#!/usr/bin/env python3
"""predict-fullanon-reaudit: generic "full anonymization + strong-detector multi-round re-audit" deep re-wash.

== What this is ==
Re-runs step0's closed loop of "anonymize -> multi-round audit -> derive new rules ->
re-anonymize -> re-check", with two upgrades:
  1) **Stronger detector**: gpt-5.5 + reasoning enabled + a strengthened prompt, to recall
     **long-tail** leaks — compound/derived forms (美帝/反美/美式/中美沙), homophones/aliases
     (超微 <-> 超威), acronyms/puns (SMCI = So Much Criminal Investigation), transliterated
     roman names, specific model numbers / proper names (055型驱逐舰/B-52/a named brokerage),
     descriptive identification (fifth-largest -> a specific company), cultural allusions
     (恒河 -> a country, 六芒星 -> a country). These are missed by the deterministic table and
     weaker models (kimi), and only caught by strong-model reasoning.

     STRONG MODEL AND PROMPT : 
     [PREVENT LONG TAILED LEAKS. RANDOM INFO ABOUT SOMETHING THAT MIGHT GIVE SOME HINTS ABOUT ANY COMPANY]

  2) **Scope changed to full-anon**: even **named public terms** (countries / named regulators /
     named financial-report forms / currencies / regions / indices) get anonymized; only
     numbers/amounts/percentages/dates, placeholders like 公司A/国家X, and **generic common
     nouns** that do not point to a specific party (government/court/president/analyst/
     central bank) are kept.

== Genericity ==
**No entity names are hardcoded**: LLM detection finds residuals -> `derive_replacement_rules`
produces "type + code" placeholders -> appended to the same replacement table -> applied
deterministically to **all GT files (timeline+contexts+gt) and the question bank**
(placeholders consistent throughout) -> re-detect, iterating to convergence. Works for all
events. Dates were already shifted in step0 and numbers are prediction signals — **never touched**.

== Usage ==
    python predict_fullanon_reaudit.py --workspace <anon_dir> \
        [--model gpt-5.5] [--reasoning-effort medium] [--max-rounds 4]
Usually copy the existing anon directory to a **new timestamped folder** first, then run this
script on the new folder (old artifacts untouched).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from predict_step0_prepare import derive_replacement_rules, _expand_variants
from predict_step2_anonymize_qb import (
    build_replacer, apply_table, _extract_json,
    placeholder_regex, gt_placeholder_set, prune_orphan_questions)
from common import call_llm, load_replacements

"""
Report residual content is there or not.

"""
FULLANON_PROMPT = """You are an **extremely strict** anonymization auditor. Below is a text in which real entities should already have been replaced with placeholders.
(Company A/Organisation B/Country X/Person A, etc.) The objective of this task is **Complete Anonymity**——**Identify every proper noun that could allow readers to pinpoint
a specific real-world entity.**，Including:

【Standard Named Entities】 Real-world Countries / Regions / Cities; named government departments / Regulatory agencies / International organizations; companies / Brand /
Products; personal names; media outlets; named laws and regulations / Financial Report Table / Index / Currenncy (For example, a particular country, a particular securities regulator, a particular Form 10-K, a particular currency, a particular index).
【Easy-to-Miss Long-Tail Cases—Reason Through and Check Each One】
  1. **Compounds / Derivatives / Shortened names**:The entity itself was anonymized, but its compound and derived forms weren’t.(“United States” was hidden, but “U.S. imperialism/anti-American/American-style/China–U.S.–Saudi Arabia” slipped through.).
  2. **Homophones / Aliases / Former names**:The real name was anonymized, but homophones or nicknames weren’t.(Maine / Main — Maine (US state) and main (primary).).
  3. **Abbreviations / Memes / Letter play**:Abbreviations that spell out real names or ticker symbols, humorous full names, and insider slang.
  4. **Transliterations / Romanized names**:Pinyin / English names that look like real people's names.
  5. **Specific models / Proper names**:Names of specific weapons / products / models / institutions / media outlets / companies / law firms,**Even if it only happens once or twice**.
  6. **Descriptive references**:Although the name is concealed, the **descriptive attributes** can uniquely identify the actual entity("The world’s fifth-largest accounting firm, with 115,000 employees").
  7. **Cultural / National allusions**:Using cultural symbols to allude to real countries / Ethnic groups(Ganges→A certain country,Hexagram→A certain country,Persia→A certain country).

🔒 **Do not count as residual content(Keep)**：Pure Numbers / Amounts / Percentages / Date(**Note:Keep the "numbers" in monetary amounts, but "currency names"
such as US dollar / Euro / Chinese yuan are still named entities; and must be counted as residuals**.For example, for “400 US dollars” report “US dollars”);The placeholders themselves, such as “Company A / Organization B /
Country X / Person A”;**Generic Common Nouns**——Common nouns / roles that do not refer to a specific real-world entity (government, court,
president, analyst, central bank, regulatory agency [generic], company [generic], journalist, legislator, capital city).

Please **First, mentally work through each candidate one by one.**Whether it can actually identify a specific real-world entity, Again **Output only JSON** (No extra text):
{{"clean": <true|false>, "residual_real_names": ["Residual string", ...], "note": "Single sentence"}}

Text to be audited:
{sample}
"""

# Placeholders (type + code, e.g. 国家X/公司A/审计机构B/货币A) — the detector may
# false-positive on these; filter them out instead of sending to derive.

PLACEHOLDER_RE = re.compile(r"^[A-Z][a-z]{1,15}[A-Z]{0,3}\d{0,3}$")
# Pure numbers/amounts/percentages/dates — always kept by rule; filter out.
NUMERIC_RE = re.compile(r"^[\d\s.,%+\-—~:：/'year''month''day''hour''minute''second''one''thousand''ten thousand''lakh''ten lakh''crore''ten crore''hundred crore''usd''us dollar''u.s. dollar''inr''rupee'$€¥]+$")


def verify_fullanon(samples: List[str], model: str, fallback: str, effort: str) -> Dict[str, Any]:
    """Strong detection (full-anon prompt); main model failure/refusal -> fallback (no reasoning)."""
    msg = [{"role": "user", "content": FULLANON_PROMPT.format(sample="\n".join(samples)[:14000])}]
    last = ""
    for m, eff in [(model, effort), (fallback, None)]:
        if not m:
            continue
        try:
            kw = {"reasoning_effort": eff} if eff else {}
            raw = call_llm(msg, model=m, temperature=0.0, max_tokens=8000, timeout=180, **kw)
            return _extract_json(raw)
        except Exception as e:
            last = str(e)[:120]
            continue
    return {"clean": None, "_error": last}


def detect_unique(blobs: List[str], model: str, fallback: str, effort: str,
                  char_budget: int = 2500) -> List[str]:
    """Run full-coverage detection on **deduplicated unique text lines** (the GT's contexts/gt
    replicate the timeline per point and are highly redundant; after dedup only the master-copy
    volume remains -> saves time without losing variety). Returns deduped residual candidates."""
    # Create list of unique lines in lines[]
    # 
    seen, lines = set(), []
    for blob in blobs:
        for ln in blob.splitlines():
            ln = ln.strip()
            if len(ln) >= 4 and ln not in seen:
                seen.add(ln)
                lines.append(ln)
    chunks: List[List[str]] = []
    cur, n = [], 0

    # Create chunks of lines that stay within the char budget
    # Store them in chunks list[]
    for ln in lines:
        if cur and n + len(ln) > char_budget:
            chunks.append(cur); cur, n = [], 0
        cur.append(ln); n += len(ln) + 1

    # Also append the last partial chunk as well
    if cur:
        chunks.append(cur)
    
    # For each chunk, check whether it contains residuals (non anonymised terms)
    # ks is the set of residuals encountered in text
    # out is the same list of residuals

    out, ks = [], set()
    for i, ch in enumerate(chunks):
        # Run verify_full_anon on each chunk
        v = verify_fullanon(ch, model, fallback, effort)
        for nm in (v.get("residual_real_names") or []):
            nm = str(nm).strip()
            if nm and nm.lower() not in ks:
                ks.add(nm.lower()); out.append(nm)

    # Output the list of non-anonymised terms (residuals) encountered in text 
    return out


def is_noise(s: str) -> bool:
    s = s.strip()
    # S should not be placeholder_RE or NUMERIC_RE 
    return (not s) or bool(
        PLACEHOLDER_RE.match(s)) or bool(NUMERIC_RE.match(s))


def gt_files(ws: Path) -> List[Path]:
    """ Lists and stores all files present in contexts and gt subdirectory of workspace """
    fs = [ws / "timeline_anon.md"]
    for sub in ("contexts", "gt"):
        d = ws / sub
        if d.is_dir():
            fs += sorted(p for p in d.iterdir() if p.is_file())
    return [f for f in fs if f.is_file()]


def apply_text(files: List[Path], repl) -> int:
    n = 0
    for f in files:
        t = f.read_text(encoding="utf-8")
        t2 = repl(t)
        if t2 != t:
            n += 1
        f.write_text(t2, encoding="utf-8")
    return n


def qb_corpus(qbs: List[Path]) -> List[str]:
    out = []
    for f in qbs:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for q in d.get("brier_questions", []):
            s = str(q.get("q", "")).strip()
            if s:
                out.append(s)
    return out


FULLANON_DERIVE_PROMPT = """Below is a list of "potentially identifiable information" extracted from a text where "Goal=complete anonymity."
For all items that are **proper nouns capable of identifying specific, real-world entities**, assign a “type + code” placeholder.

⚠️ This is a **completely anonymous** task: Named **Countries / Regions / Straits / Cities, Regulatory Agencies / Government Departments / 
International Organizations, Political Parties, Companies / Brands / Products, People’s Names, Media Outlets, Named Laws and Regulations /
Financial Statement Tables / Indexes / Currencies, Specific Model Numbers / Case Numbers/ Reference Numbers**——
**Assign placeholders to all of them**, **do not skip any just because they are "common / frequently used / generic terms."** 
(the point of this task is to anonymize them too).

Code naming rules (type + letter code): Country→CountryX/Y/Z; Region→RegionA; Strait→StraitA; City→CityA;
Institution/Department/Regulatory Agency/Organization→InstitutionA; Political Party→Political PartyA; Company→CompanyA; Media Outlet→Media OutletA; Person→PersonA/LeaderA;
Regulation/Provision→RegulationA; Financial Statement Table→TableA; Currency→CurrencyA; Index→IndexA; Product/Model→ModelA; Case Number/Reference Number→NumberA;
**Abbreviations / memes / wordplay (joke-like full forms or slang that can spell out real names or stock ticker symbols) → MemeA/AliasA** …
- **Use different code names for distinct entities of the same type**；**The same entity (including abbreviations/variants/homophones/substrings) reuses the same code name**；
- Placeholders must not contain real names from the original text, consecutive English letters, or multi-digit numbers.

**Only skip these (no allocation, no output)**：pure numbers / monetary amounts / percentages / dates;Existing placeholders(CountryX/CompanyA…）；
**Generic common nouns / roles**(not referring to any specific real-world entity, such as government, court, president, analyst, journalist, young people, or Indian people);
**Purely descriptive sentences**(**containing no recoverable real names or codes**——if a sentence contains a "meme" or reference that can be decoded to reveal a real name or code, that "meme" or reference must still be anonymized).

Existing mappings (real name → assigned code name). Assign unique codes to new entities—avoiding codes already in use—and reuse the same code for variants or abbreviations:
{existing_map}

Only output JSON Array (No additional text):[["Real proper names","Placeholder"], ...]；Do not include items that have been skipped.

List:
{names_block}
"""


def derive_fullanon(cands: List[str], model: str, existing_rules) -> List[List[str]]:
    """full-anon variant of derive: like derive_replacement_rules it produces "type + code"
    placeholders and passes in the current table for consistency, but it does **not skip named
    public terms** (countries/regulators/report forms/currencies/regions... all anonymized) —
    only pure common nouns / sentences are skipped."""
    names, seen = [], set()
    for c in cands:
        s = str(c).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower()); names.append(s[:80])
    if not names:
        return []
    pairs = [p for p in (existing_rules or [])
             if isinstance(p, (list, tuple)) and len(p) >= 2 and str(p[1]).strip()]
    existing_map = "\n".join(f"- {str(p[0]).strip()} → {str(p[1]).strip()}" for p in pairs[-120:]) or "(None)"
    prompt = FULLANON_DERIVE_PROMPT.format(
        names_block="\n".join(f"- {n}" for n in names), existing_map=existing_map)
    try:
        # gpt-5.5 is a reasoning model; with many candidates the output token need is large.
        # Give ample max_tokens, else reasoning eats the budget and content comes back empty.
        raw = call_llm([{"role": "user", "content": prompt}], model=model,
                       temperature=0.0, max_tokens=16000, timeout=300)
    except Exception:
        return []
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    out = []
    for it in data:
        if not (isinstance(it, list) and len(it) == 2):
            continue
        bad, good = str(it[0]).strip(), str(it[1]).strip()
        if not (bad and good) or bad == good:
            continue
        if len(good) > max(10, len(bad)):
            continue

        # THIS RE IS WRONG, NEED TO DESIGN NEW ONE
        # THIS RE ALLOWS 54,756 (entries) proper nouns to be anonymised AT MAX
        # TODO: FROM HERE!
        # (WONT WORK) : TAKE ALL STRINGS, DON´T FILTER OUT ANY GOOD STRINGS AT ALL
        # UNCOMMENT BELOW TO FILTER [NEEDS SOME DETERMINISM]

        # [HAS SOME STRUCTUREssssss]
        # ALLOWS AT_MAX 3 letters to be added in the placeholder, and atmost 3 numbers to be added
        # THEREBY CREATING A MAX OF 54756 replacements
        if re.search(r"[A-Z]{4,}|\d{4,}", good):
            continue

        if bad.lower() in good.lower():
            continue


        out.append([bad, good])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Full anonymization + strong-detector multi-round re-audit (generic, works for all events)")
    p.add_argument("--workspace", required=True, help="anon workspace directory (recommended: copy to a new timestamped folder first)")
    p.add_argument("--replacements", default=None, help="replacement table (default <ws>/replacements_effective.json)")
    p.add_argument("--model", default="gpt-5.5", help="strong detection model (default gpt-5.5)")
    p.add_argument("--fallback", default="gpt-5.2", help="fallback model (when the main model refuses/fails)")
    p.add_argument("--reasoning-effort", default="medium", help="reasoning effort low/medium/high (default medium)")
    p.add_argument("--max-rounds", type=int, default=4, help="max rounds of detect -> derive -> re-anonymize")
    p.add_argument("--char-budget", type=int, default=2500, help="chars per chunk (gpt-5.5 reasoning is super-linearly sensitive to length; small chunks are faster)")
    a = p.parse_args()

    ws = Path(a.workspace)
    tbl_path = Path(a.replacements) if a.replacements else ws / "replacements_effective.json"
    replacements = load_replacements(tbl_path)
    repl = build_replacer(replacements)
    gfs = gt_files(ws) # All files present in gt subdirectories
    qbs = sorted((ws / "questionbank").glob("P*.json")) if (ws / "questionbank").is_dir() else [] # get the path of all questionbank files
    print(f"[init] GT Files {len(gfs)} Items, Question Bank {len(qbs)} Items, Table has {len(replacements)} Rules") 

    # 0) First re-apply the existing table deterministically (rolls out full-anon rules
    #    already in the table; idempotent)
    apply_text(gfs, repl) # repl is the replacer, it replaces with anonymisation in all gtfiles 
    if qbs:
        apply_table(qbs, repl) # repl is the replacer, it replaces with anonymisation in all question bank files 

    have = {str(r[0]) for r in replacements if isinstance(r, (list, tuple)) and r}
    report: Dict[str, Any] = {"model": a.model, "effort": a.reasoning_effort, "rounds": []}
    added: List[List[str]] = []

    for rnd in range(1, a.max_rounds + 1):
        # Detect on **GT only** (~20K chars after master-copy dedup) — the question bank can be
        # thousands of questions / hundreds of thousands of chars and would stall detection,
        # and the bank's own knowledge injection is handled by step2.5 (gpt-5.5); rules derived
        # here are still applied to the bank.
        blobs = [f.read_text(encoding="utf-8") for f in gfs] # Store texts of all files present in gt folder
        resid = detect_unique(blobs, a.model, a.fallback, a.reasoning_effort, char_budget=a.char_budget) # List of residuals (non-anonymised terms present in blob/text)
        cands = [c for c in resid if not is_noise(c) and c not in have] # not existing residual, and is a proper noun not some noise
        print(f"[round {rnd}] Residuals detected {len(resid)}，Denoising/Removing known elements in preparation for anonymization {len(cands)}：{cands[:15]}")
        report["rounds"].append({"round": rnd, "residual": resid, "to_anon": cands})
        if not cands:
            print("  ✅ No new proper names to anonymize; the process has converged."); break
        new_rules = derive_fullanon(cands, model=a.model, existing_rules=replacements)
        new_rules = [r for r in new_rules if r and str(r[0]) not in have]
        # If no new residuals are found =>
        if not new_rules:
            print("Classify all instances of residuals as non-proper nouns/generic references, then stop."); break
        # Else is new residuals and rules are found =>
        print(f"Derivations generated {len(new_rules)}, New rules: {new_rules[:8]}")
        replacements += new_rules
        added += new_rules
        have |= {str(r[0]) for r in new_rules} # have = have | {str(r[0]) for r in new_rules}
        repl = build_replacer(replacements)
        apply_text(gfs, repl) # Apply anonymisation replacements to gt files
        if qbs:
            apply_table(qbs, repl) # And to question banks

    # Write the table back + deterministic correctness check (all in-table real-name LHS
    # should be washed out; count should be 0)
    if added:
        tbl_path.write_text(json.dumps(replacements, ensure_ascii=False, indent=2), encoding="utf-8")
    # Sort replacement rules in longest LHS first 
    # lhs = sorted({str(o) for o, _ in _expand_variants(replacements) if o}, key=len, reverse=True)
    # Remove expand_variants : No need for expand variants anymore
    lhs = sorted({str(o) for o, _ in replacements if o}, key=len, reverse=True)
    blob = "".join(f.read_text(encoding="utf-8") for f in gfs) + "".join(
        f.read_text(encoding="utf-8") for f in qbs) # Combine lines of question banks and gt files together
    det = {o: blob.count(o) for o in lhs if o and o in blob}
    report["deterministic_residual"] = det # Contains the residual string, and # times it appears in all texts. Eg. Musk:250

    # Off-topic question gate: question placeholders must be a subset of GT placeholders (per
    # the mapping produced by GT anonymization; drop questions about entities the GT lacks)
    if qbs:
        ph_re = placeholder_regex(replacements)
        gt_ph = gt_placeholder_set("".join(f.read_text(encoding="utf-8") for f in gfs), ph_re)
        prune = prune_orphan_questions(qbs, gt_ph, ph_re)
        report["orphan_prune"] = prune
        print(f"[离题题门禁] 剔除引用 GT 不存在实体的题 {prune['removed']}，保留 {prune['kept']}")

    report["added_rules"] = added
    (ws / "fullanon_reaudit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[确定性检查] 表内真名残留 {len(det)} 种" + (f"：{dict(list(det.items())[:8])}" if det else " ✅ 0"))
    print(json.dumps({
        "gt_files": len(gfs), "questionbanks": len(qbs),
        "rounds": len(report["rounds"]), "added_rules": len(added),
        "deterministic_residual_kinds": len(det),
        "report": str(ws / "fullanon_reaudit_report.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
