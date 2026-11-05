#!/usr/bin/env python3
"""predict-step2-anonymize-qb (1:1 of skills/predict-step2-anonymize-qb/SKILL.md).

Question-bank anonymization alignment + consistency verification (a fixed step
right after step2 question generation).

Why this step is needed:
  The GT is anonymized **deterministically** via the replacement table (e.g.
  Supermicro -> Company A, always consistent); but the brier questions are
  "written" by an LLM from event-name + its own knowledge, and the LLM often
  copies real entity names into question stems (especially when event-name
  contains real names), leaking real names and defeating anonymization.

This step does two things:
  1. **Deterministic anonymization**: apply the **exact same** replacement table
     as the GT to every text field of every question -> real names become
     placeholders identical to the GT's, character for character (Company A is
     always Company A, never scrambled across questions/points). This guarantees
     both "anonymized" and "placeholder-consistent" (same table = same mapping).
  2. **LLM consistency verification**: check the anonymized bank to confirm
     (a) no residual real person/org/company/place names, and (b) the
     placeholder scheme matches the GT. Results are written to
     questionbank/qb_anon_report.json.

The original bank is backed up to questionbank_raw/ first, then questionbank/
is modified in place.

Usage:
    python predict_step2_anonymize_qb.py --workspace <anon_dir> \\
        --replacements <replacements_effective.json> [--verify-model gpt-5.2] [--no-verify]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from predict_step0_prepare import _expand_variants, derive_replacement_rules  # reuse GT anonymization logic
from common import call_llm, load_replacements


def build_replacer(replacements: List[List[str]]):
    """Longest-first deterministic replacer (same table as GT phase1 -> identical placeholders).
    **Fixed-point iteration**: some rules' RHS re-introduces another rule's LHS (e.g. only
    after `SOX法案→法律B` runs does "法律B第404条条款" appear, whose rule was already applied
    earlier in the longest-first pass) -> a single pass misses it. So iterate until stable,
    bounded (<=6 rounds) to prevent cycles."""
    rules = sorted(_expand_variants(replacements), key=lambda r: len(r[0]), reverse=True)
    rules = [(o, n) for o, n in rules if o]

    def repl(s: str) -> str:
        for _ in range(6):
            prev = s
            for old, new in rules:
                if old in s:
                    s = s.replace(old, new)
            if s == prev:
                break
        return s

    return repl


def anon_obj(obj: Any, repl) -> Any:
    """Recursively run every string value through the replacer (only hits real-name substrings; leaves numbers/enums/structure untouched)."""
    if isinstance(obj, str):
        return repl(obj)
    if isinstance(obj, list):
        return [anon_obj(x, repl) for x in obj]
    if isinstance(obj, dict):
        return {k: anon_obj(v, repl) for k, v in obj.items()}
    return obj


VERIFY_PROMPT = """You are an **extremely strict**anonymous leak auditor.The following is a copy of **anonymous** text (the real entity should have been
Replace with placeholders, such as company A/organization B/country X/person A/city A).Task: **Find out anything that may still allow readers to locate
"Residual identifiable information" of real events / real parties**.The full name of the standard has often been hidden, and almost all the ones that leak the net are the following
Long tail form-please reason carefully one by one**：

1. **Compound / derived / abbreviation **: An entity is hidden, but its compound or derived writing method is not hidden.Example: "United States" is hidden but leaked
   "U.S. Emperor / anti-U.S. / American / pro-U.S. / Sino-U.S. / Sino-U.S. Sand"; "Japan" was hidden but "Japanese media / Japanese Side" was omitted.
2. **Homonym / alias / old name **: The proper name is hidden, and the homonym or alias is not hidden (example: "Ultra-micro" is hidden but its homonym "Chaowei" is missing).
3. **Abbreviation / stem / alphabet game**: Can spell out the abbreviation of the real name or stock code, the full name of the joke (example: Put the code SMCI
   Expanded into "So Much Criminal Investigation" and "Super Money Come In"), slang in the circle.
4. **Transliteration / Roman name **: Pinyin / English name that looks like a real person's name.
5. **Specific model / proper name**: Specific weapon / product / model / organization / media / company / law firm name (such as Type 055 destroyer、
   B-52, a brokerage firm, a TV station),**even if it only appears once or twice**.
6. **Descriptive identification**: Although the name is hidden, the **descriptive attribute** can uniquely lock the real subject (example: "The world's fifth largest, owns
   An accounting firm with 115,000 employees"→ can target specific companies).
7. **Cultural / country innuendo**: Use cultural symbols to allude to the real country / nation (for example: Ganges → a certain country, Six-pointed star → a certain country, Persia→ a certain country).

🔒** Does not count as residual **: pure number / amount / percentage / date; placeholder shaped like "Company A / organization B / country X / Person A"
Itself; generally refers to common words (such as government / court / analyst / president, which do not point to a specific real subject).

Please**First reason in your heart one by one** "Does it really allow people to locate a specific subject in reality", and then**Only output JSON** (no extra text)：
{{"clean": <true|false>, "residual_real_names": ["residual string",...], "note": "One sentence"}}

Text to be audited：
{sample}
"""


def _extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.S)  # reasoning models occasionally wrap the JSON in prose
        if m:
            return json.loads(m.group())
        raise


def verify_anon(samples: List[str], model: str = "gpt-5.5", fallback: str = "gpt-5.2",
                reasoning_effort: str = "medium") -> Dict[str, Any]:
    """**Strong detection**: gpt-5.5 + reasoning enabled (reasoning_effort, default medium:
    measured recall equals high but 7x faster) + a strengthened prompt, to recall long-tail
    leaks (compound/derived forms, homophones, acronym puns, transliterations, specific model
    numbers, descriptive identification, cultural allusions) — leaks the deterministic table
    and weaker models (kimi) miss, catchable only via strong-model reasoning. If the main
    model fails/refuses -> fall back to gpt-5.2 (no reasoning)."""
    msg = [{"role": "user", "content": VERIFY_PROMPT.format(sample="\n".join(samples)[:14000])}]
    last = ""
    for m, eff in [(model, reasoning_effort), (fallback, None)]:
        if not m:
            continue
        try:
            kw = {"reasoning_effort": eff} if eff else {}
            raw = call_llm(msg, model=m, temperature=0.0, max_tokens=4000, timeout=400, **kw)
            return _extract_json(raw)
        except Exception as e:
            last = str(e)[:120]
            continue
    return {"clean": None, "_error": last}


def deterministic_check(qbs: List[Path], replacements: List[List[str]]) -> Dict[str, int]:
    """Deterministic, full-coverage correctness check: scan **all** question text for
    residual real names from the table (LHS, incl. simplified/traditional variants).
    Should be 0 after the table is fully applied; non-zero = some in-table real name
    was not fully replaced (longest-match miss / ordering issue)."""
    lhs = sorted({str(o) for o, _ in _expand_variants(replacements) if o}, key=len, reverse=True)
    blob = "".join(f.read_text(encoding="utf-8") for f in qbs)
    return {o: blob.count(o) for o in lhs if o and o in blob}


def load_gt_text(ws: Path) -> str:
    """All actual text of the anonymized GT (timeline_anon + contexts/ + gt/).
    Used by the GT-preservation guard: semi-anonymized GT **deliberately keeps public
    terms** (美国/10-K/SEC/美元/欧洲 — generic regulatory/currency/geographic words that
    do not point to a specific party). The question bank must keep them too, otherwise
    question text won't match the GT and becomes inconsistent."""
    cands = [ws / "timeline_anon.md"]
    for sub in ("contexts", "gt"):
        d = ws / sub
        if d.is_dir():
            cands += [p for p in d.iterdir() if p.is_file()]
    parts: List[str] = []
    for f in cands:
        if f.is_file():
            try:
                parts.append(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return "\n".join(parts)


def placeholder_regex(replacements: List[List[str]]) -> "re.Pattern":
    """**Data-driven**: extract placeholder types (国家/公司/审计机构/梗/编号...) from the
    replacement-table RHS and build a regex matching "type + capital letter + optional
    digits" (e.g. 国家Y/公司A7/审计机构B). No hardcoded type list; generic across events."""
    types = set()
    for _, n in _expand_variants(replacements):
        m = re.match(r"^(.+?)[A-Z]\d*$", str(n))
        if m and m.group(1):
            types.add(m.group(1))
    if not types:
        return re.compile(r"(?!x)x")  # matches nothing
    body = "|".join(re.escape(t) for t in sorted(types, key=len, reverse=True))
    return re.compile(r"(?:" + body + r")[A-Z]\d{0,3}")


def gt_placeholder_set(gt_text: str, ph_re: "re.Pattern") -> set:
    """Set of placeholders that **actually appear** in the GT text = the mapping produced
    by the GT anonymization flow — questions may only reference placeholders in this set."""
    return set(ph_re.findall(gt_text))


def prune_orphan_questions(qbs: List[Path], gt_ph: set, ph_re: "re.Pattern") -> Dict[str, int]:
    """**Off-topic question gate**: question placeholders must be a subset of GT placeholders.
    The question-writing LLM often injects entities absent from the GT from its own knowledge
    (Microsoft/ByteDance...), which after anonymization become placeholders that do not exist
    in the GT (公司Q/国家R) -> models cannot answer = unsolvable. Any stem containing a
    placeholder absent from the GT -> drop the whole question. Returns {removed, kept}."""
    removed = kept = 0
    for f in qbs:
        d = json.loads(f.read_text(encoding="utf-8"))
        changed = False
        for key in ("brier_questions", "events"):
            items = d.get(key)
            if not isinstance(items, list):
                continue
            new_items = []
            for q in items:
                s = str(q.get("q", "") or q.get("question", "")) if isinstance(q, dict) else str(q)
                if set(ph_re.findall(s)) - gt_ph:   # contains a placeholder absent from GT = off-topic
                    removed += 1
                else:
                    new_items.append(q); kept += 1
            if len(new_items) != len(items):
                d[key] = new_items
                changed = True
        if changed:
            f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return {"removed": removed, "kept": kept}


def run_verify(qbs: List[Path], model: str, char_budget: int = 12000,
               reasoning_effort: str = "medium") -> List[str]:
    """LLM full-coverage thoroughness check: dedupe **all** question stems, chunk by
    character budget, and send each chunk to the strong model (gpt-5.5 + reasoning) to
    detect residual real names (no sampling). Finds new real names "not in the table,
    injected by the LLM from knowledge + long-tail variants". Returns deduped residuals."""
    stems, seen = [], set()
    for f in qbs:
        d = json.loads(f.read_text(encoding="utf-8"))
        for q in d.get("brier_questions", []):
            s = str(q.get("q", "")).strip()
            if s and s not in seen:
                seen.add(s)
                stems.append(s)
    chunks: List[List[str]] = []
    cur, cur_len = [], 0
    for s in stems:
        if cur and cur_len + len(s) > char_budget:
            chunks.append(cur); cur, cur_len = [], 0
        cur.append(s); cur_len += len(s) + 1
    if cur:
        chunks.append(cur)
    out, ks = [], set()
    for ch in chunks:
        v = verify_anon(ch, model, reasoning_effort=reasoning_effort)
        for nm in (v.get("residual_real_names") or []):
            nm = str(nm).strip()
            if nm and nm.lower() not in ks:
                ks.add(nm.lower()); out.append(nm)
    return out


def apply_table(qbs: List[Path], repl) -> int:
    """Deterministic anonymization (in place): returns the number of banks changed."""
    n = 0
    for f in qbs:
        d = json.loads(f.read_text(encoding="utf-8"))
        before = json.dumps(d, ensure_ascii=False)
        d = anon_obj(d, repl)
        after = json.dumps(d, ensure_ascii=False)
        if before != after:
            n += 1
        f.write_text(after, encoding="utf-8")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Question-bank anonymization alignment + residual closed-loop repair (post-step2 step)")
    p.add_argument("--workspace", required=True, help="anon workspace directory (contains questionbank/)")
    p.add_argument("--replacements", required=True, help="same replacement table as the GT (replacements_effective.json)")
    p.add_argument("--verify-model", default="gpt-5.5", help="detection model (default gpt-5.5, strong model + reasoning; few calls so affordable, recalls long-tail leaks; falls back to gpt-5.2 on failure)")
    p.add_argument("--reasoning-effort", default="medium", help="reasoning effort (for reasoning-capable models like gpt-5.5; low/medium/high)")
    p.add_argument("--no-verify", action="store_true", help="deterministic anonymization only; skip LLM verification + repair")
    p.add_argument("--max-fix-rounds", type=int, default=3, help="max rounds of detect -> derive -> re-anonymize")
    a = p.parse_args()

    ws = Path(a.workspace)
    qb_dir = ws / "questionbank"
    if not qb_dir.exists():
        sys.exit(f"no questionbank/ at {qb_dir}")
    replacements = load_replacements(Path(a.replacements))

    # GT-preservation guard: the bank's anonymization boundary must **exactly match the GT** —
    # public terms the GT keeps as real names (美国/10-K/SEC/美元/欧洲...) must NOT be anonymized
    # in the bank, else questions say "国家Y/表格B" while GT says "美国/10-K" -> mismatch.
    # The mechanism is **generic, no hardcoded entity names**: any replacement rule whose LHS
    # (length >= 2) appears in the anonymized GT text is dropped; this also **self-heals past
    # over-anonymization** (removes previously mis-added rules, combined with restore-from-raw + re-anonymize).
    gt_text = load_gt_text(ws)

    def _gt_kept(o: Any) -> bool:
        s = str(o)
        return len(s) >= 2 and s in gt_text

    def _o(r: Any):
        return r[0] if isinstance(r, (list, tuple)) and r else None

    def _n(r: Any):
        return r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else None

    # **Placeholder-group normalization**: if any rule of a placeholder has its LHS in the GT
    # (anchor = the GT keeps that entity's real name), then the entity's **other spellings /
    # short forms** (sibling rules) should be **normalized** to the GT's canonical real name
    # rather than each anonymized to a placeholder — otherwise the same entity is a real name
    # in GT but a placeholder in the bank, still inconsistent. The anchor is the longest
    # GT-kept LHS for that placeholder.
    anchor: Dict[str, str] = {}
    for r in replacements:
        o, n = _o(r), _n(r)
        if o and n and _gt_kept(o) and (n not in anchor or len(str(o)) > len(str(anchor[n]))):
            anchor[n] = o
    dropped, normalized, kept_rules = [], [], []
    for r in replacements:
        o, n = _o(r), _n(r)
        if o and _gt_kept(o):
            dropped.append(r)                     # anchor: GT already has this real name -> drop rule, bank keeps the real name
        elif o and n in anchor:
            normalized.append([o, anchor[n]])     # sibling spelling -> normalize to GT's canonical real name (no longer anonymized)
        else:
            kept_rules.append(r)                  # bank-only real name -> anonymize as usual (prevent leaks)
    if dropped or normalized:
        replacements = kept_rules + normalized
        # 保留守卫] 丢锚 --> Retain guard] Drop anchor
        print(f"[GT-Retain guard] Drop anchor {len(dropped)} Article + Brother writing method is normalized {len(normalized)} Strip"
              f"(Question bank anonymous boundary alignment GT).lost: {[_o(r) for r in dropped][:6]} Normalize: {[(r[0], r[1]) for r in normalized][:6]}")

    repl = build_replacer(replacements)
    qbs = sorted(qb_dir.glob("P*.json"))

    # 1) Original bank: back up on first run; on later runs **restore** from raw
    #    (clean re-anonymization starting point; self-heals over-anonymization)
    raw_dir = ws / "questionbank_raw"
    if not raw_dir.exists():
        shutil.copytree(qb_dir, raw_dir)
        # 原始题库 --> Original question bank
        print(f"[backup] Original question bank → {raw_dir}")
    else:
        for f in raw_dir.glob("P*.json"):
            shutil.copy2(f, qb_dir / f.name)
        print("[restore] From questionbank_raw Restore the question bank (clean starting point after a full reset, self-healing, excessive anonymity)")

    # 2) Initial deterministic anonymization (apply the existing table -> same placeholders as GT)
    n0 = apply_table(qbs, repl)
    print(f"[anon] Anonymous for the first time：{len(qbs)} Question bank，{n0} A real name is replaced")

    # 3) Closed-loop repair: **full-coverage** LLM detection of residuals (new real names the
    #    question LLM injected from knowledge, absent from the table) -> derive placeholders ->
    #    extend the table -> re-anonymize -> re-check, until convergence.
    report: Dict[str, Any] = {"model": a.verify_model, "rounds": []}
    added: List[List[str]] = []
    if not a.no_verify:
        for rnd in range(1, a.max_fix_rounds + 1):
            resid = run_verify(qbs, a.verify_model, reasoning_effort=a.reasoning_effort)  # all stems, deduped + chunked, no sampling
            report["rounds"].append({"round": rnd, "residual_candidates": resid})
            print(f"[round {rnd}] Comprehensive detection of residual candidates {len(resid)}: {resid[:15]}")
            if not resid:
                print("  ✅ No residue, convergence"); break
            have = {x[0] for x in replacements if isinstance(x, (list, tuple)) and len(x) >= 1}
            new_rules = derive_replacement_rules(
                "", [{"snippet": n} for n in resid],
                model=a.verify_model, existing_rules=replacements)
            new_rules = [r for r in new_rules if r and r[0] not in have and not _gt_kept(r[0])]
            if not new_rules:
                print("  No new rules can be derived (the remaining items are mostly false positives or public entities). Stop."); break
            print(f"  derive generate {len(new_rules)} New rules: {new_rules[:8]}")
            replacements = replacements + new_rules
            added += new_rules
            repl = build_replacer(replacements)
            apply_table(qbs, repl)
    # Write the guard results (dropped anchors / sibling normalization, self-healed
    # over-anonymization) + newly derived rules back to the replacement table
    # (keep table = source of truth, consistent going forward)
    if added or dropped or normalized:
        Path(a.replacements).write_text(
            json.dumps(replacements, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) Deterministic full-coverage correctness check: scan all questions for residual
    #    in-table real names (LHS); should be 0 = table fully applied
    det = deterministic_check(qbs, replacements)
    report["deterministic_residual"] = det
    print(f"[Certainty check] The real name remains in the table: {det if det else '✅ 0 (The replacement table has been fully applied.)'}")

    # 5) Off-topic question gate: question placeholders must be a subset of GT placeholders
    #    (per the mapping produced by GT anonymization). Entities injected by the question LLM
    #    that are absent from the GT -> placeholders the GT never defines -> unanswerable -> drop.
    ph_re = placeholder_regex(replacements)
    gt_ph = gt_placeholder_set(gt_text, ph_re)
    prune = prune_orphan_questions(qbs, gt_ph, ph_re)
    report["orphan_prune"] = prune
    print(f"[Off-topic question gate] Remove questions that reference entities not found in GT {prune['removed']}，keep {prune['kept']}")

    report["added_rules"] = added
    (qb_dir / "qb_anon_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "questionbanks": len(qbs),
        "initial_anonymized": n0,
        "auto_derived_rules": len(added),
        "deterministic_residual_kinds": len(det),
        "verify": "skipped" if a.no_verify else "done",
        "report": str(qb_dir / "qb_anon_report.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
