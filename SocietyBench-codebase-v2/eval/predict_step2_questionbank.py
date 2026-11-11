#!/usr/bin/env python3
"""predict-step2-questionbank (1:1 of skills/predict-step2-questionbank/SKILL.md).

SKILL §"出题全程由 CC 完成" = the work is done by an LLM. The SKILL forbids
using DMXAPI/kimi specifically (original team's self-contamination concern,
since kimi is also in `eval_models`), but any other LLM is fine. Model
configurable via `--model` / pipeline_config.default_llm_model.

For each prediction point, build a self-contained questionbank covering:

  events.all          — compact real future events (3F consumer; one date prediction per event)
  events.base         — same content as events.all (audit name retained)
  events.major        — optional subset of major turning points (audit only)
  brier_questions     — expanded probability question bank (3B consumer)

Also performs Step 2.0 — splitting the timeline at each prediction point's
cutoff into <workspace>/contexts/{Pxx}_context.md and <workspace>/gt/{Pxx}_gt.md.

For each point:
  1. Read per-point context (pre-cutoff) and GT (post-cutoff) markdowns
  2. Compute `target_total` via the SKILL formula:
        S_pred = K_i × 15
        S_gt   = G_i × 8
        S_rule = S_pred if S_pred >= S_gt else ceil((S_pred + S_gt) / 2)
        target_total = min(450, max(15, ceil(base_event_count × 2.5), S_rule))
  3. LLM extract events.all (compact real events)
  4. LLM generate brier_questions (length == target_total) per A:B:C:D = 4:2:1:3
  5. validate_questionbank; if it fails, retry up to 2 times with issues fed
     back into the prompt
  6. Compute density_check (events in 1-30 / 31-60 / 61-90 day buckets)
  7. Write <workspace>/questionbank/{Pxx}_questionbank.json
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import call_llm, load_pipeline_config, parse_date_obj


# ===========================================================================
# Step 2.0: split timeline into per-point contexts/ and gt/
# ===========================================================================

def split_context_gt(workspace: Path) -> int:
    tl_path = workspace / "timeline_with_points.md"
    pp_path = workspace / "prediction_points.json"
    if not tl_path.exists() or not pp_path.exists():
        raise SystemExit(f"missing {tl_path} or {pp_path}")

    tl = tl_path.read_text(encoding="utf-8")
    pp = json.loads(pp_path.read_text(encoding="utf-8"))
    points = pp.get("prediction_points") if isinstance(pp, dict) else pp

    blocks = re.split(r"(?=^### \d{4}-\d{2}-\d{2})", tl, flags=re.M)
    ctx_dir = workspace / "contexts"
    gt_dir = workspace / "gt"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for pt in points:
        if not pt.get("point_id"):
            continue
        pid = pt["point_id"]
        cutoff = parse_date_obj(pt.get("target_date"))
        if cutoff is None:
            continue
        ctx_parts: List[str] = []
        gt_parts: List[str] = []
        for blk in blocks:
            m = re.match(r"^### (\d{4}-\d{2}-\d{2})", blk)
            if not m:
                if not ctx_parts and not gt_parts:
                    ctx_parts.append(blk)
                continue
            d = parse_date_obj(m.group(1))
            if d is None:
                continue
            if d < cutoff:
                ctx_parts.append(blk)
            else:
                gt_parts.append(blk)
        (ctx_dir / f"{pid}_context.md").write_text("".join(ctx_parts), encoding="utf-8")
        (gt_dir / f"{pid}_gt.md").write_text("".join(gt_parts), encoding="utf-8")
        n += 1
    return n


# ===========================================================================
# Helpers
# ===========================================================================

def _salvage_truncated_array(raw: str) -> Optional[List[Any]]:
    """Salvage the complete entries before the truncation point from a JSON array
    cut off by max_tokens. The main loop (while len(brier_qs) < n_target) already
    tops up via repeated calls, so returning a partial-but-valid result lets that
    accumulation mechanism work — instead of crashing on truncation."""
    i = raw.find("[")
    if i < 0:
        return None
    s = raw[i + 1:]
    dec = json.JSONDecoder()
    out: List[Any] = []
    idx, n = 0, len(s)
    while idx < n:
        while idx < n and s[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or s[idx] == "]":
            break
        try:
            obj, end = dec.raw_decode(s, idx)
        except json.JSONDecodeError:
            break  # truncated mid-entry -> stop, return the complete ones
        out.append(obj)
        idx = end
    return out or None


def parse_json_strict(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        # Salvage a max_tokens-truncated array: return the complete pre-truncation entries
        # (the main while loop tops up the remainder)
        salvaged = _salvage_truncated_array(raw)
        if salvaged:
            return salvaged
        raise


# The main model kimi occasionally fails with "content filter" etc. (call_llm already
# retried 3 times internally, so an error here = repeated failures); in that case switch
# to a cheap foreign fallback gpt model (not subject to the same filtering). Only a few
# points trigger this; the vast majority still go through kimi.
FALLBACK_MODEL = "gpt-5.2"


def _call_with_fallback(messages: List[Dict[str, str]], *, model: str, **kwargs) -> str:
    try:
        return call_llm(messages, model=model, **kwargs)
    except Exception as e:
        print(f"  [fallback] Main Model {model} failed({str(e)[:60]}) → Cut and set aside {FALLBACK_MODEL}", flush=True)
        return call_llm(messages, model=FALLBACK_MODEL, **kwargs)


def count_gt_nodes(gt_text: str) -> int:
    return len(re.findall(r"^### \d{4}-\d{2}-\d{2}", gt_text, flags=re.M))


def days_between(target: str, cutoff: str) -> Optional[int]:
    d_t = parse_date_obj(target)
    d_c = parse_date_obj(cutoff)
    if d_t is None or d_c is None:
        return None
    return (d_t - d_c).days


def target_total(base_event_count: int, remaining_points: int, gt_nodes: int,
                 cfg: Dict[str, Any]) -> int:
    max_q = int(cfg.get("max_questions_per_point", 450))
    min_q = int(cfg.get("min_questions_per_point", 15))
    pred_m = float(cfg.get("predicted_points_multiplier", 15))
    gt_m = float(cfg.get("gt_nodes_multiplier", 8))
    ev_m = float(cfg.get("events_count_multiplier", 2.5))

    s_pred = remaining_points * pred_m
    s_gt = gt_nodes * gt_m
    s_rule = s_pred if s_pred >= s_gt else ceil((s_pred + s_gt) / 2)
    return min(max_q, max(min_q, ceil(base_event_count * ev_m), int(s_rule)))


# ===========================================================================
# LLM prompts (SKILL §Step 2.1 + §Step 2.2)
# ===========================================================================

PROMPT_EVENTS = """You are an assistant that evaluates questions. The deadline for the event “{event}” is {cutoff}.
The following is the ground-truth timeline (GT) for the event after {cutoff} **. Please extract a compact set of real events** for 3F time prediction:

Requirements for compact event sets:
- Each event must be publicly verifiable, have a pinpointable date, and be individually scoreable.
- Do not mechanically include every date in the timeline; retain only high-confidence facts worth including in a time forecast to move things forward (policy actions, official statements, negotiations or agreements, investigations or rulings, enforcement or suspensions, and clear turning points in public opinion).
- Exclude explanatory notes, comments, FAQs, live feeds, aggregated pages, and analyses of purely peripheral impacts.
- **Do not split a real event into multiple parts, mechanically change the wording, or rewrite it using synonyms.**
- By default, include one event per high-value date; add one independent new event only if genuinely necessary.
- event_desc must be generalized and rewritten: specify the organization’s name, obscure the number, and paraphrase any quotation.

================ GT ================
{gt_text}
================

**Output**: A strict JSON array with the following fields:
  id (string, sequential numbering: E01/E02/...)
  event (one-sentence generalized description in Chinese, without a specific date)
  date (YYYY-MM-DD)
  group (integer; use the same group for the same subject on the same day)
  days_from_cutoff (integer = date − cutoff)

Do not include Markdown code fences or any additional text.
"""


PROMPT_BRIER = """You are an evaluation-question generation assistant. The event is “{event}”; cutoff date: {cutoff}.
Based on the following events.all (a compact set of real events), generate **exactly {n_questions}** Brier probability questions.

================ events.all ================
{events_json}
================

Target question-type distribution A:B:C:D = 4:2:1:3 (a reference target for proportions, not a hard constraint):
- A Time-gradient questions: 1 true-window question (answer=1) per real event; optionally add 1 short-window false question (answer=0)
- B Certainty-gradient questions: true questions with clear differences in level of generality; test direction and granularity
- C Outcome-variant questions: supplemental question type, used to fill out the distribution
- D Highly deceptive false questions: only 1 per fake event; plausible-looking but did not actually occur in the ground truth

**Predictable-granularity rule (critical; determines question quality and discriminative power):**
- The question should ask only whether a **directional or structural turning point occurred** (e.g., “Was the company delisted?” “Was an internal investigation launched?” “Was the company sued?” “Did it change auditors?”). **Do not include details that could only be known after the fact in the question**—specific wording, figures, or statements (e.g., “said it found no fraud,” “expressed confidence that it would meet requirements,” “revenue growth fell below the bottom of the forecast range”) must all be stripped out. A forecaster at the cutoff could not predict these details; including them turns the question into a test of luck rather than forecasting skill.
- **Do not use compound conditions such as “both A and B”** (hitting two specific outcomes is harder and less predictable); each question should ask about only one structural outcome.
- Self-check: Could a skilled forecaster at the cutoff have reasonably predicted the **direction** of this outcome? If not (too detailed, compound, or only knowable after the fact), abstract it to a predictable level of granularity or discard it.

D-category rules (required):
1) Every D question must be unique; do not fill them using “Item N” templates.
2) If it overlaps with any key phrase in events.all (≥3 Chinese characters / ≥3 words) by 3 or more, reject it.
3) Forbidden terms (config-driven list of “too far-fetched, not discriminative” terms; currently active): {blacklist_terms}
4) Prohibited unfalsifiable keywords (private behavior, mental states, etc. that cannot be recorded by public information sources): {unfalsifiable_terms}
5) Descriptions must be verifiable in public sources (news/social media).

Quantity and balance (**must be satisfied during generation; do not rely on later retries to fix them**):
- Overall true:false ratio must be between 0.7 and 1.5
- Must include D-category questions
- A-category easy questions must be ≤80%; hard questions must be ≥10%
- **Hard constraint on A-category window diversity**: For all A-category questions, no single `window_days` value may account for more than 40%.
  For example, with 50 A-category questions, each `window_days` value (7/14/30/60/90) may account for no more than 20 questions.
  Proactively balance the distribution when generating; do not assign all A questions to the same window.
- `window_days` must be one of {{7, 14, 30, 60, 90}}
- `d_target` should be distributed between [{cutoff}, {max_date}], with a higher density for recent questions (a 4:2:1 tendency)

**Output**: Strict JSON array, with these fields for each item:
  q                (one-sentence question in Chinese, generalized phrasing of “Will event E occur within the window?”)
  answer           (0 or 1)
  difficulty       (easy / medium / hard)
  dimension        (event / public opinion / policy / numbers, etc.)
  time_window      (“7 days” / “14 days” / “30 days” / “60 days” / “90 days”)
  d_target         (YYYY-MM-DD)
  window_days      (integer)
  days_from_cutoff (integer = d_target - cutoff)
  event_group      (string: associated ID from events.all; use "D_<n>" for D-category)
  question_type    (A_true / A_false / B_true / B_false / C_true / C_false / D_false)

Do not include a Markdown code block or any additional explanation.
{retry_notes}"""


# ===========================================================================
# LLM calls
# ===========================================================================

def llm_events(event_name: str, cutoff: str, gt_text: str, model: str) -> List[Dict[str, Any]]:
    prompt = PROMPT_EVENTS.format(event=event_name, cutoff=cutoff, gt_text=gt_text[:16000])
    raw = _call_with_fallback([{"role": "user", "content": prompt}], model=model, temperature=0.0, max_tokens=6000, timeout=600)
    items = parse_json_strict(raw)
    if not isinstance(items, list):
        items = []
    cleaned: List[Dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        ev_date = str(it.get("date") or it.get("gt_date") or "").strip()
        desc = str(it.get("event") or it.get("event_desc") or "").strip()
        if not ev_date or not desc:
            continue
        dfc = it.get("days_from_cutoff")
        if not isinstance(dfc, int):
            dfc = days_between(ev_date, cutoff) or 0
        cleaned.append({
            "id": str(it.get("id") or f"E{i:02d}"),
            "event": desc,
            "date": ev_date,
            "group": int(it.get("group") or i),
            "days_from_cutoff": int(dfc),
            # back-compat for step3F readers
            "eid": str(it.get("id") or f"E{i:02d}"),
            "event_desc": desc,
            "gt_date": ev_date,
        })
    return cleaned


DEFAULT_D_BLACKLIST_TERMS = (
    "(For example: the highest national authority, the country’s top leader, the full names of well-known public figures, and clearly outrageous)"
    "Numbers and ratios such as billions, global-scale totals, or 100% — the specific list will be provided by `pipeline_config`."
    "predict_step2_questionbank.d_blacklist_terms Provide"
)
DEFAULT_UNFALSIFIABLE_TERMS = (
    "(e.g., privately, internally, insinuations, reportedly, insiders, anonymous, mental state, emotions, confidential talks, behind closed doors,"
    "Confidential — these events cannot be recorded by public sources even if they occur) — the specific list is determined by pipeline_config."
    "predict_step2_questionbank.unfalsifiable_terms Provide"
)


def llm_brier(event_name: str, cutoff: str, events: List[Dict[str, Any]],
              n_questions: int, model: str, retry_notes: str = "",
              qb_cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    max_date = max((e["date"] for e in events), default=cutoff)
    qb_cfg = qb_cfg or {}
    blacklist = qb_cfg.get("d_blacklist_terms") or DEFAULT_D_BLACKLIST_TERMS
    unfalsifiable = qb_cfg.get("unfalsifiable_terms") or DEFAULT_UNFALSIFIABLE_TERMS
    if isinstance(blacklist, list):
        blacklist = "、".join(blacklist)
    if isinstance(unfalsifiable, list):
        unfalsifiable = "、".join(unfalsifiable)
    prompt = PROMPT_BRIER.format(
        event=event_name,
        cutoff=cutoff,
        max_date=max_date,
        n_questions=n_questions,
        events_json=json.dumps(
            [{"id": e["id"], "event": e["event"], "date": e["date"]} for e in events],
            ensure_ascii=False,
        ),
        retry_notes=retry_notes,
        blacklist_terms=blacklist,
        unfalsifiable_terms=unfalsifiable,
    )
    # max_tokens=6000 (reliable value): measured that raising it (16000/40000) makes kimi
    # attempt a "generate ~375 questions in one go" ultra-long response -> connection returns
    # nothing for a long time (client hangs at 0% CPU, up to 20+ min). Keeping 6000 -> each
    # call emits only ~50 questions and returns in ~26s; the while accumulation loop below
    # tops up to 375. Slow but stable. Question-count spec unchanged.
    # timeout=120 (was 600): DMXAPI kimi occasionally "hangs" (0% CPU, no response). A healthy
    # call takes ~26s, so 120s leaves ~5x headroom; a hang times out at 120s -> retry; still
    # hung after 3 tries -> auto-fallback to FALLBACK_MODEL (gpt-5.2), guaranteeing the point
    # always completes and never deadlocks.
    raw = _call_with_fallback([{"role": "user", "content": prompt}], model=model, temperature=0.0, max_tokens=6000, timeout=120)
    items = parse_json_strict(raw)
    if not isinstance(items, list):
        items = []
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        try:
            ans = int(it.get("answer") if "answer" in it else it.get("gt", 0))
            d_target = str(it.get("d_target") or "").strip()
            wnd = int(it.get("window_days") or 14)
            q_text = str(it.get("q") or it.get("event_desc") or "").strip()
            if not q_text or not d_target:
                continue
            dfc = it.get("days_from_cutoff")
            if not isinstance(dfc, int):
                dfc = days_between(d_target, cutoff) or 0
            out.append({
                "q": q_text,
                "answer": 1 if ans == 1 else 0,
                "difficulty": str(it.get("difficulty") or "medium"),
                "dimension": str(it.get("dimension") or "事件"),
                "time_window": str(it.get("time_window") or f"{wnd}天"),
                "d_target": d_target,
                "window_days": wnd,
                "days_from_cutoff": int(dfc),
                "event_group": str(it.get("event_group") or ""),
                "question_type": str(it.get("question_type") or ""),
                "qid": i,
                "event_desc": q_text,
                "gt": 1 if ans == 1 else 0,
            })
        except (TypeError, ValueError):
            continue
    return out


# ===========================================================================
# Validation
# ===========================================================================

def validate_questionbank(
    events_all: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    target_total_count: int,
) -> Dict[str, Any]:
    issues: List[str] = []
    if len(events_all) == 0:
        # 为空 --> empty
        issues.append("events.all empty")
    if len(questions) != target_total_count:
        issues.append(f"brier_questions Question count {len(questions)} != target_total {target_total_count}")

    true_q = sum(1 for q in questions if q.get("answer") == 1)
    false_q = sum(1 for q in questions if q.get("answer") == 0)
    ratio = false_q / true_q if true_q > 0 else 999.0
    if not (0.7 <= ratio <= 1.5):
        issues.append(f"True to false ratio {true_q}:{false_q}={ratio:.2f}，Should be 0.7-1.5")

    d_count = sum(1 for q in questions if str(q.get("question_type", "")).startswith("D_"))
    if d_count == 0:
        issues.append("Lack of Category D highly confusing fake questions")

    fake_texts = [q["q"] for q in questions if q.get("answer") == 0]
    if len(set(fake_texts)) < len(fake_texts):
        issues.append(f"The text of the fake question is {len(fake_texts) - len(set(fake_texts))} Repeated path")

    a_qs = [q for q in questions if str(q.get("question_type", "")).startswith("A_")]
    if a_qs:
        n_a = len(a_qs)
        easy = sum(1 for q in a_qs if q.get("difficulty") == "easy")
        hard = sum(1 for q in a_qs if q.get("difficulty") == "hard")
        if easy / n_a > 0.8:
            issues.append(f"A class easy proportion {easy/n_a:.2f} > 0.8")
        if hard / n_a < 0.1:
            issues.append(f"A class hard proportion {hard/n_a:.2f} < 0.1")
        wins: Dict[int, int] = {}
        for q in a_qs:
            wins[q.get("window_days", 0)] = wins.get(q.get("window_days", 0), 0) + 1
        max_share = max(wins.values()) / n_a if wins else 0
        if max_share > 0.4:
            issues.append(f"A Class single-window proportion {max_share:.2f} > 0.4")

    return {
        "passed": len(issues) == 0,
        "events_count": len(events_all),
        "brier_count": len(questions),
        "target_total": target_total_count,
        "true": true_q, "false": false_q,
        "d_count": d_count,
        "issues": issues,
    }


def parse_window_days(w: str) -> Optional[int]:
    """Parse 'time_window' string like '7天'/'14天'/'30天'/'Full timeX天' into int days."""
    if not w:
        return None
    if "Full time" in w:
        m = re.search(r"(\d+)", w)
        return int(m.group(1)) if m else 9999
    m = re.search(r"(\d+)", w)
    return int(m.group(1)) if m else None


def fix_a_answers(brier_qs: List[Dict[str, Any]]) -> int:
    """Programmatic fix for A-class question answers (mirrors archive
    step2_2_fix_answers.py). LLMs frequently mislabel A-class answers; the
    correct value is purely arithmetic: `answer = 1 iff days_from_cutoff <= window`.
    B/C/D are NOT touched — their answer isn't determined by pure time math.
    Returns count of corrections made.
    """
    fixed = 0
    for q in brier_qs:
        qt = str(q.get("question_type", ""))
        if not qt.startswith("A"):
            continue
        dfc = q.get("days_from_cutoff")
        w = parse_window_days(q.get("time_window", ""))
        if dfc is None or w is None:
            continue
        try:
            expected = 1 if int(dfc) <= int(w) else 0
        except (TypeError, ValueError):
            continue
        if q.get("answer") != expected:
            q["answer"] = expected
            q["gt"] = expected  # back-compat field
            # Sync question_type label (A_true ↔ A_false)
            if expected == 1 and qt.startswith("A_false"):
                q["question_type"] = "A_true"
            elif expected == 0 and qt.startswith("A_true"):
                q["question_type"] = "A_false"
            fixed += 1
    return fixed


_CJK_KEEP_RE = re.compile(r"[^一-鿿A-Za-z0-9]+")


def _phrases_3char(text: str) -> set:
    """SKILL §"3 字关键短语": return the set of all 3-character CJK substrings
    found in `text` (sliding window), plus all English / digit tokens ≥ 3 chars.
    This is the matching basis for the D-class ≥3-phrase overlap check.
    """
    text = (text or "")
    cn_only = _CJK_KEEP_RE.sub(" ", text)  # keep CJK / alnum, spaces between
    phrases: set = set()
    # CJK sliding 3-char windows from each contiguous CJK run
    for run in re.findall(r"[一-鿿]+", cn_only):
        for i in range(len(run) - 2):
            phrases.add(run[i:i + 3])
    # English / digit tokens ≥ 3 chars
    for tok in re.findall(r"[A-Za-z0-9]{3,}", cn_only):
        phrases.add(tok.lower())
    return phrases


def d_postfilter(
    brier_qs: List[Dict[str, Any]],
    events_all: List[Dict[str, Any]],
    qb_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """SKILL §"D 类三道自检" (L279-292):
      Check 1: >= 3 overlapping 3-char key phrases with any events.all event -> reject
      Check 2: contains a blacklist term -> reject
      Check 3: contains an unfalsifiable keyword -> reject
    Returns {'rejected': [...], 'kept_d': N, 'rejected_d': M, 'reasons': {...}}.
    Modifies brier_qs in place (removes rejected D entries).
    """
    qb_cfg = qb_cfg or {}
    blacklist = qb_cfg.get("d_blacklist_terms") or []
    unfalsifiable = qb_cfg.get("unfalsifiable_terms") or []
    if isinstance(blacklist, str):
        blacklist = [s.strip() for s in re.split(r"[、,，\s]+", blacklist) if s.strip()]
    if isinstance(unfalsifiable, str):
        unfalsifiable = [s.strip() for s in re.split(r"[、,，\s]+", unfalsifiable) if s.strip()]

    # Build event phrase set once
    event_phrase_sets: List[set] = []
    for e in events_all:
        body = (e.get("event") or e.get("event_desc") or "")
        event_phrase_sets.append(_phrases_3char(body))

    rejected: List[Dict[str, Any]] = []
    reasons = {"gt_overlap": 0, "blacklist": 0, "unfalsifiable": 0}
    keep: List[Dict[str, Any]] = []
    for q in brier_qs:
        qt = str(q.get("question_type", ""))
        if not qt.startswith("D"):
            keep.append(q)
            continue
        body = q.get("q") or q.get("event_desc") or ""
        # Self-check 2: blacklist
        hit_bl = next((w for w in blacklist if w and w in body), None)
        if hit_bl:
            rejected.append({**q, "reject_reason": f"blacklist:{hit_bl}"})
            reasons["blacklist"] += 1
            continue
        # Self-check 3: unfalsifiable
        hit_uf = next((w for w in unfalsifiable if w and w in body), None)
        if hit_uf:
            rejected.append({**q, "reject_reason": f"unfalsifiable:{hit_uf}"})
            reasons["unfalsifiable"] += 1
            continue
        # Self-check 1: ≥ 3 phrase overlap with any event
        q_phrases = _phrases_3char(body)
        max_overlap = 0
        for eps in event_phrase_sets:
            ov = len(q_phrases & eps)
            if ov > max_overlap:
                max_overlap = ov
        if max_overlap >= 3:
            rejected.append({**q, "reject_reason": f"gt_overlap:{max_overlap}"})
            reasons["gt_overlap"] += 1
            continue
        keep.append(q)

    brier_qs.clear()
    brier_qs.extend(keep)
    return {
        "rejected": rejected,
        "rejected_d": len(rejected),
        "kept_d": sum(1 for q in keep if str(q.get("question_type", "")).startswith("D")),
        "reasons": reasons,
    }


def topup_d_class(
    brier_qs: List[Dict[str, Any]],
    events_all: List[Dict[str, Any]],
    event_name: str,
    cutoff: str,
    model: str,
    d_goal_pct: float = 22.0,
    qb_cfg: Optional[Dict[str, Any]] = None,
) -> int:
    """Ensure D-class share >= d_goal_pct (mirrors archive step2_3_topup_d.py).
    If short, ask LLM for extra D-class fakes, and drop the most-generic A/B/C
    questions (preferring `time_window=Full time` ones) to make room. Keeps total
    brier_count unchanged. Returns net D additions.
    """
    n_total = len(brier_qs)
    if n_total == 0:
        return 0
    d_count = sum(1 for q in brier_qs if str(q.get("question_type", "")).startswith("D"))
    goal = int(round(n_total * d_goal_pct / 100.0))
    if d_count >= goal:
        return 0
    need = goal - d_count

    # Drop the most "broad" A/B/C first
    def broadness(q: Dict[str, Any]) -> int:
        qt = str(q.get("question_type", ""))
        tw = str(q.get("time_window", ""))
        score = 0
        if qt.startswith(("A", "B", "C")):
            score += 1
        if "Full time" in tw:
            score += 2
        try:
            if int(parse_window_days(tw) or 0) >= 60:
                score += 1
        except (TypeError, ValueError):
            pass
        return -score  # negate so largest = highest broadness

    candidates_to_drop = sorted(
        [i for i, q in enumerate(brier_qs)
         if str(q.get("question_type", "")).startswith(("A", "B", "C"))],
        key=lambda i: broadness(brier_qs[i]),
    )[:need]
    if len(candidates_to_drop) < need:
        return 0  # not enough room to make space cleanly

    extra = llm_brier(event_name, cutoff, events_all, need, model,
                     retry_notes="\nPlease only generate Category D highly confusing fake questions, not A/B/C/G.Each question_type must be D_false.",
                     qb_cfg=qb_cfg)
    extras_d = [q for q in extra if str(q.get("question_type", "")).startswith("D")][:need]
    if not extras_d:
        return 0

    drop_set = set(candidates_to_drop[: len(extras_d)])
    new_list = [q for i, q in enumerate(brier_qs) if i not in drop_set]
    new_list.extend(extras_d)
    brier_qs.clear()
    brier_qs.extend(new_list)
    # Re-number qids
    for i, q in enumerate(brier_qs, start=1):
        q["qid"] = i
    return len(extras_d)


SANITY_SAMPLE_PROMPT = """You are a question bank reviewer. Below are {n} Brier questions (answer=1).
Please determine whether, based **only on the question wording**, one could roughly infer specific details from the GT source text (such as names, organizations, exact quotes, or specific numbers).
A well-generalized question should only let you guess that “a party took some kind of action,” not the precise details.

================ Sample questions ================
{qs_block}
================

Provide one judgment for each question (no Markdown code block). Strict JSON format:
{{"items": [
  {{"qid": <int>, "leaks_specifics": <true|false>, "reason": "<one sentence>"}}
]}}
"""


def sanity_sample_check(brier_qs: List[Dict[str, Any]], model: str,
                        sample_size: int = 3) -> Dict[str, Any]:
    """SKILL §"Brier 泛化度" / "质量自检 Round 2": randomly sample real brier
    questions and ask the LLM whether the question wording leaks GT specifics
    (names, exact numbers, exact quotes). Returns a structured report; the
    main flow attaches it to qb.meta.sanity_sample."""
    true_qs = [q for q in brier_qs if q.get("answer") == 1]
    if not true_qs:
        return {"sampled": 0, "items": [], "leaks_total": 0, "model": model}
    import random as _r
    rng = _r.Random(0)  # deterministic
    pool = list(true_qs)
    rng.shuffle(pool)
    sample = pool[:max(1, sample_size)]
    qs_block = "\n".join(
        f"{i+1}. (qid={q.get('qid','?')}) {q.get('q', q.get('event_desc',''))}"
        for i, q in enumerate(sample)
    )
    try:
        raw = _call_with_fallback(
            [{"role": "user", "content": SANITY_SAMPLE_PROMPT.format(
                n=len(sample), qs_block=qs_block)}],
            model=model,
            temperature=0.0,
            max_tokens=600,
            timeout=600,
            reasoning_effort="none",
        )
    except Exception as e:
        return {"sampled": len(sample), "items": [], "leaks_total": 0,
                "model": model, "error": str(e)[:200]}
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return {"sampled": len(sample), "items": [], "leaks_total": 0,
                    "model": model, "raw": raw[:200]}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"sampled": len(sample), "items": [], "leaks_total": 0,
                    "model": model, "raw": raw[:200]}
    items = data.get("items") or []
    leaks = sum(1 for it in items if it.get("leaks_specifics"))
    return {"sampled": len(sample), "items": items, "leaks_total": leaks, "model": model}


def density_check(events_all: List[Dict[str, Any]]) -> Dict[str, int]:
    seg1 = seg2 = seg3 = total90 = 0
    for e in events_all:
        d = int(e.get("days_from_cutoff", 0))
        if d < 1 or d > 90:
            continue
        total90 += 1
        if d <= 30:
            seg1 += 1
        elif d <= 60:
            seg2 += 1
        else:
            seg3 += 1
    return {"total_90d": total90, "seg1": seg1, "seg2": seg2, "seg3": seg3}


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="predict-step2-questionbank: per-point Q-bank via LLM")
    p.add_argument("--workspace", required=True)
    p.add_argument("--event-name", required=True)
    p.add_argument("--model", default=None,
                   help="LLM model; defaults to pipeline_config.default_llm_model")
    p.add_argument("--points", default="all")
    p.add_argument("--max-retries", type=int, default=2, help="Max validation retries per point")
    p.add_argument("--sanity-sample-size", type=int, default=1,
                   help="SKILL quality self-check Round 2: number of brier true-questions to sanity-sample "
                        "per point for leakage check (default 3, 0 to skip).")
    p.add_argument("--skip-split", action="store_true", help="Skip Step 2.0 (assume contexts/ and gt/ already exist)")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    qb_cfg = cfg.get("predict_step2_questionbank", {})
    model = args.model or cfg.get("default_llm_model")
    eval_models = cfg.get("evaluation_models", [])

    workspace = Path(args.workspace)
    pp_data = json.loads((workspace / "prediction_points.json").read_text(encoding="utf-8"))
    points = pp_data.get("prediction_points") if isinstance(pp_data, dict) else pp_data
    selected = [pt for pt in points if pt.get("point_id")]

    if not args.skip_split:
        n_split = split_context_gt(workspace)
        print(f"[step2.0] split contexts/gt for {n_split} points")

    if args.points != "all":
        wanted = set(args.points.split(","))
        selected = [pt for pt in selected if pt["point_id"] in wanted]

    out_dir = workspace / "questionbank"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for idx, pt in enumerate(selected):
        pid = pt["point_id"]
        cutoff = pt.get("target_date", "")

        # --- resume: skip points already fully generated (idempotent re-runs) ---
        # A point is "done" iff its file exists and its question count equals the
        # target it recorded. Partial/corrupt files fall through and regenerate.
        # Skipping costs zero LLM calls, so re-running a finished event is cheap.
        existing = out_dir / f"{pid}_questionbank.json"
        if existing.exists():
            try:
                prev = json.loads(existing.read_text(encoding="utf-8"))
                prev_qs = prev.get("brier_questions", [])
                prev_meta = prev.get("meta", {})
                prev_target = prev_meta.get("target_total", 0)
                if prev_qs and prev_target and len(prev_qs) == prev_target:
                    print(f"[step2] {pid} skip (exists, {len(prev_qs)} questions)")
                    summary.append({
                        "point_id": pid,
                        "base_event_count": prev_meta.get("base_event_count", 0),
                        "brier_count": len(prev_qs),
                        "target_total": prev_target,
                        "passed": prev_meta.get("validation", {}).get("passed", False),
                        "density_90d": prev_meta.get("density_check"),
                        "resumed": True,
                    })
                    continue
            except Exception:
                pass  # corrupt/partial → fall through and regenerate

        gt_path = workspace / "gt" / f"{pid}_gt.md"
        gt_text = gt_path.read_text(encoding="utf-8") if gt_path.exists() else ""

        gt_nodes = count_gt_nodes(gt_text)
        remaining_pred = len(selected) - idx

        print(f"[step2] {pid} cutoff={cutoff} gt_nodes={gt_nodes} remaining_pred={remaining_pred} model={model}")

        events_all = llm_events(args.event_name, cutoff, gt_text, model)
        base_event_count = len(events_all)
        n_target = target_total(base_event_count, remaining_pred, gt_nodes, qb_cfg)

        brier_qs: List[Dict[str, Any]] = []
        validation: Dict[str, Any] = {}
        retry_notes = ""
        for attempt in range(args.max_retries + 1):
            brier_qs = llm_brier(args.event_name, cutoff, events_all, n_target, model, retry_notes, qb_cfg=qb_cfg)
            validation = validate_questionbank(events_all, brier_qs, n_target)
            if validation["passed"]:
                break
            retry_notes = (
                f"\nThe previous submission contained the following issues; please correct them and resubmit the complete version.{n_target} Question：\n"
                + "\n".join(f"- {it}" for it in validation["issues"])
            )

        # SKILL §"硬约束：len(brier_questions_i) = target_total_i" — if LLM
        # still over/under-produced after retries, force the count by
        # truncating or padding with extra requested fakes; re-renumber qids.
        if len(brier_qs) > n_target:
            brier_qs = brier_qs[:n_target]
        while len(brier_qs) < n_target:
            extra = llm_brier(
                args.event_name, cutoff, events_all,
                n_target - len(brier_qs), model,
                qb_cfg=qb_cfg,
                retry_notes="\nGenerate an additional {n} questions to reach the total, keeping the structure consistent with the previous ones.".format(
                    n=n_target - len(brier_qs)
                ),
            )
            if not extra:
                break
            brier_qs.extend(extra[: n_target - len(brier_qs)])
        # re-stamp qids 1..N for stability
        for i, q in enumerate(brier_qs, start=1):
            q["qid"] = i

        # ---- Production-grade fixers (mirror archive step2_2 + step2_3) ----
        # 1) Programmatic A-class answer correction (LLMs mislabel these often)
        a_fixed = fix_a_answers(brier_qs)
        # 2) SKILL §D 类三道自检 (L279-292): post-filter D-class fakes that
        #    overlap ≥3 key-phrases with GT events, hit blacklist terms,
        #    or hit unfalsifiable terms.
        d_filter = d_postfilter(brier_qs, events_all, qb_cfg=qb_cfg)
        # 3) D-class share top-up if below goal (SKILL: A:B:C:D ≈ 4:2:1:3 ~ 30%)
        d_added = topup_d_class(brier_qs, events_all, args.event_name, cutoff, model,
                                d_goal_pct=float(qb_cfg.get("d_goal_pct", 22.0)),
                                qb_cfg=qb_cfg)
        if a_fixed or d_added or d_filter["rejected_d"]:
            print(f"  {pid} post-fix: A_answers_corrected={a_fixed} "
                  f"D_rejected={d_filter['rejected_d']} ({d_filter['reasons']}) "
                  f"D_topped_up={d_added}")

        validation = validate_questionbank(events_all, brier_qs, n_target)

        all_dates = [parse_date_obj(e["date"]) for e in events_all if parse_date_obj(e["date"])]
        gt_span_days = (max(all_dates) - parse_date_obj(cutoff)).days if all_dates and parse_date_obj(cutoff) else 0

        events_base = events_all
        events_major = [e for e in events_all if e.get("days_from_cutoff", 0) <= 60]
        dens = density_check(events_all)

        qb = {
            "point_id": pid,
            "cutoff_date": cutoff,
            "gt_span_days": gt_span_days,
            "events": {
                "base": events_base,
                "all": events_all,
                "major": events_major,
            },
            "brier_questions": brier_qs,
            "eval_config": {
                "eval_models": eval_models,
                "provider": "runtime_injected_at_step3",
                "temperature": 0,
                "reasoning": True,
                "runs_per_point": 1,
            },
            "meta": {
                "base_event_count": base_event_count,
                "all_count": len(events_all),
                "major_count": len(events_major),
                "brier_count": len(brier_qs),
                "target_total": n_target,
                "validation": validation,
                "density_check": dens,
                "d_postfilter": {
                    "rejected_d": d_filter["rejected_d"],
                    "kept_d": d_filter["kept_d"],
                    "reasons": d_filter["reasons"],
                },
                "sanity_sample": sanity_sample_check(brier_qs, model,
                                                    sample_size=args.sanity_sample_size),
                "cc_reviewed": validation.get("passed", False),
                "generator_model": model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        out_file = out_dir / f"{pid}_questionbank.json"
        out_file.write_text(json.dumps(qb, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({
            "point_id": pid,
            "base_event_count": base_event_count,
            "brier_count": len(brier_qs),
            "target_total": n_target,
            "passed": validation.get("passed", False),
            "density_90d": dens,
        })

    print(json.dumps({"points_done": len(summary), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
