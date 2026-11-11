#!/usr/bin/env python3
"""predict-step3F-time (1:1 of skills/predict-step3F-time/SKILL.md, v6.2).

For each prediction point × evaluated model:

  1. Read context (pre-cutoff) and questionbank's `events.all` (compact real
     events; sanity-checked against `meta.base_event_count`).
  2. Filter to days_from_cutoff <= 90 (WINDOW_DAYS); annotate each event with
     its 30-day segment date range using SEGMENTS = [(0,30), (31,60), (61,90)].
  3. Ask the eval model for one date per event in `1. YYYY-MM-DD` line format,
     temperature=0, reasoning_effort="high", include_reasoning=True,
     thinking.budget_tokens=24000, max_tokens=32000.
  4. **Run 1 time per point** (2026-07-01 user decision; was 2-run-avg —
     measured 1-vs-2 mean diff < run-to-run noise, single-point variance is
     diluted by cross-point pooling instead).
  5. Compute baseline wmae using segment midpoints (15.5 / 45.5 / 75.5) and
     score_100 = 100 / (1 + wmae_days / baseline_wmae).
  6. Post-run self-scan: verify every (model × point) has the expected number
     of runs; if any missing, re-run those up to 2 attempts.
  7. After all points done, emit aggregated.json with the pooled score.

Output per-point: <workspace>/results/run_<ts>/time/<model_slug>/P*_time.json
  Top-level fields (consumed by predict-step4-scorecard):
    events    : list with {eid, event_desc, gt_date, pred_date, error_days, delta_days, n_runs}
    score_100 / weighted_mae_days / baseline_wmae / predictions / events_total
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import call_llm, load_pipeline_config, parse_date_obj


WINDOW_DAYS = 90
ALPHA = 0.04  # time-weight coefficient of paper Eq. (3): w_time = 1/(1 + 0.04·t_e)
# The three 30-day buckets [l_e, u_e] of paper Sec. 3.3 "Temporal score"
# (days 0-30 / 31-60 / 61-90 after the cutoff); third tuple entry = midpoint.
SEGMENTS = [(0, 30, 15.5), (31, 60, 45.5), (61, 90, 75.5)]  # 2026-07-01 d=0 fix: segment 1 includes the cutoff day itself (old (1,30) missed d=0 → prompt had no window + baseline midpoint wrongly fell to 45.5)

# Whole-point retry policy (fixed by user 2026-06-30, aligned with 3B) -----------
# Each prediction point gets at most 3 requests, waiting RETRY_WAIT_SECONDS (default
# 60s) between attempts; attempt 1 asks all events of the point, later attempts only
# re-ask events "still without a date / request failed". Still no date after 3
# attempts → default-fill with the midpoint date of the event's 30-day segment and
# mark answer_attempt=null. Each event carries answer_attempt: 1/2/3 = which attempt
# yielded the date; null = segment-midpoint default. Each point is evaluated only
# once (fixed by user 2026-07-01, replacing the old "average of 2 runs"). Request
# failures and parse failures are accounted separately.
MAX_ANSWER_ATTEMPTS = int(os.environ.get("SB_TIME_MAX_ANSWER_ATTEMPTS", "3"))  # env-tunable; agent time_point reads it via getattr(F,...), B policy sets it very large
RETRY_WAIT_SECONDS = int(os.environ.get("SB_RETRY_WAIT_SECONDS", "60"))  # Fixed by user 2026-06-30: 300→60; can be small (e.g. 1) for tests


# ===========================================================================
# Segment helpers
# ===========================================================================

def segment_midpoint(day: int) -> float:
    for lo, hi, mid in SEGMENTS:
        if lo <= day <= hi:
            return mid
    return 45.5  # fallback; >90 events should be filtered upstream


def segment_date_range(cutoff_str: str, days_from_cutoff: int) -> Tuple[Optional[str], Optional[str]]:
    cutoff = parse_date_obj(cutoff_str)
    if cutoff is None:
        return None, None
    for lo, hi, _mid in SEGMENTS:
        if lo <= days_from_cutoff <= hi:
            return (
                (cutoff + timedelta(days=lo)).isoformat(),
                (cutoff + timedelta(days=hi)).isoformat(),
            )
    return None, None


# ===========================================================================
# Prompt + parsing
# ===========================================================================

PROMPT_TEMPLATE = """{context}

---

以下 {n} 件事情**都会在未来发生**（列表顺序不代表时间先后）。每件事标注了发生的时间区间，请在该区间内预测最可能的具体日期（YYYY-MM-DD 格式）。时间区间仅表示该事件会在此区间内发生，无暗示一定在中间附近。

只输出编号和日期，格式：
1. 2025-08-12
2. 2025-08-15

事件列表：
{event_list}
"""


# B-style "exam" prompt (fixed by user 2026-07-01: consistent with 3B, explains the scoring principle). Enabled by default (SB_EXAM_PROMPT=0 falls back to the basic template).
# Explains only clean principles: smaller error = higher score, perfect = 100, beating the naive reference scores high otherwise low (hyperbolic, never negative), near-cutoff events weigh more;
# deliberately omits concrete numbers like "midpoint = 50" (measured: half-day midpoint + whole-day predictions + front-loaded events → not exactly 50, would mislead); guides the model to derive a specific date from the materials rather than being lazy.
EXAM_PROMPT_TEMPLATE = """你正在参加一场**预测考试**。以下是一个公共事件截至 {cutoff} 的已知信息：

================ 已知信息（截止 {cutoff}）================
{context}
================

下面 {n} 件事**都会在未来发生**（列表顺序不代表时间先后），每件都标注了一个 30 天时间区间——事件一定落在该区间内，但**不一定在区间中点附近**。请为每件事预测**最可能的具体发生日期**（YYYY-MM-DD 格式）。

================ 评分规则（这是考试，请据此争取最高分）================
- 单题误差 = |你预测的日期 − 真实发生日期|（单位：天）。**误差越小得分越高，完美（0 天误差）= 100 分；越不准分越低**（分数永不为负）。
- 加权：越临近 {cutoff} 的事件占分越重，请优先把这类判准。
================

所以**别敷衍**（别不看材料、别随手在区间里给个日期）——区间只是个粗范围，真正考的是你能否据已知信息推理出每件事**最可能的那一天**，越准越高分。

作答前请**好好思考、好好推演**，想清楚每件事再下笔（思考在心里即可，输出只给日期）。

**输出格式**：只输出题号和日期，每题一行，例如：
1. 2025-08-12
2. 2025-08-15
不要添加任何额外说明、不要解释。

================ 待预测事件 ================
{event_list}
"""


# ---- English prompts (2026-07-01: English evals use English instructions, sentence-by-sentence mirror of the Chinese version above; main() switches when the workspace path contains "英文") ----
PROMPT_TEMPLATE_EN = """{context}

---

The following {n} events **will all happen in the future** (list order does NOT imply chronological order). Each event is annotated with the time window in which it occurs; predict the single most likely specific date within that window (YYYY-MM-DD format). The window only means the event happens within it — it does NOT imply the date is near the middle.

Output only the number and date, format:
1. 2025-08-12
2. 2025-08-15

Event list:
{event_list}
"""

EXAM_PROMPT_TEMPLATE_EN = """You are taking a **forecasting exam**. Below is the known information about a public event, as of {cutoff}:

================ Known information (as of {cutoff}) ================
{context}
================

The following {n} events **will all happen in the future** (list order does NOT imply chronological order). Each is annotated with a 30-day time window — the event is guaranteed to fall within it, but **not necessarily near the middle**. For each event, predict the single **most likely specific date** (YYYY-MM-DD format).

================ Scoring rules (this is an exam — aim for the highest score) ================
- Per-event error = |your predicted date − the true date| (in days). **The smaller the error, the higher the score; a perfect prediction (0 days off) = 100; the less accurate, the lower** (the score is never negative).
- Weighting: events closer to {cutoff} count more — prioritize getting those right.
================

So **do not be lazy** (do not ignore the materials, and do not just throw out a random date in the window) — the window is only a rough range; what is tested is whether you can reason from the known information to the single **most likely day** for each event. The closer, the higher.

Take your time to **think it through and reason carefully** before answering (reason internally; output only the dates).

**Output format**: output only the number and date, one per line, e.g.:
1. 2025-08-12
2. 2025-08-15
Do not add any extra explanation.

================ Events to predict ================
{event_list}
"""


DATE_LINE_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})", re.M)


def parse_dates(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in DATE_LINE_RE.finditer(raw):
        idx = m.group(1)
        date_str = m.group(2).replace("/", "-").replace(".", "-")
        parts = date_str.split("-")
        if len(parts) == 3:
            try:
                date_str = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            except ValueError:
                continue
        out[idx] = date_str
    return out


def render_event_list(events: List[Dict[str, Any]], cutoff: str) -> str:
    lines = []
    for i, e in enumerate(events, start=1):
        seg_start, seg_end = segment_date_range(cutoff, int(e.get("days_from_cutoff", 0)))
        body = e.get("event") or e.get("event_desc") or ""
        if seg_start and seg_end:
            lines.append(f"{i}. [{seg_start} ~ {seg_end}] {body}")
        else:
            lines.append(f"{i}. {body}")
    return "\n".join(lines)


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


# ===========================================================================
# Single-run call
# ===========================================================================

def call_for_dates(
    context: str,
    cutoff: str,
    events: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> Dict[str, str]:
    if os.environ.get("SB_EXAM_PROMPT", "1") != "0":   # default: B-style exam prompt (explains the principle); set SB_EXAM_PROMPT=0 for the basic template
        prompt = EXAM_PROMPT_TEMPLATE.format(
            n=len(events),
            cutoff=cutoff,
            context=context,
            event_list=render_event_list(events, cutoff),
        )
    else:
        prompt = PROMPT_TEMPLATE.format(
            n=len(events),
            context=context,
            event_list=render_event_list(events, cutoff),
        )
    kwargs = dict(model=model, temperature=0.0, max_tokens=max_tokens)
    # Fixed by user 2026-06-30: non-thinking models don't think. SB_NO_THINKING=1 → omit thinking params.
    if os.environ.get("SB_NO_THINKING", "0") == "0":
        kwargs["reasoning_effort"] = "high"
        kwargs["include_reasoning"] = True
        if "gpt" not in model.lower():
            kwargs["thinking_budget_tokens"] = thinking_budget
    raw = call_llm([{"role": "user", "content": prompt}], **kwargs)
    return parse_dates(raw)


# ===========================================================================
# Scoring
# ===========================================================================

def score_events(events: List[Dict[str, Any]], dates: Dict[str, str]) -> Dict[str, Any]:
    """Compute MAE / weighted MAE / baseline wmae / score_100 for one run."""
    weighted_errors = []
    weights = []
    errors = []
    details = []
    for i, e in enumerate(events, start=1):
        gt_date = e.get("date") or e.get("gt_date")
        gt_d = parse_date_obj(gt_date)
        pd_str = dates.get(str(i))
        pd_d = parse_date_obj(pd_str) if pd_str else None
        if gt_d and pd_d:
            err = abs((pd_d - gt_d).days)
            errors.append(err)
            days_fc = int(e.get("days_from_cutoff", 0) or 0)
            w = 1.0 / (1.0 + ALPHA * max(0, days_fc))
            weighted_errors.append(err * w)
            weights.append(w)
            details.append({**e, "pred_date": pd_str, "error_days": err, "weight": round(w, 4)})
        else:
            details.append({**e, "pred_date": pd_str, "error_days": None})

    if not errors:
        return {
            "mae_days": None, "weighted_mae_days": None,
            "baseline_wmae": None, "score_100": None,
            "events_scored": 0, "events_total": len(events),
            "sum_weights": 0, "details": details,
        }

    mae = sum(errors) / len(errors)
    weighted_mae = sum(weighted_errors) / sum(weights) if sum(weights) > 0 else mae

    # Segment-midpoint baseline
    w_sum = 0.0
    base_w_err = 0.0
    for e in events:
        d = int(e.get("days_from_cutoff", 0) or 0)
        w = 1.0 / (1.0 + ALPHA * max(0, d))  # 2026-07-02: consistent with the model side + guard against negative d / division by zero
        mid = segment_midpoint(d)
        w_sum += w
        base_w_err += abs(d - mid) * w
    baseline_wmae = base_w_err / w_sum if w_sum > 0 else 0
    score_100 = 100.0 / (1.0 + weighted_mae / baseline_wmae) if baseline_wmae > 0 else 0.0

    return {
        "mae_days": round(mae, 2),
        "weighted_mae_days": round(weighted_mae, 2),
        "baseline_wmae": round(baseline_wmae, 2),
        "score_100": round(score_100, 2),
        "events_scored": len(errors),
        "events_total": len(events),
        "sum_weights": round(sum(weights), 4),
        "details": details,
    }


def average_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in runs if r.get("weighted_mae_days") is not None]
    if not valid:
        return runs[0] if runs else {}
    avg_wmae = sum(r["weighted_mae_days"] for r in valid) / len(valid)
    baseline = valid[0]["baseline_wmae"]
    score_100 = 100.0 / (1.0 + avg_wmae / baseline) if baseline and baseline > 0 else 0.0
    # Average error_days per event index across runs
    n = max(len(r.get("details", [])) for r in valid)
    averaged_details: List[Dict[str, Any]] = []
    for i in range(n):
        per_event_errors: List[int] = []
        seen_meta: Optional[Dict[str, Any]] = None
        last_pred: Optional[str] = None
        for r in valid:
            d_list = r.get("details", [])
            if i < len(d_list):
                d = d_list[i]
                seen_meta = seen_meta or d
                if d.get("error_days") is not None:
                    per_event_errors.append(d["error_days"])
                if d.get("pred_date"):
                    last_pred = d["pred_date"]
        if seen_meta:
            averaged_details.append({
                **{k: v for k, v in seen_meta.items() if k != "error_days"},
                "pred_date": last_pred,
                "error_days": (sum(per_event_errors) / len(per_event_errors))
                              if per_event_errors else None,
                "n_runs": len(per_event_errors),
            })
    return {
        "mae_days": round(sum(r["mae_days"] for r in valid) / len(valid), 2),
        "weighted_mae_days": round(avg_wmae, 2),
        "baseline_wmae": baseline,
        "score_100": round(score_100, 2),
        "events_scored": valid[0].get("events_scored", 0),
        "events_total": valid[0].get("events_total", 0),
        "sum_weights": valid[0].get("sum_weights", 0),
        "details": averaged_details,
        "n_runs": len(valid),
    }


# ===========================================================================
# Per-point orchestration (1 run per point, 2026-07-01)
# ===========================================================================

def run_point(
    pid: str,
    qb_path: Path,
    ctx_path: Path,
    model: str,
    runs_per_point: int,
    max_tokens: int,
    thinking_budget: int,
) -> Dict[str, Any]:
    qb = json.loads(qb_path.read_text(encoding="utf-8"))
    cutoff = qb.get("cutoff_date") or ""
    all_events_raw = qb.get("events", {}).get("all", []) if isinstance(qb.get("events"), dict) else qb.get("events", [])
    base_count = qb.get("meta", {}).get("base_event_count")
    if base_count is not None and len(all_events_raw) != base_count:
        print(f"  ⚠️ events.all ({len(all_events_raw)}) ≠ base_event_count ({base_count})")

    events = [e for e in all_events_raw if int(e.get("days_from_cutoff", 0) or 0) <= WINDOW_DAYS]
    # SKILL section "every point must participate, no skipping": even when no events fall in the 90-day
    # window, return a structured record (n=0) so the point is recorded — Step 4
    # / Step 4 pooled scorers filter it out via wmae=None, but the point ISN'T
    # silently dropped from the bookkeeping.
    if not events:
        print(f"  ⚠️ {pid} has no events in 90-day window (raw_total={len(all_events_raw)}) — "
              f"recording empty result, will be pooled-out at step4 by wmae=None.")
        return {
            "point_id": pid, "model": model, "events_total": 0,
            "score_100": None, "events": [], "predictions": [],
            "per_run": [], "n_runs_completed": 0,
            "density_check": {"total_90d": 0, "seg1": 0, "seg2": 0, "seg3": 0,
                              "raw_total": len(all_events_raw), "status": "empty"},
        }

    # Density check warning — record verdict, propagate to step4
    seg_counts = [0, 0, 0]
    for e in events:
        d = int(e.get("days_from_cutoff", 0) or 0)
        if 0 <= d <= 30:
            seg_counts[0] += 1
        elif 31 <= d <= 60:
            seg_counts[1] += 1
        elif 61 <= d <= 90:
            seg_counts[2] += 1
    density_status = "ok"
    if len(events) < 5 or any(c == 0 for c in seg_counts):
        density_status = "weak"
        print(f"  ⚠️ {pid} density check weak: total_90d={len(events)} "
              f"seg=[{seg_counts[0]},{seg_counts[1]},{seg_counts[2]}] — "
              "scoring will still run; consider rerunning Step 2 with denser events.all "
              "(SKILL §3F 密度自检 says: don't fabricate events, but Step 2 should aim ≥5 / "
              "all 3 segments non-empty).")

    context = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""

    # SKILL section on tiered max_tokens rules (line 24-26):
    # Default short events:    max_tokens=6000, thinking.budget_tokens=4000
    # Long events (context >= 10K tokens): max_tokens=8000, thinking.budget_tokens=4000
    # Going above 10000 is discouraged (avoids reasoning divergence)
    approx_tokens = len(context) / 2.5
    if approx_tokens >= 10000 and max_tokens < 8000:
        max_tokens = 8000
        print(f"  {pid} long context (~{int(approx_tokens)} tok) → max_tokens=8000, thinking=4000")

    # ---- F = evaluate runs_per_point times (default 1); each run internally uses "whole-point 3-attempt retry" ----
    # (Fixed by user 2026-07-01: F runs each point only once — measured 1-vs-2 mean diff ~0.9
    #  < run-to-run noise (±2.3), saving ~1/3 of cost; each run retries at most 3 times, 60s
    #  apart; still no date after 3 attempts → segment-midpoint default and mark
    #  answer_attempt=null. Request failures and parse failures are accounted separately.)
    n_ev = len(events)
    cutoff_d = parse_date_obj(cutoff)

    def one_run(run_idx: int):
        """One evaluation run: each event gets up to 4 retries (3 normal; a 4th if DMX questions remain after the 3rd) to obtain a date.
        v3 (fixed by user 2026-07-02): failed events are NOT midpoint-filled into the primary score; primary counts answered events only, secondary = all events (failures filled with midpoint)."""
        pred_dates: List[Optional[str]] = [None] * n_ev   # real answered dates (failure = None)
        answer_attempt: List[Optional[int]] = [None] * n_ev
        last_fail_kind: List[Optional[str]] = [None] * n_ev
        attempt_records: List[Dict[str, Any]] = []
        missing = set(range(n_ev))
        attempt = 0
        while missing:
            if attempt >= MAX_ANSWER_ATTEMPTS + 1:
                break
            if attempt >= MAX_ANSWER_ATTEMPTS:
                if not any((last_fail_kind[i] or "").startswith("request_error") for i in missing):
                    break  # 3 full attempts and no DMX questions → stop (no 4th attempt)
            attempt += 1
            if attempt > 1:
                print(f"  {model} / {pid} run{run_idx+1} attempt {attempt}: "
                      f"等待 {RETRY_WAIT_SECONDS}s 后重问 {len(missing)} 个未给日期事件 ...", flush=True)
                time.sleep(RETRY_WAIT_SECONDS)
            sub_idx = sorted(missing)
            sub_events = [events[i] for i in sub_idx]
            kind = None
            try:
                sub_dates = call_for_dates(context, cutoff, sub_events, model, max_tokens, thinking_budget)
            except Exception as e:  # request failure (incl. quota): accounted separately from "model gave no date"
                sub_dates = {}
                kind = f"request_error:{str(e)[:120]}"
                print(f"  {model} / {pid} run{run_idx+1} attempt {attempt} 请求失效({len(sub_events)}): {e}", flush=True)
            n_got = 0
            for pos, orig in enumerate(sub_idx, start=1):
                ds = sub_dates.get(str(pos))
                if ds:
                    pred_dates[orig] = ds
                    answer_attempt[orig] = attempt
                    missing.discard(orig)
                    n_got += 1
                else:
                    last_fail_kind[orig] = kind or "parse_fail"
            attempt_records.append({
                "run": run_idx + 1, "attempt": attempt, "asked": len(sub_idx),
                "answered": n_got, "still_missing": len(missing), "request_error": 1 if kind else 0,
            })
            print(f"  {model} / {pid} run{run_idx+1} attempt {attempt}: 拿到 {n_got}/{len(sub_idx)}, 仍缺 {len(missing)}", flush=True)
        # v3: classify failed events (dmx=request_error/quota / model=parse_fail); no midpoint fill into the primary score
        fail_kinds: List[Optional[str]] = [None] * n_ev
        for i in missing:
            lk = last_fail_kind[i] or ""
            fail_kinds[i] = "dmx" if lk.startswith(("request_error", "quota")) else "model"
        n_dmx = sum(1 for i in missing if fail_kinds[i] == "dmx")
        n_model = sum(1 for i in missing if fail_kinds[i] == "model")
        if missing:
            print(f"  {model} / {pid} run{run_idx+1} 重试后 {len(missing)} 事件无日期 → 标无效"
                  f"(dmx={n_dmx}, model={n_model};主分排除、副分填中点)", flush=True)
        # Primary score: feed only the real answered dates
        answered_map = {str(i + 1): pred_dates[i] for i in range(n_ev) if pred_dates[i]}
        scoring_ans = score_events(events, answered_map)
        # Secondary score: answered + failures filled with segment midpoint (= legacy logic, denominator = all events)
        withdef = list(pred_dates)
        for i in missing:
            d = int(events[i].get("days_from_cutoff", 0) or 0)
            mid = int(round(segment_midpoint(d)))
            withdef[i] = (cutoff_d + timedelta(days=mid)).isoformat() if cutoff_d else None
        withdef_map = {str(i + 1): withdef[i] for i in range(n_ev) if withdef[i]}
        scoring_def = score_events(events, withdef_map)
        return {
            "scoring_answered": scoring_ans, "scoring_with_default": scoring_def,
            "pred_dates": pred_dates, "answer_attempt": answer_attempt,
            "last_fail_kind": last_fail_kind, "fail_kinds": fail_kinds,
            "attempt_records": attempt_records, "n_dmx": n_dmx, "n_model": n_model,
        }

    run_results: List[Dict[str, Any]] = [one_run(ri) for ri in range(max(1, runs_per_point))]
    summary_ans = average_runs([r["scoring_answered"] for r in run_results])       # primary score: answered only
    summary_def = average_runs([r["scoring_with_default"] for r in run_results])   # secondary score: all events, midpoint-filled

    legacy_events: List[Dict[str, Any]] = []
    avg_details = summary_ans.get("details", [])
    def_details = summary_def.get("details", [])
    for i, e in enumerate(events):
        gt_d = parse_date_obj(e.get("date") or e.get("gt_date"))
        delta = (gt_d - cutoff_d).days if (gt_d and cutoff_d) else 0
        det = avg_details[i] if i < len(avg_details) else {}
        det_def = def_details[i] if i < len(def_details) else {}
        per_run_attempt = [r["answer_attempt"][i] for r in run_results]
        per_run_pred = [r["pred_dates"][i] for r in run_results]
        answered_any = any(r["pred_dates"][i] for r in run_results)
        fk = run_results[-1]["fail_kinds"][i]
        legacy_events.append({
            "eid": str(e.get("id") or e.get("eid") or (i + 1)),
            "event_desc": e.get("event") or e.get("event_desc", ""),
            "gt_date": (e.get("date") or e.get("gt_date") or ""),
            "pred_date": det.get("pred_date") or (per_run_pred[-1] or ""),
            "error_days": det.get("error_days") if det.get("error_days") is not None else 999,   # primary: failure=999 (excluded by pooling)
            "error_days_with_default": det_def.get("error_days"),   # secondary: error after midpoint fill (present for all events)
            "delta_days": delta,
            "n_runs": det.get("n_runs", len(run_results)),
            "answer_attempt": per_run_attempt,
            "per_run_pred_date": per_run_pred,
            "answered": answered_any,
            "fail_kind": None if answered_any else fk,
            "is_default_fill": not answered_any,
        })

    attempt_hist = {"1": 0, "2": 0, "3": 0, "4": 0, "null": 0}
    for r in run_results:
        for a in r["answer_attempt"]:
            attempt_hist[str(a) if a is not None else "null"] += 1
    n_ev2 = len(events)
    n_dmx = run_results[-1]["n_dmx"]; n_model = run_results[-1]["n_model"]
    dmx_frac = n_dmx / n_ev2 if n_ev2 else 0.0
    model_frac = n_model / n_ev2 if n_ev2 else 0.0
    return {
        "point_id": pid,
        "model": model,
        "cutoff_date": cutoff,
        "events_total": len(events),
        "events_scored": summary_ans.get("events_scored"),
        "weighted_mae_days": summary_ans.get("weighted_mae_days"),
        "baseline_wmae": summary_ans.get("baseline_wmae"),
        # primary = score_answered (answered only); secondary = score_with_default (all midpoint-filled, = legacy logic)
        "score_100": summary_ans.get("score_100"),
        "score_answered": summary_ans.get("score_100"),
        "score_with_default": summary_def.get("score_100"),
        "weighted_mae_with_default": summary_def.get("weighted_mae_days"),
        "predictions": [le["pred_date"] for le in legacy_events],
        "events": legacy_events,
        "per_run": [r["scoring_answered"] for r in run_results],
        "per_run_with_default": [r["scoring_with_default"] for r in run_results],
        "n_runs_completed": len(run_results),
        "answer_attempt_histogram": attempt_hist,
        "attempt_records": [rec for r in run_results for rec in r["attempt_records"]],
        "max_answer_attempts": MAX_ANSWER_ATTEMPTS,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "runs_per_point": max(1, runs_per_point),
        "n_dmx_fail": n_dmx,
        "n_model_fail": n_model,
        # Three invalid-fraction entries (fixed by user 2026-07-02)
        "dmx_error_frac": round(dmx_frac, 4),
        "model_fail_frac": round(model_frac, 4),
        "invalid_frac": round(dmx_frac + model_frac, 4),
        "validity_policy": "1run_retry4_3plus1dmx_fail_excluded_default_as_secondary",
        "density_check": {
            "total_90d": len(events), "seg1": seg_counts[0],
            "seg2": seg_counts[1], "seg3": seg_counts[2],
            "status": density_status,
        },
    }


# ===========================================================================
# Cross-point pooled aggregation (SKILL section "pooled scoring")
# ===========================================================================

def pooled_score(per_point: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """v3 (fixed by user 2026-07-02): pool both scores + three invalid fractions (event-level sum-of-weights normalization).
      - score_answered: pooled over answered events only (real error_days; failure=999 excluded).
      - score_with_default: all events (failures midpoint-filled, using error_days_with_default) = legacy logic.
      - dmx_error_frac / model_fail_frac / invalid_frac (classified across points by event fail_kind, summed).
    """
    def _pool(err_key):
        mw = 0.0; bw = 0.0; tw = 0.0
        for pr in per_point.values():
            for e in pr.get("events", []):
                err = e.get(err_key)
                if err is None or err == 999:
                    continue
                d = int(e.get("delta_days", 0) or 0)
                w = 1.0 / (1.0 + ALPHA * max(0, d))
                mw += err * w
                bw += abs(d - segment_midpoint(d)) * w
                tw += w
        if tw <= 0:
            return None, None, None
        model_wmae = mw / tw
        base_wmae = bw / tw
        score = 100.0 / (1.0 + model_wmae / base_wmae) if base_wmae > 0 else 0.0
        return round(score, 2), round(model_wmae, 2), round(base_wmae, 2)

    score_ans, wmae_ans, base_ans = _pool("error_days")               # primary score: answered
    # Secondary: prefer error_days_with_default (new); fall back to error_days for legacy data lacking the field
    _has_wd = any(e.get("error_days_with_default") is not None
                  for pr in per_point.values() for e in pr.get("events", []))
    score_def, wmae_def, base_def = _pool("error_days_with_default" if _has_wd else "error_days")
    n_ev = 0; n_dmx = 0; n_model = 0
    for pr in per_point.values():
        for e in pr.get("events", []):
            n_ev += 1
            fk = e.get("fail_kind")
            if fk == "dmx":
                n_dmx += 1
            elif fk == "model":
                n_model += 1
    dmx_frac = n_dmx / n_ev if n_ev else 0.0
    model_frac = n_model / n_ev if n_ev else 0.0
    return {
        "score_100": score_ans,
        "score_answered": score_ans,
        "weighted_mae_days": wmae_ans,
        "baseline_wmae": base_ans,
        "score_with_default": score_def,
        "weighted_mae_with_default": wmae_def,
        # Three invalid-fraction entries (go into the total-score file)
        "dmx_error_frac": round(dmx_frac, 4),
        "model_fail_frac": round(model_frac, 4),
        "invalid_frac": round(dmx_frac + model_frac, 4),
    }


# v2_finalscore (designated by user 2026-07-06 as the time axis's PRIMARY DISPLAY SCORE): a linearly normalized score of the same family as the b axis.
# Error = [span/30] of a band centered on the ground truth with radius = days off, clipped into its 30-day segment (denominator fixed at 30; clipping = wall-adjacent errors
# discounted / halved where no symmetric point exists; a perfect hit has span 0 = full marks), then multiplied by the same time weight as old f, 1/(1+ALPHA*d); score = 100*(1 - weighted mean error).
# The old pooled.score_100 (hyperbolic) is kept untouched. History: v1 (error / distance-to-far-end; had a middle-hard/edge-easy artifact) → v2 (band/half-window, symmetric, artifact-free).
V2_FINALSCORE_META = {
    "name": "v2_finalscore", "adopted": "2026-07-06", "axis": "time(3F)", "role": "primary_display_score",
    "formula": "100*(1 - weighted_mean(norm_err));  norm_err = span([d-err,d+err] ∩ [seg_lo,seg_hi]) / 30",
    "denominator": 30, "clip_to_window": True, "perfect_equals_zero": True,
    "time_weight": "1/(1+0.04*delta_days)  (与旧f/与b时间权重同)",
    "note": "误差归一化[0,1]、线性、与b轴同族;band以真值为心半径=差天数裁进窗口(贴墙打折);完美=满分。旧pooled.score_100(双曲)保留。",
}


def v2_finalscore(per_point: Dict[str, Dict[str, Any]], err_key: str = "error_days") -> Optional[float]:
    """v2_finalscore: band span / 30, window-clipped, perfect = 0, time weight 1/(1+ALPHA*d). Defaults to error_days (same basis as the primary score).

    This is the temporal score S_time reported in the paper (Sec. 3.3,
    "Temporal score"): Eq. (2) gives the window-clipped band B_e and the
    normalized miss m_e per event; Eq. (3) aggregates them with w_time.
    """
    num = 0.0
    den = 0.0
    for pr in per_point.values():
        for e in pr.get("events", []):
            err = e.get(err_key)
            if err is None or err == 999:
                continue
            d = int(e.get("delta_days", 0) or 0)
            if d < 0 or d > 90:
                continue
            # Paper Sec. 3.3: the event's 30-day bucket [l_e, u_e] (days
            # 0-30 / 31-60 / 61-90 after the cutoff), chosen by offset t_e (=d).
            seg = next(((lo, hi) for lo, hi, _ in SEGMENTS if lo <= d <= hi), None)
            if seg is None:
                continue
            lo, hi = seg
            # Paper Eq. (2): B_e = min(u_e, t_e+Δ_e) − max(l_e, t_e−Δ_e);
            #                m_e = min(1, B_e / 30).
            span = min(hi, d + err) - max(lo, d - err)
            norm = min(1.0, span / 30.0)
            # Paper Eq. (3): w_time = 1/(1 + 0.04·t_e);
            # S_time = 100 · max(0, 1 − Σ w·m / Σ w).
            w = 1.0 / (1.0 + ALPHA * max(0, d))
            num += w * norm
            den += w
    if den <= 0:
        return None
    return round(100.0 * max(0.0, 1.0 - num / den), 2)


def is_complete_time_result(res: Dict[str, Any], runs_per_point: int, coverage_only: bool = False) -> bool:
    if coverage_only and isinstance(res, dict):
        return True
    if isinstance(res, dict) and int(res.get("events_total", 0) or 0) == 0:
        return True  # 2026-07-02: a point with no events in the window has nothing to evaluate by nature, it's not a missing run → treated as complete (avoids pointless recomputation each round; consistent with run_agents.reusable_time)
    if res.get("score_100") is None:
        return False
    # F evaluates runs_per_point times (fixed by user 2026-07-01: once; 1-vs-2 mean diff < run-to-run noise, saves ~1/3); each run uses whole-point 3-attempt retry internally.
    # Complete = has score_100 AND ran the full runs_per_point runs AND every run produced a wmae.
    per_run = res.get("per_run") or []
    if len(per_run) < runs_per_point:
        return False
    return all(sub.get("weighted_mae_days") is not None for sub in per_run)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="predict-step3F-time: temporal accuracy via LLM date predictions")
    p.add_argument("--workspace", required=True)
    p.add_argument("--event-name", required=True)
    p.add_argument("--models", required=True, help="comma-separated model IDs")
    p.add_argument("--points", default="all")
    p.add_argument("--runs-per-point", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--thinking-budget", type=int, default=None)
    p.add_argument("--max-parallel", type=int, default=300,
                   help="Global concurrency over (point × model) tasks. Fixed by user 2026-06-30: single key + 300 threads; effective concurrency = min(300, task count)")
    p.add_argument("--pipeline-config", default=None)
    p.add_argument("--run-id", default=None,
                   help="Reuse a shared run_<ts> dir (set by orchestrator)")
    p.add_argument("--coverage-only", action="store_true",
                   help="Reuse existing point files even if incomplete; only fill missing coverage.")
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    s3 = cfg.get("predict_step3_evaluation", {})
    runs_per_point = args.runs_per_point or int(s3.get("temporal_runs_per_point", 1))
    max_tokens = args.max_tokens or int(s3.get("max_tokens_per_call", 32000))       # Fixed by user 2026-06-25: use the max
    thinking_budget = args.thinking_budget or int(s3.get("thinking_budget_tokens", 24000))  # thinking maxed out (reasoning_effort is already "high")
    coverage_only = args.coverage_only or os.environ.get("SB_COVERAGE_ONLY", "0") != "0"

    workspace = Path(args.workspace)
    if "英文" in workspace.parts:          # English eval → use English prompts (2026-07-01)
        global PROMPT_TEMPLATE, EXAM_PROMPT_TEMPLATE
        PROMPT_TEMPLATE, EXAM_PROMPT_TEMPLATE = PROMPT_TEMPLATE_EN, EXAM_PROMPT_TEMPLATE_EN
    qb_dir = workspace / "questionbank"
    ctx_dir = workspace / "contexts"
    if not qb_dir.exists():
        raise SystemExit(f"No questionbank/ under {workspace}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = workspace / "results" / f"run_{run_id}" / "time"

    qb_files = sorted(qb_dir.glob("P*_questionbank.json"))
    if args.points != "all":
        wanted = set(args.points.split(","))
        qb_files = [f for f in qb_files if f.stem.replace("_questionbank", "") in wanted]

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Global concurrency over (point × model)
    tasks = []
    for model in models:
        for qb_path in qb_files:
            pid = qb_path.stem.replace("_questionbank", "")
            ctx_path = ctx_dir / f"{pid}_context.md"
            tasks.append((pid, model, qb_path, ctx_path))

    results_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in models}

    def process(pid: str, model: str, qb_path: Path, ctx_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        out_file = out_root / model_slug(model) / f"{pid}_time.json"
        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text(encoding="utf-8"))
                if is_complete_time_result(existing, runs_per_point, coverage_only=coverage_only):
                    print(f"[time] {model} / {pid} reuse existing")
                    return pid, model, existing
                print(f"[time] {model} / {pid} existing incomplete; rerun")
            except Exception:
                print(f"[time] {model} / {pid} existing unreadable; rerun")
        print(f"[time] {model} / {pid} ...")
        res = run_point(pid, qb_path, ctx_path, model, runs_per_point, max_tokens, thinking_budget)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return pid, model, res

    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
        futs = [ex.submit(process, *t) for t in tasks]
        for f in as_completed(futs):
            try:
                pid, model, res = f.result()
                results_by_model[model][pid] = res
            except Exception as e:
                print(f"  task error: {e}")

    # SKILL section "a failed single run must be re-run" (2026-04-25):
    # Self-scan after main loop. A point is considered incomplete if any of:
    #   - file missing
    #   - score_100 None
    #   - per_run length < runs_per_point
    #   - per_run contains entries where weighted_mae_days is None
    # Up to 2 re-run attempts per point.
    all_pids = [qb_path.stem.replace("_questionbank", "") for qb_path in qb_files]
    for model in models:
        slug = model_slug(model)
        per_point = results_by_model[model]
        out_model_dir = out_root / slug
        out_model_dir.mkdir(parents=True, exist_ok=True)
        if coverage_only:
            for point_path in sorted(out_model_dir.glob("P*_time.json")):
                pid = point_path.stem.replace("_time", "")
                if pid in per_point:
                    continue
                try:
                    per_point[pid] = json.loads(point_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

        def needs_rerun(pid: str) -> bool:
            r = per_point.get(pid)
            if not r:
                return True
            return not is_complete_time_result(r, runs_per_point, coverage_only=coverage_only)

        for attempt in range(2):
            missing = [pid for pid in all_pids if needs_rerun(pid)]
            if not missing:
                break
            for pid in missing:
                qb_path = qb_dir / f"{pid}_questionbank.json"
                ctx_path = ctx_dir / f"{pid}_context.md"
                print(f"[time-rerun-scan-{attempt+1}] {model} / {pid} (missing or incomplete)")
                res = run_point(pid, qb_path, ctx_path, model, runs_per_point, max_tokens, thinking_budget)
                out_file = out_model_dir / f"{pid}_time.json"
                out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                per_point[pid] = res

        # Sanity log: report incomplete points after all retries
        still_incomplete = [pid for pid in all_pids if needs_rerun(pid)]
        if still_incomplete:
            print(f"[time-rerun-scan] {model}: {len(still_incomplete)} points still incomplete after retries: {still_incomplete}")

        pooled = pooled_score(per_point)
        (out_root / slug / "aggregated.json").write_text(
            json.dumps({
                "model": model,
                "n_points": len(per_point),
                "batch_size": None,
                "validity_policy": "1run_retry4_3plus1dmx_fail_excluded_default_as_secondary",
                "max_answer_attempts": MAX_ANSWER_ATTEMPTS,
                "retry_wait_seconds": RETRY_WAIT_SECONDS,
                "fallback_filled_count": sum(int(r.get("fallback_filled_count") or 0) for r in per_point.values()),
                "answer_attempt_histogram": {
                    k: sum(int((r.get("answer_attempt_histogram") or {}).get(k, 0)) for r in per_point.values())
                    for k in ("1", "2", "3", "null")
                },
                "pooled": pooled,
                "v2_finalscore": v2_finalscore(per_point),
                "v2_finalscore_meta": V2_FINALSCORE_META,
                "per_point": {pid: {k: v for k, v in r.items() if k not in ("events", "per_run")}
                              for pid, r in per_point.items()},
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"run_dir": str(out_root.parent), "models": models,
                      "points_done": len(qb_files), "runs_per_point": runs_per_point},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
