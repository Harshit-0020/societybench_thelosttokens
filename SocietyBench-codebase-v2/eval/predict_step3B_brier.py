#!/usr/bin/env python3
"""predict-step3B-brier (1:1 of skills/predict-step3B-brier/SKILL.md).

For each prediction point and each evaluated model:
  1. Read context (pre-cutoff) + brier_questions from Step 2.
  2. Stable-shuffle by MD5(point_id), build batches using the configured or
     CLI-provided batch size. For formal runs, prefer keeping a prediction
     point as intact as the model window and parser allow; fall back to smaller
     batches only on window/timeout/parse issues.
  3. Run batches in parallel, asking the eval model to answer
     `1. 75%` per line (reasoning_effort="high"=top tier, include_reasoning=True, max_tokens=32000).
  4. Whole-point retry policy (fixed by user 2026-06-30): per point, up to 3 batched attempts with a
     5-minute wait between attempts; each answered question records answer_attempt
     (1/2/3). Questions still unanswered after attempt 3 are filled with the 0.5
     default and MARKED (answer_attempt=null, is_default_fill=true) — never a silent
     fallback. A 25-point B run issues at most 25×3 = 75 requests (one batch / point
     at batch=999999). Request failures (exceptions) and parse failures are tracked
     separately so "request failed" is distinguishable from "model gave no answer".
  5. Compute per-question MAE + dual weight (time × window) → weighted_mae
     → score_100 = 100 × max(0, 1 - weighted_mae / 1.0).
  6. After all points done, emit aggregated.json (single pool of all questions
     across selected points, recomputed with the same weights).

Input  : <workspace>/questionbank/P*_questionbank.json
         <workspace>/contexts/P*_context.md
Output : <workspace>/results/run_<ts>/brier/<model_slug>/P*_brier.json
         <workspace>/results/run_<ts>/brier/<model_slug>/aggregated.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any, Dict, List, Optional, Tuple

from common import call_llm, load_pipeline_config, parse_date_obj, LLMQuotaError


# ===========================================================================
# Constants (kept centralized so they're easy to audit against the SKILL)
# ===========================================================================

DEFAULT_BATCH_SIZE = 999999
DEFAULT_SAME_GROUP = 1  # Anti-leakage: questions of the same event don't share a batch, so the model can't reverse-engineer answers from "same event, multiple windows" (cost up ~1.9x)
DEFAULT_MAX_PARALLEL = 300   # Fixed by user 2026-06-30: single key + 300 threads (measured: no per-key concurrency cap); effective concurrency = min(300, task count)
DEFAULT_MAX_TOKENS = 32000       # Fixed by user 2026-06-25: use max max_tokens so reasoning-model thinking is never truncated
DEFAULT_THINKING_BUDGET = 24000  # Thinking budget maxed out (< max_tokens); reasoning_effort is already "high" (top tier)
MAX_REUSABLE_PARSE_FAIL_RATE = 0.02

# Whole-point retry policy (fixed by user 2026-06-30) ---------------------------
# Each prediction point gets at most 3 batched requests, waiting RETRY_WAIT_SECONDS
# (default 60s) between attempts; attempt 1 asks all questions of the point, later
# attempts only re-ask questions still "unanswered / request failed" (a P1 point can
# finish in one shot). Still no answer after 3 attempts → fill DEFAULT_FALLBACK_PROB
# (0.5) and mark answer_attempt=null. Each question carries answer_attempt:
# 1/2/3 = which attempt yielded the real answer; null = system default 0.5.
# Request failures (exceptions) and parse failures are accounted separately.
MAX_ANSWER_ATTEMPTS = 3
RETRY_WAIT_SECONDS = int(os.environ.get("SB_RETRY_WAIT_SECONDS", "60"))  # Fixed by user 2026-06-30: 300→60; can be small (e.g. 1) for tests
DEFAULT_FALLBACK_PROB = 0.5
# Fixed by user (2026-06-30): a "whole round with zero responses caused by exhausted
# quota" does NOT count toward retry attempts → it goes into a separate wait-and-retry
# track. This is the max number of such waits (each waits RETRY_WAIT_SECONDS) before
# giving up (avoids an infinite loop if the balance never recovers).
MAX_QUOTA_WAITS = int(os.environ.get("SB_MAX_QUOTA_WAITS", "6"))


# ===========================================================================
# Prompt
# ===========================================================================

ANSWER_PROMPT = """你正在参加一场**预测考试**。以下是一个公共事件截至 {cutoff} 的已知信息：

================ 已知信息（截止 {cutoff}）================
{context}
================

================ 评分规则（这是考试，请据此争取最高分）================
- 你为每道题给出一个概率（0%~100%）。每题最终只有两种真实结果：发生（=100%）或未发生（=0%）。
- 单题误差 = |你给的概率 − 真实结果|，误差越小得分越高。
- 总分 = 100 ×（1 − 加权平均误差）。即：误差为 0 得满分 100；全程蒙 50% 得 50 分；自信地把方向猜反（误差趋近 1）会掉到 0 分。
- 加权：越临近 {cutoff}、时间窗越长的题，占分越重，请优先把这类题判准。
================

你的任务是预测：下面每个问题描述的事件在未来实际发生的概率是多少？请基于已知信息推理，给出能让你**考试总分最大化**的概率：
- 方向有把握时就**果断**给高/低概率（越接近 0% 或 100% 越赚分）；
- 真正不确定（已知信息不足以判断方向）时再给接近 50%——别刻意回避，也别为显得果断而瞎押。

作答前请**好好思考、好好推演**，想清楚每题方向再下笔（思考在心里即可，输出只给答案）。

**输出格式**：只输出题号和百分比，每题一行，例如：
1. 75%
2. 30%

不要添加任何额外说明、不要解释。

================ 待回答问题 ================
{question_list}
================
"""

RETRY_PROMPT = """你刚才的回答中，第 {missing_ids} 题的概率未能识别。
请只回答这几题，严格按格式输出：
{id}. {{百分比}}%

例如：
1. 75%
2. 30%
"""


# ---- English prompts (2026-07-01: English evals use English instructions, sentence-by-sentence mirror of the Chinese version above; main() switches when the workspace path contains "英文") ----
ANSWER_PROMPT_EN = """You are taking a **forecasting exam**. Below is the known information about a public event, as of {cutoff}:

================ Known information (as of {cutoff}) ================
{context}
================

================ Scoring rules (this is an exam — aim for the highest score) ================
- For each question you give a probability (0%~100%). Each question has only two possible true outcomes: it happened (=100%) or it did not (=0%).
- Per-question error = |your probability − the true outcome|; the smaller the error, the higher the score.
- Total = 100 × (1 − weighted mean error). I.e.: zero error scores a perfect 100; guessing 50% on everything scores 50; confidently betting the wrong direction (error → 1) drops toward 0.
- Weighting: questions closer to {cutoff} and with a longer time window count more — prioritize getting those right.
================

Your task: predict the probability that each event described below actually happens in the future. Reason from the known information and give the probabilities that **maximize your exam score**:
- When you are confident about the direction, be **decisive** and give a high/low probability (closer to 0% or 100% earns more);
- Only when genuinely uncertain (the known information is insufficient to judge the direction) give something near 50% — do not deliberately hedge, and do not bluff just to look decisive.

Take your time to **think it through and reason carefully** before answering (reason internally; output only the answers).

**Output format**: output only the question number and a percentage, one per line, e.g.:
1. 75%
2. 30%

Do not add any extra explanation.

================ Questions ================
{question_list}
================
"""

RETRY_PROMPT_EN = """In your previous answer, the probability for question(s) {missing_ids} could not be parsed.
Please answer only these, strictly in this format:
{id}. {{percentage}}%

For example:
1. 75%
2. 30%
"""

LANG = "zh"   # Set by main() from the workspace path ("中文"/"英文"); decides whether model-facing text (render_question etc.) is Chinese or English


# ===========================================================================
# Question formatting
# ===========================================================================

def render_question(q: Dict[str, Any]) -> str:
    body = q.get("q") or q.get("event_desc") or ""
    d_target = q.get("d_target") or ""
    if LANG == "en":
        wd = q.get("window_days", 14)
        return f"Event: {body}  Target window: [within {wd} days before {d_target}]"
    window = q.get("time_window") or f"{q.get('window_days', 14)}天"
    return f"事件：{body}  目标日期窗口：[{d_target} 前 {window} 内]"


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def reported_batch_size(cli_batch_size: Optional[int], effective_batch_size: int) -> Optional[int]:
    """External metadata: large sentinel values mean "no fixed batch limit"."""
    if effective_batch_size >= 999999:
        return None
    return effective_batch_size


def batch_size_note(batch_size: Optional[int]) -> str:
    return "unlimited" if batch_size is None else f"fixed:{batch_size}"


def summarize_point_batch_size(
    per_point: Dict[str, Dict[str, Any]],
    default_batch_size: Optional[int],
) -> Tuple[Optional[int] | str, Dict[str, Optional[int] | str], str]:
    by_point: Dict[str, Optional[int] | str] = {}
    for pid, result in sorted(per_point.items()):
        if isinstance(result, dict) and "batch_size" in result:
            by_point[pid] = result.get("batch_size")
        else:
            by_point[pid] = "__unknown__"
    known_values = {v for v in by_point.values() if v != "__unknown__"}
    if not by_point:
        return default_batch_size, by_point, "empty"
    if not known_values and "__unknown__" in set(by_point.values()):
        return default_batch_size, by_point, "run_default_unverified_for_legacy_points"
    if "__unknown__" in set(by_point.values()):
        return "mixed", by_point, "mixed_with_unknown_legacy_points"
    if len(known_values) == 1:
        value = next(iter(known_values))
        return value, by_point, "point_metadata"
    return "mixed", by_point, "point_metadata_mixed"


# ===========================================================================
# Batch construction
# ===========================================================================

def build_batches(
    items: List[Tuple[int, Dict[str, Any]]],
    point_id: str,
    batch_size: int,
    max_same_group: int,
) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Same-event grouped batching (changed 2026-06-22, replaces the max_same_group
    splitting logic): put ALL questions of one event_group (multi-tier time windows
    7/14/30/60/90 days etc.) into the SAME batch, asked in one shot, never split —
    so the model sees all windows of the event at once and gives monotonically
    self-consistent probabilities (P(<=7) <= P(<=30) <= ...), removing the
    non-monotonic scrambling caused by the old split-batch (max=1) scheme. Within a
    group, questions are presented in ascending window order; different event groups
    are greedily packed into batches by batch_size (each group placed whole; a single
    group exceeding batch_size becomes its own batch, never split).
    The `max_same_group` parameter is kept for compatibility and no longer used."""
    # 1. Aggregate by event_group (dict preserves insertion order; ungrouped questions each form their own group)
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for orig_i, q in items:
        g = str(q.get("event_group") or f"__solo_{orig_i}")
        groups.setdefault(g, []).append((orig_i, q))
    # Within each group, ascending window days → the event's tiers appear monotonically
    for g in groups:
        groups[g].sort(key=lambda t: int(t[1].get("window_days") or 0))
    # 2. Stable-shuffle the order of GROUPS (seed from point_id for reproducibility)
    seed = int.from_bytes(hashlib.md5(point_id.encode("utf-8")).digest()[:8], "big")
    glist = list(groups.values())
    random.Random(seed).shuffle(glist)
    # 3. Greedy packing: a group goes into one batch whole, never split; open a new batch if it doesn't fit; a group larger than batch_size becomes its own batch
    batches: List[List[Tuple[int, Dict[str, Any]]]] = []
    cur: List[Tuple[int, Dict[str, Any]]] = []
    for grp in glist:
        if cur and len(cur) + len(grp) > batch_size:
            batches.append(cur)
            cur = []
        cur.extend(grp)
        if len(cur) >= batch_size:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    return [b for b in batches if b]


# ===========================================================================
# Answer parsing
# ===========================================================================

NUM_PAT = re.compile(
    r"^\s*(?:[\"']?(?:第\s*)?)?(\d{1,3})(?:\s*题)?[\"']?\s*"
    r"(?:[\.\):：、-]|=>|=)\s*[\"']?(\d+(?:\.\d+)?)\s*%?",
    re.M,
)
KV_PAT = re.compile(
    r"[\"'](?:q|question|题号|id)?\s*(\d{1,3})[\"']\s*:\s*"
    r"[\"']?(\d+(?:\.\d+)?)\s*%?[\"']?",
    re.I,
)


def parse_answers(raw: str) -> Dict[int, float]:
    """Return {1-based id → probability in [0,1]}."""
    out: Dict[int, float] = {}
    text = str(raw or "")
    for m in NUM_PAT.finditer(text):
        idx = int(m.group(1))
        pct = float(m.group(2))
        out[idx] = max(0.0, min(1.0, pct / 100.0))
    for m in KV_PAT.finditer(text):
        idx = int(m.group(1))
        pct = float(m.group(2))
        out.setdefault(idx, max(0.0, min(1.0, pct / 100.0)))
    return out


# ===========================================================================
# Per-batch call + retry
# ===========================================================================

def _call_with_retry(prompt: str, model: str, max_tokens: int, thinking_budget: int,
                     max_attempts: int = 3) -> str:
    """SKILL section "auto-retry network failures twice": wrap the LLM call with up to 2 retries
    on exceptions (3 attempts total) with exponential backoff. Returns the raw
    string; on persistent failure, propagates the last exception."""
    import time as _t
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            kwargs = dict(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            # Fixed by user 2026-06-30: non-thinking models don't think. SB_NO_THINKING=1 →
            # omit reasoning_effort/thinking_budget (avoids forcing thinking on
            # non-thinking models, or triggering a 400).
            if os.environ.get("SB_NO_THINKING", "0") == "0":
                kwargs["reasoning_effort"] = "high"
                kwargs["include_reasoning"] = True
                # GPT/OpenAI-family models control thinking via reasoning_effort and
                # return HTTP 400 on Anthropic-style thinking_budget_tokens; only pass
                # the budget for non-GPT models.
                if "gpt" not in model.lower():
                    kwargs["thinking_budget_tokens"] = thinking_budget
            # Retry once at this lower layer and let this wrapper own retries.
            # Otherwise 3 outer attempts × 3 common.call_llm attempts can turn
            # one slow batch into a very long opaque wait.
            kwargs["retries"] = 1
            return call_llm([{"role": "user", "content": prompt}], **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1:
                _t.sleep(2 ** attempt)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def call_batch(
    batch: List[Tuple[int, Dict[str, Any]]],
    context: str,
    cutoff: str,
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> Tuple[Dict[int, float], List[int]]:
    """Returns (probabilities_by_batch_idx, parse_fail_batch_idx_list)."""
    questions_block = "\n".join(
        f"{i + 1}. {render_question(q)}"
        for i, (_orig, q) in enumerate(batch)
    )
    prompt = ANSWER_PROMPT.format(
        cutoff=cutoff, context=context, question_list=questions_block
    )
    raw = _call_with_retry(prompt, model, max_tokens, thinking_budget)
    answers = parse_answers(raw)

    missing = [i + 1 for i in range(len(batch)) if (i + 1) not in answers]
    if missing:
        retry_block = "\n".join(
            f"{i}. {render_question(batch[i - 1][1])}"
            for i in missing
        )
        retry_prompt = (
            RETRY_PROMPT.format(missing_ids=", ".join(str(i) for i in missing), id=missing[0])
            + "\n\n" + retry_block
        )
        try:
            raw2 = _call_with_retry(retry_prompt, model, max_tokens, thinking_budget)
            for k, v in parse_answers(raw2).items():
                answers[k] = v
        except Exception:
            pass  # parse-fail fallback handled by caller

    parse_fail = [i + 1 for i in range(len(batch)) if (i + 1) not in answers]
    return answers, parse_fail


def call_batch_once(
    batch: List[Tuple[int, Dict[str, Any]]],
    context: str,
    cutoff: str,
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> Tuple[Dict[int, float], List[int]]:
    """ONE LLM call for `batch` — NO inner targeted retry (the cross-attempt
    retry in run_point owns retries, so a 25-point run stays within 25×3
    requests). Returns (answers_by_1based_pos, parse_fail_1based_list); raises
    on request failure so the caller can record it as request_error."""
    questions_block = "\n".join(
        f"{i + 1}. {render_question(q)}" for i, (_orig, q) in enumerate(batch)
    )
    prompt = ANSWER_PROMPT.format(cutoff=cutoff, context=context, question_list=questions_block)
    kwargs: Dict[str, Any] = dict(
        model=model, temperature=0.0, max_tokens=max_tokens, retries=1,
    )
    # 2026-07-02, aligned with 3F (call_for_dates): SB_NO_THINKING=1 → omit thinking
    # params (truly non-thinking models). Default (SB_NO_THINKING=0) still forces
    # reasoning_effort=high + thinking; main-experiment behavior unchanged.
    if os.environ.get("SB_NO_THINKING", "0") == "0":
        kwargs["reasoning_effort"] = "high"
        kwargs["include_reasoning"] = True
        # GPT/OpenAI family controls thinking via reasoning_effort; Anthropic-style thinking_budget causes HTTP 400
        if "gpt" not in model.lower():
            kwargs["thinking_budget_tokens"] = thinking_budget
    raw = call_llm([{"role": "user", "content": prompt}], **kwargs)
    answers = parse_answers(raw)
    parse_fail = [i + 1 for i in range(len(batch)) if (i + 1) not in answers]
    return answers, parse_fail


# ===========================================================================
# Weight + scoring
# ===========================================================================

def compute_weight(days_from_cutoff: int, window_days: int,
                   time_coef: float, window_offset: float) -> float:
    # Paper Sec. 3.3 "Calibration score" — the per-question weight w_q of
    # Eq. (1): w_q = w_time * w_win, with w_time = 1/(1 + 0.04*Δt) and
    # w_win = W/(W + 4) (coefficients read from pipeline_config.json).
    tw = 1.0 / (1.0 + time_coef * max(0, days_from_cutoff))
    ww = window_days / (window_days + window_offset)
    return tw * ww


def score_point(
    questions: List[Dict[str, Any]],
    probs: List[Optional[float]],
    cfg_cal: Dict[str, Any],
    parse_fail_indices: Optional[set[int]] = None,
    batch_by_index: Optional[Dict[int, int]] = None,
    answer_attempts: Optional[List[Optional[int]]] = None,
    default_indices: Optional[set] = None,
    fail_kinds: Optional[List[Optional[str]]] = None,
) -> Dict[str, Any]:
    """v3 (fixed by user 2026-07-02): produce two scores + three invalid fractions.
      - probs[i]=None means the question was NOT answered (failed); otherwise a real probability. NO default pre-fill.
      - fail_kinds[i]: None=answered / "dmx"=upstream failure / "model"=model gave no answer; unanswered defaults to "model".
      - score_answered (primary): weighted MAE → score computed on ANSWERED questions only (denominator = answered count).
      - score_with_default (secondary): ALL questions (failures filled with 0.5) = legacy logic (denominator = total).
      - dmx_error_frac / model_fail_frac / invalid_frac (= their sum; mutually exclusive classification).
    """
    time_coef = float(cfg_cal.get("time_factor_coefficient", 0.04))
    window_offset = float(cfg_cal.get("window_factor_offset", 4))
    baseline = float(cfg_cal.get("baseline_uniform_predictor_wmae", 1.0))  # zero-score line = 1.0

    n_total = len(questions)
    batch_by_index = batch_by_index or {}
    answer_attempts = answer_attempts or [None] * n_total
    fail_kinds = fail_kinds or [None] * n_total
    default_indices = default_indices or set()  # legacy compatibility, no longer drives scoring

    ans_total_ae = 0.0; ans_werr = 0.0; ans_wsum = 0.0   # primary score: answered questions only
    def_werr = 0.0; def_wsum = 0.0                        # secondary score: all questions (failures filled with 0.5)
    answered_n = 0; correct = 0; n_dmx = 0; n_model = 0
    details: List[Dict[str, Any]] = []
    for qi, (q, p) in enumerate(zip(questions, probs)):
        ans = int((q.get("answer") if q.get("answer") is not None else q.get("gt", 0)) or 0)
        days_fc = int(q.get("days_from_cutoff", 0) or 0)
        window_days = int(q.get("window_days", 14) or 14)
        w = compute_weight(days_fc, window_days, time_coef, window_offset)
        answered = p is not None
        fk = fail_kinds[qi] if qi < len(fail_kinds) else None
        if not answered and fk is None:
            fk = "model"
        p_def = float(p) if answered else DEFAULT_FALLBACK_PROB   # secondary score: failed questions filled with 0.5
        def_werr += abs(p_def - ans) * w
        def_wsum += w
        if answered:
            ae = abs(float(p) - ans)
            ans_total_ae += ae
            ans_werr += ae * w
            ans_wsum += w
            answered_n += 1
            if (float(p) >= 0.5 and ans == 1) or (float(p) < 0.5 and ans == 0):
                correct += 1
        elif fk == "dmx":
            n_dmx += 1
        else:
            n_model += 1
        details.append({
            "q": q.get("q") or q.get("event_desc", ""),
            "answer": ans,
            "prob": float(p) if answered else None,
            "mae": (abs(float(p) - ans) if answered else None),
            "weight": w,
            "days_from_cutoff": days_fc,
            "window_days": window_days,
            "dimension": q.get("dimension"),
            "difficulty": q.get("difficulty"),
            "question_type": q.get("question_type"),
            "event_group": q.get("event_group"),
            "question_index": qi,
            "batch_idx": batch_by_index.get(qi),
            "answer_attempt": answer_attempts[qi] if qi < len(answer_attempts) else None,
            "fail_kind": None if answered else fk,
            "is_default_fill": not answered,   # filled with default in secondary score (excluded from primary)
            "parse_failed": not answered,
            "valid_for_scoring": answered,     # primary score: only answered questions are valid
            "invalid_reason": None if answered else ("upstream_error" if fk == "dmx" else "model_no_answer"),
            "correct": (((float(p) >= 0.5 and ans == 1) or (float(p) < 0.5 and ans == 0)) if answered else None),
        })

    def _sc(werr, wsum):
        if wsum <= 0:
            return None, None
        # Paper Eq. (1): wMAE_cal = Σ w_q|p̂_q − y_q| / Σ w_q, then
        # S_cal = 100 * max(0, 1 − wMAE_cal)  (baseline = 1.0).
        wmae = werr / wsum
        return wmae, 100.0 * max(0.0, 1.0 - wmae / baseline)
    wmae_ans, score_ans = _sc(ans_werr, ans_wsum)
    wmae_def, score_def = _sc(def_werr, def_wsum)
    dmx_frac = n_dmx / n_total if n_total else 0.0
    model_frac = n_model / n_total if n_total else 0.0
    return {
        "n": n_total,
        "n_total": n_total,
        "n_answered": answered_n,
        "n_valid": answered_n,
        "n_invalid": n_total - answered_n,
        "accuracy": correct / answered_n if answered_n else None,
        "accuracy_pct": 100.0 * correct / answered_n if answered_n else None,
        "avg_mae": ans_total_ae / answered_n if answered_n else None,
        "weighted_mae": wmae_ans,
        # primary = score_answered (answered only); secondary = score_with_default (all filled with default, = legacy logic)
        "score_100": score_ans,
        "score_answered": score_ans,
        "score_with_default": score_def,
        "weighted_mae_with_default": wmae_def,
        # Three invalid-fraction entries (fixed by user 2026-07-02, present in both per-point and aggregate)
        "dmx_error_frac": round(dmx_frac, 4),
        "model_fail_frac": round(model_frac, 4),
        "invalid_frac": round(dmx_frac + model_frac, 4),
        "details": details,
    }


# No-signal gate (A2, 2026-07-01): detect model cells that "gave no usable signal at all" (e.g. all parses failed / all 0.5) and report N/A instead of a fake 50
NO_SIGNAL_STD_EPS = 0.01        # probability std < this → treated as "all 0.5 / no discrimination"
NO_SIGNAL_DEFAULT_FRAC = 0.9    # default-fill fraction >= this → treated as "almost all failed"


def aggregate_across_points(per_point: Dict[str, Dict[str, Any]],
                            cfg_cal: Dict[str, Any]) -> Dict[str, Any]:
    """v3 (fixed by user 2026-07-02): aggregate both scores + three invalid fractions across points. NO hard gate (no_signal) anymore; fractions are only recorded.
      - score_answered (primary): merged weighted MAE over answered questions only.
      - score_with_default (secondary): merged over all questions (failures filled with 0.5) = legacy logic.
      - dmx_error_frac / model_fail_frac / invalid_frac (classified by per-detail invalid_reason, summed).
    """
    baseline = float(cfg_cal.get("baseline_uniform_predictor_wmae", 1.0))
    merged: List[Dict[str, Any]] = []
    for pid, res in per_point.items():
        merged.extend(res.get("details", []))
    n_total = len(merged)
    if not merged:
        return {"n": 0, "n_total": 0, "n_valid": 0, "n_invalid": 0, "score_100": None}
    answered = [d for d in merged
                if d.get("valid_for_scoring") and d.get("prob") is not None and d.get("mae") is not None]
    # Primary score: answered questions only.
    # Paper Eq. (1), pooled across all questions of the selected points:
    # wMAE_cal = Σ w_q|p̂_q − y_q| / Σ w_q;  S_cal = 100 * max(0, 1 − wMAE_cal).
    ans_wsum = sum(d["weight"] for d in answered)
    ans_werr = sum(d["mae"] * d["weight"] for d in answered)
    avg_mae = (sum(d["mae"] for d in answered) / len(answered)) if answered else None
    wmae_ans = (ans_werr / ans_wsum) if ans_wsum > 0 else None
    score_ans = 100.0 * max(0.0, 1.0 - wmae_ans / baseline) if wmae_ans is not None else None
    # Secondary score: all questions (failed ones filled with 0.5 → mae=|0.5-answer|)
    def _mae_def(d):
        if d.get("valid_for_scoring") and d.get("mae") is not None:
            return d["mae"]
        return abs(DEFAULT_FALLBACK_PROB - int(d.get("answer", 0) or 0))
    def_wsum = sum(d["weight"] for d in merged)
    def_werr = sum(_mae_def(d) * d["weight"] for d in merged)
    wmae_def = (def_werr / def_wsum) if def_wsum > 0 else None
    score_def = 100.0 * max(0.0, 1.0 - wmae_def / baseline) if wmae_def is not None else None
    # Three invalid fractions (classified by per-detail invalid_reason, mutually exclusive, summed)
    n_dmx = sum(1 for d in merged if d.get("invalid_reason") == "upstream_error")
    n_model = sum(1 for d in merged if d.get("invalid_reason") == "model_no_answer")
    dmx_frac = n_dmx / n_total
    model_frac = n_model / n_total
    invalid_frac = dmx_frac + model_frac
    probs = [float(d["prob"]) for d in answered if d.get("prob") is not None]
    prob_std = ((sum((x - sum(probs) / len(probs)) ** 2 for x in probs) / len(probs)) ** 0.5) if len(probs) > 1 else 0.0
    return {
        "n": len(answered),
        "n_total": n_total,
        "n_answered": len(answered),
        "n_valid": len(answered),
        "n_invalid": n_total - len(answered),
        "avg_mae": avg_mae,
        "weighted_mae": wmae_ans,
        "score_100": score_ans,                 # primary = score_answered (no longer hard-gated to None)
        "score_answered": score_ans,
        "score_with_default": score_def,
        "weighted_mae_with_default": wmae_def,
        # Three invalid-fraction entries (go into the total-score file)
        "dmx_error_frac": round(dmx_frac, 4),
        "model_fail_frac": round(model_frac, 4),
        "invalid_frac": round(invalid_frac, 4),
        "prob_std": round(prob_std, 4),
        # --- Legacy compatibility fields (downstream scorecard may read them) ---
        "score_100_raw": round(score_ans, 2) if score_ans is not None else None,
        "no_signal": False,
        "default_frac": round(invalid_frac, 4),
        "default_fill_frac": round(invalid_frac, 4),
        "parse_fail_frac": round(invalid_frac, 4),
        "upstream_excluded_frac": round(dmx_frac, 4),
    }


def load_reusable_point(out_file: Path, coverage_only: bool = False) -> Optional[Dict[str, Any]]:
    if not out_file.exists():
        return None
    try:
        res = json.loads(out_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    n = int(res.get("n") or len(res.get("questions") or []) or len(res.get("details") or []) or 0)
    parse_fail = int(res.get("parse_fail_count") or 0)
    if n <= 0:
        return None
    if coverage_only:
        return res
    if parse_fail / n > MAX_REUSABLE_PARSE_FAIL_RATE:
        print(f"  existing {out_file.name} parse_fail={parse_fail}/{n}; rerun")
        return None
    return res


def existing_point_probs(res: Dict[str, Any], n_questions: int) -> Optional[List[float]]:
    """Best-effort extraction of previous probabilities for batch-level repair."""
    rows = res.get("questions") or res.get("details") or []
    if len(rows) != n_questions:
        return None
    probs: List[float] = []
    for row in rows:
        if not isinstance(row, dict) or "prob" not in row:
            return None
        try:
            probs.append(float(row["prob"]))
        except Exception:
            return None
    return probs


def load_existing_points(out_dir: Path, coverage_only: bool = False) -> Dict[str, Dict[str, Any]]:
    """Load already materialized point files for full-run aggregation.

    This is mainly used when a resume job is scoped to a subset of points:
    after filling that subset, aggregated.json should still represent every
    point available in the run directory, not just the points in this process.
    """
    per_point: Dict[str, Dict[str, Any]] = {}
    if not out_dir.exists():
        return per_point
    for path in sorted(out_dir.glob("P*_brier.json")):
        pid = path.stem.replace("_brier", "")
        reusable = load_reusable_point(path, coverage_only=coverage_only)
        if reusable is not None:
            per_point[pid] = reusable
    return per_point


def failed_indices_from_existing(
    res: Dict[str, Any],
    batches: List[List[Tuple[int, Dict[str, Any]]]],
    existing_probs: List[float],
) -> tuple[set[int], list[int], str]:
    """Find the smallest reliable repair unit for an existing bad point.

    New outputs carry exact parse-fail indices. Some legacy outputs only have
    a count, so we reconstruct the same batches and use all-0.5 batches as a
    best-effort marker for old fallback-filled answers. Current scoring never
    inserts 0.5 for failures; missing answers remain invalid for scoring.
    """
    exact = res.get("parse_fail_indices")
    if isinstance(exact, list):
        idxs = {int(i) for i in exact if isinstance(i, int) or str(i).isdigit()}
        batch_ids = sorted({
            bi
            for bi, batch in enumerate(batches)
            if any(orig in idxs for orig, _ in batch)
        })
        return idxs, batch_ids, "exact-parse-fail-indices"

    records = res.get("batch_records")
    if isinstance(records, list):
        idxs: set[int] = set()
        batch_ids: set[int] = set()
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_pf = rec.get("parse_fail_indices")
            rec_status = rec.get("status")
            rec_batch_idx = rec.get("batch_idx")
            if isinstance(rec_pf, list) and rec_pf:
                idxs.update(int(i) for i in rec_pf if isinstance(i, int) or str(i).isdigit())
            if rec_status != "ok" and isinstance(rec_batch_idx, int):
                batch_ids.add(rec_batch_idx)
        for bi, batch in enumerate(batches):
            if any(orig in idxs for orig, _ in batch):
                batch_ids.add(bi)
        if idxs or batch_ids:
            if not idxs:
                for bi in batch_ids:
                    idxs.update(orig for orig, _ in batches[bi])
            return idxs, sorted(batch_ids), "batch-records"

    pf_count = int(res.get("parse_fail_count") or 0)
    if pf_count <= 0:
        return set(), [], "no-parse-fail"

    all_half_batches: list[int] = []
    for bi, batch in enumerate(batches):
        if batch and all(abs(existing_probs[orig] - 0.5) < 1e-12 for orig, _ in batch):
            all_half_batches.append(bi)
    all_half_total = sum(len(batches[bi]) for bi in all_half_batches)
    if all_half_batches and all_half_total >= pf_count:
        idxs = {orig for bi in all_half_batches for orig, _ in batches[bi]}
        return idxs, all_half_batches, "legacy-all-half-batches"

    # Last resort for legacy outputs: keep repair scoped to batches containing
    # old 0.5 fallback markers.
    any_half_batches = [
        bi for bi, batch in enumerate(batches)
        if any(abs(existing_probs[orig] - 0.5) < 1e-12 for orig, _ in batch)
    ]
    if any_half_batches:
        idxs = {orig for bi in any_half_batches for orig, _ in batches[bi]}
        return idxs, any_half_batches, "legacy-any-half-batches"

    # If legacy metadata is too weak, preserve correctness by rerunning the
    # point rather than guessing which successful answers to retain.
    return set(range(len(existing_probs))), list(range(len(batches))), "legacy-full-point-fallback"


# ===========================================================================
# Per-point orchestration
# ===========================================================================

def run_point(
    pid: str,
    qb_path: Path,
    ctx_path: Path,
    model: str,
    cfg_cal: Dict[str, Any],
    batch_size: int,
    same_group: int,
    max_parallel: int,
    max_tokens: int,
    thinking_budget: int,
    existing_bad: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    qb = json.loads(qb_path.read_text(encoding="utf-8"))
    cutoff = qb.get("cutoff_date") or ""
    questions = qb.get("brier_questions") or []
    if not questions:
        return {"point_id": pid, "model": model, "n": 0, "score_100": None,
                "batch_size": reported_batch_size(None, batch_size),
                "batch_size_note": batch_size_note(reported_batch_size(None, batch_size)),
                "questions": [], "details": []}
    context = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""

    # SKILL section on tiered max_tokens rules (2026-04-22):
    # context ≥ 10K tokens → max_tokens=10000, thinking.budget_tokens=5000.
    # Approx tokens ≈ characters / 2.5 for mixed Chinese+English markdown.
    approx_tokens = len(context) / 2.5
    if approx_tokens >= 10000 and max_tokens < 10000:
        max_tokens = 10000
        thinking_budget = max(thinking_budget, 5000)
        print(f"  {pid} long context (~{int(approx_tokens)} tok) → max_tokens=10000, thinking=5000")

    indexed = list(enumerate(questions))
    n_q = len(questions)
    # Stable batch layout for bookkeeping + leakage-safe grouping. With
    # batch=999999 a point is one batch ⇒ one request per attempt ⇒ ≤3 / point.
    batches = build_batches(indexed, pid, batch_size, same_group)
    batch_by_index = {
        orig_idx: batch_idx
        for batch_idx, batch in enumerate(batches)
        for orig_idx, _q in batch
    }

    probs: List[Optional[float]] = [None] * n_q
    answer_attempt: List[Optional[int]] = [None] * n_q  # 1/2/3 = attempt that yielded the real answer; None = default 0.5
    last_fail_kind: List[Optional[str]] = [None] * n_q  # 'parse_fail' | 'request_error:..' | None
    call_gate = Semaphore(max(1, max_parallel))
    attempt_records: List[Dict[str, Any]] = []

    def call_one(batch: List[Tuple[int, Dict[str, Any]]]) -> Tuple[Dict[int, float], List[int]]:
        with call_gate:
            return call_batch_once(batch, context, cutoff, model, max_tokens, thinking_budget)

    # ---- Whole-point retry loop: at most MAX_ANSWER_ATTEMPTS attempts, waiting RETRY_WAIT_SECONDS between attempts ----
    # Attempt 1 asks all questions of the point; later attempts only re-ask "still missing / request failed" ones (a P1 point finishes in one shot).
    missing = set(range(n_q))
    genuine_attempt = 0          # genuine attempt count (whole-round quota outages don't count)
    quota_waits = 0              # consecutive "whole round out of balance" waits
    # v3 (fixed by user 2026-07-02): at most 4 rounds per point — 3 normal rounds; if after
    # 3 full rounds there are still DMX (request_error) questions, add a 4th round (one last
    # chance for DMX), capped at 4. Quota (out-of-balance) rounds don't count toward these 4 and wait separately.
    while missing and quota_waits <= MAX_QUOTA_WAITS:
        if genuine_attempt >= MAX_ANSWER_ATTEMPTS + 1:
            break
        if genuine_attempt >= MAX_ANSWER_ATTEMPTS:
            if not any((last_fail_kind[i] or "").startswith("request_error") for i in missing):
                break  # 3 full rounds done and no DMX questions → stop (no 4th round)
        if genuine_attempt > 0 or quota_waits > 0:
            print(f"  {model} / {pid} 等待 {RETRY_WAIT_SECONDS}s 后重问 {len(missing)} 题 "
                  f"(真·尝试 {genuine_attempt}/{MAX_ANSWER_ATTEMPTS}, quota等待 {quota_waits}) ...", flush=True)
            time.sleep(RETRY_WAIT_SECONDS)
        sub_items = [(i, questions[i]) for i in sorted(missing)]
        sub_batches = build_batches(sub_items, pid, batch_size, same_group)
        got: Dict[int, float] = {}
        fail_kind: Dict[int, str] = {}
        result_lock = Lock()

        def process(batch: List[Tuple[int, Dict[str, Any]]]) -> None:
            try:
                ans, _pf = call_one(batch)
            except LLMQuotaError as e:    # out of balance (all keys' quota exhausted): tracked separately, not a retry
                with result_lock:
                    for orig, _q in batch:
                        fail_kind[orig] = f"quota:{str(e)[:80]}"
                return
            except Exception as e:        # other request failures (network/timeout/400 etc.)
                with result_lock:
                    for orig, _q in batch:
                        fail_kind[orig] = f"request_error:{str(e)[:120]}"
                print(f"  {model} / {pid} 请求失效({len(batch)}题): {e}", flush=True)
                return
            with result_lock:
                for pos, (orig, _q) in enumerate(batch, start=1):
                    if pos in ans:
                        got[orig] = ans[pos]
                    else:
                        fail_kind[orig] = "parse_fail"

        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = [ex.submit(process, b) for b in sub_batches]
            for f in as_completed(futs):
                f.result()

        n_quota = sum(1 for k in fail_kind.values() if k.startswith("quota"))
        # Whole round out of balance (0 answers + all failures are quota) → does NOT count as a genuine attempt; wait then retry
        if not got and n_quota > 0 and n_quota == len(fail_kind):
            quota_waits += 1
            for orig in missing:
                last_fail_kind[orig] = "quota_no_balance"
            print(f"  {model} / {pid} ⚠️整轮无余额(quota)→ 不计重试,稍后重试(第 {quota_waits}/{MAX_QUOTA_WAITS} 次)", flush=True)
            continue
        genuine_attempt += 1
        quota_waits = 0
        for orig, p in got.items():
            probs[orig] = p
            answer_attempt[orig] = genuine_attempt
            missing.discard(orig)
        for orig, kind in fail_kind.items():
            if orig in missing:
                last_fail_kind[orig] = kind
        n_req_err = sum(1 for k in fail_kind.values() if k.startswith("request_error"))
        n_parse = sum(1 for k in fail_kind.values() if k == "parse_fail")
        attempt_records.append({
            "attempt": genuine_attempt, "requests": len(sub_batches), "asked": len(sub_items),
            "answered": len(got), "still_missing": len(missing),
            "parse_fail": n_parse, "request_error": n_req_err, "quota_skipped": n_quota,
        })
        print(f"  {model} / {pid} 真·尝试 {genuine_attempt}: 拿到 {len(got)}/{len(sub_items)}, "
              f"仍缺 {len(missing)} (parse_fail={n_parse}, request_error={n_req_err})", flush=True)

    # ---- Still missing after 4 rounds → mark all invalid (v3: no default fill, excluded from primary score); classified by last failure reason (fixed by user 2026-07-02) ----
    # dmx = request_error/quota (upstream failure / out of money); model = parse_fail (model gave no answer / refused).
    # Secondary score (score_with_default) is computed separately inside score_point by filling failed questions with 0.5 (denominator = total = legacy logic).
    fail_kinds: List[Optional[str]] = [None] * n_q
    for i in missing:
        lk = last_fail_kind[i] or ""
        fail_kinds[i] = "dmx" if lk.startswith(("request_error", "quota")) else "model"
    n_dmx_fail = sum(1 for i in missing if fail_kinds[i] == "dmx")
    n_model_fail = sum(1 for i in missing if fail_kinds[i] == "model")
    if missing:
        print(f"  {model} / {pid} 重试后 {len(missing)} 题无答案 → 标无效排除"
              f"(dmx={n_dmx_fail}, model={n_model_fail};主分不算、副分填{DEFAULT_FALLBACK_PROB})", flush=True)

    cutoff_d = parse_date_obj(cutoff)
    enriched: List[Dict[str, Any]] = []
    for qi, (q, p) in enumerate(zip(questions, probs)):
        d_target = parse_date_obj(q.get("d_target"))
        delta_days = (d_target - cutoff_d).days if (d_target and cutoff_d) else 0
        answered = p is not None
        fk = fail_kinds[qi]
        enriched.append({
            **q,
            "prob": p,
            "delta_days": delta_days,
            "gt": int((q.get("answer") if q.get("answer") is not None else q.get("gt", 0)) or 0),
            "question_index": qi,
            "batch_idx": batch_by_index.get(qi),
            "answer_attempt": answer_attempt[qi],
            "is_default_fill": not answered,
            "fail_kind": fk if not answered else None,
            "parse_failed": not answered,
            "valid_for_scoring": answered,
            "invalid_reason": None if answered else ("upstream_error" if fk == "dmx" else "model_no_answer"),
        })

    scoring = score_point(questions, probs, cfg_cal, None, batch_by_index,
                          answer_attempts=answer_attempt, fail_kinds=fail_kinds)
    attempt_hist: Dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "null": 0}
    for a in answer_attempt:
        attempt_hist[str(a) if a is not None else "null"] += 1
    n_answered = int(scoring.get("n_answered", 0))
    return {
        "point_id": pid,
        "model": model,
        "cutoff_date": cutoff,
        "batch_size": reported_batch_size(None, batch_size),
        "batch_size_note": batch_size_note(reported_batch_size(None, batch_size)),
        "questions": enriched,  # legacy field consumed by predict_step4_scorecard
        **scoring,
        "answer_attempt_histogram": attempt_hist,
        "attempt_records": attempt_records,
        "max_answer_attempts": MAX_ANSWER_ATTEMPTS,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "n_dmx_fail": n_dmx_fail,
        "n_model_fail": n_model_fail,
        "invalid_indices": sorted(missing),
        "valid_for_scoring": bool(n_answered),
        "invalid_reason": "all_questions_invalid" if not n_answered else None,
        "validity_policy": "retry4_3plus1dmx_fail_excluded_default_as_secondary",
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="predict-step3B-brier: calibration via LLM probability estimates")
    p.add_argument("--workspace", required=True)
    p.add_argument("--event-name", required=True)
    p.add_argument("--models", required=True, help="comma-separated model IDs")
    p.add_argument("--points", default="all", help="comma-separated point IDs or 'all'")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--same-group", type=int, default=None)
    p.add_argument("--max-parallel", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--thinking-budget", type=int, default=None)
    p.add_argument("--pipeline-config", default=None)
    p.add_argument("--run-id", default=None,
                   help="Reuse a shared run_<ts> dir (orchestrator pins this so step3B/3F coexist)")
    p.add_argument("--coverage-only", action="store_true",
                   help="Reuse existing point files even if they fail the parse-fail quality gate; only fill missing coverage.")
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    s3 = cfg.get("predict_step3_evaluation", {})
    cal_cfg = cfg.get("scoring_calibration", {})

    batch_size = args.batch_size or int(s3.get("batch_max_size", DEFAULT_BATCH_SIZE))
    batch_size_for_metadata = reported_batch_size(args.batch_size, batch_size)
    same_group = args.same_group or int(s3.get("batch_max_same_group", DEFAULT_SAME_GROUP))
    max_parallel = args.max_parallel or int(s3.get("max_parallel_batches", DEFAULT_MAX_PARALLEL))
    max_tokens = args.max_tokens or int(s3.get("max_tokens_per_call", DEFAULT_MAX_TOKENS))
    thinking_budget = args.thinking_budget or int(s3.get("thinking_budget_tokens", DEFAULT_THINKING_BUDGET))
    coverage_only = args.coverage_only or os.environ.get("SB_COVERAGE_ONLY", "0") != "0"

    workspace = Path(args.workspace)
    global LANG, ANSWER_PROMPT, RETRY_PROMPT          # English eval → all-English prompts (2026-07-01)
    LANG = "en" if "英文" in workspace.parts else "zh"
    if LANG == "en":
        ANSWER_PROMPT, RETRY_PROMPT = ANSWER_PROMPT_EN, RETRY_PROMPT_EN
    qb_dir = workspace / "questionbank"
    ctx_dir = workspace / "contexts"
    if not qb_dir.exists():
        raise SystemExit(f"No questionbank/ under {workspace}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = workspace / "results" / f"run_{run_id}" / "brier"

    qb_files = sorted(qb_dir.glob("P*_questionbank.json"))
    if args.points != "all":
        wanted = set(args.points.split(","))
        qb_files = [f for f in qb_files if f.stem.replace("_questionbank", "") in wanted]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    # Points are mutually independent (one request per point) → parallelize over (model × point), using max_parallel to saturate the 3 keys.
    # run_point's internal batch concurrency is set to 1 (batch=999999 means 1 batch/point anyway), avoiding "point-level × batch-level" nested amplification.
    tasks_3b = [
        (model, model_slug(model), qb_path.stem.replace("_questionbank", ""), qb_path)
        for model in models for qb_path in qb_files
    ]
    results_by_model: Dict[str, Dict[str, Any]] = {m: {} for m in models}
    results_lock = Lock()

    def process_point(model: str, slug: str, pid: str, qb_path: Path) -> None:
        ctx_path = ctx_dir / f"{pid}_context.md"
        out_file = out_root / slug / f"{pid}_brier.json"
        reusable = load_reusable_point(out_file, coverage_only=coverage_only)
        if reusable is not None:
            print(f"[brier] {model} / {pid} reuse existing", flush=True)
            with results_lock:
                results_by_model[model][pid] = reusable
            return
        existing_bad: Optional[Dict[str, Any]] = None
        if out_file.exists():
            try:
                existing_bad = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception:
                existing_bad = None
        print(f"[brier] {model} / {pid} ...", flush=True)
        res = run_point(
            pid, qb_path, ctx_path, model, cal_cfg,
            batch_size, same_group, 1, max_tokens, thinking_budget,
            existing_bad=existing_bad,
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        with results_lock:
            results_by_model[model][pid] = res

    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        futs = [ex.submit(process_point, *t) for t in tasks_3b]
        for f in as_completed(futs):
            f.result()

    for model in models:
        slug = model_slug(model)
        per_point = results_by_model[model]
        # Cross-point aggregation
        if coverage_only:
            per_point.update(load_existing_points(out_root / slug, coverage_only=True))
        agg = aggregate_across_points(per_point, cal_cfg)
        agg_batch_size, batch_size_by_point, batch_size_source = summarize_point_batch_size(
            per_point,
            batch_size_for_metadata,
        )
        agg_path = out_root / slug / "aggregated.json"
        agg_path.write_text(
            json.dumps({
                "model": model,
                "n_points": len(per_point),
                "batch_size": agg_batch_size,
                "batch_size_source": batch_size_source,
                "batch_size_by_point": batch_size_by_point,
                "validity_policy": "retry4_3plus1dmx_fail_excluded_default_as_secondary",
                "max_answer_attempts": MAX_ANSWER_ATTEMPTS,
                "retry_wait_seconds": RETRY_WAIT_SECONDS,
                "fallback_filled_count": sum(int(r.get("fallback_filled_count") or 0) for r in per_point.values()),
                "answer_attempt_histogram": {
                    k: sum(int((r.get("answer_attempt_histogram") or {}).get(k, 0)) for r in per_point.values())
                    for k in ("1", "2", "3", "null")
                },
                "aggregated": agg,
                "per_point": {pid: {k: v for k, v in r.items() if k not in ("questions", "details")}
                              for pid, r in per_point.items()},
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"run_dir": str(out_root.parent), "models": models,
                      "points_done": len(qb_files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
