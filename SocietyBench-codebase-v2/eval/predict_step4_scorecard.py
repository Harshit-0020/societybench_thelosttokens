#!/usr/bin/env python3
"""predict-step4-scorecard (1:1 of skills/predict-step4-scorecard/SKILL.md).

Purely algorithmic — no LLM. Reads per-point Brier outputs (3B) and per-point
temporal outputs (3F) from a workspace's `results/run_<ts>/` directory and
emits the two-axis scorecard.

Cross-point aggregation is **pooled**, not per-point averaged (per SKILL §跨点
汇总标准):
  - 3B:  pooled weighted MAE across all valid questions of all selected points,
         score = 100 × max(0, 1 - wMAE / baseline)        (baseline = 0.50)
  - 3F:  pooled wMAE_days vs pooled baseline wMAE_days,
         score = 100 / (1 + wMAE_days / baseline_wmae)

**No combined / overall score is emitted** — 3B and 3F rank independently.

Mapping to the paper (Sec. 3.3): the reported calibration score S_cal is
Eq. (1), computed in predict_step3B_brier.py; the reported temporal score
S_time is Eqs. (2)-(3), computed as `v2_finalscore` in predict_step3F_time.py
(the hyperbolic pooled score above is kept as a legacy secondary field).
Per Sec. 3.3 "Aggregation", scores are computed independently per event and
the paper's headline number is the cross-event mean of the five per-event
scores.

Output:
  <run_dir>/scorecard.json     — overall summary with separate 3B/3F rankings
  <run_dir>/scorecard.md       — human-readable markdown
  <run_dir>/scorecards/scorecard_<model_slug>.json  — per-model breakdown
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import load_pipeline_config


# ===========================================================================
# Pooled calibration (3B)
# ===========================================================================

def pool_calibration(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    coef = float(cfg.get("time_factor_coefficient", 0.04))
    offset = float(cfg.get("window_factor_offset", 4))
    baseline = float(cfg.get("baseline_uniform_predictor_wmae", 0.50))

    num = 0.0
    denom = 0.0
    n = 0
    invalid = 0
    for r in records:
        if not r.get("valid_for_scoring", not r.get("parse_failed", False)):
            invalid += 1
            continue
        try:
            p = float(r.get("prob"))
            y = float(r.get("gt") if "gt" in r else r.get("answer", 0))
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not (0.0 <= p <= 1.0) or y not in (0.0, 1.0):
            invalid += 1
            continue
        days_fc = float(r.get("days_from_cutoff", r.get("delta_days", 0)) or 0)
        window_days = float(r.get("window_days", 1) or 1)
        w_t = 1.0 / (1.0 + coef * max(0.0, days_fc))
        w_w = window_days / (window_days + offset)
        w = w_t * w_w
        num += w * abs(p - y)
        denom += w
        n += 1

    if denom <= 0 or n == 0:
        return {"wmae": None, "n": 0, "n_total": len(records), "n_invalid": invalid, "score_100": None}
    wmae = num / denom
    score = 100.0 * max(0.0, 1.0 - wmae / baseline)
    return {"wmae": wmae, "n": n, "n_total": len(records), "n_invalid": invalid, "score_100": score}


# ===========================================================================
# Pooled temporal (3F)
# ===========================================================================

ALPHA = 0.04
SEGMENTS = [(0, 30, 15.5), (31, 60, 45.5), (61, 90, 75.5)]  # 2026-07-01 d=0 fix: matches 3F, first segment includes the cutoff day itself


def segment_midpoint(day: int) -> float:
    for lo, hi, mid in SEGMENTS:
        if lo <= day <= hi:
            return mid
    return 45.5


def pool_temporal(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    w_total = 0.0
    model_werr = 0.0
    baseline_werr = 0.0
    v2_num = 0.0   # v2_finalscore (main score since 2026-07-06): band span / 30, clipped to segment window, perfect = 0
    n = 0
    for r in records:
        try:
            err = float(r.get("error_days"))
        except (TypeError, ValueError):
            continue
        if err < 0 or err >= 999:
            continue
        d = int(r.get("delta_days", 0) or 0)
        if d > 90 or d < 0:   # 2026-07-01: includes d=0 (consistent with the pooled_score convention)
            continue
        w = 1.0 / (1.0 + ALPHA * d)
        w_total += w
        model_werr += err * w
        baseline_werr += abs(d - segment_midpoint(d)) * w
        _seg = next(((lo, hi) for lo, hi, _ in SEGMENTS if lo <= d <= hi), None)
        if _seg is not None:
            _lo, _hi = _seg
            v2_num += w * min(1.0, (min(_hi, d + err) - max(_lo, d - err)) / 30.0)
        n += 1
    if w_total <= 0 or n == 0:
        return {"wmae_days": None, "n": 0, "baseline_wmae_days": None, "score_100": None,
                "v2_finalscore": None}
    wmae_days = model_werr / w_total
    baseline_wmae = baseline_werr / w_total
    score = 100.0 / (1.0 + wmae_days / baseline_wmae) if baseline_wmae > 0 else 0.0
    return {
        "wmae_days": wmae_days,
        "baseline_wmae_days": baseline_wmae,
        "n": n,
        "score_100": score,
        "v2_finalscore": round(100.0 * max(0.0, 1.0 - v2_num / w_total), 2),
    }


# ===========================================================================
# Discovery
# ===========================================================================

def read_aggregated(run_dir: Path, axis: str) -> Dict[str, Dict[str, Any]]:
    """SKILL §"多模型总榜读 pooled.3B / pooled.3F": if Step 3B / 3F has already
    written aggregated.json per model under brier/<model>/ or time/<model>/,
    prefer that as the pooled score source instead of re-aggregating here.
    Returns {model_slug: aggregated_payload} (empty when not available).
    """
    axis_root = run_dir / axis  # "brier" or "time"
    if not axis_root.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for model_dir in sorted([d for d in axis_root.iterdir() if d.is_dir()]):
        f = model_dir / "aggregated.json"
        if f.exists() and f.stat().st_size > 0:
            try:
                out[model_dir.name] = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return out


def collect_brier(run_dir: Path, points_filter: Optional[set]) -> Dict[str, List[Dict[str, Any]]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    brier_root = run_dir / "brier"
    if not brier_root.exists():
        return by_model
    for model_dir in sorted([d for d in brier_root.iterdir() if d.is_dir()]):
        records: List[Dict[str, Any]] = []
        for f in sorted(model_dir.glob("P*_brier.json")):
            pid = f.stem.replace("_brier", "")
            if points_filter and pid not in points_filter:
                continue
            payload = json.loads(f.read_text(encoding="utf-8"))
            qs = payload.get("questions") if isinstance(payload, dict) else payload
            if isinstance(qs, list):
                parse_fail_indices = set()
                parse_fail_count = 0
                has_row_validity = False
                if isinstance(payload, dict):
                    parse_fail_count = int(payload.get("parse_fail_count") or 0)
                    raw_indices = payload.get("parse_fail_indices") or []
                    if isinstance(raw_indices, list):
                        parse_fail_indices = {
                            int(x) for x in raw_indices
                            if isinstance(x, int) or str(x).isdigit()
                        }
                for idx, q in enumerate(qs):
                    if not isinstance(q, dict):
                        continue
                    row = dict(q)
                    has_row_validity = has_row_validity or (
                        "valid_for_scoring" in row or "parse_failed" in row
                    )
                    if idx in parse_fail_indices:
                        # 2026-06-30 user decision: questions filled with the default (3B=0.5)
                        # after 3 hourly retries are scored as usual. As long as a usable
                        # probability exists (default 0.5) the row stays valid; invalidate only
                        # when there is truly no probability.
                        row["parse_failed"] = True
                        row.setdefault("is_default_fill", True)
                        if row.get("prob") is None:
                            row["valid_for_scoring"] = False
                            row.setdefault("invalid_reason", "no_model_answer_no_default")
                        else:
                            row["valid_for_scoring"] = True
                    records.append(row)
                if parse_fail_count and not parse_fail_indices and not has_row_validity:
                    for row in records[-len(qs):]:
                        if row.get("prob") is None:
                            row["parse_failed"] = True
                            row["valid_for_scoring"] = False
                            row.setdefault("invalid_reason", "legacy_parse_fail_without_indices")
        by_model[model_dir.name] = records
    return by_model


def collect_temporal(run_dir: Path, points_filter: Optional[set]) -> Dict[str, List[Dict[str, Any]]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for model_dir in sorted([d for d in (run_dir / "time").iterdir() if d.is_dir()]) \
            if (run_dir / "time").exists() else []:
        records: List[Dict[str, Any]] = []
        for f in sorted(model_dir.glob("P*_time.json")):
            pid = f.stem.replace("_time", "")
            if points_filter and pid not in points_filter:
                continue
            payload = json.loads(f.read_text(encoding="utf-8"))
            evs = payload.get("events") if isinstance(payload, dict) else payload
            if isinstance(evs, list):
                records.extend(e for e in evs if isinstance(e, dict))
        by_model[model_dir.name] = records
    return by_model


def collect_run_stats(run_dir: Path, points_filter: Optional[set]) -> Dict[str, Dict[str, Any]]:
    """SKILL §"当前标准" (current standard): the scorecard must separately report
    `3B_runs` (1 run) and `3F_runs` (2 runs, with `3F_avg` / `3F_range`).
    Aggregate per-run stats from each Pxx_time.json file's `per_run` list.
    """
    out: Dict[str, Dict[str, Any]] = {}
    time_root = run_dir / "time"
    if not time_root.exists():
        return out
    for model_dir in sorted([d for d in time_root.iterdir() if d.is_dir()]):
        per_run_wmaes: List[float] = []
        n_complete = 0
        n_total = 0
        for f in sorted(model_dir.glob("P*_time.json")):
            pid = f.stem.replace("_time", "")
            if points_filter and pid not in points_filter:
                continue
            n_total += 1
            payload = json.loads(f.read_text(encoding="utf-8"))
            per_run = payload.get("per_run") or []
            if len(per_run) >= 2 and all(r.get("weighted_mae_days") is not None for r in per_run):
                n_complete += 1
                for r in per_run:
                    per_run_wmaes.append(float(r["weighted_mae_days"]))
        avg = sum(per_run_wmaes) / len(per_run_wmaes) if per_run_wmaes else None
        rng = (max(per_run_wmaes) - min(per_run_wmaes)) if per_run_wmaes else None
        out[model_dir.name] = {
            "points_total": n_total,
            "points_with_2_runs": n_complete,
            "3F_avg_wmae": avg,
            "3F_range_wmae": rng,
        }
    return out


def find_latest_run(results_dir: Path) -> Optional[Path]:
    cands = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    return cands[-1] if cands else None


# ===========================================================================
# Rendering
# ===========================================================================

def _fmt(v: Any, spec: str = ".2f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return "N/A"


def render_md(scorecard: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Scorecard\n")
    lines.append(f"- workspace: `{scorecard['workspace']}`")
    lines.append(f"- run: `{scorecard['run_dir']}`")
    lines.append("")
    lines.append("## 总览（pooled across all selected points）\n")
    lines.append("| Model | 3B (S_cal) | 3B wMAE | n_3B | 3F (S_time) | 3F wMAE_days | baseline | n_3F |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in scorecard["entries"]:
        c, t = row["cal"], row["time"]
        lines.append(
            f"| {row['model']} | {_fmt(c['score_100'])} | {_fmt(c['wmae'], '.4f')} | "
            f"{c['n']} | {_fmt(t['score_100'])} | "
            f"{_fmt(t['wmae_days'])} | {_fmt(t['baseline_wmae_days'])} | {t['n']} |"
        )

    medals = ["🥇", "🥈", "🥉"]

    def _rank(key: str, label: str) -> None:
        lines.append(f"\n## {label}\n")
        eligible = [r for r in scorecard["entries"] if r[key]["score_100"] is not None]
        eligible.sort(key=lambda r: -r[key]["score_100"])
        if not eligible:
            lines.append("_no eligible models_")
            return
        for i, r in enumerate(eligible):
            badge = medals[i] if i < len(medals) else "  "
            lines.append(f"- {badge} **{r['model']}** — {r[key]['score_100']:.2f}")

    _rank("cal", "3B 榜 (Brier 概率校准)")
    _rank("time", "3F 榜 (时间精度)")
    lines.append("")
    lines.append("> 注：3B / 3F 都是跨预测点 **pooled** 后的最终分数（非 per-point 平均），且不输出综合排名。")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="predict-step4-scorecard: dual-axis pooled aggregation")
    p.add_argument("--workspace", required=True)
    p.add_argument("--run-id", default=None,
                   help="Specific run_<ts> directory; defaults to latest run.")
    p.add_argument("--models", default=None,
                   help="Comma-separated model slugs (filesystem names) to include")
    p.add_argument("--points", default=None,
                   help="Comma-separated point ids to include (e.g. P01,P02)")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    cal_cfg = cfg.get("scoring_calibration", {})

    workspace = Path(args.workspace)
    results_dir = workspace / "results"
    if not results_dir.exists():
        raise SystemExit(f"No results/ under workspace: {workspace}")

    if args.run_id:
        run_dir = results_dir / (args.run_id if args.run_id.startswith("run_")
                                 else f"run_{args.run_id}")
    else:
        run_dir = find_latest_run(results_dir)
    if run_dir is None or not run_dir.exists():
        raise SystemExit(f"No run_* directory in {results_dir}")

    points_filter: Optional[set] = (set(args.points.split(",")) if args.points else None)
    model_filter: Optional[set] = (set(args.models.split(",")) if args.models else None)

    brier_by_model = collect_brier(run_dir, points_filter)
    time_by_model = collect_temporal(run_dir, points_filter)
    run_stats = collect_run_stats(run_dir, points_filter)

    # SKILL §"多模型总榜读 pooled": prefer aggregated.json if Step 3B/3F wrote it
    brier_agg = read_aggregated(run_dir, "brier")
    time_agg = read_aggregated(run_dir, "time")

    models = sorted(set(brier_by_model) | set(time_by_model))
    if model_filter:
        models = [m for m in models if m in model_filter]

    def _coerce_cal_from_agg(agg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not agg:
            return None
        if agg.get("validity_policy") not in (
            "parse_failed_or_missing_answers_excluded",   # old convention: default-filled questions excluded
            "retry3_5min_then_default_0.5_marked",        # new convention (2026-06-30 user decision): default-filled questions scored as 0.5, already correctly included by aggregate_across_points
        ):
            return None
        a = agg.get("aggregated") or {}
        if a.get("score_100") is None:
            return None
        return {"wmae": a.get("weighted_mae"), "n": a.get("n", 0),
                "n_total": a.get("n_total", a.get("n", 0)),
                "n_invalid": a.get("n_invalid", 0),
                "score_100": a.get("score_100"),
                "source": "aggregated.json"}

    def _coerce_time_from_agg(agg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not agg:
            return None
        a = agg.get("pooled") or {}
        if a.get("score_100") is None:
            return None
        # Count n by summing per-point events scored
        n = 0
        for v in (agg.get("per_point") or {}).values():
            n += int(v.get("events_scored", 0) or 0)
        return {"wmae_days": a.get("weighted_mae_days"), "n": n,
                "baseline_wmae_days": a.get("baseline_wmae"),
                "score_100": a.get("score_100"), "source": "aggregated.json"}

    entries: List[Dict[str, Any]] = []
    scorecards_dir = run_dir / "scorecards"
    scorecards_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        cal_from_agg = _coerce_cal_from_agg(brier_agg.get(m))
        tim_from_agg = _coerce_time_from_agg(time_agg.get(m))
        cal = cal_from_agg if cal_from_agg else pool_calibration(brier_by_model.get(m, []), cal_cfg)
        tim = tim_from_agg if tim_from_agg else pool_temporal(time_by_model.get(m, []))
        # SKILL §当前标准 (2026-04-22):
        # report `3B_runs` (=1) and `3F_runs` (=2) along with 3F_avg/3F_range.
        rs = run_stats.get(m, {})
        runs_attestation = {
            "3B_runs": 1,
            "3F_runs": 2,
            "3F_points_with_2_runs": rs.get("points_with_2_runs"),
            "3F_points_total": rs.get("points_total"),
            "3F_avg_wmae": rs.get("3F_avg_wmae"),
            "3F_range_wmae": rs.get("3F_range_wmae"),
        }
        entries.append({"model": m, "cal": cal, "time": tim, "runs": runs_attestation})
        (scorecards_dir / f"scorecard_{m}.json").write_text(
            json.dumps({"model": m, "cal": cal, "time": tim, "runs": runs_attestation},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Order rows alphabetically; ranking happens separately in render_md.
    entries.sort(key=lambda r: r["model"])

    scorecard = {
        "workspace": str(workspace),
        "run_dir": str(run_dir),
        "config_used": {"scoring_calibration": cal_cfg, "scoring_temporal_segments": SEGMENTS},
        "points_filter": sorted(points_filter) if points_filter else None,
        "entries": entries,
    }
    (run_dir / "scorecard.json").write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render_md(scorecard)
    (run_dir / "scorecard.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
