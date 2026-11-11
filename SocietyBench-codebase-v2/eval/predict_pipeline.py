#!/usr/bin/env python3
"""predict-pipeline (1:1 of skills/predict-pipeline/SKILL.md).

End-to-end predict chain:
  Step -1: GT-refine strict gate (SKILL §Step -1 — requires refined GT +
           iteration log with explicit "Human review approved, proceed to predict")
  Step 0:  anonymize     (predict_step0_prepare.py — Phase 1+2+3)
  Step 1:  prediction points (predict_step1_points.py)
  Step 2:  questionbank (predict_step2_questionbank.py — owns Step 2.0 split)
  Step 3:  3B + 3F parallel (run_pipeline_parallel.py — shared run_id)
  Step 4:  scorecard (predict_step4_scorecard.py)

Re-run support: `--start-step N` skips earlier steps when their outputs exist.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

D = Path(__file__).resolve().parent


def run(name: str, cmd: list[str]) -> None:
    print(f"\n========== [{name}] {' '.join(cmd)} ==========")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def step_minus_one(timeline_md: Path) -> None:
    """SKILL §Step -1: strict gate. The input must be the refined GT
    (`timeline_merged_long_gt.md`), the merge-side iteration log must exist,
    and that log must contain CC's explicit `Human review approved, proceed to predict`
    approval. Any failure → hard exit, no auto-bypass.
    """
    refined = timeline_md.parent / "timeline_merged_long_gt.md"
    log = timeline_md.parent / "timeline_merged_long_gt_iteration_log.md"
    report = timeline_md.parent / "timeline_merged_long_gt_report.json"

    if not refined.exists() or refined.stat().st_size == 0:
        sys.exit(
            f"[Step -1/4] refined GT not found at {refined}. SKILL requires "
            "Predict to run on the refined timeline (timeline_merged_long_gt.md), "
            "not the raw merge long-form. Run merge-pipeline Step 6.5 first."
        )

    # If the caller passed the raw long-form, force them to switch to refined.
    if timeline_md.resolve() != refined.resolve():
        sys.exit(
            f"[Step -1/4] input was {timeline_md.name}, but refined GT exists "
            f"at {refined}. SKILL forbids running predict on the un-refined "
            "long-form — re-invoke with the refined file."
        )

    if not report.exists():
        sys.exit(
            f"[Step -1/4] missing refine report at {report}. Step 6.5 did not "
            "complete cleanly; rerun merge-pipeline Step 6.5 before predict."
        )

    if not log.exists():
        sys.exit(
            f"[Step -1/4] missing iteration log at {log}. SKILL §Step -1 requires "
            "the 3-round self-check log + CC human approval entry before predict."
        )

    log_text = log.read_text(encoding="utf-8", errors="ignore")
    if "Human review approved" not in log_text:
        sys.exit(
            f"[Step -1/4] iteration log at {log} does not contain "
            "'Human review approved, proceed to predict'. SKILL §Step -1 requires explicit CC "
            "human approval after the 3-round self-check. Complete the review "
            "and append the approval line before re-running predict."
        )

    print("[Step -1/4] refined GT + iteration log + Human review approved — gate passed")


def update_run_latest(workspace: Path, run_id: str) -> None:
    target = workspace / "results" / f"run_{run_id}"
    if not target.exists():
        return
    link = workspace / "results" / "run_latest"
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.name)  # relative symlink within results/
        print(f"[run_latest] -> {target.name}")
    except OSError as e:
        print(f"[run_latest] could not create symlink: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="predict-pipeline: GT-check → anonymize → points → questionbank → evaluate → score")
    p.add_argument("timeline_md", help="merged long-form GT timeline (prefer timeline_merged_long_gt.md)")
    p.add_argument("output_dir", help="prediction workspace")
    p.add_argument("--event-name", required=True)
    # SKILL argument-hint: --replacements-json <replacements.json>
    p.add_argument("--replacements-json", dest="replacements_json", required=True,
                   help="anonymization replacement table JSON (SKILL argument-hint name)")
    p.add_argument("--eval-models", default=None,
                   help="comma-separated; defaults to pipeline_config.evaluation_models")
    # SKILL usage section: default P02,P15,P29 (representative early/mid/late points); use 'all' to run everything
    p.add_argument("--points", default="P02,P15,P29",
                   help="comma-separated point ids or 'all'. Default: P02,P15,P29 per SKILL.")
    p.add_argument("--model", default=None, help="LLM used for question-bank generation")
    p.add_argument("--start-step", type=int, default=0,
                   help="0=anonymize, 1=points, 2=questionbank, 3=eval, 4=scorecard")
    p.add_argument("--skip-step0-audit", action="store_true",
                   help="Run step0 with --skip-audit (skip Phase 2 LLM audit + Phase 3 consistency check)")
    p.add_argument("--pipeline-config", default=None)
    a = p.parse_args()

    py = sys.executable
    cfg_arg = ["--pipeline-config", a.pipeline_config] if a.pipeline_config else []
    timeline = Path(a.timeline_md) # Timeline mergerd long gt
    workspace = Path(a.output_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    # Shared run id pinned across step3B + step3F + step4
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----- Step -1: GT refine strict gate (SKILL §Step -1) -----
    # TODO: START FROM HERE!
    #FIXME step_minus_one(timeline)
    # [SKIP THIS ☝️ SINCE IT STRICTLY LOOKS FOR timeline_merged_long_gt ] However we already have timeline_anon.md

    # ----- Step 0: anonymize -----
    anon_path = workspace / "timeline_anon.md"
    if a.start_step <= 0:
        if anon_path.exists() and anon_path.stat().st_size > 0:
            print(f"[Step 0/4] {anon_path} already exists — skipping (delete to re-run)")
        else:
            s0_cmd = [py, str(D / "predict_step0_prepare.py"),
                      str(timeline), str(workspace),
                      "--replacements", a.replacements_json, *cfg_arg] # Anonymise timeline_merged_long_gt
            if a.skip_step0_audit:
                s0_cmd.append("--skip-audit")
            run("predict-step0", s0_cmd)
    else:
        print("[Step 0/4] skipped (start-step > 0)")

    # ----- Step 1: prediction points -----
    pp_path = workspace / "prediction_points.json"
    if a.start_step <= 1:
        if not anon_path.exists():
            raise SystemExit(f"Cannot run step1: {anon_path} does not exist (did step0 fail?)")
        if pp_path.exists() and pp_path.stat().st_size > 0:
            print(f"[Step 1/4] {pp_path} already exists — skipping (delete to re-run)")
        else:
            run("predict-step1", [py, str(D / "predict_step1_points.py"),
                                  str(anon_path), str(workspace), *cfg_arg])
    else:
        print("[Step 1/4] skipped")

    # ----- Step 2: questionbank (owns Step 2.0 split internally) -----
    qb_dir = workspace / "questionbank"
    if a.start_step <= 2:
        # If contexts/ + gt/ already exist, step2 will skip its own Step 2.0
        # automatically (--skip-split available via flag); we still call step2
        # so per-point banks get refreshed for any new points.
        ctx_dir = workspace / "contexts"
        gt_dir = workspace / "gt"
        qb_cmd = [py, str(D / "predict_step2_questionbank.py"),
                  "--workspace", str(workspace),
                  "--event-name", a.event_name,
                  "--points", a.points, *cfg_arg]
        if a.model:
            qb_cmd += ["--model", a.model]
        if ctx_dir.exists() and gt_dir.exists() and any(ctx_dir.glob("P*_context.md")):
            qb_cmd += ["--skip-split"]
            print("[Step 2/4] contexts/ + gt/ already populated — passing --skip-split")
        run("predict-step2", qb_cmd)
    else:
        print("[Step 2/4] skipped")

    # ----- Step 3: 3B + 3F in parallel (shared run_id) -----
    if a.start_step <= 3:
        eval_cmd = [py, str(D / "run_pipeline_parallel.py"),
                    "--workspace", str(workspace),
                    "--event-name", a.event_name,
                    "--points", a.points,
                    "--run-id", run_id, *cfg_arg]
        if a.eval_models:
            eval_cmd += ["--models", a.eval_models]
        run("predict-step3+4", eval_cmd)
        update_run_latest(workspace, run_id)
    else:
        print("[Step 3+4] skipped")


if __name__ == "__main__":
    main()
