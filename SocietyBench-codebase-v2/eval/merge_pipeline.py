#!/usr/bin/env python3
"""merge-pipeline (1:1 of skills/merge-pipeline/SKILL.md).

Runs the merge chain:
  Step 5  — combine web + media short timelines (per-day [W]/[M]/[WM] merge)
  Step 6  — expand to long-form (events + opinion)
  Step 6.5 — GT refine for predict input (Round 1 baseline + iteration log)

Step 6.5 only runs Round 1 automatically; the iteration log seeds Rounds 2/3
plus the mandatory human-approval entry that gates predict.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parent


def run(name: str, cmd: list[str]) -> None:
    print(f"\n========== [{name}] {' '.join(cmd)} ==========")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    # SKILL argument-hint: <web_timeline_short> <media_timeline_short>
    # <web_step1_jsonl> <media_step1_jsonl> <media_step3_jsonl> <output_dir>
    p = argparse.ArgumentParser(description="merge-pipeline: short merge + long expand + GT refine")
    p.add_argument("web_timeline_short", help="web-side short timeline (produced by web-step5)")
    p.add_argument("media_timeline_short", help="media-side short timeline (CC-reviewed & compressed on the media side)")
    p.add_argument("web_step1_jsonl", help="web-side step1 original summaries (for long-form expansion)")
    p.add_argument("media_step1_jsonl", help="media-side step1 original summaries (for long-form expansion)")
    p.add_argument("media_step3_jsonl", help="media-side step3 deduped data (for opinion attachment; do not delete)")
    p.add_argument("output_dir", help="merged output directory")
    # SKILL §argument-hint lists only the 6 positional args; event-name is
    # optional, defaulting to a generic label so standalone runs don't fail.
    p.add_argument("--event-name", default="事件")
    p.add_argument("--skip-gt-refine", action="store_true",
                   help="Skip Step 6.5 GT refine (only run when you'll handle predict-GT separately)")
    p.add_argument("--articles-json", default=None,
                   help="Optional gt_workspace/timeline_articles.json — when given, Step 6.5 "
                        "performs SKILL §date fallback bundle repair for short GT nodes.")
    p.add_argument("--strict-approval", action="store_true",
                   help="Per SKILL §completion check: hard-exit if iteration log lacks '人工审核通过，可进入 predict'. "
                        "Default behavior is to warn; predict-pipeline's Step -1 gate still enforces this strictly.")
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    a = p.parse_args()

    py = sys.executable
    base = ["--event-name", a.event_name]
    if a.model:
        base += ["--model", a.model]
    if a.pipeline_config:
        base += ["--pipeline-config", a.pipeline_config]

    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    web_short = Path(a.web_timeline_short)
    media_short = Path(a.media_timeline_short)
    web_step1 = Path(a.web_step1_jsonl)
    media_step1 = Path(a.media_step1_jsonl)
    media_step3 = Path(a.media_step3_jsonl)

    for required in (web_short, media_short, web_step1, media_step1, media_step3):
        if not required.exists():
            raise SystemExit(f"missing required input: {required}")

    merged_short = output_dir / "timeline_merged_short.md"
    merged_long = output_dir / "timeline_merged_long.md"

    # Step 5
    if not merged_short.exists() or merged_short.stat().st_size == 0:
        run("merge-step5", [py, str(D / "merge_step5_short.py"),
                            str(web_short), str(media_short), str(output_dir), *base])
    else:
        print(f"[merge-step5] {merged_short} already exists — skipping")

    # Step 6
    if not merged_long.exists() or merged_long.stat().st_size == 0:
        run("merge-step6", [py, str(D / "merge_step6_long.py"),
                            str(merged_short),
                            str(web_step1), str(media_step1), str(media_step3),
                            str(output_dir), *base])
    else:
        print(f"[merge-step6] {merged_long} already exists — skipping")

    # SKILL §"completion check" — Step 6 produced an md whose date-node count is close
    # to Step 5's. Catches silent truncation where the file exists but only a
    # handful of nodes made it through.
    date_re = re.compile(r"^###\s+\d{4}-\d{2}-\d{2}")
    short_nodes = sum(1 for ln in merged_short.read_text(encoding="utf-8").splitlines()
                      if date_re.match(ln.strip()))
    long_nodes = sum(1 for ln in merged_long.read_text(encoding="utf-8").splitlines()
                     if date_re.match(ln.strip()))
    if short_nodes == 0:
        raise SystemExit(f"[merge-pipeline §完成检查] {merged_short} has 0 date nodes — Step 5 broken")
    retention = long_nodes / short_nodes
    print(f"[merge-pipeline §完成检查] date nodes: short={short_nodes} long={long_nodes} ({retention:.0%})")
    if retention < 0.8:
        raise SystemExit(
            f"[merge-pipeline §完成检查] merged_long has {long_nodes} date nodes vs "
            f"merged_short {short_nodes} (<80% retained) — Step 6 likely truncated"
        )

    # Step 6.5: GT refine
    if a.skip_gt_refine:
        print("[merge-step6.5] skipped (--skip-gt-refine)")
        return

    refined = output_dir / "timeline_merged_long_gt.md"
    report_json = output_dir / "timeline_merged_long_gt_report.json"
    iter_log = output_dir / "timeline_merged_long_gt_iteration_log.md"
    if not refined.exists() or refined.stat().st_size == 0:
        s65_cmd = [py, str(D / "merge_step6_5_gt_refine.py"),
                   str(merged_long), str(output_dir), *base]
        if a.articles_json:
            s65_cmd += ["--articles-json", a.articles_json]
        run("merge-step6.5", s65_cmd)
    else:
        print(f"[merge-step6.5] {refined} already exists — skipping (delete to re-run)")

    # SKILL §"completion check": refined GT + report.json + iteration log all required
    missing = [str(p) for p in (refined, report_json, iter_log) if not p.exists()]
    if missing:
        raise SystemExit("Step 6.5 did not produce all required outputs: " + ", ".join(missing))

    # SKILL §"completion check": iteration log must contain explicit CC approval entry.
    # In strict mode, hard-exit; otherwise warn (predict-pipeline's Step -1 gate
    # will still refuse to run on an unapproved GT).
    log_text = iter_log.read_text(encoding="utf-8")
    approved = "人工审核通过" in log_text
    if not approved:
        msg = (
            "iteration log at {p} does NOT contain '人工审核通过，可进入 predict'. "
            "Complete Rounds 2/3 + CC review and append the approval line."
        ).format(p=iter_log)
        if a.strict_approval:
            raise SystemExit("[merge-pipeline §strict-approval] " + msg)
        print("\n⚠️  " + msg + "\n   (use --strict-approval to make this a hard error)")

    print("\n═══════════════════════════════════")
    print(f"  Merge Pipeline 完成 — {a.event_name}")
    print("═══════════════════════════════════")
    print(f"  合并短版：{merged_short}")
    print(f"  合并长版：{merged_long}")
    print(f"  预测 GT： {refined}")
    print(f"  报告：    {report_json}")
    print(f"  迭代记录：{iter_log}  {'✅ approved' if approved else '⚠️ pending CC review'}")


if __name__ == "__main__":
    main()
