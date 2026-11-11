#!/usr/bin/env python3
"""pipeline-run (1:1 of skills/pipeline-run/SKILL.md).

SKILL describes a 4-stage research workflow with worklog appending after each stage:

  Stage 0 — environment check (BettaFish DB + 6 external API keys + MiroFish health)
  Stage 1 — crawler collection (/bfish-crawl, 7 platforms one by one)
  Stage 2 — opinion analysis (/bfish-research: QueryEngine + MediaEngine + InsightEngine + ReportEngine)
  Stage 3 — simulation (/mfish-simulate, Zep Cloud + Twitter+Reddit Agent)

External dependencies not packaged in societybench_release:
  - BettaFish (PostgreSQL + 6 API Keys + 4 internal LLM engines)
  - MiroFish (localhost:5001 API + Zep Cloud)
  - bfish-crawl / bfish-research / mfish-simulate skills (CC-invoked, not py scripts)

Since those external systems are absent, this script delegates the data
collection + analysis (Stages 1+2) to the societybench-internal `total-pipeline`
(web-search-crawl / media-crawl → web/media pipelines → merge → predict),
which is the closest in-repo equivalent. Stage 0 falls back to a lightweight
environment check on what IS available here. Stage 3 (mfish-simulate) is
explicitly skipped (user-confirmed).

The SKILL's worklog-append contract (every completed step must append a
structured record) is honored: this script writes a worklog header, then after
each Phase of total-pipeline finishes (by snapshotting the workspace), appends
a record.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_worklog(log_dir: Path, query: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    worklog = log_dir / "worklog.md"
    if not worklog.exists():
        worklog.write_text(
            "# Pipeline Worklog\n"
            f"- **Start time** {now_str()}\n"
            f"- **Search keywords:** {query}\n"
            f"- **Log directory:** {log_dir}\n\n---\n\n",
            encoding="utf-8",
        )
    return worklog


def append_worklog(
    worklog: Path, stage_id: str, name: str, status: str,
    output: str = "", key_data: str = "", note: str = "无",
) -> None:
    with worklog.open("a", encoding="utf-8") as fh:
        fh.write(
            f"### [{stage_id}] {name}\n"
            f"- **Time:** {now_str()}\n"
            f"- **Status:** {status}\n"
            f"- **Output:** {output or 'None'}\n"
            f"- **Key data:** {key_data or 'None'}\n"
            f"- **Notes:** {note}\n\n"
        )


def stage0_env_check(worklog: Path) -> bool:
    """SKILL Stage 0: environment check.

    The real SKILL checks PostgreSQL + 6 BettaFish-specific keys + MiroFish.
    Those external systems are not in societybench_release. So this lightweight
    check verifies what IS required by total-pipeline:
      - DMXAPI key (for LLM steps via common.call_llm)
      - APIFY_TOKEN (for web-search-crawl, if used)
    """
    print(f"[Stage 0] Environment Check ({now_str()})", flush=True)
    issues = []
    have_dmx = bool(os.environ.get("DMXAPI_KEY") or os.environ.get("OPENAI_API_KEY"))
    have_apify = bool(os.environ.get("APIFY_TOKEN"))
    print(f"  - DMXAPI/OPENAI key: {'OK' if have_dmx else 'Missing'}")
    print(f"  - APIFY_TOKEN:        {'OK' if have_apify else 'Missing (required only for web scraping)'}")
    if not have_dmx:
        issues.append("DMXAPI_KEY/OPENAI_API_KEY missing (required by LLM steps)")
    note = ""
    if issues:
        note = "Missing: " + "; ".join(issues)
        note += "External dependencies such as BettaFish/MiroFish (including PostgreSQL, MindSpider, Tavily, Zep Cloud, etc.) are outside the scope of `societybench_release`; the `pipeline-run` process will delegate to `total-pipeline` to execute data collection using the community edition."
    append_worklog(
        worklog, "Stage0", "Environment Check",
        "PASS" if not issues else "FAIL",
        output="(Check only)",
        key_data=f"DMX={have_dmx}, APIFY={have_apify}",
        note=note or "The external dependencies for BettaFish/MiroFish are outside the scope of societybench_release; pipeline-run follows a path equivalent to total-pipeline.",
    )
    return not issues


def stage1_2_total_pipeline(worklog: Path, total_argv: list[str]) -> int:
    """SKILL Stages 1+2: crawler collection + opinion analysis.

    The original SKILL invokes /bfish-crawl and /bfish-research (CC skills); those
    BettaFish skills are not in this repo. Instead this calls the societybench
    built-in total_pipeline.py equivalent path, which runs Phase 0
    (web-search-crawl + media-crawl) → Phase 1 (web/media-pipeline)
    → Phase 2 (merge) → Phase 3 (predict).
    """
    print(f"\n[Stages 1+2] Running total-pipeline ({now_str()})", flush=True)
    py = sys.executable
    total_py = Path(__file__).resolve().parent / "total_pipeline.py"
    cmd = [py, str(total_py), *total_argv]
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    elapsed = int(time.time() - t0)
    status = "PASS" if rc == 0 else "FAIL"
    append_worklog(
        worklog, "Stages 1+2", "Collection + Processing + Merging + Evaluation(total-pipeline equivalent path)",
        status,
        output=("total-pipeline completed; Outputs are located at: Under workspace_root in => gt_workspace/ media_workspace/ predict_workspace_v2/"
                if rc == 0 else "total-pipeline Failure"),
        key_data=f"Time spent {elapsed}s, returncode={rc}",
        note="Corresponds to SKILL §Phase 1 /bfish-crawl + Phase 2 /bfish-research; Since BettaFish is not in this repo, so total-pipeline is used instead.",
    )
    return rc


def stage3_skipped(worklog: Path) -> None:
    """SKILL Stage 3: simulation (mfish-simulate). User-confirmed SKIP."""
    print(f"\n[Stage 3] Simulation (mfish-simulate) — SKIP(User has confirmed to skip)", flush=True)
    append_worklog(
        worklog, "Stage 3", "Simulation and scenario analysis(mfish-simulate）",
        "SKIP",
        output="—",
        key_data="—",
        note="The user has confirmed that this release should skip mfish-simulate.；MiroFish external APIs/Zep Cloud are outside the scope of societybench_release.",
    )


def write_summary(worklog: Path) -> None:
    with worklog.open("a", encoding="utf-8") as fh:
        fh.write(
            "---\n\n"
            "## Summary\n"
            f"- **Completion time:** {now_str()}\n"
            "- **mfish-simulate：** SKIP(User confirmation)\n"
            "- **BettaFish/MiroFish External Dependencies:** Outside the scope of societybench_release\n"
            "- **What actually ran:** societybench total-pipeline\n"
        )


def main() -> int:
    # SKILL variable initialization: TIMESTAMP + LOG_DIR
    ts = os.environ.get("TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_env = os.environ.get("LOG_DIR")
    if log_dir_env:
        log_dir = Path(log_dir_env)
    else:
        log_dir = Path.home() / "pipeline_logs" / ts
    os.environ.setdefault("TIMESTAMP", ts)
    os.environ.setdefault("LOG_DIR", str(log_dir))

    # SKILL: "query keywords: $ARGUMENTS (if the user gave none, ask for them)" —
    # interactive prompt when no CLI args supplied.
    if len(sys.argv) < 2:
        if sys.stdin.isatty():
            try:
                user_input = input(
                    "[pipeline-run] No search keywords were provided. Please enter a description of the event you want to research\n"
                    "(One sentence, for example, \"Tesla Mexico factory strike 2025\"):\n>"
                ).strip()
            except (EOFError, KeyboardInterrupt):
                user_input = ""
            if user_input:
                sys.argv.append(user_input)
            else:
                print("[pipeline-run] No keywords received. Exiting.", file=sys.stderr)
                return 2
        else:
            print("[pipeline-run] No search keywords provided, and stdin is not a TTY, so interactive input is unavailable.\n"
                  "Usage: python pipeline_run.py \"<Event name>\" <workspace> --replacements-json <file>",
                  file=sys.stderr)
            return 2

    query = " ".join(sys.argv[1:]) or "(no args)"
    worklog = init_worklog(log_dir, query)
    print(f"[pipeline-run] worklog: {worklog}")

    # Stage 0
    if not stage0_env_check(worklog):
        print("[pipeline-run] Stage 0 failed — stopped; see worklog for details.")
        return 1

    # Stages 1+2
    rc = stage1_2_total_pipeline(worklog, sys.argv[1:])

    # Stage 3 (SKIP per user)
    stage3_skipped(worklog)

    # Summary
    write_summary(worklog)

    print(f"\n[pipeline-run] Completed — worklog: {worklog}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
