#!/usr/bin/env python3
"""total-pipeline (1:1 of skills/total-pipeline/SKILL.md).

End-to-end runner for one event:
  Phase 0 — data collection: web-search-crawl + media-crawl (in parallel)
  Phase 1 — per-source processing: web-pipeline + media-pipeline (in parallel)
  Phase 2 — merge-pipeline (includes Step 6.5 GT refine)
  Phase 3 — predict-pipeline (uses timeline_merged_long_gt.md by default)

Use --start-phase to skip earlier phases when intermediates already exist.
Phase 0 crawl tools (Apify, MediaCrawlerPro) must be installed and configured
via .env (APIFY_TOKEN, MCP_HOME, MCP_SIGNSRV_URL).
"""
from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

D = Path(__file__).resolve().parent

# SKILL section "10-min project self-check": warn if a single phase produces no new output for
# this many seconds. Override with --selfcheck-seconds.
DEFAULT_SELFCHECK_SECONDS = 600


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_worklog(worklog: Optional[Path], phase: str, status: str,
                    note: str = "", elapsed_s: Optional[float] = None) -> None:
    """SKILL section "run feedback write-back": append a one-block entry per phase to worklog.md."""
    if worklog is None:
        return
    parts = [
        f"\n### [{phase}] {_now()}",
        f"- 状态：{status}",
    ]
    if elapsed_s is not None:
        parts.append(f"- 耗时：{int(elapsed_s)}s")
    if note:
        parts.append(f"- 备注：{note}")
    parts.append("")
    try:
        with worklog.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(parts) + "\n")
    except OSError:
        pass


def run(name: str, cmd: List[str], worklog: Optional[Path] = None,
        selfcheck_seconds: int = DEFAULT_SELFCHECK_SECONDS) -> int:
    """Run a subprocess and tee its stdout/stderr. Implements the SKILL 10-min
    self-check: if more than `selfcheck_seconds` pass without a new output
    line, print a soft warning (does NOT abort the child)."""
    print(f"\n========== [{name}] {' '.join(cmd)} ==========")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, text=True)
    last_output_t = time.time()
    warned = False
    assert proc.stdout is not None
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 5.0)
        if ready:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                last_output_t = time.time()
                warned = False
        else:
            if proc.poll() is not None:
                break
            idle = time.time() - last_output_t
            if idle > selfcheck_seconds and not warned:
                warned = True
                msg = (f"\n⚠️  [{name}] 自检：>{selfcheck_seconds}s 无新输出。"
                       "继续等待 — 这可能是 LLM reasoning / Apify run / 大批量 IO 的正常慢，"
                       "也可能是单 batch 卡死。父进程不会强制中断；如需停止请 Ctrl+C。\n")
                print(msg, flush=True)
                _append_worklog(worklog, name + "/self-check",
                               "STALL", f"no new output for {int(idle)}s")
    rc = proc.wait()
    elapsed = time.time() - t0
    _append_worklog(worklog, name, "PASS" if rc == 0 else f"FAIL rc={rc}", elapsed_s=elapsed)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="total-pipeline: end-to-end SocietyBench build for one event")
    p.add_argument("topic", help="research topic, one-line description")
    p.add_argument("workspace_root", help="root directory for all intermediates + outputs")
    # SKILL argument-hint: --replacements-json <replacements.json>
    p.add_argument("--replacements-json", dest="replacements_json", required=True,
                   help="anonymization replacement table JSON (SKILL argument-hint name)")
    p.add_argument("--event-name", default=None, help="defaults to <topic>")
    p.add_argument("--web-crawl-config", default=None,
                   help="JSON config for web-search-crawl (keywords + segments)")
    p.add_argument("--media-keywords-config", default=None,
                   help="JSON config for media-crawl keywords")
    p.add_argument("--media-platforms", default="dy,bili,wb,zhihu,tieba,xhs,ks")
    p.add_argument("--media-validator", default="v2", choices=["v1", "v2"],
                   help="media validity validator: v2 = kimi per-post semantic (default/standard) | v1 = LLM layered keyword OR (baseline)")
    p.add_argument("--no-media-validate", action="store_true",
                   help="Skip media validity validation (assumes raw_media/valid_data_llm_verified.csv already exists)")
    p.add_argument("--stages", default=None, help="optional stages JSON for media/web step5")
    p.add_argument("--eval-models", default=None)
    p.add_argument("--points", default="P02,P15,P29",
                   help="comma-separated point ids or 'all'. Default: P02,P15,P29 per SKILL.")
    p.add_argument("--model", default=None, help="LLM used for processing / question-bank generation")
    # SKILL usage section: default P02,P15,P29 (representative early/middle/late points)
    p.add_argument("--start-phase", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--skip-step0-audit", action="store_true",
                   help="Skip Phase 3 step0 LLM audit + consistency check")
    p.add_argument("--selfcheck-seconds", type=int, default=DEFAULT_SELFCHECK_SECONDS,
                   help="SKILL 10-min project self-check: warn when a phase has no new output for "
                        "this many seconds (default 600 = 10 min). Set 0 to disable.")
    p.add_argument("--worklog", default=None,
                   help="Optional worklog.md path; total-pipeline appends a [phase] block "
                        "to it after each phase finishes (SKILL run-feedback write-back).")
    p.add_argument("--pipeline-config", default=None)
    a = p.parse_args()

    py = sys.executable
    root = Path(a.workspace_root); root.mkdir(parents=True, exist_ok=True)
    event_name = a.event_name or a.topic

    # Resolve worklog: explicit > $LOG_DIR/worklog.md > workspace_root/worklog.md
    if a.worklog:
        worklog = Path(a.worklog)
    elif os.environ.get("LOG_DIR"):
        worklog = Path(os.environ["LOG_DIR"]) / "worklog.md"
    else:
        worklog = root / "worklog.md"
    worklog.parent.mkdir(parents=True, exist_ok=True)
    if not worklog.exists():
        worklog.write_text(
            f"# Total-Pipeline Worklog\n- topic: {a.topic}\n- event_name: {event_name}\n"
            f"- workspace: {root}\n- started: {_now()}\n\n---\n",
            encoding="utf-8",
        )

    base_event = ["--event-name", event_name]
    base_optional = []
    if a.model:
        base_optional += ["--model", a.model]
    if a.pipeline_config:
        base_optional += ["--pipeline-config", a.pipeline_config]

    # SKILL directory layout: gt_workspace / media_workspace / predict_workspace_v2.
    # Merge products live inside media_workspace/ (SKILL line: "merge output goes into media_workspace/ by convention").
    web_raw = root / "raw_web"
    media_raw = root / "raw_media"
    web_ws = root / "gt_workspace"
    media_ws = root / "media_workspace"
    merge_ws = media_ws  # merge products go into media_workspace per SKILL
    predict_ws = root / "predict_workspace_v2"
    # SKILL section "parallel isolation (mandatory)": per-event runtime copies of external crawl tools so
    # multiple events running concurrently never share writable working dirs.
    web_runtime = root / "_tool_web_runtime"
    media_runtime = root / "_tool_media_runtime"
    for d in (web_raw, media_raw, web_ws, media_ws, predict_ws):
        d.mkdir(parents=True, exist_ok=True)

    def _clone_runtime(tool_home_env: str, dest: Path) -> Optional[Path]:
        """rsync the external tool's directory into per-event runtime, excluding
        outputs/data/__pycache__ so each event has a clean writable workspace.
        Returns the dest path on success, or None if env var not set / source
        missing (in which case caller falls back to invoking the tool directly).
        """
        src = os.environ.get(tool_home_env, "").strip()
        if not src:
            return None
        src_path = Path(src)
        if not src_path.exists():
            print(f"[runtime] {tool_home_env}={src} does not exist — falling back to direct invocation")
            return None
        if dest.exists() and any(dest.iterdir()):
            print(f"[runtime] {dest} already populated — reusing")
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        # Use rsync if available; fall back to shutil.copytree
        try:
            rc = subprocess.run(
                ["rsync", "-a", "--delete",
                 "--exclude=outputs/", "--exclude=data/", "--exclude=__pycache__/",
                 "--exclude=.git/", "--exclude=node_modules/",
                 f"{src_path}/", f"{dest}/"],
                check=False,
            ).returncode
            if rc != 0:
                raise RuntimeError(f"rsync rc={rc}")
        except (FileNotFoundError, RuntimeError):
            import shutil
            shutil.copytree(src_path, dest, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("outputs", "data", "__pycache__",
                                                          ".git", "node_modules"))
        print(f"[runtime] cloned {src_path} → {dest}")
        return dest

    # ---- Phase 0: data collection ----
    if a.start_phase <= 0:
        # SKILL parallel isolation: clone external crawl tool dirs to per-event runtime
        # BEFORE launching parallel threads. Sets MCP_HOME / APIFY_TOOL_HOME
        # (consumed by the wrapper py scripts) to the per-event copy if cloned.
        web_runtime_path = _clone_runtime("APIFY_TOOL_HOME", web_runtime)
        media_runtime_path = _clone_runtime("MCP_HOME", media_runtime)
        # Pass these to child processes via environment
        child_env = os.environ.copy()
        if web_runtime_path:
            child_env["APIFY_TOOL_HOME"] = str(web_runtime_path)
        if media_runtime_path:
            child_env["MCP_HOME"] = str(media_runtime_path)
        # Apply override (Popen inherits parent env by default; we update parent
        # process env so subsequent subprocess.Popen calls inside run() see it).
        if web_runtime_path:
            os.environ["APIFY_TOOL_HOME"] = str(web_runtime_path)
        if media_runtime_path:
            os.environ["MCP_HOME"] = str(media_runtime_path)
        _append_worklog(worklog, "Phase 0/runtime-clone",
                        "PASS",
                        note=f"web_runtime={'cloned' if web_runtime_path else 'skipped'}, "
                             f"media_runtime={'cloned' if media_runtime_path else 'skipped'}")
        results: Dict[str, int] = {}

        def web_crawl() -> None:
            # SKILL web-search-crawl is now a full 8-step orchestrator
            # (see eval/crawl/web_search_crawl_pipeline.py). Skip if no topic
            # provided beyond what's already in raw_web/.
            if not a.web_crawl_config and not a.topic:
                print("[Phase 0/web] no --web-crawl-config or topic — skipping web search; assuming articles already in raw_web/")
                results["web"] = 0
                return
            cmd = [py, str(D / "crawl" / "web_search_crawl_pipeline.py"),
                   "--project", "event",
                   "--topic", a.topic,
                   "--output-dir", str(web_raw)]
            if a.web_crawl_config:
                cmd += ["--keywords-config", a.web_crawl_config]
            results["web"] = run("crawl-web", cmd, worklog=worklog, selfcheck_seconds=a.selfcheck_seconds)

        def media_crawl() -> None:
            cmd = [py, str(D / "crawl" / "media_crawl.py"),
                   "--platforms", a.media_platforms]
            if a.media_keywords_config:
                cmd += ["--keywords-config", a.media_keywords_config]
            else:
                cmd += ["--topic", a.topic]
            results["media"] = run("crawl-media", cmd, worklog=worklog, selfcheck_seconds=a.selfcheck_seconds)
            # ---- media Step 4: validity validation (default LLM = export_llm_validated_v2.py) ----
            # Wires "raw social-media DB → verified valid_data_llm_verified.csv" into the main pipeline.
            # The main pipeline previously lacked this step (Phase 1 read valid_csv directly) and relied on ad-hoc CC rule scripts → now unified as the default LLM validator.
            if not a.no_media_validate:
                out_csv = media_raw / "valid_data_llm_verified.csv"
                db = None
                for cand in [media_raw / "media_crawler.db",
                             (media_runtime_path / "media_crawler.db") if media_runtime_path else None,
                             (media_runtime_path / "data" / "media_crawler.db") if media_runtime_path else None]:
                    if cand and cand.exists() and cand.stat().st_size > 0:
                        db = cand; break
                if db is None and media_runtime_path:
                    hits = list(Path(media_runtime_path).rglob("media_crawler.db"))
                    db = hits[0] if hits else None
                if db is None:
                    print("[Phase 0/media] 未找到 media_crawler.db — 跳过验证（假定 valid_csv 已存在）")
                else:
                    validator = ("export_llm_validated_v2.py" if a.media_validator == "v2"
                                 else "export_llm_validated_v1.py")
                    vcmd = [py, str(D / "crawl" / "media_scripts" / validator),
                            "--db", str(db), "--topic", a.topic, "--output", str(out_csv)]
                    results["media-validate"] = run(f"media-validate({a.media_validator})", vcmd,
                                                    worklog=worklog, selfcheck_seconds=a.selfcheck_seconds)

        threads = [threading.Thread(target=web_crawl), threading.Thread(target=media_crawl)]
        for t in threads: t.start()
        for t in threads: t.join()
        if any(v != 0 for v in results.values()):
            print("⚠ Phase 0 returned non-zero — continuing only if intermediates exist")

    # ---- Phase 1: per-source processing ----
    if a.start_phase <= 1:
        results = {}

        def web_pipe() -> None:
            articles = web_raw / "articles.json"
            if not articles.exists():
                # search-crawl outputs <project>_full.json; user can rename or skip
                candidates = sorted(web_raw.glob("*_full.json"))
                if candidates:
                    articles = candidates[-1]
                    print(f"[Phase 1/web] using {articles}")
            if not articles.exists():
                print(f"[Phase 1/web] no articles JSON found in {web_raw} — skipping")
                results["web"] = 0
                return
            cmd = [py, str(D / "web_pipeline.py"), str(articles), str(web_ws),
                   *base_event, *base_optional]
            if a.stages:
                cmd += ["--stages", a.stages]
            results["web"] = run("web-pipeline", cmd, worklog=worklog, selfcheck_seconds=a.selfcheck_seconds)

        def media_pipe() -> None:
            input_csv = media_raw / "valid_data_llm_verified.csv"
            if not input_csv.exists():
                candidates = sorted(media_raw.glob("valid_data*.csv"))
                if candidates:
                    input_csv = candidates[-1]
            if not input_csv.exists():
                print(f"[Phase 1/media] no valid_data CSV found in {media_raw} — skipping")
                results["media"] = 0
                return
            cmd = [py, str(D / "media_pipeline.py"), str(input_csv), str(media_ws),
                   *base_event, *base_optional]
            if a.stages:
                cmd += ["--stages", a.stages]
            results["media"] = run("media-pipeline", cmd, worklog=worklog, selfcheck_seconds=a.selfcheck_seconds)

        threads = [threading.Thread(target=web_pipe), threading.Thread(target=media_pipe)]
        for t in threads: t.start()
        for t in threads: t.join()
        if any(v != 0 for v in results.values()):
            sys.exit(1)

    # ---- Phase 2: merge ----
    # SKILL: 6 positional args (web_short, media_short, web_step1, media_step1, media_step3, output_dir)
    if a.start_phase <= 2:
        cmd = [py, str(D / "merge_pipeline.py"),
               str(web_ws / "timeline_short.md"),
               str(media_ws / "media_timeline_short.md"),
               str(web_ws / "timeline_step1.jsonl"),
               str(media_ws / "media_step1.jsonl"),
               str(media_ws / "media_step3.jsonl"),
               str(merge_ws),
               *base_event, *base_optional]
        if run("merge-pipeline", cmd, worklog=worklog, selfcheck_seconds=a.selfcheck_seconds) != 0:
            sys.exit(1)

    # ---- Phase 3: predict ----
    if a.start_phase <= 3:
        refined = merge_ws / "timeline_merged_long_gt.md"
        original = merge_ws / "timeline_merged_long.md"
        predict_input = refined if refined.exists() else original
        if not predict_input.exists():
            sys.exit(f"Neither {refined} nor {original} exists — merge-pipeline must have failed")
        print(f"[Phase 3] predict input = {predict_input}")
        cmd = [py, str(D / "predict_pipeline.py"),
               str(predict_input), str(predict_ws),
               "--event-name", event_name,
               "--replacements-json", a.replacements_json,
               "--points", a.points]
        if a.eval_models:
            cmd += ["--eval-models", a.eval_models]
        if a.skip_step0_audit:
            cmd += ["--skip-step0-audit"]
        if a.model:
            cmd += ["--model", a.model]
        if a.pipeline_config:
            cmd += ["--pipeline-config", a.pipeline_config]
        if run("predict-pipeline", cmd, worklog=worklog, selfcheck_seconds=a.selfcheck_seconds) != 0:
            sys.exit(1)

    print(f"\n✅ total-pipeline complete: results in {predict_ws}/results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
