#!/usr/bin/env python3
"""SocietyBench experiment driver.

This is an execution wrapper around the locked runbook in
`runs_new/实验方案.md`.  It intentionally keeps policy decisions here and
leaves scoring logic in the existing step scripts.

Default behavior is safe: without `--yes`, the driver only prints the plan.
Use `--dry-run` for an explicit non-executing plan and `--preflight-only` for
the small API smoke run before a formal run.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
ROOT = CODE_DIR.parent
RUNS_NEW = ROOT / "runs_new"
FINAL_RESULT = ROOT / "final_result"
AGENTS_PY = Path.home() / "societybench_agents_venv" / "bin" / "python"

CHINESE_MODELS = [
    "qwen3.5-plus-2026-02-15",
    "kimi-k2.5",
    "doubao-seed-2-0-pro-260215",
]
DOUBAO = "doubao-seed-2-0-pro-260215"
QWEN = "qwen3.5-plus-2026-02-15"
M3_BASES = DOUBAO
A4_VARIANTS = ["noentity", "nodate", "plain"]

EVENTS = [
    {"dir": "event1_library", "slug": "library", "aliases": {"event1", "event1_library", "library"}},
    {"dir": "event2_trump_tariff", "slug": "trump_tariff", "aliases": {"event2", "event2_trump_tariff", "trump_tariff"}},
    {"dir": "event3_tiktok", "slug": "tiktok", "aliases": {"event3", "event3_tiktok", "tiktok"}},
    {"dir": "event4_us_iran", "slug": "us_iran", "aliases": {"event4", "event4_us_iran", "us_iran"}},
    {"dir": "event5_smci", "slug": "smci", "aliases": {"event5", "event5_smci", "smci"}},
]


def model_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def display_cmd(cmd: Iterable[object]) -> str:
    parts = []
    for item in cmd:
        s = str(item)
        if re.search(r"\s", s):
            parts.append(repr(s))
        else:
            parts.append(s)
    return " ".join(parts)


def workspace(event: dict[str, object], lang: str) -> Path:
    return RUNS_NEW / str(event["dir"]) / "final" / lang


def run_dir(ws: Path, run_id: str) -> Path:
    return ws / "results" / (run_id if run_id.startswith("run_") else f"run_{run_id}")


def brier_agg_exists(ws: Path, run_id: str, models: Iterable[str]) -> bool:
    root = run_dir(ws, run_id) / "brier"
    return all((root / model_slug(m) / "aggregated.json").exists() for m in models)


def time_agg_exists(ws: Path, run_id: str, models: Iterable[str]) -> bool:
    root = run_dir(ws, run_id) / "time"
    return all((root / model_slug(m) / "aggregated.json").exists() for m in models)


def scorecard_exists(ws: Path, run_id: str) -> bool:
    return (run_dir(ws, run_id) / "scorecard.json").exists()


def resolve_events(raw: str) -> list[dict[str, object]]:
    if raw == "all":
        return EVENTS
    selected: list[dict[str, object]] = []
    for token in split_csv(raw):
        found = None
        for event in EVENTS:
            if token in event["aliases"]:  # type: ignore[operator]
                found = event
                break
        if not found:
            raise SystemExit(f"Unknown event: {token}")
        selected.append(found)
    return selected


def resolve_scope(raw: str, langs: list[str]) -> list[str]:
    if raw == "full":
        scope: list[str] = []
        if "中文" in langs:
            scope.extend(["M1", "M4", "A1", "A2", "M3", "A4"])
        if "英文" in langs:
            scope.extend(["M2", "M4", "A1", "A2", "M3"])
        return list(dict.fromkeys(scope))
    return [x.upper() for x in split_csv(raw)]


@dataclass
class Task:
    name: str
    cmd: list[object]
    api: bool = False
    complete: Callable[[], bool] | None = None
    cwd: Path = CODE_DIR
    env: dict[str, str] | None = None

    def is_complete(self) -> bool:
        return bool(self.complete and self.complete())


def make_scorecard_task(ws: Path, run_id_value: str, name: str) -> Task:
    return Task(
        name=name,
        cmd=[sys.executable, EVAL_DIR / "predict_step4_scorecard.py", "--workspace", ws, "--run-id", run_id_value],
        api=False,
        complete=lambda ws=ws, rid=run_id_value: scorecard_exists(ws, rid),
    )


def gate_tasks(events: list[dict[str, object]]) -> list[Task]:
    tasks = [
        Task(
            name="gate: date consistency",
            cmd=[sys.executable, EVAL_DIR / "date_consistency_gate.py", "--root", RUNS_NEW, "--show", "10"],
            api=False,
        )
    ]
    for event in events:
        tasks.append(
            Task(
                name=f"gate: finalize scan {event['dir']}",
                cmd=[sys.executable, EVAL_DIR / "finalize_event.py", "--event-dir", RUNS_NEW / str(event["dir"]), "--scan-only"],
                api=False,
            )
        )
    return tasks


def build_m1_tasks(
    events: list[dict[str, object]],
    lang: str,
    models: list[str],
    points: str,
    max_parallel: int,
    brier_batch_size: int,
) -> list[Task]:
    run_id_value = "M1" if lang == "中文" else "M2"
    tasks: list[Task] = []
    for event in events:
        ws = workspace(event, lang)
        ev = str(event["slug"])
        tasks.append(Task(
            name=f"{run_id_value}: 3B {event['dir']} {lang}",
            cmd=[sys.executable, EVAL_DIR / "predict_step3B_brier.py", "--workspace", ws, "--event-name", ev,
                 "--models", ",".join(models), "--points", points, "--run-id", run_id_value,
                 "--max-parallel", str(max_parallel), "--max-tokens", "32000", "--thinking-budget", "24000",
                 "--batch-size", str(brier_batch_size)],
            api=True,
            complete=lambda ws=ws, rid=run_id_value, ms=models: brier_agg_exists(ws, rid, ms),
        ))
        tasks.append(Task(
            name=f"{run_id_value}: 3F {event['dir']} {lang}",
            cmd=[sys.executable, EVAL_DIR / "predict_step3F_time.py", "--workspace", ws, "--event-name", ev,
                 "--models", ",".join(models), "--points", points, "--run-id", run_id_value,
                 "--runs-per-point", "1", "--max-parallel", str(max_parallel), "--max-tokens", "32000",
                 "--thinking-budget", "24000"],
            api=True,
            complete=lambda ws=ws, rid=run_id_value, ms=models: time_agg_exists(ws, rid, ms),
        ))
        tasks.append(make_scorecard_task(ws, run_id_value, f"{run_id_value}: scorecard {event['dir']} {lang}"))
    return tasks


def build_m4_tasks(events: list[dict[str, object]], lang: str, points: str) -> list[Task]:
    tasks: list[Task] = []
    for event in events:
        ws = workspace(event, lang)
        ev = str(event["slug"])
        tasks.append(Task(
            name=f"M4: baseline {event['dir']} {lang}",
            cmd=[sys.executable, EVAL_DIR / "baseline" / "baseline_freq_momentum.py", "--workspace", ws,
                 "--event-name", ev, "--points", points, "--run-id", "M4"],
            api=False,
            complete=lambda ws=ws: brier_agg_exists(ws, "M4", ["frequency", "momentum"]),
        ))
        tasks.append(make_scorecard_task(ws, "M4", f"M4: scorecard {event['dir']} {lang}"))
    return tasks


def brier_dirs(events: list[dict[str, object]], lang: str, run_id_value: str, model: str) -> str:
    return ",".join(
        str(workspace(event, lang) / "results" / f"run_{run_id_value}" / "brier" / model_slug(model))
        for event in events
    )


def build_a1_tasks(events: list[dict[str, object]], lang: str, models: list[str]) -> list[Task]:
    source_run = "M1" if lang == "中文" else "M2"
    tasks: list[Task] = []
    for model in models:
        out = RUNS_NEW / "_ablation" / lang / f"run_{source_run}" / "A1_qtype" / model_slug(model)
        tasks.append(Task(
            name=f"A1: qtype {lang} {model}",
            cmd=[sys.executable, EVAL_DIR / "ablation" / "ablation_qtype.py", "--brier-dirs",
                 brier_dirs(events, lang, source_run, model), "--out", out],
            api=False,
            complete=lambda out=out: out.with_suffix(".json").exists() and out.with_suffix(".md").exists(),
        ))
    return tasks


def build_a2_tasks(events: list[dict[str, object]], lang: str, models: list[str]) -> list[Task]:
    if not ({DOUBAO, QWEN} <= set(models)):
        print("[warn] A2 skipped: requires Doubao-pro and Qwen3.5 in --models")
        return []
    source_run = "M1" if lang == "中文" else "M2"
    out = RUNS_NEW / "_ablation" / lang / f"run_{source_run}" / "A2_scoring" / f"{model_slug(DOUBAO)}__vs__{model_slug(QWEN)}"
    return [Task(
        name=f"A2: scoring {lang} Doubao vs Qwen",
        cmd=[sys.executable, EVAL_DIR / "ablation" / "ablation_scoring.py",
             "--model1-dirs", brier_dirs(events, lang, source_run, DOUBAO), "--model1-name", "doubao-pro",
             "--model2-dirs", brier_dirs(events, lang, source_run, QWEN), "--model2-name", "qwen3.5-plus",
             "--out", out],
        api=False,
        complete=lambda out=out: out.with_suffix(".json").exists() and out.with_suffix(".md").exists(),
    )]


def build_m3_tasks(
    events: list[dict[str, object]],
    lang: str,
    points: str,
    preflight: bool = False,
    brier_batch_size: int = 999999,
) -> list[Task]:
    tasks: list[Task] = []
    py = AGENTS_PY if AGENTS_PY.exists() else Path(sys.executable)
    for event in events:
        ws = workspace(event, lang)
        rid = "preflight_M3" if preflight else "M3"
        # Global optimization 2026-07-01: M3 runs only the three agents (no --with-bare). The bare-Doubao baseline comes from main experiment M1, not re-run inside M3.
        cmd: list[object] = [py, EVAL_DIR / "agents" / "run_full_m3.py", "--event", event["slug"], "--lang", lang,
                             "--bases", M3_BASES, "--run-id", rid, "--points", points,
                             "--batch-size", str(brier_batch_size)]
        if preflight:
            cmd += ["--axes", "brier", "--mirofish-agents", "4", "--mirofish-steps", "2"]
        tasks.append(Task(
            name=f"M3: agents {event['dir']} {lang}",
            cmd=cmd,
            api=True,
            complete=lambda ws=ws, rid=rid: scorecard_exists(ws, rid) if not preflight else False,
        ))
        if not preflight:
            tasks.append(make_scorecard_task(ws, rid, f"M3: scorecard {event['dir']} {lang}"))
    return tasks


def build_a4_tasks(
    events: list[dict[str, object]],
    points: str,
    max_parallel: int,
    preflight: bool = False,
    brier_batch_size: int = 999999,
) -> list[Task]:
    tasks: list[Task] = []
    for event in events:
        ev = str(event["slug"])
        for variant in A4_VARIANTS:
            ws = RUNS_NEW / str(event["dir"]) / "final" / "匿名化消融实验" / variant
            rid = f"preflight_A4_{variant}" if preflight else f"A4_{variant}"
            tasks.append(Task(
                name=f"A4: {variant} {event['dir']}",
                cmd=[sys.executable, EVAL_DIR / "predict_step3B_brier.py", "--workspace", ws, "--event-name", ev,
                     "--models", DOUBAO, "--points", points, "--run-id", rid, "--max-parallel", str(max_parallel),
                     "--max-tokens", "32000", "--thinking-budget", "24000",
                     "--batch-size", str(brier_batch_size)],
                api=True,
                complete=lambda ws=ws, rid=rid: brier_agg_exists(ws, rid, [DOUBAO]),
            ))
            tasks.append(make_scorecard_task(ws, rid, f"A4: scorecard {variant} {event['dir']}"))
    return tasks


def build_preflight_tasks(
    events: list[dict[str, object]],
    scopes: list[str],
    langs: list[str],
    models: list[str],
    points: str,
    max_parallel: int,
    brier_batch_size: int,
) -> list[Task]:
    tasks = gate_tasks(events)
    for lang in langs:
        if (lang == "中文" and "M1" in scopes) or (lang == "英文" and "M2" in scopes):
            run_id_value = "preflight_M1" if lang == "中文" else "preflight_M2"
            for event in events:
                ws = workspace(event, lang)
                ev = str(event["slug"])
                tasks.append(Task(
                    name=f"preflight {run_id_value}: 3B {event['dir']} {lang}",
                    cmd=[sys.executable, EVAL_DIR / "predict_step3B_brier.py", "--workspace", ws, "--event-name", ev,
                         "--models", ",".join(models), "--points", points, "--run-id", run_id_value,
                         "--max-parallel", str(max_parallel), "--max-tokens", "32000", "--thinking-budget", "24000",
                         "--batch-size", str(brier_batch_size)],
                    api=True,
                ))
                tasks.append(Task(
                    name=f"preflight {run_id_value}: 3F {event['dir']} {lang}",
                    cmd=[sys.executable, EVAL_DIR / "predict_step3F_time.py", "--workspace", ws, "--event-name", ev,
                         "--models", ",".join(models), "--points", points, "--run-id", run_id_value,
                         "--runs-per-point", "1", "--max-parallel", str(max_parallel), "--max-tokens", "32000",
                         "--thinking-budget", "24000"],
                    api=True,
                ))
                tasks.append(make_scorecard_task(ws, run_id_value, f"preflight {run_id_value}: scorecard {event['dir']} {lang}"))
        if "M3" in scopes:
            tasks.extend(build_m3_tasks(events, lang, points, preflight=True, brier_batch_size=brier_batch_size))
    if "A4" in scopes and "中文" in langs:
        tasks.extend(build_a4_tasks(events, points, max_parallel=1, preflight=True, brier_batch_size=brier_batch_size))
    return tasks


def build_formal_tasks(events: list[dict[str, object]], scopes: list[str], langs: list[str], models: list[str], points: str, args: argparse.Namespace) -> list[Task]:
    tasks = gate_tasks(events)
    for lang in langs:
        if lang == "中文" and "M1" in scopes:
            tasks.extend(build_m1_tasks(events, lang, models, points, args.m1_max_parallel, args.brier_batch_size))
        if lang == "英文" and "M2" in scopes:
            tasks.extend(build_m1_tasks(events, lang, models, points, args.m1_max_parallel, args.brier_batch_size))
        if "M4" in scopes:
            tasks.extend(build_m4_tasks(events, lang, points))
        if "A1" in scopes:
            tasks.extend(build_a1_tasks(events, lang, models))
        if "A2" in scopes:
            tasks.extend(build_a2_tasks(events, lang, models))
        if "M3" in scopes:
            tasks.extend(build_m3_tasks(events, lang, points, brier_batch_size=args.brier_batch_size))
    if "A4" in scopes:
        if "中文" not in langs:
            print("[warn] A4 skipped: A4 variants exist only for 中文 workspaces")
        else:
            tasks.extend(build_a4_tasks(events, points, args.a4_max_parallel, brier_batch_size=args.brier_batch_size))
    return tasks


def print_plan(tasks: list[Task], resume: bool) -> None:
    api_count = sum(1 for t in tasks if t.api)
    print(f"tasks={len(tasks)} api_tasks={api_count} resume={resume}")
    for i, task in enumerate(tasks, start=1):
        state = "SKIP(done)" if resume and task.is_complete() else "RUN"
        marker = "API" if task.api else "local"
        print(f"\n[{i:03d}] {state} {marker} {task.name}")
        print("  " + display_cmd(task.cmd))


def _run_one_task(task: "Task", idx: int, resume: bool) -> None:
    if resume and task.is_complete():
        print(f"\n[{idx:03d}] skip complete: {task.name}", flush=True)
        return
    print(f"\n[{idx:03d}] run: {task.name}", flush=True)
    print("  " + display_cmd(task.cmd), flush=True)
    env = os.environ.copy()
    if task.env:
        env.update(task.env)
    proc = subprocess.run([str(x) for x in task.cmd], cwd=task.cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Task failed (exit {proc.returncode}): {task.name}")


def execute(tasks: list[Task], resume: bool, jobs: int = 1) -> None:
    # Task order = [gates..., per event×lang (3B, 3F, scorecard)...]. Gates run serially first;
    # API tasks (3B/3F) are mutually independent → run `jobs` subprocesses concurrently
    # (each subprocess runs max_parallel concurrent points → total concurrency ≈ jobs×max_parallel,
    # saturating multiple keys); scorecards run serially after all API tasks (read-only, fast).
    # Job-level parallelism added 2026-06-30.
    first_api = next((i for i, t in enumerate(tasks) if t.api), len(tasks))
    leading = list(enumerate(tasks[:first_api], start=1))
    rest = list(enumerate(tasks[first_api:], start=first_api + 1))
    rest_api = [(i, t) for i, t in rest if t.api]
    rest_local = [(i, t) for i, t in rest if not t.api]
    for i, t in leading:
        _run_one_task(t, i, resume)
    if jobs > 1 and len(rest_api) > 1:
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_run_one_task, t, i, resume): t.name for i, t in rest_api}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
        if errors:
            raise SystemExit("并行 API 任务有失败:\n  " + "\n  ".join(errors))
    else:
        for i, t in rest_api:
            _run_one_task(t, i, resume)
    for i, t in rest_local:
        _run_one_task(t, i, resume)


def main() -> None:
    ap = argparse.ArgumentParser(description="SocietyBench locked-run experiment driver")
    ap.add_argument("--models", default=",".join(CHINESE_MODELS), help="comma-separated model ids for M1/M2")
    ap.add_argument("--scope", default="M1,M4", help="comma-separated scope list or full")
    ap.add_argument("--langs", default="中文", help="中文, 英文, or comma-separated")
    ap.add_argument("--events", default="all", help="all or comma-separated event aliases")
    ap.add_argument("--points", default="all", help="all or comma-separated point ids")
    ap.add_argument("--preflight-points", default="P01,P07")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually execute the plan")
    # 2026-06-30 measured per-key concurrency ceiling: doubao/kimi at N=80 still 0 errors, no latency rise (ceiling not reached).
    # Take 0.8×ceiling = 0.8×80 = 64/key; 3 keys fully loaded = 64×3 = 192 total concurrency (common.py
    # round-robins keys per call → 192 spreads automatically over 3 keys ≈ 64/key, still within the measured safe zone).
    # Both 3F and 3B now parallelize BY POINT (max_parallel effective; points are independent, one request each;
    # 3B was also switched to parallel points on 2026-06-30). run_all remains sequential at the task level (one
    # event×lang at a time), so effective per-event concurrency ≈ that event's task count (25 points × model count,
    # ~50); 192 is the cap and converges automatically with task count. To push 3 keys to ~64/key full load,
    # run_all could parallelize across events (separate discussion).
    ap.add_argument("--m1-max-parallel", type=int, default=300)   # 2026-06-30: single key + 300 threads (with --jobs 1, total concurrency <= 300)
    ap.add_argument("--a4-max-parallel", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of API tasks (per event×lang 3B/3F) to run concurrently → total concurrency ≈ jobs×max_parallel, saturating multiple keys")
    ap.add_argument("--brier-batch-size", type=int, default=999999,
                    help="Brier batch size for newly launched Brier tasks; 999999 records as batch_size=null")
    args = ap.parse_args()

    langs = split_csv(args.langs)
    for lang in langs:
        if lang not in {"中文", "英文"}:
            raise SystemExit(f"Unknown lang: {lang}")
    events = resolve_events(args.events)
    scopes = resolve_scope(args.scope, langs)
    models = split_csv(args.models)

    if not (RUNS_NEW / "LOCKED_20260630.md").exists():
        raise SystemExit("Missing runs_new/LOCKED_20260630.md; refusing to run unlocked data")
    if "M3" in scopes and not AGENTS_PY.exists():
        print(f"[warn] agents venv python not found: {AGENTS_PY}; will use {sys.executable}")

    if args.preflight_only:
        tasks = build_preflight_tasks(
            events, scopes, langs, models, args.preflight_points,
            args.m1_max_parallel, args.brier_batch_size,
        )
    else:
        tasks = build_formal_tasks(events, scopes, langs, models, args.points, args)

    print_plan(tasks, args.resume)
    if args.dry_run:
        print("\nDRY RUN: no commands executed.")
        return
    if not args.yes:
        print("\nPLAN ONLY: pass --yes to execute. For API smoke, use --preflight-only --yes first.")
        return
    execute(tasks, args.resume, jobs=args.jobs)
    print("\nDONE")


if __name__ == "__main__":
    main()
