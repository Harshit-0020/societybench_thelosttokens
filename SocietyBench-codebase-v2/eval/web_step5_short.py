#!/usr/bin/env python3
"""web-step5-short (1:1 of skills/web-step5-short/SKILL.md).

SKILL §"CC does it itself" = the work is done by an LLM. The SKILL forbids using
kimi specifically (self-contamination concern on the original team), but any
other LLM is fine. The model is configurable via `--model` / pipeline_config.
default_llm_model — open-source users plug in their own.

Pipeline:
  1. Filter step4 rows by `new_info` noise patterns (pure-Python pre-filter)
  2. For each candidate, ask the LLM to re-judge "假新进展" (fake new progress)
     and compress to 1-2 sentences (~50-100 chars) per SKILL §"execution step 2.
     per-item review" + §"3"
  3. Group survivors by Stage (if --stages provided) and date, write
     `timeline_short.md` with `[ID: ...]` markers preserved (SKILL §"4. output format")
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import call_llm, load_pipeline_config, parse_date_obj, read_jsonl


NOISE_PATTERNS = [r"^无新进展$", r"^提取失败", r"^\[.*error", r"^N/A$"]

PROMPT = """\
请二次判断下面这条已被标为"有新进展"的信息，并按需精炼：

- 是不是真的有 {date} 前后新发生的事情（不是回顾、不是评论、不是情绪、不是事件背景）？
- 如果真有，用 1-2 句话精炼写出新进展核心，约 50-100 字
- 如果其实是旧事 / 评论 / 情绪 / 与事件无关，只输出：假新进展

只输出精炼版或"假新进展"，不要任何额外说明。

发布日期：{date}
new_info：{new_info}
"""

WRITE_LOCK = threading.Lock()
DONE_LOCK = threading.Lock()


def log(msg: str, log_path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with WRITE_LOCK:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def is_noise(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if "无新进展" in s or "提取失败" in s:
        return True
    return any(re.search(p, s, flags=re.I) for p in NOISE_PATTERNS)


def load_stages(path: Optional[Path]) -> List[Dict[str, str]]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("stages", []) if isinstance(data, dict) else []


def stage_for(date_str: str, stages: List[Dict[str, str]]) -> str:
    d = parse_date_obj(date_str)
    if d is None or not stages:
        return "未分阶段"
    for s in stages:
        start = parse_date_obj(s.get("start"))
        end = parse_date_obj(s.get("end"))
        if start and end and start <= d <= end:
            return s.get("name") or "未命名"
    return "未分阶段"


def shrink_one(date: str, new_info: str, model: str) -> str:
    try:
        raw = call_llm(
            [{"role": "user", "content": PROMPT.format(date=date, new_info=new_info)}],
            model=model,
            temperature=0.0,
            max_tokens=300,
            reasoning_effort="none",
        )
    except Exception:
        return new_info[:120]
    return raw.strip().lstrip("`").rstrip("`")


def main() -> None:
    p = argparse.ArgumentParser(description="web-step5-short: filter + compress to short timeline")
    p.add_argument("input_jsonl")
    p.add_argument("output_dir")
    p.add_argument("--event-name", default="")
    p.add_argument("--stages", default=None, help="optional stages config JSON")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model", default=None,
                   help="LLM model; defaults to pipeline_config.default_llm_model")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")
    stages = load_stages(Path(args.stages)) if args.stages else []

    rows = read_jsonl(Path(args.input_jsonl))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "timeline_short.log"

    candidates = [r for r in rows if not is_noise(r.get("new_info", ""))]
    counter = {"done": 0}
    total = len(candidates)
    log(f"Step 5: total_rows={len(rows)} candidates={total} threads={args.threads} model={model}", log_path)

    kept: List[Dict[str, Any]] = []
    kept_lock = threading.Lock()

    def process(r: Dict[str, Any]) -> None:
        short = shrink_one(r.get("date", ""), r.get("new_info", ""), model)
        with DONE_LOCK:
            counter["done"] += 1
            done = counter["done"]
        log(f"[{done}/{total}] {r.get('date','')} | {r.get('title','')[:40]}", log_path)
        if "假新进展" in short[:6]:
            return
        with kept_lock:
            kept.append({**r, "short": short})

    if candidates:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = [pool.submit(process, r) for r in candidates]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    kept.sort(key=lambda x: (x.get("date") or ""))
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in kept:
        stage = stage_for(r.get("date", ""), stages) if stages else "全部进展"
        groups[stage][r.get("date", "")].append(r)

    title_prefix = (args.event_name + " — ") if args.event_name else ""
    md_lines = [f"# {title_prefix}Timeline（短版）", ""]
    stage_order = [s.get("name") for s in stages] if stages else list(groups.keys())
    seen = set()
    for stage in list(stage_order) + ["未分阶段"]:
        if stage in seen or stage not in groups:
            continue
        seen.add(stage)
        md_lines.append(f"## {stage}")
        md_lines.append("")
        for date in sorted(groups[stage].keys()):
            md_lines.append(f"### {date}")
            for r in groups[stage][date]:
                md_lines.append(f"- [ID: {r.get('id','')}] {r.get('short','')}")
            md_lines.append("")

    out_md = out_dir / "timeline_short.md"
    out_md.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({
        "input": len(rows), "candidates": total, "kept": len(kept),
        "stages_used": [s.get("name") for s in stages],
        "output_md": str(out_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
