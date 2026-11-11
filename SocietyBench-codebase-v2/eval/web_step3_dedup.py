#!/usr/bin/env python3
"""web-step3-dedup (1:1 of skills/web-step3-dedup/SKILL.md).

Group step2.jsonl by date. For each date with ≥2 entries, ask the LLM to pick
which ones have unique new info (filtering same-day repeats of older news).
Dates with ≤1 entry are kept as-is. Irrelevant entries are dropped first.

Per the SKILL: 4 threads, resume per-date, reasoning_effort="none".
The LLM answers in `保留:n1,n2,n3` form (numeric indices into the group).
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
from typing import Any, Dict, List, Set

from common import call_llm, load_pipeline_config, read_jsonl, write_progress


PROMPT = """\
以下是 {date} 这一天关于「{event}」的 {n} 条报道摘要。

很多报道在重复讲述过去已经发生的事。请你：

1. 找出哪些报道包含了**这一天的新进展、新信息、新观点**（而不只是重复旧事）
2. 返回应该保留的编号，格式：保留：1,3,5
3. 如果全都是重复旧事没有新信息，选最完整的 1-2 条，格式：保留：1
4. 只返回"保留：x,x,x"这一行，不要其他内容

{items_text}
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


def load_existing_dates(out_path: Path) -> Set[str]:
    dates: Set[str] = set()
    if not out_path.exists():
        return dates
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                d = (row.get("date") or "")[:10]
                if d:
                    dates.add(d)
            except json.JSONDecodeError:
                continue
    return dates


def dedup_group(date: str, articles: List[Dict[str, Any]],
                event_name: str, model: str) -> List[Dict[str, Any]]:
    if len(articles) <= 1:  # threshold ≥3→≥2: same-day pairs (often near-duplicates) also go through LLM dedup, eliminating "one node, two facts"
        return articles
    items_lines = []
    for i, a in enumerate(articles):
        items_lines.append(f"[{i+1}] (ID: {a.get('id', '')}) {a.get('summary','')[:400]}")
    items_text = "\n".join(items_lines)
    prompt = PROMPT.format(date=date, event=event_name, n=len(articles), items_text=items_text)
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=100,
            reasoning_effort="none",
        )
    except Exception:
        return articles[:2]
    for line in raw.strip().splitlines():
        if "保留" in line:
            tail = re.split(r"[：:]", line, maxsplit=1)
            if len(tail) < 2:
                continue
            nums = [s.strip() for s in tail[1].split(",")]
            indices: List[int] = []
            for n in nums:
                try:
                    idx = int(n) - 1
                    if 0 <= idx < len(articles):
                        indices.append(idx)
                except ValueError:
                    pass
            if indices:
                return [articles[i] for i in indices]
    return articles[:2]


def process_date(
    date: str, articles: List[Dict[str, Any]],
    event_name: str, model: str,
    out_path: Path, log_path: Path,
    counter: Dict[str, int], total_dates: int,
) -> None:
    kept = dedup_group(date, articles, event_name, model)
    with WRITE_LOCK:
        with out_path.open("a", encoding="utf-8") as fh:
            for a in kept:
                fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    with DONE_LOCK:
        counter["done"] += 1
        done = counter["done"]
    pct = done / total_dates * 100 if total_dates else 0
    log(f"[{done}/{total_dates}] ({pct:.1f}%) {date}: {len(articles)}篇 → {len(kept)}篇", log_path)


def main() -> None:
    p = argparse.ArgumentParser(description="web-step3-dedup: per-day deduplication via LLM")
    p.add_argument("input_jsonl")
    p.add_argument("output_dir")
    p.add_argument("--event-name", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    rows = read_jsonl(Path(args.input_jsonl))
    seen_url: Set[str] = set()
    relevant: List[Dict[str, Any]] = []
    for r in rows:
        url = r.get("url")
        if not url or url in seen_url:
            continue
        seen_url.add(url)
        summary = (r.get("summary") or "").strip()
        if summary in ("不相关", "与事件无关", "摘要生成失败", ""):
            continue
        relevant.append(r)

    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in relevant:
        by_date[(r.get("date") or "unknown")[:10]].append(r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timeline_step3.jsonl"
    log_path = out_dir / "timeline_step3.log"

    done_dates = load_existing_dates(out_path)
    todo_dates = {d: arts for d, arts in by_date.items() if d not in done_dates}
    counter = {"done": len(done_dates)}
    total_dates = len(by_date)

    log(f"Step 3: {len(relevant)} rows / {total_dates} dates / done={len(done_dates)} / todo={len(todo_dates)} threads={args.threads}", log_path)

    if todo_dates:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = {
                pool.submit(process_date, d, arts, args.event_name, model,
                            out_path, log_path, counter, total_dates): d
                for d, arts in sorted(todo_dates.items())
            }
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    # Final tally
    n_kept = 0
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for _ in fh:
                n_kept += 1
    # SKILL §progress.json: explicit status snapshot for audit
    try:
        _final_done = (sum(1 for _ in open(out_path, encoding="utf-8"))
                       if "out_path" in dir() and Path(out_path).exists() else 0)
        write_progress(Path(out_path).with_name(Path(out_path).stem + "_progress.json"),
                       done_count=_final_done, total=total_dates, completed_ids=sorted(done_dates) if 'done_dates' in dir() else None)
    except Exception:
        pass
    print(json.dumps({
        "input_rows": len(rows),
        "after_dedup_url": len(relevant),
        "dates": total_dates,
        "kept": n_kept,
        "output_file": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
