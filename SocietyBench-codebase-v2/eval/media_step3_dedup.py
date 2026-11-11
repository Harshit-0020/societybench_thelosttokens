#!/usr/bin/env python3
"""media-step3-dedup (1:1 of skills/media-step3-dedup/SKILL.md).

Groups media_step2.jsonl by date. For each date with ≥3 entries, asks the LLM
to pick which IDs to keep (unique fact or distinct opinion). ≤2 entries pass
through; big days (>30) split into batches of 30. Uses index-number answers
(`保留：1,3,5`).

8 threads (one per date), resume per-date, reasoning_effort="none".
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
以下是 {date} 这一天关于「{event}」的 {n} 条社交媒体摘要。
很多帖子在重复讨论相同的事情，请你：

1. 找出哪些包含独特的新信息或独特的舆论视角（而不是重复其他条已有的内容）
2. 返回应该保留的编号，格式：保留：1,3,5
3. 如果全都重复，选最完整的 2-3 条
4. 只返回"保留：x,x,x"这一行，不要其他内容

{items_text}
"""

BATCH_SIZE = 30
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
                d = row.get("date") or ""
                if d:
                    dates.add(d)
            except json.JSONDecodeError:
                continue
    return dates


def dedup_batch(date: str, event_name: str, batch: List[Dict[str, Any]], model: str) -> Set[str]:
    if len(batch) <= 2:
        return {item["id"] for item in batch}
    items_lines = []
    for i, item in enumerate(batch, start=1):
        body = item.get("summary_clean") or item.get("summary") or ""
        items_lines.append(f"{i}. [ID: {item.get('id','')}]\n{body[:300]}")
    items_text = "\n\n".join(items_lines)
    prompt = PROMPT.format(date=date, event=event_name, n=len(batch), items_text=items_text)
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=200,
            reasoning_effort="none",
        )
    except Exception:
        return {item["id"] for item in batch}
    m = re.search(r"保留[：:]\s*([\d,，\s]+)", raw)
    if not m:
        return {item["id"] for item in batch}
    nums_str = m.group(1).replace("，", ",")
    keep: Set[str] = set()
    for n in nums_str.split(","):
        n = n.strip()
        if n.isdigit():
            idx = int(n) - 1
            if 0 <= idx < len(batch):
                keep.add(batch[idx]["id"])
    return keep or {item["id"] for item in batch}


def process_date(
    date: str, items: List[Dict[str, Any]], event_name: str, model: str,
    out_path: Path, log_path: Path,
    counter: Dict[str, int], total_dates: int,
) -> None:
    keep_ids: Set[str] = set()
    for i in range(0, len(items), BATCH_SIZE):
        keep_ids |= dedup_batch(date, event_name, items[i:i + BATCH_SIZE], model)
    kept = [it for it in items if it["id"] in keep_ids]
    with WRITE_LOCK:
        with out_path.open("a", encoding="utf-8") as fh:
            for item in kept:
                row = {
                    "id": item["id"],
                    "platform": item.get("platform"),
                    "stage": item.get("stage"),
                    "date": item.get("date"),
                    "title": item.get("title", ""),
                    "liked_count": item.get("liked_count", 0),
                    "total_comments": item.get("total_comments", 0),
                    "url": item.get("url", ""),
                    "summary_clean": item.get("summary_clean") or item.get("summary", ""),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with DONE_LOCK:
        counter["done"] += 1
        done = counter["done"]
    pct = done / total_dates * 100 if total_dates else 0
    log(f"[{done}/{total_dates}] ({pct:.1f}%) {date}: {len(items)}条 → {len(kept)}条", log_path)


def main() -> None:
    p = argparse.ArgumentParser(description="media-step3-dedup: per-day dedup of social-media summaries")
    p.add_argument("input_jsonl")
    p.add_argument("output_dir")
    p.add_argument("--event-name", required=True)
    p.add_argument("--threads", type=int, default=8)  # SKILL: THREADS=8
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    rows = read_jsonl(Path(args.input_jsonl))
    items_all: List[Dict[str, Any]] = []
    for r in rows:
        summary = (r.get("summary_clean") or r.get("summary") or "").strip()
        if summary in ("不相关", "清洗失败", ""):
            continue
        items_all.append(r)

    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items_all:
        by_date[(it.get("date") or "unknown")].append(it)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "media_step3.jsonl"
    log_path = out_dir / "media_step3.log"

    done_dates = load_existing_dates(out_path)
    todo_dates = {d: arts for d, arts in by_date.items() if d not in done_dates}
    counter = {"done": len(done_dates)}
    total_dates = len(by_date)

    log(f"Step 3: {len(items_all)} rows / {total_dates} dates / done={len(done_dates)} todo={len(todo_dates)} threads={args.threads}", log_path)

    if todo_dates:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = {pool.submit(process_date, d, arts, args.event_name, model,
                                 out_path, log_path, counter, total_dates): d
                    for d, arts in sorted(todo_dates.items())}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

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
    print(json.dumps({"input_rows": len(rows), "after_filter": len(items_all),
                      "dates": total_dates, "kept": n_kept,
                      "output_file": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
