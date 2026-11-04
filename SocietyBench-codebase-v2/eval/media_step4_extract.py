#!/usr/bin/env python3
"""media-step4-extract (1:1 of skills/media-step4-extract/SKILL.md).

For each media_step3.jsonl row, ask the LLM whether the post reports any new
fact (vs opinion/recap/emotion). Tag `new_info` — filtering happens in step5.

8 threads, resume by id, reasoning_effort="none".
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set

from common import call_llm, load_pipeline_config, read_jsonl, write_progress


PROMPT = """\
This social media post was published on {date}. Below is a summary of it.

Please assess: Does this content include any new developments around {date}\
(a new court ruling, official notice, response, revelation, policy, or turning point in public opinion)?

Rules:
- If there are any new developments, state what has happened in 2–3 sentences; do not recap past events.
- If the entire post merely reviews or summarizes past events,\
or is purely an emotional outburst without factual information,\
reply: No new developments
- Do not use metatextual language such as "summary," "this post," or "this content."
- Just write the event details directly.

Abstract: {summary}
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


def load_existing(out_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not out_path.exists():
        return ids
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("id"):
                    ids.add(row["id"])
            except json.JSONDecodeError:
                continue
    return ids


def extract_new(date: str, summary: str, model: str) -> str:
    try:
        raw = call_llm(
            [{"role": "user", "content": PROMPT.format(date=date, summary=summary[:1500])}],
            model=model,
            temperature=0.0,
            max_tokens=300,
            reasoning_effort="none",
        )
    except Exception as e:
        return f"[extract_error] {e}"
    return raw.strip().lstrip("`").rstrip("`")


def process_one(item: Dict[str, Any], event_name: str, model: str,
                out_path: Path, log_path: Path,
                counter: Dict[str, int], total: int) -> None:
    date = (item.get("date") or "")[:10]
    summary = item.get("summary_clean") or item.get("summary") or ""
    if summary.strip() in ("Irrelevant", "Cleaning failed", ""):
        new_info = "No new developments"
    else:
        new_info = extract_new(date, summary, model)
    row = {
        "id": item["id"],
        "platform": item.get("platform"),
        "stage": item.get("stage"),
        "date": date,
        "title": item.get("title", ""),
        "liked_count": item.get("liked_count", 0),
        "total_comments": item.get("total_comments", 0),
        "url": item.get("url", ""),
        "summary_clean": summary,
        "new_info": new_info,
    }
    with WRITE_LOCK:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with DONE_LOCK:
        counter["done"] += 1
        done = counter["done"]
    pct = done / total * 100 if total else 0
    status = "New" if ("No new developments" not in new_info and not new_info.startswith("[extract_error]")) else "Old"
    log(f"[{done}/{total}] ({pct:.1f}%) [{status}] {date} | {item.get('title','')[:40]}", log_path)


def main() -> None:
    p = argparse.ArgumentParser(description="media-step4-extract: tag new_info per media item")
    p.add_argument("input_jsonl")
    p.add_argument("output_dir")
    p.add_argument("--event-name", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    rows = read_jsonl(Path(args.input_jsonl))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "media_step4.jsonl"
    log_path = out_dir / "media_step4.log"

    existing = load_existing(out_path)
    todo = [r for r in rows if r.get("id") and r["id"] not in existing]
    counter = {"done": len(existing)}
    total = len(rows)

    log(f"Step 4: total={total} done={len(existing)} todo={len(todo)} threads={args.threads}", log_path)

    if todo:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = {pool.submit(process_one, r, args.event_name, model,
                                 out_path, log_path, counter, total): r for r in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    new_count = old_count = 0
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ni = row.get("new_info", "")
                if "无新进展" in ni or ni.startswith("[extract_error]"):
                    old_count += 1
                else:
                    new_count += 1
    # SKILL §progress.json: explicit status snapshot for audit
    try:
        _final_done = (sum(1 for _ in open(out_path, encoding="utf-8"))
                       if "out_path" in dir() and Path(out_path).exists() else 0)
        write_progress(Path(out_path).with_name(Path(out_path).stem + "_progress.json"),
                       done_count=_final_done, total=len(rows))
    except Exception:
        pass
    print(json.dumps({
        "input": len(rows), "has_new_info": new_count, "no_new_info": old_count,
        "output_file": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
