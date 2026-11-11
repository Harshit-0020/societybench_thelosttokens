#!/usr/bin/env python3
"""media-step2-clean (1:1 of skills/media-step2-clean/SKILL.md).

Rewrites each media_step1.jsonl summary to drop meta-language and platform
boilerplate. Items whose summary is already "Irrelevant" / "Summary generation failed" / blank
pass through marked irrelevant. 8 threads, resume by id, reasoning_effort="none".
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
Please rewrite the following summary of social media sentiment, with the following requirements:
1. Directly narrate the event content and public opinion, just like writing a public opinion brief.
2. Delete all meta-language, including but not limited to: "comment section," "publisher," "this post," "summary," "this article," "this Weibo," "post content," "video content," "this video," "Douyin user," "Weibo user," "Zhihu user," "the following is," "summarized as follows," "it is worth noting that."
3. Change "The comment section presents the XX viewpoint" to "Public opinion presents the XX viewpoint" or directly state the viewpoint.
4. Change "the publisher believes" to a direct statement of opinion or use "it is believed that."
5. If the content is truly unrelated to {event}", simply reply: Irrelevant
6. Return only the rewritten text; do not add any prefixes or suffixes. And keep length between 100-200 words

Original text: {summary}
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


def clean_one(summary: str, event_name: str, model: str) -> str:
    try:
        raw = call_llm(
            [{"role": "user", "content": PROMPT.format(event=event_name, summary=summary)}],
            model=model,
            temperature=0.0,
            max_tokens=500,
            reasoning_effort="none",
        )
    except Exception as e:
        return f"[clean_error] {e}"
    return raw.strip().lstrip("`").rstrip("`")


def process_one(item: Dict[str, Any], event_name: str, model: str,
                out_path: Path, log_path: Path,
                counter: Dict[str, int], total: int) -> None:
    # Per SKILL §features: LLM does a 2nd-pass relevance judgment. If it returns
    # "Irrelevant" we still record it (so downstream string-match filter can act on
    # `summary_clean == "Irrelevant"`), but we do NOT add a `relevant` field —
    # SKILL output schema only has summary_clean.
    summary = item.get("summary", "")
    cleaned = clean_one(summary, event_name, model)
    if cleaned.startswith("[clean_error]"):
        cleaned = "Cleaning failed"
    row = {
        "id": item["id"],
        "platform": item.get("platform"),
        "stage": item.get("stage"),
        "date": item.get("date"),
        "title": item.get("title", ""),
        "liked_count": item.get("liked_count", 0),
        "total_comments": item.get("total_comments", 0),
        "url": item.get("url", ""),
        "summary_clean": cleaned,
    }
    with WRITE_LOCK:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with DONE_LOCK:
        counter["done"] += 1
        done = counter["done"]
    pct = done / total * 100 if total else 0
    tag = "Irrelevant" if cleaned == "Irrelevant" else cleaned[:40]
    log(f"[{done}/{total}] ({pct:.1f}%) {item.get('platform','')}|{item.get('date','')} | {tag}", log_path)


def main() -> None:
    p = argparse.ArgumentParser(description="media-step2-clean: strip meta-language & platform boilerplate")
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
    # SKILL §features: "auto-skip: entries marked 'Irrelevant' or 'Summary generation failed' in Step 1 are not sent to cleaning"
    # So we filter at load time, dropping those rows entirely from output.
    items = [
        r for r in rows
        if (r.get("summary") or "").strip() not in ("Irrelevant", "Summary generation failed", "")
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "media_step2.jsonl"
    log_path = out_dir / "media_step2.log"
    # Touch the output file so an empty run (all upstream rows sentinel-filtered
    # or zero-input) still produces a readable artifact — otherwise downstream
    # media-step3 raises FileNotFoundError on open.
    out_path.touch(exist_ok=True)

    existing = load_existing(out_path)
    todo = [r for r in items if r.get("id") and r["id"] not in existing]
    counter = {"done": len(existing)}
    total = len(items)

    log(f"Step 2: total={total} done={len(existing)} todo={len(todo)} threads={args.threads}", log_path)

    if todo:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = {pool.submit(process_one, r, args.event_name, model,
                                 out_path, log_path, counter, total): r for r in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    # SKILL §progress.json: explicit status snapshot for audit
    try:
        _final_done = (sum(1 for _ in open(out_path, encoding="utf-8"))
                       if "out_path" in dir() and Path(out_path).exists() else 0)
        write_progress(Path(out_path).with_name(Path(out_path).stem + "_progress.json"),
                       done_count=_final_done, total=len(rows))
    except Exception:
        pass
    print(json.dumps({"input": len(rows), "rows_in_output": len(load_existing(out_path)),
                      "output_file": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
