#!/usr/bin/env python3
"""web-step2-clean (1:1 of skills/web-step2-clean/SKILL.md).

Reads step1.jsonl, asks the LLM to rewrite each summary to remove meta-language
("Summary"/"This article"/"The article" etc.) and web junk (By/Updated/nav bars/captions/Subscribe etc.),
flagging irrelevant items as "Irrelevant".

Per the SKILL: 8 threads, resume via output JSONL on disk, per-row append,
reasoning_effort="none".
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from common import call_llm, load_pipeline_config, read_jsonl, write_progress


PROMPT = """\
Please rewrite the following text. Requirements:
1. Describe the sequence of events directly, as if writing a news report
2. Remove all metadiscursive language, such as "summary", "this paper", "the article", "this report", "news article", "the following is", "in summary", "it is worth noting", "it should be pointed out" and "the full text is approximately".
3. Remove all web clutter / portal remnants, including but not limited to:
   - Author's byline：By / Updated on / Read Full Bio / Skip to content / Section Navigation
   - Subscription/Login Guide：Subscribe / Log in / Sign up / Continue reading
   - Image label: Image credit / Figure caption / Caption / Photo by
   - Website tagline：Site slogan, section names, broken nav links
   - Information card fragments：View Details / Reading time / Key points / Ask AI / live ticker
   - Multilingual mixed field：cookies notice, privacy policy prompt
4. Delete all judgments such as "irrelevant to the event" and "cannot be determined"
5. If the content is indeed Irrelevant to "{event}", just reply: Irrelevant
6. Directly output the rewritten text, do not add any prefixes and suffixes

Original text：
{summary}
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


def load_existing(out_path: Path) -> Dict[str, Dict[str, Any]]:
    by_url: Dict[str, Dict[str, Any]] = {}
    if not out_path.exists():
        return by_url
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("url"):
                    by_url[row["url"]] = row
            except json.JSONDecodeError:
                continue
    return by_url


def clean_one(summary: str, event_name: str, model: str) -> str:
    try:
        raw = call_llm(
            [{"role": "user", "content": PROMPT.format(event=event_name, summary=summary)}],
            model=model,
            temperature=0.0,
            max_tokens=800,
            reasoning_effort="none",
        )
    except Exception as e:
        return f"[clean_error] {e}"
    return raw.strip().lstrip("`").rstrip("`")


def process_one(item: Dict[str, Any], event_name: str, model: str,
                out_path: Path, log_path: Path, counter: Dict[str, int], total: int) -> None:
    # SKILL §script: items already labelled "Irrelevant to the event / Summary generation failed / Irrelevant"
    # pass through as "Irrelevant" without re-asking the LLM. SKILL output schema
    # has no boolean `relevant` field — only the input-passthrough `relevance`
    # string and the (possibly rewritten) `summary`.
    summary = item.get("summary", "")
    if summary.strip() in ("Irrelevant to the event", "Summary generation failed", "Irrelevant"):
        cleaned = "Irrelevant"
    else:
        cleaned = clean_one(summary, event_name, model)
        if cleaned.startswith("[clean_error]"):
            cleaned = "Irrelevant"
    row = {
        "id": item.get("id", ""),
        "url": item.get("url"),
        "title": item.get("title", ""),
        "date": item.get("date", ""),
        "relevance": item.get("relevance", ""),  # input-passthrough per SKILL output schema
        "summary": cleaned,
    }
    with WRITE_LOCK:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with DONE_LOCK:
        counter["done"] += 1
        done = counter["done"]
    pct = done / total * 100 if total else 0
    log(f"[{done}/{total}] ({pct:.1f}%) {item.get('date','')[:10]} | {item.get('title','')[:40]}", log_path)


def main() -> None:
    p = argparse.ArgumentParser(description="web-step2-clean: strip meta-language & web junk")
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
    # de-dup by url
    seen = set()
    items: List[Dict[str, Any]] = []
    for r in rows:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u)
            items.append(r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timeline_step2.jsonl"
    log_path = out_dir / "timeline_step2.log"

    existing = load_existing(out_path)
    todo = [it for it in items if it.get("url") and it["url"] not in existing]
    counter = {"done": len(existing)}
    total = len(items)

    log(f"Step 2: total={total} done={len(existing)} todo={len(todo)} threads={args.threads}", log_path)

    if todo:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = {pool.submit(process_one, it, args.event_name, model,
                                 out_path, log_path, counter, total): it for it in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    final = load_existing(out_path)
    irrelevant = sum(1 for r in final.values() if str(r.get("summary", "")).strip() == "Irrelevant")
    # SKILL §progress.json: explicit status snapshot for audit
    try:
        _final_done = (sum(1 for _ in open(out_path, encoding="utf-8"))
                       if "out_path" in dir() and Path(out_path).exists() else 0)
        write_progress(Path(out_path).with_name(Path(out_path).stem + "_progress.json"),
                       done_count=_final_done, total=len(rows), completed_ids=list(final.keys()) if 'final' in dir() else None)
    except Exception:
        pass
    print(json.dumps({
        "input_rows": len(rows),
        "unique_urls": len(items),
        "rows_in_output": len(final),
        "relevant": len(final) - irrelevant,
        "irrelevant": irrelevant,
        "output_file": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
