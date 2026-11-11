#!/usr/bin/env python3
"""web-step1-summarize (1:1 of skills/web-step1-summarize/SKILL.md).

LLM-driven. Reads articles JSON (array of {url, title, text/text_preview, date, ...})
and for each article asks the LLM to produce a 300-400-char Chinese factual
summary. Articles judged irrelevant to the supplied event are flagged so
downstream steps can drop them.

Per the SKILL:
  - 20-way threaded execution
  - resume: skip URLs already present in timeline_step1.jsonl
  - per-result append to output JSONL (so a crash leaves partial progress)
  - per-date 0-indexed ID format `{date}_{seq}` (e.g. 2025-07-29_2)
  - reasoning explicitly disabled (effort="none") — this step is summarization,
    not deep reasoning

Usage:
    python web_step1_summarize.py <input_articles_json> <output_dir> \\
        --event-name "<your event>" [--threads 20] [--model kimi-k2.5]
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import call_llm, load_json, load_pipeline_config, parse_date_str, write_progress


PROMPT = """\
请根据以下新闻内容，直接描述发生了什么事情，约 300-400 字。

要求：
- 直接叙述事件经过，像写新闻稿一样
- 禁止出现"摘要""本文""该文章""这篇报道""与事件无关"等元语言
- 禁止出现"以下是摘要""总结如下"等开头
- 如果内容跟「{event}」无关，只回复三个字：不相关
- 直接从事件内容开始写，不要任何前缀

标题：{title}

正文：
{text}
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
    """Return {url: row} for already-completed entries (for resume)."""
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
            except json.JSONDecodeError:
                continue
            url = row.get("url")
            if url:
                by_url[url] = row
    return by_url


def summarize_text(title: str, text: str, event_name: str, model: str) -> str:
    prompt = PROMPT.format(event=event_name, title=title[:200] or "无标题", text=text[:3000])
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=800,
            reasoning_effort="none",  # SKILL: disable reasoning, otherwise extremely slow
        )
    except Exception as e:
        return f"[summarize_error] {e}"
    return raw.strip()


def process_one(
    article: Dict[str, Any],
    article_id: str,
    event_name: str,
    model: str,
    out_path: Path,
    log_path: Path,
    counter_ref: Dict[str, int],
    total: int,
) -> Dict[str, Any]:
    title = article.get("title", "") or ""
    text = article.get("text") or article.get("text_preview") or article.get("content") or ""
    date = (article.get("date") or article.get("publish_date") or "")[:10]
    summary = summarize_text(title, text, event_name, model)
    # SKILL §"keep original on failure" + §output format: normalize error sentinel to the
    # SKILL-defined string so downstream string-match filters work.
    if summary.startswith("[summarize_error]"):
        summary = "摘要生成失败"
    result = {
        "id": article_id,
        "url": article.get("url"),
        "title": title[:200],
        "date": date,
        "summary": summary,
    }
    with WRITE_LOCK:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    with DONE_LOCK:
        counter_ref["done"] += 1
        done = counter_ref["done"]
    pct = done / total * 100 if total else 0
    log(f"[{done}/{total}] ({pct:.1f}%) {date} | {title[:50]}", log_path)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="web-step1-summarize: 300-400 char factual summary")
    p.add_argument("input_json", help="articles JSON (array of dicts)")
    p.add_argument("output_dir")
    p.add_argument("--event-name", required=True, help="research event name")
    p.add_argument("--threads", type=int, default=20)
    p.add_argument("--model", default=None,
                   help="LLM for summarization; defaults to pipeline_config.default_llm_model")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    articles = load_json(args.input_json)
    if not isinstance(articles, list):
        raise SystemExit("Input must be a JSON array of articles.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timeline_step1.jsonl"
    log_path = out_dir / "timeline_step1.log"

    # Assign stable `{date}_{idx}` IDs (0-indexed per date)
    date_counter: Dict[str, int] = defaultdict(int)
    for art in articles:
        d = (art.get("date") or art.get("publish_date") or "unknown")[:10]
        art["_id"] = f"{d}_{date_counter[d]}"
        date_counter[d] += 1

    # Resume: skip URLs already completed
    existing = load_existing(out_path)
    todo = [a for a in articles if a.get("url") and a["url"] not in existing]
    counter_ref = {"done": len(existing)}
    total = len(articles)

    log(f"Step 1: total={total} done={len(existing)} todo={len(todo)} threads={args.threads}", log_path)

    if todo:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = {
                pool.submit(process_one, a, a["_id"], args.event_name, model,
                            out_path, log_path, counter_ref, total): a
                for a in todo
            }
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    # Final stats — use SKILL string contract: summary == "不相关" → irrelevant
    final = load_existing(out_path)
    irrelevant = sum(1 for r in final.values() if str(r.get("summary", "")).strip() == "不相关")
    failed = sum(1 for r in final.values() if str(r.get("summary", "")).strip() == "摘要生成失败")
    rel = len(final) - irrelevant - failed
    # SKILL §"append progress after each step" — explicit status file for human/audit
    write_progress(out_path.with_name(out_path.stem + "_progress.json"),
                   done_count=len(final), total=len(articles),
                   completed_ids=list(final.keys()),
                   extra={"relevant": rel, "irrelevant": irrelevant, "failed": failed})
    print(json.dumps({
        "input_articles": len(articles),
        "rows_in_output": len(final),
        "relevant": rel,
        "irrelevant": irrelevant,
        "failed": failed,
        "output_file": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
