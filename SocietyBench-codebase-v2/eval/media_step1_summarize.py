#!/usr/bin/env python3
"""media-step1-summarize (1:1 of skills/media-step1-summarize/SKILL.md).

For each post (with its top comments from step0), produce a 100-200 char
Chinese summary covering both the post content and the dominant opinion in the
comments. Irrelevant ones → "不相关".

8 threads, resume by id, reasoning_effort="none".
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from common import call_llm, load_pipeline_config, read_jsonl, write_progress


PLATFORM_CN = {
    "douyin": "抖音", "weibo": "微博", "zhihu": "知乎",
    "bili": "B站", "tieba": "贴吧", "xhs": "小红书", "ks": "快手",
}

PROMPT = """\
请根据以下社交媒体帖子及其评论，用 100-200 字概括这条帖子在讨论什么，反映了什么舆论观点或事件进展。

要求：
- 先概括帖子本身在说什么，再综合评论区的主要观点和情绪
- 直接叙述，禁止出现"摘要""本文""该帖子""这条微博"等元语言
- 如果内容跟「{event}」无关，只回复三个字：不相关
- 直接从内容开始写，不要任何前缀

{post_block}
{comments_block}
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


def build_prompt(post: Dict[str, Any], event_name: str) -> str:
    platform = PLATFORM_CN.get(post.get("platform"), post.get("platform", ""))
    title = (post.get("title") or "").strip()
    content = (post.get("content") or "").strip()
    user_name = post.get("user_name", "")
    date = post.get("date", "")
    liked = post.get("liked_count", 0)
    cn = post.get("total_comments", 0)

    parts: List[str] = []
    if title:
        parts.append(f"标题：{title}")
    if content and content != title:
        parts.append(f"正文：{content[:1500]}")
    parts.append(f"发布者：{user_name} | 平台：{platform} | 日期：{date} | 点赞：{liked} | 评论数：{cn}")
    post_block = "\n".join(parts)

    top = post.get("top_comments", []) or []
    # Accept either string format "[赞N] content" (from step0) or {content, liked_count} dict
    norm_comments: List[str] = []
    for c in top[:20]:
        if isinstance(c, str):
            norm_comments.append(c)
        elif isinstance(c, dict):
            txt = (c.get("content") or "").strip()
            if txt:
                norm_comments.append(f"[赞{c.get('liked_count', 0)}] {txt}")
    comments_block = ("\n\n热门评论:\n" + "\n".join(norm_comments)) if norm_comments else ""

    return PROMPT.format(event=event_name, post_block=post_block, comments_block=comments_block)


def summarize_one(post: Dict[str, Any], event_name: str, model: str) -> str:
    try:
        raw = call_llm(
            [{"role": "user", "content": build_prompt(post, event_name)}],
            model=model,
            temperature=0.0,
            max_tokens=500,
            reasoning_effort="none",
        )
    except Exception as e:
        return f"[summarize_error] {e}"
    return raw.strip().lstrip("`").rstrip("`")


def process_one(post: Dict[str, Any], event_name: str, model: str,
                out_path: Path, log_path: Path,
                counter: Dict[str, int], total: int) -> None:
    summary = summarize_one(post, event_name, model)
    # If LLM returned an error sentinel, normalize to SKILL's "摘要生成失败"
    # so downstream (media-step2) can detect via simple string match per SKILL.
    if summary.startswith("[summarize_error]"):
        summary = "摘要生成失败"
    row = {
        "id": post["id"],
        "platform": post.get("platform"),
        "stage": post.get("stage"),
        "date": post.get("date"),
        "title": (post.get("title") or "")[:100],
        "liked_count": post.get("liked_count", 0),
        "total_comments": post.get("total_comments", 0),
        "url": post.get("url", ""),
        "summary": summary,
    }
    with WRITE_LOCK:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with DONE_LOCK:
        counter["done"] += 1
        done = counter["done"]
    pct = done / total * 100 if total else 0
    tag = "不相关" if summary == "不相关" else summary[:40]
    log(f"[{done}/{total}] ({pct:.1f}%) {post.get('platform','')}|{post.get('date','')} | {tag}", log_path)


def main() -> None:
    p = argparse.ArgumentParser(description="media-step1-summarize: post+comments → 100-200 char summary")
    p.add_argument("input_jsonl")
    p.add_argument("output_dir")
    p.add_argument("--event-name", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    posts = read_jsonl(Path(args.input_jsonl))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "media_step1.jsonl"
    log_path = out_dir / "media_step1.log"

    existing = load_existing(out_path)
    todo = [post for post in posts if post.get("id") and post["id"] not in existing]
    counter = {"done": len(existing)}
    total = len(posts)

    log(f"Step 1: total={total} done={len(existing)} todo={len(todo)} threads={args.threads}", log_path)

    if todo:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = {pool.submit(process_one, post, args.event_name, model,
                                 out_path, log_path, counter, total): post for post in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    n_total = len(load_existing(out_path))
    # SKILL §progress.json: explicit status snapshot for audit
    try:
        _final_done = (sum(1 for _ in open(out_path, encoding="utf-8"))
                       if "out_path" in dir() and Path(out_path).exists() else 0)
        write_progress(Path(out_path).with_name(Path(out_path).stem + "_progress.json"),
                       done_count=_final_done, total=len(posts))
    except Exception:
        pass
    print(json.dumps({"input": len(posts), "rows_in_output": n_total,
                      "output_file": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
