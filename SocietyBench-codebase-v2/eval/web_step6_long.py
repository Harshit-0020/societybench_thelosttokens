#!/usr/bin/env python3
"""web-step6-long (1:1 of skills/web-step6-long/SKILL.md).

SKILL §"CC does it itself" = the work is done by an LLM. The SKILL forbids using
kimi specifically, but any other LLM is fine. Model configurable via `--model` /
pipeline_config.default_llm_model.

Pipeline:
  1. Build {id → step1.summary} index (SKILL §1)
  2. Parse timeline_short.md, extract every `[ID: ...]` bullet (SKILL §2)
  3. For each bullet, ask the LLM to expand into a 200-300 char news-style
     paragraph **using only the original step1 summary** as source. Multiple
     IDs are joined. (SKILL §3)
  4. Rebuild markdown — preserve stage / date headers, DROP `[ID: ...]` markers
     per SKILL §"4. output format" ("final file carries no IDs")
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from common import call_llm, load_pipeline_config, read_jsonl


PROMPT = """\
请把下面这条"短版"扩写成 200-300 字的事件叙述：

- 像新闻稿，语言直接，包含具体的时间、人物、机构、行为、结果
- **只写该日期的新进展**，不回顾旧事
- 禁止出现"摘要""本文""该报道"等元语言
- **信息只能来自给定的原始摘要**，绝不凭记忆或推断补充任何细节
- 如果原始摘要不足以撑起 200 字，宁可少于 200 字也不要编造

只输出扩写后的中文段落，不要任何额外说明。

短版要点：{short}

原始摘要：
{src}
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


def expand_one(short: str, src_summary: str, model: str) -> str:
    if not src_summary.strip():
        return short  # no source → preserve short text rather than hallucinate
    try:
        raw = call_llm(
            [{"role": "user", "content": PROMPT.format(short=short, src=src_summary[:5000])}],
            model=model,
            temperature=0.0,
            max_tokens=800,
            reasoning_effort="none",
        )
    except Exception as e:
        return f"[expand_error] {e}"
    return raw.strip().lstrip("`").rstrip("`")


def main() -> None:
    p = argparse.ArgumentParser(description="web-step6-long: expand short → 200-300 char paragraphs")
    p.add_argument("short_md", help="timeline_short.md from step5")
    p.add_argument("step1_jsonl", help="timeline_step1.jsonl (original summaries)")
    p.add_argument("output_dir")
    p.add_argument("--event-name", default="")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model", default=None,
                   help="LLM model; defaults to pipeline_config.default_llm_model")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    src_by_id: Dict[str, Dict[str, Any]] = {
        r["id"]: r for r in read_jsonl(Path(args.step1_jsonl)) if r.get("id")
    }
    text = Path(args.short_md).read_text(encoding="utf-8")
    lines = text.splitlines()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "timeline.log"

    items: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        m_id = re.search(r"\[ID:\s*([^\]]+)\]", line)
        if not m_id:
            continue
        ids = [s.strip() for s in m_id.group(1).split(",") if s.strip()]
        short = re.sub(r"^\s*[-*]\s*\[ID:[^\]]+\]\s*", "", line).strip()
        short = re.sub(r"\s*\[ID:[^\]]+\]\s*$", "", short).strip()
        items.append({"line_no": idx, "ids": ids, "short": short, "expanded": None})

    log(f"Step 6: lines={len(lines)} bullets={len(items)} threads={args.threads} model={model}", log_path)

    counter = {"done": 0}

    def process(item: Dict[str, Any]) -> None:
        # SKILL: "if an ID is missing from step1, keep the short-version text as-is (tag it as source-missing)"
        missing_ids = [i for i in item["ids"] if i not in src_by_id]
        item["src_missing"] = bool(missing_ids) and len(missing_ids) == len(item["ids"])
        item["partial_src"] = bool(missing_ids) and not item["src_missing"]
        src_summary = "\n\n".join(
            src_by_id[i].get("summary", "") for i in item["ids"] if i in src_by_id
        )
        item["expanded"] = expand_one(item["short"], src_summary, model)
        with DONE_LOCK:
            counter["done"] += 1
        miss_tag = " [来源缺失]" if item["src_missing"] else (
            " [部分来源缺失]" if item["partial_src"] else ""
        )
        log(f"[{counter['done']}/{len(items)}] line={item['line_no']} ids={item['ids']}{miss_tag}",
            log_path)

    if items:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = [pool.submit(process, it) for it in items]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"thread error: {e}", log_path)

    by_line = {it["line_no"]: it for it in items}
    out_lines: List[str] = []
    saw_title = any(ln.startswith("# ") for ln in lines)
    if not saw_title and args.event_name:
        out_lines.append(f"# {args.event_name} — Timeline")
        out_lines.append("")
    for i, line in enumerate(lines):
        if i in by_line:
            it = by_line[i]
            text = it["expanded"] or it["short"]
            # SKILL: "ID not found → keep original text (tag as source-missing)"
            if it.get("src_missing"):
                text = f"{text} ⚠️[来源缺失]"
            elif it.get("partial_src"):
                text = f"{text} ⚠️[部分来源缺失]"
            out_lines.append(f"- {text}")
        else:
            out_lines.append(line)

    out_md = "\n".join(out_lines).rstrip() + "\n"
    out_path = out_dir / "timeline.md"
    out_path.write_text(out_md, encoding="utf-8")
    print(json.dumps({"input_bullets": len(items), "output_file": str(out_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
