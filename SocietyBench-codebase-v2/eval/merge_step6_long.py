#!/usr/bin/env python3
"""merge-step6-long (1:1 of skills/merge-step6-long/SKILL.md).

SKILL §"CC does it itself" = the work is done by an LLM. The SKILL says not to
call kimi specifically (original team's self-contamination concern),
but any other LLM is fine. Model configurable via `--model` /
pipeline_config.default_llm_model.

Pipeline:
  1. Build three indexes (SKILL §"execution steps"):
       - web_step1[id] → original summary  (for **事件** expansion)
       - media_step1[id] → original summary
       - media_step3 grouped by date       (for **舆论** synthesis)
  2. Walk merged_short_md, collect each date section with its `[ID: ...]` bullets
  3. For each bullet, ask LLM to expand to 200-300 char **事件** (fact) paragraph
     based **only** on the source summaries (SKILL §"3. expand the fact layer")
  4. For each date with ≥1 media_step3 post (±1 day window), ask LLM to
     synthesize a 100-200 char **舆论** (opinion) paragraph (SKILL §"4. attach the opinion layer")
  5. Rebuild markdown — preserve stage / date headers, replace bullets with
     **事件**/**舆论** paragraphs, DROP `[ID: ...]` markers

Pre-check: merged short input must contain `[ID:` markers; otherwise abort.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import call_llm, load_pipeline_config, parse_date_obj, read_jsonl


PROMPT_FACT = """\
请把下面这条"短版"扩写成 200-300 字的事件叙述：

- 像新闻稿，包含具体时间、人物、机构、行为、结果
- **只写该日期的新进展**，不回顾旧事
- 禁止"摘要""本文""该报道""该帖子"等元语言
- **信息只能来自给定的原始摘要**，绝不凭记忆或推断补充
- 如果原始摘要内容不够撑起 200 字，宁可少于 200 字也不要编造

只输出扩写后的中文段落，不要任何额外说明。

短版要点：{short}

原始摘要（可能多条，请综合）：
{src}
"""

PROMPT_OPINION = """\
请基于下面 {date} 当天的多条社媒舆论摘要，写一段 100-200 字的公众反应概括：

- 主流观点（支持谁、反对谁、为什么）
- 情绪倾向（愤怒/同情/讽刺/疲惫等）
- 争议焦点（具体在争什么）
- 典型言论/梗

只写舆论本身，不要复述事件经过。只输出综合后的中文段落，不要任何额外说明。

================ 舆论摘要 ================
{posts}
================
"""

OPINION_MIN_POSTS = 1  # threshold ≥3→≥1: social-media opinion is sparse anyway; synthesize an opinion paragraph from ≥1 post to reduce "missing opinion" nodes
OPINION_WINDOW_DAYS = 1

DATE_RE = re.compile(r"^###\s+(\d{4}-\d{2}-\d{2})")
WRITE_LOCK = threading.Lock()


def log(msg: str, log_path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with WRITE_LOCK:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def expand_fact(short: str, src: str, model: str) -> str:
    if not src.strip():
        return short
    try:
        raw = call_llm(
            [{"role": "user", "content": PROMPT_FACT.format(short=short, src=src[:6000])}],
            model=model,
            temperature=0.0,
            max_tokens=800,
            reasoning_effort="none",
        )
    except Exception as e:
        return f"[expand_error] {e}"
    return raw.strip().lstrip("`").rstrip("`")


def synth_opinion(date: str, posts: List[str], model: str) -> Optional[str]:
    if len(posts) < OPINION_MIN_POSTS:
        return None
    block = "\n".join(f"- {p[:200]}" for p in posts[:15])
    try:
        raw = call_llm(
            [{"role": "user", "content": PROMPT_OPINION.format(date=date, posts=block)}],
            model=model,
            temperature=0.0,
            max_tokens=500,
            reasoning_effort="none",
        )
    except Exception:
        return None
    text = raw.strip().lstrip("`").rstrip("`")
    return text or None


def main() -> None:
    p = argparse.ArgumentParser(description="merge-step6-long: expand facts + attach public opinion")
    p.add_argument("merged_short_md")
    p.add_argument("web_step1_jsonl")
    p.add_argument("media_step1_jsonl")
    p.add_argument("media_step3_jsonl")
    p.add_argument("output_dir")
    p.add_argument("--event-name", default="")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model", default=None,
                   help="LLM model; defaults to pipeline_config.default_llm_model")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    short_path = Path(args.merged_short_md)
    short_text = short_path.read_text(encoding="utf-8")
    if "[ID:" not in short_text:
        raise SystemExit(f"{short_path} has no [ID: ...] markers — rerun merge-step5 first")

    web_by_id = {r["id"]: r for r in read_jsonl(Path(args.web_step1_jsonl)) if r.get("id")}
    media_by_id = {r["id"]: r for r in read_jsonl(Path(args.media_step1_jsonl)) if r.get("id")}

    by_date: Dict[str, List[str]] = defaultdict(list)
    for r in read_jsonl(Path(args.media_step3_jsonl)):
        body = r.get("summary_clean") or r.get("summary") or ""
        if body.strip() in ("不相关", "清洗失败", ""):
            continue
        date_str = (r.get("date") or "")[:10]
        if date_str:
            by_date[date_str].append(body)

    def opinion_posts_for(date_str: str) -> List[str]:
        d = parse_date_obj(date_str)
        if d is None:
            return []
        out: List[str] = []
        for off in range(-OPINION_WINDOW_DAYS, OPINION_WINDOW_DAYS + 1):
            out.extend(by_date.get((d + timedelta(days=off)).isoformat(), []))
        return out

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "timeline_merged_long.log"

    # SKILL §4. keep intermediate files: dump the three indexes to disk so reruns and
    # audits can inspect what merge-step6 saw at this moment.
    (out_dir / "_idx_web.json").write_text(
        json.dumps({"count": len(web_by_id), "ids": sorted(web_by_id.keys())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "_idx_media.json").write_text(
        json.dumps({"count": len(media_by_id), "ids": sorted(media_by_id.keys())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "_idx_opinion.json").write_text(
        json.dumps({"dates": sorted(by_date.keys()),
                    "posts_per_date": {d: len(v) for d, v in by_date.items()},
                    "total_posts": sum(len(v) for v in by_date.values())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = short_text.splitlines()
    sections: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for idx, line in enumerate(lines):
        m = DATE_RE.match(line.strip())
        if m:
            if cur:
                cur["end_idx"] = idx
                sections.append(cur)
            cur = {"date": m.group(1), "header_idx": idx, "end_idx": idx, "items": []}
            continue
        if cur is None:
            continue
        m_id = re.search(r"\[ID:\s*([^\]]+)\]", line)
        if not m_id:
            continue
        ids = [s.strip() for s in m_id.group(1).split(",") if s.strip()]
        tag_m = re.search(r"\[(W|M|WM)\]", line)
        tag = f"[{tag_m.group(1)}]" if tag_m else ""
        short = re.sub(r"^\s*[-*]\s*", "", line)
        short = re.sub(r"\[W?M?\]", "", short, count=1)
        short = re.sub(r"\[ID:[^\]]+\]", "", short).strip()
        srcs: List[str] = []
        for i in ids:
            if i in web_by_id:
                s = web_by_id[i].get("summary") or ""
                if s:
                    srcs.append(s)
            elif i in media_by_id:
                s = media_by_id[i].get("summary") or ""
                if s:
                    srcs.append(s)
        cur["items"].append({"tag": tag, "short": short, "ids": ids, "src_block": "\n\n".join(srcs)})
    if cur:
        cur["end_idx"] = len(lines)
        sections.append(cur)

    log(f"merge-step6: sections={len(sections)} items={sum(len(s['items']) for s in sections)} threads={args.threads} model={model}", log_path)

    tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [(sec, it) for sec in sections for it in sec["items"]]

    def process_fact(it: Dict[str, Any]) -> None:
        it["expanded"] = expand_fact(it["short"], it["src_block"], model)

    def process_opinion(sec: Dict[str, Any]) -> None:
        sec["opinion"] = synth_opinion(sec["date"], opinion_posts_for(sec["date"]), model)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = [pool.submit(process_fact, it) for _, it in tasks]
        futs += [pool.submit(process_opinion, sec) for sec in sections]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                log(f"thread error: {e}", log_path)

    out_lines: List[str] = []
    saw_title = any(ln.startswith("# ") for ln in lines)
    if not saw_title and args.event_name:
        out_lines.append(f"# {args.event_name} — 长版合并 Timeline")
        out_lines.append("")
    sec_map = {sec["header_idx"]: sec for sec in sections}
    bullet_lines: set = set()
    for sec in sections:
        for off in range(sec["header_idx"] + 1, sec["end_idx"]):
            if "[ID:" in lines[off]:
                bullet_lines.add(off)

    for i, line in enumerate(lines):
        if i in sec_map:
            sec = sec_map[i]
            out_lines.append(line)
            out_lines.append("")
            for it in sec["items"]:
                prefix = f"{it['tag']} " if it["tag"] else ""
                out_lines.append(f"**事件**：{prefix}{it.get('expanded') or it['short']}")
                out_lines.append("")
            if sec.get("opinion"):
                out_lines.append(f"**舆论**：{sec['opinion']}")
                out_lines.append("")
            continue
        if i in bullet_lines:
            continue
        out_lines.append(line)

    # SKILL §"4. keep intermediate files": split into timeline_long_part1.md (first half
    # of stages) and timeline_long_part2.md (second half). Each represents one
    # of the two parallel writer-Agents in the SKILL's two-Agent design. The
    # final merged_long is the concatenation, per SKILL §"3. merge output".
    stage_idx = [i for i, ln in enumerate(out_lines) if ln.startswith("## ")]
    if len(stage_idx) >= 2:
        mid = stage_idx[len(stage_idx) // 2]
        part1_lines = out_lines[:mid]
        part2_lines = out_lines[mid:]
    else:
        part1_lines = list(out_lines)
        part2_lines = []

    part1_md = out_dir / "timeline_long_part1.md"
    part2_md = out_dir / "timeline_long_part2.md"
    part1_md.write_text("\n".join(part1_lines).rstrip() + "\n", encoding="utf-8")
    part2_md.write_text(
        ("\n".join(part2_lines).rstrip() + "\n") if part2_lines else "",
        encoding="utf-8",
    )

    out_md = out_dir / "timeline_merged_long.md"
    merged = part1_md.read_text(encoding="utf-8")
    if part2_lines:
        merged += part2_md.read_text(encoding="utf-8")
    out_md.write_text(merged.rstrip() + "\n", encoding="utf-8")

    print(json.dumps({
        "sections": len(sections),
        "items": sum(len(s["items"]) for s in sections),
        "opinions_attached": sum(1 for s in sections if s.get("opinion")),
        "part1_file": str(part1_md),
        "part2_file": str(part2_md),
        "output_file": str(out_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
