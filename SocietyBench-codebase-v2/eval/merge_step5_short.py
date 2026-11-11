#!/usr/bin/env python3
"""merge-step5-short (1:1 of skills/merge-step5-short/SKILL.md).

SKILL §"CC does it itself" = the work is done by an LLM. The SKILL says not to
use kimi specifically (original team's self-contamination concern),
but any other LLM is fine. Model configurable via `--model` /
pipeline_config.default_llm_model.

Pipeline:
  1. Parse both web and media short timelines, group bullets by date
  2. For each date with both-side bullets, ask the LLM to decide which
     same-event pairs to MERGE, KEEP_W, KEEP_M (per SKILL §2 manual merge
     logic — automated via LLM here)
  3. Tag survivors [W]/[M]/[WM], preserve [ID: ...] lists (SKILL §"⚠️ mandatory requirements")
  4. Group by stage and write `timeline_merged_short.md` with per-source counts
     at the end (SKILL §"output format")
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
from typing import Any, Dict, List, Set, Tuple

from common import call_llm, load_pipeline_config


PROMPT = """\
请决定下面同一天 ({date}) web 与 media 列表里哪些条目说的是**同一件事**：

- 同一件事 → 输出一行：`MERGE: w_idx, m_idx`（可多 web 多 media，如 `MERGE: w0, w2, m1`）
- 独立条目 → 输出 `KEEP_W: w_idx` 或 `KEEP_M: m_idx`
- 每行一条指令，不要带其他文字

================ 日期 {date} ================
[web 候选]
{web_block}
[media 候选]
{media_block}
================
"""

DATE_RE = re.compile(r"^###\s*(\d{4}-\d{2}-\d{2})")
STAGE_RE = re.compile(r"^##\s+(.+)$")
WRITE_LOCK = threading.Lock()


def log(msg: str, log_path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with WRITE_LOCK:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def parse_short_md(path: Path) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """Return (date → stage_name, date → list of {text, ids})."""
    date_stage: Dict[str, str] = {}
    items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return date_stage, items
    cur_stage = ""
    cur_date = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        m_stage = STAGE_RE.match(line.rstrip())
        if m_stage and not line.startswith("###"):
            cur_stage = m_stage.group(1)
            continue
        m_date = DATE_RE.match(line.strip())
        if m_date:
            cur_date = m_date.group(1)
            date_stage.setdefault(cur_date, cur_stage)
            continue
        m_id = re.search(r"\[ID:\s*([^\]]+)\]", line)
        if m_id:
            # date comes from either: (1) a preceding ### header (web short version), or (2) inline - **YYYY-MM-DD** (media short version)
            m_inline = re.search(r"\*\*(\d{4}-\d{2}-\d{2})\*\*", line)
            d = m_inline.group(1) if m_inline else cur_date
            if d:
                date_stage.setdefault(d, cur_stage)
                text = re.sub(r"^\s*[-*]\s*", "", line).strip()
                text = re.sub(r"\*\*\d{4}-\d{2}-\d{2}\*\*", "", text).strip()
                text = re.sub(r"\[ID:[^\]]+\]", "", text).strip()
                ids = [s.strip() for s in m_id.group(1).split(",") if s.strip()]
                items[d].append({"text": text, "ids": ids})
    return date_stage, items


def assert_parse_not_silent(path: Path, label: str, items: Dict[str, List[Dict[str, Any]]]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "[ID:" in text and not items:
        raise SystemExit(
            f"{label} timeline contains [ID: ...] markers but parsed 0 dated items: {path}. "
            "Check date format; supported formats are '### YYYY-MM-DD' headers and "
            "inline '- **YYYY-MM-DD** ... [ID: ...]' bullets."
        )


def merge_day(
    date: str, wlist: List[Dict[str, Any]], mlist: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    if not wlist and not mlist:
        return []
    if not wlist:
        return [{"tag": "[M]", "text": m["text"], "ids": m["ids"]} for m in mlist]
    if not mlist:
        return [{"tag": "[W]", "text": w["text"], "ids": w["ids"]} for w in wlist]

    wb = "\n".join(f"  w{i}: {w['text']}" for i, w in enumerate(wlist))
    mb = "\n".join(f"  m{i}: {m['text']}" for i, m in enumerate(mlist))
    prompt = PROMPT.format(date=date, web_block=wb, media_block=mb)
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=600,
            reasoning_effort="none",
        )
    except Exception:
        return ([{"tag": "[W]", **w} for w in wlist]
                + [{"tag": "[M]", **m} for m in mlist])

    used_w: Set[int] = set()
    used_m: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("MERGE"):
            idxs = re.findall(r"([wm])(\d+)", line)
            ws = [int(i) for t, i in idxs if t == "w" and int(i) < len(wlist)]
            ms = [int(i) for t, i in idxs if t == "m" and int(i) < len(mlist)]
            if not ws and not ms:
                continue
            text = wlist[ws[0]]["text"] if ws else mlist[ms[0]]["text"]
            ids = sum((wlist[i]["ids"] for i in ws), []) + sum((mlist[i]["ids"] for i in ms), [])
            tag = "[WM]" if (ws and ms) else ("[W]" if ws else "[M]")
            out.append({"tag": tag, "text": text, "ids": ids})
            used_w |= set(ws)
            used_m |= set(ms)
        elif line.startswith("KEEP_W"):
            m = re.search(r"\d+", line)
            if m:
                i = int(m.group())
                if 0 <= i < len(wlist):
                    out.append({"tag": "[W]", "text": wlist[i]["text"], "ids": wlist[i]["ids"]})
                    used_w.add(i)
        elif line.startswith("KEEP_M"):
            m = re.search(r"\d+", line)
            if m:
                i = int(m.group())
                if 0 <= i < len(mlist):
                    out.append({"tag": "[M]", "text": mlist[i]["text"], "ids": mlist[i]["ids"]})
                    used_m.add(i)

    for i, w in enumerate(wlist):
        if i not in used_w:
            out.append({"tag": "[W]", "text": w["text"], "ids": w["ids"]})
    for i, m in enumerate(mlist):
        if i not in used_m:
            out.append({"tag": "[M]", "text": m["text"], "ids": m["ids"]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="merge-step5-short: combine web + media short timelines")
    p.add_argument("web_short_md")
    p.add_argument("media_short_md")
    p.add_argument("output_dir")
    p.add_argument("--event-name", default="")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--model", default=None,
                   help="LLM model; defaults to pipeline_config.default_llm_model")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    web_stage_by_date, web = parse_short_md(Path(args.web_short_md))
    media_stage_by_date, media = parse_short_md(Path(args.media_short_md))
    assert_parse_not_silent(Path(args.web_short_md), "web", web)
    assert_parse_not_silent(Path(args.media_short_md), "media", media)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "timeline_merged_short.log"

    all_dates = sorted(set(web) | set(media))
    log(f"merge-step5: web_dates={len(web)} media_dates={len(media)} union={len(all_dates)} threads={args.threads} model={model}", log_path)

    merged_by_date: Dict[str, List[Dict[str, Any]]] = {}
    lock = threading.Lock()

    def process(date: str) -> None:
        merged = merge_day(date, web.get(date, []), media.get(date, []), model)
        with lock:
            merged_by_date[date] = merged
        log(f"merged {date}: w={len(web.get(date, []))} m={len(media.get(date, []))} → {len(merged)}", log_path)

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = [pool.submit(process, d) for d in all_dates]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                log(f"thread error: {e}", log_path)

    def stage_of(date: str) -> str:
        return web_stage_by_date.get(date) or media_stage_by_date.get(date) or "未分阶段"

    by_stage: Dict[str, List[str]] = defaultdict(list)
    for d in all_dates:
        by_stage[stage_of(d)].append(d)

    seen_stages: List[str] = []
    for stage in list(web_stage_by_date.values()) + list(media_stage_by_date.values()):
        if stage and stage not in seen_stages and stage in by_stage:
            seen_stages.append(stage)
    for stage in by_stage:
        if stage not in seen_stages:
            seen_stages.append(stage)

    title_prefix = (args.event_name + " — ") if args.event_name else ""
    out_lines: List[str] = [
        f"# {title_prefix}合并 Timeline（短版）",
        "",
        "> 来源：Web（新闻文章）+ Media（社媒帖子）",
        "> [W] = 仅新闻 | [M] = 仅社媒 | [WM] = 两边都有",
        "",
        "---",
        "",
    ]

    counts = {"[W]": 0, "[M]": 0, "[WM]": 0}
    for stage in seen_stages:
        out_lines.append(f"## {stage}")
        out_lines.append("")
        for date in sorted(by_stage[stage]):
            items = merged_by_date.get(date, [])
            if not items:
                continue
            out_lines.append(f"### {date}")
            for it in items:
                ids = ", ".join(it["ids"]) if it.get("ids") else ""
                out_lines.append(f"- {it['tag']} {it['text']} [ID: {ids}]")
                counts[it["tag"]] = counts.get(it["tag"], 0) + 1
            out_lines.append("")
        out_lines.append("")

    out_lines.append("---")
    out_lines.append("")
    out_lines.append("**统计**：")
    for tag in ("[W]", "[M]", "[WM]"):
        out_lines.append(f"- {tag}：{counts[tag]} 条")

    out_path = out_dir / "timeline_merged_short.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")

    # SKILL §"completion check": every event bullet (one tagged [W]/[M]/[WM] followed
    # by content) must carry [ID: xxx] — merge-step6 depends on these markers.
    # The trailing "**统计**" summary lines (e.g. "- [W]：8 条") are NOT event
    # bullets, so the regex requires a space (not a colon) after the tag.
    event_bullet_re = re.compile(r"^\s*-\s+\[(W|M|WM)\]\s+")
    missing_id_rows: List[str] = []
    for ln in out_path.read_text(encoding="utf-8").splitlines():
        if event_bullet_re.match(ln) and "[ID:" not in ln:
            missing_id_rows.append(ln)
    if missing_id_rows:
        sample = "\n  ".join(missing_id_rows[:5])
        raise SystemExit(
            f"[merge-step5 §完成检查] {len(missing_id_rows)} event bullet(s) in "
            f"{out_path} are missing [ID: ...] markers. First few:\n  {sample}"
        )

    print(json.dumps({
        "web_dates": len(web), "media_dates": len(media),
        "union_dates": len(all_dates),
        "counts": counts,
        "rows_with_id": sum(1 for ln in out_lines if ln.lstrip().startswith("- ") and "[ID:" in ln),
        "output_file": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
