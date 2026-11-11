#!/usr/bin/env python3
"""media-step0-aggregate (1:1 of skills/media-step0-aggregate/SKILL.md).

Purely algorithmic — no LLM. Two outputs:
  1. posts_aggregated.jsonl — one record per post, with top-N most-liked
     comments (and sub_comments) attached
  2. stage_stats.json + a printed table — per-platform post/comment counts
     grouped by event stage (if --stages is supplied)

`type` values handled:  `post`  vs  `comment` / `sub_comment` (latter both
treated as comments under their parent_id).

Stages config (optional, event-specific): a JSON file like:
    {
      "stages": [
        {"name": "S01_prelude", "start": "2025-04-01", "end": "2025-07-05"},
        {"name": "S02_incident1", "start": "2025-07-06", "end": "2025-07-18"}
      ]
    }
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import load_pipeline_config, parse_date_obj


def load_stages(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    stages = data.get("stages", []) if isinstance(data, dict) else []
    out = []
    for s in stages:
        out.append({
            "name": s.get("name") or "",
            "start": parse_date_obj(s.get("start")),
            "end": parse_date_obj(s.get("end")),
        })
    return out


def find_stage(d: Optional[date], stages: List[Dict[str, Any]]) -> str:
    if d is None or not stages:
        return "unknown"
    for s in stages:
        if s["start"] and s["end"] and s["start"] <= d <= s["end"]:
            return s["name"]
    return "out_of_range"


def aggregate(
    rows: List[Dict[str, Any]],
    top_comments: int,
    stages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    posts: Dict[str, Dict[str, Any]] = {}
    comments_by_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        rtype = (r.get("type") or "").strip().lower()
        if rtype == "post":
            posts[str(r.get("id"))] = r
        elif rtype in ("comment", "sub_comment"):
            parent = str(r.get("parent_id") or "")
            if parent:
                comments_by_parent[parent].append(r)

    aggregated: List[Dict[str, Any]] = []
    for pid, post in posts.items():
        cmts = sorted(
            comments_by_parent.get(pid, []),
            key=lambda c: int(c.get("liked_count") or 0),
            reverse=True,
        )
        top = cmts[:top_comments]
        ct_str = post.get("create_time") or ""
        d = parse_date_obj(ct_str)
        stage = find_stage(d, stages)
        top_comment_strs: List[str] = []
        for c in top:
            txt = (c.get("content") or "").strip()
            if not txt:
                continue
            top_comment_strs.append(f"[赞{c.get('liked_count', 0)}] {txt}")
        aggregated.append({
            "id": f"{post.get('platform','')}_{pid}",
            "platform": post.get("platform"),
            "stage": stage,
            "date": ct_str[:10] if ct_str else "",
            "create_time": ct_str,
            "title": (post.get("title") or "").strip(),
            "content": (post.get("content") or "").strip(),
            "liked_count": int(post.get("liked_count") or 0),
            "comment_count": int(post.get("comment_count") or 0),
            "share_count": int(post.get("share_count") or 0),
            "user_name": post.get("user_name", ""),
            "ip_location": post.get("ip_location", ""),
            "source_keyword": post.get("source_keyword", ""),
            "url": post.get("url", ""),
            "total_comments": len(cmts),
            "top_comments": top_comment_strs,
        })
    aggregated.sort(key=lambda x: x.get("create_time") or "")

    # Stats: per stage × platform × {post, comment}
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        rtype = (r.get("type") or "").strip().lower()
        if rtype not in ("post", "comment", "sub_comment"):
            continue
        d = parse_date_obj(r.get("create_time"))
        stage = find_stage(d, stages)
        plat = (r.get("platform") or "unknown")
        suffix = "post" if rtype == "post" else "comment"
        stats[stage][f"{plat}_{suffix}"] += 1
        stats[stage][f"total_{suffix}"] += 1

    return aggregated, dict(stats)


def main() -> None:
    p = argparse.ArgumentParser(description="media-step0-aggregate: post + top-N comments + stage stats")
    p.add_argument("input_csv")
    p.add_argument("output_dir")
    p.add_argument("--top-comments", type=int, default=None)
    p.add_argument("--stages", default=None, help="Optional stages JSON config")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    top_n = args.top_comments if args.top_comments is not None else int(
        cfg.get("media_step0_aggregate", {}).get("top_comments_per_post", 30)
    )
    stages = load_stages(Path(args.stages)) if args.stages else []

    rows: List[Dict[str, Any]] = []
    with open(args.input_csv, "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    aggregated, stats = aggregate(rows, top_n, stages)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "posts_aggregated.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in aggregated:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats_path = out_dir / "stage_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print stage table
    if stages:
        print(f"{'stage':<28} {'post':>6} {'comment':>8}")
        print("-" * 50)
        for s in stages + [{"name": "out_of_range"}, {"name": "unknown"}]:
            name = s["name"]
            row = stats.get(name, {})
            print(f"{name:<28} {row.get('total_post', 0):>6} {row.get('total_comment', 0):>8}")
        print("-" * 50)

    with_comments = sum(1 for r in aggregated if r["total_comments"] > 0)
    print(json.dumps({
        "input_rows": len(rows),
        "output_posts": len(aggregated),
        "with_comments": with_comments,
        "without_comments": len(aggregated) - with_comments,
        "top_comments_per_post": top_n,
        "stages_used": len(stages),
        "output_file": str(out_path),
        "stats_file": str(stats_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
