#!/usr/bin/env python3
"""merge-step6.5 — GT refine for predict input (1:1 of skills/merge-pipeline §Step 6.5).

Reads `timeline_merged_long.md` and produces `timeline_merged_long_gt.md`:
the predict-ready refined version. Per the SKILL:

  - Sentence-anchor at fact (no leading And/But/While/When/As/Though)
  - Strip title shells (`-新华网`, `北京 X 月 X 日电`, `综合消息`, `据 XX 报道`)
  - Strip portal / caption / author-bio fragments
  - Strip merged_short residue, truncation tails (.m., ..., trailing quotes)
  - Each **事件** paragraph target 200-300 chars; under 200 / over 300 → repair
  - Only delete a node if all candidate sources fail to reconstitute it
  - Round 1 baseline + auto check; Rounds 2/3 + human approval are NOT
    auto-completed — the iteration log is seeded with TODOs.

Outputs:
  <output_dir>/timeline_merged_long_gt.md
  <output_dir>/timeline_merged_long_gt_report.json
  <output_dir>/timeline_merged_long_gt_iteration_log.md
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import call_llm, load_pipeline_config


REFINE_PROMPT = """\
Please rewrite the following event narrative into a 200-300-word prediction-specific GT paragraph：

Hard requirements：
1. The beginning of the sentence must fall at the anchor point of the fact ("Who did what when"), and cannot start with conjunctions or half clauses such as And/But/While/When/As/Though.
2. Delete the title shell, column, header, and signature (such as "-Xinhua", "Beijing X Month X Day", "Comprehensive News", "According to XX Report", "AGENCE France-Presse)")
3. Delete portal fragments ("View details", "Reading time", "key points", "Ask AI"), links, URL, quotation mark fragments, markdown
4. Keep 1-3 complementary fact sentences, and the overall pressure is 200-300 words
5. It is not allowed to make up facts by memory; only the information in the given original text can be used
6. Meta-languages such as "abstract", "Article", and "The report" are prohibited

Only output the rewritten paragraphs, without any additional instructions.

Original text：
{body}
"""

DATE_RE = re.compile(r"^###\s+(\d{4}-\d{2}-\d{2})")
STAGE_RE = re.compile(r"^##\s+(.+)$")
EVENT_PREFIX = re.compile(r"^\*\*Event\*\*:\s*")
OPINION_PREFIX = re.compile(r"^\*\*Public Opinion\*\*:\s*")

JUNK_PATTERNS = [
    r"^\s*(And|But|While|When|As|Though|However)\b",
    r"^\s*[—\-]+\s*",
    r"^…+",
]
TITLE_SHELL_PATTERNS = [
    r"-(Reuters|Associated Press|AFP|BBC|CNN|Xinhua)",
    r"(News roundup|According to [^,.\\n]{2,40} reports)",
    r"Agence France-Presse[）)]",
    r"\b(Beijing|Washington|Tehran|New York|London)\s+\w+\s+\d{1,2},?\s+\d{4}\b",
    r"\b(View Details|Read Time|Key Points|Ask AI)\b",
]
JUNK_ANY_PATTERNS = [r"https?://\S+", r"ref_(src|url)=", r"granuleid:"]

WRITE_LOCK = threading.Lock()


def log(msg: str, log_path: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with WRITE_LOCK:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def hard_fail_tags(body: str) -> List[str]:
    tags: List[str] = []
    for pat in JUNK_PATTERNS:
        if re.search(pat, body):
            tags.append(f"bad_start::{pat[:20]}")
    for pat in TITLE_SHELL_PATTERNS:
        if re.search(pat, body):
            tags.append("title_shell")
            break
    for pat in JUNK_ANY_PATTERNS:
        if re.search(pat, body):
            tags.append("ref_residue")
            break
    n = len(re.sub(r"\s+", "", body))
    if n < 200:
        tags.append(f"gt_below_min_chars:{n}")
    elif n > 300:
        tags.append(f"gt_above_max_chars:{n}")
    return tags


def refine_body(body: str, event_name: str, model: str) -> str:
    try:
        raw = call_llm(
            [{"role": "user", "content": REFINE_PROMPT.format(body=body[:4000])}],
            model=model,
            temperature=0.0,
            max_tokens=600,
            reasoning_effort="none",
        )
    except Exception:
        return body
    return raw.strip().lstrip("`").rstrip("`")


# SKILL §"date fallback same-day multi-source bundle repair":
# when a refined node is < 200 chars, look up that day's content_text /
# content_markdown / text_preview in timeline_articles.json and bundle the
# candidate sources for one more LLM refine pass.
def _load_date_fallback_index(articles_json: Optional[Path]) -> Dict[str, List[str]]:
    """date(YYYY-MM-DD) -> list of long-form article bodies for that day."""
    out: Dict[str, List[str]] = {}
    if not articles_json or not articles_json.exists():
        return out
    try:
        data = __import__("json").loads(articles_json.read_text(encoding="utf-8"))
    except Exception:
        return out
    items = data if isinstance(data, list) else data.get("articles", []) if isinstance(data, dict) else []
    for art in items:
        if not isinstance(art, dict):
            continue
        date = (art.get("date") or art.get("publish_date") or "")[:10]
        if not date:
            continue
        body = (art.get("content_text") or art.get("content_markdown")
                or art.get("text") or art.get("markdown") or art.get("text_preview") or "")
        body = str(body).strip()
        if len(body) < 80:
            continue
        out.setdefault(date, []).append(body[:3000])
    return out


def date_bundle_repair(short_body: str, date_str: str, fallback_idx: Dict[str, List[str]],
                       event_name: str, model: str) -> Tuple[str, bool]:
    """Try a small bundle repair: gather up to 3 same-day candidate bodies,
    ask LLM to weave a 200-300 char GT paragraph. Returns (new_body, used)."""
    bodies = fallback_idx.get(date_str, [])
    if not bodies:
        return short_body, False
    bundle = "\n\n---\n\n".join(bodies[:3])
    prompt = REFINE_PROMPT.format(body=(short_body + "\n\n[Full text candidate source on the same day]\n" + bundle)[:8000])
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=700,
            reasoning_effort="none",
        )
    except Exception:
        return short_body, False
    return raw.strip().lstrip("`").rstrip("`"), True


def main() -> None:
    p = argparse.ArgumentParser(description="merge-step6.5: GT refine for predict input (Round 1)")
    p.add_argument("merged_long_md")
    p.add_argument("output_dir")
    p.add_argument("--event-name", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-rounds", type=int, default=3,
                   help="SKILL §three-round loop: 1=R1 baseline only, 2=R1+R2 main-trunk, 3=R1+R2+R3 edge closing. "
                        "Default 3 — all three rounds run automatically via the same LLM API.")
    p.add_argument("--articles-json", default=None,
                   help="optional gt_workspace/timeline_articles.json for date-fallback bundle repair "
                        "(SKILL §\"when the _idx_web summary/preview cannot support 200-300 chars...\")")
    p.add_argument("--model", default=None)
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    model = args.model or cfg.get("default_llm_model")

    in_path = Path(args.merged_long_md)
    if not in_path.exists():
        raise SystemExit(f"missing {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "timeline_merged_long_gt.log"

    lines = in_path.read_text(encoding="utf-8").splitlines()

    # Parse into sections: stage > date > {event_body, opinion_body}
    sections: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    mode: Optional[str] = None
    cur_stage = ""
    for ln in lines:
        m_stage = STAGE_RE.match(ln.rstrip())
        m_date = DATE_RE.match(ln.strip())
        if m_stage and not ln.startswith("###"):
            cur_stage = m_stage.group(1)
            continue
        if m_date:
            if cur:
                sections.append(cur)
            cur = {"stage": cur_stage, "date": m_date.group(1), "event": [], "opinion": []}
            mode = None
            continue
        if cur is None:
            continue
        if EVENT_PREFIX.match(ln):
            mode = "event"
            cur["event"].append(EVENT_PREFIX.sub("", ln).strip())
        elif OPINION_PREFIX.match(ln):
            mode = "opinion"
            cur["opinion"].append(OPINION_PREFIX.sub("", ln).strip())
        elif mode and ln.strip():
            cur[mode].append(ln.strip())
    if cur:
        sections.append(cur)

    # SKILL §"date fallback": preload same-day article bodies if provided
    fallback_idx = _load_date_fallback_index(
        Path(args.articles_json) if args.articles_json else None
    )
    if fallback_idx:
        log(f"Loaded date-fallback index: {sum(len(v) for v in fallback_idx.values())} bodies "
            f"across {len(fallback_idx)} dates", log_path)

    log(f"Round 1 GT refine: {len(sections)} sections", log_path)

    report_nodes: List[Dict[str, Any]] = []
    rep_lock = threading.Lock()

    def process(sec: Dict[str, Any]) -> None:
        original = " ".join(sec["event"]).strip()
        if not original:
            with rep_lock:
                report_nodes.append({
                    "date": sec["date"], "stage": sec["stage"],
                    "selected_source": "merged_long_event",
                    "rewrite_mode": "drop",
                    "hard_fail_tags": ["empty_event"],
                    "drop_reasons": ["empty"],
                    "passed": False,
                })
            sec["refined_event"] = None
            return
        # Round 1: rewrite via LLM, check, repair once if hard-fail
        refined = refine_body(original, args.event_name, model)
        tags = hard_fail_tags(refined)
        mode_used = "primary"
        selected_source = "merged_long_event"
        if any(t.startswith(("gt_below_min", "gt_above_max", "bad_start", "title_shell")) for t in tags):
            refined = refine_body(
                refined + "\n\n(Please change the beginning of the sentence to the fact anchor, control the length to 200-300 words, and delete the title shell/header/link/signature fragment.）",
                args.event_name, model,
            )
            tags = hard_fail_tags(refined)
            mode_used = "repair"
            # SKILL §"date fallback": if still under-length and articles.json
            # is supplied, bundle same-day candidate bodies for one more pass.
            still_short = any(t.startswith("gt_below_min") for t in tags)
            if still_short and fallback_idx:
                bundled, used = date_bundle_repair(refined, sec["date"], fallback_idx,
                                                   args.event_name, model)
                if used:
                    refined = bundled
                    tags = hard_fail_tags(refined)
                    mode_used = "date_bundle_repair"
                    selected_source = "merged_long_event+timeline_articles_same_day"
        sec["refined_event"] = refined
        passed = not any(t.startswith(("title_shell", "ref_residue", "bad_start")) for t in tags) and \
                 not any(t.startswith("gt_below_min") for t in tags)
        with rep_lock:
            report_nodes.append({
                "date": sec["date"], "stage": sec["stage"],
                "selected_source": selected_source,
                "rewrite_mode": mode_used,
                "hard_fail_tags": tags,
                "passed": passed,
                "original_chars": len(re.sub(r"\s+", "", original)),
                "refined_chars": len(re.sub(r"\s+", "", refined)),
            })

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = [pool.submit(process, sec) for sec in sections]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                log(f"thread error: {e}", log_path)

    # SKILL §"three-round loop": Round 2 (main-trunk revision) + Round 3 (edge closing).
    # Run these automatically using the same LLM pipeline. Round 2 targets nodes
    # that still flag `title_shell` / `bad_start` / `gt_below_min` after Round 1.
    # Round 3 sweeps the remaining `passed=False` nodes for a milder rewrite.
    if args.max_rounds >= 2:
        log(f"Round 2 main-trunk revision starting (targeting high-priority failures)", log_path)
        round2_targets = [
            sec for sec, n in zip(sections, report_nodes)
            if not n.get("passed") and any(
                t.startswith(("title_shell", "bad_start", "gt_below_min"))
                for t in n.get("hard_fail_tags", [])
            ) and sec.get("refined_event")
        ]

        def round2_process(sec: Dict[str, Any]) -> None:
            curr = sec.get("refined_event") or ""
            # Stronger prompt: emphasize fact-anchor at sentence start + 200-300 chars
            redo = refine_body(
                curr + "\n\n(This is a high-impact node in the main chain. Please rewrite it completely: the sentence must begin with a factual anchor—who did what, and when.)"
                       "Delete any title shells, electric heads, link fragments, and quotation fragments; the final length is 200-300 words, and no less than 200 is allowed.）",
                args.event_name, model,
            )
            tags = hard_fail_tags(redo)
            sec["refined_event"] = redo
            sec["round2_applied"] = True
            sec["round2_tags"] = tags

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = [pool.submit(round2_process, sec) for sec in round2_targets]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"round 2 thread error: {e}", log_path)
        log(f"Round 2 done: revised {len(round2_targets)} nodes", log_path)

    if args.max_rounds >= 3:
        log(f"Round 3 edge closing starting (remaining failures)", log_path)
        # Recompute pass status after Round 2
        round3_targets = []
        for sec, n in zip(sections, report_nodes):
            if not sec.get("refined_event"):
                continue
            curr_tags = (sec.get("round2_tags")
                         if sec.get("round2_applied") else n.get("hard_fail_tags", []))
            still_failing = any(
                t.startswith(("title_shell", "ref_residue", "bad_start", "gt_below_min"))
                for t in curr_tags
            )
            if still_failing:
                round3_targets.append(sec)

        def round3_process(sec: Dict[str, Any]) -> None:
            curr = sec.get("refined_event") or ""
            # Milder prompt for the residual edge cases
            redo = refine_body(
                curr + "\n\n(This is a soft boundary node.Please do a minimum cleanup: remove the remaining title shell/quotation half sentence/link，"
                       "Keep the meaning of the facts unchanged, and finally fall to 200-300 words.）",
                args.event_name, model,
            )
            sec["refined_event"] = redo
            sec["round3_applied"] = True
            sec["round3_tags"] = hard_fail_tags(redo)

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futs = [pool.submit(round3_process, sec) for sec in round3_targets]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    log(f"round 3 thread error: {e}", log_path)
        log(f"Round 3 done: revised {len(round3_targets)} nodes", log_path)

    # Build refined GT markdown
    out_lines: List[str] = [f"# {args.event_name} — Long-form forecast GT（Round 1+2+3 refined）", ""]
    stage_seen: List[str] = []
    for sec in sections:
        if sec.get("refined_event") is None:
            continue
        if sec["stage"] and sec["stage"] not in stage_seen:
            stage_seen.append(sec["stage"])
            out_lines.append(f"## {sec['stage']}\n")
        out_lines.append(f"### {sec['date']}\n")
        out_lines.append(f"**event**：{sec['refined_event']}")
        out_lines.append("")
        if sec["opinion"]:
            out_lines.append(f"**Public opinion**：{' '.join(sec['opinion'])}")
            out_lines.append("")

    gt_md_path = out_dir / "timeline_merged_long_gt.md"
    gt_md_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    # Recompute pass status with R2/R3 tags taken into account
    for sec, node in zip(sections, report_nodes):
        if sec.get("round3_applied"):
            node["round3_tags"] = sec.get("round3_tags", [])
            node["rewrite_mode"] = "round3_edge_close"
            node["passed"] = not any(
                t.startswith(("title_shell", "ref_residue", "bad_start", "gt_below_min"))
                for t in node["round3_tags"]
            )
        elif sec.get("round2_applied"):
            node["round2_tags"] = sec.get("round2_tags", [])
            node["rewrite_mode"] = "round2_main_trunk"
            node["passed"] = not any(
                t.startswith(("title_shell", "ref_residue", "bad_start", "gt_below_min"))
                for t in node["round2_tags"]
            )

    report = {
        "event_name": args.event_name,
        "rounds_executed": args.max_rounds,
        "total_sections": len(sections),
        "kept": sum(1 for n in report_nodes if n.get("passed")),
        "failed": sum(1 for n in report_nodes if not n.get("passed")),
        "round2_revised": sum(1 for s in sections if s.get("round2_applied")),
        "round3_revised": sum(1 for s in sections if s.get("round3_applied")),
        "nodes": report_nodes,
    }
    (out_dir / "timeline_merged_long_gt_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Seed iteration log with Round 1 + TODOs for Rounds 2/3 + human approval
    log_md = out_dir / "timeline_merged_long_gt_iteration_log.md"
    log_lines: List[str] = []
    if log_md.exists():
        log_lines.append(log_md.read_text(encoding="utf-8").rstrip())
        log_lines.append("\n---\n")
    log_lines.append(f"# GT refine iteration log — {args.event_name}\n")
    log_lines.append(f"## Round 1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_lines.append(f"- baseline Automatic generation is complete：{len(sections)} nodes，{report['kept']} Passed the initial validation，{report['failed']} Still encountering a hard failure.")
    log_lines.append("- Before beginning the main Round 2 revisions, please manually spot-check the following hard-failure nodes: ")
    bad = [n for n in report_nodes if not n.get("passed")]
    for n in bad[:25]:
        log_lines.append(f"  - {n['date']} — tags={n['hard_fail_tags']}")
    if len(bad) > 25:
        log_lines.append(f"  - ...and {len(bad) - 25} more (see timeline_merged_long_gt_report.json)")
    log_lines.append("")
    if args.max_rounds >= 2:
        n_r2 = sum(1 for s in sections if s.get("round2_applied"))
        log_lines.append(f"## Round 2 — Main Revision (automatic, {n_r2} nodes rewrite)")
        log_lines.append("- Use the same LLM API to aggressively rewrite nodes from Round 1 whose `hard_fail_tags` contain `title_shell`, `bad_start`, or `gt_below_min`.")
        log_lines.append("")
    else:
        log_lines.append("## Round 2 — skip（--max-rounds < 2）\n")

    if args.max_rounds >= 3:
        n_r3 = sum(1 for s in sections if s.get("round3_applied"))
        log_lines.append(f"## Round 3 — Boundary Refinement (automatic, {n_r3} nodes lightly rewrite)")
        log_lines.append("- correct Round 2 After that, the soft boundary node that still hits the hard and fails is minimized and cleaned up.")
        log_lines.append("")
    else:
        log_lines.append("## Round 3 — skip（--max-rounds < 3）\n")

    log_lines.append("## Final conclusion (manual review section after the three automated rounds are completed)")
    log_lines.append("Final Manual Review Decision (required; otherwise, predict-pipeline Step -1 will reject the submission):")
    log_lines.append("-Candidate conclusion：")
    log_lines.append("  - `After manual review, you can enter predict`")
    log_lines.append("  - `Manual review failed, continue to revise`")
    log_lines.append("  - `Manual review failed, there is external hard blocking`")
    log_lines.append("")
    log_md.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "sections": len(sections),
        "passed": report["kept"],
        "failed": report["failed"],
        "gt_md": str(gt_md_path),
        "report_json": str(out_dir / "timeline_merged_long_gt_report.json"),
        "iteration_log": str(log_md),
        "note": "Round 1 only — Rounds 2/3 + Manual review must be completed manually before predict",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
