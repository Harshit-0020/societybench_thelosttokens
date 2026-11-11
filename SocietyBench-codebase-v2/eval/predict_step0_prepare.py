#!/usr/bin/env python3
"""predict-step0-prepare (1:1 of skills/predict-step0-prepare/SKILL.md).

Anonymizes the merged timeline through three phases:

  Phase 1: rule-based replacement (entities) + date shift
           - longest-match-first
           - hit-count statistics + zero-hit reporting
           - double-replacement repair (e.g. "导师导师J" → "导师J")
           - role-prefix double repair (auto-extracted from replacement targets)
           - multi-pass re-run on post-replacement text

  Phase 2: LLM semantic audit (one round per invocation; user iterates)
           - reads the anonymized text and reports "direct search keys" (high)
             and "replacement quality issues" (mid)
           - stops when high == 0 AND mid == 0, or when --max-audit-rounds hit
           - audit_report.json records each round

  Phase 3: LLM consistency check (compares original vs anonymized)
           - paragraph completeness, event completeness, semantic equivalence,
             accidental damage, format consistency

Usage:
    python predict_step0_prepare.py <timeline_md> <output_dir> \\
        --replacements <replacements.json> \\
        [--date-shift-max-days 180] [--date-shift-override <int>] [--seed N] \\
        [--skip-audit] [--max-audit-rounds 5] [--audit-only] \\
        [--audit-model kimi-k2.5] \\
        [--pipeline-config <path>]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import (
    apply_replacements,
    call_llm,
    load_pipeline_config,
    load_replacements,
    parse_date_obj,
)


# ===========================================================================
# Code hard gate (1) Simplified/Traditional variant coverage: each entity rule is
# auto-expanded to both Simplified and Traditional Chinese spellings → same
# placeholder. Prevents escapes where one variant has a rule but the other slips
# through. Uses opencc for general conversion when installed; otherwise falls
# back to a built-in Simplified↔Traditional char map for common entity-name
# characters (zero external dependency in release/deploy environments).
# ===========================================================================
try:  # general path
    from opencc import OpenCC  # type: ignore
    _CC_T2S, _CC_S2T = OpenCC("t2s"), OpenCC("s2t")
    def _to_simp(s: str) -> str: return _CC_T2S.convert(s)
    def _to_trad(s: str) -> str: return _CC_S2T.convert(s)
except Exception:  # fallback: covers common Simplified↔Traditional char pairs in Chinese entity names
    _S2T = {"兴": "興", "辉": "輝", "达": "達", "见": "見", "后": "後", "电": "電",
            "脑": "腦", "纳": "納", "马": "馬", "尔": "爾", "会": "會", "计": "計",
            "师": "師", "报": "報", "证": "證", "团": "團", "构": "構", "机": "機",
            "谋": "謀", "卖": "賣", "购": "購", "债": "債", "盘": "盤", "涨": "漲",
            "亿": "億", "万": "萬", "户": "戶", "财": "財", "务": "務", "国": "國",
            "权": "權", "学": "學", "员": "員", "长": "長", "张": "張", "刘": "劉",
            "陈": "陳", "杨": "楊", "韩": "韓", "贤": "賢", "见": "見"}
    _T2S = {v: k for k, v in _S2T.items()}
    def _to_simp(s: str) -> str: return "".join(_T2S.get(c, c) for c in s)
    def _to_trad(s: str) -> str: return "".join(_S2T.get(c, c) for c in s)


def _expand_variants(replacements: List[List[str]]) -> List[List[str]]:
    """Add Simplified/Traditional variants of each rule's original name (mapped to the
    same placeholder). Added variant rules that never match are harmless."""
    out: List[List[str]] = []
    seen = set()
    for pair in replacements:
        if not pair or not pair[0]:
            continue
        orig = str(pair[0])
        repl = str(pair[1]) if len(pair) > 1 else ""
        for variant in (orig, _to_simp(orig), _to_trad(orig)):
            key = (variant, repl)
            if variant and key not in seen:
                seen.add(key)
                out.append([variant, repl])
    return out


# ===========================================================================
# Phase 1: rule-based replacement (with hit counts + double-fix + multi-pass)
# ===========================================================================

def phase1_replace(
    text: str,
    replacements: List[List[str]],
) -> Tuple[str, Dict[str, Any]]:
    """Apply the replacement table to `text`, return (new_text, stats).

    Stats include per-rule hit counts, zero-hit rules, and the role-prefix
    fixes that were performed.
    """
    # Gate (1): expand each rule into Simplified/Traditional variants first.
    # NOT NEEDED SINCE, WE DON´T HAVE CHINESE TEXT ANYMORE
    # replacements = _expand_variants(replacements)


    # Gate (2): drop dangerous single-char keys. Chinese has no word boundaries, so a
    # single-char key matches as a substring and corrupts unrelated common words
    # (e.g. "加"→国家Z destroys 加征/加强/加拿大; "中"→国家X destroys 中国/其中/中美).
    # A single Latin/digit key ("X"→公司A) would also re-hit letters inside already
    # inserted placeholders (国家X → 国家公司A). Such keys are never safe: an entity
    # this short must be covered by multi-char rules (中国, 加拿大, X平台); anything
    # missed is caught by reaudit adding multi-char rules.
    # FIXME: drop very short strings [MAYNOT BE RELATED TO ENGLISH LANGUAGE]

    # All the stuff that was dropped for being too short
    dropped_short = sorted({old for old, _new in replacements if len(str(old).strip()) <= 1})
    # All of the remaining replacements
    reps = [(old, new) for old, new in replacements if len(str(old).strip()) >= 2]

    # Gate (3): close replacement values under the rule set — first fully resolve rules
    # whose value still contains a real name. Audits often append two kinds of
    # half-finished rules: (1) a compound rule anonymized only halfway (the value still
    # holds a real name); (2) a two-level chain (alias → real name, then real name →
    # placeholder). The old multi-pass replacement finished these via re-scanning, but
    # re-scanning was exactly the source of chained placeholder pollution. With
    # single-pass replacement, each value must first be resolved through the other rules
    # until it contains no key, so one pass emits fully anonymized text.
    _kv: Dict[str, str] = {}
    for _o, _n in reps:
        _kv.setdefault(_o, _n)
    # Key value pairs for reps, with longest keys first
    _keys_long_first = sorted(_kv, key=len, reverse=True)


    def _resolve(val: str, own: str) -> str:
        for _ in range(10):  # bounded fixed point, guards against self-reference/cycles
            changed = False
            for k in _keys_long_first:
                if k != own and k in val:
                    # longest key contained within another is replaced first
                    nv = val.replace(k, _kv[k])
                    if nv != val:
                        val, changed = nv, True
            if not changed:
                break
        return val

    # create the replacements over here
    reps = [(old, _resolve(new, old)) for old, new in reps]

    # Longest-first: at each position take the longest key matching there. Bucket by first char for speed.
    sorted_reps = sorted(reps, key=lambda r: len(r[0]), reverse=True)
    by_first: Dict[str, List[Any]] = defaultdict(list)
    for old, new in sorted_reps:
        by_first[old[0]].append((old, new))

    # --- Single pass, longest match, scans the original text only (never re-scans replaced text)
    # Scan the original left to right: on a hit, emit the placeholder and skip the
    # matched length of the ORIGINAL text; emitted placeholders are never scanned again
    # → no chained stacking (an inserted "国家X" cannot be eaten by a later "X→…" rule)
    # and no substring damage (single-char keys were dropped in gate (2)). The old
    # doubling/prefix/second-pass fallbacks (all cleanup for dirt produced by
    # sequential replacement) are therefore no longer needed.
    counts: Dict[str, int] = defaultdict(int)
    out_parts: List[str] = []
    i, n = 0, len(text)
    while i < n:
        hit = None
        for old, new in by_first.get(text[i], ()):  # same first char, already longest-first
            if text.startswith(old, i):
                hit = (old, new)
                break
        if hit:
            old, new = hit
            out_parts.append(new)
            counts[old] += 1
            i += len(old)
        else:
            out_parts.append(text[i])
            i += 1
    out = "".join(out_parts)

    zero_hit = [old for old, _new in sorted_reps if old not in counts]

    stats = {
        "n_rules": len(sorted_reps),
        "n_rules_with_hits": sum(1 for v in counts.values() if v > 0),
        "total_replacements": sum(counts.values()),
        "hit_counts": dict(counts),
        "zero_hit_rules": zero_hit,
        "dropped_short_keys": dropped_short,
        "double_fixes": {},
        "role_prefix_fixes": {},
        "pass2_hits": {},
    }
    
    return out, stats

# ===========================================================================
# Date shifting (idempotent on YYYY-MM-DD and YYYY年MM月DD日 forms)
# ===========================================================================

def shift_dates(text: str, delta_days: int) -> str:
    if delta_days == 0:
        return text

    # Year-less "M月D日" dates (common in body text) need a reference year for the delta:
    # take the first 4-digit year in the text, default 2026.
    
    # FIXME: Change the dates RE later. Something than can look for dates ?
    # _ym = re.search(r"(\d{4})\s*[-年]", text)
    _ym = re.search(r"(?<!\d)(\d{4})(?!\d)", text)
    ref_year = int(_ym.group(1)) if _ym else 2026

    shifted_ranges: List[str] = []

    def _stash_shifted_range(value: str) -> str:
        shifted_ranges.append(value)
        return f"__SHIFTED_DATE_RANGE_{len(shifted_ranges) - 1}__"

    def _month_after(year: int, month: int) -> Tuple[int, int]:
        return (year + 1, 1) if month == 12 else (year, month + 1)

    def _shift_en_range(m: re.Match) -> str:
        year_s, mon_s, day_s, sep, end_day_s = m.groups()
        year = int(year_s) if year_s else ref_year
        mon = int(mon_s)
        day = int(day_s)
        end_day = int(end_day_s)
        sep = "->"

        try:
            start = date(year, mon, day)
        except ValueError:
            return m.group(0)
        end_year, end_mon = year, mon

        if end_day < day:
            end_year, end_mon = _month_after(year, mon)
        try:
            end = date(end_year, end_mon, end_day)
        except ValueError:
            return m.group(0)
        start2 = start + timedelta(days=delta_days)
        end2 = end + timedelta(days=delta_days)
        if year_s:
            if start2.year == end2.year:
                if start2.month == end2.month:
                    out = f"{start2.year}-{start2.month}-{start2.day}{sep}{end2.day}"
                else:
                    out = f"{start2.year}-{start2.month}-{start2.day}{sep}{end2.month}-{end2.day}"
            else:
                out = f"{start2.year}-{start2.month}-{start2.day}{sep}{end2.year}-{end2.month}-{end2.day}"
        elif start2.month == end2.month:
            out = f"{start2.month}-{start2.day}{sep}{end2.day}"
        else:
            out = f"{start2.month}-{start2.day}{sep}{end2.month}-{end2.day}"
        return _stash_shifted_range(out)

    def _shift_cn_range(m: re.Match) -> str:
        mon_s, day_s,end_day_s, year_s = m.groups()
        sep = "-"
        year = int(year_s) if year_s else ref_year
        mon = int(mon_s)
        day = int(day_s)
        end_day = int(end_day_s)
        try:
            start = date(year, mon, day)
        except ValueError:
            return m.group(0)
        end_year, end_mon = year, mon
        if end_day < day:
            end_year, end_mon = _month_after(year, mon)
        try:
            end = date(end_year, end_mon, end_day)
        except ValueError:
            return m.group(0)
        start2 = start + timedelta(days=delta_days)
        end2 = end + timedelta(days=delta_days)
        if year_s:
            if start2.year == end2.year:
                if start2.month == end2.month:
                    out = f"{start2.year}年{start2.month}月{start2.day}日{sep}{end2.day}日"
                else:
                    out = f"{start2.year}年{start2.month}月{start2.day}日{sep}{end2.month}月{end2.day}日"
            else:
                out = f"{start2.year}年{start2.month}月{start2.day}日{sep}{end2.year}年{end2.month}月{end2.day}日"
        elif start2.month == end2.month:
            out = f"{start2.month}月{start2.day}日{sep}{end2.day}日"
        else:
            out = f"{start2.month}月{start2.day}日{sep}{end2.month}月{end2.day}日"
        return _stash_shifted_range(out)

    def _shift_iso(m: re.Match) -> str:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return m.group(0).replace(
            f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
            d2.isoformat(),
        )

    def _shift_cn(m: re.Match) -> str:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.year}年{d2.month}月{d2.day}日"
    
    def _shift_en(m: re.Match) -> str:
        try:
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.year}-{d2.month}-{d2.day}"
    
    def _shift_cn_md(m: re.Match) -> str:  # year-less "M月D日" → shift in sync, output "M月D日" only
        try:
            d = date(ref_year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.month}月{d2.day}日"

    def _shift_en_md(m: re.Match) -> str:  # year-less "M月D日" → shift in sync, output "M月D日" only
        try:
            d = date(ref_year, int(m.group(2)), int(m.group(1)))
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.month} {d2.day}"

    def _shift_relative_cn_md(m: re.Match) -> str:
        prefix, mon_s, day_s = m.groups()
        try:
            d = date(ref_year, int(mon_s), int(day_s))
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{prefix}{d2.month}月{d2.day}日"

    def _shift_relative_en_md(m: re.Match) -> str:
        prefix, mon_s, day_s = m.groups()
        try:
            d = date(ref_year, int(mon_s), int(day_s))
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{prefix}{d2.month}-{d2.day}"

    def _shift_cn_ym(m: re.Match) -> str:
        try:
            d = date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.year}年{d2.month}月"

    def _shift_en_ym(m: re.Match) -> str:
        try:
            "date needs: YEAR, MONTH, DAY"
            d = date(int(m.group(2)), int(m.group(1)), 1)
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.year} {d2.month}"

    def _shift_cn_month(m: re.Match) -> str:
        try:
            d = date(ref_year, int(m.group(1)), 1)
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.month}月"

    def _shift_en_month(m: re.Match) -> str:
        try:
            d = date(ref_year, int(m.group(1)), 1)
        except ValueError:
            return m.group(0)
        d2 = d + timedelta(days=delta_days)
        return f"{d2.month}"

    # Handle bare-end-day ranges like "1月17日至18日 / 5月9到12日" first and stash the
    # results, so later single-date rules neither shift only the start date nor
    # re-shift an already shifted range.

    # CHINESE RE : 
    # This RE will not be used, since we´re not using chinese news. We will not encounter chinese dates over there. 
    # These RE will not match
    # text = re.sub(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日?\s*(至|到|[-—~])\s*(\d{1,2})日", _shift_cn_range, text)

    # SUBS(DATE RANGES): Matching following format : March 15–20, 2025 & 15–20 March 2025
    RE_EN_FM1 = r"(?:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?(\d{1,2})(?:st|nd|rd|th)?\s*(?:-|–|—|~|to|through)\s*(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?"
    RE_EN_FM2 = r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:-|–|—|~|to|through)\s*(\d{1,2})(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,?\s*(\d{4}))?"

    text = re.sub(RE_EN_FM1, _shift_en_range, text)
    text = re.sub(RE_EN_FM2, _shift_en_range, text)

    # RE for ISO date format. This ISO format will match
    text = re.sub(r"(\d{4})-(\d{2})-(\d{2})", _shift_iso, text)

    # RE for chinese date format. These chinese dates RE will not match.
    # """This Chinese regex matches a full date consisting of year, month, and day, such as 2025年3月15日"""
    # text = re.sub(r"(\d{4})年(\d{1,2})月(\d{1,2})日", _shift_cn, text)

    "Format A: Month Day, Year, Format B: Day Month Year"
    RE2_EN_FM1 = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})"
    RE2_EN_FM2 = r"(\d{1,2})(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})"
    text = re.sub(RE2_EN_FM1, _shift_en, text)
    text = re.sub(RE_EN_FM2, _shift_en, text)
    

    # RE for (SAMEYEAR)MM/DD format. These chinese dates RE will also not match.
    # This Chinese regex matches dates where the year is expressed relatively as "the current year" or "the same year", followed by a month and day."
    # text = re.sub(r"(当年|同年)(\d{1,2})月(\d{1,2})日", _shift_relative_cn_md, text)
    RE3_EN_FM1 = r"(?:(?:in|later in|earlier in|during)\s+)?(that year|the same year),?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?"
    RE3_EN_FM2 = r"(?:on\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(that year|the same year)"

    text = re.sub(RE3_EN_FM1, _shift_relative_en_md, text)
    text = re.sub(RE3_EN_FM2, _shift_relative_en_md, text)


    # RE for YYYY/MM/DD format. These chinese RE will not match
    "This Chinese regex matches a year and month expression, without requiring a day, while excluding certain unwanted continuations."
    # text = re.sub(r"(\d{4})年(\d{1,2})月(?!\d|日|个)", _shift_cn_ym, text)

    RE4_EN_FM1 = r"(?:(?:in|during)\s+)?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec),?\s+(?:of\s+)?(\d{4})(?!\d)"
    text = re.sub(RE4_EN_FM1, _shift_en_ym, text)

    # RE for 
    # Year-less "M月D日" in body text must shift in sync too; n
    # Negative lookbehinds skip the already handled "…年M月D日"

    """
    Matches only MM\DD format
    This is a Regular Expression (Regex) used to match dates
    written in Japanese or Chinese-style month/day format, 
    such as 3月15日 or 12月25日."""
    """
    ?<!\d) -> Negative lookbehind: The preceding character must NOT be a digit (0–9).
    (?<!年) -> Negative lookbehind: The preceding character must NOT be 年 (year character).
    """

    "This Chinese regex matches a month-day date without an explicitly specified year, such as 3月15日 or 12月25日."
    # text = re.sub(r"(?<!\d)(?<!年)(\d{1,2})月(\d{1,2})日", _shift_cn_md, text)
    "MONTH/DAY Pattern"
    RE5_EN_FM1 = r"(?<!\d)(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?!\d)"
    "DAY/MONTH Pattern"
    RE5_EN_FM2 = r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(?![a-z])"
    text = re.sub(RE5_EN_FM1, _shift_en_md, text)


    
    """
    This is a Regular Expression (Regex) designed to match a month expression
    in Chinese/Japanese text, such as 3月 or 12月, while avoiding certain unwanted matches.

    REGEX : (?<!\d)(?<!年)(\d{1,2})月(?!\d|日|个)

    1. Explanation of each part
    Meaning

    (?<!\d) : Negative lookbehind: Previous character must not be a digit.
    (?<!年) : Negative lookbehind: Previous character must not be 年 (year).
    (\d{1,2}) : Captures 1 or 2 digits (month number).
    月 : Matches the literal character 月 (month).
    (?!\d\|日\|个) : Negative lookahead: Next character must not be a digit, 日, or 个.
    """
    # text = re.sub(r"(?<!\d)(?<!年)(\d{1,2})月(?!\d|日|个)", _shift_cn_month, text)

    RE6_EN_FM1 = r"(?<![a-z0-9])(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(?![a-z]|\s+\d{1,2}(?:st|nd|rd|th)?\b)"
    text = re.sub(RE6_EN_FM1, _shift_en_month ,text)

    for idx, value in enumerate(shifted_ranges):
        text = text.replace(f"__SHIFTED_DATE_RANGE_{idx}__", value)
    return text

# ===========================================================================
# Phase 2: LLM semantic audit
# ===========================================================================

AUDIT_PROMPT = """你是真实实体识别员。下面是一段**已部分匿名化**的事件时间线（部分真实实体已被替换为"公司A""高管A"等占位符）。

**不要判断"是否合格"。请逐段穷举式地列出文中所有"仍是真实名称、能用于识别真实事件或当事方"的专有名词**（这是关键：列举，不是判断）：
- 真实人名：任何具体个人（高管/董事/创始人/CFO/合规官/分析师/律师/政商人物…）
- 真实机构名：公司/律所/投行/券商/审计所/基金/政府或监管机构/媒体…
- 真实产品/型号/项目/平台名、真实地名、社媒账号、可搜索的真实标题、能搜到的原话引语
- 残留的简称/别名/繁简体变体；同一真实实体被替换成了不同占位符的情况

为每个真实名分配一个**类别化匿名占位符**：人名→高管B/高管C…或当事人B…；公司→公司B/公司C…；律所→律所A/律所B…；投行券商→投行A…；分析师/媒体机构→机构C…；基金→基金A…；产品/型号→产品A…；地名→地点A…。
⚠️ **已经是占位符的（公司A/高管A/机构B/审计机构A…）一律跳过，不要列。** 技术问题（双重替换/乱码/不通顺）也一并列出（suggestion 给修正写法）。
⚠️ **防漏检（重要）**：本文是真实事件时间线，几乎必然含有多个未匿名的真实**地名（国家/首都/城市/海峡/海域/地区/基地）**、人名、机构名、舰船名。请**逐段从头扫到尾、宁可多列不可漏**；除非你已逐段确认一个真实专名都不剩，否则**不要返回空列表**。

🔒 **铁律——只匿"身份"，保留"事实数字"**：你只列**真实专名/身份**（人名、公司/机构/律所/投行/基金、产品型号专名、真实地名、账号、可搜索标题/原话）。
**绝对不要列、不要动**以下（它们是预测信号，抹掉=信息缺失）：**数字、百分比(如 78%、+178%)、金额(如 400美元/59亿)、股价与涨跌幅、财务/业绩数据、具体日期(整体平移已在别处统一做、单条日期不要替换)、事实经过与因果**。
例：安永→审计机构A ✅；把"暴跌78%"列出来要求匿成"暴跌某跌幅" ❌（这是信息缺失，禁止）。

================ 待审核文本 ================
{body}
================

{prior_findings_block}

**输出 JSON**（无额外文字、无 markdown）：
{{
  "leaks": [
    {{"snippet": "真实原名", "suggestion": "占位符", "severity": "high"}}
  ],
  "stats": {{"high": <列出的真实实体数>, "mid": <技术问题数>}}
}}
若确已完全匿名干净（再也找不到任何真实专名），返回 {{"leaks": [], "stats": {{"high": 0, "mid": 0}}}}。

如果没有问题，返回 {{"leaks": [], "stats": {{"high": 0, "mid": 0}}}}。
"""


def _build_prior_findings_block(prior_findings: List[Dict[str, Any]]) -> str:
    if not prior_findings:
        return ""
    lines = ["", "**上一轮发现的问题（已修复，请重点确认是否修复干净，并检查是否引入新问题）**："]
    for it in prior_findings:
        sev = it.get("severity", "?")
        snip = (it.get("snippet") or "")[:200]
        lines.append(f"- [{sev}] {snip}")
    lines.append("")
    return "\n".join(lines)


def _parse_audit_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # tolerate prose before/after the JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


# Lens A — categorized check (scan type by type; restores the original scheme plus the added "participants of this event")
AUDIT_PROMPT_A = """你是匿名化审核员（**分类视角 A**）。下面是一段**已部分匿名化**的事件时间线（真实实体部分已替换为"公司A""高管A"等占位符）。
请**按以下类型逐项扫描**，列出**原文中仍残留、未被替换的真实内容**（已是占位符的一律跳过）：
- 真实人名：任何具体个人（高管/董事/创始人/CFO/合规官/分析师/律师/政商人物…）
- 真实机构名：公司/律所/投行/券商/审计所/基金/媒体/政府或监管机构
- 真实地名、真实文章/文件标题、可搜索的直接引语、可反算的周期数字（N周年→建立年）、事件特有的网络梗/谐音
- **本事件的具体参与方（重点·最常漏）**
- 占位符不一致（同一实体多个占位符 / 残留简称或繁简体变体）、替换技术问题（双重替换/乱码/不通顺）
为每项给类别化占位符建议（人名→高管B…；公司→公司B…；律所→律所A…；投行→投行A…；分析师/媒体→机构C…；基金→基金A…；产品→产品A…）。

🔒 **铁律——只匿"身份"，保留"事实数字"**：只列**真实专名/身份**。
**绝对不要列**：数字、百分比(如 78%、+178%)、金额(400美元/59亿)、股价与涨跌幅、财务/业绩数据、具体日期(整体平移已统一做)、事实经过——它们是预测信号，抹掉=信息缺失。例：安永→审计机构A ✅；"暴跌78%"→"某跌幅" ❌。

================ 待审核文本 ================
{body}
================
{prior_findings_block}
**输出 JSON**（无额外文字）：{{"leaks":[{{"snippet":"真实片段","suggestion":"占位符","severity":"high|mid"}}],"stats":{{"high":<真实残留项数>,"mid":<技术问题数>}}}}
全部干净则 {{"leaks":[],"stats":{{"high":0,"mid":0}}}}。"""

# Lens B — open-ended quality judgment (read the whole text: is the quality acceptable, and if not, what to fix)
AUDIT_PROMPT_B = """你是匿名化审核员（**整体视角 B**）。下面是一段**已部分匿名化**的事件时间线。
请**整体读一遍、自由判断**：**这版匿名化质量行不行？读完还能不能识别出这是哪个真实事件、牵涉哪些真实当事方？**
- 若**不合格**：指出**具体在哪些地方还没匿干净**（残留了什么真实**专名/身份**、或哪段上下文拼起来仍能认出真实事件），并给**改进建议**——可以是替换成占位符、也可以是泛化/改写/删除。
- 已经是占位符的（公司A/高管A…）不算问题。

🔒 **铁律——只匿"身份"，不要碰"事实数字"**：你判"能不能认出真实事件"时，**靠的是真实专名/身份**，不是数字。
**绝对不要建议动**：数字、百分比(如 78%、+178%)、金额(400美元/59亿)、股价与涨跌幅、财务/业绩数据、具体日期(整体平移已在别处统一做)、事实经过与因果——**这些是预测信号，要原样保留；把它们匿成"某跌幅/某价格/某月份"是信息缺失，不合格**。质量判断只针对"身份是否还认得出"，不针对数字。

================ 待审核文本 ================
{body}
================
{prior_findings_block}
**输出 JSON**（无额外文字）：{{"qualified":<true|false>,"leaks":[{{"snippet":"问题片段/残留真实名","suggestion":"改进建议(替换给占位符；泛化/删除给目标词或空串)","severity":"high"}}],"stats":{{"high":<问题数>,"mid":0}}}}
完全合格则 {{"qualified":true,"leaks":[],"stats":{{"high":0,"mid":0}}}}。"""

# Lens C = the existing AUDIT_PROMPT (exhaustive enumeration-style extraction)
AUDIT_LENSES = [("A", AUDIT_PROMPT_A), ("B", AUDIT_PROMPT_B), ("C", AUDIT_PROMPT)]


def _run_one_lens(name: str, prompt_tmpl: str, anon_text: str, prior_block: str, model: str) -> Dict[str, Any]:
    # A single-lens failure (LLM 400/timeout/parse error) must never sink the multi-lens
    # audit — the whole point of multi-lens redundancy is this fallback.
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt_tmpl.format(body=anon_text, prior_findings_block=prior_block)}],
            model=model, temperature=0.0, max_tokens=4000,
        )
        result = _parse_audit_json(raw)
        if not isinstance(result, dict):
            raise ValueError("non-dict")
    except Exception as e:
        return {"lens": name, "leaks": [], "stats": {"high": 0, "mid": 0}, "_error": str(e)[:120]}
    leaks = result.get("leaks") if isinstance(result.get("leaks"), list) else []
    return {"lens": name, "leaks": leaks, "qualified": result.get("qualified")}


# Iron rule (code hard gate): pure number/percentage/amount/stock-price/date snippets
# carry factual content — never counted as leaks, never turned into rules.
# If stripping all such tokens leaves nothing, the snippet is just a numeric fact → protect it.
_FACT_STRIP_RE = re.compile(
    r"[\d.,，、:：%％\s~\-—()（）]|美元|美金|港元|人民币|元|亿|万|千|百|股|倍|个百分点|"
    r"基点|bps|点|年|月|日|周[一二三四五六日天]|星期[一二三四五六日天]|季度|财年|"
    r"约|逾|超|近|达|暴跌|大跌|重挫|下跌|跌|暴涨|大涨|飙升|上涨|涨|创|刷新|"
    r"新低|新高|历史高点|历史低点|高点|低点|盘前|盘后|收盘|开盘"
)


def _is_pure_fact(s: str) -> bool:
    return len(_FACT_STRIP_RE.sub("", s or "").strip()) == 0


def run_audit_round(
    anon_text: str,
    prior_findings: List[Dict[str, Any]],
    *,
    model: str,
    raw_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Multi-lens fusion: run A (category) + B (quality) + C (enumeration) in parallel,
    UNION-dedup; only leaks that pass the code-level verification gate count as real."""
    from concurrent.futures import ThreadPoolExecutor
    prior_block = _build_prior_findings_block(prior_findings)
    with ThreadPoolExecutor(max_workers=3) as pool:
        lens_results = list(pool.map(
            lambda nb: _run_one_lens(nb[0], nb[1], anon_text, prior_block, model),
            AUDIT_LENSES,
        ))
    # UNION + dedup by snippet (keep the first suggestion)
    merged: List[Dict[str, Any]] = []
    seen = set()
    per_lens = {}
    for lr in lens_results:
        per_lens[lr["lens"]] = len(lr.get("leaks") or [])
        for it in (lr.get("leaks") or []):
            if not isinstance(it, dict):
                continue
            snip = str(it.get("snippet") or "").strip()
            if not snip or snip.lower() in seen:
                continue
            seen.add(snip.lower())
            merged.append({"snippet": snip,
                           "suggestion": it.get("suggestion"),
                           "severity": (it.get("severity") or "high"),
                           "_lens": lr["lens"]})
    # Code-level verification gate: a leak counts as real only if it is "in the current
    # anonymized text AND in the original text AND not a pure numeric fact".
    #   - not in the anon text → auditor hallucination / re-listing;
    #   - in the anon text but not in the original → it is a placeholder (the auditor
    #     anonymizing its own output);
    #   - pure number/percentage/amount/date → iron-rule protected, never anonymized.
    # → kills auditor noise so convergence is anchored on the real text.
    def _is_real(it: Dict[str, Any]) -> bool:
        snip = str(it.get("snippet") or "").strip()
        # Iron rule: contains digits (percentage/amount/stock price/date) or only fact
        # words → factual content, protect, never anonymize. Real entity names contain
        # no digits, so this gate only blocks facts and never lets an entity through.
        if not snip or _is_pure_fact(snip) or re.search(r"\d", snip):
            return False
        if snip not in anon_text:
            return False
        if raw_text is not None and snip not in raw_text:
            return False
        return True

    verified = [it for it in merged if _is_real(it)]
    high = sum(1 for it in verified if (it.get("severity") or "").lower() != "mid")
    mid = len(verified) - high
    return {"leaks": verified, "stats": {"high": high, "mid": mid},
            "raw_union": len(merged), "verified": len(verified),
            "per_lens_found": per_lens,
            "qualified_B": next((lr.get("qualified") for lr in lens_results if lr["lens"] == "B"), None)}


# ===========================================================================
# Phase 2 iteration: constrained rewrite that addresses reported leaks
# ===========================================================================

AUDIT_FIX_PROMPT = """你是匿名化规则补充员。审核发现以下问题（残留搜索键或质量瑕疵）。
请把每个问题转化为一条 **[原文片段, 替换占位符]** 规则，下一轮会把这些规则加入
replacements.json 后**从原文重新执行 Phase 1 替换**，不允许直接改文本。

**严格规则**：
- 只针对"直接搜索键"和"替换质量问题"两类问题输出规则
- 原文片段必须是文本中真实出现的、可被 str.replace() 匹配的字符串
- 替换占位符要保留信息结构（领域 / 方向 / 类型），不必保留原措辞
- "替换质量问题"的修法是给出 [bad_pattern → good_pattern]（如 "导师导师J" → "导师J"）
- 不要输出注释、不要输出 markdown 代码块、不要解释

================ 上一轮发现的问题 ================
{issues_block}
================

================ 当前匿名化文本（仅供定位）================
{body}
================

只输出 JSON 数组，每个元素 `[原文片段, 替换占位符]`：
[["原文片段1", "替换占位符1"], ["原文片段2", "替换占位符2"], ...]
如果没有可程序化处理的规则，输出 `[]`。
"""


DERIVE_PH_PROMPT = """下面是一份匿名文本里仍残留的「疑似真实专有名词」清单。请为每一个分配一个**类别化匿名占位符 = 类型 + 代号**（保留类型、隐去身份）。

代号规则（类型 + 字母代号）：
- 国家→国家X/国家Y/国家Z…；首都/城市→城市A/城市B…；海峡/海/湾/港→海峡A/海域A…；地区/州/省/岛/基地→地区A/基地A…
- 人物：领导人→领导人A/领导人B…；部长/官员→官员A/官员B…；将领/司令→将领A…；顾问→顾问A…；发言人→发言人A…
- 机构/部门→机构A…；军队/部队→部队A…；舰船→舰A…；公司→公司A…；媒体→媒体A…
- **同类的不同实体必须用不同代号**（两个不同国家不能都叫"某国"，要给不同代号 国家X / 国家Y）；
- **同一个真实实体（含不同写法/简称/子串）→ 必须用同一个代号**；
- 占位符不含原文真名、不含连续英文串、不含多位数字。
- ⚠️ **不要照单全收**：若某项其实不是需匿名的专有名词（通用词、泛称、已是占位符如"国家X/城市A"、纯职务名）→ 跳过、不输出。
- 🔒 数字/百分比/金额/日期一律不处理。

**已有映射（真名 → 已分配代号）。两条铁则**：
① 给**新**实体分配代号时，避开这里已用过的代号（顺延下一个未用的）；
② 若本次清单里的真名其实是下面某个已映射实体的**全称/简称/变体/子串**（例：已映射"某海峡全称→海峡A"，本次又出现它的简称），**必须复用它的同一代号**，绝不另起新码。
{existing_map}

只输出 JSON 数组（无额外文字）：[["真实专名","占位符"], ...]；跳过的不要出现。

清单：
{names_block}
"""


def derive_replacement_rules(anon_text: str, leaks: List[Dict[str, Any]],
                             *, model: str, existing_rules=None) -> List[List[str]]:
    """Hand the gate-verified "clean real proper-name snippets" to the LLM to generate
    type+code placeholders. Lens suggestions (especially B's) are not trusted directly —
    the LLM regenerates them all. The current replacement table (existing_rules,
    real name → code) is passed in so the LLM both avoids codes already in use (new
    entities get the next free code) and reuses the same code for variants/abbreviations,
    keeping the whole document and all rounds consistent. The LLM also filters out
    non-proper-names (nothing is accepted wholesale); placeholders are then locally
    validated. Failures never crash."""
    names: List[str] = []
    seen = set()
    for it in leaks:
        s = str(it.get("snippet") or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            names.append(s[:80])
    if not names:
        return []
    # Existing mapping (real name → code): the tail (codes newly added by audits) is the
    # most relevant; cap at 120 entries to keep the prompt from growing too large.
    pairs = [p for p in (existing_rules or [])
             if isinstance(p, (list, tuple)) and len(p) >= 2 and str(p[1]).strip()]
    map_lines = [f"- {str(p[0]).strip()} → {str(p[1]).strip()}" for p in pairs]
    existing_map = "\n".join(map_lines[-120:]) if map_lines else "（暂无）"
    prompt = DERIVE_PH_PROMPT.format(
        names_block="\n".join(f"- {n}" for n in names),
        existing_map=existing_map,
    )
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=model, temperature=0.0, max_tokens=2000,
        )
    except Exception:
        return []
    raw = raw.strip().lstrip("`").rstrip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    out: List[List[str]] = []
    for it in data:
        if not (isinstance(it, list) and len(it) == 2):
            continue
        bad, good = str(it[0]).strip(), str(it[1]).strip()
        # Locally validate the placeholder: non-empty, differs from the original, not too
        # long, no English runs / multi-digit numbers, does not contain the original name
        if not (bad and good) or bad == good:
            continue
        if len(good) > max(10, len(bad)):
            continue
        if re.search(r"[A-Za-z]{2,}|\d{2,}", good):
            continue
        if bad.lower() in good.lower():
            continue
        out.append([bad, good])
    return out


# ===========================================================================
# Phase 3: LLM consistency check (original vs anonymized)
# ===========================================================================

CONSISTENCY_PROMPT = """你是信息完整度校验员。请对比以下两个文本，检查匿名化版本是否**完整保留了原文的信息**。

⚠️ **重要前提（这些是有意的匿名化处理，绝对不算损伤/不一致，不要因此判 inconsistent）**：
- **真实实体被替换为占位符**（如 超微电脑→公司A、安永→审计机构A）——这是匿名化目的本身。
- **所有日期被整体平移了固定天数**（如全部 -162 天）——这是防止靠日期识别事件的有意处理。**因此请按"节点出现的先后顺序 + 事件内容"来对应原文与匿名版，不要按日期数值对应**；日期值不同是正常的，节点之间的相对间隔与因果顺序保持即可。

只检查"信息有没有真的丢失或被破坏"：

1. **⭐数字/事实保真（最重要·重点查）**：原文里的**数字、百分比、金额、股价涨跌幅、财务/业绩数据、具体日期**，在匿名版里是否**被错误地抹成了占位符**（如 "暴跌78%"→"暴跌某跌幅"、"400美元"→"某价格"、"2024-08-28"→"某日期/某月份"）？**这类数字/事实有明确信息指向、是预测信号，被匿掉就是严重信息缺失**，必须逐条在 damages 报告。（注意：日期被**整体平移**成另一个具体日期是有意的、不算缺失；但被抹成"某月份"这种占位符算缺失。）
2. **节点数与顺序**：匿名版的日期节点数量是否≈原文？节点先后顺序是否一致？
3. **事件完整性**：每个节点的"事件/舆论"段落是否都保留？有没有**整段事实被删掉**？
4. **语义一致性**：每段是否仍在讲原文对应的同一件事？（实体换成占位符可以，但事件经过、数字、因果不应被改变或丢失）
5. **格式**：阶段(##)/节点(###)/分隔线 结构是否完好。

**输出格式必须为 JSON**（不要带任何额外文字、不要 markdown 代码块）：
{{
  "node_count_original": <int>,
  "node_count_anon": <int>,
  "missing_nodes": ["YYYY-MM-DD", ...],
  "damages": [
    {{"location": "位置", "issue": "类型(段落缺失/事件改变/格式不一致/意外删除)", "detail": "原文 vs 匿名化版本的差异"}}
  ],
  "verdict": "consistent|inconsistent",
  "summary": "一句话总结"
}}
"""


def run_consistency_check(original: str, anon: str, *, model: str) -> Dict[str, Any]:
    # The LLM completeness check is advisory only (the authoritative verdict comes from the
    # deterministic number gate); call/parse failures must not crash — return unknown.
    prompt = CONSISTENCY_PROMPT.format(original=original, anon=anon)
    try:
        raw = call_llm(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=4000,
        )
        return _parse_audit_json(raw)
    except Exception as e:
        return {
            "verdict": "unknown",
            "_error": str(e)[:120],
        }


# Numeric-fact tokens: a value plus a unit (percentage/currency/scale word/shares/multiple/points).
# Bare dates (no unit) are excluded — dates are intentionally shifted.
# _FACT_NUM_RE = re.compile(
#     r"[0-9][0-9.,]*\s*(?:%|％|个百分点|个基点|美元|美金|港元|港币|人民币|欧元|日元|"
#     r"元|亿|万|千|股|倍|bps|点)"
# )

_FACT_NUM_EN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:"
    # Currency prefix: $5 million, €10 billion
    r"[$€£¥]\s*[0-9][0-9,.]*\s*"
    r"(?:thousand|million|billion|trillion|k|m|bn)?"
    r"|"
    # Number followed by unit: 15%, 200 bps, 5 million shares
    r"[0-9][0-9,.]*\s*"
    r"(?:"
    r"%|percent(?:age)?(?:\s+points?)?|pp|"
    r"basis\s+points?|bps|bp|"
    r"USD|EUR|GBP|JPY|CNY|INR|"
    r"dollars?|euros?|pounds?|yen|"
    r"thousand|million|billion|trillion|k|m|bn|"
    r"shares?|stocks?|times?|x|points?|pts"
    r")"
    r")"
    r"(?![A-Za-z])",
    re.IGNORECASE
)
def number_preservation_check(original: str, anon: str) -> Dict[str, Any]:
    """Code hard gate (2), deterministic fact-preservation gate: compare the multisets of
    numeric facts in the original vs the anonymized text.

    The LLM completeness check hallucinates (invents "erased numbers") and cannot be
    trusted. Here a deterministic multiset difference decides: a numeric fact present in
    the original but missing from the anonymized text = information erased by
    anonymization. This is the authoritative verdict.
    """
    from collections import Counter
    norm = lambda s: s.replace(" ", "").replace(",", "").replace("，", "")
    # co = Counter(norm(t) for t in _FACT_NUM_RE.findall(original))
    # ca = Counter(norm(t) for t in _FACT_NUM_RE.findall(anon))

    co = Counter(norm(t) for t in _FACT_NUM_EN_RE.findall(original))
    ca = Counter(norm(t) for t in _FACT_NUM_EN_RE.findall(anon))


    lost = co - ca       # in the original but lost in the anon text = factual number erased (severe)
    gained = ca - co     # appears only in the anon text (usually a replacement side effect, minor)
    ok = sum(lost.values()) == 0
    return {
        "fact_verdict": "consistent" if ok else "inconsistent",
        "n_original": sum(co.values()),
        "n_anon": sum(ca.values()),
        "numbers_lost": dict(lost),
        "numbers_gained": dict(gained),
        "_note": "Deterministic numeric multiset comparison; \
            nonempty numbers_lost = factual numbers were removed during anonymization",
    }


DATE_SHIFT_JUDGE_PROMPT = """你在校验一份匿名时间线的"日期整体平移"做得对不对。
全文所有日期本应被平移了**同样的 {delta} 天**：标题 `### 日期`、正文里的 `YYYY-MM-DD` / `X年X月X日` / 无年份的 `X月X日` 都要同步平移、保持彼此相对关系。

逐节点检查：
1. 每个节点标题(### 日期) 和 该节点正文里提到的日期，平移是否**一致**（相对间隔/先后顺序自洽）？
2. 有没有**漏平移**的日期（标题变了正文某日期没变，或反之）？
3. 平移后有没有不合理（顺序错乱、明显穿越）？

只输出 JSON（无额外文字）：{{"consistent":<true|false>,"issues":["具体问题(注明节点)",...],"summary":"一句话"}}

匿名文本：
{anon}
"""


def judge_date_shift(anon_text: str, delta_days: int, *, model: str) -> Dict[str, Any]:
    """LLM re-check of whether the date shift is consistent and reasonable
    (user-requested). Call/parse failures must not crash; returns consistent=None."""
    if delta_days in (0, None):
        return {"consistent": None, "_note": "未平移日期，跳过"}
    try:
        raw = call_llm(
            [{"role": "user", "content": DATE_SHIFT_JUDGE_PROMPT.format(delta=delta_days, anon=anon_text[:14000])}],
            model=model, temperature=0.0, max_tokens=1500,
        )
        return _parse_audit_json(raw)
    except Exception as e:
        return {"consistent": None, "_error": str(e)[:120]}


# ===========================================================================
# Orchestration
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="predict-step0-prepare: 3-phase anonymization")
    p.add_argument("timeline_md")
    p.add_argument("output_dir")
    p.add_argument("--replacements", required=True, help="JSON: list of [original, replacement] pairs")
    p.add_argument("--date-shift-max-days", type=int, default=None)
    p.add_argument("--date-shift-override", type=int, default=None,
                   help="Use this exact delta instead of sampling (e.g. 0 to disable date shift)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-audit", action="store_true",
                   help="Skip Phase 2 (LLM audit) and Phase 3 (consistency check)")
    p.add_argument("--audit-only", action="store_true",
                   help="Skip Phase 1; re-audit existing <output_dir>/timeline_anon.md")
    p.add_argument("--skip-consistency", action="store_true",
                   help="Run Phase 2 but skip Phase 3 (consistency check)")
    p.add_argument("--max-audit-rounds", type=int, default=5,
                   help="Cap for Phase 2 audit rounds within this invocation (default 5)")
    p.add_argument("--audit-model", default=None,
                   help="LLM used for the anonymization audit / consistency check. Default = "
                        "config.anonymization.audit_model, falling back to default_llm_model when null. "
                        "The audit fuses lenses A+B+C and relies on redundancy to absorb model uncertainty; "
                        "to give the audit a stronger dedicated model, set it in the config — no code change needed.")
    p.add_argument("--pipeline-config", default=None)
    args = p.parse_args()

    cfg = load_pipeline_config(args.pipeline_config) if args.pipeline_config else load_pipeline_config()
    anon_cfg = cfg.get("anonymization", {})
    max_d = (args.date_shift_max_days
             if args.date_shift_max_days is not None
             else int(anon_cfg.get("date_shift_max_days", 180)))
    # The audit needs a strong model: enumeration-style entity auditing and the
    # consistency check are sensitive to model capability (weaker models proved
    # unreliable in practice: missed leaks / drift). No model is hardcoded — precedence:
    # CLI > optional audit-specific model (anonymization.audit_model) > default LLM
    # (default_llm_model).
    audit_model = (args.audit_model
                   or anon_cfg.get("audit_model")
                   or cfg.get("default_llm_model"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = Path(args.timeline_md).read_text(encoding="utf-8")
    replacements = load_replacements(Path(args.replacements))

    # ----------------------------- Phase 1 ------------------------------
    if args.audit_only:
        anon_path = out_dir / "timeline_anon.md"
        if not anon_path.exists():
            raise SystemExit(f"--audit-only requires {anon_path} to already exist")
        anon = anon_path.read_text(encoding="utf-8")
        phase1_stats: Dict[str, Any] = {"skipped": True, "reason": "audit-only"}
        delta = None
    else:
        if args.date_shift_override is not None:
            delta = int(args.date_shift_override)
        else:
            rnd = random.Random(args.seed)
            delta = rnd.randint(-max_d, max_d)

        # Perform Anonymisation replacements
        anon, phase1_stats = phase1_replace(raw_text, replacements)
        # Perform delta shifting
        anon = shift_dates(anon, delta)

        (out_dir / "timeline_anon.md").write_text(anon, encoding="utf-8")
        (out_dir / "replacements_effective.json").write_text(
            json.dumps(replacements, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ----------------------------- Phase 2 ------------------------------
    # SKILL §Phase 2 (L250): "After each round of fixes ... add the newly found
    # replacement rules to the table and re-run Phase 1 from the original text."
    # — That is, each round we derive [bad, good] rules from the
    # audit findings, append them to `replacements`, then re-run phase1_replace
    # on the RAW original text (not the in-place anonymized version). This keeps
    # all variants of the same word handled consistently.
    audit_rounds: List[Dict[str, Any]] = []
    audit_passed = False
    appended_rules_total: List[List[str]] = []
    if not args.skip_audit:
        prior_findings: List[Dict[str, Any]] = []
        for r in range(1, max(1, args.max_audit_rounds) + 1):
            round_result = run_audit_round(anon, prior_findings, model=audit_model, raw_text=raw_text)
            round_result["round"] = r
            audit_rounds.append(round_result)
            stats = round_result.get("stats", {})
            if stats.get("high", 0) == 0 and stats.get("mid", 0) == 0:
                audit_passed = True
                break
            leaks = round_result.get("leaks") or []
            # Gate-verified real leaks (clean real names) are all handed to the LLM to
            # generate "type+code" placeholders (e.g. 城市A/海峡A/领导人B): lens
            # suggestions (especially B's) are not trusted directly; different entities
            # of the same type get different codes, the same entity keeps one code, and
            # placeholders already used in the table are passed in to avoid collisions
            # and keep whole-document / cross-round consistency; the LLM also filters
            # out non-proper-names (nothing is accepted wholesale).
            new_rules = derive_replacement_rules(
                anon, leaks, model=audit_model,
                existing_rules=replacements,
            )
            # re-run phase1_replace from the RAW original text with augmented rules.
            round_result["new_rules"] = new_rules
            if new_rules:
                replacements = replacements + new_rules
                appended_rules_total.extend(new_rules)
                anon, phase1_stats = phase1_replace(raw_text, replacements)
                if delta is not None:
                    anon = shift_dates(anon, delta)
                (out_dir / "timeline_anon.md").write_text(anon, encoding="utf-8")
                (out_dir / "replacements_effective.json").write_text(
                    json.dumps(replacements, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            # Per-round completeness gate (user-requested "check every round"): confirm
            # this round's fixes did not corrupt or drop information.
            #  Gate (2) (authoritative, deterministic): were any numeric facts erased?
            round_result["fact_preservation"] = number_preservation_check(raw_text, anon)
            #  LLM completeness (advisory; hallucinates)
            if not args.skip_consistency:
                round_result["consistency"] = run_consistency_check(raw_text, anon, model=audit_model)
            prior_findings = leaks

        (out_dir / "audit_report.json").write_text(
            json.dumps({
                "model": audit_model,
                "passed": audit_passed,
                "max_rounds": args.max_audit_rounds,
                "rounds": audit_rounds,
                "appended_rules_total": appended_rules_total,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----------------------------- Phase 3 ------------------------------
    consistency_result: Dict[str, Any] = {}
    if not args.skip_audit:
        # Gate (2): deterministic number-preservation gate = the authoritative fact
        # verdict; always runs (deterministic, zero cost, cannot hallucinate)
        fact_check = number_preservation_check(raw_text, anon)
        consistency_result["fact_preservation"] = fact_check
        consistency_result["verdict"] = fact_check["fact_verdict"]  # the code gate is the authoritative verdict
        # LLM completeness = advisory only (invents damaged numbers out of thin air); skippable via --skip-consistency
        if not args.skip_consistency:
            llm_res = run_consistency_check(raw_text, anon, model=audit_model)
            consistency_result["llm_verdict"] = llm_res.get("verdict")
            consistency_result["llm_detail"] = llm_res
        # Date-shift recheck: LLM judges whether heading/body dates were shifted in sync
        # and reasonably (user-requested)
        date_shift_check = judge_date_shift(anon, delta, model=audit_model)
        consistency_result["date_shift_check"] = date_shift_check
        (out_dir / "consistency_report.json").write_text(
            json.dumps(consistency_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----------------------------- Meta ---------------------------------
    meta = {
        "input_timeline": str(args.timeline_md),
        "replacements_file": str(args.replacements),
        "n_replacements": len(replacements),
        "date_shift_days": delta,
        "date_shift_max_days": max_d,
        "phase1": phase1_stats,
        "phase2": {
            "skipped": args.skip_audit,
            "passed": audit_passed,
            "rounds_executed": len(audit_rounds),
            "model": audit_model if not args.skip_audit else None,
        },
        "phase3": {
            "skipped": args.skip_audit,
            "fact_verdict": consistency_result.get("verdict") if consistency_result else None,
            "numbers_lost": (consistency_result.get("fact_preservation") or {}).get("numbers_lost") if consistency_result else None,
            "llm_verdict_advisory": consistency_result.get("llm_verdict") if consistency_result else None,
            "model": audit_model if (consistency_result and not args.skip_consistency) else None,
        },
    }
    (out_dir / "predict_input_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "output_timeline": str(out_dir / "timeline_anon.md"),
        "date_shift_days": delta,
        "phase1_total_replacements": phase1_stats.get("total_replacements"),
        "phase1_zero_hit_count": len(phase1_stats.get("zero_hit_rules", []) or []),
        "phase2_passed": audit_passed,
        "phase3_fact_verdict": consistency_result.get("verdict") if consistency_result else None,
        "phase3_numbers_lost": (consistency_result.get("fact_preservation") or {}).get("numbers_lost") if consistency_result else None,
        "phase3_llm_verdict_advisory": consistency_result.get("llm_verdict") if consistency_result else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
