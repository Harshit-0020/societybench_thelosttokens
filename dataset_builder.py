import sys, traceback
from pathlib import Path

def _crash(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    print(msg)
    (Path.home() / "scraper_crash.txt").write_text(msg, encoding="utf-8")

sys.excepthook = _crash

import requests
import trafilatura
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import hashlib
import json
import os
import re
import time
from urllib.parse import urlparse

# Pretty source names for common Hindi news domains (add more as needed)
SOURCE_MAP = {
    "navbharattimes.indiatimes.com": "Navbharat Times",
    "jagran.com": "Dainik Jagran",
    "jagranjosh.com": "Jagran Josh",
    "bhaskar.com": "Dainik Bhaskar",
    "amarujala.com": "Amar Ujala",
    "livehindustan.com": "Hindustan",
    "hindi.news18.com": "News18 Hindi",
    "aajtak.in": "Aaj Tak",
    "abplive.com": "ABP Live",
    "zeenews.india.com": "Zee News",
    "ndtv.in": "NDTV India",
    "jansatta.com": "Jansatta",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "hi-IN,hi;q=0.9,en;q=0.7",
}


# ---------------------------------------------------------------- helpers
def get_source_name(url):
    domain = urlparse(url).netloc.lower().replace("www.", "")
    for key, name in SOURCE_MAP.items():
        if domain == key or domain.endswith("." + key):
            return name
    return domain.split(".")[0].replace("-", " ").title()


def clean_text(value):
    return BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True)


def normalize_date(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            ts = value / 1000 if value > 10**12 else value
            return datetime.datetime.fromtimestamp(ts).date().isoformat()
        except (ValueError, OSError, OverflowError):
            return None

    value = str(value).strip()
    if not value or value.lower() in {"n/a", "unknown", "none"}:
        return None

    try:
        return datetime.datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).date().isoformat()
    except ValueError:
        pass

    if re.fullmatch(r"\d{10}|\d{13}", value):
        try:
            ts = int(value)
            if len(value) == 13:
                ts /= 1000
            return datetime.datetime.fromtimestamp(ts).date().isoformat()
        except (ValueError, OSError, OverflowError):
            pass

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
                "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %b, %Y",
                "%d %B %Y", "%d %B, %Y"):
        try:
            return datetime.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def extract_published_date(soup):
    # 1) meta tags
    for attrs in (
        {"property": "article:published_time"},
        {"name": "date"},
        {"itemprop": "datePublished"},
        {"property": "og:article:published_time"},
        {"name": "publish-date"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag:
            parsed = normalize_date(tag.get("content"))
            if parsed:
                return parsed

    # 2) JSON-LD
    def find_date(obj):
        if isinstance(obj, dict):
            for key in ("datePublished", "dateCreated"):
                if obj.get(key):
                    return obj[key]
            for v in obj.values():
                r = find_date(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for item in obj:
                r = find_date(item)
                if r:
                    return r
        return None

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or script.get_text())
            parsed = normalize_date(find_date(data))
            if parsed:
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # 3) <time datetime="...">
    tag = soup.find("time", attrs={"datetime": True})
    if tag:
        parsed = normalize_date(tag.get("datetime"))
        if parsed:
            return parsed

    return "Unknown"


def extract_author(soup):
    tag = soup.find("meta", attrs={"name": re.compile(r"^author$", re.I)})
    if tag and tag.get("content"):
        return clean_text(tag["content"])

    tag = soup.find("meta", attrs={"property": "article:author"})
    if tag and tag.get("content") and not tag["content"].startswith("http"):
        return clean_text(tag["content"])

    byline = soup.find("div", class_="bl")
    if byline:
        link = byline.find("a")
        if link and link.get_text(strip=True):
            return link.get_text(" ", strip=True)

    tag = soup.find(attrs={"itemprop": re.compile(r"^author$", re.I)})
    if tag:
        name_tag = tag.find(attrs={"itemprop": "name"}) or tag
        if name_tag.get_text(strip=True):
            return name_tag.get_text(" ", strip=True)

    return "Unknown"


def extract_title(soup):
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return clean_text(og["content"])
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    if soup.title:
        return soup.title.get_text(strip=True)
    return "Unknown"


def extract_text_fallback(soup):
    """Paragraph-based extraction (used if trafilatura returns too little)."""
    node = soup.find(attrs={"itemprop": re.compile(r"articleBody", re.I)})
    if node is None:
        node = soup.find(
            ["article", "div"],
            class_=re.compile(
                r"(story|article).*(content|body)|(content|body).*(story|article)", re.I
            ),
        )
    if node is None:
        node = soup.find("article") or soup.find("main")

    if node:
        for bad in node.select(
            "script, style, noscript, iframe, figure, button, form, nav, footer, aside"
        ):
            bad.decompose()
        paragraphs = node.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    out, seen = [], set()
    for p in paragraphs:
        t = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(t) < 20 or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return "\n".join(out)


def devanagari_count(text):
    return len(re.findall(r"[\u0900-\u097F]", text or ""))


# ---------------------------------------------------------------- scraper
def scrape_article(url, query, session, require_hindi=True):
    response = session.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    # Fix wrong/missing charset (common cause of garbled Hindi)
    if not response.encoding or response.encoding.lower() in ("iso-8859-1", "ascii"):
        response.encoding = response.apparent_encoding or "utf-8"
    html = response.text

    soup = BeautifulSoup(html, "lxml")

    # Primary extraction: trafilatura; fallback: paragraph extractor
    text = trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_recall=True
    )
    if not text or len(text) < 200:
        fallback = extract_text_fallback(BeautifulSoup(html, "lxml"))
        if len(fallback) > len(text or ""):
            text = fallback

    text = (text or "").strip()

    if require_hindi and devanagari_count(text) < 15:
        raise ValueError("Insufficient Hindi (Devanagari) text found on page")

    return {
        "record_id": "news_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:8],
        "source_type": "news",
        "source": get_source_name(url),
        "url": url,
        "title": extract_title(soup),
        "published_date": extract_published_date(soup),
        "author": extract_author(soup),
        "text": text,
        "query": query,
        "scraped_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "language" : "hindi"
    }


def scrape_from_excel(excel_path, output_file="articles.json",
                      delay=1.5, require_hindi=True):
    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if not {"url", "query"}.issubset(df.columns):
        raise ValueError(f"Excel must have columns 'url' and 'query'. Found: {list(df.columns)}")

    session = requests.Session()
    articles, failed, seen = [], [], set()

    for idx, row in df.iterrows():
        url = str(row["url"]).strip()
        query = "" if pd.isna(row["query"]) else str(row["query"]).strip()

        if not url or url.lower() == "nan":
            continue
        if url in seen:
            print(f"[{idx + 1}/{len(df)}] Duplicate, skipping: {url}")
            continue
        seen.add(url)

        print(f"[{idx + 1}/{len(df)}] {url}")
        try:
            art = scrape_article(url, query, session, require_hindi=require_hindi)
            articles.append(art)
            print(f"   OK: {art['title'][:70]}")
        except Exception as e:
            print(f"   FAILED: {e}")
            failed.append({"url": url, "query": query, "error": str(e)})

        time.sleep(delay)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(articles)} saved, {len(failed)} failed.")
    print("Saved to:", output_file)

    if failed:
        failed_file = output_file.replace(".json", "_failed.json")
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print("Failed URLs logged to:", failed_file)

    return articles


if __name__ == "__main__":
    import sys, traceback
    from pathlib import Path

    BASE = Path(__file__).resolve().parent
    excel_path = BASE / "russia_ukraine_english_urls_2024.xlsx"       # must match your actual Excel filename
    output_file = BASE / "articles_3.json"

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
        def flush(self):
            for st in self.streams: st.flush()

    log = open(BASE / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    sys.stderr = Tee(sys.__stderr__, log)

    try:
        print("Working folder:", BASE)
        print("Excel exists:", excel_path.exists())
        scrape_from_excel(
            excel_path=str(excel_path),
            output_file=str(output_file),
            delay=1.5,
            require_hindi=True,
        )
    except Exception:
        traceback.print_exc()
    finally:
        print("Finished.")
        log.close()