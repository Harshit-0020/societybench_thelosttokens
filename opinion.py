"""
Usage:
    python reddit_opinions.py
Output: reddit_opinions.csv with columns:
    type,id,parent_id,platform,create_time,title,content,liked_count,
    comment_count,share_count,user_name,ip_location,source_keyword,url
"""
import csv
import os
from datetime import datetime, timezone

import praw

# ---- Credentials ------------------------------------------------------------
CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")       # or paste here for a local test
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
USER_AGENT = "news-opinion-research by u/Mountain_Version2257"

# ---- What to collect --------------------------------------------------------
TOPICS = [
    "Asian Games 2026"
]
ARTICLE_URLS = [
    # "https://www.example.com/news/some-article",   # finds Reddit threads sharing this link
]
SUBREDDITS = "technology+artificial+economics+india+worldnews"                       # e.g. "technology+artificial+economics+india+worldnews"
THREADS_PER_TOPIC = 5                    # top threads per topic
COMMENTS_PER_THREAD = 200                # max comments kept per thread
OUTPUT_FILE = "reddit_opinions.csv"
# -----------------------------------------------------------------------------

COLUMNS = [
    "type", "id", "parent_id", "platform", "create_time", "title", "content",
    "liked_count", "comment_count", "share_count", "user_name", "ip_location",
    "source_keyword", "url",
]

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
seen_ids = set()


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def clean(text):
    return (text or "").replace("\r", " ").replace("\n", " ").strip()


def collect_thread(sub, keyword):
    """Return one 'post' row plus 'comment' rows for a submission."""
    rows = []
    if sub.id in seen_ids:
        return rows
    seen_ids.add(sub.id)

    # For link posts, content is the article URL being discussed; for text posts it's the body.
    post_content = clean(sub.selftext) if sub.is_self else sub.url
    rows.append({
        "type": "post",
        "id": sub.id,
        "parent_id": "",
        "platform": "reddit",
        "create_time": iso(sub.created_utc),
        "title": clean(sub.title),
        "content": post_content,
        "liked_count": sub.score,
        "comment_count": sub.num_comments,
        "share_count": getattr(sub, "num_crossposts", 0),  # crossposts = closest to "shares"
        "user_name": str(sub.author) if sub.author else "[deleted]",
        "ip_location": "",  # Reddit does not provide location
        "source_keyword": keyword,
        "url": f"https://www.reddit.com{sub.permalink}",
    })

    sub.comment_sort = "top"
    sub.comments.replace_more(limit=0)  # skip "load more" stubs (keeps API calls low)
    for c in sub.comments.list()[:COMMENTS_PER_THREAD]:
        text = clean(c.body)
        if not text or text in ("[deleted]", "[removed]"):
            continue
        rows.append({
            "type": "comment",
            "id": c.id,
            "parent_id": c.parent_id.split("_", 1)[-1],  # id of parent comment or the post
            "platform": "reddit",
            "create_time": iso(c.created_utc),
            "title": "",
            "content": text,
            "liked_count": c.score,
            "comment_count": len(c.replies),  # direct replies loaded
            "share_count": 0,
            "user_name": str(c.author) if c.author else "[deleted]",
            "ip_location": "",
            "source_keyword": keyword,
            "url": f"https://www.reddit.com{c.permalink}",
        })
    return rows


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise SystemExit("Missing Reddit credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")

    rows = []
    for topic in TOPICS:
        print(f"Topic: {topic}")
        for sub in reddit.subreddit(SUBREDDITS).search(topic, sort="top", time_filter="month", limit=THREADS_PER_TOPIC):
            print(f"  - {sub.title[:80]} ({sub.num_comments} comments)")
            rows += collect_thread(sub, topic)

    for url in ARTICLE_URLS:
        print(f"Article: {url}")
        for sub in list(reddit.info(url=url))[:THREADS_PER_TOPIC]:
            print(f"  - {sub.title[:80]} ({sub.num_comments} comments)")
            rows += collect_thread(sub, url)

    if not rows:
        print("Nothing found.")
        return

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel reads it correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    posts = sum(r["type"] == "post" for r in rows)
    print(f"\nSaved {posts} posts and {len(rows) - posts} comments to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()