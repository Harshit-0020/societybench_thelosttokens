"""
Usage:
    pip install requests
    set BSKY_HANDLE=yourname.bsky.social          (Windows)  /  export ... (Mac/Linux)
    set BSKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx     (create at Bluesky > Settings > App Passwords)
    python bluesky_opinions.py
Output: bluesky_opinions.csv with columns:
    type,id,parent_id,platform,create_time,title,content,liked_count,
    comment_count,share_count,user_name,ip_location,source_keyword,url
"""
import csv
import os
import time
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

# ---- Credentials ------------------------------------------------------------
# HANDLE = os.environ.get("BSKY_HANDLE", "")
HANDLE = "kobe08-24.bsky.social"
# APP_PASSWORD = os.environ.get("BSKY_APP_PASSWORD", "")
APP_PASSWORD = "Jaipur11@04"


# ---- What to collect --------------------------------------------------------
TOPICS = [
    "NEET_Paper_Leak",
]
LANGUAGE = "en"               # set to None for all languages
SORT = "top"                  # "top" or "latest"
POSTS_PER_TOPIC = 100         # max posts kept per topic
FETCH_REPLIES = True          # also collect replies for each post
MIN_REPLIES_TO_FETCH = 1      # only open threads with at least this many replies
REPLIES_PER_POST = 100        # max replies kept per post
SINCE = None                  # e.g. "2026-09-01T00:00:00Z" or None
OUTPUT_FILE = "bluesky_opinions_NEET_Paper_Leak.csv"
# -----------------------------------------------------------------------------

BASE = "https://bsky.social/xrpc"
COLUMNS = [
    "type", "id", "parent_id", "platform", "create_time", "title", "content",
    "liked_count", "comment_count", "share_count", "user_name", "ip_location",
    "source_keyword", "url",
]

session = requests.Session()
seen_ids = set()


def clean(text):
    return (text or "").replace("\r", " ").replace("\n", " ").strip()


def login():
    r = session.post(f"{BASE}/com.atproto.server.createSession",
                     json={"identifier": HANDLE, "password": APP_PASSWORD}, timeout=30)
    r.raise_for_status()
    session.headers["Authorization"] = f"Bearer {r.json()['accessJwt']}"


def get(endpoint, params):
    """GET with simple retry on rate limits / temporary errors."""
    for attempt in range(5):
        r = session.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        if r.status_code in (429, 502, 503):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Bluesky request kept failing: {endpoint}")


def post_url(post):
    return f"https://bsky.app/profile/{post['author']['handle']}/post/{post['uri'].split('/')[-1]}"


def to_row(post, row_type, keyword, parent_id=""):
    rec = post.get("record", {})
    created = rec.get("createdAt", post.get("indexedAt", ""))
    return {
        "type": row_type,
        "id": post["uri"].split("/")[-1],
        "parent_id": parent_id,
        "platform": "bluesky",
        "create_time": created,
        "title": "",
        "content": clean(rec.get("text", "")),
        "liked_count": post.get("likeCount", 0),
        "comment_count": post.get("replyCount", 0),
        "share_count": post.get("repostCount", 0) + post.get("quoteCount", 0),
        "user_name": post["author"]["handle"],
        "ip_location": "",  # Bluesky does not provide location
        "source_keyword": keyword,
        "url": post_url(post),
    }


def walk_replies(node, keyword, parent_rkey, out, limit):
    """Recursively collect replies from a getPostThread tree."""
    for child in node.get("replies", []):
        if len(out) >= limit:
            return
        post = child.get("post")
        if not post:  # blocked / not-found placeholder
            continue
        rkey = post["uri"].split("/")[-1]
        if rkey in seen_ids:
            continue
        seen_ids.add(rkey)
        out.append(to_row(post, "comment", keyword, parent_rkey))
        walk_replies(child, keyword, rkey, out, limit)


def collect_replies(post, keyword):
    data = get("app.bsky.feed.getPostThread", {"uri": post["uri"], "depth": 6, "parentHeight": 0})
    out = []
    walk_replies(data["thread"], keyword, post["uri"].split("/")[-1], out, REPLIES_PER_POST)
    return out


def search_topic(topic):
    rows, posts, cursor = [], [], None
    while len(posts) < POSTS_PER_TOPIC:
        params = {"q": topic, "sort": SORT, "limit": min(100, POSTS_PER_TOPIC - len(posts))}
        if LANGUAGE:
            params["lang"] = LANGUAGE
        if SINCE:
            params["since"] = SINCE
        if cursor:
            params["cursor"] = cursor
        data = get("app.bsky.feed.searchPosts", params)
        batch = data.get("posts", [])
        if not batch:
            break
        posts += batch
        cursor = data.get("cursor")
        if not cursor:
            break

    for p in posts:
        rkey = p["uri"].split("/")[-1]
        if rkey in seen_ids:
            continue
        seen_ids.add(rkey)
        rows.append(to_row(p, "post", topic))
        print(f"  - {clean(p['record'].get('text', ''))[:80]} ({p.get('replyCount', 0)} replies)")
        if FETCH_REPLIES and p.get("replyCount", 0) >= MIN_REPLIES_TO_FETCH:
            rows += collect_replies(p, topic)
    return rows


def main():
    if not HANDLE or not APP_PASSWORD:
        raise SystemExit("Missing credentials. Set BSKY_HANDLE and BSKY_APP_PASSWORD.")
    login()
    
    print("Hi")

    rows = []
    for topic in TOPICS:
        print(f"Topic: {topic}")
        rows += search_topic(topic)

    if not rows:
        print("Nothing found.")
        return

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    posts = sum(r["type"] == "post" for r in rows)
    print(f"\nSaved {posts} posts and {len(rows) - posts} replies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()



"""
Bluesky opinion scraper

Install:
    pip install requests

Set credentials before running.

Windows PowerShell:
    $env:BSKY_HANDLE="yourname.bsky.social"
    $env:BSKY_APP_PASSWORD="your-new-app-password"
    python bluesky_opinions.py

Ubuntu / WSL:
    export BSKY_HANDLE="yourname.bsky.social"
    export BSKY_APP_PASSWORD="your-new-app-password"
    python3 bluesky_opinions.py

Output:
    bluesky_opinions_NEET_Paper_Leak.csv
"""

# import sys

# sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# import csv
# import os
# import time
# import random
# import requests

# from datetime import datetime, timezone
# from email.utils import parsedate_to_datetime


# # --------------------------------------------------
# # Configuration
# # --------------------------------------------------

# HANDLE = os.environ.get("BSKY_HANDLE", "").strip()
# APP_PASSWORD = os.environ.get("BSKY_APP_PASSWORD", "").strip()

# TOPICS = [
#     "NEET Paper Leak",
# ]

# LANGUAGE = "en"       # Set to None for all languages
# SORT = "latest"       # "top" or "latest"

# POSTS_PER_TOPIC = 20
# FETCH_REPLIES = True
# MIN_REPLIES_TO_FETCH = 1
# REPLIES_PER_POST = 20

# # Example: "2026-09-01T00:00:00Z"
# # None means no lower date filter.
# SINCE = None

# OUTPUT_FILE = "bluesky_opinions_NEET_Paper_Leak.csv"

# BASE = "https://bsky.social/xrpc"

# COLUMNS = [
#     "type",
#     "id",
#     "parent_id",
#     "platform",
#     "create_time",
#     "title",
#     "content",
#     "liked_count",
#     "comment_count",
#     "share_count",
#     "user_name",
#     "ip_location",
#     "source_keyword",
#     "url",
# ]

# MAX_RETRIES = 5
# REQUEST_TIMEOUT = 30

# session = requests.Session()

# # Track full AT URIs to avoid collisions between different authors.
# seen_uris = set()


# # --------------------------------------------------
# # Helpers
# # --------------------------------------------------

# def clean(text):
#     """Normalize text for CSV storage."""
#     return (text or "").replace("\r", " ").replace("\n", " ").strip()


# def get_rkey(post):
#     """Extract the record key from a Bluesky post."""
#     return post["uri"].split("/")[-1]


# def post_url(post):
#     """Construct the public URL for a Bluesky post."""
#     handle = post["author"]["handle"]
#     return f"https://bsky.app/profile/{handle}/post/{get_rkey(post)}"


# def normalize_timestamp(value):
#     """
#     Preserve ISO 8601 timestamps in UTC format.
#     Example: 2026-09-05T08:44:27Z
#     """
#     if not value:
#         return ""

#     try:
#         dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

#         if dt.tzinfo is None:
#             dt = dt.replace(tzinfo=timezone.utc)

#         dt = dt.astimezone(timezone.utc)

#         # Remove fractional seconds to produce the requested format.
#         return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

#     except (ValueError, TypeError):
#         # Preserve the original value if parsing fails.
#         return value


# # --------------------------------------------------
# # Authentication
# # --------------------------------------------------

# def login():
#     """Authenticate using a Bluesky app password."""
#     if not HANDLE or not APP_PASSWORD:
#         raise RuntimeError(
#             "Missing credentials. Set BSKY_HANDLE and "
#             "BSKY_APP_PASSWORD environment variables."
#         )

#     response = session.post(
#         f"{BASE}/com.atproto.server.createSession",
#         json={
#             "identifier": HANDLE,
#             "password": APP_PASSWORD,
#         },
#         timeout=REQUEST_TIMEOUT,
#     )

#     response.raise_for_status()
#     data = response.json()

#     session.headers.update({
#         "Authorization": f"Bearer {data['accessJwt']}"
#     })

#     print(f"Logged in successfully as {HANDLE}")


# # --------------------------------------------------
# # API requests with retries
# # --------------------------------------------------

# def retry_delay(response, attempt):
#     """Determine how long to wait before retrying."""
#     retry_after = response.headers.get("Retry-After")

#     if retry_after:
#         try:
#             return max(1.0, float(retry_after))
#         except ValueError:
#             try:
#                 retry_date = parsedate_to_datetime(retry_after)
#                 return max(
#                     1.0,
#                     (retry_date - datetime.now(timezone.utc)).total_seconds()
#                 )
#             except (ValueError, TypeError, OverflowError):
#                 pass

#     # Exponential backoff with a small random delay.
#     return min(60, 2 ** (attempt + 1)) + random.random()


# def get(endpoint, params):
#     """GET with retries for temporary failures and rate limits."""
#     url = f"{BASE}/{endpoint}"

#     for attempt in range(MAX_RETRIES):
#         try:
#             response = session.get(
#                 url,
#                 params=params,
#                 timeout=REQUEST_TIMEOUT,
#             )

#             if response.status_code == 429 or response.status_code in (
#                 500, 502, 503, 504
#             ):
#                 wait = retry_delay(response, attempt)

#                 print(
#                     f"HTTP {response.status_code} for {endpoint}. "
#                     f"Retry {attempt + 1}/{MAX_RETRIES} "
#                     f"in {wait:.1f}s."
#                 )

#                 if attempt == MAX_RETRIES - 1:
#                     response.raise_for_status()

#                 time.sleep(wait)
#                 continue

#             response.raise_for_status()
#             return response.json()

#         except (requests.Timeout, requests.ConnectionError) as exc:
#             if attempt == MAX_RETRIES - 1:
#                 raise RuntimeError(
#                     f"Network request failed: {endpoint}"
#                 ) from exc

#             wait = min(60, 2 ** (attempt + 1))
#             print(
#                 f"Network error for {endpoint}: {exc}. "
#                 f"Retrying in {wait}s."
#             )
#             time.sleep(wait)

#     raise RuntimeError(f"Request failed after retries: {endpoint}")


# # --------------------------------------------------
# # Convert a post into the output schema
# # --------------------------------------------------

# def to_row(post, row_type, keyword, parent_id=""):
#     record = post.get("record", {})

#     created = normalize_timestamp(
#         record.get("createdAt", post.get("indexedAt", ""))
#     )

#     return {
#         "type": row_type,
#         "id": get_rkey(post),
#         "parent_id": parent_id,
#         "platform": "bluesky",
#         "create_time": created,
#         "title": "",
#         "content": clean(record.get("text", "")),
#         "liked_count": post.get("likeCount", 0),
#         "comment_count": post.get("replyCount", 0),
#         "share_count": (
#             post.get("repostCount", 0)
#             + post.get("quoteCount", 0)
#         ),
#         "user_name": post.get("author", {}).get("handle", ""),
#         "ip_location": "",
#         "source_keyword": keyword,
#         "url": post_url(post),
#     }


# # --------------------------------------------------
# # Recursively extract replies
# # --------------------------------------------------

# def walk_replies(node, keyword, parent_rkey, out, limit):
#     """
#     Collect replies from a getPostThread response.
#     Stop when the per-thread reply limit is reached.
#     """
#     if len(out) >= limit:
#         return

#     for child in node.get("replies", []):
#         if len(out) >= limit:
#             return

#         post = child.get("post")

#         # Skip blocked, deleted, or unavailable posts.
#         if not post or not post.get("uri"):
#             continue

#         uri = post["uri"]

#         if uri in seen_uris:
#             continue

#         seen_uris.add(uri)
#         rkey = get_rkey(post)

#         out.append(to_row(post, "comment", keyword, parent_rkey))

#         walk_replies(
#             child,
#             keyword,
#             rkey,
#             out,
#             limit,
#         )


# def collect_replies(post, keyword):
#     """Fetch a thread and return its replies."""
#     try:
#         data = get(
#             "app.bsky.feed.getPostThread",
#             {
#                 "uri": post["uri"],
#                 "depth": 6,
#                 "parentHeight": 0,
#             },
#         )

#         thread = data.get("thread", {})

#         if not isinstance(thread, dict):
#             return []

#         out = []

#         walk_replies(
#             thread,
#             keyword,
#             get_rkey(post),
#             out,
#             REPLIES_PER_POST,
#         )

#         return out

#     except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
#         print(f"Could not fetch replies for {post.get('uri', '')}: {exc}")
#         return []


# # --------------------------------------------------
# # Search posts
# # --------------------------------------------------

# def search_topic(topic):
#     """Yield rows as posts are processed, limiting memory use."""
#     cursor = None
#     post_count = 0

#     while post_count < POSTS_PER_TOPIC:
#         params = {
#             "q": topic,
#             "sort": SORT,
#             "limit": min(100, POSTS_PER_TOPIC - post_count),
#         }

#         if LANGUAGE:
#             params["lang"] = LANGUAGE

#         if SINCE:
#             params["since"] = SINCE

#         if cursor:
#             params["cursor"] = cursor

#         data = get("app.bsky.feed.searchPosts", params)
#         batch = data.get("posts", [])

#         if not batch:
#             print("No more posts returned by the API.")
#             break

#         for post in batch:
#             if post_count >= POSTS_PER_TOPIC:
#                 break

#             uri = post.get("uri")

#             if not uri or uri in seen_uris:
#                 continue

#             seen_uris.add(uri)
#             post_count += 1

#             print(
#                 f"[{post_count}/{POSTS_PER_TOPIC}] "
#                 f"{clean(post.get('record', {}).get('text', ''))[:90]} "
#                 f"({post.get('replyCount', 0)} replies)"
#             )

#             # Yield the post immediately so it can be saved.
#             yield to_row(post, "post", topic)

#             if (
#                 FETCH_REPLIES
#                 and post.get("replyCount", 0) >= MIN_REPLIES_TO_FETCH
#             ):
#                 replies = collect_replies(post, topic)

#                 for reply in replies:
#                     yield reply

#                 print(f"    Collected {len(replies)} replies")

#             # Small pause between posts to reduce request pressure.
#             time.sleep(0.3)

#         cursor = data.get("cursor")

#         if not cursor:
#             break

#     print(f"Finished topic: {topic}. Posts collected: {post_count}")


# # --------------------------------------------------
# # Main: save each row as it arrives
# # --------------------------------------------------

# def main():
#     total_posts = 0
#     total_replies = 0
#     file_has_data = os.path.exists(OUTPUT_FILE) and os.path.getsize(
#         OUTPUT_FILE
#     ) > 0

#     # Start a fresh output file for each run.
#     # This avoids mixing old results with new results.
#     with open(
#         OUTPUT_FILE,
#         "w",
#         newline="",
#         encoding="utf-8-sig",
#     ) as output:
#         writer = csv.DictWriter(output, fieldnames=COLUMNS)
#         writer.writeheader()
#         output.flush()

#         try:
#             login()

#             for topic in TOPICS:
#                 print(f"\nSearching topic: {topic}")

#                 for row in search_topic(topic):
#                     writer.writerow(row)
#                     output.flush()

#                     if row["type"] == "post":
#                         total_posts += 1
#                     else:
#                         total_replies += 1

#         except KeyboardInterrupt:
#             print("\nStopped by user. Saving collected results...")
#         except Exception as exc:
#             print(f"\nScraper stopped due to an error: {exc}")
#             print("All rows written before the error remain in the CSV.")
#             raise
#         finally:
#             output.flush()

#     print("\nCollection finished.")
#     print(f"Posts saved: {total_posts}")
#     print(f"Replies saved: {total_replies}")
#     print(f"Output file: {os.path.abspath(OUTPUT_FILE)}")


# if __name__ == "__main__":
#     main()
