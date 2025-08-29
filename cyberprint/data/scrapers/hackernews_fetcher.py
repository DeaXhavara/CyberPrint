import requests
import time

def fetch_hackernews_comments_by_user(username, limit=50):
    """
    Fetches latest comments from a Hacker News user.

    Args:
        username (str): Hacker News username
        limit (int): Max number of comments to fetch

    Returns:
        List[dict]: Each dict contains platform, type, content, author, timestamp
    """
    comments_data = []

    # Step 1: Get user info to find submitted items
    user_url = f"https://hacker-news.firebaseio.com/v0/user/{username}.json"
    response = requests.get(user_url)
    if response.status_code != 200:
        print(f"[HackerNews Fetcher] User {username} not found")
        return []

    user_data = response.json()
    if not user_data or "submitted" not in user_data:
        print(f"[HackerNews Fetcher] No submissions for user {username}")
        return []

    submitted_ids = user_data["submitted"]
    count = 0

    # Step 2: Fetch items (comments only)
    for item_id in submitted_ids:
        if count >= limit:
            break

        item_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
        item_resp = requests.get(item_url)
        if item_resp.status_code != 200:
            continue

        item_data = item_resp.json()
        if item_data is None:
            continue

        if item_data.get("type") == "comment":
            comments_data.append({
                "platform": "HackerNews",
                "type": "comment",
                "content": item_data.get("text", ""),
                "author": username,
                "timestamp": item_data.get("time")
            })
            count += 1

        time.sleep(0.1)  # polite rate limiting

    return comments_data[:limit]
