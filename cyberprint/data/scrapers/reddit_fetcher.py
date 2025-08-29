import praw
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_reddit_comments_by_user(username, limit=50):
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    comments_data = []
    try:
        for comment in reddit.redditor(username).comments.new(limit=limit):
            comments_data.append({
                "platform": "Reddit",
                "text": comment.body,
                "author": comment.author.name if comment.author else "deleted",
                "timestamp": comment.created_utc
            })
    except Exception as e:
        print(f"[Reddit Fetcher] Error fetching user {username}: {e}")
    return comments_data
