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
        user = reddit.redditor(username)
        
        # Check if user exists
        try:
            user.id  # This will raise an exception if user doesn't exist
        except Exception:
            print(f"[Reddit Fetcher] User '{username}' not found or suspended")
            return []
        
        # Fetch comments
        comment_count = 0
        for comment in user.comments.new(limit=limit):
            if comment.body and comment.body != "[deleted]" and comment.body != "[removed]":
                comments_data.append({
                    "platform": "Reddit",
                    "text": comment.body,
                    "author": comment.author.name if comment.author else "deleted",
                    "timestamp": comment.created_utc
                })
                comment_count += 1
                if comment_count >= limit:  # Ensure we don't exceed the requested limit
                    break
        
        print(f"[Reddit Fetcher] Successfully fetched {comment_count} comments from {username}")
        
    except Exception as e:
        print(f"[Reddit Fetcher] Error fetching user {username}: {e}")
    return comments_data
