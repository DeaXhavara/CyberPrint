# scrapers/youtube_fetcher.py
import os
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def extract_channel_id(url: str) -> str:
    """
    Extracts channel ID from any YouTube URL:
    - https://www.youtube.com/channel/UCxxxx
    - https://www.youtube.com/user/username
    - https://www.youtube.com/@handle
    """
    # Direct channel ID
    match = re.search(r"youtube\.com/channel/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    # Username
    match = re.search(r"youtube\.com/user/([a-zA-Z0-9_-]+)", url)
    if match:
        username = match.group(1)
        response = youtube.channels().list(part="id", forUsername=username).execute()
        if response["items"]:
            return response["items"][0]["id"]

    # Handle (@handle)
    match = re.search(r"youtube\.com/@([a-zA-Z0-9_-]+)", url)
    if match:
        handle = match.group(1)
        response = youtube.search().list(part="snippet", q=handle, type="channel", maxResults=1).execute()
        if response["items"]:
            return response["items"][0]["snippet"]["channelId"]

    return None


def fetch_youtube_comments(channel_url: str, max_comments: int = 50):
    """
    Fetch comments that others have made on a YouTube channel's videos.
    This analyzes the feedback/sentiment the channel receives from viewers.
    """
    if not YOUTUBE_API_KEY:
        print("[YouTube Fetcher] YouTube API key not found. Please set YOUTUBE_API_KEY in .env file")
        return []
        
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        print("[YouTube Fetcher] Could not resolve channel ID.")
        return []

    # Step 1: Get the uploads playlist
    try:
        channel_response = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()
        
        if not channel_response.get("items"):
            print(f"[YouTube Fetcher] Channel not found: {channel_id}")
            return []
            
        uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError as e:
        print(f"[YouTube Fetcher] API Error getting channel: {e}")
        return []
    except Exception as e:
        print(f"[YouTube Fetcher] Could not get uploads playlist: {e}")
        return []

    # Step 2: Get recent videos from the uploads playlist
    video_ids = []
    try:
        playlist_response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=20  # Get recent videos
        ).execute()

        for item in playlist_response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
            
    except HttpError as e:
        print(f"[YouTube Fetcher] API Error getting videos: {e}")
        return []
    except Exception as e:
        print(f"[YouTube Fetcher] Error getting videos: {e}")
        return []

    if not video_ids:
        print(f"[YouTube Fetcher] No videos found for channel {channel_id}")
        return []

    # Step 3: Fetch comments from videos (comments others made about this channel)
    comments = []
    for vid in video_ids:
        if len(comments) >= max_comments:
            break
            
        try:
            remaining_comments = max_comments - len(comments)
            comment_response = youtube.commentThreads().list(
                part="snippet",
                videoId=vid,
                maxResults=min(25, remaining_comments),
                order="relevance",  # Get most relevant comments
                textFormat="plainText"
            ).execute()
            
            for item in comment_response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                comment_text = top_comment.get("textDisplay", "").strip()
                
                # Filter out very short comments (likely spam/low quality)
                if len(comment_text) > 10:
                    comments.append({
                        "videoId": vid,
                        "author": top_comment.get("authorDisplayName"),
                        "text": comment_text
                    })
                    
                if len(comments) >= max_comments:
                    break
                    
        except HttpError as e:
            if e.resp.status == 403:
                print(f"[YouTube Fetcher] Comments disabled for video {vid}")
            else:
                print(f"[YouTube Fetcher] API Error fetching comments for video {vid}: {e}")
        except Exception as e:
            print(f"[YouTube Fetcher] Error fetching comments for video {vid}: {e}")

    print(f"[YouTube Fetcher] Successfully fetched {len(comments)} comments from {len(video_ids)} videos")
    return comments
