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
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        print("[YouTube Fetcher] Could not resolve channel ID.")
        return []

    # 1️⃣ Get the uploads playlist
    try:
        channel_response = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()
        uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except Exception as e:
        print(f"[YouTube Fetcher] Could not get uploads playlist: {e}")
        return []

    # 2️⃣ Get videos from the uploads playlist
    video_ids = []
    next_page_token = None
    while len(video_ids) < 50:  # fetch enough videos
        playlist_response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in playlist_response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
            if len(video_ids) >= 50:
                break

        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    if not video_ids:
        print(f"[YouTube Fetcher] No videos found for channel {channel_id}")
        return []

    # 3️⃣ Fetch comments from videos
    comments = []
    for vid in video_ids:
        try:
            comment_response = youtube.commentThreads().list(
                part="snippet",
                videoId=vid,
                maxResults=50,
                textFormat="plainText"
            ).execute()
            for item in comment_response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "videoId": vid,
                    "author": top_comment.get("authorDisplayName"),
                    "text": top_comment.get("textDisplay")
                })
                if len(comments) >= max_comments:
                    return comments
        except Exception as e:
            print(f"[YouTube Fetcher] Error fetching comments for video {vid}: {e}")

    return comments
