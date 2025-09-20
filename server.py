#!/usr/bin/env python3
"""
CyberPrint FastAPI Server
========================

FastAPI server that wraps the existing CyberPrint pipeline for web frontend integration.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cyberprint.pipeline import CyberPrintPipeline
from cyberprint.data.scrapers.reddit_fetcher import fetch_reddit_comments_by_user
from cyberprint.data.scrapers.youtube_fetcher import fetch_youtube_comments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CyberPrint API", description="Advanced Sentiment Analysis Pipeline", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://127.0.0.1:53646", "https://cyberprintapp-production.up.railway.app"],  # React dev server + proxy + Railway frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static files (PDFs)
output_dir = Path(__file__).parent / "cyberprint" / "data" / "output"
app.mount("/static", StaticFiles(directory=str(output_dir)), name="static")

class AnalysisRequest(BaseModel):
    url: str
    num_comments: int = 50

class ProfileInfo(BaseModel):
    username: str
    platform: str
    profile_url: str
    avatar_url: str = ""

def infer_platform_from_url(url: str) -> str:
    """Infer platform from the given URL."""
    url_lower = url.lower()
    if "reddit.com" in url_lower or "redd.it" in url_lower:
        return "reddit"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    else:
        return "unknown"

def extract_username_from_url(url: str, platform: str) -> str:
    """Extract username/channel from URL based on platform."""
    if platform == "reddit":
        if "/user/" in url:
            return url.split("/user/")[1].split("/")[0]
        elif "/u/" in url:
            return url.split("/u/")[1].split("/")[0]
    elif platform == "youtube":
        # Extract clean channel name from YouTube URL
        import re
        # Handle @handle format
        if "/@" in url:
            handle = url.split("/@")[1].split("/")[0].split("?")[0]
            return handle
        # Handle /user/ format
        elif "/user/" in url:
            username = url.split("/user/")[1].split("/")[0].split("?")[0]
            return username
        # Handle /channel/ format
        elif "/channel/" in url:
            channel_id = url.split("/channel/")[1].split("/")[0].split("?")[0]
            return channel_id
        # Handle /c/ format
        elif "/c/" in url:
            channel_name = url.split("/c/")[1].split("/")[0].split("?")[0]
            return channel_name
        else:
            # Fallback: return the full URL (will be sanitized later)
            return url
    return None

def get_profile_info(platform: str, identifier: str, url: str) -> ProfileInfo:
    """Get profile information for display."""
    avatar_url = ""
    
    # Generate avatar URLs based on platform
    if platform == "reddit":
        avatar_url = f"https://www.reddit.com/user/{identifier}/about.json"  # Reddit API for avatar
    elif platform == "youtube":
        avatar_url = ""  # YouTube avatars require API key
    
    return ProfileInfo(
        username=identifier,
        platform=platform,
        profile_url=url,
        avatar_url=avatar_url
    )

def fetch_user_comments(platform: str, identifier: str, num_comments: int, url: str = None) -> list:
    """Fetch comments based on platform."""
    comments = []
    
    try:
        if platform == "reddit":
            logger.info(f"Fetching {num_comments} comments from Reddit user: {identifier}")
            reddit_comments = fetch_reddit_comments_by_user(identifier, num_comments)
            for comment in reddit_comments:
                comments.append({
                    'tweet': comment['text'],
                    'source': f"reddit_{identifier}",
                    'platform': 'reddit'
                })
                
        elif platform == "youtube":
            logger.info(f"Fetching {num_comments} comments from YouTube channel: {identifier}")
            # For YouTube, we need to pass the full URL to the fetcher, not just the identifier
            # The fetcher handles URL parsing internally
            youtube_comments = fetch_youtube_comments(url if url else identifier, num_comments)
            for comment in youtube_comments:
                comments.append({
                    'tweet': comment['text'],
                    'source': f"youtube_{identifier}",
                    'platform': 'youtube'
                })
                
                
    except Exception as e:
        logger.error(f"Error fetching comments: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch comments: {str(e)}")
    
    return comments

def generate_insights(analytics: Dict[str, Any], platform: str) -> Dict[str, list]:
    """Generate key insights from analytics data."""
    insights = {
        "if_this_is_you": [],
        "if_this_is_stranger": []
    }
    
    sentiment_dist = analytics.get('sentiment_distribution', {})
    total_comments = analytics.get('total_comments', 0)
    mental_health_warnings = analytics.get('mental_health_warnings', 0)
    
    # Calculate percentages
    positive_pct = (sentiment_dist.get('positive', 0) / total_comments * 100) if total_comments > 0 else 0
    negative_pct = (sentiment_dist.get('negative', 0) / total_comments * 100) if total_comments > 0 else 0
    
    # "If this is you" insights - always provide comprehensive advice
    if positive_pct > 60:
        insights["if_this_is_you"].extend([
            "You maintain a positive online presence! Keep spreading good vibes.",
            "Your positive communication style creates a welcoming environment for others.",
            "Consider mentoring others who might benefit from your positive approach."
        ])
    elif positive_pct > 30:
        insights["if_this_is_you"].extend([
            "You show a balanced approach to online communication.",
            "Your mix of positive and constructive feedback demonstrates emotional maturity.",
            "Continue fostering healthy online discussions."
        ])
    else:
        insights["if_this_is_you"].extend([
            "Consider adding more positive interactions to improve your online presence.",
            "Small changes like expressing gratitude can significantly impact your digital footprint.",
            "Try to balance criticism with constructive suggestions."
        ])
    
    if negative_pct > 40:
        insights["if_this_is_you"].extend([
            "High negative sentiment detected. Consider taking breaks from heated discussions.",
            "Practice the 24-hour rule: wait before responding to controversial topics.",
            "Focus on solutions rather than problems in your communications."
        ])
    elif negative_pct > 20:
        insights["if_this_is_you"].append("Monitor your tone in online discussions to maintain positive relationships.")
    
    if mental_health_warnings > 0 and platform != "youtube":
        insights["if_this_is_you"].extend([
            "Mental health support resources are available. Consider reaching out for help.",
            "Your wellbeing matters - don't hesitate to seek professional support if needed."
        ])
    
    if analytics.get('yellow_flags', 0) > total_comments * 0.2:
        insights["if_this_is_you"].extend([
            "High sarcasm detected. Consider clearer communication to avoid misunderstandings.",
            "Add context or tone indicators to help others understand your intent."
        ])
    
    # Always add general digital wellness advice
    insights["if_this_is_you"].extend([
        "Regular self-reflection on your online behavior promotes healthy digital habits.",
        "Your communication style can always evolve - embrace growth and learning."
    ])
    
    # Platform-specific insights
    if platform == "youtube":
        insights["if_this_is_you"].extend([
            "This analysis shows how your audience responds to your content.",
            "Use this feedback to understand what resonates with your viewers.",
            "Consider adjusting your content strategy based on audience sentiment."
        ])
        insights["if_this_is_stranger"].extend([
            "This shows the feedback this YouTube channel receives from their audience.",
            "Remember: Comments reflect viewer opinions, not necessarily the creator's character.",
            "Content creators often face varied audience reactions.",
            "Consider the context - controversial topics may generate more negative feedback."
        ])
    else:
        # "If this is stranger" insights for other platforms
        insights["if_this_is_stranger"].extend([
            "Remember: Online behavior often doesn't reflect someone's true character.",
            "Don't take negative comments personally - they're about the commenter, not you.",
            "Consider sharing CyberPrint with them so they can see their own patterns.",
            "Approach online interactions with empathy and understanding."
        ])
    
    if negative_pct > 50:
        if platform == "youtube":
            insights["if_this_is_stranger"].append("This channel receives significant negative feedback. Content may be controversial or audience may be highly critical.")
        else:
            insights["if_this_is_stranger"].append("This person may be going through a difficult time. Show compassion.")
    
    return insights

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "CyberPrint API is running", "version": "1.0.0"}

@app.post("/analyze")
async def analyze_profile(request: AnalysisRequest) -> Dict[str, Any]:
    """
    Analyze a user profile and return sentiment analysis results.
    """
    try:
        # Validate input
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        if request.num_comments <= 0 or request.num_comments > 1000:
            raise HTTPException(status_code=400, detail="Number of comments must be between 1 and 1000")
        
        # Detect platform
        platform = infer_platform_from_url(request.url)
        if platform == "unknown":
            raise HTTPException(
                status_code=400, 
                detail="Unsupported platform. Please use Reddit or YouTube URLs"
            )
        
        # Extract identifier
        identifier = extract_username_from_url(request.url, platform)
        if not identifier:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract username/channel from URL"
            )
        
        logger.info(f"Analyzing {platform} profile: {identifier}")
        
        # Initialize pipeline
        pipeline = CyberPrintPipeline()
        
        # Fetch comments
        user_comments = fetch_user_comments(platform, identifier, request.num_comments, request.url)
        
        if not user_comments:
            raise HTTPException(
                status_code=404, 
                detail=f"No comments found for {platform} user '{identifier}'. The profile may not exist, be private, or have no recent comments."
            )
        
        logger.info(f"Successfully fetched {len(user_comments)} comments")
        
        # Convert to DataFrame
        user_data = pd.DataFrame(user_comments)
        logger.info(f"DataFrame created with {len(user_data)} rows")
        
        # Process and generate reports
        analytics = pipeline.process_and_report(
            user_data,
            user_id=identifier,
            platform=platform,
            profile_url=request.url,
            generate_pdf=True,
            generate_html=False
        )
        
        # Get profile info
        profile_info = get_profile_info(platform, identifier, request.url)
        
        # Generate insights
        insights = generate_insights(analytics, platform)
        
        # Prepare response
        response = {
            "success": True,
            "profile": profile_info.dict(),
            "analytics": {
                "total_comments": analytics['total_comments'],
                "sentiment_distribution": analytics['sentiment_distribution'],
                "sub_label_distribution": analytics.get('sub_label_distribution', {}),
                "mental_health_warnings": analytics.get('mental_health_warnings', 0),
                "yellow_flags": analytics.get('yellow_flags', 0)
            },
            "insights": insights,
            "pdf_path": f"/static/{os.path.basename(analytics['pdf_report_path'])}" if 'pdf_report_path' in analytics and analytics['pdf_report_path'] else None
        }
        
        logger.info(f"Analysis complete for {identifier}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files."""
    file_path = output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/pdf'
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
