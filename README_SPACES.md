# CyberPrint - Hugging Face Spaces Deployment

CyberPrint is a sentiment analysis application using DeBERTa transformer model.

## Live Demo
🚀 **Live on Hugging Face Spaces:** https://huggingface.co/spaces/deabyte/cyberprint

## Environment Variables Required

Add these to your Hugging Face Space settings under "Variables and secrets":

```
REDDIT_CLIENT_ID = your_reddit_client_id
REDDIT_CLIENT_SECRET = your_reddit_client_secret
REDDIT_USERNAME = your_reddit_username
REDDIT_PASSWORD = your_reddit_password
REDDIT_USER_AGENT = CyberPrint/0.1 by your_username
YOUTUBE_API_KEY = your_youtube_api_key
```

## Deployment Notes

- Uses Docker with port 7860 (Hugging Face Spaces standard)
- Requires Git LFS for large DeBERTa model files (737MB)
- Auto-syncs from GitHub via GitHub Actions
- Updated: Sync triggered via GitHub Actions + React frontend
- 16GB RAM on CPU Basic tier (free)
