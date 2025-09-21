# CyberPrint - Hugging Face Spaces Deployment

CyberPrint is a sentiment analysis application using DeBERTa transformer model.

## Live Demo
🚀 **Live on Hugging Face Spaces:** https://huggingface.co/spaces/deabyte/cyberprint

## Environment Variables Required

Add these to your Hugging Face Space settings under "Variables and secrets":

```
REDDIT_CLIENT_ID = HeasTXaPckYo0kbnvhcKig
REDDIT_CLIENT_SECRET = NTA26_bJtjiG2-GzFpi1UJyAX1ZV9g
REDDIT_USERNAME = SearchTraditional695
REDDIT_PASSWORD = SearchTraditional695
REDDIT_USER_AGENT = CyberPrint/0.1 by SearchTraditional695
YOUTUBE_API_KEY = AIzaSyCYEaEIo1e2DSzR2B0-vJGoP4MZEtbFNPA
```

## Deployment Notes

- Uses Docker with port 7860 (Hugging Face Spaces standard)
- Requires Git LFS for large DeBERTa model files (737MB)
- Auto-syncs from GitHub via GitHub Actions
- Updated: Sync triggered via GitHub Actions + React frontend
- 16GB RAM on CPU Basic tier (free)
