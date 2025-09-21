# CyberPrint Hugging Face Spaces Deployment

## Environment Variables Required

Add these to your Hugging Face Space settings:

```
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USERNAME=your_reddit_username_here
REDDIT_PASSWORD=your_reddit_password_here
REDDIT_USER_AGENT=CyberPrint/0.1 by YourUsername
YOUTUBE_API_KEY=your_youtube_api_key_here
```

## Deployment Notes

- Uses port 7860 (Hugging Face Spaces standard)
- Git LFS enabled for DeBERTa model files (737MB)
- Full-stack deployment: FastAPI backend + React frontend
- 16GB RAM on CPU Basic tier (free)
