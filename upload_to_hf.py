#!/usr/bin/env python3
"""
Upload CyberPrint to Hugging Face Space directly
"""
import os
from huggingface_hub import HfApi

def upload_to_space():
    # Get token from environment or prompt
    token = os.getenv('HF_TOKEN')
    if not token:
        token = input("Enter your Hugging Face token: ").strip()
        if not token:
            print("❌ No token provided!")
            return False
    
    # Initialize API
    api = HfApi(token=token)
    
    # Upload entire folder to Space
    print("🚀 Uploading CyberPrint to Hugging Face Space...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id="deabyte/cyberprint",
            repo_type="space",
            ignore_patterns=[
                ".git/*",
                ".github/*", 
                "__pycache__/*",
                "*.pyc",
                ".env*",
                "frontend/node_modules/*",
                "frontend/.cache/*",
                "frontend/build/*",
                ".DS_Store",
                "upload_to_hf.py",
                "**/.cache/*",
                "**/node_modules/*"
            ]
        )
        print("✅ Successfully uploaded to https://huggingface.co/spaces/deabyte/cyberprint")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    upload_to_space()
