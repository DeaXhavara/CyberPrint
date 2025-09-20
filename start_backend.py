#!/usr/bin/env python3
"""
Start the CyberPrint FastAPI backend server
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
