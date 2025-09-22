<div align="center">

  <img width="200" height="200" alt="CyberPrint" src="https://github.com/user-attachments/assets/40e8edcd-50ac-479b-ba03-d0ec97e2f551" />

  <h3><i>CyberPrint - Your Online Behavior, Decoded</i></h3>

</div>



## Table of Contents
1. [Quick Start for Judges](#quick-start-for-judges)  
2. [Inspiration](#inspiration)  
3. [What it does](#what-it-does)  
4. [How I built it](#how-i-built-it)  
   - [AI/ML Architecture](#aiml-architecture)  
   - [Backend Infrastructure](#backend-infrastructure)  
   - [Frontend Experience](#frontend-experience)  
   - [Technical Architecture](#technical-architecture)  
5. [Challenges I ran into](#challenges-i-ran-into)  
6. [Accomplishments that I'm proud of](#accomplishments-that-im-proud-of)  
7. [What I learned](#what-i-learned)  
8. [What's next for CyberPrint](#whats-next-for-cyberprint)  

---

## Quick Start for Judges

### Prerequisites
- Python 3.8+ 
- Node.js 16+ and npm
- Git with Git LFS enabled

### 1. Clone and Setup
```bash
git clone https://github.com/DeaXhavara/CyberPrint.git
cd CyberPrint
git lfs pull  # Download large model files
```

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables (optional - app works without API keys)
cp .env.example .env
# Edit .env with your Reddit/YouTube API keys if you want to test those features
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### 4. Run the Application
```bash
# Terminal 1: Start backend server
python server.py

# Terminal 2: Start frontend (in a new terminal)
cd frontend
npm start
```

### 5. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Quick Test
1. Open http://localhost:3000
2. Enter any Reddit URL (e.g., "https://www.reddit.com/user/PukTuki") or YouTube channel("https://www.youtube.com/user/PewDiePie")
3. Click "Analyze" to see the sentiment analysis in action
4. Download the generated PDF report

### Troubleshooting
- **Model loading issues**: The app automatically falls back to logistic regression if DeBERTa models fail
- **API rate limits**: The app works with sample data even without API keys
- **Port conflicts**: Change ports in `server.py` (backend) or `package.json` (frontend)

---

## Inspiration
While browsing social media, I encountered an individual who consistently made negative comments about various creators. Each creator took these comments seriously and attempted to provide explanations, yet the commenter’s behavior remained consistently negative.  

This prompted me to investigate the existence of online tools that could analyze such behavior and potentially assist creators in not taking it as a personal attack.  

In our hyper-connected world, our digital communications reveal profound insights about our mental state and well-being. **CyberPrint** was born from the recognition that millions of people leave digital breadcrumbs (comments, posts, interactions) that collectively paint a picture of their emotional landscape.  

The goal: analyze these patterns and provide **meaningful insights** for digital wellbeing, mental health awareness, and personal growth.  

---

## What it does
CyberPrint is an advanced **sentiment analysis platform** that analyzes users' digital communications across Reddit and YouTube to generate **comprehensive mental health and wellbeing reports**.  

Key features:
- **Analyzes Digital Footprints** → Processes Reddit & YouTube comments  
- **Sentiment Classification** → DeBERTa transformer models with >92% accuracy  
- **Granular Sub-Label Detection** → Gratitude, sarcasm, concern, excitement, etc.  
- **Mental Health Monitoring** → Detects potential concerns & provides resources  
- **Personalized PDF Reports** → Professional, visually appealing insights  
- **Dual Perspective Analysis** → For both *self-reflection* and *understanding others*

<img width="300" height="500" alt="Screenshot 2025-09-20 at 9 56 06 PM" src="https://github.com/user-attachments/assets/7b02d543-344b-457d-8104-4c9567c60e79" /> <img width="300" height="500" alt="Screenshot 2025-09-20 at 9 56 30 PM" src="https://github.com/user-attachments/assets/cdaff081-9094-4bf6-a9a0-cf5e346916c0" />

<img width="300" height="500" alt="Screenshot 2025-09-20 at 9 57 08 PM" src="https://github.com/user-attachments/assets/6eff2ca4-69c0-4f2f-8137-dfc95c342341" /> <img width="300" height="500" alt="Screenshot 2025-09-20 at 9 57 37 PM" src="https://github.com/user-attachments/assets/9a51a0fa-145a-47e0-b82a-ef1257a59f17" />



---

## How I built it

### AI/ML Architecture
- Fine-tuned **DeBERTa** model → 94–97% sentiment classification accuracy  
- **Fallback System** → Logistic regression for reliability  
- **Sub-Label Classification** → Rule-based for fine-grained emotions  
- **Active Learning Pipeline** → Improves continuously via misclassification detection  

### Backend Infrastructure
- **FastAPI** → High-performance API  
- **Text Processing** → Robust preprocessing & batch processing  
- **APIs** → Reddit API + YouTube Data API v3  
- **PDF Generation** → Professional reports with ReportLab  

### Frontend Experience
- **React** → Responsive single-page app  
- **Animated UI** → Smooth, engaging experience  
- **Real-time Analysis** → Progress indicators for live feedback  
- **Mobile-Responsive** → Optimized across devices  

### Technical Architecture
- **Microservices Design** → Separate backend and frontend services  
- **Docker Containerization** → Consistent development environments  
- **Environment Management** → Secure API key handling  
- **CORS Configuration** → Seamless frontend-backend communication  

---

## Challenges I ran into

### Model Performance Optimization
- **Problem**: Initial accuracy inconsistencies across different text types  
- **Cause**: Training data imbalance and model bias  
- **Solution**: Enhanced rule-based fallback system with comprehensive keyword matching  

### Gratitude Classification Bias
- **Problem**: Model mislabeled gratitude expressions as neutral  
- **Cause**: Training data imbalance toward neutral predictions  
- **Solution**: Post-processing override system for gratitude detection  

### Multi-Service Architecture
- **Problem**: Coordinating React frontend with FastAPI backend  
- **Solution**: Proper CORS configuration and environment management  

---

## Accomplishments that I'm proud of
- Advanced sentiment analysis with DeBERTa transformer model  
- Sophisticated gratitude detection and bias correction system  
- Beautiful, share-worthy PDF reports with professional styling  
- Active learning pipeline for continuous improvement  
- Reddit + YouTube API integration for real-world data  
- Comprehensive sub-label classification system  
- Seamless UX from input → full analysis  
- Designed CyberPrint logo in Canva  
- Most importantly: **I brought my vision to life**  

---

## What I learned
- Advanced transformer fine-tuning (DeBERTa)  
- ML model optimization and fallback strategies  
- Bias detection & mitigation in sentiment analysis  
- Full-stack integration (React + FastAPI)  
- Designing intuitive user experiences  
- Building responsibly for mental health & wellbeing  

---

## What's next for CyberPrint

### Enhanced AI Capabilities
- Multi-language support  
- Temporal analysis → track sentiment over time  
- Specialized models → depression/anxiety detection  

### Platform Expansion
- Twitter/X integration  
- Instagram & TikTok comment analysis  
- LinkedIn career sentiment insights  

### Enterprise Features
- Team/Org-level analytics  
- Developer API access  
- Industry-specific custom models  

### Wellness Integration
- Personalized recommendations  
- Therapist dashboards  
- Long-term wellbeing tracking  


## Contributing

CyberPrint is an open project.  
If you find it useful, feel free to **fork it**, **star it**, or submit a **pull request**.  

Bug reports, feature suggestions, and improvements are always welcome! As my inaugural full-stack web application, I am eager to receive constructive feedback.

CyberPrint - Where AI meets digital well-being

