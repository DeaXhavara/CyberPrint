# CyberPrint

**Your online behavior, decoded**

---

## Table of Contents
1. [Inspiration](#inspiration)  
2. [What it does](#what-it-does)  
3. [How I built it](#how-i-built-it)  
   - [AI/ML Architecture](#aiml-architecture)  
   - [Backend Infrastructure](#backend-infrastructure)  
   - [Frontend Experience](#frontend-experience)  
   - [Deployment & DevOps](#deployment--devops)  
4. [Challenges I ran into](#challenges-i-ran-into)  
5. [Accomplishments that I'm proud of](#accomplishments-that-im-proud-of)  
6. [What I learned](#what-i-learned)  
7. [What's next for CyberPrint](#whats-next-for-cyberprint)  

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

### Deployment & DevOps
- **Railway Deployment** → Dual service (backend + frontend)  
- **Docker** → Consistent environments  
- **Secure Env Management** → API keys & configs  
- **CORS Config** → Seamless frontend-backend communication  

---

## Challenges I ran into

### Model Performance Crisis
- **Problem**: Accuracy collapsed from 94–97% → 38% during deployment  
- **Cause**: DeBERTa model files (737MB+) weren’t available in production  
- **Solution**: Added fallback + optimized deployment strategy  

### Gratitude Classification Bias
- **Problem**: Model mislabeled gratitude as neutral  
- **Cause**: Training data imbalance  
- **Solution**: Post-processing override system for gratitude detection  

### Deployment Complexity
- **Problem**: Multi-service architecture (React + FastAPI) was tough  
- **Solution**: Dual Railway services + proper environment management  

---

## Accomplishments that I'm proud of
- ~94% sentiment accuracy with DeBERTa  
- Production-ready deployment on Railway  
- Sophisticated gratitude detection system  
- Beautiful, share-worthy PDF reports  
- Active learning pipeline for continuous improvement  
- Reddit + YouTube integration  
- Seamless UX from input → full analysis  
- Designed CyberPrint logo in Canva  
- Most importantly: **I brought my vision to life**  

---

## What I learned
- Advanced transformer fine-tuning (DeBERTa)  
- Production ML deployment strategies  
- Bias detection & mitigation  
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

