#!/usr/bin/env python3
"""
CyberPrint Main Pipeline
========================

This is the main pipeline for CyberPrint that:
1. Loads and merges all datasets into a clean database
2. Deduplicates entries using normalized text
3. Classifies comments into main sentiment labels and sub-labels
4. Generates PDF reports per user/channel with analytics and insights
5. Provides instructor-ready analytics with counts and examples

Main sentiment categories:
- Positive: Gratitude, Compliments, Positive Actions
- Negative: Toxic, Personal Attacks, Harsh Language
- Neutral: Fact-based, Question-based, Unbiased
- Yellow Flag: Irony, Sarcasm, Humor
- Mental Health: Emotional distress (warning flag)

Sub-labels provide granular classification within each main category.
"""

import os
import pandas as pd
import numpy as np
import re
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from pathlib import Path
import json
from datetime import datetime

# Import our dataset builder
from .data.build_balanced_dataset import CyberPrintDatasetBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubLabelClassifier:
    """Classifies text into detailed sub-labels within main sentiment categories."""
    
    def __init__(self):
        # Define sub-label patterns
        self.sub_label_patterns = {
            'positive': {
                'gratitude': [
                    'thank', 'thanks', 'grateful', 'appreciate', 'blessed', 'gratitude',
                    'thankful', 'indebted', 'obliged', 'acknowledge'
                ],
                'compliments': [
                    'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'great',
                    'wonderful', 'outstanding', 'superb', 'marvelous', 'fabulous',
                    'terrific', 'incredible', 'phenomenal', 'exceptional', 'remarkable',
                    'splendid', 'magnificent', 'perfect', 'beautiful', 'lovely'
                ],
                'positive_actions': [
                    'help', 'support', 'assist', 'encourage', 'motivate', 'inspire',
                    'uplift', 'boost', 'cheer', 'comfort', 'reassure', 'validate',
                    'recognize', 'celebrate', 'congratulate', 'praise', 'commend'
                ],
                'joy_happiness': [
                    'happy', 'joy', 'excited', 'thrilled', 'delighted', 'pleased',
                    'satisfied', 'content', 'cheerful', 'upbeat', 'positive', 'optimistic'
                ]
            },
            'negative': {
                'toxic': [
                    'hate', 'despise', 'loathe', 'disgusting', 'revolting', 'vile',
                    'disgust', 'repulsive', 'abhorrent', 'detestable'
                ],
                'personal_attacks': [
                    'stupid', 'idiot', 'moron', 'dumb', 'fool', 'loser', 'pathetic',
                    'worthless', 'useless', 'incompetent', 'failure'
                ],
                'harsh_language': [
                    'damn', 'hell', 'crap', 'sucks', 'terrible', 'awful', 'horrible',
                    'disgusting', 'nasty', 'vile', 'revolting'
                ],
                'anger_frustration': [
                    'angry', 'mad', 'furious', 'rage', 'frustrated', 'annoyed',
                    'irritated', 'pissed', 'livid', 'outraged'
                ]
            },
            'neutral': {
                'fact_based': [
                    'according to', 'research shows', 'studies indicate', 'data suggests',
                    'statistics show', 'evidence shows', 'facts', 'information',
                    'based on', 'according to the', 'it is', 'there is', 'there are'
                ],
                'question_based': [
                    'what', 'how', 'why', 'when', 'where', 'who', 'which', '?',
                    'can you', 'could you', 'would you', 'do you', 'are you',
                    'is it', 'was it', 'will it', 'should i', 'could i'
                ],
                'unbiased': [
                    'neutral', 'objective', 'impartial', 'fair', 'balanced',
                    'unbiased', 'unprejudiced', 'dispassionate'
                ],
                'informational': [
                    'update', 'news', 'report', 'announcement', 'notice',
                    'information', 'details', 'specifications', 'description'
                ]
            },
            'yellow_flag': {
                'irony': [
                    'oh great', 'wonderful', 'perfect', 'exactly what i needed',
                    'just what i wanted', 'fantastic', 'brilliant', 'amazing'
                ],
                'sarcasm': [
                    'sure', 'right', 'of course', 'obviously', 'clearly',
                    'totally', 'absolutely', 'definitely', 'certainly'
                ],
                'humor': [
                    'lol', 'haha', 'funny', 'hilarious', 'comedy', 'joke',
                    'humor', 'wit', 'sarcastic', 'ironic', 'satirical'
                ],
                'skeptical': [
                    'really', 'seriously', 'come on', 'give me a break',
                    'you must be kidding', 'no way', 'unbelievable'
                ]
            },
            'mental_health': {
                'emotional_distress': [
                    'sad', 'depressed', 'hopeless', 'despair', 'lonely', 'isolated',
                    'empty', 'numb', 'broken', 'hurt', 'pain', 'suffering'
                ],
                'anxiety': [
                    'anxious', 'worried', 'nervous', 'scared', 'afraid', 'fear',
                    'panic', 'overwhelmed', 'stressed', 'tense', 'uneasy'
                ],
                'crisis': [
                    'suicide', 'kill myself', 'end it all', 'give up', 'no point',
                    'worthless', 'burden', 'better off without', 'end my life'
                ],
                'seeking_help': [
                    'help me', 'need help', 'support', 'therapy', 'counseling',
                    'treatment', 'recovery', 'healing', 'getting better'
                ]
            }
        }
    
    def classify_sub_label(self, text: str, main_sentiment: str) -> str:
        """Classify text into a sub-label within the main sentiment category."""
        if main_sentiment not in self.sub_label_patterns:
            return 'general'
        
        text_lower = text.lower()
        sentiment_patterns = self.sub_label_patterns[main_sentiment]
        
        # Score each sub-label
        sub_label_scores = {}
        for sub_label, patterns in sentiment_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1
            sub_label_scores[sub_label] = score
        
        # Return the sub-label with highest score, or 'general' if no matches
        if sub_label_scores and max(sub_label_scores.values()) > 0:
            return max(sub_label_scores, key=sub_label_scores.get)
        else:
            return 'general'

class CyberPrintPipeline:
    """Main CyberPrint pipeline for dataset processing and report generation."""
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.output_dir = output_dir or os.path.join(self.data_dir, "output")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analytics"), exist_ok=True)
        
        # Initialize components
        self.dataset_builder = CyberPrintDatasetBuilder()
        self.sub_label_classifier = SubLabelClassifier()
        
        # Load or build the main dataset
        self.main_dataset = None
        self.load_main_dataset()
    
    def load_main_dataset(self):
        """Load the main balanced dataset."""
        dataset_path = os.path.join(self.data_dir, "processed", "final_database_clean.csv")
        
        if os.path.exists(dataset_path):
            logger.info("Loading existing main dataset...")
            self.main_dataset = pd.read_csv(dataset_path)
        else:
            logger.info("Building main dataset...")
            self.main_dataset = self.dataset_builder.build_dataset()
            self.dataset_builder.save_dataset(self.main_dataset)
        
        logger.info(f"Loaded main dataset with {len(self.main_dataset)} examples")
    
    def add_sub_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sub-labels to the dataset."""
        logger.info("Adding sub-labels to dataset...")
        
        df = df.copy()
        df['sub_label'] = df.apply(
            lambda row: self.sub_label_classifier.classify_sub_label(
                row['tweet'], row['sentiment']
            ), axis=1
        )
        
        return df
    
    def predict_sentiments(self, df: pd.DataFrame, enable_gpt_enhancement: bool = False) -> pd.DataFrame:
        """Predict sentiments for the user data using the trained ML model."""
        logger.info("Predicting sentiments with enhanced sub-label classification...")
        
        try:
            # Import the ML predictor
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from cyberprint_ml_predictor import predict_text
            
            df = df.copy()
            texts = df['tweet'].tolist()
            
            # Enhanced prediction with sub-labels
            predictions = predict_text(
                texts, 
                include_sub_labels=True,
                confidence_threshold=0.6
            )
            
            sentiments = []
            sub_labels = []
            sub_label_confidences = []
            mental_health_alerts = []
            yellow_flags = []
            confidence_scores = []
            applied_rules_list = []
            enhanced_flags = []
            
            for i, pred in enumerate(predictions):
                predicted_label = pred['predicted_label']
                predicted_score = pred['predicted_score']
                sub_label = pred.get('sub_label', 'general')
                sub_label_confidence = pred.get('sub_label_confidence', 0.0)
                applied_rules = pred.get('applied_rules', [])
                enhanced = pred.get('enhanced', False)
                
                sentiments.append(predicted_label)
                sub_labels.append(sub_label)
                sub_label_confidences.append(sub_label_confidence)
                confidence_scores.append(predicted_score)
                applied_rules_list.append(applied_rules)
                enhanced_flags.append(enhanced)
                
                # Enhanced mental health detection (disabled for YouTube)
                mental_health_alert = 0
                
                # Only apply mental health detection for non-YouTube platforms
                if not (hasattr(self, '_current_platform') and self._current_platform == 'youtube'):
                    text_lower = texts[i].lower()
                    
                    # Check for threatening sub-label or high-confidence negative with mental health keywords
                    if (sub_label == 'threatening' or 
                        (predicted_label == 'negative' and predicted_score > 0.7) or
                        any(keyword in text_lower for keyword in [
                            'depressed', 'suicide', 'kill myself', 'hopeless', 'worthless',
                            'end it all', 'not worth living', 'want to die', 'harm myself'
                        ])):
                        mental_health_alert = 1
                mental_health_alerts.append(mental_health_alert)
                
                yellow_flag = 1 if predicted_label == 'yellow_flag' else 0
                yellow_flags.append(yellow_flag)
                
                # Store probability distributions
                for label in ['positive', 'negative', 'neutral', 'yellow_flag']:
                    col_name = f'{label}_prob'
                    if col_name not in df.columns:
                        df[col_name] = 0.0
                    df.loc[i, col_name] = pred['probs'].get(label, 0.0)
            
            # Add all new columns
            df['sentiment'] = sentiments
            df['sub_label'] = sub_labels
            df['sub_label_confidence'] = sub_label_confidences
            df['confidence_score'] = confidence_scores
            df['mental_health_alert'] = mental_health_alerts
            df['yellow_flag'] = yellow_flags
            df['applied_rules'] = applied_rules_list
            df['gpt_enhanced'] = enhanced_flags
            
            # Log rule application statistics
            total_rules_applied = sum(len(rules) for rules in applied_rules_list)
            enhanced_count = sum(enhanced_flags)
            
            logger.info(f"Successfully predicted sentiments for {len(df)} comments")
            logger.info(f"Sub-label rules applied: {total_rules_applied} total rules")
            # GPT-OSS removed from system
            
            # Log sub-label distribution
            sub_label_dist = pd.Series(sub_labels).value_counts()
            logger.info(f"Sub-label distribution: {dict(sub_label_dist)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error predicting sentiments: {e}")
            import traceback
            traceback.print_exc()
            df = df.copy()
            df['sentiment'] = 'neutral'
            df['sub_label'] = 'general'
            df['sub_label_confidence'] = 0.0
            df['confidence_score'] = 0.0
            df['mental_health_alert'] = 0
            df['yellow_flag'] = 0
            df['applied_rules'] = [[] for _ in range(len(df))]
            df['gpt_enhanced'] = [False for _ in range(len(df))]
            for label in ['positive', 'negative', 'neutral', 'yellow_flag']:
                df[f'{label}_prob'] = [0.0 for _ in range(len(df))]
            return df
    
    def process_user_data(self, user_data: pd.DataFrame, user_id: str, platform: str = "unknown") -> Dict[str, Any]:
        """Process data for a specific user/channel and generate analytics."""
        logger.info(f"Processing data for user {user_id} on {platform}")
        logger.info(f"Input DataFrame has {len(user_data)} rows")
        
        # Set current platform for mental health detection logic
        self._current_platform = platform
        
        # Predict sentiments using the ML model (includes sub-labels)
        user_data = self.predict_sentiments(user_data)
        logger.info(f"After sentiment prediction: {len(user_data)} rows")
        
        # Generate analytics
        analytics = {
            'user_id': user_id,
            'platform': platform,
            'total_comments': len(user_data),
            'timestamp': datetime.now().isoformat(),
            'sentiment_distribution': user_data['sentiment'].value_counts().to_dict(),
            'sub_label_distribution': user_data['sub_label'].value_counts().to_dict(),
            'mental_health_warnings': int(user_data['mental_health_alert'].sum()) if 'mental_health_alert' in user_data.columns else 0,
            'yellow_flags': int(user_data['yellow_flag'].sum()) if 'yellow_flag' in user_data.columns else 0,
            'sentiment_by_sub_label': {},
            'sample_comments': {},
            'insights': []
        }
        
        logger.info(f"Analytics generated - Total comments: {analytics['total_comments']}")
        logger.info(f"Sentiment distribution: {analytics['sentiment_distribution']}")
        
        # Generate sentiment by sub-label breakdown
        for sentiment in user_data['sentiment'].unique():
            sentiment_data = user_data[user_data['sentiment'] == sentiment]
            analytics['sentiment_by_sub_label'][sentiment] = sentiment_data['sub_label'].value_counts().to_dict()
        
        # Generate sample comments for each sentiment/sub-label combination
        for sentiment in user_data['sentiment'].unique():
            sentiment_data = user_data[user_data['sentiment'] == sentiment]
            for sub_label in sentiment_data['sub_label'].unique():
                sub_data = sentiment_data[sentiment_data['sub_label'] == sub_label]
                if len(sub_data) > 0:
                    key = f"{sentiment}_{sub_label}"
                    analytics['sample_comments'][key] = sub_data['tweet'].head(3).tolist()
        
        print("\nGenerating insights...")
        analytics['insights'] = self.generate_insights(analytics)
        
        return analytics
    
    def generate_insights(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate personalized insights and recommendations."""
        insights = []
        
        total = analytics['total_comments']
        mental_health = analytics['mental_health_warnings']
        yellow_flags = analytics['yellow_flags']
        
        # Mental health insights (disabled for YouTube)
        platform = analytics.get('platform', 'unknown')
        if mental_health > 0 and platform != 'youtube':
            mental_health_pct = (mental_health / total) * 100
            if mental_health_pct > 10:
                insights.append(f"HIGH MENTAL HEALTH ALERT: {mental_health_pct:.1f}% of comments show signs of emotional distress. Consider immediate intervention.")
            elif mental_health_pct > 5:
                insights.append(f"Mental health concerns detected in {mental_health_pct:.1f}% of comments. Monitor closely and offer support resources.")
            else:
                insights.append(f"Mental health monitoring: {mental_health} comments flagged for review.")
        
        # Sentiment insights
        sentiment_dist = analytics['sentiment_distribution']
        if 'negative' in sentiment_dist:
            negative_pct = (sentiment_dist['negative'] / total) * 100
            if negative_pct > 30:
                if platform == "youtube":
                    insights.append(f"High negative feedback: {negative_pct:.1f}% negative comments. Review content strategy and community management approach.")
                else:
                    insights.append(f"High negative sentiment: {negative_pct:.1f}% negative comments. Consider content moderation strategies.")
            elif negative_pct > 15:
                insights.append(f"Moderate negative sentiment: {negative_pct:.1f}% negative comments. Monitor for escalation.")
        
        if 'positive' in sentiment_dist:
            positive_pct = (sentiment_dist['positive'] / total) * 100
            if positive_pct > 50:
                insights.append(f"Excellent positive engagement: {positive_pct:.1f}% positive comments. Great community health!")
            elif positive_pct > 30:
                insights.append(f"Good positive sentiment: {positive_pct:.1f}% positive comments. Maintain current approach.")
        
        # Yellow flag insights
        if yellow_flags > 0:
            yellow_pct = (yellow_flags / total) * 100
            if yellow_pct > 20:
                insights.append(f"High sarcasm/irony detected: {yellow_pct:.1f}% of comments. Monitor for potential misunderstandings.")
            else:
                insights.append(f"Sarcasm/irony present: {yellow_flags} comments flagged. Context may be important.")
        
        # Engagement recommendations
        if total < 10:
            insights.append("Low activity: Consider strategies to increase engagement and participation.")
        elif total > 1000:
            insights.append("High activity: Consider automated moderation tools for efficient management.")
        
        return insights
    
    def generate_pdf_report(self, analytics: Dict[str, Any], user_data: pd.DataFrame, output_path: str = None):
        """Generate a PDF report with the specified layout."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, PageTemplate, Frame
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
            from reportlab.graphics.shapes import Drawing, Rect
            from reportlab.graphics.charts.barcharts import VerticalBarChart
            from reportlab.graphics import renderPDF
            from reportlab.platypus.doctemplate import BaseDocTemplate
            from reportlab.platypus.frames import Frame
            
            if output_path is None:
                output_path = os.path.join(
                    self.output_dir, "cyberprint_report.pdf"
                )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create custom PDF document with footer
            class FooterDocTemplate(BaseDocTemplate):
                def __init__(self, filename, **kwargs):
                    BaseDocTemplate.__init__(self, filename, **kwargs)
                    
                def afterPage(self):
                    """Add footer to each page"""
                    self.canv.saveState()
                    
                    # Footer line
                    self.canv.setStrokeColor(colors.Color(0.7, 0.7, 0.7))
                    self.canv.setLineWidth(0.5)
                    self.canv.line(0.8*inch, 0.6*inch, A4[0]-0.8*inch, 0.6*inch)
                    
                    # Footer text - left side
                    self.canv.setFont('Times-Roman', 9)
                    self.canv.setFillColor(colors.Color(0.5, 0.5, 0.5))
                    self.canv.drawString(0.8*inch, 0.4*inch, "CyberPrint Analysis Report")
                    
                    # Footer text - right side (page number)
                    page_num = self.canv.getPageNumber()
                    self.canv.drawRightString(A4[0]-0.8*inch, 0.4*inch, f"Page {page_num}")
                    
                    self.canv.restoreState()
            
            # Create document with custom footer template
            doc = FooterDocTemplate(output_path, pagesize=A4, 
                                  topMargin=0.7*inch, bottomMargin=1.0*inch,
                                  leftMargin=0.8*inch, rightMargin=0.8*inch)
            
            # Add page template with frame
            frame = Frame(0.8*inch, 1.0*inch, A4[0]-1.6*inch, A4[1]-1.7*inch, 
                         leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
            template = PageTemplate(id='normal', frames=frame)
            doc.addPageTemplates([template])
            
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles with pastel colors and fancy fonts
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=32,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.Color(0.2, 0.3, 0.5),  # Deep pastel blue
                fontName='Times-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=20,
                spaceAfter=20,
                spaceBefore=15,
                textColor=colors.Color(0.3, 0.4, 0.6),  # Soft blue-gray
                fontName='Times-Bold',
                borderWidth=0,
                borderPadding=8,
                backColor=colors.Color(0.95, 0.97, 0.99)  # Very light blue background
            )
            
            profile_style = ParagraphStyle(
                'ProfileStyle',
                parent=styles['Normal'],
                fontSize=13,
                spaceAfter=15,
                spaceBefore=10,
                alignment=TA_CENTER,
                backColor=colors.Color(0.96, 0.98, 0.96),  # Very light mint green
                borderColor=colors.Color(0.7, 0.85, 0.7),  # Soft green border
                borderWidth=1,
                borderPadding=15,
                fontName='Times-Roman',
                textColor=colors.Color(0.2, 0.4, 0.3)  # Dark green text
            )
            
            # Add small CyberPrint logo at the very top right
            logo_path = '/Users/deaxhavara/CyberPrint/frontend/public/cyberprintblack.png'
            
            if os.path.exists(logo_path):
                try:
                    # Create a small logo and position it at the very top right
                    logo = Image(logo_path, width=1.0*inch, height=1.0*inch)
                    
                    # Create a table to position logo on the right with minimal space
                    logo_table = Table([['', logo]], colWidths=[5.0*inch, 1.0*inch])
                    logo_table.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 0),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                        ('TOPPADDING', (0, 0), (-1, -1), 0),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                    ]))
                    
                    story.append(logo_table)
                except Exception as e:
                    print(f"DEBUG: Logo failed with error: {e}")
            
            # Compact title section with minimal spacing
            compact_title_style = ParagraphStyle(
                'CompactTitle',
                parent=styles['Heading1'],
                fontSize=28,
                spaceAfter=8,
                spaceBefore=5,
                alignment=TA_CENTER,
                textColor=colors.Color(0.2, 0.3, 0.5),
                fontName='Times-Bold'
            )
            
            story.append(Paragraph("CyberPrint Report", compact_title_style))
            story.append(Paragraph("Advanced Sentiment Analysis & Digital Wellbeing Assessment", 
                                 ParagraphStyle('subtitle', alignment=TA_CENTER, fontSize=12, 
                                              textColor=colors.Color(0.4, 0.5, 0.6), 
                                              fontName='Times-Italic', spaceAfter=12)))
            story.append(Spacer(1, 8))
            
            # Profile info section
            profile_info = f"""
            <b>Profile URL:</b> {analytics.get('profile_url', 'N/A')}<br/>
            <b>Platform:</b> {analytics['platform'].title()}<br/>
            <b>User/Channel:</b> {analytics['user_id']}<br/>
            <b>Analysis Date:</b> {analytics['timestamp'][:10]}
            """
            story.append(Paragraph(profile_info, profile_style))
            story.append(Spacer(1, 15))
            
            # Sentiment cards section - create 4 colored boxes with pastel colors
            sentiment_colors = {
                'positive': colors.Color(0.7, 0.9, 0.7),    # Soft pastel green
                'negative': colors.Color(0.95, 0.7, 0.7),  # Soft pastel pink/red
                'neutral': colors.Color(0.85, 0.9, 0.95),  # Soft pastel blue-gray
                'yellow_flag': colors.Color(0.95, 0.9, 0.6)  # Soft pastel yellow
            }
            
            # Text colors for better contrast on pastel backgrounds
            sentiment_text_colors = {
                'positive': colors.Color(0.2, 0.5, 0.2),    # Dark green
                'negative': colors.Color(0.6, 0.2, 0.2),   # Dark red
                'neutral': colors.Color(0.3, 0.4, 0.5),    # Dark blue-gray
                'yellow_flag': colors.Color(0.6, 0.5, 0.1) # Dark yellow/brown
            }
            
            sentiment_descriptions = {
                'positive': 'Indicators of encouragement, gratitude, and positivity',
                'negative': 'Possible offensive, harmful, or critical content',
                'neutral': 'Factual, unbiased, or informational content',
                'yellow_flag': 'Sarcasm, irony, or complex sentiment patterns'
            }
            
            # Create sentiment cards table
            cards_data = []
            row1 = []
            row2 = []
            
            for sentiment in ['positive', 'negative', 'neutral', 'yellow_flag']:
                count = analytics['sentiment_distribution'].get(sentiment, 0)
                percentage = (count / analytics['total_comments']) * 100 if analytics['total_comments'] > 0 else 0
                
                card_content = f"""
                <b><font size="14" face="Times-Bold">{sentiment.replace('_', ' ').title()}: {count}</font></b><br/>
                <font size="10" face="Times-Roman">{sentiment_descriptions[sentiment]}</font><br/>
                <font size="12" face="Times-Bold">{percentage:.1f}%</font>
                """
                
                if len(row1) < 2:
                    row1.append(Paragraph(card_content, styles['Normal']))
                else:
                    row2.append(Paragraph(card_content, styles['Normal']))
            
            cards_data = [row1, row2]
            cards_table = Table(cards_data, colWidths=[2.8*inch, 2.8*inch])
            
            # Apply pastel colors to cards with better styling
            card_styles = [
                ('BACKGROUND', (0, 0), (0, 0), sentiment_colors['positive']),
                ('BACKGROUND', (1, 0), (1, 0), sentiment_colors['negative']),
                ('BACKGROUND', (0, 1), (0, 1), sentiment_colors['neutral']),
                ('BACKGROUND', (1, 1), (1, 1), sentiment_colors['yellow_flag']),
                # Individual text colors for better contrast
                ('TEXTCOLOR', (0, 0), (0, 0), sentiment_text_colors['positive']),
                ('TEXTCOLOR', (1, 0), (1, 0), sentiment_text_colors['negative']),
                ('TEXTCOLOR', (0, 1), (0, 1), sentiment_text_colors['neutral']),
                ('TEXTCOLOR', (1, 1), (1, 1), sentiment_text_colors['yellow_flag']),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Soft borders instead of harsh grid
                ('BOX', (0, 0), (-1, -1), 1, colors.Color(0.8, 0.8, 0.8)),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.Color(0.9, 0.9, 0.9)),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 20),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
                # Rounded corners effect with subtle shadow
                ('ROUNDEDCORNERS', (0, 0), (-1, -1), 5)
            ]
            
            cards_table.setStyle(TableStyle(card_styles))
            story.append(cards_table)
            story.append(Spacer(1, 30))
            
            # Bar chart section with better spacing
            story.append(Paragraph("Main Sentiments Distribution", heading_style))
            story.append(Spacer(1, 10))  # Reduced space before chart
            
            # Create prettier bar chart with pastel colors
            chart_data = [analytics['sentiment_distribution'].get(s, 0) for s in ['positive', 'negative', 'neutral', 'yellow_flag']]
            chart_labels = ['Positive', 'Negative', 'Neutral', 'Yellow Flags']
            
            drawing = Drawing(400, 200)  # Reduced height to fit better
            chart = VerticalBarChart()
            chart.x = 40
            chart.y = 40
            chart.height = 120  # Reduced height
            chart.width = 320
            chart.data = [chart_data]
            chart.categoryAxis.categoryNames = chart_labels
            chart.valueAxis.valueMin = 0
            chart.valueAxis.valueMax = max(chart_data) + 1 if chart_data else 1
            
            # Apply pastel colors to bars
            chart.bars[0].fillColor = colors.Color(0.7, 0.9, 0.85)  # Soft mint
            chart.categoryAxis.labels.fontName = 'Times-Roman'
            chart.categoryAxis.labels.fontSize = 9
            chart.valueAxis.labels.fontName = 'Times-Roman'
            chart.valueAxis.labels.fontSize = 8
            chart.valueAxis.gridStrokeColor = colors.Color(0.95, 0.95, 0.95)
            chart.valueAxis.gridStrokeWidth = 0.5
            
            drawing.add(chart)
            story.append(drawing)
            story.append(Spacer(1, 15))  # Reduced space after chart
            
            # Mental health warning section with enhanced styling (disabled for YouTube)
            if analytics.get('mental_health_warnings', 0) > 0 and analytics['platform'] != 'youtube':
                # Mental health warning box
                warning_style = ParagraphStyle(
                    'WarningStyle',
                    parent=styles['Normal'],
                    fontSize=13,
                    alignment=TA_CENTER,
                    backColor=colors.Color(0.99, 0.95, 0.95),  # Very soft pink
                    borderColor=colors.Color(0.9, 0.6, 0.6),   # Soft red border
                    borderWidth=2,
                    borderPadding=20,
                    spaceAfter=15,
                    fontName='Times-Roman',
                    textColor=colors.Color(0.6, 0.2, 0.2)  # Deep red text
                )
                
                warning_text = """
                <b>🚨 Mental Health Support Notice</b><br/><br/>
                Our analysis detected potential signs of emotional distress in some comments.<br/>
                This is a gentle reminder that support is available.<br/><br/>
                <b>Crisis Support:</b> Call 988 (US) or visit https://findahelpline.com<br/>
                <b>Remember:</b> You matter, and help is always available.
                """
                story.append(Paragraph(warning_text, warning_style))
                story.append(Spacer(1, 15))
            
            # YouTube-specific note
            if analytics['platform'] == 'youtube':
                youtube_style = ParagraphStyle(
                    'YouTubeStyle',
                    parent=styles['Normal'],
                    fontSize=13,
                    alignment=TA_CENTER,
                    backColor=colors.Color(0.92, 0.96, 0.99),  # Very soft blue
                    borderColor=colors.Color(0.7, 0.8, 0.9),   # Soft blue border
                    borderWidth=2,
                    borderPadding=18,
                    spaceAfter=25,
                    fontName='Times-Roman',
                    textColor=colors.Color(0.2, 0.3, 0.5)  # Deep blue text
                )
                
                youtube_text = """
                <b>YouTube Channel Analysis:</b> This report analyzes the feedback and comments that viewers have left on this channel's recent videos. The sentiment analysis reflects how the audience responds to and engages with this channel's content.
                """
                story.append(Paragraph(youtube_text, youtube_style))
                story.append(Spacer(1, 20))
            
            # Personalized recommendations section with better spacing
            story.append(Paragraph("Personalized Recommendations", heading_style))
            story.append(Spacer(1, 8))  # Reduced space
            
            # Generate personalized tips based on sentiment analysis
            tips_content = self._generate_personalized_tips(analytics)
            
            tips_style = ParagraphStyle(
                'TipsStyle',
                parent=styles['Normal'],
                fontSize=11,  # Slightly smaller font
                alignment=TA_LEFT,
                backColor=colors.Color(0.95, 0.98, 0.95),  # Very soft green
                borderColor=colors.Color(0.7, 0.85, 0.7),  # Soft green border
                borderWidth=1,
                borderPadding=12,  # Reduced padding
                spaceAfter=15,  # Reduced space
                fontName='Times-Roman',
                textColor=colors.Color(0.2, 0.4, 0.2),  # Dark green text
                leftIndent=8,
                rightIndent=8
            )
            
            story.append(Paragraph(tips_content, tips_style))
            story.append(Spacer(1, 15))  # Reduced space
            
            # Analysis section with better spacing
            story.append(Paragraph("Analysis of the Comments", heading_style))
            story.append(Spacer(1, 8))  # Reduced space
            
            # Get predictions with confidence scores
            try:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from cyberprint_ml_predictor import predict_text
                
                texts = user_data['tweet'].tolist()
                predictions = predict_text(texts)
                
                # Calculate overall confidence using the stored confidence scores
                if 'confidence_score' in user_data.columns:
                    overall_confidence = user_data['confidence_score'].mean() * 100
                else:
                    overall_confidence = np.mean([pred['predicted_score'] for pred in predictions]) * 100
                
                confidence_style = ParagraphStyle(
                    'ConfidenceStyle',
                    parent=styles['Normal'],
                    fontSize=15,
                    alignment=TA_CENTER,
                    backColor=colors.Color(0.93, 0.98, 0.93),  # Very soft green
                    borderColor=colors.Color(0.7, 0.85, 0.7),  # Soft green border
                    borderWidth=1,
                    borderPadding=15,
                    spaceAfter=25,
                    fontName='Times-Bold',
                    textColor=colors.Color(0.2, 0.5, 0.3)  # Forest green text
                )
                
                story.append(Paragraph(f"Confidence in the model result: {overall_confidence:.1f}%", confidence_style))
                story.append(Spacer(1, 12))  # Reduced space
                
                # Individual comments with enhanced predictions
                comment_heading_style = ParagraphStyle(
                    'CommentHeading',
                    parent=styles['Heading3'],
                    fontSize=16,
                    spaceAfter=15,
                    spaceBefore=10,
                    textColor=colors.Color(0.3, 0.4, 0.6),
                    fontName='Times-Bold',
                    backColor=colors.Color(0.97, 0.98, 0.99),
                    borderPadding=8
                )
                story.append(Paragraph("Individual Comment Analysis", comment_heading_style))
                story.append(Spacer(1, 8))  # Add small space after heading
                
                for i, (_, row) in enumerate(user_data.iterrows(), 1):
                    # Show all comments, not just first 10
                    pass
                    
                    text = row['tweet'][:150] + ('...' if len(row['tweet']) > 150 else '')
                    
                    # Use stored confidence, sentiment, and sub-label from the dataframe
                    if 'confidence_score' in row and 'sentiment' in row:
                        confidence = row['confidence_score'] * 100
                        sentiment = row['sentiment']
                        sub_label = row.get('sub_label', 'general')
                        sub_confidence = row.get('sub_label_confidence', 0.0) * 100
                        enhanced = row.get('gpt_enhanced', False)
                    else:
                        # Fallback to predictions if columns don't exist
                        pred = predictions[i-1] if i-1 < len(predictions) else {
                            'predicted_score': 0.0, 'predicted_label': 'neutral', 
                            'sub_label': 'general', 'sub_label_confidence': 0.0, 'enhanced': False
                        }
                        confidence = pred['predicted_score'] * 100
                        sentiment = pred['predicted_label']
                        sub_label = pred.get('sub_label', 'general')
                        sub_confidence = pred.get('sub_label_confidence', 0.0) * 100
                        enhanced = pred.get('enhanced', False)
                    
                    comment_text = f"{i}. {text}"
                    
                    # Enhanced label with sub-label information
                    main_sentiment_display = sentiment.replace('_', ' ').title()
                    sub_label_display = sub_label.replace('_', ' ').title()
                    
                    if sub_label != 'general':
                        label_text = f"({main_sentiment_display} - {sub_label_display}: {confidence:.1f}%"
                        if sub_confidence > 0:
                            label_text += f", Sub-label confidence: {sub_confidence:.1f}%"
                        if enhanced:
                            label_text += " [GPT-Enhanced]"
                        label_text += ")"
                    else:
                        label_text = f"({main_sentiment_display}: {confidence:.1f}%"
                        if enhanced:
                            label_text += " [GPT-Enhanced]"
                        label_text += ")"
                    
                    # Create prettier comment styling
                    comment_style = ParagraphStyle(
                        'CommentStyle',
                        parent=styles['Normal'],
                        fontSize=11,
                        fontName='Times-Roman',
                        textColor=colors.Color(0.2, 0.2, 0.3),
                        spaceAfter=5,
                        leftIndent=10
                    )
                    
                    label_style = ParagraphStyle(
                        'LabelStyle',
                        parent=styles['Italic'],
                        fontSize=10,
                        fontName='Times-Italic',
                        textColor=colors.Color(0.5, 0.6, 0.7),
                        spaceAfter=12,
                        leftIndent=20
                    )
                    
                    # Remove bullet points and improve formatting
                    clean_comment_text = text  # Remove the "i. " prefix
                    story.append(Paragraph(clean_comment_text, comment_style))
                    story.append(Paragraph(label_text, label_style))
                    story.append(Spacer(1, 12))  # More spacing between comments
                    
            except Exception as e:
                logger.error(f"Error adding predictions to PDF: {e}")
                story.append(Paragraph("Individual comment analysis unavailable.", styles['Normal']))
            
            # Add personalized AI advice section
            story.append(PageBreak())
            story.append(Paragraph("Personalized Advice from CyberPrint AI Agent", title_style))
            story.append(Spacer(1, 20))
            
            # Generate personalized advice based on user's sentiment patterns
            advice_text = self._generate_personalized_ai_advice(analytics)
            advice_style = ParagraphStyle(
                'AdviceStyle',
                parent=styles['Normal'],
                fontSize=12,
                fontName='Times-Roman',
                textColor=colors.Color(0.2, 0.3, 0.4),
                spaceAfter=15,
                leftIndent=15,
                rightIndent=15,
                backColor=colors.Color(0.97, 0.98, 0.99),
                borderWidth=1,
                borderColor=colors.Color(0.8, 0.85, 0.9),
                borderPadding=15
            )
            
            story.append(Paragraph(advice_text, advice_style))
            story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except ImportError:
            logger.error("reportlab not installed. Install with: pip install reportlab")
            return None
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None
    
    def _generate_personalized_ai_advice(self, analytics: Dict[str, Any]) -> str:
        """Generate personalized advice based on user's sentiment patterns."""
        try:
            sentiment_dist = analytics.get('sentiment_distribution', {})
            sub_label_dist = analytics.get('sub_label_distribution', {})
            total_comments = analytics.get('total_comments', 0)
            
            # Calculate percentages
            positive_pct = (sentiment_dist.get('positive', 0) / total_comments * 100) if total_comments > 0 else 0
            negative_pct = (sentiment_dist.get('negative', 0) / total_comments * 100) if total_comments > 0 else 0
            neutral_pct = (sentiment_dist.get('neutral', 0) / total_comments * 100) if total_comments > 0 else 0
            yellow_flag_pct = (sentiment_dist.get('yellow_flag', 0) / total_comments * 100) if total_comments > 0 else 0
            
            # Get top sub-labels
            top_sub_labels = sorted(sub_label_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            
            advice_parts = []
            
            # Header
            advice_parts.append("<b>Your Digital Communication Profile Analysis</b><br/><br/>")
            
            # Overall sentiment analysis
            if positive_pct > 60:
                advice_parts.append("<b>Strengths:</b> You maintain a predominantly positive online presence! Your comments show genuine engagement and supportive interactions. This creates a welcoming digital environment for others.<br/><br/>")
            elif positive_pct > 40:
                advice_parts.append("<b>Balanced Approach:</b> You demonstrate a healthy mix of positive and constructive communication. This shows emotional maturity in your digital interactions.<br/><br/>")
            else:
                advice_parts.append("<b>Growth Opportunity:</b> Consider incorporating more positive language in your online interactions. Small changes like expressing gratitude or offering encouragement can significantly improve your digital presence.<br/><br/>")
            
            # Sub-label specific advice
            if top_sub_labels:
                advice_parts.append("<b>Communication Pattern Insights:</b><br/>")
                
                for sub_label, count in top_sub_labels:
                    if sub_label == 'general':
                        continue
                        
                    pct = (count / total_comments * 100) if total_comments > 0 else 0
                    if pct < 5:  # Skip very small percentages
                        continue
                        
                    if sub_label == 'gratitude':
                        advice_parts.append(f"• <b>Gratitude ({pct:.1f}%):</b> Excellent! You regularly express appreciation. This builds strong relationships and positive community engagement.<br/>")
                    elif sub_label == 'compliments':
                        advice_parts.append(f"• <b>Compliments ({pct:.1f}%):</b> You're great at recognizing others' contributions. This encourages positive interactions and builds community spirit.<br/>")
                    elif sub_label == 'question_based':
                        advice_parts.append(f"• <b>Inquiry-Focused ({pct:.1f}%):</b> You ask thoughtful questions, showing genuine curiosity and engagement with topics and people.<br/>")
                    elif sub_label == 'harsh_criticism':
                        advice_parts.append(f"• <b>Critical Feedback ({pct:.1f}%):</b> While feedback is valuable, consider framing criticism more constructively. Try the 'sandwich method': positive-constructive-positive.<br/>")
                    elif sub_label == 'disagreement_opposition':
                        advice_parts.append(f"• <b>Disagreement ({pct:.1f}%):</b> Healthy debate is good! Consider acknowledging valid points before presenting counterarguments to maintain respectful discourse.<br/>")
                    elif sub_label == 'casual_conversation':
                        advice_parts.append(f"• <b>Social Engagement ({pct:.1f}%):</b> You're socially active online! This helps build connections and community presence.<br/>")
                    elif sub_label == 'internet_expressions':
                        advice_parts.append(f"• <b>Internet Culture ({pct:.1f}%):</b> You're fluent in online communication styles. Balance this with more substantive contributions for deeper engagement.<br/>")
                
                advice_parts.append("<br/>")
            
            # Personalized recommendations
            advice_parts.append("<b>Personalized Recommendations:</b><br/>")
            
            if negative_pct > 30:
                advice_parts.append("• <b>Emotional Regulation:</b> Take a moment before responding to controversial topics. Consider how your words might affect others and the conversation.<br/>")
            
            if yellow_flag_pct > 20:
                advice_parts.append("• <b>Clarity in Communication:</b> Your humor and sarcasm are noted, but ensure your intent is clear to avoid misunderstandings.<br/>")
            
            if neutral_pct > 50:
                advice_parts.append("• <b>Emotional Expression:</b> Consider sharing more of your personality and opinions. Authentic engagement creates more meaningful connections.<br/>")
            
            # General advice
            advice_parts.append("• <b>Digital Wellness:</b> Continue monitoring your online interactions. Positive digital habits contribute to both personal well-being and community health.<br/>")
            advice_parts.append("• <b>Growth Mindset:</b> Your communication style can always evolve. Regular self-reflection helps maintain healthy online relationships.<br/><br/>")
            
            # Footer
            advice_parts.append("<i>Remember: This analysis reflects patterns in your recent online activity. Use these insights as a starting point for conscious communication choices that align with your values and goals.</i>")
            
            return "".join(advice_parts)
            
        except Exception as e:
            logger.error(f"Error generating personalized advice: {e}")
            return "Personalized advice generation is currently unavailable. Please try again later."

    def _generate_personalized_tips(self, analytics: Dict[str, Any]) -> str:
        """Generate personalized tips based on sentiment analysis results."""
        tips = []
        
        print("\nAnalyzing sentiment patterns...")
        sentiment_dist = analytics.get('sentiment_distribution', {})
        sub_label_dist = analytics.get('sub_label_distribution', {})
        total_comments = analytics.get('total_comments', 0)
        platform = analytics.get('platform', 'unknown')
        
        # Calculate percentages
        negative_pct = (sentiment_dist.get('negative', 0) / total_comments * 100) if total_comments > 0 else 0
        positive_pct = (sentiment_dist.get('positive', 0) / total_comments * 100) if total_comments > 0 else 0
        yellow_flag_pct = (sentiment_dist.get('yellow_flag', 0) / total_comments * 100) if total_comments > 0 else 0
        
        # Platform-specific header
        if platform == 'youtube':
            tips.append("<b>YouTube Channel Feedback Analysis & Content Strategy Tips</b><br/><br/>")
        else:
            tips.append("<b>Personalized Digital Wellbeing Tips</b><br/><br/>")
        
        # Mental health support (if warnings detected, disabled for YouTube)
        if analytics.get('mental_health_warnings', 0) > 0 and platform != 'youtube':
            tips.append("<b>Immediate Support:</b><br/>")
            tips.append("• Crisis Text Line: Text HOME to 741741<br/>")
            tips.append("• National Suicide Prevention Lifeline: 988<br/>")
            tips.append("• International support: https://findahelpline.com<br/><br/>")
        
        # Tips based on negative sentiment
        if negative_pct > 30:
            if platform == 'youtube':
                tips.append("<b>High Negative Feedback Analysis:</b><br/>")
                tips.append("• Review content topics - controversial subjects may generate more criticism<br/>")
                tips.append("• Consider moderating comments more actively to maintain community standards<br/>")
                tips.append("• Engage constructively with valid criticism to show responsiveness<br/>")
                tips.append("• Focus on content that historically receives positive audience response<br/><br/>")
            else:
                tips.append("<b>🛡️ Managing Negative Interactions:</b><br/>")
                tips.append("• Take breaks from social media when feeling overwhelmed<br/>")
                tips.append("• Use privacy settings to limit who can contact you<br/>")
                tips.append("• Block or mute accounts that consistently post negative content<br/>")
                tips.append("• Report harassment or abusive behavior to platform moderators<br/><br/>")
        
        # Tips for handling criticism
        if negative_pct > 20 or sub_label_dist.get('harsh_criticism', 0) > 0:
            if platform == 'youtube':
                tips.append("<b>🛡️ Managing Negative Feedback:</b><br/>")
                tips.append("• Distinguish between constructive criticism and harassment<br/>")
                tips.append("• Use YouTube's comment moderation tools to filter inappropriate content<br/>")
                tips.append("• Consider disabling comments temporarily if harassment becomes overwhelming<br/>")
                tips.append("• Focus on feedback from your core, supportive audience<br/>")
                tips.append("• Report genuinely abusive comments to YouTube for review<br/><br/>")
            else:
                tips.append("<b>Cyberbullying Prevention & Response:</b><br/>")
                tips.append("• Document evidence: Screenshot harmful messages before blocking<br/>")
                tips.append("• Don't engage with bullies - it often escalates the situation<br/>")
                tips.append("• Reach out to trusted friends, family, or counselors for support<br/>")
                tips.append("• Use platform reporting tools to flag abusive content<br/>")
                tips.append("• Consider making your profile private temporarily<br/><br/>")
        
        # Tips for improving engagement
        if positive_pct < 40:
            if platform == 'youtube':
                tips.append("<b>Improving Audience Engagement:</b><br/>")
                tips.append("• Create content that provides clear value to your target audience<br/>")
                tips.append("• Respond to positive comments to encourage more engagement<br/>")
                tips.append("• Ask questions in your videos to prompt viewer interaction<br/>")
                tips.append("• Consider collaborating with creators who have positive communities<br/>")
                tips.append("• Analyze your most positively received videos for successful patterns<br/><br/>")
            else:
                tips.append("<b>Improving Your Online Presence:</b><br/>")
                tips.append("• Share positive content: achievements, gratitude, helpful resources<br/>")
                tips.append("• Engage constructively: ask questions, offer support, share knowledge<br/>")
                tips.append("• Use positive language: 'thank you', 'great point', 'I appreciate'<br/>")
                tips.append("• Join communities aligned with your interests and values<br/>")
                tips.append("• Practice digital empathy: consider others' perspectives<br/><br/>")
        
        # Tips for sarcasm/yellow flags
        if yellow_flag_pct > 15:
            if platform == 'youtube':
                tips.append("<b>Audience Tone Analysis:</b><br/>")
                tips.append("• High sarcasm may indicate audience skepticism or humor<br/>")
                tips.append("• Consider if your content tone matches your audience's communication style<br/>")
                tips.append("• Sarcastic comments aren't always negative - context matters<br/>")
                tips.append("• Engage with witty comments to build community rapport<br/><br/>")
            else:
                tips.append("<b>Communication Clarity:</b><br/>")
                tips.append("• Use clear language to avoid misunderstandings<br/>")
                tips.append("• Add context when using humor or sarcasm<br/>")
                tips.append("• Consider using emojis to convey tone or add /s for sarcasm<br/>")
                tips.append("• Remember: text lacks vocal tone and body language<br/><br/>")
        
        # General tips
        if platform == 'youtube':
            tips.append("<b>Content Creator Wellness:</b><br/>")
            tips.append("• Don't let negative comments define your content strategy entirely<br/>")
            tips.append("• Focus on creating content you're passionate about<br/>")
            tips.append("• Build a supportive community by setting clear community guidelines<br/>")
            tips.append("• Take breaks from reading comments when needed for mental health<br/>")
            tips.append("• Remember: vocal critics often represent a small portion of your audience<br/><br/>")
        else:
            tips.append("<b>General Digital Wellness:</b><br/>")
            tips.append("• Set daily time limits for social media use<br/>")
            tips.append("• Curate your feed: unfollow accounts that make you feel bad<br/>")
            tips.append("• Practice the 'pause before posting' rule<br/>")
            tips.append("• Engage in offline activities and hobbies regularly<br/>")
            tips.append("• Remember: social media is a highlight reel, not reality<br/><br/>")
        
        # Positive reinforcement
        if positive_pct > 50:
            if platform == 'youtube':
                tips.append("<b>Excellent Audience Response!</b><br/>")
                tips.append("Your content is resonating well with your audience. Keep creating valuable content that your viewers appreciate!<br/>")
            else:
                tips.append("<b>Keep Up the Great Work!</b><br/>")
                tips.append("Your positive online presence is making a difference. Continue spreading kindness and positivity in digital spaces!<br/>")
        
        return ''.join(tips)
    
    def save_analytics(self, analytics: Dict[str, Any], output_path: str = None):
        """Save analytics to JSON file."""
        if output_path is None:
            # Sanitize user_id to create valid filename
            import re
            safe_user_id = re.sub(r'[^\w\-_\.]', '_', str(analytics['user_id']))
            safe_user_id = re.sub(r'_+', '_', safe_user_id)  # Replace multiple underscores with single
            safe_user_id = safe_user_id.strip('_')  # Remove leading/trailing underscores
            
            output_path = os.path.join(
                self.output_dir, "analytics", 
                f"analytics_{safe_user_id}_{analytics['platform']}.json"
            )
        
        with open(output_path, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)
        
        logger.info(f"Analytics saved: {output_path}")
        return output_path
    
    def generate_html_report(self, analytics: Dict[str, Any], user_data: pd.DataFrame, output_path: str = None):
        """Generate an HTML report with the specified layout."""
        try:
            # Import prediction module for confidence scores
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from predict import predict_text
            
            if output_path is None:
                output_path = os.path.join(
                    self.output_dir, "reports", 
                    f"cyberprint_report_{analytics['user_id']}_{analytics['platform']}.html"
                )
            
            # Define sentiment colors
            colors = {
                'positive': '#4CAF50',  # Green
                'negative': '#F44336',  # Red
                'neutral': '#9E9E9E',   # Gray
                'yellow_flag': '#FF9800'  # Orange
            }
            
            # Get predictions with confidence scores for each comment
            comments_with_predictions = []
            texts = user_data['tweet'].tolist()
            predictions = predict_text(texts)
            
            for i, (_, row) in enumerate(user_data.iterrows()):
                pred = predictions[i]
                comments_with_predictions.append({
                    'text': row['tweet'],
                    'predicted_label': pred['predicted_label'],
                    'confidence': pred['predicted_score'] * 100,
                    'all_probs': pred['probs']
                })
            
            # Calculate overall model confidence
            overall_confidence = np.mean([pred['predicted_score'] for pred in predictions]) * 100
            
            # Generate HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CyberPrint Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .profile-info {{
            text-align: center;
            margin-bottom: 30px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 8px;
        }}
        .sentiment-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .sentiment-card {{
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .sentiment-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .sentiment-card p {{
            margin: 0;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .chart-container {{
            margin: 40px 0;
            text-align: center;
        }}
        .chart-wrapper {{
            max-width: 600px;
            margin: 0 auto;
        }}
        .mental-health-warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
            text-align: center;
        }}
        .mental-health-warning h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .mental-health-warning p {{
            color: #856404;
            margin-bottom: 10px;
        }}
        .mental-health-warning a {{
            color: #007bff;
            text-decoration: none;
        }}
        .mental-health-warning a:hover {{
            text-decoration: underline;
        }}
        .analysis-section {{
            margin-top: 40px;
        }}
        .analysis-section h2 {{
            text-align: center;
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
        }}
        .confidence-display {{
            text-align: center;
            font-size: 1.3em;
            color: #2c3e50;
            margin: 30px 0;
            padding: 15px;
            background-color: #e8f4fd;
            border-radius: 8px;
        }}
        .comments-list {{
            margin-top: 30px;
        }}
        .comment-item {{
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #ddd;
            background-color: #f9f9f9;
            border-radius: 0 8px 8px 0;
        }}
        .comment-text {{
            margin-bottom: 8px;
            line-height: 1.5;
        }}
        .comment-label {{
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            display: inline-block;
        }}
        .youtube-note {{
            background-color: #e3f2fd;
            border: 1px solid #90caf9;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
            text-align: center;
            color: #1565c0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CyberPrint Report</h1>
        </div>
        
        <div class="profile-info">
            <strong>Profile URL:</strong> {analytics.get('profile_url', 'N/A')}<br>
            <strong>Platform:</strong> {analytics['platform'].title()}<br>
            <strong>User/Channel:</strong> {analytics['user_id']}<br>
            print("\nAnalysis complete!"):</strong> {analytics['timestamp'][:10]}
        </div>"""
            
            # Import color scheme for consistent colors
            try:
                print("\nGenerating visualizations...")
                from cyberprint.visualization.color_scheme import CyberPrintColorScheme
                color_scheme = CyberPrintColorScheme()
            except ImportError:
                # Fallback colors if color scheme module not available
                color_scheme = None
            
            # Create sentiment squares in a 2x2 grid with consistent colors
            if color_scheme:
                sentiment_data = [
                    ('Positive', analytics.get('positive_count', 0), 
                     color_scheme.get_sentiment_color('positive', 'reportlab'), 
                     'Encouraging, supportive, or appreciative content'),
                    ('Negative', analytics.get('negative_count', 0), 
                     color_scheme.get_sentiment_color('negative', 'reportlab'), 
                     'Critical, offensive, or harmful content'),
                    ('Neutral', analytics.get('neutral_count', 0), 
                     color_scheme.get_sentiment_color('neutral', 'reportlab'), 
                     'Factual, unbiased, or informational content'),
                    ('Yellow Flag', analytics.get('yellow_flag_count', 0), 
                     color_scheme.get_sentiment_color('yellow_flag', 'reportlab'), 
                     'Sarcastic, ironic, or contains internet slang')
                ]
            else:
                sentiment_data = [
                    ('Positive', analytics.get('positive_count', 0), colors.Color(0.2, 0.8, 0.4), 'Encouraging, supportive, or appreciative content'),
                    ('Negative', analytics.get('negative_count', 0), colors.Color(0.9, 0.3, 0.3), 'Critical, offensive, or harmful content'),
                    ('Neutral', analytics.get('neutral_count', 0), colors.Color(0.6, 0.6, 0.6), 'Factual, unbiased, or informational content'),
                    ('Yellow Flag', analytics.get('yellow_flag_count', 0), colors.Color(1.0, 0.8, 0.2), 'Sarcastic, ironic, or contains internet slang')
                ]
            
            html_content += '\n        <div class="sentiment-grid">\n'
            
            for sentiment, count, color, description in sentiment_data:
                percentage = (count / analytics['total_comments']) * 100 if analytics['total_comments'] > 0 else 0
                
                html_content += f"""
            <div class="sentiment-card" style="background-color: {colors[sentiment]};">
                <h3>{sentiment.replace('_', ' ').title()}: {count}</h3>
                <p>{sentiment_descriptions[sentiment]}</p>
                <p><strong>{percentage:.1f}%</strong></p>
            </div>"""
            
            html_content += '\n        </div>\n'
            
            # Add chart
            chart_data = [analytics['sentiment_distribution'].get(s, 0) for s in ['positive', 'negative', 'neutral', 'yellow_flag']]
            chart_colors = [colors[s] for s in ['positive', 'negative', 'neutral', 'yellow_flag']]
            
            html_content += f"""
        <div class="chart-container">
            <h2>Main Sentiments Distribution</h2>
            <div class="chart-wrapper">
                <canvas id="sentimentChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <script>
            const ctx = document.getElementById('sentimentChart').getContext('2d');
            const chart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Positive', 'Negative', 'Neutral', 'Yellow Flags'],
                    datasets: [{{
                        data: {chart_data},
                        backgroundColor: {chart_colors},
                        borderColor: {chart_colors},
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{
                                stepSize: 1
                            }}
                        }}
                    }}
                }}
            }});
        </script>"""
            
            # Add mental health warning (not for YouTube)
            if analytics['platform'] != 'youtube' and analytics.get('mental_health_warnings', 0) > 0:
                html_content += """
        <div class="mental-health-warning">
            <h3>Potential Signs of Emotional Distress Detected</h3>
            <p>Some comments this user has made show possible indicators of mental & emotional distress.</p>
            <p>Please remember: if you or someone you know needs help, reach out. <a href="https://findahelpline.com" target="_blank">Find support in your area</a></p>
        </div>"""
            
            # Add YouTube-specific note
            if analytics['platform'] == 'youtube':
                html_content += """
        <div class="youtube-note">
            <p><strong>Note:</strong> These are the recent comments and overall tone feedback this channel has received on their YouTube platform.</p>
        </div>"""
            
            # Add analysis section
            html_content += f"""
        <div class="analysis-section">
            <h2>Analysis of the Comments</h2>
            <div class="confidence-display">
                Confidence in the model result: {overall_confidence:.1f}%
            </div>
            
            <div class="comments-list">"""
            
            # Add individual comments with predictions
            for i, comment_data in enumerate(comments_with_predictions, 1):
                sentiment = comment_data['predicted_label']
                confidence = comment_data['confidence']
                text = comment_data['text'][:200] + ('...' if len(comment_data['text']) > 200 else '')
                
                html_content += f"""
                <div class="comment-item">
                    <div class="comment-text">{i}. {text}</div>
                    <span class="comment-label" style="background-color: {colors.get(sentiment, '#666')};">
                        {sentiment.replace('_', ' ').title()}: {confidence:.1f}%
                    </span>
                </div>"""
            
            html_content += """
            </div>
        </div>
    </div>
</body>
</html>"""
            
            # Write HTML file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_and_report(self, user_data: pd.DataFrame, user_id: str, platform: str = "unknown", 
                          profile_url: str = None, generate_pdf: bool = True, generate_html: bool = False) -> Dict[str, Any]:
        """Complete processing pipeline: analyze data and generate reports."""
        logger.info(f"Starting complete processing for {user_id} on {platform}")
        
        # Process user data
        analytics = self.process_user_data(user_data, user_id, platform)
        
        # Add profile URL to analytics
        if profile_url:
            analytics['profile_url'] = profile_url
        
        # Save analytics
        self.save_analytics(analytics)
        
        # Generate PDF report with unique filename
        if generate_pdf:
            # Create unique PDF filename based on user_id and timestamp
            timestamp = analytics.get('timestamp', '').replace(':', '-').replace(' ', '_')
            if not timestamp:  # Fallback if timestamp is empty
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Clean user_id for filename (remove special characters)
            clean_user_id = ''.join(c for c in user_id if c.isalnum() or c in '-_')[:20]
            pdf_filename = f"cyberprint_report_{clean_user_id}_{timestamp}.pdf"
            pdf_path = os.path.join(self.output_dir, pdf_filename)
            
            # Always generate fresh PDF with current data
            generated_pdf_path = self.generate_pdf_report(analytics, user_data, pdf_path)
            if generated_pdf_path:
                analytics['pdf_report_path'] = generated_pdf_path
                logger.info(f"PDF generated successfully: {pdf_filename}")
            else:
                logger.error(f"PDF generation failed for {user_id}")
                analytics['pdf_report_path'] = None
        
        # Generate HTML report
        if generate_html:
            html_path = self.generate_html_report(analytics, user_data)
            analytics['html_report_path'] = html_path
        
        logger.info(f"Complete processing finished for {user_id}")
        return analytics
    
    def get_instructor_analytics(self) -> Dict[str, Any]:
        """Generate instructor-ready analytics for the entire dataset."""
        logger.info("Generating instructor-ready analytics...")
        
        # Add sub-labels to main dataset
        dataset_with_sub_labels = self.add_sub_labels(self.main_dataset)
        
        print("\nGenerating comprehensive PDF report...")
        instructor_analytics = {
            'dataset_overview': {
                'total_examples': len(dataset_with_sub_labels),
                'unique_users': dataset_with_sub_labels['source'].nunique(),
                'platforms': dataset_with_sub_labels['source'].value_counts().to_dict()
            },
            'sentiment_breakdown': {
                'main_sentiments': dataset_with_sub_labels['sentiment'].value_counts().to_dict(),
                'sub_labels': dataset_with_sub_labels['sub_label'].value_counts().to_dict(),
                'sentiment_by_sub_label': {}
            },
            'mental_health_analysis': {
                'total_warnings': int(dataset_with_sub_labels['mental_health_alert'].sum()),
                'warning_rate': (dataset_with_sub_labels['mental_health_alert'].sum() / len(dataset_with_sub_labels)) * 100,
                'warnings_by_sentiment': dataset_with_sub_labels[dataset_with_sub_labels['mental_health_alert']]['sentiment'].value_counts().to_dict()
            },
            'yellow_flag_analysis': {
                'total_flags': int(dataset_with_sub_labels['yellow_flag'].sum()),
                'flag_rate': (dataset_with_sub_labels['yellow_flag'].sum() / len(dataset_with_sub_labels)) * 100,
                'flags_by_sentiment': dataset_with_sub_labels[dataset_with_sub_labels['yellow_flag']]['sentiment'].value_counts().to_dict()
            },
            'sample_examples': {}
        }
        
        # Generate sentiment by sub-label breakdown
        for sentiment in dataset_with_sub_labels['sentiment'].unique():
            sentiment_data = dataset_with_sub_labels[dataset_with_sub_labels['sentiment'] == sentiment]
            instructor_analytics['sentiment_breakdown']['sentiment_by_sub_label'][sentiment] = sentiment_data['sub_label'].value_counts().to_dict()
        
        # Generate sample examples for each sentiment/sub-label combination
        for sentiment in dataset_with_sub_labels['sentiment'].unique():
            sentiment_data = dataset_with_sub_labels[dataset_with_sub_labels['sentiment'] == sentiment]
            for sub_label in sentiment_data['sub_label'].unique():
                sub_data = sentiment_data[sentiment_data['sub_label'] == sub_label]
                if len(sub_data) > 0:
                    key = f"{sentiment}_{sub_label}"
                    instructor_analytics['sample_examples'][key] = {
                        'count': len(sub_data),
                        'examples': sub_data['tweet'].head(5).tolist()
                    }
        
        # Save instructor analytics
        instructor_path = os.path.join(self.output_dir, "instructor_analytics.json")
        with open(instructor_path, 'w') as f:
            json.dump(instructor_analytics, f, indent=2, default=str)
        
        logger.info(f"Instructor analytics saved: {instructor_path}")
        return instructor_analytics

def main():
    """Main function to demonstrate the CyberPrint pipeline."""
    # Initialize pipeline
    pipeline = CyberPrintPipeline()
    
    # Generate instructor analytics
    instructor_analytics = pipeline.get_instructor_analytics()
    
    print("Creating timeline analysis...")
    print(f"Total examples processed: {instructor_analytics['dataset_overview']['total_examples']:,}")
    print(f"Mental health warnings: {instructor_analytics['mental_health_analysis']['total_warnings']:,}")
    print(f"Yellow flags: {instructor_analytics['yellow_flag_analysis']['total_flags']:,}")
    print("Creating sentiment distribution chart...")
    for sentiment, count in instructor_analytics['sentiment_breakdown']['main_sentiments'].items():
        print("\nStarting sentiment analysis...")
    
    # Example: Process a sample of data as if it were from a specific user
    sample_data = pipeline.main_dataset.sample(n=min(100, len(pipeline.main_dataset)))
    sample_analytics = pipeline.process_and_report(
        sample_data, 
        user_id="sample_user", 
        platform="twitter",
        generate_pdf=True
    )
    
    print("PDF report generated successfully!")
    print(f"PDF report: {sample_analytics.get('pdf_report_path', 'Not generated')}")

if __name__ == "__main__":
    main()