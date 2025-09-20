#!/usr/bin/env python3
"""
Consistent Color Scheme for CyberPrint Visualizations
====================================================

Centralized color definitions for consistent visualization across
PDF reports, charts, and UI elements.
"""

from typing import Dict, Tuple
from reportlab.lib import colors

class CyberPrintColorScheme:
    """Centralized color scheme for consistent visualization."""
    
    # Main sentiment colors (RGB tuples)
    SENTIMENT_COLORS = {
        'positive': (0.2, 0.8, 0.4),      # Green
        'negative': (0.9, 0.3, 0.3),      # Red  
        'neutral': (0.6, 0.6, 0.6),       # Gray
        'yellow_flag': (1.0, 0.8, 0.2)    # Yellow/Orange
    }
    
    # Sub-label colors (lighter/darker variants of main colors)
    SUB_LABEL_COLORS = {
        # Positive sub-labels (green variants)
        'gratitude': (0.1, 0.7, 0.3),
        'compliments': (0.3, 0.9, 0.5),
        'reinforcing_positive_actions': (0.2, 0.8, 0.4),
        'joy_happiness': (0.4, 1.0, 0.6),
        
        # Negative sub-labels (red variants)
        'offensive': (0.8, 0.2, 0.2),
        'insulting': (0.9, 0.3, 0.3),
        'threatening': (0.7, 0.1, 0.1),
        'harsh_criticism': (1.0, 0.4, 0.4),
        
        # Neutral sub-labels (gray variants)
        'fact_based': (0.5, 0.5, 0.5),
        'question_based': (0.7, 0.7, 0.7),
        'lack_of_bias': (0.6, 0.6, 0.6),
        'informational': (0.8, 0.8, 0.8),
        
        # Yellow flag sub-labels (yellow/orange variants)
        'sarcasm': (1.0, 0.7, 0.1),
        'irony': (1.0, 0.8, 0.2),
        'internet_slang': (0.9, 0.9, 0.3),
        'humor': (1.0, 0.9, 0.4),
        
        # Default
        'general': (0.5, 0.5, 0.5)
    }
    
    # Pastel versions for cards/backgrounds
    PASTEL_COLORS = {
        'positive': (0.8, 0.95, 0.85),    # Light green
        'negative': (0.98, 0.85, 0.85),   # Light red
        'neutral': (0.9, 0.9, 0.9),       # Light gray
        'yellow_flag': (1.0, 0.95, 0.8)   # Light yellow
    }
    
    # Hex colors for HTML/CSS
    HEX_COLORS = {
        'positive': '#33CC66',
        'negative': '#E64D4D',
        'neutral': '#999999',
        'yellow_flag': '#FFCC33'
    }
    
    # ReportLab Color objects
    REPORTLAB_COLORS = {
        'positive': colors.Color(0.2, 0.8, 0.4),
        'negative': colors.Color(0.9, 0.3, 0.3),
        'neutral': colors.Color(0.6, 0.6, 0.6),
        'yellow_flag': colors.Color(1.0, 0.8, 0.2)
    }
    
    REPORTLAB_PASTEL_COLORS = {
        'positive': colors.Color(0.8, 0.95, 0.85),
        'negative': colors.Color(0.98, 0.85, 0.85),
        'neutral': colors.Color(0.9, 0.9, 0.9),
        'yellow_flag': colors.Color(1.0, 0.95, 0.8)
    }
    
    @classmethod
    def get_sentiment_color(cls, sentiment: str, format: str = 'rgb') -> Tuple[float, float, float] or str or colors.Color:
        """
        Get color for a sentiment in the specified format.
        
        Args:
            sentiment: The sentiment label
            format: 'rgb', 'hex', 'reportlab', or 'pastel'
            
        Returns:
            Color in the requested format
        """
        sentiment = sentiment.lower()
        
        if format == 'rgb':
            return cls.SENTIMENT_COLORS.get(sentiment, cls.SENTIMENT_COLORS['neutral'])
        elif format == 'hex':
            return cls.HEX_COLORS.get(sentiment, cls.HEX_COLORS['neutral'])
        elif format == 'reportlab':
            return cls.REPORTLAB_COLORS.get(sentiment, cls.REPORTLAB_COLORS['neutral'])
        elif format == 'pastel':
            return cls.PASTEL_COLORS.get(sentiment, cls.PASTEL_COLORS['neutral'])
        else:
            return cls.SENTIMENT_COLORS.get(sentiment, cls.SENTIMENT_COLORS['neutral'])
    
    @classmethod
    def get_sub_label_color(cls, sub_label: str, format: str = 'rgb') -> Tuple[float, float, float] or str or colors.Color:
        """
        Get color for a sub-label in the specified format.
        
        Args:
            sub_label: The sub-label
            format: 'rgb', 'hex', or 'reportlab'
            
        Returns:
            Color in the requested format
        """
        sub_label = sub_label.lower()
        rgb_color = cls.SUB_LABEL_COLORS.get(sub_label, cls.SUB_LABEL_COLORS['general'])
        
        if format == 'rgb':
            return rgb_color
        elif format == 'hex':
            return f"#{int(rgb_color[0]*255):02x}{int(rgb_color[1]*255):02x}{int(rgb_color[2]*255):02x}"
        elif format == 'reportlab':
            return colors.Color(rgb_color[0], rgb_color[1], rgb_color[2])
        else:
            return rgb_color
    
    @classmethod
    def get_color_palette(cls) -> Dict[str, Dict[str, any]]:
        """Get the complete color palette for reference."""
        return {
            'sentiments': cls.SENTIMENT_COLORS,
            'sub_labels': cls.SUB_LABEL_COLORS,
            'pastels': cls.PASTEL_COLORS,
            'hex': cls.HEX_COLORS
        }
    
    @classmethod
    def get_chart_colors(cls, labels: list) -> list:
        """Get a list of colors for chart data based on sentiment labels."""
        colors_list = []
        for label in labels:
            if label.lower() in cls.SENTIMENT_COLORS:
                colors_list.append(cls.get_sentiment_color(label, 'reportlab'))
            else:
                colors_list.append(cls.REPORTLAB_COLORS['neutral'])
        return colors_list
