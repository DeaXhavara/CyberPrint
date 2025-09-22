#!/usr/bin/env python3
"""
Enhanced Sub-label Classifier for CyberPrint
============================================

Rule-based filtering system for accurate sub-label assignment based on
keyword patterns and linguistic features.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SubLabelRule:
    """Represents a rule for sub-label classification."""
    name: str
    keywords: List[str]
    patterns: List[str]
    weight: float = 1.0
    requires_all: bool = False  # If True, all keywords must be present

class EnhancedSubLabelClassifier:
    """Enhanced sub-label classifier using rule-based filtering."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.applied_rules_log = []
    
    def get_applied_rules(self) -> List[str]:
        """Return the list of applied rules for the last classification."""
        return self.applied_rules_log.copy()
    
    def _initialize_rules(self) -> Dict[str, Dict[str, SubLabelRule]]:
        """Initialize comprehensive rule sets for each sentiment category."""
        
        rules = {
            'positive': {
                'gratitude': SubLabelRule(
                    name='gratitude',
                    keywords=[
                        'thank', 'thanks', 'grateful', 'appreciate', 'blessed', 'gratitude',
                        'thankful', 'indebted', 'obliged', 'acknowledge', 'credit', 'props',
                        'kudos', 'cheers', 'much appreciated', 'big thanks', 'many thanks'
                    ],
                    patterns=[
                        r'\bthank\s+you\b',
                        r'\bthanks\s+for\b',
                        r'\bappreciate\s+it\b',
                        r'\bgrateful\s+for\b',
                        r'\bmuch\s+appreciated\b'
                    ],
                    weight=2.0
                ),
                'compliments': SubLabelRule(
                    name='compliments',
                    keywords=[
                        'amazing', 'awesome', 'brilliant', 'excellent', 'fantastic', 'great',
                        'wonderful', 'outstanding', 'superb', 'marvelous', 'fabulous',
                        'terrific', 'incredible', 'phenomenal', 'exceptional', 'remarkable',
                        'splendid', 'magnificent', 'perfect', 'beautiful', 'lovely',
                        'impressive', 'stunning', 'breathtaking', 'genius', 'masterpiece',
                        'good', 'nice', 'cool', 'solid', 'decent', 'fine', 'sweet',
                        'rad', 'dope', 'fire', 'lit', 'sick', 'tight', 'fresh',
                        'clean', 'smooth', 'slick', 'sharp', 'crisp', 'neat'
                    ],
                    patterns=[
                        r'\byou\s+are\s+(amazing|awesome|great|brilliant|good|cool)\b',
                        r'\bthis\s+is\s+(amazing|awesome|incredible|perfect|great|good|cool)\b',
                        r'\bwell\s+done\b',
                        r'\bgreat\s+job\b',
                        r'\bnice\s+work\b',
                        r'\bgood\s+(job|work|stuff|point|idea)\b',
                        r'\bpretty\s+(good|cool|nice|awesome)\b',
                        r'\bnot\s+bad\b',
                        r'\bi\s+like\s+(this|it|that)\b'
                    ],
                    weight=1.5
                ),
                'reinforcing_positive_actions': SubLabelRule(
                    name='reinforcing_positive_actions',
                    keywords=[
                        'help', 'support', 'assist', 'encourage', 'motivate', 'inspire',
                        'uplift', 'boost', 'cheer', 'comfort', 'reassure', 'validate',
                        'recognize', 'celebrate', 'congratulate', 'praise', 'commend',
                        'keep going', 'keep up', 'well done', 'good job', 'proud of'
                    ],
                    patterns=[
                        r'\bkeep\s+(it\s+up|going|trying)\b',
                        r'\byou\s+can\s+do\s+it\b',
                        r'\bproud\s+of\s+you\b',
                        r'\bwell\s+done\b',
                        r'\bgood\s+(job|work)\b',
                        r'\bkeep\s+up\s+the\s+good\s+work\b'
                    ],
                    weight=1.8
                ),
                'joy_happiness': SubLabelRule(
                    name='joy_happiness',
                    keywords=[
                        'happy', 'joy', 'excited', 'thrilled', 'delighted', 'pleased',
                        'satisfied', 'content', 'cheerful', 'upbeat', 'positive', 'optimistic',
                        'elated', 'ecstatic', 'overjoyed', 'blissful', 'euphoric',
                        'glad', 'joyful', 'merry', 'bright', 'sunny', 'bubbly',
                        'pumped', 'stoked', 'hyped', 'amped', 'psyched', 'buzzing'
                    ],
                    patterns=[
                        r'\bso\s+happy\b',
                        r'\bfeel\s+great\b',
                        r'\bmade\s+my\s+day\b',
                        r'\bput\s+a\s+smile\b',
                        r'\bi\s+love\s+(this|it|that)\b',
                        r'\bfeeling\s+(good|great|awesome|happy)\b',
                        r'\bthis\s+makes\s+me\s+(happy|smile)\b',
                        r'\byay\b|\bwoo\b|\byes\b'
                    ],
                    weight=1.3
                ),
                'agreement_support': SubLabelRule(
                    name='agreement_support',
                    keywords=[
                        'agree', 'exactly', 'absolutely', 'definitely', 'totally',
                        'completely', 'perfectly', 'spot on', 'right on', 'nail on head',
                        'couldn\'t agree more', 'my thoughts exactly', 'well said',
                        'this', 'yes', 'yep', 'yeah', 'true', 'correct', 'right'
                    ],
                    patterns=[
                        r'\bi\s+agree\b',
                        r'\bexactly\b',
                        r'\babsolutely\b',
                        r'\bthis\s+is\s+(true|right|correct)\b',
                        r'\byou\'re\s+(right|correct)\b',
                        r'\bcouldn\'t\s+agree\s+more\b',
                        r'\bwell\s+said\b',
                        r'\bmy\s+thoughts\s+exactly\b',
                        r'^(this|yes|yep|yeah|true|exactly)$'
                    ],
                    weight=1.4
                )
            },
            
            'negative': {
                'offensive': SubLabelRule(
                    name='offensive',
                    keywords=[
                        'hate', 'despise', 'loathe', 'disgusting', 'revolting', 'vile',
                        'disgust', 'repulsive', 'abhorrent', 'detestable', 'offensive',
                        'inappropriate', 'unacceptable', 'disturbing', 'sick', 'twisted',
                        'gross', 'nasty', 'foul', 'repugnant', 'appalling', 'horrid'
                    ],
                    patterns=[
                        r'\bhate\s+this\b',
                        r'\bso\s+disgusting\b',
                        r'\bmakes\s+me\s+sick\b',
                        r'\bcompletely\s+inappropriate\b',
                        r'\bthis\s+is\s+(disgusting|gross|sick|wrong)\b',
                        r'\bi\s+hate\s+(this|it|that)\b'
                    ],
                    weight=2.5
                ),
                'insulting': SubLabelRule(
                    name='insulting',
                    keywords=[
                        'stupid', 'idiot', 'moron', 'dumb', 'fool', 'loser', 'pathetic',
                        'worthless', 'useless', 'incompetent', 'failure', 'trash', 'garbage',
                        'waste', 'joke', 'clown', 'amateur', 'clueless', 'braindead'
                    ],
                    patterns=[
                        r'\byou\s+are\s+(stupid|an\s+idiot|pathetic|worthless)\b',
                        r'\bwhat\s+a\s+(loser|joke|waste)\b',
                        r'\bso\s+(dumb|stupid|pathetic)\b'
                    ],
                    weight=2.0
                ),
                'threatening': SubLabelRule(
                    name='threatening',
                    keywords=[
                        'kill', 'die', 'death', 'hurt', 'harm', 'destroy', 'violence',
                        'attack', 'fight', 'beat', 'punch', 'threat', 'dangerous',
                        'weapon', 'blood', 'murder', 'revenge'
                    ],
                    patterns=[
                        r'\bgoing\s+to\s+(kill|hurt|destroy)\b',
                        r'\bwish\s+you\s+were\s+dead\b',
                        r'\bmake\s+you\s+pay\b',
                        r'\bwatch\s+your\s+back\b'
                    ],
                    weight=3.0
                ),
                'harsh_criticism': SubLabelRule(
                    name='harsh_criticism',
                    keywords=[
                        'terrible', 'awful', 'horrible', 'worst', 'trash', 'garbage',
                        'complete failure', 'disaster', 'nightmare', 'catastrophe',
                        'abysmal', 'atrocious', 'deplorable', 'appalling',
                        'bad', 'sucks', 'lame', 'weak', 'boring', 'dull',
                        'disappointing', 'underwhelming', 'mediocre', 'subpar'
                    ],
                    patterns=[
                        r'\bthis\s+is\s+(terrible|awful|horrible|trash|bad)\b',
                        r'\bworst\s+(thing|idea|decision)\b',
                        r'\bcomplete\s+(disaster|failure)\b',
                        r'\bthis\s+sucks\b',
                        r'\bso\s+(bad|lame|boring|disappointing)\b',
                        r'\bi\s+don\'t\s+like\s+(this|it|that)\b'
                    ],
                    weight=1.5
                ),
                'disagreement_opposition': SubLabelRule(
                    name='disagreement_opposition',
                    keywords=[
                        'disagree', 'wrong', 'incorrect', 'false', 'untrue', 'mistaken',
                        'not true', 'bullshit', 'nonsense', 'ridiculous', 'absurd',
                        'no way', 'absolutely not', 'i don\'t think so', 'doubt it'
                    ],
                    patterns=[
                        r'\bi\s+disagree\b',
                        r'\bthat\'s\s+(wrong|incorrect|false|not\s+true)\b',
                        r'\byou\'re\s+wrong\b',
                        r'\bno\s+way\b',
                        r'\babsolutely\s+not\b',
                        r'\bi\s+don\'t\s+think\s+so\b',
                        r'\bthat\'s\s+(bullshit|nonsense|ridiculous)\b',
                        r'^(no|nope|nah|wrong)$'
                    ],
                    weight=1.6
                )
            },
            
            'neutral': {
                'fact_based': SubLabelRule(
                    name='fact_based',
                    keywords=[
                        'according to', 'research shows', 'studies indicate', 'data suggests',
                        'statistics show', 'evidence shows', 'facts', 'information',
                        'based on', 'documented', 'verified', 'confirmed', 'reported',
                        'analysis', 'findings', 'results', 'conclusion'
                    ],
                    patterns=[
                        r'\baccording\s+to\b',
                        r'\bresearch\s+(shows|indicates|suggests)\b',
                        r'\bstudies\s+(show|indicate|suggest)\b',
                        r'\bdata\s+(shows|indicates|suggests)\b',
                        r'\bevidence\s+(shows|suggests|indicates)\b',
                        r'\bin\s+fact\b',
                        r'\bstatistically\b'
                    ],
                    weight=2.0
                ),
                'simple_agreement': SubLabelRule(
                    name='simple_agreement',
                    keywords=[
                        'agree', 'exactly', 'yes', 'yep', 'yeah', 'true', 'correct', 'right',
                        'absolutely', 'definitely', 'totally', 'completely', 'sure',
                        'this', 'that', 'it', 'same', 'likewise'
                    ],
                    patterns=[
                        r'^(yes|yep|yeah|true|exactly|right|correct|this|that|it)$',
                        r'\bi\s+agree\b',
                        r'\bexactly\b',
                        r'\babsolutely\b',
                        r'\btotally\b',
                        r'\bcompletely\b'
                    ],
                    weight=1.8
                ),
                'simple_positive': SubLabelRule(
                    name='simple_positive',
                    keywords=[
                        'good', 'nice', 'cool', 'great', 'awesome', 'sweet', 'solid',
                        'decent', 'fine', 'ok', 'okay', 'alright', 'not bad'
                    ],
                    patterns=[
                        r'^(good|nice|cool|great|awesome|ok|okay|alright)$',
                        r'\bnot\s+bad\b',
                        r'\bpretty\s+(good|nice|cool)\b'
                    ],
                    weight=1.6
                ),
                'simple_negative': SubLabelRule(
                    name='simple_negative',
                    keywords=[
                        'bad', 'wrong', 'nope', 'nah', 'disagree', 'false',
                        'incorrect', 'untrue', 'not true', 'not right'
                    ],
                    patterns=[
                        r'^(bad|wrong|nope|nah|false)$',
                        r'\bi\s+disagree\b',
                        r'\bnot\s+(true|right|correct)\b'
                    ],
                    weight=1.6
                ),
                'question_based': SubLabelRule(
                    name='question_based',
                    keywords=[
                        'what', 'how', 'why', 'when', 'where', 'who', 'which',
                        'can you', 'could you', 'would you', 'do you', 'are you',
                        'is it', 'was it', 'will it', 'should i', 'could i'
                    ],
                    patterns=[
                        r'\?',  # Contains question mark
                        r'\bwhat\s+(is|are|do|does|did|will|would|can|could)\b',
                        r'\bhow\s+(do|does|did|will|would|can|could|to)\b',
                        r'\bwhy\s+(is|are|do|does|did|will|would)\b',
                        r'\bcan\s+you\b',
                        r'\bcould\s+you\b',
                        r'\bwould\s+you\b'
                    ],
                    weight=2.5
                ),
                'lack_of_bias': SubLabelRule(
                    name='lack_of_bias',
                    keywords=[
                        'neutral', 'objective', 'impartial', 'fair', 'balanced',
                        'unbiased', 'unprejudiced', 'dispassionate', 'factual',
                        'straightforward', 'matter-of-fact', 'informative'
                    ],
                    patterns=[
                        r'\bin\s+my\s+opinion\b',
                        r'\bobjectiv(e|ely)\b',
                        r'\bfrom\s+a\s+neutral\s+perspective\b',
                        r'\bto\s+be\s+fair\b'
                    ],
                    weight=1.5
                ),
                'informational': SubLabelRule(
                    name='informational',
                    keywords=[
                        'update', 'news', 'report', 'announcement', 'notice',
                        'information', 'details', 'specifications', 'description',
                        'explanation', 'clarification', 'summary', 'overview'
                    ],
                    patterns=[
                        r'\bhere\s+is\s+the\b',
                        r'\bfor\s+your\s+information\b',
                        r'\bjust\s+to\s+clarify\b',
                        r'\blet\s+me\s+explain\b'
                    ],
                    weight=1.3
                ),
                'casual_conversation': SubLabelRule(
                    name='casual_conversation',
                    keywords=[
                        'hey', 'hi', 'hello', 'sup', 'what\'s up', 'how are you',
                        'how\'s it going', 'good morning', 'good afternoon', 'good evening',
                        'see you', 'talk to you', 'catch you', 'later', 'bye', 'goodbye',
                        'take care', 'have a good', 'nice to meet', 'pleasure to meet',
                        'there', 'here', 'post', 'comment', 'thread', 'topic'
                    ],
                    patterns=[
                        r'^(hey|hi|hello|sup)\b',
                        r'\bhow\s+(are\s+you|is\s+it\s+going)\b',
                        r'\bgood\s+(morning|afternoon|evening|night)\b',
                        r'\bsee\s+you\s+(later|soon)\b',
                        r'\btalk\s+to\s+you\s+(later|soon)\b',
                        r'\btake\s+care\b',
                        r'\bhave\s+a\s+good\b',
                        r'^(bye|goodbye|later)$',
                        r'\b(nice|good)\s+(post|comment|thread|topic)\b'
                    ],
                    weight=1.2
                ),
                'internet_expressions': SubLabelRule(
                    name='internet_expressions',
                    keywords=[
                        'lol', 'lmao', 'rofl', 'lmfao', 'omg', 'wtf', 'tbh', 'imo', 'imho',
                        'fwiw', 'afaik', 'iirc', 'tl;dr', 'tldr', 'btw', 'fyi', 'smh'
                    ],
                    patterns=[
                        r'^(lol|lmao|rofl|omg|wtf|tbh|imo|btw|fyi|smh)$',
                        r'\b(lol|lmao|omg|tbh|imo|btw|fyi)\b'
                    ],
                    weight=1.5
                )
            },
            
            'yellow_flag': {
                'sarcasm': SubLabelRule(
                    name='sarcasm',
                    keywords=[
                        'sure', 'right', 'of course', 'obviously', 'clearly',
                        'totally', 'absolutely', 'definitely', 'certainly',
                        'yeah right', 'oh sure', 'real smart', 'genius move'
                    ],
                    patterns=[
                        r'\boh\s+(sure|right|great|wonderful)\b',
                        r'\byeah\s+right\b',
                        r'\breal\s+(smart|clever|genius)\b',
                        r'\bgenius\s+move\b',
                        r'\bobviously\b.*\bnot\b',
                        r'\bsure\s+thing\b'
                    ],
                    weight=2.0
                ),
                'irony': SubLabelRule(
                    name='irony',
                    keywords=[
                        'oh great', 'wonderful', 'perfect', 'exactly what i needed',
                        'just what i wanted', 'fantastic', 'brilliant', 'amazing',
                        'just my luck', 'how convenient', 'what a surprise'
                    ],
                    patterns=[
                        r'\boh\s+great\b',
                        r'\bjust\s+perfect\b',
                        r'\bexactly\s+what\s+i\s+needed\b',
                        r'\bjust\s+what\s+i\s+wanted\b',
                        r'\bjust\s+my\s+luck\b',
                        r'\bhow\s+convenient\b',
                        r'\bwhat\s+a\s+surprise\b'
                    ],
                    weight=2.5
                ),
                'internet_slang': SubLabelRule(
                    name='internet_slang',
                    keywords=[
                        'lol', 'lmao', 'rofl', 'lmfao', 'omg', 'wtf', 'smh', 'fml',
                        'tbh', 'imo', 'imho', 'afaik', 'tl;dr', 'ftw', 'facepalm',
                        'epic fail', 'noob', 'pwned', 'rekt', 'savage', 'lit', 'fire'
                    ],
                    patterns=[
                        r'\blol\b',
                        r'\blmao\b',
                        r'\bomg\b',
                        r'\bwtf\b',
                        r'\bsmh\b',
                        r'\btbh\b',
                        r'\bimo\b',
                        r'\bepic\s+fail\b',
                        r'\bget\s+rekt\b'
                    ],
                    weight=1.8
                ),
                'humor': SubLabelRule(
                    name='humor',
                    keywords=[
                        'lol', 'lmao', 'rofl', 'haha', 'hehe', 'funny', 'hilarious',
                        'joke', 'comedy', 'amusing', 'witty', 'clever', 'entertaining',
                        'laugh', 'giggle', 'chuckle', 'crack up', 'burst out laughing'
                    ],
                    patterns=[
                        r'\bha+h+a+\b',
                        r'\bhe+h+e+\b',
                        r'\bso\s+funny\b',
                        r'\bcracking\s+up\b',
                        r'\bdying\s+of\s+laughter\b',
                        r'\bi\s+can\'t\s+even\b'
                    ],
                    weight=1.5
                )
            }
        }
    
        return rules
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Main classify method for compatibility.
        """
        sub_label, confidence, _ = self.classify_sub_label(text, "neutral", log_rules=False)
        return sub_label, confidence
    
    def classify_sub_label(self, text: str, main_sentiment: str, log_rules: bool = True) -> Tuple[str, float, List[str]]:
        """
        Classify sub-label using rule-based filtering.
        
        Args:
            text: The comment text
            main_sentiment: The main sentiment category
            log_rules: Whether to log applied rules
            
        Returns:
            Tuple of (sub_label, confidence_score, applied_rules)
        """
        if main_sentiment not in self.rules:
            return 'general', 0.0, []
        
        text_lower = text.lower()
        sentiment_rules = self.rules[main_sentiment]
        
        # Score each sub-label
        sub_label_scores = {}
        applied_rules = []
        
        for sub_label, rule in sentiment_rules.items():
            score = 0.0
            rule_matches = []
            
            # Check keywords
            keyword_matches = 0
            for keyword in rule.keywords:
                if keyword.lower() in text_lower:
                    keyword_matches += 1
                    rule_matches.append(f"keyword:{keyword}")
            
            # Check patterns
            pattern_matches = 0
            for pattern in rule.patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1
                    rule_matches.append(f"pattern:{pattern}")
            
            # Calculate score based on rule requirements
            if rule.requires_all:
                if keyword_matches == len(rule.keywords):
                    score = rule.weight * (keyword_matches + pattern_matches)
            else:
                score = rule.weight * (keyword_matches + pattern_matches)
            
            if score > 0:
                sub_label_scores[sub_label] = score
                applied_rules.extend([f"{sub_label}:{match}" for match in rule_matches])
        
        # Determine best sub-label
        if sub_label_scores:
            best_sub_label = max(sub_label_scores, key=sub_label_scores.get)
            max_score = sub_label_scores[best_sub_label]
            
            # Normalize confidence score (0-1 range) - very generous to minimize 'general' fallback
            confidence = min(max_score / 2.0, 1.0)  # Even more generous scoring
            
            if log_rules:
                self.applied_rules_log.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'main_sentiment': main_sentiment,
                    'sub_label': best_sub_label,
                    'confidence': confidence,
                    'applied_rules': applied_rules
                })
            
            return best_sub_label, confidence, applied_rules
        else:
            return 'general', 0.0, []
    
    def get_rules_log(self) -> List[Dict]:
        """Get the log of applied rules for traceability."""
        return self.applied_rules_log
    
    def clear_rules_log(self):
        """Clear the rules log."""
        self.applied_rules_log = []
    
    def get_sub_label_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Get descriptions for all sub-labels."""
        descriptions = {
            'positive': {
                'gratitude': 'Expressions of thanks, appreciation, or acknowledgment',
                'compliments': 'Praise, admiration, or positive evaluation',
                'reinforcing_positive_actions': 'Encouragement, support, or validation of good behavior',
                'joy_happiness': 'Expressions of happiness, joy, or positive emotions',
                'agreement_support': 'Agreement, validation, or supportive responses'
            },
            'negative': {
                'offensive': 'Inappropriate, disturbing, or morally objectionable content',
                'insulting': 'Personal attacks, name-calling, or derogatory remarks',
                'threatening': 'Threats of violence, harm, or intimidation',
                'harsh_criticism': 'Severe negative judgment or condemnation',
                'disagreement_opposition': 'Disagreement, opposition, or contradictory statements'
            },
            'neutral': {
                'fact_based': 'Objective information, research, or documented evidence',
                'question_based': 'Inquiries, requests for information, or clarification',
                'lack_of_bias': 'Impartial, balanced, or objective statements',
                'informational': 'Updates, announcements, or explanatory content',
                'casual_conversation': 'Greetings, farewells, and casual social interactions',
                'simple_agreement': 'Basic agreement or affirmative responses',
                'simple_positive': 'Simple positive expressions or approval',
                'simple_negative': 'Simple negative expressions or disagreement',
                'internet_expressions': 'Common internet slang and abbreviations'
            },
            'yellow_flag': {
                'sarcasm': 'Ironic or mocking tone with opposite intended meaning',
                'irony': 'Situational irony or unexpected contradictions',
                'internet_slang': 'Online jargon, memes, or digital communication patterns',
                'humor': 'Jokes, amusing content, or entertainment-focused remarks',
                'confusion_uncertainty': 'Expressions of confusion, uncertainty, or lack of clarity'
            }
        }
        return descriptions
