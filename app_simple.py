"""
Simple Email Spam Detector for Render Deployment
Pure Python implementation without heavy ML dependencies
"""

from flask import Flask, render_template, request, jsonify
import os
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SimpleSpamDetector:
    def __init__(self):
        self.spam_keywords = {
            'urgency': ['urgent', 'immediate', 'act now', 'limited time', 'expires', 'deadline', 'within', 'hours', 'minutes', 'before', 'face arrest', 'avoid'],
            'money': ['money', 'cash', 'prize', 'winner', 'lottery', 'million', 'thousand', 'owe', 'refund', 'payment', 'taxes', 'fees'],
            'free': ['free', 'no cost', 'complimentary', 'gratis'],
            'call_to_action': ['click', 'call', 'download', 'visit', 'claim', 'order', 'verify', 'update', 'contact', 'restore', 'log in', 'confirm'],
            'personal_info': ['bank', 'account', 'password', 'ssn', 'credit', 'social security', 'identity', 'billing', 'benefits', 'activity', 'login'],
            'threats': ['arrest', 'warrant', 'seizure', 'suspended', 'restricted', 'compromise', 'compromised', 'lien', 'court', 'summons', 'penalty', 'detected', 'breach'],
            'authority': ['irs', 'federal', 'government', 'medicare', 'social security', 'bank of america', 'paypal', 'amazon', 'wells fargo', 'microsoft', 'apple', 'chase', 'citibank'],
            'phishing_verbs': ['suspicious', 'detected', 'restricted', 'expires', 'required', 'immediately', 'verify', 'confirm', 'update', 'restore']
        }
        
        self.spam_domains = ['secure-', 'verify-', '-secure', '-verification', '-update', '-portal', '-center']
    
    def predict_spam(self, email_text):
        """Analyze email for spam using keyword and pattern matching"""
        text_lower = email_text.lower()
        spam_score = 0
        flagged_keywords = []
        custom_flags = []
        
        # Check keywords by category
        category_weights = {
            'urgency': 15, 'money': 12, 'threats': 20, 'authority': 18,
            'call_to_action': 8, 'personal_info': 10, 'phishing_verbs': 15, 'free': 5
        }
        
        for category, keywords in self.spam_keywords.items():
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
                    if keyword not in flagged_keywords:
                        flagged_keywords.append(keyword)
            
            if matches > 0:
                spam_score += matches * category_weights.get(category, 5)
                
                # Add custom flags
                if category == 'authority':
                    custom_flags.append('authority_impersonation')
                elif category == 'threats':
                    custom_flags.append('threat_language_detected')
                elif category == 'phishing_verbs':
                    custom_flags.append('phishing_language_detected')
                elif category == 'urgency':
                    custom_flags.append('urgency_tactics')
        
        # Check for suspicious domains
        for domain in self.spam_domains:
            if domain in text_lower:
                spam_score += 25
                custom_flags.append('suspicious_domains_detected')
                break
        
        # Check formatting patterns
        upper_ratio = sum(1 for c in email_text if c.isupper()) / len(email_text) if email_text else 0
        if upper_ratio > 0.3:
            spam_score += 15
            custom_flags.append('excessive_capitalization')
        
        exclamation_count = email_text.count('!')
        if exclamation_count > 2:
            spam_score += 10
            custom_flags.append('excessive_punctuation')
        
        # Calculate probability
        probability = min(100, spam_score / 2)
        
        # Boost sophisticated phishing
        if ('authority_impersonation' in custom_flags and 
            'threat_language_detected' in custom_flags and
            ('suspicious_domains_detected' in custom_flags or 'phishing_language_detected' in custom_flags)):
            probability = max(probability, 85)
        
        is_spam = probability > 50
        
        if probability > 80 or probability < 20:
            confidence = 'High'
        elif probability > 60 or probability < 40:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return {
            'spam_probability': round(probability, 1),
            'is_spam': is_spam,
            'classification': 'SPAM' if is_spam else 'NOT SPAM',
            'confidence': confidence,
            'flagged_keywords': flagged_keywords[:10],
            'custom_flags': list(set(custom_flags)),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Initialize detector
detector = SimpleSpamDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_email():
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text.strip():
            return jsonify({'error': 'Email text required'}), 400
        
        logger.info(f"Analyzing email of length {len(email_text)}")
        result = detector.predict_spam(email_text)
        
        logger.info(f"Analysis completed: {result['classification']} ({result['spam_probability']}%)")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        is_spam = data.get('is_spam', False)
        
        logger.info(f"Feedback received: {'SPAM' if is_spam else 'HAM'}")
        
        return jsonify({
            'message': 'Thank you for your feedback! This helps improve our spam detection.',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': 'Failed to process feedback'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting Simple Spam Detector on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)