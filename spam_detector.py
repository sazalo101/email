"""
Advanced Email Spam Detector with Custom Feature Engineering

This module implements a machine learning-based spam detection system using:
1. Custom TF-IDF vectorization with domain-specific preprocessing
2. Logistic Regression with L2 regularization
3. Feature engineering including text statistics and pattern matching
4. Explainable AI with keyword importance scoring

Author: DevPost Competition Team
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import re
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class SpamDetector:
    """
    Advanced spam detection system with custom feature engineering.
    
    This class implements a machine learning pipeline that:
    - Preprocesses email text with domain-specific cleaning
    - Extracts TF-IDF features and custom statistical features
    - Trains a logistic regression model with cross-validation
    - Provides explainable predictions with keyword analysis
    """
    
    def __init__(self, max_features: int = 5000, test_size: float = 0.2):
        """
        Initialize the spam detector with configurable parameters.
        
        Args:
            max_features: Maximum number of TF-IDF features to extract
            test_size: Proportion of data to use for testing
        """
        # Core ML components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english', 
            lowercase=True,
            ngram_range=(1, 2),  # Use both unigrams and bigrams for better context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Logistic Regression with L2 regularization to prevent overfitting
        self.model = LogisticRegression(
            random_state=42, 
            C=1.0,  # Regularization strength
            max_iter=1000,  # Increased iterations for convergence
            solver='liblinear'  # Good for small datasets
        )
        
        # Feature scaling for custom numerical features
        self.scaler = StandardScaler()
        
        # Model state and metadata
        self.spam_keywords = []
        self.feature_names = []
        self.is_trained = False
        self.test_size = test_size
        self.training_history = {}
        
        # Spam pattern detection regexes (hand-crafted rules)
        self.spam_patterns = {
            'urgency': r'\b(urgent|immediate|act now|limited time|expires|deadline|within \d+|before|face arrest|avoid|hours|minutes)\b',
            'money': r'\$[\d,]+|\b(money|cash|prize|winner|lottery|million|thousand|owe|refund|payment|taxes|fees)\b',
            'free': r'\b(free|no cost|complimentary|gratis)\b',
            'suspicious_chars': r'[!]{2,}|[A-Z]{5,}|[0-9]{8,}',
            'call_to_action': r'\b(click|call|download|visit|claim|order|verify|update|contact|restore|log in|confirm)\b',
            'personal_info': r'\b(bank|account|password|ssn|credit|social security|identity|billing|benefits|activity|login)\b',
            'threats': r'\b(arrest|warrant|seizure|suspended|restricted|compromise|compromised|lien|court|summons|penalty|detected|breach)\b',
            'authority': r'\b(irs|federal|government|medicare|social security|bank of america|paypal|amazon|wells fargo|microsoft|apple)\b',
            'suspicious_domains': r'\b\w*-\w+(-\w+)*\.(com|net|org|gov)\b',
            'phone_pressure': r'\b(call|contact).{0,20}(\d{3}[-.]?\d{3}[-.]?\d{4}|\d{1}-\d{3}-\d{3}-\d{4})\b',
            'spoofed_domains': r'\b(secure-\w+|verify-\w+|\w+-secure|\w+-verification|\w+-update)\.(com|net|org)\b',
            'phishing_verbs': r'\b(suspicious|detected|restricted|expires|required|immediately|verify|confirm|update|restore)\b'
        }
        
    def extract_custom_features(self, text: str) -> np.ndarray:
        """
        Extract custom statistical and pattern-based features from email text.
        
        This method creates domain-specific features that complement TF-IDF:
        - Text statistics (length, word count, etc.)
        - Spam pattern matches using regex
        - Character and formatting analysis
        
        Args:
            text: Raw email text
            
        Returns:
            numpy array of custom features
        """
        features = []
        
        # Basic text statistics (indices 0-5)
        features.append(len(text))  # 0: Total character count
        features.append(len(text.split()))  # 1: Word count
        features.append(len([c for c in text if c.isupper()]) / len(text) if text else 0)  # 2: Uppercase ratio
        features.append(len([c for c in text if c.isdigit()]) / len(text) if text else 0)  # 3: Digit ratio
        features.append(text.count('!') / len(text) if text else 0)  # 4: Exclamation ratio
        features.append(text.count('$'))  # 5: Dollar sign count
        
        # Pattern matching features
        for pattern_name, pattern in self.spam_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            features.append(matches)
        
        # Email-specific features
        features.append(len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)))  # Email count
        features.append(len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)))  # URL count
        features.append(len(re.findall(r'\b\d{3}-\d{3}-\d{4}\b', text)))  # Phone number count
        
        return np.array(features)
    
    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing with domain-specific cleaning.
        
        This method applies multiple preprocessing steps:
        - Converts to lowercase for consistency
        - Removes excessive punctuation while preserving meaning
        - Normalizes whitespace
        - Removes non-ASCII characters that might be used for obfuscation
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive punctuation (more than 2 consecutive)
        text = re.sub(r'[!]{3,}', '!!', text)
        text = re.sub(r'[?]{3,}', '??', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-ASCII characters (potential obfuscation)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', ' ', text)
        
        return text.strip()
    
    def load_training_data(self) -> Tuple[List[str], List[int]]:
        """
        Load and prepare training data with comprehensive spam/ham examples.
        
        This method creates a diverse dataset covering:
        - Financial scams and phishing attempts
        - Adult content and dating spam
        - Fake products and services
        - Tech support scams
        - Legitimate business communications
        - Personal correspondence
        
        Returns:
            Tuple of (email_texts, labels) where labels are 1 for spam, 0 for ham
        """
        # Comprehensive spam examples covering various categories
        spam_emails = [
            # Financial scams
            "URGENT! Limited time offer! Click now to claim your prize of $1,000,000!",
            "You've won $1000000! Send your bank details immediately for processing!",
            "FREE MONEY! No strings attached! Act now before offer expires!",
            "Nigerian prince needs help! Share $10 million inheritance! Wire transfer required!",
            "Cash advance! Get $5000 today! No credit check required! Apply now!",
            "Debt consolidation! Reduce payments by 50%! Bad credit OK! Call immediately!",
            "IRS NOTICE: You owe back taxes! Pay immediately or face arrest and penalties!",
            "Lottery winner! $50 million prize waiting! Send processing fee to claim!",
            "Bitcoin doubler! Send 1 BTC, get 2 BTC back! Elon Musk endorses this system!",
            "Binary options trading! Make millions! 99% success rate guaranteed!",
            
            # Product and service scams
            "LOSE WEIGHT FAST! Miracle pill burns fat! No diet or exercise required!",
            "Penis enlargement guaranteed! Doctors hate this one weird trick! Order now!",
            "VIAGRA! CIALIS! Best prices online! No prescription needed! Discreet shipping!",
            "PHARMACY ONLINE! Cheap medications! FDA approved! Order without prescription!",
            "Home refinance! Lowest rates ever! Bad credit welcome! Call now for pre-approval!",
            "Foreclosure help! Save your home! Government program available! Act fast!",
            
            # Tech and dating scams
            "HOT SINGLES in your area! Click to meet them tonight! 18+ only!",
            "MICROSOFT SECURITY ALERT: Computer compromised! Call tech support immediately!",
            "WARNING: Your computer is infected with 47 viruses! Download antivirus now!",
            "Apple iPhone winner! Congratulations! Claim before midnight or lose prize!",
            "Survey reward! $100 Amazon gift card! Complete 5 questions! Limited time offer!",
            
            # Work from home and MLM
            "Work from home! $2000/week guaranteed! No experience required! Start immediately!",
            "Make money fast! Work part-time! Guaranteed income of $5000 monthly!",
            "MLM opportunity! Be your own boss! Recruit friends and family! Easy money!",
            "AMAZING DEAL! 90% OFF everything! Don't miss out on this incredible sale!",
            "Exclusive offer just for you! Limited slots available! VIP access granted!",
            "Get rich quick scheme! Investment opportunity! High returns guaranteed!",
            
            # Fake emergencies and threats
            "Your account will be closed! Update payment information now or lose access!",
            "Security breach detected! Change password immediately! Click here to secure account!",
            "Package delivery failed! Reschedule immediately or return to sender!",
            "Inheritance claim! Deceased relative left $2 million! Contact lawyer immediately!",
            
            # Sophisticated phishing (formal language, specific threats)
            "IRS NOTICE: You owe $5,247 in back taxes. Pay immediately at irs-secure-payment.com or face arrest within 24 hours!",
            "Your Social Security benefits will be suspended. Call 1-800-555-0123 immediately to verify your identity and avoid suspension.",
            "Federal Tax Lien Notice: Property seizure scheduled for next Monday. Contact our office at 555-0199 to arrange payment plan.",
            "Medicare enrollment deadline approaching. Visit medicare-gov-update.com to avoid coverage gaps and penalty fees.",
            "Bank of America: Suspicious activity detected on account ending 4567. Verify at secure-boa-verification.net within 48 hours.",
            "PayPal Security Alert: Account access restricted due to unusual login. Click here to restore access: paypal-security-center.org",
            "Amazon: Your Prime membership payment failed. Update billing info at amazon-billing-update.com to avoid service interruption.",
            "Wells Fargo: Account compromise detected. Immediate action required. Log in at wellsfargo-secure-access.net to secure account.",
            "Chase Bank Alert: Unusual login activity detected. Verify your identity at chase-secure-verify.com within 24 hours.",
            "Apple ID Security: Your account has been restricted. Confirm your identity at apple-id-verification.net immediately.",
            "Microsoft Security: Suspicious activity detected on your account. Update security info at microsoft-secure-portal.com.",
            "Citibank Notice: Account access suspended due to security concerns. Restore access at citi-account-restore.net.",
            "Venmo Alert: Payment authorization required. Verify account details at venmo-secure-auth.com to continue service.",
            "Gmail Security: Unauthorized access detected. Secure your account at gmail-security-center.net within 6 hours."
            "IRS Refund Status: $3,847 tax refund pending. Claim at irs-refund-processing.gov before expiration date.",
            "Court Summons: You are required to appear in federal court. Contact clerk at 555-0188 or face warrant for arrest."
        ]
        
        # Comprehensive legitimate email examples
        ham_emails = [
            # Business communications
            "Hi John, let's schedule our coffee meeting for tomorrow at 3pm downtown.",
            "The quarterly financial report is attached. Please review by Friday and provide feedback.",
            "Thanks for your help with the Johnson project. Your insights were much appreciated!",
            "Reminder: Team meeting scheduled for Monday at 10am in conference room B.",
            "Here's the agenda for tomorrow's board meeting. Please prepare your department updates.",
            "Your flight booking confirmation for March 15th. Gate information will be updated.",
            "Invoice #12345 for consulting services rendered in February is attached for payment.",
            "Monthly newsletter from our engineering team with updates on recent developments.",
            "Annual performance review meeting scheduled with HR for next Tuesday at 2pm.",
            "Project milestone completed successfully. Moving to next phase as planned.",
            
            # Customer service and notifications
            "Your Amazon order #789456123 has been shipped and will arrive in 2-3 business days.",
            "Dear customer, your subscription renewal is due next month for continued service.",
            "Password reset request for your account was completed successfully at 2:34pm today.",
            "Thank you for attending our workshop. Please find the feedback survey attached.",
            "Appointment confirmed with Dr. Smith for Tuesday at 2pm. Bring insurance card.",
            "Your library books are due for return next week. Renewal available online.",
            "Bank statement for account ending in 1234 is now available in your online portal.",
            "Your warranty claim has been approved and processed. Replacement ships Monday.",
            "Software update completed successfully on your device. No action required.",
            "Your gym membership expires at the end of this month. Renewal options available.",
            
            # Personal and social
            "Happy birthday! Hope you have a wonderful celebration with family and friends!",
            "The conference call with the client has been rescheduled to next week Thursday.",
            "Please find the updated project document attached for your review and comments.",
            "Thank you for your recent purchase. Your receipt and warranty info are attached.",
            "Looking forward to working with you on this exciting new project initiative.",
            "Weather alert: Snow expected tomorrow morning. Drive safely and allow extra time.",
            "School parent-teacher conference scheduled for next Friday at 4pm in room 205.",
            "Office closure notification for the Memorial Day holiday weekend. Enjoy your break!",
            "Reminder to submit your timesheet by end of day Friday for payroll processing.",
            "Your package delivery is scheduled for tomorrow between 2-4pm. Please be available.",
            
            # Legitimate financial and government communications
            "Your tax documents are ready for pickup at our office. Please bring photo ID.",
            "Social Security Administration: Annual statement now available online at ssa.gov",
            "IRS Notice: Your tax return has been processed and approved. Refund will be direct deposited.",
            "Medicare Annual Notice: Review your coverage options during open enrollment period.",
            "Bank statement available: Your monthly statement for checking account 1234 is ready online.",
            "Credit card payment confirmation: Your $247.83 payment was received and processed successfully.",
            "Legitimate PayPal notification: Payment of $29.99 received from buyer for item #12345.",
            "Amazon order confirmation: Your order will arrive Tuesday. Track package with number ABC123.",
            "Wells Fargo alert: Low balance notification for checking account ending in 5678.",
            "Court clerk reminder: Jury duty scheduled for March 15th. Please arrive at 8:30 AM."
        ]
        
        # Combine datasets and create labels
        emails = spam_emails + ham_emails
        labels = [1] * len(spam_emails) + [0] * len(ham_emails)
        
        return emails, labels
    
    def train_model(self) -> Dict[str, Any]:
        """
        Train the spam detection model with comprehensive evaluation.
        
        This method implements the complete training pipeline:
        1. Load and preprocess training data
        2. Extract TF-IDF and custom features
        3. Split data for training and testing
        4. Train logistic regression model
        5. Evaluate performance with multiple metrics
        6. Extract important features for explainability
        
        Returns:
            Dictionary containing training metrics and model performance
        """
        print("Loading training data...")
        emails, labels = self.load_training_data()
        
        print(f"Dataset: {len(emails)} emails ({sum(labels)} spam, {len(labels) - sum(labels)} ham)")
        
        # Preprocess text data
        print("Preprocessing text data...")
        processed_emails = [self.preprocess_text(email) for email in emails]
        
        # Extract TF-IDF features
        print("Extracting TF-IDF features...")
        tfidf_features = self.vectorizer.fit_transform(processed_emails)
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        
        # Extract custom features
        print("Extracting custom features...")
        custom_features = np.array([self.extract_custom_features(email) for email in emails])
        custom_features_scaled = self.scaler.fit_transform(custom_features)
        
        # Combine TF-IDF and custom features
        combined_features = np.hstack([tfidf_features.toarray(), custom_features_scaled])
        
        print(f"Total features: {combined_features.shape[1]} (TF-IDF: {tfidf_features.shape[1]}, Custom: {custom_features.shape[1]})")
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, labels, test_size=self.test_size, random_state=42, stratify=labels
        )
        
        # Train the model
        print("Training logistic regression model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate performance metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Cross-validation for robust evaluation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Confusion matrix for detailed analysis
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Extract top spam-indicating features
        feature_importance = self.model.coef_[0]
        tfidf_importance = feature_importance[:len(self.feature_names)]
        
        # Get top 30 spam-indicating words
        top_spam_indices = np.argsort(tfidf_importance)[-30:]
        self.spam_keywords = [self.feature_names[i] for i in top_spam_indices]
        
        # Store training metrics
        self.training_history = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred_test),
            'total_features': combined_features.shape[1],
            'tfidf_features': tfidf_features.shape[1],
            'custom_features': custom_features.shape[1]
        }
        
        # Print training results
        print(f"\n=== Training Results ===")
        print(f"Training Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Top spam keywords: {', '.join(self.spam_keywords[-10:])}")
        
        return self.training_history
    
    def predict_spam(self, email_text: str) -> Dict[str, Any]:
        """
        Predict whether an email is spam with detailed explanation.
        
        This method provides comprehensive spam analysis:
        - Preprocesses the input text
        - Extracts all feature types used in training
        - Generates probability scores
        - Identifies specific keywords that influenced the decision
        - Provides confidence assessment
        
        Args:
            email_text: Raw email content to analyze
            
        Returns:
            Dictionary with prediction results and explanations
        """
        if not self.is_trained:
            print("Model not trained yet. Training now...")
            self.train_model()
        
        # Preprocess the input text
        processed_text = self.preprocess_text(email_text)
        
        # Extract TF-IDF features
        tfidf_features = self.vectorizer.transform([processed_text])
        
        # Extract custom features  
        custom_features = self.extract_custom_features(email_text)
        custom_features_scaled = self.scaler.transform([custom_features])
        
        # Combine features
        combined_features = np.hstack([tfidf_features.toarray(), custom_features_scaled])
        
        # Generate predictions
        spam_probability = self.model.predict_proba(combined_features)[0][1]
        prediction = self.model.predict(combined_features)[0]
        
        # Analyze which keywords contributed to spam classification
        flagged_keywords = []
        words = processed_text.split()
        for word in words:
            if word in self.spam_keywords:
                flagged_keywords.append(word)
        
        # Analyze custom feature contributions
        # Feature mapping: 0-5=basic stats, 6-17=spam patterns, 18-20=email-specific
        custom_flags = []
        if custom_features[2] > 0.3:  # High uppercase ratio
            custom_flags.append("excessive_capitalization")
        if custom_features[5] > 2:  # Multiple dollar signs  
            custom_flags.append("multiple_money_symbols")
        if sum(custom_features[6:18]) > 2:  # Multiple spam patterns (indices 6-17)
            custom_flags.append("spam_patterns_detected")
        if custom_features[14] > 0:  # suspicious_domains pattern 
            custom_flags.append("suspicious_domains_detected")  
        if custom_features[13] > 0:  # authority pattern 
            custom_flags.append("authority_impersonation")
        if custom_features[12] > 0:  # threats pattern 
            custom_flags.append("threat_language_detected")
        if len(custom_features) > 16 and custom_features[16] > 0:  # spoofed_domains pattern 
            custom_flags.append("domain_spoofing_detected")
        if len(custom_features) > 17 and custom_features[17] > 1:  # phishing_verbs pattern 
            custom_flags.append("phishing_language_detected")
        
        # Determine confidence level
        if spam_probability > 0.8 or spam_probability < 0.2:
            confidence = "High"
        elif spam_probability > 0.6 or spam_probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            'is_spam': bool(prediction),
            'spam_probability': float(spam_probability),
            'confidence': confidence,
            'flagged_keywords': flagged_keywords,
            'custom_flags': custom_flags,
            'processed_text': processed_text,
            'feature_count': combined_features.shape[1]
        }
    
    def update_model_with_feedback(self, email_text: str, is_spam_feedback: bool) -> None:
        """
        Update model with user feedback for continuous learning.
        
        Note: This is a simplified implementation. In production, you would:
        - Collect feedback over time
        - Retrain periodically with batches of new data
        - Use online learning algorithms for real-time updates
        
        Args:
            email_text: Email text that was classified
            is_spam_feedback: True if user says it's spam, False otherwise
        """
        feedback_type = "SPAM" if is_spam_feedback else "HAM"
        print(f"Feedback received: Email classified as {feedback_type}")
        print(f"Email preview: {email_text[:100]}...")
        
        # In a production system, you would store this feedback
        # and retrain the model periodically
        
    def save_model(self, filepath: str = 'spam_detector_model.joblib') -> None:
        """
        Save the trained model and all components to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'scaler': self.scaler,
            'spam_keywords': self.spam_keywords,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'spam_patterns': self.spam_patterns
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath: str = 'spam_detector_model.joblib') -> bool:
        """
        Load a previously trained model from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                return False
                
            model_data = joblib.load(filepath)
            
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.spam_keywords = model_data['spam_keywords']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data.get('training_history', {})
            self.spam_patterns = model_data.get('spam_patterns', self.spam_patterns)
            self.is_trained = True
            
            print(f"Model loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the trained model.
        
        Returns:
            Dictionary with model architecture and performance details
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}
            
        return {
            "model_type": "Logistic Regression with Custom Feature Engineering",
            "features": {
                "tfidf_features": len(self.feature_names),
                "custom_features": 21,  # Updated: 6 basic + 12 patterns + 3 email-specific
                "total_features": len(self.feature_names) + 21
            },
            "training_performance": self.training_history,
            "top_spam_indicators": self.spam_keywords[-10:],
            "spam_patterns": list(self.spam_patterns.keys())
        }

# Example usage and testing
if __name__ == "__main__":
    print("=== Email Spam Detector Training ===")
    
    # Initialize and train the detector
    detector = SpamDetector()
    training_results = detector.train_model()
    
    # Save the trained model
    detector.save_model()
    
    # Test with example emails
    test_emails = [
        "URGENT! You've won a million dollars! Click here now to claim your prize!",
        "Hi Sarah, can we schedule our meeting for next Tuesday at 2pm?",
        "FREE MONEY! Limited time offer! Act fast before it's too late!",
        "Your Amazon order will arrive tomorrow. Track your package online.",
        "VIAGRA! CIALIS! Best prices! No prescription needed! Order now!"
    ]
    
    print("\n=== Testing Predictions ===")
    for i, email in enumerate(test_emails, 1):
        result = detector.predict_spam(email)
        print(f"\nTest {i}: {email[:50]}...")
        print(f"Spam probability: {result['spam_probability']:.3f}")
        print(f"Classification: {'SPAM' if result['is_spam'] else 'HAM'}")
        print(f"Confidence: {result['confidence']}")
        print(f"Flagged keywords: {result['flagged_keywords']}")
        print(f"Custom flags: {result['custom_flags']}")
        
    # Display model information
    print("\n=== Model Information ===")
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")