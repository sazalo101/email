#!/usr/bin/env python3
"""
Force retrain the spam detector with enhanced patterns and examples.
This script will delete the existing model and train a new one with all the improvements.
"""

import os
from spam_detector import SpamDetector

def main():
    # Remove existing model to force retraining
    model_file = 'spam_detector_model.joblib'
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"Removed existing model: {model_file}")
    
    # Initialize and train new model
    print("Training enhanced spam detector...")
    detector = SpamDetector()
    training_results = detector.train_model()
    
    # Save the new model
    detector.save_model(model_file)
    print(f"New model saved to: {model_file}")
    
    # Test with the problematic email
    test_email = "Bank of America: Suspicious activity detected on account ending 4567. Verify at secure-boa-verification.net within 48 hours"
    
    print(f"\n=== Testing Enhanced Model ===")
    result = detector.predict_spam(test_email)
    
    print(f"Email: {test_email}")
    print(f"Spam Probability: {result['spam_probability']:.1%}")
    print(f"Classification: {'SPAM' if result['is_spam'] else 'HAM'}")
    print(f"Confidence: {result['confidence']}")
    print(f"Flagged Keywords: {result['flagged_keywords']}")
    print(f"Custom Flags: {result['custom_flags']}")
    
    print(f"\n=== Model Performance ===")
    print(f"Training Accuracy: {training_results['train_accuracy']:.3f}")
    print(f"Test Accuracy: {training_results['test_accuracy']:.3f}")
    print(f"Cross-validation: {training_results['cv_mean']:.3f} ± {training_results['cv_std']:.3f}")

if __name__ == "__main__":
    main()