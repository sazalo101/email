"""
Flask Web Application for Advanced Email Spam Detector

This module provides a web interface and REST API for the spam detection system.
It integrates the machine learning model with a modern web frontend, offering:
- Real-time email spam analysis
- Detailed explanations of classification decisions
- User feedback collection for model improvement
- Responsive web interface for cross-platform compatibility

Key Features:
- RESTful API endpoints for analysis and feedback
- Comprehensive error handling and input validation
- Production-ready deployment configuration
- Detailed logging for debugging and monitoring

Author: DevPost Competition Team
Date: 2025
"""

from flask import Flask, render_template, request, jsonify
from spam_detector import SpamDetector
import os
import logging
from datetime import datetime
import traceback
from typing import Dict, Any

# Configure logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spam_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application with configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Global spam detector instance
detector = None

def initialize_detector() -> SpamDetector:
    """
    Initialize or load the spam detection model.
    
    This function attempts to load a pre-trained model from disk.
    If no saved model exists, it trains a new model from scratch.
    The model is cached globally for efficient request handling.
    
    Returns:
        SpamDetector: Initialized and ready-to-use spam detector instance
        
    Raises:
        Exception: If model initialization or training fails
    """
    global detector
    
    if detector is None:
        try:
            logger.info("Initializing spam detector...")
            detector = SpamDetector()
            
            # Train model on first deployment
            model_path = 'spam_detector_model.joblib'
            if detector.load_model(model_path):
                logger.info(f"Successfully loaded pre-trained model from {model_path}")
            else:
                logger.info("No pre-trained model found. Training new model...")
                training_results = detector.train_model()
                detector.save_model(model_path)
                
                # Log training performance for monitoring
                logger.info(f"Model training completed:")
                logger.info(f"  - Test Accuracy: {training_results['test_accuracy']:.3f}")
                logger.info(f"  - Cross-validation: {training_results['cv_mean']:.3f} ± {training_results['cv_std']:.3f}")
                logger.info(f"  - Total Features: {training_results['total_features']}")
                
        except Exception as e:
            logger.error(f"Failed to initialize spam detector: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    return detector

def validate_email_input(email_text: str) -> Dict[str, Any]:
    """
    Validate email input for analysis.
    
    This function performs comprehensive input validation to ensure:
    - Input is not empty or just whitespace
    - Content length is within reasonable bounds
    - Text contains meaningful content for analysis
    
    Args:
        email_text: Raw email content from user input
        
    Returns:
        Dict with validation result and error message if invalid
    """
    if not email_text or not isinstance(email_text, str):
        return {'valid': False, 'error': 'Email text is required'}
    
    # Strip whitespace and check if content remains
    cleaned_text = email_text.strip()
    if not cleaned_text:
        return {'valid': False, 'error': 'Email text cannot be empty or just whitespace'}
    
    # Check minimum content length for meaningful analysis
    if len(cleaned_text) < 5:
        return {'valid': False, 'error': 'Email text is too short for analysis (minimum 5 characters)'}
    
    # Check maximum content length to prevent resource exhaustion
    if len(cleaned_text) > 10000:
        return {'valid': False, 'error': 'Email text is too long (maximum 10,000 characters)'}
    
    return {'valid': True, 'cleaned_text': cleaned_text}

@app.route('/')
def index():
    """
    Serve the main web interface.
    
    This endpoint renders the HTML interface for spam detection.
    The template includes all necessary JavaScript and CSS for
    interactive analysis and result visualization.
    
    Returns:
        Rendered HTML template for the web interface
    """
    try:
        logger.info("Serving main web interface")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving main page: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_email():
    """
    Analyze email for spam detection via REST API.
    
    This endpoint accepts email content and returns detailed spam analysis:
    - Spam probability percentage
    - Binary classification (spam/not spam)
    - Confidence level assessment
    - Flagged keywords that influenced the decision
    - Custom pattern flags (capitalization, money symbols, etc.)
    
    Expected JSON Input:
        {
            "email_text": "Email content to analyze"
        }
    
    JSON Response Format:
        {
            "spam_probability": 85.7,
            "is_spam": true,
            "classification": "SPAM",
            "confidence": "High",
            "flagged_keywords": ["urgent", "winner", "click"],
            "custom_flags": ["excessive_capitalization"],
            "analysis_timestamp": "2025-01-XX XX:XX:XX",
            "processing_time_ms": 45
        }
    
    Returns:
        JSON response with analysis results or error message
    """
    start_time = datetime.now()
    
    try:
        # Parse and validate JSON request
        if not request.is_json:
            logger.warning("Received non-JSON request to /analyze endpoint")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        email_text = data.get('email_text', '')
        
        # Validate email input
        validation_result = validate_email_input(email_text)
        if not validation_result['valid']:
            logger.info(f"Invalid input rejected: {validation_result['error']}")
            return jsonify({'error': validation_result['error']}), 400
        
        cleaned_text = validation_result['cleaned_text']
        
        # Initialize detector if needed
        spam_detector = initialize_detector()
        
        # Perform spam analysis
        logger.info(f"Analyzing email of length {len(cleaned_text)} characters")
        result = spam_detector.predict_spam(cleaned_text)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Format response with additional metadata
        response = {
            'spam_probability': round(result['spam_probability'] * 100, 1),
            'is_spam': result['is_spam'],
            'classification': 'SPAM' if result['is_spam'] else 'NOT SPAM',
            'confidence': result['confidence'],
            'flagged_keywords': result['flagged_keywords'],
            'custom_flags': result.get('custom_flags', []),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time_ms': round(processing_time, 1),
            'feature_count': result.get('feature_count', 0)
        }
        
        # Log analysis result for monitoring
        logger.info(f"Analysis completed: {response['classification']} "
                   f"({response['spam_probability']}%) in {response['processing_time_ms']}ms")
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Error during email analysis: {str(e)}")
        logger.error(f"Processing time before error: {processing_time:.1f}ms")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'Internal server error during analysis',
            'processing_time_ms': round(processing_time, 1)
        }), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback for model improvement.
    
    This endpoint collects user corrections on spam classifications.
    The feedback is used to:
    - Log misclassifications for model retraining
    - Track model performance over time
    - Identify areas for improvement
    
    Expected JSON Input:
        {
            "email_text": "Original email content",
            "is_spam": true,
            "original_prediction": false,
            "user_comment": "Optional feedback comment"
        }
    
    JSON Response:
        {
            "message": "Feedback received successfully",
            "feedback_id": "unique_identifier",
            "timestamp": "2025-01-XX XX:XX:XX"
        }
    
    Returns:
        JSON confirmation of feedback submission
    """
    try:
        # Parse and validate JSON request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        email_text = data.get('email_text', '')
        is_spam_feedback = data.get('is_spam', False)
        original_prediction = data.get('original_prediction', None)
        user_comment = data.get('user_comment', '')
        
        # Validate required fields
        validation_result = validate_email_input(email_text)
        if not validation_result['valid']:
            return jsonify({'error': f'Invalid email text: {validation_result["error"]}'}), 400
        
        # Initialize detector if needed
        spam_detector = initialize_detector()
        
        # Process feedback
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update model with feedback (simplified implementation)
        spam_detector.update_model_with_feedback(email_text, is_spam_feedback)
        
        # Log feedback for analysis and future model retraining
        feedback_log = {
            'feedback_id': feedback_id,
            'timestamp': timestamp,
            'email_length': len(email_text),
            'user_classification': 'SPAM' if is_spam_feedback else 'HAM',
            'original_prediction': original_prediction,
            'correction_made': original_prediction != is_spam_feedback if original_prediction is not None else None,
            'user_comment': user_comment[:500] if user_comment else None  # Limit comment length
        }
        
        logger.info(f"Feedback received: {feedback_log}")
        
        response = {
            'message': 'Thank you for your feedback! This helps improve our spam detection accuracy.',
            'feedback_id': feedback_id,
            'timestamp': timestamp
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'Internal server error while processing feedback'
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get comprehensive information about the trained model.
    
    This endpoint provides technical details about the model:
    - Architecture and algorithm details
    - Feature engineering approach
    - Training performance metrics
    - Model configuration parameters
    
    This information is useful for:
    - Technical users understanding the system
    - Model debugging and optimization
    - Transparency and explainability
    
    Returns:
        JSON response with detailed model information
    """
    try:
        # Initialize detector if needed
        spam_detector = initialize_detector()
        
        # Get comprehensive model information
        model_info = spam_detector.get_model_info()
        
        # Add runtime information
        model_info['runtime_info'] = {
            'python_version': os.sys.version,
            'flask_version': '3.0+',
            'scikit_learn_version': '1.4+',
            'deployment_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("Model information requested and served")
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        return jsonify({'error': 'Unable to retrieve model information'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring and deployment.
    
    This endpoint provides system status information:
    - Model initialization status
    - Basic system health indicators
    - Response time measurement
    
    Used by:
    - Load balancers for health monitoring
    - Deployment systems for readiness checks
    - Monitoring tools for uptime tracking
    
    Returns:
        JSON response with health status
    """
    start_time = datetime.now()
    
    try:
        # Check if detector is initialized
        detector_status = detector is not None and detector.is_trained
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        health_info = {
            'status': 'healthy' if detector_status else 'initializing',
            'model_loaded': detector_status,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'response_time_ms': round(response_time, 1),
            'version': '1.0.0'
        }
        
        status_code = 200 if detector_status else 503
        return jsonify(health_info), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with JSON response."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors with JSON response."""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle request size limit exceeded."""
    return jsonify({'error': 'Request too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

# Production deployment configuration
if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 10000))  # Render default port
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Starting Flask application on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Initialize detector on startup for faster first requests
    try:
        initialize_detector()
        logger.info("Spam detector initialized successfully on startup")
    except Exception as e:
        logger.error(f"Failed to initialize detector on startup: {str(e)}")
        logger.error("Application will attempt to initialize on first request")
    
    # Start the Flask server
    app.run(host=host, port=port, debug=debug, threaded=True)