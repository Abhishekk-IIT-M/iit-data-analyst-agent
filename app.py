import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile
import traceback

from data_analyst_agent import DataAnalystAgent
from utils import validate_file, create_error_response, create_success_response

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv', 'txt', 'parquet'}

# Initialize Data Analyst Agent
data_agent = DataAnalystAgent()

@app.route('/')
def index():
    """Serve API documentation page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Main endpoint for data analysis requests
    Accepts file uploads and analysis parameters
    Returns structured JSON with analysis results
    """
    try:
        logger.info("Received analysis request")
        
        # Parse request data
        request_data = {}
        
        # Handle file upload if present
        uploaded_file = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Validate file
                validation_result = validate_file(file, ALLOWED_EXTENSIONS)
                if not validation_result['valid']:
                    return jsonify(create_error_response(str(validation_result['error']))), 400
                
                # Save file temporarily
                filename = secure_filename(file.filename or "uploaded_file")
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                uploaded_file = filepath
                logger.info(f"File uploaded: {filename}")
        
        # Get other parameters from form data or JSON
        if request.is_json:
            request_data = request.get_json() or {}
        else:
            request_data = request.form.to_dict()
        
        # Extract analysis parameters
        analysis_type = request_data.get('analysis_type', 'basic')
        web_url = request_data.get('web_url')
        s3_path = request_data.get('s3_path')
        query = request_data.get('query')
        visualization_type = request_data.get('visualization_type', 'auto')
        statistical_tests_raw = request_data.get('statistical_tests', [])
        statistical_tests = statistical_tests_raw if isinstance(statistical_tests_raw, list) else []
        
        # Validate that at least one data source is provided
        if not any([uploaded_file, web_url, s3_path]):
            return jsonify(create_error_response(
                "No data source provided. Please upload a file, provide a web URL, or specify an S3 path."
            )), 400
        
        # Perform analysis
        result = data_agent.analyze(
            file_path=uploaded_file,
            web_url=web_url,
            s3_path=s3_path,
            analysis_type=analysis_type,
            query=query,
            visualization_type=visualization_type,
            statistical_tests=statistical_tests
        )
        
        # Clean up temporary file
        if uploaded_file and os.path.exists(uploaded_file):
            os.remove(uploaded_file)
        
        logger.info("Analysis completed successfully")
        return jsonify(create_success_response(result))
    
    except RequestEntityTooLarge:
        return jsonify(create_error_response("File too large. Maximum size is 100MB.")), 413
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up temporary file in case of error
        if 'uploaded_file' in locals() and uploaded_file and os.path.exists(uploaded_file):
            os.remove(uploaded_file)
        
        return jsonify(create_error_response(
            f"Analysis failed: {str(e)}"
        )), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify(create_error_response("Endpoint not found")), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify(create_error_response("Method not allowed")), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify(create_error_response("Internal server error")), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
