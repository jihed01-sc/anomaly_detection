# Network Anomaly Detection - Deployment Guide

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended for model training
- 1GB+ disk space for dataset and model files

### Installation

1. **Install required packages**:
```bash
pip install -r requirements.txt
```

2. **Run the enhanced Streamlit application**:
```bash
streamlit run enhanced_app.py
```

3. **Access the application**:
Open your browser and navigate to `http://localhost:8501`

## File Structure

```
anomaly_detection/
├── enhanced_app.py              # Enhanced Streamlit application
├── app.py                       # Original Streamlit app
├── config.py                    # Configuration settings
├── utils.py                     # Utility functions
├── model_manager.py             # Model management
├── enhanced_training.py         # Enhanced training script
├── requirements.txt             # Python dependencies
├── network_anomaly_detection_model.joblib  # Pre-trained model
├── Anomaly detection.ipynb      # Original Jupyter notebook
├── ANALYSIS_REPORT.md           # Comprehensive analysis
└── DEPLOYMENT_GUIDE.md          # This file
```

## Application Features

### 🔧 Manual Input Mode
- Interactive form for single prediction
- Real-time input validation
- Confidence scoring with visualization
- Support for all network traffic parameters

### 📁 File Upload Mode
- Batch processing of CSV files
- Data preview and summary statistics
- Comprehensive results analysis
- Export functionality with timestamps

### 📋 Sample Data Mode
- Demonstration with pre-loaded examples
- Quick testing of application features
- Educational purposes

### 🎯 Advanced Features
- **Feature Importance**: Understand which features contribute most to predictions
- **Confidence Scoring**: Get prediction probabilities for all classes
- **Interactive Visualizations**: Plotly-based charts and graphs
- **Model Health Monitoring**: Track model performance and usage
- **Export Capabilities**: Download results as CSV files

## Model Information

### Attack Categories
The model classifies network traffic into 5 categories:

1. **Normal**: Legitimate network traffic
2. **DoS**: Denial of Service attacks
3. **Probe**: Information gathering attacks
4. **Privilege Escalation**: Unauthorized access elevation
5. **Access**: Unauthorized access to resources

### Technical Specifications
- **Algorithm**: Random Forest Classifier
- **Features**: 39 network traffic characteristics
- **Training Dataset**: NSL-KDD (enhanced KDD Cup 1999)
- **Model Size**: ~21MB
- **Expected Accuracy**: >95% on test data

## Deployment Options

### 1. Local Development

**For development and testing**:
```bash
# Clone repository
git clone <repository-url>
cd anomaly_detection

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run enhanced_app.py
```

### 2. Streamlit Community Cloud

**Best for: Free hosting, simple deployment**

1. **Push to GitHub**: Ensure your code is in a public GitHub repository
2. **Connect to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: 
   - Select your repository
   - Choose `enhanced_app.py` as the main file
   - Set Python version to 3.11
   - Deploy with one click

**Configuration** (`.streamlit/config.toml`):
```toml
[server]
maxUploadSize = 10
enableCORS = false

[theme]
primaryColor = "#1e3c72"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### 3. Heroku Deployment

**Best for: More control, custom domains**

**Required files**:

`Procfile`:
```
web: streamlit run enhanced_app.py --server.port=$PORT --server.address=0.0.0.0
```

`runtime.txt`:
```
python-3.11.0
```

**Deployment steps**:
```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku main

# Open application
heroku open
```

### 4. Docker Deployment

**Best for: Consistent environments, scaling**

`Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "enhanced_app.py", "--server.address=0.0.0.0"]
```

**Build and run**:
```bash
# Build image
docker build -t anomaly-detector .

# Run container
docker run -p 8501:8501 anomaly-detector
```

### 5. AWS/GCP/Azure Cloud Deployment

**For production environments**:

- **AWS**: Use ECS, Lambda, or EC2 with Application Load Balancer
- **GCP**: Deploy on Cloud Run or Compute Engine
- **Azure**: Use Container Instances or App Service

## Production Considerations

### Performance Optimization

1. **Caching**: Enable Streamlit caching for better performance
```python
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')
```

2. **Memory Management**: Monitor memory usage with large files
3. **Load Balancing**: Use multiple instances for high traffic
4. **CDN**: Serve static assets via CDN for global users

### Security

1. **HTTPS**: Always use HTTPS in production
2. **Input Validation**: Comprehensive input sanitization (already implemented)
3. **Rate Limiting**: Implement API rate limiting for file uploads
4. **Error Handling**: Secure error messages (no sensitive information exposure)

### Monitoring

1. **Application Logs**: Monitor application logs for errors
2. **Performance Metrics**: Track response times and resource usage
3. **Model Drift**: Monitor prediction accuracy over time
4. **Health Checks**: Implement health check endpoints

### Environment Variables

Set these environment variables for production:

```bash
export LOG_LEVEL=INFO
export MAX_UPLOAD_SIZE=10
export MODEL_PATH=/path/to/model.joblib
```

## Troubleshooting

### Common Issues

1. **Model Not Found**:
   - Ensure `network_anomaly_detection_model.joblib` exists
   - Check file permissions
   - Verify model path in config

2. **Memory Issues**:
   - Reduce file upload size limit
   - Use streaming for large files
   - Increase container memory allocation

3. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Ensure proper virtual environment

4. **Performance Issues**:
   - Enable Streamlit caching
   - Optimize data processing
   - Use appropriate server resources

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Usage

### Single Prediction

```python
import requests

# Prepare data
data = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',
    'src_bytes': 500,
    'dst_bytes': 1000,
    # ... other features
}

# Make prediction (if API endpoint is available)
response = requests.post('http://localhost:8501/predict', json=data)
result = response.json()
```

### Batch Processing

Upload CSV file via the web interface or programmatically:

```python
files = {'file': open('network_data.csv', 'rb')}
response = requests.post('http://localhost:8501/batch_predict', files=files)
```

## Model Retraining

To retrain the model with new data:

```bash
# Run enhanced training script
python enhanced_training.py

# This will:
# 1. Download fresh NSL-KDD dataset
# 2. Preprocess data
# 3. Train new model
# 4. Evaluate performance
# 5. Save updated model
```

## Support

For issues and questions:
1. Check the application logs
2. Review the troubleshooting section
3. Consult the comprehensive analysis report
4. Check model health status in the application

## Performance Benchmarks

### Expected Performance
- **Single Prediction**: <100ms
- **Batch Processing**: ~1000 records/second
- **Model Loading**: <3 seconds
- **Memory Usage**: <512MB (without large uploads)

### Scaling Recommendations
- **Small Scale**: Single instance, 1GB RAM
- **Medium Scale**: Load balancer + 2-3 instances, 2GB RAM each
- **Large Scale**: Auto-scaling group, CDN, database for logging