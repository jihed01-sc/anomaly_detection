"""
Utility functions for Network Anomaly Detection Application.

This module provides reusable utility functions for data validation,
preprocessing, and common operations used throughout the application.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import streamlit as st
from config import EXPECTED_COLUMNS, VALIDATION_RANGES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', mode='a')
        ]
    )


def validate_input_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data against expected ranges and types.
    
    Args:
        data: Dictionary containing input features
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for required numeric fields
    for field, config in VALIDATION_RANGES.items():
        if field in data:
            value = data[field]
            if not isinstance(value, (int, float)) or value < config['min'] or value > config['max']:
                errors.append(f"{field} must be between {config['min']} and {config['max']}")
    
    # Validate rate fields (should be between 0 and 1)
    rate_fields = [col for col in EXPECTED_COLUMNS if 'rate' in col]
    for field in rate_fields:
        if field in data:
            value = data[field]
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(f"{field} must be between 0.0 and 1.0")
    
    return len(errors) == 0, errors


def preprocess_input_data(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess input data for model prediction.
    
    Args:
        input_data: Dictionary containing raw input features
        
    Returns:
        Preprocessed DataFrame ready for model prediction
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        # Create a copy to avoid modifying original
        processed_data = input_data.copy()
        
        # Fill missing columns with default values
        for col in EXPECTED_COLUMNS:
            if col not in processed_data:
                if col in VALIDATION_RANGES:
                    processed_data[col] = VALIDATION_RANGES[col]['default']
                elif 'rate' in col:
                    processed_data[col] = 0.0
                else:
                    processed_data[col] = 0
        
        # Convert to DataFrame
        df = pd.DataFrame([processed_data])
        
        # Ensure correct column order
        df = df[EXPECTED_COLUMNS]
        
        # Handle categorical columns if any encoding is needed
        # Note: This assumes the model was trained with encoded categorical features
        # If using label encoding or one-hot encoding, implement that here
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing input data: {str(e)}")
        raise ValueError(f"Failed to preprocess input data: {str(e)}")


def format_prediction_confidence(prediction: int, probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Format prediction results with confidence information.
    
    Args:
        prediction: Predicted class index
        probabilities: Model prediction probabilities (optional)
        
    Returns:
        Dictionary containing formatted prediction information
    """
    from config import CLASS_LABELS
    
    result = {
        'predicted_class': CLASS_LABELS[prediction],
        'predicted_index': prediction,
        'is_anomaly': prediction != 0,
        'confidence': None,
        'all_probabilities': None
    }
    
    if probabilities is not None:
        result['confidence'] = float(probabilities[prediction])
        result['all_probabilities'] = {
            CLASS_LABELS[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
    
    return result


def create_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics for uploaded data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    try:
        stats = {
            'total_records': len(df),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        return stats
    except Exception as e:
        logger.error(f"Error creating summary stats: {str(e)}")
        return {'error': str(e)}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove special characters and replace with underscores
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    return sanitized


def check_model_health(model: Any) -> Dict[str, Any]:
    """
    Perform basic health checks on the loaded model.
    
    Args:
        model: Loaded ML model
        
    Returns:
        Dictionary containing health check results
    """
    health_info = {
        'model_type': type(model).__name__,
        'has_predict_method': hasattr(model, 'predict'),
        'has_predict_proba_method': hasattr(model, 'predict_proba'),
        'model_attributes': []
    }
    
    # Check for common model attributes
    common_attrs = ['n_estimators', 'max_depth', 'random_state', 'feature_importances_']
    for attr in common_attrs:
        if hasattr(model, attr):
            health_info['model_attributes'].append(attr)
    
    return health_info


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for demonstration purposes.
    
    Returns:
        Sample DataFrame with example network traffic data
    """
    sample_data = {
        'duration': [0, 5, 10],
        'protocol_type': ['tcp', 'udp', 'tcp'],
        'service': ['http', 'dns', 'ftp'],
        'flag': ['SF', 'SF', 'S0'],
        'src_bytes': [500, 100, 0],
        'dst_bytes': [1000, 50, 0],
        'count': [1, 2, 3],
        'serror_rate': [0.0, 0.0, 1.0]
    }
    
    # Fill in remaining columns with defaults
    for col in EXPECTED_COLUMNS:
        if col not in sample_data:
            sample_data[col] = [0] * 3
    
    return pd.DataFrame(sample_data)


def display_error_with_details(error_msg: str, details: Optional[str] = None) -> None:
    """
    Display error message with optional details in Streamlit.
    
    Args:
        error_msg: Main error message
        details: Optional detailed error information
    """
    st.error(f"❌ {error_msg}")
    if details:
        with st.expander("Error Details"):
            st.code(details)


def display_success_with_details(success_msg: str, details: Optional[Dict] = None) -> None:
    """
    Display success message with optional details in Streamlit.
    
    Args:
        success_msg: Main success message
        details: Optional detailed information
    """
    st.success(f"✅ {success_msg}")
    if details:
        with st.expander("Details"):
            for key, value in details.items():
                st.write(f"**{key}**: {value}")