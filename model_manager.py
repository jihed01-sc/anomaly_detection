"""
Model management module for Network Anomaly Detection Application.

This module handles model loading, validation, and prediction operations
with comprehensive error handling and performance monitoring.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict, Any, List
import logging
import os
from datetime import datetime

from config import (
    MODEL_FILE_PATH, 
    BACKUP_MODEL_PATH, 
    CLASS_LABELS, 
    CONFIDENCE_THRESHOLD,
    EXPECTED_COLUMNS
)
from utils import check_model_health, format_prediction_confidence

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Enhanced model management class with caching, validation, and monitoring.
    """
    
    def __init__(self):
        self.model = None
        self.model_info = {}
        self.load_timestamp = None
        self.prediction_count = 0
        
    @st.cache_resource
    def load_model(_self) -> Tuple[Any, Dict[str, Any]]:
        """
        Load the trained model with comprehensive error handling and validation.
        
        Returns:
            Tuple of (model_object, model_info_dict)
            
        Raises:
            FileNotFoundError: If model file cannot be found
            Exception: If model loading fails
        """
        model_paths = [MODEL_FILE_PATH, BACKUP_MODEL_PATH]
        
        for model_path in model_paths:
            try:
                if os.path.exists(model_path):
                    logger.info(f"Loading model from {model_path}")
                    
                    # Load the model
                    model = joblib.load(model_path)
                    
                    # Perform health checks
                    health_info = check_model_health(model)
                    
                    # Get model metadata
                    model_info = {
                        'file_path': model_path,
                        'file_size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2),
                        'load_timestamp': datetime.now().isoformat(),
                        'health_check': health_info,
                        'model_type': type(model).__name__
                    }
                    
                    # Additional model-specific information
                    if hasattr(model, 'n_estimators'):
                        model_info['n_estimators'] = model.n_estimators
                    if hasattr(model, 'max_depth'):
                        model_info['max_depth'] = model.max_depth
                    if hasattr(model, 'feature_importances_'):
                        model_info['has_feature_importance'] = True
                        model_info['n_features'] = len(model.feature_importances_)
                    
                    logger.info(f"Model loaded successfully: {model_info}")
                    return model, model_info
                    
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                continue
        
        # If we reach here, no model could be loaded
        raise FileNotFoundError(
            f"Could not load model from any of the paths: {model_paths}. "
            "Please ensure the model file exists and is accessible."
        )
    
    def initialize(self) -> bool:
        """
        Initialize the model manager and load the model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.model, self.model_info = self.load_model()
            self.load_timestamp = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            return False
    
    def predict_single(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make a single prediction with confidence scores.
        
        Args:
            input_data: Preprocessed DataFrame with a single row
            
        Returns:
            Dictionary containing prediction results and metadata
            
        Raises:
            ValueError: If model is not loaded or input is invalid
            Exception: If prediction fails
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please initialize the model first.")
        
        if len(input_data) != 1:
            raise ValueError("Input data must contain exactly one row for single prediction.")
        
        try:
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_data)[0]
            
            # Format results
            result = format_prediction_confidence(prediction, probabilities)
            
            # Add metadata
            result.update({
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_info.get('model_type', 'Unknown'),
                'prediction_id': self.prediction_count + 1
            })
            
            self.prediction_count += 1
            logger.info(f"Prediction completed: {result['predicted_class']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions on multiple rows.
        
        Args:
            input_data: Preprocessed DataFrame with multiple rows
            
        Returns:
            DataFrame with original data plus prediction columns
            
        Raises:
            ValueError: If model is not loaded
            Exception: If batch prediction fails
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please initialize the model first.")
        
        if input_data.empty:
            raise ValueError("Input data is empty.")
        
        try:
            logger.info(f"Starting batch prediction for {len(input_data)} records")
            
            # Make predictions
            predictions = self.model.predict(input_data)
            
            # Get prediction probabilities if available
            prediction_probabilities = None
            if hasattr(self.model, 'predict_proba'):
                prediction_probabilities = self.model.predict_proba(input_data)
            
            # Create results DataFrame
            result_df = input_data.copy()
            result_df['Predicted_Class'] = [CLASS_LABELS[pred] for pred in predictions]
            result_df['Predicted_Index'] = predictions
            result_df['Is_Anomaly'] = predictions != 0
            
            # Add confidence scores if available
            if prediction_probabilities is not None:
                result_df['Confidence'] = [
                    prediction_probabilities[i][predictions[i]] 
                    for i in range(len(predictions))
                ]
                
                # Add individual class probabilities
                for j, class_name in enumerate(CLASS_LABELS):
                    result_df[f'Prob_{class_name}'] = prediction_probabilities[:, j]
            
            # Add metadata
            result_df['Prediction_Timestamp'] = datetime.now().isoformat()
            
            self.prediction_count += len(input_data)
            logger.info(f"Batch prediction completed for {len(input_data)} records")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise Exception(f"Batch prediction failed: {str(e)}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the model if available.
        
        Returns:
            DataFrame with feature names and importance scores, or None
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        try:
            importance_df = pd.DataFrame({
                'Feature': EXPECTED_COLUMNS,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {str(e)}")
            return None
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive model statistics and metadata.
        
        Returns:
            Dictionary containing model statistics
        """
        if self.model is None:
            return {'status': 'Model not loaded'}
        
        stats = {
            'status': 'Loaded',
            'load_timestamp': self.load_timestamp.isoformat() if self.load_timestamp else None,
            'predictions_made': self.prediction_count,
            'model_info': self.model_info,
            'expected_features': len(EXPECTED_COLUMNS),
            'class_labels': CLASS_LABELS
        }
        
        return stats
    
    def validate_input_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that input DataFrame has the expected features.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_missing_features)
        """
        missing_features = []
        for feature in EXPECTED_COLUMNS:
            if feature not in df.columns:
                missing_features.append(feature)
        
        return len(missing_features) == 0, missing_features


# Global model manager instance
model_manager = ModelManager()