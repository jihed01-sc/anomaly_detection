#!/usr/bin/env python3
"""
Enhanced Network Anomaly Detection Model Training Script.

This script provides an improved version of the original Jupyter notebook code
with better error handling, logging, type hints, and modular design.
Based on the NSL-KDD dataset for network intrusion detection.

Author: Enhanced Code Analysis
Version: 2.0
"""

import numpy as np
import pandas as pd
import logging
import joblib
import requests
import zipfile
import io
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# Visualization imports
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class NetworkAnomalyDetector:
    """
    Enhanced Network Anomaly Detection Model with comprehensive training pipeline.
    
    This class encapsulates the complete machine learning pipeline for network
    intrusion detection including data loading, preprocessing, training, and evaluation.
    """
    
    def __init__(self, 
                 dataset_url: str = "https://academy.hackthebox.com/storage/modules/292/KDD_dataset.zip",
                 random_state: int = 1337,
                 test_size: float = 0.2,
                 validation_size: float = 0.3):
        """
        Initialize the Network Anomaly Detector.
        
        Args:
            dataset_url: URL to download the NSL-KDD dataset
            random_state: Random seed for reproducibility
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
        """
        self.dataset_url = dataset_url
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.model: Optional[RandomForestClassifier] = None
        
        # Training splits
        self.train_X: Optional[pd.DataFrame] = None
        self.train_y: Optional[pd.Series] = None
        self.val_X: Optional[pd.DataFrame] = None
        self.val_y: Optional[pd.Series] = None
        self.test_X: Optional[pd.DataFrame] = None
        self.test_y: Optional[pd.Series] = None
        
        # Results storage
        self.training_history: Dict[str, Any] = {}
        
        # Define feature columns
        self.feature_columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
        ]
        
        # Define attack categorization
        self.attack_categories = {
            'dos_attacks': ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
                           'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'],
            'probe_attacks': ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'],
            'privilege_attacks': ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 
                                 'rootkit', 'sqlattack', 'xterm'],
            'access_attacks': ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 
                              'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 
                              'snmpguess', 'spy', 'warezclient', 'warezmaster', 
                              'xclock', 'xsnoop']
        }
        
        self.class_labels = ['Normal', 'DoS', 'Probe', 'Privilege Escalation', 'Access']
        
        logger.info("NetworkAnomalyDetector initialized successfully")
    
    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download and extract the NSL-KDD dataset.
        
        Args:
            force_download: Whether to download even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        dataset_file = "KDD+.txt"
        
        if os.path.exists(dataset_file) and not force_download:
            logger.info(f"Dataset file {dataset_file} already exists. Skipping download.")
            return True
        
        try:
            logger.info(f"Downloading dataset from {self.dataset_url}")
            
            response = requests.get(self.dataset_url, timeout=30)
            response.raise_for_status()
            
            # Extract the zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall('.')
                
            if os.path.exists(dataset_file):
                file_size = os.path.getsize(dataset_file) / (1024 * 1024)
                logger.info(f"Dataset downloaded successfully. Size: {file_size:.2f} MB")
                return True
            else:
                logger.error("Dataset file not found after extraction")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            return False
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during dataset download: {str(e)}")
            return False
    
    def load_data(self, file_path: str = "KDD+.txt") -> bool:
        """
        Load the NSL-KDD dataset from file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading dataset from {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"Dataset file {file_path} not found")
                return False
            
            # Load data with proper column names
            self.raw_data = pd.read_csv(file_path, names=self.feature_columns)
            
            logger.info(f"Dataset loaded successfully. Shape: {self.raw_data.shape}")
            logger.info(f"Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Basic data quality checks
            self._perform_data_quality_checks()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return False
    
    def _perform_data_quality_checks(self) -> None:
        """Perform basic data quality checks on the loaded dataset."""
        if self.raw_data is None:
            return
        
        # Check for missing values
        missing_values = self.raw_data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found {missing_values.sum()} missing values")
            logger.debug(f"Missing values by column:\n{missing_values[missing_values > 0]}")
        
        # Check data types
        logger.info(f"Data types:\n{self.raw_data.dtypes.value_counts()}")
        
        # Check target distribution
        attack_distribution = self.raw_data['attack'].value_counts()
        logger.info(f"Attack type distribution:\n{attack_distribution.head(10)}")
    
    def preprocess_data(self) -> bool:
        """
        Preprocess the raw data for machine learning.
        
        Returns:
            True if successful, False otherwise
        """
        if self.raw_data is None:
            logger.error("No raw data available. Please load data first.")
            return False
        
        try:
            logger.info("Starting data preprocessing")
            
            # Create a copy for processing
            df = self.raw_data.copy()
            
            # Create multi-class target variable
            df['attack_map'] = df['attack'].apply(self._map_attack_to_category)
            
            # Encode categorical variables
            features_to_encode = ['protocol_type', 'service']
            encoded_features = pd.get_dummies(df[features_to_encode])
            
            # Select numeric features
            numeric_features = [
                'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 
                'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
                'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                'dst_host_srv_rerror_rate'
            ]
            
            # Combine features
            self.processed_data = encoded_features.join(df[numeric_features + ['attack_map']])
            
            logger.info(f"Preprocessing completed. Final shape: {self.processed_data.shape}")
            logger.info(f"Feature columns: {len(self.processed_data.columns) - 1}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            return False
    
    def _map_attack_to_category(self, attack: str) -> int:
        """
        Map attack types to category indices.
        
        Args:
            attack: Attack type string
            
        Returns:
            Category index (0=Normal, 1=DoS, 2=Probe, 3=Privilege, 4=Access)
        """
        if attack == 'normal':
            return 0
        elif attack in self.attack_categories['dos_attacks']:
            return 1
        elif attack in self.attack_categories['probe_attacks']:
            return 2
        elif attack in self.attack_categories['privilege_attacks']:
            return 3
        elif attack in self.attack_categories['access_attacks']:
            return 4
        else:
            logger.warning(f"Unknown attack type: {attack}. Mapping to Normal.")
            return 0
    
    def split_data(self) -> bool:
        """
        Split the processed data into training, validation, and test sets.
        
        Returns:
            True if successful, False otherwise
        """
        if self.processed_data is None:
            logger.error("No processed data available. Please preprocess data first.")
            return False
        
        try:
            logger.info("Splitting data into train/validation/test sets")
            
            # Separate features and target
            X = self.processed_data.drop('attack_map', axis=1)
            y = self.processed_data['attack_map']
            
            # First split: train+val vs test
            train_val_X, self.test_X, train_val_y, self.test_y = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Second split: train vs validation
            self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(
                train_val_X, train_val_y, 
                test_size=self.validation_size, 
                random_state=self.random_state, 
                stratify=train_val_y
            )
            
            logger.info(f"Data split completed:")
            logger.info(f"  Training set: {self.train_X.shape[0]} samples")
            logger.info(f"  Validation set: {self.val_X.shape[0]} samples")
            logger.info(f"  Test set: {self.test_X.shape[0]} samples")
            
            # Log class distribution
            for name, y_subset in [("Train", self.train_y), ("Validation", self.val_y), ("Test", self.test_y)]:
                class_dist = y_subset.value_counts().sort_index()
                logger.info(f"  {name} class distribution: {dict(class_dist)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}")
            return False
    
    def train_model(self, 
                   n_estimators: int = 100,
                   max_depth: Optional[int] = None,
                   min_samples_split: int = 2,
                   min_samples_leaf: int = 1,
                   n_jobs: int = -1) -> bool:
        """
        Train the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            n_jobs: Number of parallel jobs
            
        Returns:
            True if successful, False otherwise
        """
        if any(x is None for x in [self.train_X, self.train_y]):
            logger.error("Training data not available. Please split data first.")
            return False
        
        try:
            logger.info("Starting model training")
            
            # Initialize model
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=n_jobs,
                verbose=1
            )
            
            # Record training start time
            start_time = datetime.now()
            
            # Train the model
            self.model.fit(self.train_X, self.train_y)
            
            # Record training duration
            training_duration = datetime.now() - start_time
            
            logger.info(f"Model training completed in {training_duration}")
            
            # Store training metadata
            self.training_history = {
                'training_start': start_time.isoformat(),
                'training_duration': str(training_duration),
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'training_samples': len(self.train_X),
                'n_features': self.train_X.shape[1],
                'model_size_mb': self._estimate_model_size()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB."""
        if self.model is None:
            return 0.0
        
        try:
            # Serialize model to estimate size
            model_bytes = joblib.dumps(self.model)
            size_mb = len(model_bytes) / (1024 * 1024)
            return round(size_mb, 2)
        except Exception:
            return 0.0
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on validation and test sets.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            logger.error("No trained model available.")
            return {}
        
        try:
            logger.info("Evaluating model performance")
            
            evaluation_results = {}
            
            # Evaluate on validation set
            val_predictions = self.model.predict(self.val_X)
            val_metrics = self._calculate_metrics(self.val_y, val_predictions, "Validation")
            evaluation_results['validation'] = val_metrics
            
            # Evaluate on test set
            test_predictions = self.model.predict(self.test_X)
            test_metrics = self._calculate_metrics(self.test_y, test_predictions, "Test")
            evaluation_results['test'] = test_metrics
            
            # Log results
            logger.info("Evaluation completed:")
            for dataset, metrics in evaluation_results.items():
                logger.info(f"  {dataset.capitalize()} Set:")
                logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"    Precision: {metrics['precision']:.4f}")
                logger.info(f"    Recall: {metrics['recall']:.4f}")
                logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {}
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics for predictions."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_labels,
            output_dict=True
        )
        metrics['classification_report'] = class_report
        
        return metrics
    
    def visualize_results(self, dataset: str = 'test') -> None:
        """
        Create visualizations for model evaluation results.
        
        Args:
            dataset: Which dataset to visualize ('validation' or 'test')
        """
        if self.model is None:
            logger.error("No trained model available for visualization.")
            return
        
        try:
            # Select appropriate dataset
            if dataset == 'validation':
                X, y_true = self.val_X, self.val_y
                title_suffix = "Validation Set"
            else:
                X, y_true = self.test_X, self.test_y
                title_suffix = "Test Set"
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Create confusion matrix visualization
            plt.figure(figsize=(10, 8))
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(
                conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.class_labels,
                yticklabels=self.class_labels
            )
            plt.title(f'Network Anomaly Detection - {title_suffix}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{dataset}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Feature importance visualization
            if hasattr(self.model, 'feature_importances_'):
                self._visualize_feature_importance()
            
            logger.info(f"Visualizations saved for {dataset} set")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
    
    def _visualize_feature_importance(self, top_n: int = 20) -> None:
        """Visualize feature importance."""
        try:
            feature_names = self.train_X.columns
            importances = self.model.feature_importances_
            
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Feature importance visualization failed: {str(e)}")
    
    def save_model(self, filepath: str = 'network_anomaly_detection_model.joblib') -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No trained model available to save.")
            return False
        
        try:
            logger.info(f"Saving model to {filepath}")
            
            # Save model
            joblib.dump(self.model, filepath)
            
            # Verify save
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"Model saved successfully. File size: {file_size:.2f} MB")
                
                # Save metadata
                metadata = {
                    'model_type': type(self.model).__name__,
                    'feature_names': list(self.train_X.columns),
                    'class_labels': self.class_labels,
                    'training_history': self.training_history,
                    'save_timestamp': datetime.now().isoformat()
                }
                
                metadata_file = filepath.replace('.joblib', '_metadata.json')
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Model metadata saved to {metadata_file}")
                return True
            else:
                logger.error("Model file was not created")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete machine learning pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting complete ML pipeline")
        
        steps = [
            ("Download Dataset", self.download_dataset),
            ("Load Data", self.load_data),
            ("Preprocess Data", self.preprocess_data),
            ("Split Data", self.split_data),
            ("Train Model", self.train_model),
            ("Evaluate Model", self.evaluate_model),
            ("Save Model", self.save_model)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Executing step: {step_name}")
            try:
                result = step_func()
                if result is False:
                    logger.error(f"Step '{step_name}' failed")
                    return False
                logger.info(f"Step '{step_name}' completed successfully")
            except Exception as e:
                logger.error(f"Step '{step_name}' failed with exception: {str(e)}")
                return False
        
        # Generate visualizations
        logger.info("Generating visualizations")
        self.visualize_results('validation')
        self.visualize_results('test')
        
        logger.info("Complete ML pipeline finished successfully")
        return True


def main():
    """Main function to run the enhanced network anomaly detection training."""
    print("Enhanced Network Anomaly Detection Model Training")
    print("=" * 50)
    
    # Initialize detector
    detector = NetworkAnomalyDetector()
    
    # Run complete pipeline
    success = detector.run_complete_pipeline()
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"Model saved as: network_anomaly_detection_model.joblib")
        print(f"Visualizations saved as PNG files")
        print(f"Training logs saved to: training.log")
    else:
        print("\n❌ Training failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())