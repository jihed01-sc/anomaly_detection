"""
Configuration settings for Network Anomaly Detection Application.

This module contains all configuration constants and settings used throughout
the application, following the principle of centralized configuration management.
"""

from typing import List, Dict, Any
import os

# Model Configuration
MODEL_FILE_PATH: str = "network_anomaly_detection_model.joblib"
BACKUP_MODEL_PATH: str = "model_artifacts.joblib"  # Fallback for legacy naming

# Feature Configuration - Expected input columns for the trained model
EXPECTED_COLUMNS: List[str] = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Classification Labels
CLASS_LABELS: List[str] = ['Normal', 'DoS', 'Probe', 'Privilege Escalation', 'Access']

# UI Configuration
APP_TITLE: str = "🛡️ Network Anomaly Detection System"
APP_DESCRIPTION: str = """
This advanced application uses a trained Random Forest model to detect network intrusions 
and classify traffic as normal or various types of attacks. Built on the NSL-KDD dataset, 
it provides comprehensive anomaly detection with confidence scores and detailed analysis.
"""

# Input Field Configurations
PROTOCOL_TYPES: List[str] = ["tcp", "udp", "icmp"]
FLAG_TYPES: List[str] = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3"]
COMMON_SERVICES: List[str] = [
    "http", "ftp", "ftp_data", "smtp", "telnet", "domain_u", "private", 
    "pop_3", "finger", "imap4", "other", "ssh", "eco_i", "auth"
]

# Validation Ranges
VALIDATION_RANGES: Dict[str, Dict[str, Any]] = {
    'duration': {'min': 0, 'max': 3600, 'default': 0},
    'src_bytes': {'min': 0, 'max': 1e9, 'default': 0},
    'dst_bytes': {'min': 0, 'max': 1e9, 'default': 0},
    'count': {'min': 0, 'max': 1000, 'default': 1},
    'srv_count': {'min': 0, 'max': 1000, 'default': 1},
    'dst_host_count': {'min': 0, 'max': 255, 'default': 1},
    'dst_host_srv_count': {'min': 0, 'max': 255, 'default': 1}
}

# Performance thresholds
CONFIDENCE_THRESHOLD: float = 0.8
ANOMALY_THRESHOLD: float = 0.5

# File upload settings
MAX_FILE_SIZE_MB: int = 10
ALLOWED_FILE_TYPES: List[str] = ["csv", "txt"]

# Logging configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"