"""
Enhanced Network Anomaly Detection Streamlit Application.

This application provides a comprehensive interface for network intrusion detection
using machine learning. It includes advanced features like confidence scoring,
batch processing, data visualization, and model performance metrics.

Author: Enhanced by Code Analysis
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import io

# Import custom modules
from config import (
    APP_TITLE, 
    APP_DESCRIPTION, 
    PROTOCOL_TYPES, 
    FLAG_TYPES, 
    COMMON_SERVICES,
    VALIDATION_RANGES,
    CLASS_LABELS,
    MAX_FILE_SIZE_MB,
    ALLOWED_FILE_TYPES
)
from utils import (
    setup_logging, 
    validate_input_data, 
    preprocess_input_data,
    create_summary_stats,
    load_sample_data,
    display_error_with_details,
    display_success_with_details
)
from model_manager import model_manager

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Network Anomaly Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1e3c72;
    }
    
    .prediction-normal {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .prediction-anomaly {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)


def initialize_app() -> bool:
    """
    Initialize the application and load the model.
    
    Returns:
        True if initialization successful, False otherwise
    """
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False
    
    if not st.session_state.model_initialized:
        with st.spinner("Initializing model..."):
            success = model_manager.initialize()
            st.session_state.model_initialized = success
            if success:
                st.session_state.model_stats = model_manager.get_model_stats()
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check the model file.")
        return success
    
    return True


def render_header():
    """Render the application header with styling."""
    st.markdown(f"""
    <div class="main-header">
        <h1>{APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
    </div>
    """, unsafe_allow_html=True)


def render_model_info():
    """Render model information in the sidebar."""
    st.sidebar.header("🤖 Model Information")
    
    if st.session_state.get('model_initialized', False):
        stats = st.session_state.get('model_stats', {})
        
        st.sidebar.success("Model Status: Loaded ✅")
        
        with st.sidebar.expander("Model Details"):
            st.write(f"**Type**: {stats.get('model_info', {}).get('model_type', 'Unknown')}")
            st.write(f"**Features**: {stats.get('expected_features', 'Unknown')}")
            st.write(f"**Classes**: {len(CLASS_LABELS)}")
            st.write(f"**Predictions Made**: {stats.get('predictions_made', 0)}")
            
            model_info = stats.get('model_info', {})
            if 'file_size_mb' in model_info:
                st.write(f"**Model Size**: {model_info['file_size_mb']} MB")
            if 'n_estimators' in model_info:
                st.write(f"**Estimators**: {model_info['n_estimators']}")
    else:
        st.sidebar.error("Model Status: Not Loaded ❌")


def render_manual_input() -> Optional[Dict[str, Any]]:
    """
    Render manual input form for single prediction.
    
    Returns:
        Dictionary containing input data or None
    """
    st.header("🔧 Manual Data Input")
    
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Connection Details")
            duration = st.number_input(
                "Duration (seconds)", 
                min_value=0, 
                max_value=3600, 
                value=0,
                help="Length of the connection in seconds"
            )
            
            protocol_type = st.selectbox(
                "Protocol Type", 
                PROTOCOL_TYPES,
                help="Network protocol used"
            )
            
            service = st.selectbox(
                "Service", 
                COMMON_SERVICES,
                help="Network service being accessed"
            )
            
            flag = st.selectbox(
                "Connection Flag", 
                FLAG_TYPES,
                help="Status of the connection"
            )
        
        with col2:
            st.subheader("Traffic Volume")
            src_bytes = st.number_input(
                "Source Bytes", 
                min_value=0, 
                value=0,
                help="Number of bytes sent from source"
            )
            
            dst_bytes = st.number_input(
                "Destination Bytes", 
                min_value=0, 
                value=0,
                help="Number of bytes sent to destination"
            )
            
            count = st.number_input(
                "Connection Count", 
                min_value=0, 
                max_value=1000, 
                value=1,
                help="Number of connections to the same host"
            )
            
            srv_count = st.number_input(
                "Service Count", 
                min_value=0, 
                max_value=1000, 
                value=1,
                help="Number of connections to the same service"
            )
        
        with col3:
            st.subheader("Error Rates")
            serror_rate = st.slider(
                "SYN Error Rate", 
                0.0, 1.0, 0.0,
                help="Percentage of connections with SYN errors"
            )
            
            rerror_rate = st.slider(
                "REJ Error Rate", 
                0.0, 1.0, 0.0,
                help="Percentage of connections with REJ errors"
            )
            
            same_srv_rate = st.slider(
                "Same Service Rate", 
                0.0, 1.0, 1.0,
                help="Percentage of connections to the same service"
            )
            
            dst_host_count = st.number_input(
                "Destination Host Count", 
                min_value=0, 
                max_value=255, 
                value=1,
                help="Number of destination hosts"
            )
        
        submit_button = st.form_submit_button(
            "🔍 Detect Anomaly", 
            type="primary",
            use_container_width=True
        )
        
        if submit_button:
            input_data = {
                'duration': duration,
                'protocol_type': protocol_type,
                'service': service,
                'flag': flag,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'count': count,
                'srv_count': srv_count,
                'serror_rate': serror_rate,
                'rerror_rate': rerror_rate,
                'same_srv_rate': same_srv_rate,
                'dst_host_count': dst_host_count
            }
            
            # Validate input
            is_valid, errors = validate_input_data(input_data)
            if not is_valid:
                display_error_with_details("Invalid input data", "\n".join(errors))
                return None
            
            return input_data
    
    return None


def render_file_upload() -> Optional[pd.DataFrame]:
    """
    Render file upload interface for batch processing.
    
    Returns:
        Uploaded DataFrame or None
    """
    st.header("📁 File Upload for Batch Processing")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=ALLOWED_FILE_TYPES,
        help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB. The CSV should contain network traffic data with appropriate columns."
    )
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)")
                return None
            
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.subheader("📊 Data Preview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                stats = create_summary_stats(df)
                st.markdown("**Dataset Summary**")
                st.metric("Total Records", stats.get('total_records', 0))
                st.metric("Columns", len(df.columns))
                st.metric("Missing Values", stats.get('missing_values', 0))
            
            return df
            
        except Exception as e:
            display_error_with_details("Error processing file", str(e))
            return None
    
    return None


def render_prediction_result(result: Dict[str, Any]):
    """
    Render prediction result with enhanced visualization.
    
    Args:
        result: Prediction result dictionary
    """
    st.subheader("🎯 Prediction Result")
    
    predicted_class = result['predicted_class']
    confidence = result.get('confidence')
    is_anomaly = result['is_anomaly']
    
    # Main prediction display
    if is_anomaly:
        st.markdown(f"""
        <div class="prediction-anomaly">
            <h3>🚨 ANOMALY DETECTED</h3>
            <p><strong>Attack Type:</strong> {predicted_class}</p>
            {f'<p><strong>Confidence:</strong> {confidence:.1%}</p>' if confidence else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-normal">
            <h3>✅ NORMAL TRAFFIC</h3>
            <p><strong>Classification:</strong> {predicted_class}</p>
            {f'<p><strong>Confidence:</strong> {confidence:.1%}</p>' if confidence else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence visualization
    if result.get('all_probabilities'):
        st.subheader("📈 Classification Probabilities")
        
        probs = result['all_probabilities']
        prob_df = pd.DataFrame(list(probs.items()), columns=['Class', 'Probability'])
        
        fig = px.bar(
            prob_df, 
            x='Class', 
            y='Probability',
            title="Prediction Confidence by Class",
            color='Probability',
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_batch_results(results_df: pd.DataFrame):
    """
    Render batch prediction results with analytics.
    
    Args:
        results_df: DataFrame containing batch prediction results
    """
    st.subheader("📊 Batch Prediction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_predictions = len(results_df)
    anomaly_count = results_df['Is_Anomaly'].sum()
    normal_count = total_predictions - anomaly_count
    avg_confidence = results_df['Confidence'].mean() if 'Confidence' in results_df.columns else None
    
    with col1:
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        st.metric("Normal Traffic", normal_count)
    
    with col3:
        st.metric("Anomalies Detected", anomaly_count)
    
    with col4:
        if avg_confidence:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        class_counts = results_df['Predicted_Class'].value_counts()
        fig1 = px.pie(
            values=class_counts.values, 
            names=class_counts.index,
            title="Prediction Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Confidence distribution
        if 'Confidence' in results_df.columns:
            fig2 = px.histogram(
                results_df, 
                x='Confidence',
                title="Confidence Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_only_anomalies = st.checkbox("Show only anomalies")
    with col2:
        if 'Confidence' in results_df.columns:
            min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0)
            results_df = results_df[results_df['Confidence'] >= min_confidence]
    
    if show_only_anomalies:
        display_df = results_df[results_df['Is_Anomaly'] == True]
    else:
        display_df = results_df
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def render_feature_importance():
    """Render feature importance visualization."""
    importance_df = model_manager.get_feature_importance()
    
    if importance_df is not None:
        st.subheader("🎯 Feature Importance")
        
        # Top 20 features
        top_features = importance_df.head(20)
        
        fig = px.bar(
            top_features, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Top 20 Most Important Features"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View All Feature Importances"):
            st.dataframe(importance_df, use_container_width=True)


def main():
    """Main application function."""
    render_header()
    
    # Initialize the application
    if not initialize_app():
        st.stop()
    
    # Render model information in sidebar
    render_model_info()
    
    # Input method selection
    st.sidebar.header("🔧 Input Method")
    input_method = st.sidebar.radio(
        "Choose your input method:",
        ("Manual Input", "File Upload", "Sample Data"),
        help="Select how you want to provide data for analysis"
    )
    
    # Additional options
    st.sidebar.header("⚙️ Options")
    show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=False)
    
    # Main content area
    if input_method == "Manual Input":
        input_data = render_manual_input()
        
        if input_data:
            try:
                # Preprocess and predict
                processed_data = preprocess_input_data(input_data)
                result = model_manager.predict_single(processed_data)
                render_prediction_result(result)
                
            except Exception as e:
                display_error_with_details("Prediction failed", str(e))
    
    elif input_method == "File Upload":
        uploaded_df = render_file_upload()
        
        if uploaded_df is not None:
            if st.button("🔍 Analyze File", type="primary"):
                try:
                    with st.spinner("Processing file..."):
                        # Validate features
                        is_valid, missing_features = model_manager.validate_input_features(uploaded_df)
                        
                        if not is_valid:
                            st.warning(f"Missing features: {', '.join(missing_features)}")
                            st.info("Adding default values for missing features...")
                            
                            # Add missing features with default values
                            for feature in missing_features:
                                uploaded_df[feature] = 0
                        
                        # Make predictions
                        results_df = model_manager.predict_batch(uploaded_df)
                        render_batch_results(results_df)
                        
                except Exception as e:
                    display_error_with_details("Batch prediction failed", str(e))
    
    elif input_method == "Sample Data":
        st.header("📋 Sample Data Analysis")
        sample_df = load_sample_data()
        
        st.info("Using sample network traffic data for demonstration.")
        st.dataframe(sample_df, use_container_width=True)
        
        if st.button("🔍 Analyze Sample Data", type="primary"):
            try:
                results_df = model_manager.predict_batch(sample_df)
                render_batch_results(results_df)
            except Exception as e:
                display_error_with_details("Sample data analysis failed", str(e))
    
    # Feature importance section
    if show_feature_importance:
        render_feature_importance()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <p><strong>Network Anomaly Detection System</strong> - Powered by Random Forest Machine Learning</p>
        <p>This application detects network intrusions and classifies them into different attack categories using the NSL-KDD dataset.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()