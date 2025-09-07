import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration ---
MODEL_ARTIFACTS_PATH = "model_artifacts.joblib"
# These are the columns the model was trained on, in order.
# We need this to create a DataFrame from user input.
EXPECTED_COLUMNS = [
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
CLASS_LABELS = ['Normal', 'DoS', 'Probe', 'Privilege Escalation', 'Access']

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """Load the trained model pipeline from disk."""
    try:
        model = joblib.load(MODEL_ARTIFACTS_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{MODEL_ARTIFACTS_PATH}'. Please run the training script first.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

def predict(model, input_df: pd.DataFrame) -> np.ndarray:
    """Make predictions using the loaded model."""
    try:
        predictions = model.predict(input_df)
        return predictions
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return np.array([])

# --- Streamlit App UI ---
st.set_page_config(page_title="Network Anomaly Detector", layout="wide")

st.title("🌐 Network Anomaly Detector")
st.markdown("""
This application uses a trained Random Forest model to predict whether network traffic is **normal** or an **attack**.
You can either upload a CSV file with network data or enter the parameters manually to get a prediction.
""")

# Load the model
model = load_model()

if model:
    # --- Input Method Selection ---
    input_method = st.radio(
        "Choose your input method:",
        ("Upload a CSV file", "Enter data manually"),
        horizontal=True
    )

    st.sidebar.header("Manual Input")
    
    if input_method == "Enter data manually":
        st.header("Manual Data Input")
        st.info("Enter the network traffic parameters below. Hover over the (?) for more info.")

        # Create columns for layout
        col1, col2, col3 = st.columns(3)

        with col1:
            duration = st.number_input("Duration", min_value=0, help="Length (seconds) of the connection.")
            protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"], help="Protocol used for the connection.")
            service = st.text_input("Service", "http", help="Destination network service used (e.g., http, ftp).")
            flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "SH"], help="Normal or error status of the connection.")

        with col2:
            src_bytes = st.number_input("Source Bytes", min_value=0)
            dst_bytes = st.number_input("Destination Bytes", min_value=0)
            count = st.number_input("Count", min_value=0, help="Number of connections to the same host in the last two seconds.")
            serror_rate = st.slider("Serror Rate", 0.0, 1.0, 0.0, help="Percentage of connections that have 'S0' errors.")

        with col3:
            dst_host_count = st.number_input("Dst Host Count", min_value=0)
            dst_host_srv_count = st.number_input("Dst Host Srv Count", min_value=0)
            dst_host_same_srv_rate = st.slider("Dst Host Same Srv Rate", 0.0, 1.0, 0.0)
            dst_host_diff_srv_rate = st.slider("Dst Host Diff Srv Rate", 0.0, 1.0, 0.0)

        if st.button("Detect Anomaly", type="primary"):
            # Create a dictionary with all expected columns, filling non-inputted ones with 0
            input_data = {col: 0 for col in EXPECTED_COLUMNS}
            # Update with user inputs
            input_data.update({
                'duration': duration, 'protocol_type': protocol_type, 'service': service, 'flag': flag,
                'src_bytes': src_bytes, 'dst_bytes': dst_bytes, 'count': count, 'serror_rate': serror_rate,
                'dst_host_count': dst_host_count, 'dst_host_srv_count': dst_host_srv_count,
                'dst_host_same_srv_rate': dst_host_same_srv_rate, 'dst_host_diff_srv_rate': dst_host_diff_srv_rate
            })
            
            input_df = pd.DataFrame([input_data])
            
            prediction = predict(model, input_df)
            if prediction.size > 0:
                predicted_class = CLASS_LABELS[prediction[0]]
                st.subheader("Prediction Result")
                if predicted_class == 'Normal':
                    st.success(f"✅ The network traffic is predicted to be **{predicted_class}**.")
                else:
                    st.error(f"🚨 The network traffic is predicted to be an **{predicted_class}** attack.")

    elif input_method == "Upload a CSV file":
        st.header("CSV File Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="The CSV should have the same columns as the NSL-KDD dataset."
        )
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file, names=EXPECTED_COLUMNS, header=None)
                st.write("Uploaded Data Preview:")
                st.dataframe(input_df.head())

                if st.button("Detect Anomalies in File", type="primary"):
                    predictions = predict(model, input_df)
                    if predictions.size > 0:
                        result_df = input_df.copy()
                        result_df['Prediction'] = [CLASS_LABELS[p] for p in predictions]
                        
                        st.subheader("Prediction Results")
                        st.dataframe(result_df)
                        
                        st.download_button(
                            label="Download Results as CSV",
                            data=result_df.to_csv(index=False).encode('utf-8'),
                            file_name='predictions.csv',
                            mime='text/csv',
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}. Please ensure it's a valid CSV with the correct columns.")
else:
    st.warning("Model could not be loaded. Please ensure `model_artifacts.joblib` is in the same directory.")
