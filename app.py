import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from io import StringIO
import sys

# Set page config
st.set_page_config(
    page_title="InterBug - Sei Network Bug Detection",
    page_icon="🐞",
    layout="wide"
)

# Inject custom CSS for styling
custom_css = """
<style>
/* Overall background color (more grayish white) */
body {
    background-color: #f5f5f5;
}

/* Sidebar styles */
[data-testid="stSidebar"] > div:first-child {
    background-image: linear-gradient(135deg, #4e54c8 0%, #8f94fb 100%);
    color: white;
}

/* Sidebar text adjustments */
.sidebar-content {
    color: white;
}

/* Custom button style */
.stButton>button {
    background-color: #4e54c8;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.5em 1em;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #3e43a8;
}

/* Metric styling */
.css-1d391kg { 
    background-color: #ffffff;
    border: 2px solid #4e54c8;
    border-radius: 10px;
    padding: 10px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Set up logging to capture outputs
log_output = StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=log_output
)
logger = logging.getLogger(__name__)

# Function to check if required data files exist
def check_data_files():
    required_files = {
        'combined_datas.csv': 'Combined raw data',
        'official_sei_data3.csv': 'Cleaned data',
        'preprocessed_features.csv': 'Preprocessed features',
        'training_data.csv': 'Training data with bug detection'
    }
    
    existing_files = {}
    for file, description in required_files.items():
        if os.path.exists(file):
            existing_files[file] = {
                'exists': True,
                'description': description,
                'modified': os.path.getmtime(file),
                'size': os.path.getsize(file) / 1024  # Size in KB
            }
        else:
            existing_files[file] = {
                'exists': False,
                'description': description
            }
    
    return existing_files

# Function to fetch new data - only call this when requested
def fetch_new_data():
    try:
        # Import data fetching module dynamically
        from Sei_Module.data_collection import fetch_data
        
        logger.info("Starting to fetch new data from API...")
        old_stdout = sys.stdout
        capture = StringIO()
        sys.stdout = capture
        
        fetch_data.fetch_sei_data()  # Call the fetch_data function
        
        sys.stdout = old_stdout
        fetch_log = capture.getvalue()
        
        logger.info("Data fetching completed successfully")
        return True, fetch_log
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return False, str(e)

# Function to load model
@st.cache_resource
def load_model():
    try:
        with open("interbug_model.pkl", "rb") as f:
            model, imputer, selector = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return model, imputer, selector, feature_names
    except FileNotFoundError:
        return None, None, None, None

# Function to create feature importance plot
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        return fig
    return None

# Function to visualize anomaly detection thresholds
def plot_anomaly_thresholds(df, feature, threshold):
    fig = px.histogram(
        df, x=feature, 
        color=df[feature].abs() > threshold,
        marginal="box",
        labels={"color": "Anomaly"},
        color_discrete_map={True: "red", False: "blue"}
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig.add_vline(x=-threshold, line_dash="dash", line_color="red")
    return fig

# Sidebar with navigation and data management
st.sidebar.title("InterBug AI")
st.sidebar.markdown("Sei Network Cross-Chain Bug Detection")

existing_files = check_data_files()

st.sidebar.markdown("---")
with st.sidebar.expander("Data Management", expanded=False):
    for file, info in existing_files.items():
        if info['exists']:
            st.sidebar.success(f"✅ {info['description']} exists")
        else:
            st.sidebar.warning(f"❌ {info['description']} missing")

# Option to fetch new data
fetch_new = st.sidebar.checkbox("Fetch new data from API", value=False)
if fetch_new:
    if st.sidebar.button("Start Data Fetch"):
        with st.spinner("Fetching data from API..."):
            success, fetch_log = fetch_new_data()
            if success:
                st.sidebar.success("Data fetched successfully!")
                existing_files = check_data_files()
                # Check if the fetched file exists and show download button
                if existing_files.get('combined_datas.csv', {}).get('exists'):
                    with open('combined_datas.csv', 'rb') as f:
                        st.download_button(
                            label="Download Combined Data",
                            data=f,
                            file_name="combined_datas.csv",
                            mime="text/csv"
                        )
            else:
                st.sidebar.error(f"Error fetching data: {fetch_log}")


page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Data Upload & Processing", "Bug Detection", "Model Training", "Live Monitoring"]
)


# Dashboard page
if page == "Dashboard":
    st.title("InterBug AI Dashboard")
    st.markdown("""
    ## Sei Network Cross-Chain Bug Detection System
    
    This system monitors and analyzes transaction data from both EVM and Cosmos chains 
    to detect potential bugs and anomalies in the Sei Network.
    """)
    
    if existing_files['training_data.csv']['exists']:
        try:
            df = pd.read_csv('training_data.csv')
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                if 'bug_detected' in df.columns:
                    bug_count = df['bug_detected'].sum()
                    st.metric("Bugs Detected", bug_count)
            with col3:
                if 'bug_detected' in df.columns:
                    bug_rate = f"{(bug_count / len(df) * 100):.1f}%"
                    st.metric("Bug Rate", bug_rate)
            with col4:
                if 'anomaly_score' in df.columns:
                    avg_score = f"{df['anomaly_score'].mean():.2f}"
                    st.metric("Avg. Anomaly Score", avg_score)
            
            if 'bug_detected' in df.columns:
                st.subheader("Recent Anomalies")
                anomalies = df[df['bug_detected'] == 1].tail(5)
                st.dataframe(anomalies)
                st.subheader("Anomaly Score Distribution")
                fig = px.histogram(
                    df, x='anomaly_score',
                    color='bug_detected',
                    labels={"bug_detected": "Bug Detected"},
                    color_discrete_map={1: "red", 0: "blue"}
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
    else:
        st.info("No training data available yet. Start by uploading and processing data.")
        st.subheader("Sample Bug Detection Visualization")
        sample_data = pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'transaction_mismatch': np.random.normal(0, 1, 30),
            'fee_ratio': np.random.normal(0, 1, 30),
            'anomaly_score': np.random.gamma(2, 1, 30)
        })
        sample_data['bug_detected'] = (sample_data['anomaly_score'] > 3).astype(int)
        fig = px.line(
            sample_data, x='time', y=['transaction_mismatch', 'fee_ratio'],
            color_discrete_map={'transaction_mismatch': 'blue', 'fee_ratio': 'green'}
        )
        fig.add_scatter(
            x=sample_data[sample_data['bug_detected']==1]['time'],
            y=sample_data[sample_data['bug_detected']==1]['transaction_mismatch'],
            mode='markers',
            marker=dict(color='red', size=12, symbol='x'),
            name='Anomalies'
        )
        st.plotly_chart(fig, use_container_width=True)

# Data Upload & Processing Page
elif page == "Data Upload & Processing":
    st.title("Data Upload & Preprocessing")
    uploaded_file = st.file_uploader("Upload Sei Network transaction data (CSV)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.subheader("Column Information")
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            st.subheader("Data Cleaning & Preprocessing")
            st.markdown("Select data cleaning and preprocessing options:")
            clean_option = st.checkbox("Clean data", value=True)
            preprocess_option = st.checkbox("Preprocess data", value=True)
            if st.button("Process Data"):
                with st.spinner("Processing data..."):
                    df.to_csv('combined_data.csv', index=False)
                    if clean_option:
                        try:
                            from Sei_Module.data_cleaning import clean_data
                            clean_data.clean_and_save_data()
                            st.success("Data cleaning completed")
                            if os.path.exists('official_sei_data3.csv'):
                                cleaned_df = pd.read_csv('official_sei_data3.csv')
                                st.subheader("Cleaned Data Preview")
                                st.dataframe(cleaned_df.head())
                            else:
                                st.error("Cleaned data file not created")
                        except ImportError:
                            st.warning("Could not import clean_and_save_data. Running simplified cleaning.")
                            df.dropna(how='all', inplace=True)
                            df.to_csv('official_sei_data3.csv', index=False)
                            st.success("Basic data cleaning completed")
                    if preprocess_option:
                        try:
                            from Sei_Module.data_preprocess import preprocessing
                            cleaned_df = pd.read_csv('official_sei_data3.csv')
                            processed_df = preprocessing.preprocess_data(cleaned_df)
                            processed_df.to_csv('preprocessed_features.csv', index=False)
                            st.success("Data preprocessing completed")
                            st.subheader("Preprocessed Data Preview")
                            st.dataframe(processed_df.head())
                        except Exception as e:
                            st.error(f"Preprocessing failed: {str(e)}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    st.subheader("Or Use Existing Data")
    if st.button("Load Existing Data"):
        if os.path.exists('official_sei_data3.csv'):
            df = pd.read_csv('official_sei_data3.csv')
            st.success(f"Loaded existing data with {len(df)} rows and {len(df.columns)} columns")
            st.dataframe(df.head())
        else:
            st.error("No existing data found")

# Bug Detection Page
elif page == "Bug Detection":
    st.title("Bug Detection")
    if not os.path.exists('preprocessed_features.csv'):
        st.error("No preprocessed data found. Please upload and process data first.")
    else:
        df = pd.read_csv('preprocessed_features.csv')
        st.success(f"Loaded preprocessed data with {len(df)} rows and {len(df.columns)} features")
        st.subheader("Feature Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Feature Analysis")
        features = st.multiselect("Select features to analyze", df.columns.tolist(),
                                  default=df.columns.tolist()[:3])
        if features:
            st.subheader("Feature Correlation")
            corr = df[features].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Feature Distributions")
            feature_to_plot = st.selectbox("Select feature to plot distribution", features)
            fig = px.histogram(df, x=feature_to_plot, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Bug Detection")
        threshold_option = st.slider("Anomaly threshold percentile", 70, 95, 80)
        threshold_multiplier = st.slider("Anomaly Threshold Multiplier", 
                                         min_value=0.5, max_value=2.0, value=1.5, step=0.5)

        # Use session state to store bug detection results
        if "bug_detection_result" not in st.session_state:
            if st.button("Detect Bugs"):
                with st.spinner("Detecting bugs..."):
                    try:
                        from Sei_Module.testing import framework
                        result_df = framework.detect_bugs(df, threshold_multiplier=threshold_multiplier)
                        result_df.to_csv('training_data.csv', index=False)
                        st.session_state['bug_detection_result'] = result_df
                        st.success(f"Bug detection completed. Found {result_df['bug_detected'].sum()} bugs in {len(result_df)} records")
                    except Exception as e:
                        st.error(f"Bug detection failed: {str(e)}")
                        st.code(str(e))
        else:
            st.success("Bug detection already run. See results below.")

        # If results are stored, display them
        if "bug_detection_result" in st.session_state:
            result_df = st.session_state['bug_detection_result']
            st.subheader("Bug Detection Results")
            if st.checkbox("Show only detected bugs"):
                display_df = result_df[result_df['bug_detected'] == 1]
            else:
                display_df = result_df
            st.dataframe(display_df)

            st.subheader("Anomaly Score Distribution")
            fig = px.histogram(
                result_df, x='anomaly_score',
                color='bug_detected', 
                labels={"bug_detected": "Bug Detected"},
                color_discrete_map={1: "red", 0: "blue"}
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Feature Contribution to Bugs")
            feature_for_anomaly = st.selectbox(
                "Select feature to analyze contribution to bugs",
                [f for f in result_df.columns if f not in ['bug_detected', 'anomaly_score']]
            )
            fig = px.scatter(
                result_df, x=feature_for_anomaly, y='anomaly_score',
                color='bug_detected',
                labels={"bug_detected": "Bug Detected"},
                color_discrete_map={1: "red", 0: "blue"}
            )
            st.plotly_chart(fig, use_container_width=True)
            feature_threshold = result_df[feature_for_anomaly].abs().quantile(threshold_option/100)
            fig = plot_anomaly_thresholds(result_df, feature_for_anomaly, feature_threshold)
            st.plotly_chart(fig, use_container_width=True)


# Model Training Page
elif page == "Model Training":
    st.title("Model Training")
    if not os.path.exists('training_data.csv'):
        st.error("No training data found. Please run bug detection first.")
    else:
        training_df = pd.read_csv('training_data.csv')
        st.success(f"Loaded training data with {len(training_df)} records")
        if 'bug_detected' in training_df.columns:
            class_counts = training_df['bug_detected'].value_counts()
            st.subheader("Class Distribution")
            fig = px.pie(
                values=class_counts.values, 
                names=class_counts.index.map({0: "Normal", 1: "Bug"}),
                color=class_counts.index.map({0: "Normal", 1: "Bug"}),
                color_discrete_map={"Normal": "blue", "Bug": "red"}
            )
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Training Options")
        n_estimators = st.slider("Number of estimators", 50, 200, 100)
        max_depth = st.slider("Maximum depth", 3, 10, 5)
        feature_selection = st.checkbox("Use feature selection", value=True)
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    old_stdout = sys.stdout
                    sys.stdout = mystdout = StringIO()
                    from Sei_Module.model import train_model
                    train_model.train_model()
                    sys.stdout = old_stdout
                    
                    # Debug: Check if the model file was created
                    if os.path.exists("interbug_model.pkl"):
                        st.write("Model file created successfully!")
                    else:
                        st.write("Model file not found after training!")

                    st.text("Training Log:")
                    st.code(mystdout.getvalue())
                    model, imputer, selector, feature_names = load_model()
                    if model is not None:
                        st.success("Model training completed successfully")
                        st.subheader("Feature Importance")
                        fig = plot_feature_importance(model, feature_names['selected_features'])
                        if fig:
                            st.pyplot(fig)
                        if os.path.exists('feature_importance.png'):
                            st.subheader("Feature Importance Visualization")
                            st.image('feature_importance.png')
                        st.subheader("Training Logs")
                        st.text(mystdout.getvalue())
                        st.subheader("Model Details")
                        st.write(f"Model Type: {type(model).__name__}")
                        st.write(f"Number of Estimators: {model.n_estimators}")
                        st.write(f"Selected Features: {len(feature_names['selected_features'])}")
                        st.write("Selected Features:")
                        for i, feat in enumerate(feature_names['selected_features']):
                            st.write(f"{i+1}. {feat}")
                        else:
                            st.error("Model training did not produce a model file")
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")


# Live Monitoring Page
elif page == "Live Monitoring":
    st.title("Live Monitoring")
    model, imputer, selector, feature_names = load_model()
    if model is None:
        st.error("No trained model found. Please train the model first.")
    else:
        st.success("Model loaded successfully")
        st.subheader("Monitoring Options")
        threshold_multiplier = st.slider(
            "Anomaly Threshold Multiplier", 
            min_value=0.5, 
            max_value=2.0, 
            value=1.5, 
            step=0.5
        )
        uploaded_file = st.file_uploader("Upload new data for monitoring", type="csv")
        use_synthetic = st.checkbox("Use synthetic data for demo", value=False)
        if uploaded_file or use_synthetic:
            try:
                if uploaded_file:
                    new_data = pd.read_csv(uploaded_file)
                elif use_synthetic:
                    if os.path.exists('preprocessed_features.csv'):
                        template_df = pd.read_csv('preprocessed_features.csv')
                        features = template_df.columns.tolist()
                        np.random.seed(42)
                        n_samples = st.slider("Number of synthetic samples", 10, 100, 30)
                        normal_data = pd.DataFrame(
                            np.random.normal(0, 1, (n_samples - 3, len(features))),
                            columns=features
                        )
                        anomaly_data = pd.DataFrame(
                            np.random.normal(5, 2, (3, len(features))),
                            columns=features
                        )
                        new_data = pd.concat([normal_data, anomaly_data])
                    else:
                        st.error("No template data found for synthetic generation")
                        new_data = None
                if new_data is not None:
                    st.success(f"Data loaded with {len(new_data)} records")
                    st.subheader("Data Preview")
                    st.dataframe(new_data.head())
                    if st.button("Detect Anomalies"):
                        with st.spinner("Processing..."):
                            try:
                                all_features = feature_names['all_features']
                                missing_features = [f for f in all_features if f not in new_data.columns]
                                if missing_features:
                                    st.warning(f"Missing features: {missing_features}")
                                    for feat in missing_features:
                                        new_data[feat] = 0
                                X = new_data[all_features].copy()
                                X_imputed = imputer.transform(X)
                                X_selected = selector.transform(X_imputed)
                                predictions = model.predict(X_selected)
                                probabilities = model.predict_proba(X_selected)[:, 1]
                                threshold = np.mean(probabilities) * threshold_multiplier
                                bug_detected = (probabilities > threshold).astype(int)
                                new_data['anomaly_probability'] = probabilities
                                new_data['bug_detected'] = bug_detected
                                st.subheader("Detection Results")
                                st.write(f"Detected {bug_detected.sum()} anomalies in {len(new_data)} records")
                                if st.checkbox("Show only detected anomalies"):
                                    display_df = new_data[new_data['bug_detected'] == 1]
                                else:
                                    display_df = new_data
                                st.dataframe(display_df)
                                st.subheader("Anomaly Probability Distribution")
                                fig = px.histogram(
                                    new_data, x='anomaly_probability',
                                    color='bug_detected',
                                    color_discrete_map={1: "red", 0: "blue"}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.subheader("Anomaly Timeline")
                                if 'timestamp' not in new_data.columns:
                                    new_data['timestamp'] = pd.date_range(
                                        start='2023-01-01', 
                                        periods=len(new_data), 
                                        freq='H'
                                    )
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=new_data['timestamp'],
                                    y=new_data['anomaly_probability'],
                                    mode='lines+markers',
                                    name='Anomaly Probability',
                                    marker=dict(
                                        color=new_data['anomaly_probability'],
                                        colorscale='Viridis',
                                        size=8
                                    )
                                ))
                                bug_data = new_data[new_data['bug_detected'] == 1]
                                if len(bug_data) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=bug_data['timestamp'],
                                        y=bug_data['anomaly_probability'],
                                        mode='markers',
                                        name='Detected Bugs',
                                        marker=dict(
                                            color='red',
                                            size=12,
                                            symbol='x'
                                        )
                                    ))
                                fig.add_shape(
                                    type="line",
                                    x0=new_data['timestamp'].min(),
                                    y0=threshold,
                                    x1=new_data['timestamp'].max(),
                                    y1=threshold,
                                    line=dict(
                                        color="red",
                                        width=2,
                                        dash="dash"
                                    )
                                )
                                fig.update_layout(
                                    title="Anomaly Detection Timeline",
                                    xaxis_title="Time",
                                    yaxis_title="Anomaly Probability",
                                    legend_title="Legend"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error during anomaly detection: {str(e)}")
            except Exception as e:
                st.error(f"Error processing new data: {str(e)}")

# Footer: Display logs in sidebar if desired
st.sidebar.markdown("---")
st.sidebar.markdown("### Logs")
if st.sidebar.checkbox("Show logs"):
    st.sidebar.text_area("Log Output", log_output.getvalue(), height=300)
st.sidebar.markdown("---")
st.sidebar.info("Interbug AI - Sei Network Cross-Chain Bug Detection System")
