import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from PIL import Image
from gtts import gTTS

# Import custom backend modules
from src.data_processor import optimize_memory, get_dataset_stats, get_column_recommendations
from src.ai_engine import generate_ml_readiness_report

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DataGuard AI | ML-Readiness Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Custom CSS Styling (Enterprise UI Upgrade) ---
st.markdown("""
    <style>
    /* 1. Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* 2. Apply Font to the entire app */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* 3. Hide Default Streamlit Branding (Menu & Footer) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 4. Custom Headers */
    .main-header {
        font-size: 2.6rem;
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* 5. Style the Metric Cards (KPIs) */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* 6. Style Primary Buttons (Like the Gemini Submit Button) */
    .stButton > button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1E40AF;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        color: white;
    }

    /* 7. Health Review Container */
    .health-container {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #CBD5E1;
        margin-bottom: 24px;
    }
    </style>
""", unsafe_allow_html=True)
# --- Sidebar Configuration (Branding & Sample Datasets Only) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=70)
    st.title("DataGuard AI")
    st.markdown("**Multimodal Data Collection & ML-Readiness Studio**")
    st.markdown("---")
    
    st.subheader("📚 Sample Datasets")
    st.markdown("Choose a pre-loaded sample dataset or upload your own on the main page:")
    
    sample_choice = st.selectbox(
        "Select Sample Dataset:",
        ["None (Upload Custom CSV)", "Telco Customer Churn (Sample)", "Housing Prices (Sample)"]
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the main page uploader to analyze custom enterprise CSV files.")

# --- Main Page Layout ---
st.markdown('<p class="main-header">🛡️ DataGuard AI Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Inspect, clean, and validate your dataset before machine learning model training.</p>', unsafe_allow_html=True)

# --- Main Page Data Ingestion Section ---
st.markdown("### 📁 Data Ingestion")
uploaded_file = st.file_uploader("Upload your raw CSV dataset", type=["csv"])

df = None

# Handle Sample Dataset Selection vs Custom Upload
if sample_choice == "Telco Customer Churn (Sample)" and not uploaded_file:
    try:
        # Load sample data if available or create a mock DataFrame fallback
        df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
        st.success("Loaded Telco Customer Churn Sample Dataset!")
    except Exception:
        st.warning("Could not load online sample. Please upload a CSV file directly.")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset successfully loaded from main page upload!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

# --- Main Dashboard Execution ---
if df is not None:
    df = optimize_memory(df)
    
    # Dataset Health Metrics
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    st.markdown("---")
    st.subheader("📊 Dataset Health Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{total_rows:,}", delta="Records")
    col2.metric("Total Features", f"{total_cols:,}", delta="Columns")
    col3.metric("Missing Values", f"{missing_cells:,}", delta=f"-{missing_cells} cells", delta_color="inverse")
    col4.metric("Duplicate Rows", f"{duplicate_rows:,}", delta=f"-{duplicate_rows} rows", delta_color="inverse")

    # Interactive Data Editor
    st.markdown("---")
    st.subheader("🔍 Preview & Clean Raw Dataset (Interactive Editor)")
    st.markdown("You can edit values directly in the grid below to fix inconsistencies before analysis.")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # Gemini AI ML-Readiness Engine Form
    st.markdown("---")
    st.subheader("🤖 Gemini AI ML-Readiness Engine")
    
    with st.form("gemini_form"):
        schema_image_file = st.file_uploader(
            "Capture Data Schema / Dictionary Image (Optional):", 
            type=["png", "jpg", "jpeg"],
            help="Upload an image of your database schema or data dictionary."
        )
        
        project_objective = st.text_area(
            "Describe your intended ML Project Objective:",
            placeholder="e.g., Predict customer churn using Telco data to identify high-risk accounts before cancellation..."
        )
        
        submitted = st.form_submit_button("🚀 Generate Tailored ML-Readiness Report")

    schema_image = None
    if schema_image_file is not None:
        try:
            schema_image = Image.open(schema_image_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    if submitted:
        if not project_objective.strip():
            st.warning("Please describe your intended ML project objective first.")
        else:
            with st.spinner("Analyzing dataset health and querying Gemini AI..."):
                df_stats = {
                    "total_rows": total_rows,
                    "total_cols": total_cols,
                    "missing_cells": missing_cells,
                    "duplicate_rows": duplicate_rows,
                    "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
                }
                
                report = generate_ml_readiness_report(df_stats, project_objective, schema_image)
                st.session_state["gemini_report"] = report

    # --- Display Report & Text-to-Speech ---
    if "gemini_report" in st.session_state:
        st.markdown("---\n### 📋 Tailored ML-Readiness Assessment Report")
        st.info(st.session_state["gemini_report"])

        def text_to_speech(text: str) -> BytesIO:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer

        if st.button("🔊 Read ML-Readiness Report Aloud"):
            with st.spinner("Synthesizing audio report..."):
                audio_output = text_to_speech(st.session_state["gemini_report"])
                st.audio(audio_output, format="audio/mp3")

        # --- Cleaned Data Download Option ---
        st.markdown("---")
        st.subheader("📥 Export Cleaned Dataset")
        csv_data = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV Dataset",
            data=csv_data,
            file_name="dataguard_cleaned_dataset.csv",
            mime="text/csv"
        )
else:
    st.info("📁 Please upload a CSV dataset or pick a sample dataset above to start your DataGuard AI analysis.")