import streamlit as st
import pandas as pd
from io import BytesIO
from gtts import gTTS
from PIL import Image

# Import backend modules
from src.ai_engine import generate_ml_readiness_report, configure_gemini

# Page Configuration
st.set_page_config(
    page_title="DataGuard AI Studio",
    page_icon="🛡️",
    layout="wide"
)

# Initialize API configuration check
configure_gemini()

st.title("🛡️ DataGuard AI: Multimodal Data Collection & ML Readiness Studio")
st.markdown("Inspect your raw datasets, analyze data quality, and generate tailored Gemini AI readiness reports.")

# --- Sidebar / Data Ingestion ---
st.sidebar.header("📁 Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload your raw CSV dataset", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(uploaded_file)
    
    st.success("Dataset successfully loaded!")
    
    # --- Dataset Health Overview ---
    st.subheader("📊 Dataset Health Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("Total Features", f"{total_cols}")
    col3.metric("Missing Values", f"{missing_cells}")
    col4.metric("Duplicate Rows", f"{duplicate_rows}")

    # --- Interactive Data Editor ---
    with st.expander("🔍 Preview & Clean Raw Dataset (Interactive Editor)", expanded=False):
        st.markdown("You can edit values directly in the grid below to fix inconsistencies before analysis.")
        df = st.data_editor(df, key="editable_df")

    # --- Gemini AI ML-Readiness Engine ---
    st.markdown("---")
    st.subheader("🤖 Gemini AI ML-Readiness Engine")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        schema_file = st.file_uploader("Capture / Upload Data Schema / Dictionary Image", type=["png", "jpg", "jpeg"])
        schema_image = Image.open(schema_file) if schema_file else None
        if schema_image:
            st.image(schema_image, caption="Uploaded Schema / Data Dictionary", width=250)

    with col_input2:
        project_objective = st.text_area(
            "Describe your intended ML Project Objective:",
            placeholder="e.g., Predict customer churn using Telco data..."
        )

    if st.button("🚀 Run AI ML-Readiness Analysis", type="primary"):
        if not project_objective.strip():
            st.warning("Please describe your intended ML project objective first.")
        else:
            with st.spinner("Analyzing dataset health and querying Gemini AI..."):
                df_stats = {
                    "total_rows": total_rows,
                    "total_cols": total_cols,
                    "missing_cells": missing_cells,
                    "duplicate_rows": duplicate_rows,
                    "dtypes": df.dtypes.to_dict()
                }
                
                report = generate_ml_readiness_report(df_stats, project_objective, schema_image)
                st.session_state["gemini_report"] = report

    # --- Display Report & Text-to-Speech ---
    if "gemini_report" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Tailored ML-Readiness Assessment Report")
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
else:
    st.info("👈 Please upload a CSV dataset from the sidebar to start your DataGuard AI analysis.")