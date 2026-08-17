import os
import google.generativeai as genai
from PIL import Image

def configure_gemini():
    """Configures the Gemini API using environment variables or Streamlit secrets safely."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
            
    if api_key:
        genai.configure(api_key=api_key)
    return api_key

def generate_ml_readiness_report(df_stats: dict, objective: str, schema_image: Image.Image = None) -> str:
    """
    Sends dataset health metrics, optional schema image, and project objective 
    to Google Gemini to generate a tailored ML-readiness assessment.
    """
    api_key = configure_gemini()
    if not api_key:
        return "⚠️ Error: Gemini API key not found. Please configure GEMINI_API_KEY in your environment variables or Streamlit secrets."

    try:
        # Use gemini-1.5-flash for fast, multimodal analysis
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        You are an expert Data Engineer and Machine Learning Architect. 
        Analyze the following dataset health report against the user's intended ML project objective.

        Project Objective:
        "{objective}"

        Dataset Health Statistics:
        - Total Rows: {df_stats.get('total_rows', 'N/A')}
        - Total Columns: {df_stats.get('total_cols', 'N/A')}
        - Missing Cells: {df_stats.get('missing_cells', 'N/A')}
        - Duplicate Rows: {df_stats.get('duplicate_rows', 'N/A')}
        - Column Names & Types: {df_stats.get('dtypes', 'N/A')}

        Please provide a professional, structured ML-Readiness Assessment Report including:
        1. Readiness Status (e.g., Ready, Conditional Pass, or Requires Significant Cleaning)
        2. Critical Data Quality Actions Required before training an ML model
        3. Potential Target Class Balance or Data Leakage Risks
        4. Recommended Next Steps
        """

        contents = [prompt]
        if schema_image:
            contents.append(schema_image)

        response = model.generate_content(contents)
        return response.text

    except Exception as e:
        return f"⚠️ Error communicating with Gemini API: {str(e)}"