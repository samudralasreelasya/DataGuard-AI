import os
import time
import google.generativeai as genai
from PIL import Image

def configure_gemini():
    """Configures the Gemini API using environment variable or st.secrets"""
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

# Configure Gemini on module load
configure_gemini()
model = genai.GenerativeModel('gemini-3.5-flash')   

def generate_ml_readiness_report(df_stats: dict, objective: str, schema_image = None) -> str:
    """
    Sends dataset health metrics, optional schema image, and project objective 
    to Google Gemini to generate a tailored ML-Readiness report.
    """
    prompt = f"""
    Analyze the following dataset health statistics for an ML project with objective: {objective}
    Stats: {df_stats}

    Provide your assessment structured clearly with:
    1. Readiness Status (e.g., Ready, Conditional Pass, or Action Required)
    2. Critical Data Quality Actions Needed
    3. Target Balance & Potential Bias Risks
    4. Recommended Next Steps Before Training
    """
    
    try:
        if schema_image is not None:
            response = model.generate_content([prompt, schema_image])
        else:
            response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        if "429" in str(e):
            return "⚠️ **Rate Limit Exceeded (HTTP 429):** You have temporarily exceeded your Gemini API free-tier quota limit. Please wait about 60 seconds before trying again, or switch your model to `gemini-3.5-flash`."
        return f"⚠️ Error communicating with Gemini API: {e}"