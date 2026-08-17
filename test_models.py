import os
import google.generativeai as genai

# Ensure your API key is picked up
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # Fallback if using Streamlit secrets locally
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

genai.configure(api_key=api_key)

print("--- Available Models ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)