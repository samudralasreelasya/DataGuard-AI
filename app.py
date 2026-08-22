import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from PIL import Image
from gtts import gTTS
import datetime

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DataGuard AI | ML-Readiness Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"  # Ensures the sidebar stays open normally
)

# Dummy backend functions for standalone execution
def optimize_memory(df):
    return df

def get_dataset_stats(df):
    return {
        "total_rows": df.shape[0],
        "total_cols": df.shape[1],
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

def generate_ml_readiness_report(df_stats, objective, image=None):
    return f"**ML Readiness Report for Objective:** '{objective}'\n\n" \
           f"- **Data Completeness:** {((1 - df_stats['missing_cells']/(df_stats['total_rows']*df_stats['total_cols']))*100):.1f}%\n" \
           f"- **Duplicate Check:** Found {df_stats['duplicate_rows']} duplicates.\n" \
           f"- **Recommendation:** Dataset structure appears viable for initial baseline model training."

# --- Initialize Session States ---
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user_email"] = ""

if "dataset_history" not in st.session_state:
    st.session_state["dataset_history"] = {}

# Dynamic Theme Colors Definition
if st.session_state["theme"] == "dark":
    bg_app = "#0F172A"
    bg_card = "#1E293B"
    text_color = "#F8FAFC"
    sub_text = "#94A3B8"
    border_color = "#334155"
    input_bg = "#0F172A"
    input_text = "#FFFFFF"
    input_border = "#475569"
    accent_color = "#3B82F6"
    accent_hover = "#2563EB"
    metric_bg = "#182232"
else:
    bg_app = "#F8FAFC"
    bg_card = "#FFFFFF"
    text_color = "#0F172A"
    sub_text = "#475569"
    border_color = "#E2E8F0"
    input_bg = "#FFFFFF"
    input_text = "#0F172A"
    input_border = "#CBD5E1"
    accent_color = "#2563EB"
    accent_hover = "#1D4ED8"
    metric_bg = "#F1F5F9"

# Apply Custom Responsive CSS
st.markdown("""
    <style>
    /* 1. Import Material Symbols Outlined Font */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* 2. Target the Sidebar Collapse and Header Control Buttons */
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="collapsedControl"] span,
    [data-testid="stHeader"] span {
        font-family: 'Material Symbols Outlined' !important;
        font-weight: normal !important;
        font-style: normal !important;
        display: inline-block !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        word-wrap: normal !important;
        white-space: nowrap !important;
        direction: ltr !important;
        color: #3B82F6 !important; /* Button icon color */
    }

    /* 3. Give the icon button a clean rectangular pill background */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 6px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        background-color: #334155 !important;
        border-color: #3B82F6 !important;
    }
    </style>
""", unsafe_allow_html=True)
# Helper function for Text-to-Speech Voice Audit
def text_to_speech(text: str) -> BytesIO:
    tts = gTTS(text=text, lang='en', slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# --- 1. SIGN-UP & LOGIN SECTION ---
if not st.session_state["logged_in"]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.write("")
        st.write("")
        st.markdown(f"<h1 style='text-align: center; color: {accent_color};'>🛡️ DataGuard AI</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: {sub_text};'>Enterprise Multimodal ML-Readiness Studio</p>", unsafe_allow_html=True)
        
        # Theme Toggle Bar inside login header
        t_col1, t_col2 = st.columns([3, 1])
        with t_col2:
            theme_btn = st.button("🌙 Dark" if st.session_state["theme"] == "light" else "☀️ Light", key="login_theme_toggle")
            if theme_btn:
                st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
                st.rerun()

        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        
        # Tab switch between Login and Sign Up
        auth_tab1, auth_tab2 = st.tabs(["🔒 Sign In", "📝 Sign Up"])
        
        with auth_tab1:
            with st.form("login_form"):
                email = st.text_input("Work Email", placeholder="evaluator@mirai.edu", key="login_email")
                password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
                submit_login = st.form_submit_button("Sign In to Studio", use_container_width=True)
                
                if submit_login:
                    if "@" in email and password:
                        st.session_state["logged_in"] = True
                        st.session_state["user_email"] = email
                        st.success(f"Welcome back, {email}!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid email and password.")

        with auth_tab2:
            with st.form("signup_form"):
                new_name = st.text_input("Full Name", placeholder="John Doe", key="signup_name")
                new_email = st.text_input("Work Email", placeholder="evaluator@mirai.edu", key="signup_email")
                new_password = st.text_input("Create Password", type="password", placeholder="••••••••", key="signup_password")
                submit_signup = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit_signup:
                    if "@" in new_email and new_password:
                        st.session_state["logged_in"] = True
                        st.session_state["user_email"] = new_email
                        st.success("Account successfully created!")
                        st.rerun()
                    else:
                        st.error("Please fill out all fields correctly.")
                        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # ==========================================
    # 🚀 MAIN DASHBOARD
    # ==========================================

    # --- SIDEBAR: HISTORY & SETTINGS ---
    with st.sidebar:
        st.markdown("### 🛡️ DataGuard AI")
        st.caption(f"👤 {st.session_state['user_email']}")
        
        mode_col, logout_col = st.columns(2)
        with mode_col:
            if st.button("🌙 Dark" if st.session_state["theme"] == "light" else "☀️ Light", key="main_theme_toggle"):
                st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
                st.rerun()
        with logout_col:
            if st.button("Logout", type="primary"):
                st.session_state["logged_in"] = False
                st.session_state["user_email"] = ""
                st.rerun()

        st.markdown("---")
        st.markdown("### 📜 Dataset Audit History")
        
        if not st.session_state["dataset_history"]:
            st.info("No processing history recorded yet.")
        else:
            for ds_name, history in st.session_state["dataset_history"].items():
                with st.expander(f"📄 {ds_name}", expanded=False):
                    st.markdown(f"**Uploaded:** `{history['upload_time']}`")
                    st.markdown(f"**Model:** `{history['model_chosen'] or 'N/A'}`")
                    
                    st.markdown("**Prompts Submitted:**")
                    if history["prompts"]:
                        for idx, p in enumerate(history["prompts"], 1):
                            st.caption(f"{idx}. {p}")
                    else:
                        st.caption("No prompts logged.")

                    st.markdown("**AI & Voice Audits Run:**")
                    if history["audits"]:
                        for audit_idx, audit in enumerate(history["audits"], 1):
                            st.markdown(f"**Audit #{audit_idx}** (`{audit['timestamp']}`)")
                            st.caption(f"Engine: {audit['model']}")
                            st.text_area("Audit Log Output:", audit['report'], height=90, key=f"hist_{ds_name}_{audit_idx}")
                    else:
                        st.caption("No audit outputs logged.")

    # --- MAIN PAGE HEADER ---
    st.markdown(f"<h1 style='color: {accent_color}; font-weight: 700; margin-bottom: 0px;'>🛡️ DataGuard AI Studio</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {sub_text}; font-size: 1.05rem; margin-bottom: 24px;'>Multimodal Machine Learning Readiness and Data Quality Engine</p>", unsafe_allow_html=True)

    # --- DATA INGESTION & SEARCH BAR ---
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Data Ingestion & Search")
    
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload Raw CSV Dataset", type=["csv"], key="main_csv_uploader")
    with col_up2:
        sample_choice = st.selectbox(
            "Or Choose Sample Dataset:",
            ["None", "Telco Customer Churn Sample"]
        )
    st.markdown('</div>', unsafe_allow_html=True)

    df = None
    dataset_name = ""

    if sample_choice == "Telco Customer Churn Sample" and uploaded_file is None:
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
            dataset_name = "Telco-Customer-Churn.csv"
            st.success("Loaded Telco Customer Churn Sample Dataset!")
        except Exception:
            st.warning("Could not load online sample dataset.")
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")

    # Register Dataset to History State
    if df is not None and dataset_name:
        if dataset_name not in st.session_state["dataset_history"]:
            st.session_state["dataset_history"][dataset_name] = {
                "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stats": get_dataset_stats(df),
                "model_chosen": None,
                "prompts": [],
                "audits": []
            }

        # --- MULTI-TAB WORKSPACE ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Health & Analytics", 
            "🔍 Data Cleaning & Search", 
            "🤖 Gemini AI Audit", 
            "📥 Export & Options"
        ])

        # --- TAB 1: HEALTH & ANALYTICS ---
        with tab1:
            stats = get_dataset_stats(df)
            st.subheader("📊 Dataset Health Overview")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-box"><div class="metric-title">Total Rows</div><div class="metric-value">{stats["total_rows"]:,}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-box"><div class="metric-title">Total Features</div><div class="metric-value">{stats["total_cols"]}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-box"><div class="metric-title">Missing Cells</div><div class="metric-value">{stats["missing_cells"]:,}</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-box"><div class="metric-title">Duplicate Rows</div><div class="metric-value">{stats["duplicate_rows"]:,}</div></div>', unsafe_allow_html=True)

            st.write("")
            st.subheader("📈 Missing Values Distribution")
            missing_counts = df.isnull().sum().reset_index()
            missing_counts.columns = ['variable', 'missing_count']
            
            if missing_counts['missing_count'].sum() > 0:
                fig_miss = px.bar(
                    missing_counts[missing_counts['missing_count'] > 0],
                    x='variable',
                    y='missing_count',
                    title="Missing Values per Column",
                    template="plotly_dark" if st.session_state["theme"] == "dark" else "plotly_white"
                )
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.info("🎉 Zero missing values detected in dataset!")

        # --- TAB 2: DATA CLEANING & SEARCH ---
        with tab2:
            st.subheader("🔍 Interactive Data Editor & Search")
            
            search_query = st.text_input("🔎 Search dataset by keyword across all fields:", "", key="table_search_input")
            
            filtered_df = df
            if search_query:
                mask = np.column_stack([df[col].astype(str).str.contains(search_query, case=False, na=False) for col in df.columns])
                filtered_df = df.loc[mask.any(axis=1)]
                st.caption(f"Showing {len(filtered_df)} row(s) matching search term: '{search_query}'")

            edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True, key="main_data_editor")

        # --- TAB 3: GEMINI AI AUDIT ---# --- TAB 3: GEMINI AI AUDIT ---
        with tab3:
            st.subheader("🤖 Gemini AI ML-Readiness Engine")
            
            # 1. Initialize session state for prompt tracking
            if "audit_prompt_text" not in st.session_state:
                st.session_state["audit_prompt_text"] = ""

            # 2. Beginner Prompts Helper (MVP Feature)
            st.markdown("##### 💡 Example Audit Prompts for New Users:")
            ex_col1, ex_col2 = st.columns(2)
            
            p1_clicked = ex_col1.button("📋 Prompt 1: Customer Churn ML Readiness", use_container_width=True)
            p2_clicked = ex_col2.button("⚠️ Prompt 2: Missing Data & Imbalance Audit", use_container_width=True)

            # Update session state dynamically when an example prompt is clicked
            if p1_clicked:
                st.session_state["audit_prompt_text"] = "Audit this dataset to evaluate its readiness for predicting customer churn. Check for target variable imbalances and missing value risks."
                st.rerun()
            elif p2_clicked:
                st.session_state["audit_prompt_text"] = "Identify severe data quality bottlenecks, highly correlated features, missing value strategies, and potential data leaks for model training."
                st.rerun()

            # 3. Audit Input Form
            with st.form("gemini_audit_form"):
                aud_col1, aud_col2 = st.columns([1, 2])
                with aud_col1:
                    selected_model = st.selectbox(
                        "AI Model Engine",
                        ["Gemini 1.5 Pro", "Gemini 1.5 Flash", "Gemini Ultra"],
                        key="model_selector"
                    )
                    schema_image_file = st.file_uploader(
                        "Upload Schema / Data Dictionary Image (Optional)",
                        type=["png", "jpg", "jpeg"]
                    )
                
                with aud_col2:
                    # Explicitly bind the value parameter to st.session_state
                    prompt_input = st.text_area(
                        "Describe your ML Project Objective or Audit Prompt:",
                        value=st.session_state["audit_prompt_text"],
                        placeholder="e.g., Evaluate dataset suitability for training a supervised classification model...",
                        height=140,
                        key="audit_prompt_area"
                    )
                
                submit_audit = st.form_submit_button("🚀 Run AI ML-Readiness Audit", use_container_width=True)

            # 4. Handle Execution & Output on Main Page
            if submit_audit:
                # Save manual inputs back into session state
                st.session_state["audit_prompt_text"] = prompt_input
                
                if not st.session_state["audit_prompt_text"].strip():
                    st.warning("Please provide a prompt or select an example above.")
                else:
                    with st.spinner("Analyzing dataset health and executing Gemini audit..."):
                        schema_img = Image.open(schema_image_file) if schema_image_file else None
                        
                        df_stats = {
                            "total_rows": int(df.shape[0]),
                            "total_cols": int(df.shape[1]),
                            "missing_cells": int(df.isnull().sum().sum()),
                            "duplicate_rows": int(df.duplicated().sum())
                        }
                        
                        # Generate the report text
                        report = generate_ml_readiness_report(df_stats, st.session_state["audit_prompt_text"], schema_img)
                        
                        # Log to Sidebar History
                        st.session_state["dataset_history"][dataset_name]["model_chosen"] = selected_model
                        st.session_state["dataset_history"][dataset_name]["prompts"].append(st.session_state["audit_prompt_text"])
                        st.session_state["dataset_history"][dataset_name]["audits"].append({
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "prompt": st.session_state["audit_prompt_text"],
                            "report": report,
                            "model": selected_model
                        })
                        
                        # Store report for main page rendering
                        st.session_state["latest_report"] = report

            # 5. Display Text & Voice Output Directly on Main Page
            if "latest_report" in st.session_state:
                st.markdown("---")
                st.subheader("📋 Tailored ML-Readiness Report Output")
                st.info(st.session_state["latest_report"])

                st.markdown("##### 🎙️ Voice Audit Synthesis")
                if st.button("🔊 Generate & Listen to Voice Audit", key="main_voice_btn"):
                    with st.spinner("Synthesizing audio voice report..."):
                        audio_data = text_to_speech(st.session_state["latest_report"])
                        st.audio(audio_data, format="audio/mp3")


        # --- TAB 4: EXPORT & OPTIONS ---
        with tab4:
            st.subheader("📥 Export Dataset")
            st.write("Download your processed dataset ready for model baseline training.")
            
            export_df = edited_df if 'edited_df' in locals() else df
            csv_bytes = export_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Cleaned CSV Dataset",
                data=csv_bytes,
                file_name="dataguard_cleaned_dataset.csv",
                mime="text/csv",
                type="primary"
            )
    else:
        st.info("👈 Upload a CSV file or choose a sample dataset above to start auditing.")