import os
import sys
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from audiorecorder import audiorecorder

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_processor import get_dataset_stats, search_dataset          # noqa: E402
from ai_engine import generate_ml_readiness_report, configure_gemini  # noqa: E402
from audio_utils import text_to_speech, speech_to_text                # noqa: E402

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DataGuard AI | ML-Readiness Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_gemini()

# --- Initialize Session State ---
defaults = {
    "auth_mode": "login",
    "logged_in": False,
    "user_email": "",
    "user_name": "",
    "dataset_history": {},
    "current_df": None,
    "current_dataset_name": "",
    "audit_prompt_text": "",
    "latest_report": "",
    "edited_df": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================================
# THEME TOKENS
# ==========================================================
T = {
    "bg_app": "#0F172A",
    "bg_card": "#1E293B",
    "text_color": "#F8FAFC",
    "sub_text": "#94A3B8",
    "border_color": "#334155",
    "input_bg": "#0F172A",
    "input_text": "#FFFFFF",
    "input_border": "#475569",
    "accent_color": "#3B82F6",
    "accent_hover": "#2563EB",
    "metric_bg": "#182232",
    "success_color": "#22C55E",
    "danger_color": "#F87171",
}
PLOTLY_TEMPLATE = "plotly_dark"


def inject_css(t: dict) -> None:
    """Wires the theme tokens to every part of the app -- background, sidebar,
    cards, inputs, buttons, tabs -- instead of only the sidebar icon."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        /* --- App shell --- */
        .stApp {{
            background-color: {t['bg_app']};
            color: {t['text_color']};
        }}
        [data-testid="stHeader"] {{
            background-color: transparent;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {t['bg_card']};
            border-right: 1px solid {t['border_color']};
        }}
        section[data-testid="stSidebar"] * {{
            color: {t['text_color']};
        }}

        /* --- Sidebar collapse / header control icons --- */
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="collapsedControl"] span,
        [data-testid="stHeader"] span {{
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
            color: {t['accent_color']} !important;
        }}
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button {{
            background-color: {t['bg_card']} !important;
            border: 1px solid {t['border_color']} !important;
            border-radius: 8px !important;
            padding: 6px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        }}
        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="collapsedControl"] button:hover {{
            background-color: {t['border_color']} !important;
            border-color: {t['accent_color']} !important;
        }}

        /* --- Cards --- */
        .main-card {{
            background: {t['bg_card']};
            border: 1px solid {t['border_color']};
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 20px;
        }}
        .side-card {{
            background: {t['metric_bg']};
            border: 1px solid {t['border_color']};
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }}

        /* --- Metric tiles --- */
        .metric-box {{
            background: {t['metric_bg']};
            border: 1px solid {t['border_color']};
            border-radius: 12px;
            padding: 16px 18px;
            text-align: left;
        }}
        .metric-title {{
            color: {t['sub_text']};
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }}
        .metric-value {{
            color: {t['text_color']};
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.1;
        }}

        /* --- Inputs --- */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {{
            background-color: {t['input_bg']} !important;
            color: {t['input_text']} !important;
            border: 1px solid {t['input_border']} !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: {t['input_bg']} !important;
            border-color: {t['input_border']} !important;
            border-radius: 8px !important;
            color: {t['input_text']} !important;
        }}

        /* --- Buttons --- */
        .stButton > button {{
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: 1px solid {t['border_color']} !important;
        }}
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
            background-color: {t['accent_color']} !important;
            border: 1px solid {t['accent_color']} !important;
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
            background-color: {t['accent_hover']} !important;
            border-color: {t['accent_hover']} !important;
        }}
        .stButton > button[kind="secondary"], .stFormSubmitButton > button[kind="secondary"] {{
            background-color: transparent !important;
            color: {t['text_color']} !important;
        }}

        /* --- Tabs --- */
        button[data-baseweb="tab"] {{
            color: {t['sub_text']} !important;
            font-weight: 600 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {t['accent_color']} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {t['accent_color']} !important;
        }}

        /* --- Auth link-style toggle button --- */
        .auth-toggle button {{
            background: none !important;
            border: none !important;
            color: {t['accent_color']} !important;
            text-decoration: underline;
            padding: 0 !important;
        }}

        /* --- Misc text --- */
        .subtle {{ color: {t['sub_text']}; }}
        hr {{ border-color: {t['border_color']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(T)


def metric_tile(col, label: str, value: str) -> None:
    col.markdown(
        f'<div class="metric-box"><div class="metric-title">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )





# ==========================================================
# 1. AUTH SCREENS (Login / Sign Up)
# ==========================================================
if not st.session_state["logged_in"]:
    c1, c2, c3 = st.columns([1, 1.3, 1])
    with c2:
        st.write("")
        st.write("")
        st.markdown('<div class="main-card">', unsafe_allow_html=True)

        st.markdown(
            f"<h1 style='text-align:center; color:{T['accent_color']}; margin-bottom:0;'>🛡️ DataGuard AI</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center; color:{T['sub_text']}; margin-top:4px;'>"
            "Multimodal ML-Readiness Studio</p>",
            unsafe_allow_html=True,
        )
        st.write("")

        if st.session_state["auth_mode"] == "login":
            with st.form("login_form"):
                st.markdown("##### Sign in")
                email = st.text_input("Username / Email", placeholder="evaluator@mirai.edu")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button(
                    "Sign In to Studio", use_container_width=True, type="primary"
                )
            if submit_login:
                if "@" in email and password:
                    st.session_state["logged_in"] = True
                    st.session_state["user_email"] = email
                    st.rerun()
                else:
                    st.error("Please enter a valid email and password.")

            st.markdown('<div class="auth-toggle" style="text-align:center;">', unsafe_allow_html=True)
            if st.button("No account? Sign up", key="go_signup"):
                st.session_state["auth_mode"] = "signup"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            with st.form("signup_form"):
                st.markdown("##### Create an account")
                new_name = st.text_input("Full Name", placeholder="John Doe")
                new_email = st.text_input("Work Email", placeholder="evaluator@mirai.edu")
                new_password = st.text_input("Create Password", type="password", placeholder="••••••••")
                submit_signup = st.form_submit_button(
                    "Create Account", use_container_width=True, type="primary"
                )
            if submit_signup:
                if "@" in new_email and new_password and new_name.strip():
                    st.session_state["logged_in"] = True
                    st.session_state["user_email"] = new_email
                    st.session_state["user_name"] = new_name
                    st.rerun()
                else:
                    st.error("Please fill out all fields correctly.")

            st.markdown('<div class="auth-toggle" style="text-align:center;">', unsafe_allow_html=True)
            if st.button("Already have an account? Sign in", key="go_login"):
                st.session_state["auth_mode"] = "login"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# 2. MAIN DASHBOARD
# ==========================================================
else:
    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("### 🛡️ DataGuard AI")
        st.caption(f"👤 {st.session_state['user_email']}")

        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["user_email"] = ""
            st.rerun()

        st.markdown("---")
        st.markdown("##### 📁 Datasets Used")
        if not st.session_state["dataset_history"]:
            st.caption("No datasets uploaded yet.")
        else:
            for ds_name in st.session_state["dataset_history"]:
                is_active = ds_name == st.session_state["current_dataset_name"]
                marker = "🟢" if is_active else "⚪"
                st.markdown(
                    f'<div class="side-card">{marker} <b>{ds_name}</b></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("##### 💬 AI Audit Prompt History")
        any_prompts = False
        for ds_name, history in st.session_state["dataset_history"].items():
            if history["prompts"]:
                any_prompts = True
                with st.expander(f"📄 {ds_name}", expanded=False):
                    st.caption(f"Uploaded: {history['upload_time']}")
                    st.caption(f"Model: {history['model_chosen'] or 'N/A'}")
                    st.markdown("**Prompts:**")
                    for idx, p in enumerate(history["prompts"], 1):
                        st.caption(f"{idx}. {p}")
                    if history["audits"]:
                        st.markdown("**Audit runs:**")
                        for audit_idx, audit in enumerate(history["audits"], 1):
                            st.caption(f"#{audit_idx} · {audit['timestamp']} · {audit['model']}")
        if not any_prompts:
            st.caption("No AI audit prompts submitted yet.")

    # ---------------- HEADER ----------------
    st.markdown(
        f"<h1 style='color:{T['accent_color']}; font-weight:800; margin-bottom:0px;'>🛡️ DataGuard AI Studio</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{T['sub_text']}; font-size:1.05rem; margin-bottom:24px;'>"
        "Multimodal Machine Learning Readiness and Data Quality Engine</p>",
        unsafe_allow_html=True,
    )

    tab_upload, tab_clean, tab_audit, tab_health = st.tabs(
        ["📤 Upload CSV", "🧹 Clean Data", "🤖 AI Audit", "📊 Health Overview"]
    )

    # ---------------- TAB: UPLOAD CSV ----------------
    with tab_upload:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Data Ingestion")
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader("Upload Raw CSV Dataset", type=["csv"], key="main_csv_uploader")
        with col_up2:
            sample_choice = st.selectbox(
                "Or choose a sample dataset:",
                ["None", "Telco Customer Churn Sample"],
            )
        st.markdown("</div>", unsafe_allow_html=True)

        df, dataset_name = None, ""
        if sample_choice == "Telco Customer Churn Sample" and uploaded_file is None:
            try:
                df = pd.read_csv(
                    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
                    "master/data/Telco-Customer-Churn.csv"
                )
                dataset_name = "Telco-Customer-Churn.csv"
                st.success("Loaded Telco Customer Churn sample dataset.")
            except Exception:
                st.warning("Could not load the online sample dataset.")
        elif uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name
                st.success("Dataset loaded successfully.")
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")

        if df is not None and dataset_name:
            st.session_state["current_df"] = df
            st.session_state["current_dataset_name"] = dataset_name
            if dataset_name not in st.session_state["dataset_history"]:
                st.session_state["dataset_history"][dataset_name] = {
                    "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stats": get_dataset_stats(df),
                    "model_chosen": None,
                    "prompts": [],
                    "audits": [],
                }

        if st.session_state["current_df"] is not None:
            st.markdown("---")
            st.caption(f"Active dataset: **{st.session_state['current_dataset_name']}**")
            st.dataframe(st.session_state["current_df"].head(20), use_container_width=True)
        else:
            st.info("Upload a CSV or choose a sample dataset to get started.")

    # Shared reference used by the remaining tabs
    df = st.session_state["current_df"]
    dataset_name = st.session_state["current_dataset_name"]

    # ---------------- TAB: CLEAN DATA ----------------
    with tab_clean:
        if df is None:
            st.info("👈 Upload a dataset in the **Upload CSV** tab first.")
        else:
            st.subheader("🔍 Interactive Data Editor & Search")
            search_query = st.text_input(
                "Search dataset by keyword across all fields:", "", key="table_search_input"
            )
            filtered_df = search_dataset(df, search_query) if search_query else df
            if search_query:
                st.caption(f"Showing {len(filtered_df)} row(s) matching '{search_query}'")

            edited_df = st.data_editor(
                filtered_df, num_rows="dynamic", use_container_width=True, key="main_data_editor"
            )
            st.session_state["edited_df"] = edited_df

            st.markdown("---")
            st.markdown("### 📥 Export Cleaned Dataset")
            st.caption("Download your processed dataset, ready for baseline model training.")
            export_df = st.session_state["edited_df"] if st.session_state["edited_df"] is not None else df
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned CSV",
                data=csv_bytes,
                file_name="dataguard_cleaned_dataset.csv",
                mime="text/csv",
                type="primary",
            )

    # ---------------- TAB: AI AUDIT ----------------
    with tab_audit:
        if df is None:
            st.info("👈 Upload a dataset in the **Upload CSV** tab first.")
        else:
            st.subheader("🤖 Gemini AI ML-Readiness Engine")

            st.markdown("##### 💡 Prompt Library")
            ex_col1, ex_col2 = st.columns(2)
            p1_clicked = ex_col1.button("📋 Customer Churn ML Readiness", use_container_width=True)
            p2_clicked = ex_col2.button("⚠️ Missing Data & Imbalance Audit", use_container_width=True)
            if p1_clicked:
                st.session_state["audit_prompt_text"] = (
                    "Audit this dataset to evaluate its readiness for predicting customer churn. "
                    "Check for target variable imbalances and missing value risks."
                )
                st.rerun()
            elif p2_clicked:
                st.session_state["audit_prompt_text"] = (
                    "Identify severe data quality bottlenecks, highly correlated features, missing "
                    "value strategies, and potential data leaks for model training."
                )
                st.rerun()

            st.markdown("##### 🎙️ Or Speak Your Prompt")
            rec_col, txt_col = st.columns([1, 2])
            with rec_col:
                recorded_audio = audiorecorder("Record", "Stop Recording")
            with txt_col:
                if len(recorded_audio) > 0:
                    transcribed = speech_to_text(recorded_audio)
                    if transcribed:
                        st.session_state["audit_prompt_text"] = transcribed
                        st.success(f"Transcribed: \u201c{transcribed}\u201d")
                    else:
                        st.warning("Couldn't make out any speech — please try again or type your prompt.")

            with st.form("gemini_audit_form"):
                aud_col1, aud_col2 = st.columns([1, 2])
                with aud_col1:
                    selected_model = st.selectbox(
                        "AI Model Engine",
                        ["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 1.5 Pro"],
                        key="model_selector",
                    )
                    schema_image_file = st.file_uploader(
                        "Upload Schema / Data Dictionary Image (optional)",
                        type=["png", "jpg", "jpeg"],
                    )
                with aud_col2:
                    prompt_input = st.text_area(
                        "Describe your ML project objective or audit prompt:",
                        value=st.session_state["audit_prompt_text"],
                        placeholder="e.g., Evaluate dataset suitability for training a supervised "
                        "classification model...",
                        height=140,
                        key="audit_prompt_area",
                    )
                submit_audit = st.form_submit_button(
                    "🚀 Run AI ML-Readiness Audit", use_container_width=True, type="primary"
                )

            if submit_audit:
                st.session_state["audit_prompt_text"] = prompt_input
                if not st.session_state["audit_prompt_text"].strip():
                    st.warning("Please provide a prompt or select one from the library above.")
                else:
                    with st.spinner("Analyzing dataset health and running the Gemini audit..."):
                        schema_img = Image.open(schema_image_file) if schema_image_file else None
                        df_stats = get_dataset_stats(df)
                        report = generate_ml_readiness_report(
                            df_stats, st.session_state["audit_prompt_text"], schema_img
                        )

                        hist = st.session_state["dataset_history"][dataset_name]
                        hist["model_chosen"] = selected_model
                        hist["prompts"].append(st.session_state["audit_prompt_text"])
                        hist["audits"].append(
                            {
                                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                                "prompt": st.session_state["audit_prompt_text"],
                                "report": report,
                                "model": selected_model,
                            }
                        )
                        st.session_state["latest_report"] = report

            if st.session_state["latest_report"]:
                st.markdown("---")
                st.subheader("📋 Tailored ML-Readiness Report")
                st.markdown(
                    f'<div class="main-card">{st.session_state["latest_report"]}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("##### 🔊 Voice Report")
                st.caption("Have the report read aloud — handy when presenting to a client.")
                if st.button("Generate & Listen to Voice Audit", key="main_voice_btn"):
                    with st.spinner("Synthesizing audio voice report..."):
                        audio_data = text_to_speech(st.session_state["latest_report"])
                        st.audio(audio_data, format="audio/mp3")

    # ---------------- TAB: HEALTH OVERVIEW ----------------
    with tab_health:
        if df is None:
            st.info("👈 Upload a dataset in the **Upload CSV** tab first.")
        else:
            stats = get_dataset_stats(df)
            st.subheader("📊 Dataset Health Overview")
            c1, c2, c3, c4 = st.columns(4)
            metric_tile(c1, "Total Rows", f"{stats['total_rows']:,}")
            metric_tile(c2, "Total Features", f"{stats['total_cols']}")
            metric_tile(c3, "Missing Cells", f"{stats['missing_cells']:,}")
            metric_tile(c4, "Duplicate Rows", f"{stats['duplicate_rows']:,}")

            st.write("")
            st.subheader("📈 Missing Values Distribution")
            missing_counts = df.isnull().sum().reset_index()
            missing_counts.columns = ["variable", "missing_count"]
            if missing_counts["missing_count"].sum() > 0:
                fig_miss = px.line(
                    missing_counts[missing_counts["missing_count"] > 0],
                    x="variable",
                    y="missing_count",
                    title="Missing Values per Column",
                    template=PLOTLY_TEMPLATE,
                    markers=True,
                )
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.info("🎉 Zero missing values detected in this dataset!")