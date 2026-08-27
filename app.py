import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from PIL import Image
from gtts import gTTS
import datetime
import os
from google import genai
from google.genai import types as genai_types

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DataGuard AI | ML-Readiness Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GEMINI API CONFIGURATION
# ============================================================
# Looks first at Streamlit Cloud secrets (st.secrets), then falls
# back to a local environment variable. Never hardcode the key.
# Uses the current google-genai SDK — the older google-generativeai
# package has been fully deprecated by Google and no longer receives
# updates or bug fixes.
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

GEMINI_API_KEY = get_api_key()
GEMINI_READY = bool(GEMINI_API_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_READY else None

# Maps the friendly dropdown label to a real, callable Gemini model id.
# Gemini 1.5 models were fully shut down by Google in 2026 (requests now 404).
# "Gemini Ultra" was never an available generateContent model, so it's not listed.
MODEL_MAP = {
    "Gemini 2.5 Flash (fast, cost-efficient)": "gemini-2.5-flash",
    "Gemini 3.6 Flash (latest, GA)": "gemini-3.6-flash",
}


def get_dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "total_rows": df.shape[0],
        "total_cols": df.shape[1],
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


def build_schema_summary(df: pd.DataFrame, max_cols: int = 40) -> str:
    """Turns column names/dtypes/null counts into compact text context for the prompt."""
    lines = []
    for col in df.columns[:max_cols]:
        lines.append(
            f"- {col} | dtype: {df[col].dtype} | nulls: {int(df[col].isnull().sum())} | "
            f"unique: {df[col].nunique()}"
        )
    if len(df.columns) > max_cols:
        lines.append(f"...and {len(df.columns) - max_cols} more columns")
    return "\n".join(lines)


def guess_target_column(df: pd.DataFrame) -> str:
    """
    Heuristic guess at a likely target/label column, so the example
    prompts can reference a real column name instead of a generic one.
    Prefers low-cardinality categorical/boolean columns whose name hints
    at a target (churn, label, target, class, outcome, status, default,
    fraud, converted...), falling back to the lowest-cardinality
    non-numeric column, then just the last column.
    """
    hint_words = ["churn", "target", "label", "class", "outcome",
                  "status", "default", "fraud", "convert", "response", "y"]
    candidates = []
    for col in df.columns:
        col_lower = col.lower()
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1 or nunique > 20:
            continue
        score = 0
        if any(h in col_lower for h in hint_words):
            score += 10
        if nunique == 2:
            score += 5
        if df[col].dtype == object or str(df[col].dtype) == "category":
            score += 2
        candidates.append((score, col))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return df.columns[-1] if len(df.columns) else "the target variable"


def build_dynamic_prompts(df: pd.DataFrame) -> tuple[str, str]:
    """
    Generates two example audit prompts grounded in THIS dataset's actual
    columns, instead of static churn-specific placeholder text.
    """
    target_col = guess_target_column(df)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    top_missing = (
        df.isnull().sum().sort_values(ascending=False).head(3)
    )
    top_missing_cols = [c for c in top_missing.index if top_missing[c] > 0]

    prompt_1 = (
        f"Audit this dataset to evaluate its readiness for predicting "
        f"'{target_col}'. Check for class imbalance in '{target_col}' and "
        f"missing value risks across its {df.shape[1]} columns."
    )

    if top_missing_cols:
        cols_txt = ", ".join(top_missing_cols)
        prompt_2 = (
            f"Identify severe data quality bottlenecks focused on the columns "
            f"with the most missing data ({cols_txt}), flag any highly "
            f"correlated numeric features among {', '.join(numeric_cols[:5]) or 'the numeric columns'}, "
            f"and suggest a missing-value strategy."
        )
    else:
        prompt_2 = (
            f"This dataset has zero missing values. Identify potential data "
            f"leakage risks, correlated numeric features among "
            f"{', '.join(numeric_cols[:5]) or 'the numeric columns'}, and any "
            f"class imbalance risk in '{target_col}' before model training."
        )

    return prompt_1, prompt_2


def run_gemini_audit(df_stats: dict, objective: str, model_id: str,
                      schema_summary: str, schema_image: Image.Image = None) -> str:
    """
    Real Gemini call. Builds a system-style instruction + dynamic f-string context
    from the dataset's actual stats and schema, optionally attaches a schema/data
    dictionary screenshot for multimodal grounding, and returns the model's text.
    Raises a RuntimeError with a friendly message on failure so the caller can
    show st.error() instead of crashing the app.
    """
    if not GEMINI_READY:
        raise RuntimeError(
            "No Gemini API key is configured. Add GEMINI_API_KEY to Streamlit "
            "secrets (Settings → Secrets) or your local environment."
        )

    system_instruction = (
        "You are DataGuard AI, a senior ML data-readiness auditor. You review "
        "tabular datasets and give a structured, actionable readiness report for "
        "a stated machine learning objective. Always cover: (1) data completeness "
        "and what the missing values imply, (2) duplicate/leakage risk, (3) target "
        "variable and class imbalance risk if relevant to the stated objective, "
        "(4) concrete preprocessing recommendations, and (5) an overall readiness "
        "verdict (Ready / Needs Work / Not Ready). Be specific to the numbers given, "
        "not generic. Keep it under 250 words and use markdown headers/bullets."
    )

    user_prompt = (
        f"ML Objective stated by the user: {objective}\n\n"
        f"Dataset statistics:\n"
        f"- Rows: {df_stats['total_rows']:,}\n"
        f"- Columns: {df_stats['total_cols']}\n"
        f"- Missing cells: {df_stats['missing_cells']:,}\n"
        f"- Duplicate rows: {df_stats['duplicate_rows']:,}\n\n"
        f"Column-level schema:\n{schema_summary}\n\n"
    )

    if schema_image is not None:
        user_prompt += (
            "A schema/data-dictionary screenshot is attached — use it to "
            "understand column semantics and refine the report.\n\n"
        )

    user_prompt += "Produce the ML-readiness report now."

    contents = [user_prompt]
    if schema_image is not None:
        contents.append(schema_image)

    try:
        response = gemini_client.models.generate_content(
            model=model_id,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction
            ),
        )
        return response.text
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
            raise RuntimeError(
                "Gemini rate limit hit (HTTP 429). Please wait a moment and try again."
            )
        elif "API key" in err_str or "PERMISSION_DENIED" in err_str or "API_KEY_INVALID" in err_str:
            raise RuntimeError("Gemini API key was rejected. Check GEMINI_API_KEY.")
        elif "deprecat" in err_str.lower() or "404" in err_str or "NOT_FOUND" in err_str:
            raise RuntimeError(
                f"Model '{model_id}' is unavailable or deprecated. Try a different model."
            )
        else:
            raise RuntimeError(f"Gemini request failed: {err_str}")


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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

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
    color: #3B82F6 !important;
}

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

        t_col1, t_col2 = st.columns([3, 1])
        with t_col2:
            theme_btn = st.button("🌙 Dark" if st.session_state["theme"] == "light" else "☀️ Light", key="login_theme_toggle")
            if theme_btn:
                st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
                st.rerun()

        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        auth_tab1, auth_tab2 = st.tabs(["🔒 Sign In", "📝 Sign Up"])

        with auth_tab1:
            with st.form("login_form"):
                email = st.text_input("Work Email", placeholder="evaluator@mirai.edu", key="login_email")
                password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
                submit_login = st.form_submit_button("Sign In to Studio", width='stretch')
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
                submit_signup = st.form_submit_button("Create Account", width='stretch')
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
    with st.sidebar:
        st.markdown("### 🛡️ DataGuard AI")
        st.caption(f"👤 {st.session_state['user_email']}")

        if not GEMINI_READY:
            st.warning("⚠️ GEMINI_API_KEY not set — AI Audit tab will not run.", icon="⚠️")

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

    st.markdown(f"<h1 style='color: {accent_color}; font-weight: 700; margin-bottom: 0px;'>🛡️ DataGuard AI Studio</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {sub_text}; font-size: 1.05rem; margin-bottom: 24px;'>Multimodal Machine Learning Readiness and Data Quality Engine</p>", unsafe_allow_html=True)

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

    if df is not None and dataset_name:
        if dataset_name not in st.session_state["dataset_history"]:
            st.session_state["dataset_history"][dataset_name] = {
                "upload_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stats": get_dataset_stats(df),
                "model_chosen": None,
                "prompts": [],
                "audits": []
            }

    # --- Everything below requires a loaded dataset ---
    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Health & Analytics",
            "🔍 Data Cleaning & Search",
            "🤖 Gemini AI Audit",
            "📥 Export & Options"
        ])

        with tab1:
            stats = get_dataset_stats(df)
            st.subheader("📊 Dataset Health Overview")

            # Delta vs. this dataset's previous upload/run, if one exists in history
            history_entry = st.session_state["dataset_history"].get(dataset_name, {})
            prev_stats = history_entry.get("stats") if history_entry.get("audits") else None
            missing_delta = None
            dup_delta = None
            if prev_stats:
                missing_delta = stats["missing_cells"] - prev_stats["missing_cells"]
                dup_delta = stats["duplicate_rows"] - prev_stats["duplicate_rows"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows", f"{stats['total_rows']:,}")
            c2.metric("Total Features", stats["total_cols"])
            c3.metric(
                "Missing Cells", f"{stats['missing_cells']:,}",
                delta=(f"{missing_delta:+,}" if missing_delta is not None else None),
                delta_color="inverse"
            )
            c4.metric(
                "Duplicate Rows", f"{stats['duplicate_rows']:,}",
                delta=(f"{dup_delta:+,}" if dup_delta is not None else None),
                delta_color="inverse"
            )

            st.write("")
            st.subheader("📈 Missing Values Distribution")
            missing_counts = df.isnull().sum().reset_index()
            missing_counts.columns = ['variable', 'missing_count']
            if missing_counts['missing_count'].sum() > 0:
                plot_df = missing_counts[missing_counts['missing_count'] > 0].sort_values(
                    'missing_count', ascending=False
                )
                fig_miss = px.line(
                    plot_df,
                    x='variable',
                    y='missing_count',
                    markers=True,
                    title="Missing Values per Column",
                    template="plotly_dark" if st.session_state["theme"] == "dark" else "plotly_white"
                )
                fig_miss.update_traces(line=dict(width=3), marker=dict(size=8))
                fig_miss.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig_miss, width='stretch')
            else:
                st.info("🎉 Zero missing values detected in dataset!")

        with tab2:
            st.subheader("🔍 Interactive Data Editor & Search")
            search_query = st.text_input("🔎 Search dataset by keyword across all fields:", "", key="table_search_input")
            filtered_df = df
            if search_query:
                mask = np.column_stack([df[col].astype(str).str.contains(search_query, case=False, na=False) for col in df.columns])
                filtered_df = df.loc[mask.any(axis=1)]
                st.caption(f"Showing {len(filtered_df)} row(s) matching search term: '{search_query}'")
            edited_df = st.data_editor(filtered_df, num_rows="dynamic", width='stretch', key="main_data_editor")

        with tab3:
            st.subheader("🤖 Gemini AI ML-Readiness Engine")

            # The text_area below owns its own state via key="audit_prompt_area".
            # Once a widget has a key, Streamlit ignores any `value=` argument on
            # reruns and reads only from st.session_state[key] — so to make the
            # example buttons actually populate the textbox, we must write to
            # THIS key (not a separate shadow variable) before rerunning.
            if "audit_prompt_area" not in st.session_state:
                st.session_state["audit_prompt_area"] = ""

            dyn_prompt_1, dyn_prompt_2 = build_dynamic_prompts(df)

            st.markdown("##### 💡 Example Audit Prompts (generated from this dataset):")
            ex_col1, ex_col2 = st.columns(2)
            p1_clicked = ex_col1.button(f"📋 {dyn_prompt_1[:60]}…", width='stretch', help=dyn_prompt_1)
            p2_clicked = ex_col2.button(f"⚠️ {dyn_prompt_2[:60]}…", width='stretch', help=dyn_prompt_2)

            if p1_clicked:
                st.session_state["audit_prompt_area"] = dyn_prompt_1
                st.rerun()
            elif p2_clicked:
                st.session_state["audit_prompt_area"] = dyn_prompt_2
                st.rerun()

            with st.form("gemini_audit_form"):
                aud_col1, aud_col2 = st.columns([1, 2])
                with aud_col1:
                    selected_model_label = st.selectbox(
                        "AI Model Engine",
                        list(MODEL_MAP.keys()),
                        key="model_selector"
                    )
                    schema_image_file = st.file_uploader(
                        "Upload Schema / Data Dictionary Image (Optional)",
                        type=["png", "jpg", "jpeg"]
                    )
                with aud_col2:
                    prompt_input = st.text_area(
                        "Describe your ML Project Objective or Audit Prompt:",
                        placeholder="e.g., Evaluate dataset suitability for training a supervised classification model...",
                        height=140,
                        key="audit_prompt_area"
                    )
                submit_audit = st.form_submit_button("🚀 Run AI ML-Readiness Audit", width='stretch')

            if submit_audit:
                if not prompt_input.strip():
                    st.warning("Please provide a prompt or select an example above.")
                else:
                    with st.spinner(f"Running live {selected_model_label} audit..."):
                        schema_img = Image.open(schema_image_file) if schema_image_file else None
                        df_stats = get_dataset_stats(df)
                        schema_summary = build_schema_summary(df)
                        model_id = MODEL_MAP[selected_model_label]

                        try:
                            report = run_gemini_audit(
                                df_stats=df_stats,
                                objective=prompt_input,
                                model_id=model_id,
                                schema_summary=schema_summary,
                                schema_image=schema_img
                            )
                            st.session_state["dataset_history"][dataset_name]["model_chosen"] = selected_model_label
                            st.session_state["dataset_history"][dataset_name]["prompts"].append(prompt_input)
                            st.session_state["dataset_history"][dataset_name]["audits"].append({
                                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                                "prompt": prompt_input,
                                "report": report,
                                "model": selected_model_label
                            })
                            st.session_state["latest_report"] = report
                        except RuntimeError as e:
                            st.error(f"🚫 {e}")

            if "latest_report" in st.session_state:
                st.markdown("---")
                st.subheader("📋 Tailored ML-Readiness Report Output")
                st.markdown(st.session_state["latest_report"])

                st.markdown("##### 🎙️ Voice Audit Synthesis")
                if st.button("🔊 Generate & Listen to Voice Audit", key="main_voice_btn"):
                    with st.spinner("Synthesizing audio voice report..."):
                        audio_data = text_to_speech(st.session_state["latest_report"])
                        st.audio(audio_data, format="audio/mp3")

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