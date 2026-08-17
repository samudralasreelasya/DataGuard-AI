# 🛡️ DataGuard AI | Multi-modal Data Collection & ML-Readiness Studio

> **MirAI School of Technology Capstone Project (Category D: Productivity & Enterprise Automation)** > **Live Demo:** [Streamlit Cloud Deployment URL](#) *(Replace with your live link once deployed)* > **Repository:** [samudralasreelasya/DataGuard-AI](https://github.com/samudralasreelasya/DataGuard-AI)

---

## 🚀 Overview
**DataGuard AI** is a sophisticated multi-modal data quality and machine learning readiness tool created using **Streamlit**, **Pandas**, **Plotly**, and **Google Gemini AI**. It gives developers, data scientists, and analysts the ability to validate, audit, and profile CSV files before using them for ML training or RAG systems.

---

## ✨ Key Features
1. **Interactive Data Ingestion & Profiling**: Load large CSV files (max file size 500MB) directly in the main dashboard to derive instant health metrics (missing values, duplicate rows, memory consumption).
2. **Interactive Data Editor (`st.data_editor`)**: Edit and rectify inconsistent values directly from the web interface to export a production ready dataset.
3. **Advanced Visualization Tools**: Interactive KPI cards, missing value density heatmaps, correlation matrix, and distribution graphs with Plotly.
4. **Multimodal Gemini AI Capability (`gemini-3.5-flash`)**:
   - **Vision Functionality**: Upload a data dictionary/schema screenshot in order to make Gemini understand column semantics.
   - **Custom ML Readiness Audit**: Uses dataset statistics, schema information, and project goals for structured audit reports, bias warnings, and missing values strategy.
5. **Text-to-Speech (TTS) Report Creation**: Transform your customized readiness report into audible MP3 format using `gTTS`.
## 🛠️ System Architecture Diagram
```mermaid
graph TD
    A[User / Browser] -->|Uploads CSV & Schema Image| B(Streamlit Frontend Dashboard)
    A -->|Project Objective Description| B
    B -->|State Management: st.session_state| B
    B -->|Calculates Stats via Pandas| C[Data Processing Module]
    C -->|Numeric Distributions & Heatmaps| B
    B -->|Sends Multimodal Payload: Image + Stats + Prompt| D[Google Gemini AI Engine]
    D -->|Returns Tailored ML-Readiness Report| B
    B -->|Interactive Data Editor| E[Cleaned CSV Export]
⚙️ Tech Stack & Dependencies
Python 3.10+

Streamlit (Interactive Frontend UI)

Pandas & NumPy (Data processing & memory optimization)

Plotly (Interactive data visualizations & heatmaps)

Google GenerativeAI SDK (google-generativeai)

gTTS (Google Text-to-Speech audio synthesis)

Pillow (PIL) (Image processing for schema uploads)
🚀 Local Installation & Setup
1. Clone the Repository:
    git clone [https://github.com/samudralasreelasya/DataGuard-AI.git](https://github.com/samudralasreelasya/DataGuard-AI.git)
cd DataGuard-AI
2. Create and Activate a Virtual Environment (Recommended):
    python -m venv .venv
    .venv\Scripts\activate   # On Windows PowerShell/CMD
3. Install Dependencies:
    pip install -r requirements.txt
4. Set Up Your Gemini API Key:
    Set your environment variable in your terminal:
    set GEMINI_API_KEY=your_actual_api_key_here
5. Run the Application:
    python -m streamlit run app.py
🧠 Key Features of Capstone Design
Preservation of Session State: Utilizes st.session_state to save profiling data and AI reports even after various interactions with widgets.

Minimizing Unnecessary Requests: Makes use of st.form to avoid unnecessary queries to the API, Gemini, by design.

Exception Handling: In-built exception handlers help in managing HTTP 429 rate limit and deprecation exceptions without UI failure.

© 2026 Samudrala Sreelasya | MirAI School of Technology Capstone

