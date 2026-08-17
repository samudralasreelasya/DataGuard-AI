# System archticture and Design document : DataGaurd AI
## 1.Overview
DataGaurd AI is a multi model streamlit web application designed for the evaulation of raw csv dataset so that the dataset will be ready train the ML model.
## System Arthecture Flow
``mermaid
graph TD
    A[User / Browser] -->|Uploads CSV & Schema Image| B(Streamlit Frontend Dashboard)
    A -->|Voice Input / Text Objective| B
    B -->|State Management: st.session_state| B
    B -->|Calculates Stats via Pandas| C[Data Processing Module]
    C -->|Numeric Distributions & Correlations| B
    B -->|Sends Multimodal Payload: Image + Stats + Prompt| D[Google Gemini API Vision / Pro]
    D -->|Returns Tailored ML-Readiness Report & Fixes| B
    B -->|Interactive Data Editor| E[Cleaned CSV Export]