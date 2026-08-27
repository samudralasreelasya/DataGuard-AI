```
██████╗  █████╗ ████████╗ █████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
██║  ██║███████║   ██║   ███████║██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██║  ██║██╔══██║   ██║   ██╔══██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
██████╔╝██║  ██║   ██║   ██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝

                   🛡️  AI  |  MULTI-MODAL DATA-QUALITY & ML-READINESS STUDIO
```

> **MirAI School of Technology Capstone Project** — Category D: Productivity & Enterprise Automation
> **Live Demo:** [dataguard-ai-titvzta9necybkrd4njqlq.streamlit.app](https://dataguard-ai-titvzta9necybkrd4njqlq.streamlit.app/)
> **Repository:** [github.com/samudralasreelasya/DataGuard-AI](https://github.com/samudralasreelasya/DataGuard-AI)

---

```bash
$ whoami
Samudrala Sreelasya — B.Tech capstone project, MirAI School of Technology

$ cat mission.txt
Validate, audit, and profile CSV datasets before they hit an ML pipeline —
Using an actual Gemini model for the reasoning, not a pre-written template.
```

---

## 📡 `> overview`

**DataGuard AI** is a Streamlit dashboard that ingests a CSV, profiles it with Pandas
(missing values, duplicates, memory footprint), lets you clean it interactively with
He goes to st.data_editor and then gives the live Gemini model the actual dataset statistics.
plus an optional schema-screenshot to produce a tailored **ML-readiness audit** —
Not a pre-written answer; the audit can also be converted into speech using `gTTS`.

```bash
$ ./run_pipeline.sh
[1/5] Read CSV file ................ done
[2/5] The Pandas profile... completed
[3/5] Creating the prompt using the statistics... completed
[4/5] Call gemini-1.5-flash ... done
[5/5] Render report and audio. Done.
```

---

## ✨ `> features`

| # | Feature | Implementation |
|---|---------|-----------------|
| 1 | Interactive data ingestion & profiling | `pandas` — rows, columns, missing cells, duplicate rows |
| 2 | Live keyword search with inline editing | using `st.data_editor` and a `.str.contains()` mask applied to all the columns |
| 3 | KPI cards showing the differences | using the `st.metric()` function to compare the current run with the previous run of the same dataset |
| 4 | Visualization of missing values | bar chart in `plotly.express` with theme awareness
| 5 | **Real Gemini AI audit** | `google-generativeai` — system-prompted, dataset-grounded, model-selectable |
| 6 | **Multimodal schema understanding** | Uploaded schema/data-dictionary image passed directly into `generate_content()` |
| 7 | Voice audit playback | `gTTS` converts the AI report to MP3, played in-browser |
| 8 | Session persistence | `st.session_state` keeps the theme, login, and per-dataset audit history. |
| 9 | Export of the cleaned dataset | Download in CSV format using `st.download_button` after editing |

---

## 🧠 `> ai_integration`

Unlike a static templated response, every audit is grounded in the dataset actually
uploaded:

```python
system_instruction = (
    You are DataGuard AI, a senior machine learning data-readiness auditor. Your job is to check datasets before they go into machine learning projects. You review data for quality, missing values, duplicates, and outliers. You look for bias and flag anything that could affect model training or results. You also check data labels and formats to make sure they match project goals. You write clear reports about your findings and give simple ways to fix problems. Your work helps teams use clean, trustworthy data for their projects.
    "Always cover completeness, duplicate/leakage risk, class imbalance, "
    advice on preprocessing, together with a general readiness judgment.
)

user_prompt = f