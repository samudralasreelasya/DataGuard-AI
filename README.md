# DataGuard AI — AI & RAG Data Readiness Studio

DataGuard AI is a planned Streamlit application for checking whether a tabular dataset is ready for machine-learning workflows and for giving practical, AI-assisted recommendations before a model or RAG pipeline is built.

## Planned scope

- Upload a user-provided CSV dataset (no data is stored in this repository).
- Inspect data quality: missing values, duplicate rows, inconsistent types, outliers, and class balance.
- Present interactive visual summaries and an editable cleaning review.
- Generate an AI-assisted readiness report using Gemini, based on calculated checks and user-provided context.
- Export a reviewed, cleaned dataset and a concise readiness summary.

The first release deliberately focuses on CSV/tabular data. Document, image, video, deep-learning training, and RAG ingestion are future extensions—not part of this capstone MVP.

## Planned structure

```text
DataGuard-AI/
├── app.py                 # Streamlit entry point (to be implemented together)
├── requirements.txt
├── README.md
├── .gitignore
├── src/                   # Application modules (to be added)
├── tests/                 # Tests (to be added)
└── docs/                  # Architecture and demo materials (to be added)
```

## Working schedule (IST)

| Date | Time | Planned focus |
| --- | --- | --- |
| Aug 12–16 | 20–30 minutes/day, flexible | Set the feature scope, create the project shell, sketch the interface, and prepare the build plan. |
| Aug 17–21 | Exam week; optional 10–20 minutes/day | Exam-first: only light tasks such as notes, UI decisions, or README updates if time permits. |
| Aug 22 (Saturday) | Main build session | Build the Streamlit dashboard shell and CSV workflow together. |
| Aug 23 (Sunday) | Main build session | Add the Gemini readiness report, visual polish, and testing. |
| Aug 24 | Flexible finalisation session | Deploy, document the system design, and rehearse the demo. |
| Aug 25 | Before 11:59 PM | Final testing, push final code to `main`, record demo video, and publish the required LinkedIn post. |

Exact session start/end times will be added as we agree them; this table records the planned dates and working windows so the project timeline remains visible.

## Next session

We will decide the first screen layout and create `app.py` together. No user data or sample datasets have been added.
