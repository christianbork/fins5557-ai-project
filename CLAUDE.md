# CLAUDE.md: FINS5557 group project (Track A)

## Context
- UNSW FINS5557 Applied AI in Finance, T3 2026. Group project, Track A (Tech Team), team of 3.
- Goal: a working AI-powered finance app, deployed to a public URL, with a GitHub repo, a report (15 pages max) and a 5-7 min demo video.
- **Due: Friday of Week 8, 11:59 PM.**
- Idea is not final yet. Leading option: BNPL / thin-file credit screener (XGBoost + SHAP, fairness audit, PSI monitoring, LLM-drafted adverse-action explanation).
- Course materials, Code Bank and brief/rubric PDFs live in:
  `C:\Users\chris\OneDrive\Desktop\Masters\Semester 3\Applied AI`
  (`Code Bank/README.md` indexes all seminar code by snippet ID, e.g. W05-02).

## How to work in this repo
- Python 3.10+, virtual env in `.venv`. Stack: pandas, scikit-learn, xgboost, shap, streamlit, google-generativeai.
- `app.py` is the Streamlit entry point; real logic goes in modules under `src/` (keep the UI thin).
- Explore in `notebooks/`, then move working code into `src/`.
- Explain changes clearly; I'm learning and need to explain this code in the report and video.
- When adapting course code, mention the Code Bank snippet ID.

## Marking priorities (from the rubric)
- Justified model choice, robust inference, edge-case and error handling.
- Modular, readable code; small meaningful Git commits; complete README and requirements.txt.
- Security: no hardcoded keys (use `src/config.py`, `.env`, Streamlit secrets); validate user input.
- UI shows loading / success / error states; clear instructions for users.
- At least one documented performance optimisation (e.g. `st.cache_resource` for model loading).

## Rules
- Never read, print or commit `.env` or `.streamlit/secrets.toml`.
- Don't commit data files or trained model binaries unless they're small and we agree to.
- Ask before installing new packages; add any new package to requirements.txt.
- Suggest a commit message after each finished piece of work.
