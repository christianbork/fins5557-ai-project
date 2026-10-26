# CLAUDE.md: FINS5557 group project (Track A)

## Context
- UNSW FINS5557 Applied AI in Finance, T3 2026. Group project, Track A (Tech Team), team of 3.
- Goal: a working AI-powered finance app, deployed to a public URL, with a GitHub repo, a report (15 pages max) and a 5-7 min demo video.
- **Due: Friday 6 November 2026 (Week 8), 11:59 PM.**
- **What we're building:** a credit card default screener (UCI Taiwan data). ML model → calibrated PD → approve/refer/decline from a cost-based threshold → SHAP reasons → LLM-drafted decline letter, plus a fairness audit and PSI drift monitoring.
- **The build plan is in `docs/BUILD_PLAN.md`.** Work through it one step at a time, in order, and tick off each step when its checks pass. It also lists the decisions made so far and the ideas that are not decided yet.
- Course materials, Code Bank and brief/rubric PDFs live in:
  `C:\Users\chris\OneDrive\Desktop\Masters\Semester 3\Applied AI`
  (`Code Bank/README.md` indexes all seminar code by snippet ID, e.g. W05-02).

## How to work in this repo
- Python 3.12, virtual env in `.venv`. Install with `pip install -r requirements-dev.txt` (includes `requirements.txt`).
- Stack: pandas, scikit-learn, xgboost, shap, streamlit, python-dotenv. LLM backend set by `LLM_BACKEND` in `.env` (`template`, `gemini` or `ollama`).
- `app.py` is the Streamlit entry point; real logic goes in modules under `src/` (keep the UI thin).
- **One notebook only: `notebooks/project.ipynb`.** Add a section for each step instead of creating new notebooks. Working code is moved into `src/` and imported back.
- Run `pytest -q` and `ruff check .` after each change.
- Explain changes clearly; I'm learning and need to explain this code in the report and video. Link the code to the finance concept (credit risk, expected loss, regulation).
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
- Ask before installing new packages; add any new package to the right requirements file, pinned.
- Ask before running long training jobs.
- Suggest a commit message after each finished piece of work.
