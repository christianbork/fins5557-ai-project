# FINS5557 Applied AI Project

> One-line description of the app (fill in once the idea is final).

**Live app:** _add Streamlit Cloud URL_ | **Demo video:** _add link_

## Overview
- **Problem:** ...
- **Target users:** ...
- **AI/ML technique:** ...

## Project structure
```
app.py            Streamlit entry point
src/              Application code (models, data processing, LLM calls)
  config.py       Loads API keys from .env / Streamlit secrets
notebooks/        Exploration and model development
data/             Local data (not committed)
tests/            Tests
```

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then add your API key to .env
streamlit run app.py
```

## Deployment
Deployed on Streamlit Community Cloud. API keys are stored in the app's Secrets settings, never in the code.

## Team
| Name | zID | Role |
|---|---|---|
| Christian | | |
| | | |
| | | |

## Data sources
...
