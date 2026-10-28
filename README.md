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
The app needs only the committed model in `models/`, not the Kaggle data.
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Decline letters use the standard template by default. To have a local language model draft them instead (customer data stays on your machine), install [Ollama](https://ollama.com), run `ollama pull llama3.2:3b`, and set `LLM_BACKEND=ollama` in a `.env` file. Model-written letters are drafts for the analyst to check before sending.

To retrain or run the notebook, install the dev tools (`pip install -r requirements-dev.txt`) and get the data (see **Data**). Tests: `pytest -q`; lint: `ruff check .`.

## Deployment
Deployed on Streamlit Community Cloud from the `main` branch (entry point `app.py`, Python 3.12) with `LLM_BACKEND = "template"` in the app's Secrets settings, so the public demo never calls a language model and only shows invented example applicants. No keys are stored in the code.

## Team
| Name | zID | Role |
|---|---|---|
| Christian | | |
| | | |
| | | |

## Data
**Home Credit Default Risk** (Home Credit Group, Kaggle competition, 2018): `application_train.csv`, 307,511 consumer loan applications, one row each, with what the lender knew at application and whether the client later had payment difficulties (8.1% did).
Source: https://www.kaggle.com/competitions/home-credit-default-risk/data

**The data is not in this repo and must not be shared.** The competition rules limit use to the competition and forbid passing the data to anyone who hasn't accepted them. Our lecturer approved using it for this FINS5557 project (28 October 2026). So `data/raw/` is gitignored, and each teammate downloads their own copy:

1. Log in to kaggle.com with your own account, verify your phone number, and accept the rules on the [competition rules page](https://www.kaggle.com/competitions/home-credit-default-risk/rules).
2. Create an API token (kaggle.com → Settings → API → Create New Token) and save it once per laptop. The command asks for the token without showing it and saves it to `~/.kaggle/access_token`, outside the repo:
   ```bash
   python -c "import getpass,pathlib; p=pathlib.Path.home()/'.kaggle'; p.mkdir(exist_ok=True); (p/'access_token').write_text(getpass.getpass('Kaggle token: ').strip()); print('saved')"
   ```
   (`kagglehub.login()` only keeps the token for the current Python process when run in a terminal, so it doesn't work for the next step.)
3. Download the file into `data/raw/`:
   ```bash
   pip install -r requirements-dev.txt
   python -m src.data      # prints: 307,511 applications, default rate 8.1%
   ```

The deployed app only needs the trained model in `models/`, never the raw data, and its example applicants are invented, not rows from the Kaggle file.
