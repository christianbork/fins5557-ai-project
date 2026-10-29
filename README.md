# Loan default screener

[![CI](https://github.com/christianbork/fins5557-ai-project/actions/workflows/ci.yml/badge.svg)](https://github.com/christianbork/fins5557-ai-project/actions/workflows/ci.yml)

> An AI credit-screening app for a consumer lender: a machine-learning model estimates each applicant's probability of default (PD), a cost-based rule turns it into approve / refer / decline, SHAP explains the main reasons, and a language model drafts the letter, with a fairness audit and drift monitoring behind it. FINS5557 Applied AI in Finance, UNSW, T3 2026 (Track A).

**Live app:** _add Streamlit Cloud URL_ | **Demo video:** _add link_

## Overview
- **Problem:** a consumer lender has to decide quickly and consistently which loan applicants to approve, explain every decline, and show that its decisions are profitable, fair and still valid as the applicant population changes.
- **Target users:** credit analysts at a consumer lender (cash and revolving loans).
- **AI/ML techniques:**
  - **Machine learning:** logistic regression (scorecard baseline), random forest and XGBoost, tuned with cross-validation; XGBoost chosen (test AUC 0.757), with isotonic calibration so the scores are real PDs (mean test PD 8.2% vs 8.1% actual).
  - **Decision rule from the lender's economics:** expected loss = PD × LGD × EAD; the cut-off (21.0%) maximises profit and sits next to the 20% break-even PD (LGD 60%, margin 15%); a ±2-point band is referred to an analyst.
  - **Explainable AI:** SHAP gives the top reasons behind each score, in plain English.
  - **Generative AI:** Google Gemini (public demo) or a local Ollama model (real customer data) drafts the decline or refer letter, behind guardrails: the prompt holds no applicant data, outputs are checked, and any failure falls back to a fixed template. The letter is a draft for the analyst to check.
- **Governance:** input validation, a guard that refers applicants outside the training range (the model only saw loans Home Credit had already approved), a fairness audit by sex, age, family status and education, and PSI drift monitoring.

## Using the app
| Page | What it does |
|---|---|
| **Screening** (`app.py`) | Enter one applicant, or load one of 6 invented examples, then **Assess applicant**: PD, decision, expected loss and profit, the main reasons and a chart of what moved the score. For a decline or refer, **Draft letter**, check and edit it, then download it. |
| **Batch scoring** | Upload a CSV (or use the examples): rows with errors are listed by line, the rest are scored; download the results with PD, decision, reasons and expected loss. |
| **Model and fairness** | How the model was chosen (v1 vs v2), ROC and reliability charts, the profit curve with the profit and balanced-accuracy cut-offs, and the fairness table with flags. |
| **Monitoring** | PSI per input for a new batch against the training data (a normal and a shifted synthetic sample are built in), plus how the model's decisions for the batch compare with the test split. |

_Screenshots: add once deployed._

## How it works
```
applicant (form or CSV)
  -> src/applicant.py    form in years -> model fields
  -> src/validation.py   input errors; out-of-range guard (refer, never auto-approve)
  -> src/predict.py      calibrated XGBoost pipeline -> PD -> approve / refer / decline (src/threshold.py)
  -> src/explain.py      SHAP -> top 3 reasons
  -> src/llm.py          letter: template | Ollama (local) | Gemini (demo), with guardrails
offline: src/data.py -> src/features.py -> src/train.py (tuning, calibration, cut-off)
         -> src/fairness.py, src/reporting.py, src/monitoring.py -> models/ and reports/ (aggregates only)
```

## Project structure
```
app.py                  Screening page (Streamlit entry point)
pages/                  Batch scoring, Model and fairness, Monitoring
src/                    All logic (the pages only lay it out)
models/                 Trained model, metadata, training reference statistics (no applicant rows)
reports/                Model comparison, tuning, profit curve, cut-offs, fairness, chart data, performance
archive/                Earlier model versions; see docs/MODEL_HISTORY.md
data/sample/            Invented example applicants and synthetic monitoring batches (data/raw/ is not committed)
notebooks/project.ipynb The analysis notebook, one section per step
docs/                   BUILD_PLAN.md (steps and decisions), MODEL_HISTORY.md
tests/                  pytest suite (fake data; no Kaggle file or API key needed)
.github/workflows/      CI: ruff and pytest on every pull request
```

## Run locally
The app needs only the committed model in `models/`, not the Kaggle data.
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Decline letters use the standard template by default. To have a local language model draft them instead (customer data stays on your machine), install [Ollama](https://ollama.com), run `ollama pull llama3.2:3b`, and set `LLM_BACKEND=ollama` in a `.env` file. To use Google Gemini instead (the public demo's setting), create a free API key at [Google AI Studio](https://aistudio.google.com) and put `LLM_BACKEND=gemini` and `GOOGLE_API_KEY=...` in `.env`. Model-written letters are drafts for the analyst to check before sending. Whichever backend is used, the prompt contains only the decision and fixed reason phrases, never applicant data, and any failure falls back to the template.

To retrain or run the notebook, install the dev tools (`pip install -r requirements-dev.txt`) and get the data (see **Data**). Tests: `pytest -q`; lint: `ruff check .`.

## Deployment
Deployed on Streamlit Community Cloud from the `main` branch (entry point `app.py`, Python 3.12) with `LLM_BACKEND = "gemini"` and `GOOGLE_API_KEY` in the app's Secrets settings: the public demo drafts letters with Gemini's free tier, which is acceptable because it only handles invented example applicants and the prompt holds no applicant data. Real customer data would use the local Ollama backend instead (closed environment). No keys are stored in the code.

## Team
| Name | zID | Role |
|---|---|---|
| Christian Bork | | |
| | | |
| | | |

## Licence
MIT, see [LICENSE](LICENSE). The Home Credit data is not covered: it stays under the Kaggle competition rules (see **Data**).

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
