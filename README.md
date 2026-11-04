# Loan default screener

[![CI](https://github.com/christianbork/fins5557-ai-project/actions/workflows/ci.yml/badge.svg)](https://github.com/christianbork/fins5557-ai-project/actions/workflows/ci.yml)

> An AI credit-screening app for a consumer lender: a machine-learning model estimates each applicant's probability of default (PD), a cost-based rule turns it into approve / refer / decline, SHAP explains the main reasons, and a language model drafts the letter, with a fairness audit and drift monitoring behind it. FINS5557 Applied AI in Finance, UNSW, T3 2026 (Track A).

**Live app:** https://fins5557-loan-screener.streamlit.app | **Demo video:** _add link_

**Docs:** [model card](docs/MODEL_CARD.md) (scope, metrics, fairness, limitations) · [model history](docs/MODEL_HISTORY.md) (v0 to v2) · [build plan](docs/BUILD_PLAN.md) (steps and decisions)

## Overview
- **Problem:** a consumer lender has to decide quickly and consistently which loan applicants to approve, explain every decline, and show that its decisions are profitable, fair and still valid as the applicant population changes.
- **Target users:** credit analysts at a consumer lender (cash and revolving loans).
- **AI/ML techniques:**
  - **Machine learning:** logistic regression (scorecard baseline), random forest and XGBoost, tuned with cross-validation; XGBoost chosen (test AUC 0.757), with isotonic calibration so the scores are real PDs (mean test PD 8.2% vs 8.1% actual).
  - **Lending policy limits first (stage 1):** an application is declined before the model is used if the loan would take more than 48 instalments to repay, the repayment is larger than income, or the loan is more than 30 times income. The model can't extrapolate beyond the loans Home Credit approved, so these hard rules stop it scoring impossible applications (user testing found a $1m loan on a $30k income at $500 per instalment scored a PD of 7%).
  - **Decision rule from the lender's economics:** expected loss = PD × LGD × EAD; the cut-off (21.0%) maximises profit and sits next to the 20% break-even PD (LGD 60%, margin 15%); a ±2-point band is referred to an analyst.
  - **Explainable AI:** SHAP gives the top reasons behind each score, in plain English.
  - **Generative AI:** a **Letter Agent** (Google Gemini in the public demo, a local Ollama model for real customer data) drafts the decline or refer letter. Its prompt is a fixed, role-based brief (role, task, rules, output format) plus only the decision and reason phrases from a fixed list, so no applicant data reaches the model. Six rule-based **guardrail checks** run on every letter, any failure falls back to a fixed template, and the screening page shows both the exact prompt and the check results. The letter is a draft for the analyst to check.
- **Governance:** input validation, a guard that refers applicants outside the training range (the model only saw loans Home Credit had already approved), a fairness audit by sex, age, family status and education, PSI drift monitoring, a decision log, and a [model card](docs/MODEL_CARD.md).
- **Currency:** amounts are shown in USD as a display assumption. Home Credit's data doesn't state its currency, so no conversion is applied: the numbers are the dataset's own.

## Using the app
| Page | What it does |
|---|---|
| **Screening** (`app.py`) | Enter one applicant, or load one of 6 invented examples, then **Assess applicant**: PD, decision, expected loss and profit, the main reasons and a chart of what moved the score. For a decline or refer, **Draft letter**, check and edit it, then download it. Two panels under the letter show the Letter Agent's fixed input and the guardrail checks. |
| **Batch scoring** | Upload a CSV (or use the examples): rows with errors are listed by line, the rest are scored; download the results with PD, decision, reasons and expected loss. |
| **Model and fairness** | How the model was chosen (v1 vs v2), ROC and reliability charts, the profit curve with the profit and balanced-accuracy cut-offs, the fairness table with flags, and the system mapped to the course's four DDF stations. |
| **Monitoring** | PSI per input for a new batch against the training data (a normal and a shifted synthetic sample are built in), plus how the model's decisions for the batch compare with the test split. |

_Screenshots: add once deployed._

## How it works
```
applicant (form or CSV)
  -> src/applicant.py    form in years -> model fields
  -> src/validation.py   input errors; out-of-range guard (refer, never auto-approve)
  -> src/policy.py       stage 1: lending policy limits (instalments, repayment vs income, loan vs income) -> decline
  -> src/predict.py      calibrated XGBoost pipeline -> PD -> approve / refer / decline (src/threshold.py)
  -> src/explain.py      SHAP -> top 3 reasons
  -> src/llm.py          letter: template | Ollama (local) | Gemini (demo), with guardrails
offline: src/data.py -> src/features.py -> src/train.py (tuning, calibration, cut-off)
         -> src/fairness.py, src/reporting.py, src/monitoring.py -> models/ and reports/ (aggregates only)
```

The project follows the course's **Digital Data Factory (DDF)**: four stations, each accountable for its own checks (map in `src/ddf.py`, also shown on the app's Model and fairness page).

| DDF station | What we do | Checks owned here | Code |
|---|---|---|---|
| **1. ETL** (gate keepers) | Load the Home Credit applications, check and clean them, split 60/20/20 (stratified) before anything is fitted | Source and terms documented, raw data never committed, split first to prevent leakage, selection bias recorded | `src/data.py` |
| **2. Feature Engineering** | Nine credit-risk ratios, missing-value flags, training medians, one-hot encoding in one sklearn pipeline | Sex, age and family status never used as inputs; proxies analysed | `src/features.py` |
| **3. Model Design** | Compare logistic regression, random forest and XGBoost; calibrate; set the cut-off from the lender's economics; audit fairness | 0.01 AUC rule over the baseline, Brier score, four-fifths rule, model card | `src/train.py`, `threshold.py`, `fairness.py`, `metrics.py`, `reporting.py`, `model_card.py` |
| **4. Implementation** | The app: validation, scoring, SHAP reasons, the Letter Agent with guardrails, batch scoring, PSI monitoring, decision log | Unusual applicants referred, no applicant data sent to a model, analyst checks every letter, PSI alerts trigger review | `app.py`, `pages/`, `src/predict.py`, `validation.py`, `explain.py`, `llm.py`, `monitoring.py`, `audit.py` |

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
docs/                   MODEL_CARD.md (generated), MODEL_HISTORY.md, BUILD_PLAN.md (steps and decisions)
tests/                  pytest suite (fake data; no Kaggle file or API key needed)
.github/workflows/      CI: ruff, secret scan, pytest with coverage on every pull request
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

**Decision log.** Every assessment is appended to `logs/decisions.jsonl` (gitignored), one JSON line each: time, model version, inputs, PD, decision, reasons, warnings, and which backend wrote any letter. It is the audit trail a real lender would keep, so any decision can be traced and explained later. It is on by default; set `DECISION_LOG=off` to turn it off.

To retrain or run the notebook, install the dev tools (`pip install -r requirements-dev.txt`) and get the data (see **Data**). Retraining (`python -m src.train`) also regenerates the reports and the model card (`python -m src.reporting model-card` regenerates the card on its own).

**Development checks.** Tests: `pytest -q` (with coverage: `pytest -q --cov`; **94%** of `src/` is covered, and CI prints the report). Lint and format: `ruff check .` and `ruff format .`. Once per clone, run `pre-commit install`: every commit then runs ruff, a secret scan (detect-secrets) and a 50 MB file-size check.

## Deployment
Deployed on Streamlit Community Cloud from the `main` branch (entry point `app.py`, Python 3.12) with `LLM_BACKEND = "gemini"`, `GOOGLE_API_KEY` and `DECISION_LOG = "off"` in the app's Secrets settings (the public demo keeps no log of what visitors type in): the public demo drafts letters with Gemini's free tier, which is acceptable because it only handles invented example applicants and the prompt holds no applicant data. Real customer data would use the local Ollama backend instead (closed environment). No keys are stored in the code.

## Known issues and limitations
- **Policy limits are set from the data, not from a real lender's policy.** They sit just beyond the range of Home Credit's approved loans: on the test split they affect 9 of 61,503 applicants (0.015%, 8 of whom repaid), so the reported test results are unchanged at one decimal place. A real lender would set them from its responsible-lending policy and affordability assessment (NCCP Act).
- **Inputs come from a foreign lender's data.** Some fields are Home Credit's own (the region risk rating, its employer-type codes, anonymised external scores whose providers aren't named). The form explains each one, but a deployment would map them to the lender's own data, e.g. bureau scores rescaled to 0 to 1 and the region rating looked up from the address.
- **Selection bias.** Home Credit only observed repayment for loans it had already approved, so the model has never seen the applicants its earlier policy turned away. The app is scoped as a second-stage screen, and applicants outside the training range are referred to an analyst.
- **No application dates.** The data has no dates, so there is no out-of-time test, and the monitoring page uses synthetic batches built from the training statistics.
- **Assumed economics.** The cut-off and profit figures rest on stated assumptions (LGD 60%, a 15% lifetime margin, exposure = loan amount). The data doesn't state its currency; amounts are shown in USD as a display assumption.
- **Moderate recall.** At the profit-maximising cut-off the model declines 25.5% of eventual defaulters. That is the profitable trade-off under these assumptions, not a fraud filter.
- **Fairness.** Every group passes the four-fifths rule and PDs are accurate within each group, but creditworthy applicants under 30 are approved less often than those over 60 (about 87% vs 99%), mostly through the external credit scores. Accepted and documented as a policy decision, not hidden.
- **Letters.** The guardrails catch hard violations (numbers, protected attributes, judged reasons, a missing review right), but softer wording can still get through, so every model letter is a draft the analyst must check.
- **Free-tier hosting.** Gemini's free tier can hit its rate limit (the app then uses the template and says so), and Streamlit Community Cloud puts the app to sleep after inactivity, so the first visit can take a minute.

## Planned enhancements
- Add Home Credit's bureau and previous-application tables, which carry the alternative data that matters most for thin-file applicants.
- Out-of-time validation and reject inference once dated data, including declined applications, is available.
- Scheduled PSI monitoring with alerts, and comparison of predicted PDs with realised defaults as loans mature.
- Analyst log-in with roles, and a database for the decision log instead of a local file.
- A small fine-tuned local model for letters (the Week 4 "SLM-first" idea), so real customer data never leaves the lender.
- Fairness options for the policy owner to choose from, such as reviewing referrals for younger applicants by hand.

## AI tools
We used Claude Code (Anthropic) as a coding assistant for parts of the code, tests and documentation; every change was reviewed, run and tested by the team. The report's AI-use statement describes which tasks it helped with. Google Gemini is part of the app itself (the Letter Agent).

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
