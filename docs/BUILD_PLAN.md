# Build plan: credit card default screener

This is the step-by-step build plan for the FINS5557 group project. Claude Code (and teammates) should work through it **one step at a time, in order**, ticking off each step when its checks pass. Read `CLAUDE.md` first for the working rules.

**Deadline:** Friday 6 November 2026, 11:59pm. **Gates:** basic app deployed Tue 27 Oct, full version deployed Fri 30 Oct, code freeze Mon 2 Nov.

---

## 1. What we're building

A Streamlit app for a credit analyst at a credit card lender. For each applicant it shows:

1. a **calibrated probability of default (PD)** from a machine-learning model,
2. an **approve / refer / decline** decision from a cost-based threshold,
3. the **top reasons** behind the score (SHAP),
4. a **decline letter** drafted by an LLM (with a template fallback).

Behind that sit a **fairness audit** (outcomes by sex, age band and education) and **drift monitoring** (PSI on a new batch vs the training data).

### Decisions so far

| Topic | Decision |
|---|---|
| Data | UCI "Default of Credit Card Clients" (Taiwan, 2005), 30,000 rows, CC BY 4.0. Downloaded by `src/data.py`, saved to `data/raw/` (gitignored). |
| Target | `default` (1 = defaulted next month), about 22% positive. |
| Protected attributes | `SEX`, `MARRIAGE`, `AGE`: **never model inputs**, only used for the fairness audit. `EDUCATION` stays an input but is also audited. |
| Split | Stratified 60/20/20 train/validation/test with seed 100, done before any fitting. The test set is used **once**, for final reporting. |
| Models | Logistic regression (baseline), random forest, XGBoost. Pick on validation results. If XGBoost isn't clearly better than logistic regression, say so and justify the choice. |
| Threshold | From expected loss, not 0.5: EL = PD × LGD × EAD. Choose the cut-off that maximises expected profit, with a "refer" band around it. Assumptions (LGD, margin) live in `src/config.py` and are stated in the report. |
| Generative AI | Only for the decline letter. All LLM calls go through `src/llm.py` with `LLM_BACKEND` = `template` (default, no AI), `gemini` or `ollama`. Build the template first. Gemini vs local Ollama is **undecided** (waiting on the professor). |
| App | Streamlit. `app.py` = screening page; extra pages in `pages/`. Deployed on Streamlit Community Cloud. |
| Notebooks | **One notebook only:** `notebooks/project.ipynb`, with a section per step. Don't create new notebooks; add sections. |

### Not decided yet (don't build unless asked)

- Neural network (MLP) as a fourth model in the comparison.
- Pre-application quick screen from an uploaded tax statement (two-stage flow).
- Running the LLM locally with Ollama (closed environment).

---

## 2. Target repo layout

```
app.py                  Screening page (Streamlit entry point, keep it thin)
pages/                  1_Batch_scoring.py, 2_Model_and_fairness.py, 3_Monitoring.py
src/
  config.py             Paths, constants, assumptions, API keys        [done]
  data.py               Download, validate, clean, split                [done]
  features.py           Engineered features + preprocessing pipeline
  train.py              Train and compare models, calibrate, save artefacts
  threshold.py          Expected-loss threshold and decision bands
  predict.py            predict(applicant: dict) -> dict  (used by the app)
  explain.py            SHAP explainer + plain-English reasons
  fairness.py           Group metrics, disparate-impact ratio
  monitoring.py         PSI per feature
  validation.py         Input and CSV checks with clear error messages
  llm.py                Decline letter: template / gemini / ollama
models/                 model.joblib, metadata.json, reference_stats.json
notebooks/project.ipynb The one notebook
tests/                  One test file per src module
docs/BUILD_PLAN.md      This file
.github/workflows/ci.yml
```

---

## 3. Build steps

Each step lists what to build and **checks** that must pass before ticking it off. After each step: run `pytest -q` and `ruff check .`, explain the changes to Christian in plain language (with the finance reasoning), and suggest a commit message.

### Step 0: Setup [done]
- [x] Repo, `.venv` (Python 3.12), packages installed, pinned `requirements.txt` + `requirements-dev.txt`.
- [x] `src/config.py`, `src/data.py`, `tests/test_data.py` (6 tests), `pyproject.toml`.
- [x] `notebooks/project.ipynb` with setup and data sections.

### Step 1: Real data and EDA [done]
- [x] Download the real data: `python -c "from src.data import load_raw; print(load_raw().shape)"`. Expect `(30000, 24)`, and the file at `data/raw/uci_credit_default.csv`.
- [x] Run sections 0 and 1 of `notebooks/project.ipynb`.
- [x] Fill in the **Findings** block in section 1 (default rate, strongest signal, data issues, fairness preview) with the actual numbers.
- **Checks:** `git status` does not list the CSV. Default rate is about 22%. `clean()` leaves EDUCATION in {1,2,3,4} and MARRIAGE in {1,2,3}.

### Step 2: Features (`src/features.py`) [done]
- [x] `add_features(df)`: utilisation (`BILL_AMT1 / LIMIT_BAL`), payment ratio (`PAY_AMT1 / BILL_AMT2`, the bill that September's payment settles; safe when the bill is ≤ 0), number of months late in the last 6 (`PAY_* > 0`), max months late, average utilisation over 6 months, bill trend (`BILL_AMT1 - BILL_AMT6`).
- [x] `build_preprocessor()`: a scikit-learn `ColumnTransformer`. Scale the numeric columns (for logistic regression) and one-hot encode `EDUCATION`. Fitted on training data only.
- [x] Keep the list of feature columns in `src/config.py`, so a reduced "thin-file" feature set could be added later by changing one setting.
- [x] Notebook section 2: show the new features against the default rate.
- **Checks:** tests for safe division (zero and negative bills give finite values, no NaN/inf) and for the pipeline producing the same columns on train and validation.

### Step 3: Models (`src/train.py`)
- [ ] Train three pipelines (preprocessor + model): logistic regression (`class_weight="balanced"`), random forest (`class_weight="balanced"`), XGBoost (`scale_pos_weight` = negatives / positives). Course reference: Code Bank W05-02 for the scorecard approach, topic 02 for the models.
- [ ] Light tuning with 5-fold stratified CV on the training set only (small grids, a few minutes in total; ask before anything longer).
- [ ] Compare on the **validation** set: AUC, Gini (2·AUC − 1), KS, Brier score, balanced accuracy. Save a comparison table to `reports/model_comparison.csv`.
- [ ] Choose the final model using the justification rule in section 1.
- [ ] Run as `python -m src.train`. It saves `models/model.joblib` and `models/metadata.json` (training date, data row count, chosen model, metrics, library versions, threshold).
- [ ] Notebook section 3: comparison table and ROC curves.
- **Checks:** the script runs end to end from a clean checkout plus the raw data. Test: the saved model loads and returns probabilities between 0 and 1.
- **Decision for Christian:** whether `models/model.joblib` gets committed (needed for deployment, small file). If yes, add `!models/model.joblib` and `!models/*.json` to `.gitignore`.

### Step 4: Calibration and threshold (`src/threshold.py`)
- [ ] Calibrate the chosen model on the validation set (`CalibratedClassifierCV` with isotonic or sigmoid, whichever has the lower Brier score). Plot a reliability curve in notebook section 4.
- [ ] `expected_profit(pd, threshold, lgd, margin, ead)` and `choose_threshold(...)` that picks the cut-off maximising expected profit on validation. Put the assumptions in `config.py` (e.g. `LGD = 0.60`, annual margin, EAD = credit limit).
- [ ] `decide(pd) -> "approve" | "refer" | "decline"` with a refer band around the cut-off (e.g. ±5 percentage points, in config).
- [ ] Report final test-set metrics once, at the chosen threshold, including the confusion matrix.
- **Checks:** tests for `decide()` at band edges, and that the profit curve has a sensible maximum.

### Step 5: Prediction interface (`src/predict.py`)
- [ ] `load_artifacts()` (cached by the app with `st.cache_resource`).
- [ ] `predict(applicant: dict) -> {"pd": float, "decision": str, "reasons": list[str], "shap": dict}`, plus `predict_batch(df) -> DataFrame` (vectorised, no row loop).
- **Checks:** tests with a valid applicant dict, a missing field (clear error), and a batch of 100 rows.

### Step 6: Explainability (`src/explain.py`)
- [ ] SHAP `TreeExplainer` for tree models (or `LinearExplainer` if logistic regression wins), built once and cached. Course reference: topic 03.
- [ ] `top_reasons(shap_row, k=3)` mapping features to plain-English phrases (e.g. `PAY_0` → "Recent payment was late", `utilisation` → "High use of credit limit").
- [ ] Notebook section 5: global SHAP summary plot and one worked applicant.
- **Checks:** reasons only ever list features that increase risk for a decline. Test the phrase mapping covers every model feature.

### Step 7: Input validation (`src/validation.py`)
- [ ] `validate_applicant(dict) -> list[str]`: errors naming the field and the allowed range (age 18–100, `LIMIT_BAL` > 0, valid codes for EDUCATION, PAY_*, etc.).
- [ ] `validate_batch(df) -> (valid_rows, errors)`: column and type checks, row-level errors, a maximum file size and row count.
- **Checks:** tests for each rule and for a CSV with a missing column.

### Step 8: Decline letter (`src/llm.py`)
- [ ] `draft_letter(decision, reasons) -> str` that switches on `LLM_BACKEND`.
- [ ] Build `template` first: a fixed, well-written letter that fills in the reasons.
- [ ] Then the AI backends behind the same function: `gemini` (consider the newer `google-genai` package; `google-generativeai` is the old one) and/or `ollama`. Ask Christian before installing either package.
- [ ] Guardrails: the prompt gets only the decision and the reasons, never protected attributes or the raw PD. A timeout. On any error, fall back to the template and say so in the UI.
- **Checks:** tests that the template works with no key, and that a failing backend falls back to the template.

### Step 9: Screening page (`app.py`) and first deployment
- [ ] Form (`st.form`) with grouped inputs, help text and a "Load example applicant" button (examples from a small committed sample file, e.g. `data/sample/sample_applicants.csv`; add `!data/sample/` to `.gitignore`).
- [ ] On submit: validate → spinner → PD, decision badge, SHAP waterfall, top reasons → "Draft letter" button.
- [ ] States: `st.spinner` while working, `st.success` / `st.warning` / `st.error` for outcomes and problems.
- [ ] Performance optimisation: cache the model and explainer with `st.cache_resource`, the sample data with `st.cache_data`. Time the first vs later loads with `time.perf_counter()` and record the numbers in `reports/performance.md` for the report.
- [ ] Deploy to Streamlit Community Cloud (Python 3.12, entry point `app.py`). Add the live URL to the README.
- **Checks:** the app works locally and on the public URL in a private browser window. Invalid input shows an error instead of crashing.

### Step 10: Fairness (`src/fairness.py`) and the Model page
- [ ] Group metrics on the test set by sex, age band and education: approval rate, true positive rate, false positive rate. Disparate-impact ratio (approval rate vs the most-approved group), flagged below 0.8. Course reference: topic 04.
- [ ] `pages/2_Model_and_fairness.py`: model comparison table, ROC and calibration plots, threshold and profit curve, fairness table with flags.
- **Checks:** tests on a small hand-made example where the answer is known.

### Step 11: Monitoring (`src/monitoring.py`)
- [ ] Save training reference statistics (deciles per feature) to `models/reference_stats.json` in `train.py`.
- [ ] `compute_psi(expected, actual, bins=10)` per feature: < 0.1 stable, 0.1–0.25 watch, > 0.25 alert. Course reference: Code Bank W02-10, W02-14, W05-07.
- [ ] Ship a small "shifted" sample batch for the demo (e.g. higher utilisation, younger customers).
- [ ] `pages/3_Monitoring.py`: upload a batch (or use the sample), show a PSI table and bar chart with traffic-light colours.
- **Checks:** PSI of identical distributions is 0; the shifted sample triggers at least one alert.

### Step 12: Batch scoring page
- [ ] `pages/1_Batch_scoring.py`: upload CSV → validate → score → preview → download results.
- **Checks:** a bad CSV shows row-level errors; valid rows still score.

### Step 13: CI and code quality
- [ ] `.github/workflows/ci.yml` that runs `pip install -r requirements-dev.txt`, `ruff check .` and `pytest -q` on every pull request (Python 3.12). CI must not need the raw data or an API key: use small fixtures.
- [ ] CI badge in the README.
- **Checks:** green tick on a pull request.

### Step 14: Finish
- [ ] README complete: overview, live URL, video link, screenshots, setup, architecture, team, data citation, licence.
- [ ] Fresh clone + install + run works.
- [ ] Repo made public before submission.

---

## 4. Conventions

- **Code lives in `src/`**, and the notebook and app import it. Never copy logic between the notebook and the app.
- **One notebook** (`notebooks/project.ipynb`). Clear outputs before committing.
- **Settings and assumptions** go in `src/config.py`, not hardcoded in modules.
- **Branches and pull requests:** `feat/...`, `fix/...`, merged after a teammate reviews. Small commits with messages like `feat: add PSI monitoring`.
- **Never commit** `.env`, `.streamlit/secrets.toml` or `data/raw/`.
- **Ask before** installing packages or running long training jobs. New packages go into the right requirements file, pinned.
- **Course code:** check the Code Bank first and name the snippet ID in a comment when adapting it.
