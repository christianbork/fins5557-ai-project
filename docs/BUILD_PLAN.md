# Build plan: consumer loan default screener (Home Credit)

This is the step-by-step build plan for the FINS5557 group project. Claude Code (and teammates) should work through it **one step at a time, in order**, ticking off each step when its checks pass. Read `CLAUDE.md` first for the working rules.

**Deadline:** Friday 6 November 2026, 11:59pm. **Gates (revised 28 Oct after switching data):** Home Credit pipeline retrained Thu 29 Oct, full version deployed Sun 1 Nov, code freeze Tue 3 Nov.

> **28 Oct: we switched from the UCI Taiwan data to Home Credit.** The professor confirmed we should use the best data, even if it can't go in the public repo, as long as we show where it came from. Home Credit is application data (what a lender knows when someone applies), has a realistic ~8% default rate, keeps the fairness attributes and fits the thin-file story. Steps 1 to 5 were built on Taiwan; Step 5b below redoes them on Home Credit. The training, calibration, threshold and prediction code carries over unchanged.

---

## 1. What we're building

A Streamlit app for a credit analyst at a consumer lender (cash and revolving loans). For each loan applicant it shows:

1. a **calibrated probability of default (PD)** from a machine-learning model,
2. an **approve / refer / decline** decision from a cost-based threshold,
3. the **top reasons** behind the score (SHAP),
4. a **decline letter** drafted by an LLM (with a template fallback).

Behind that sit a **fairness audit** (outcomes by gender, age band, family status and education) and **drift monitoring** (PSI on a new batch vs the training data).

### Decisions so far

| Topic | Decision |
|---|---|
| Data | **Home Credit Default Risk** (Kaggle, 2018): `application_train.csv`, 307,511 loan applications, one row each. From https://www.kaggle.com/competitions/home-credit-default-risk/data (log in, accept the competition rules, download the file into `data/raw/`). **Competition rules:** the sponsor's clause limits use to the competition and forbids passing the data to anyone who hasn't accepted the rules. Our professor approved using it for this project (28 Oct). So: `data/raw/` stays gitignored, the file is **never shared between teammates** (each teammate downloads it with their own Kaggle account after accepting the rules), and the report and README cite the source and this approval. Only `application_train.csv` is used for now (the bureau and previous-application tables could be added later). |
| Target | `TARGET` renamed `default`: 1 = payment difficulties on the loan. About 8% positive. |
| Inputs | 23 raw fields known at application, listed in `src/config.py` (`NUMERIC_INPUTS`, `CATEGORICAL_INPUTS`): income, loan amount, repayment, goods price, family size, employment, ID/registration age, car age, region rating, the three external scores (`EXT_SOURCE_1/2/3`), bureau enquiries, contract/income/education/housing/occupation/organisation type. Left out on purpose: the "defaults in social circle" fields (guilt by association) and ~50 mostly empty building statistics. |
| Protected attributes | `CODE_GENDER`, `DAYS_BIRTH` (age), `NAME_FAMILY_STATUS`: **never model inputs**, only used for the fairness audit. `NAME_EDUCATION_TYPE` stays an input but is also audited. |
| Missing values | Kept as information: numeric gaps get the training median plus a 0/1 "was missing" column; categorical gaps become "Unknown". A missing external score usually means a thin credit file. `DAYS_EMPLOYED = 365243` is Home Credit's "not employed" code and becomes a flag, not a number. |
| Population (selection bias) | **Decided (28 Oct).** Every row is a loan Home Credit already approved; applicants it rejected have no outcome, so the model has never seen them (the reject-inference problem). It may be too lenient on the weakest applicants, and some patterns reflect the lender's past choices (e.g. long loans default *less*). Proper reject inference isn't feasible: the refused applications in `previous_application.csv` lack most of our inputs. So: (1) the app is scoped as a **second-stage screen** for applicants who already pass the lender's basic policy rules, stated in the app and README; (2) the report has a limitations paragraph; (3) Step 7 refers applicants outside the training ranges instead of auto-approving; (4) Step 11 PSI monitoring flags a population unlike the training data. |
| Split | Stratified 60/20/20 train/validation/test with seed 100, done before any fitting. The test set is used **once**, for final reporting. |
| Models | Logistic regression (baseline), random forest, XGBoost. Pick on validation results. If XGBoost isn't clearly better than logistic regression, say so and justify the choice. |
| Threshold | From expected loss, not 0.5: EL = PD × LGD × EAD. Assumptions in `src/config.py`: LGD 60%, net margin over the loan's life 15%, EAD = loan amount (`AMT_CREDIT`), so break-even PD = 20%. The code picks the cut-off that maximises realised profit on validation, with a ±2-point refer band; the report compares it with the break-even PD and shows a sensitivity table (LGD 40–80%, margin 10–20%). |
| Generative AI | Only for the decline letter (and the tax-statement reader, if built). All LLM calls go through `src/llm.py` with `LLM_BACKEND` = `template` (default, no AI), `ollama` or `gemini`. **Decided (28 Oct): hybrid.** Closed environment for real customer data: `ollama` running locally, so no customer data leaves our machines (the professor leans towards closed). The public Streamlit Cloud demo can't run Ollama and only handles example applicants, so it uses `template` (or `gemini`). The report explains this choice. Build the template first, then Ollama. |
| App | Streamlit. `app.py` = screening page; extra pages in `pages/`. Deployed on Streamlit Community Cloud with `LLM_BACKEND=template`. The deployed app needs the trained model file but never the raw data; example applicants shown in the app must be invented or heavily altered, not copied rows from the Kaggle file. |
| Notebooks | **One notebook only:** `notebooks/project.ipynb`, with a section per step. Don't create new notebooks; add sections. |

### Not decided yet (don't build unless asked)

- Neural network (MLP) as a fourth model in the comparison.
- Pre-application quick screen from an uploaded tax statement (two-stage flow).
- Regime / robustness test: resample the test set to a different default rate and recalibrate (cheap), and PSI between groups.
- Adding the bureau / previous-application tables for extra features.

---

## 2. Target repo layout

```
app.py                  Screening page (Streamlit entry point, keep it thin)
pages/                  1_Batch_scoring.py, 2_Model_and_fairness.py, 3_Monitoring.py
src/
  config.py             Paths, constants, inputs, assumptions, API keys [done, Home Credit]
  data.py               Load, validate, clean, split                    [done, Home Credit]
  features.py           Engineered features + preprocessing pipeline     [done, Home Credit]
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

### Step 1: Real data and EDA [done on Taiwan; redone in Step 5b]
- [x] Download the real data: `python -c "from src.data import load_raw; print(load_raw().shape)"`. Expect `(30000, 24)`, and the file at `data/raw/uci_credit_default.csv`.
- [x] Run sections 0 and 1 of `notebooks/project.ipynb`.
- [x] Fill in the **Findings** block in section 1 (default rate, strongest signal, data issues, fairness preview) with the actual numbers.
- **Checks:** `git status` does not list the CSV. Default rate is about 22%. `clean()` leaves EDUCATION in {1,2,3,4} and MARRIAGE in {1,2,3}.

### Step 2: Features (`src/features.py`) [done on Taiwan; redone in Step 5b]
- [x] `add_features(df)`: utilisation (`BILL_AMT1 / LIMIT_BAL`), payment ratio (`PAY_AMT1 / BILL_AMT2`, the bill that September's payment settles; safe when the bill is ≤ 0), number of months late in the last 6 (`PAY_* > 0`), max months late, average utilisation over 6 months, bill trend (`BILL_AMT1 - BILL_AMT6`).
- [x] `build_preprocessor()`: a scikit-learn `ColumnTransformer`. Scale the numeric columns (for logistic regression) and one-hot encode `EDUCATION`. Fitted on training data only.
- [x] Keep the list of feature columns in `src/config.py`, so a reduced "thin-file" feature set could be added later by changing one setting.
- [x] Notebook section 2: show the new features against the default rate.
- **Checks:** tests for safe division (zero and negative bills give finite values, no NaN/inf) and for the pipeline producing the same columns on train and validation.

### Step 3: Models (`src/train.py`) [done on Taiwan; redone in Step 5b]
- [x] Train three pipelines (preprocessor + model): logistic regression (`class_weight="balanced"`), random forest (`class_weight="balanced"`), XGBoost (`scale_pos_weight` = negatives / positives). Course reference: Code Bank W05-02 for the scorecard approach, topic 02 for the models.
- [x] Light tuning with 5-fold stratified CV on the training set only (small grids, a few minutes in total; ask before anything longer).
- [x] Compare on the **validation** set: AUC, Gini (2·AUC − 1), KS, Brier score, balanced accuracy. Save a comparison table to `reports/model_comparison.csv`.
- [x] Choose the final model using the justification rule in section 1.
- [x] Run as `python -m src.train`. It saves `models/model.joblib` and `models/metadata.json` (training date, data row count, chosen model, metrics, library versions, threshold).
- [x] Notebook section 3: comparison table and ROC curves.
- **Checks:** the script runs end to end from a clean checkout plus the raw data. Test: the saved model loads and returns probabilities between 0 and 1.
- **Decided:** `models/model.joblib` (about 6 MB) and `models/*.json` are committed, because the deployed app needs them.

### Step 4: Calibration and threshold (`src/threshold.py`) [done on Taiwan; redone in Step 5b]
- [x] Calibrate the chosen model on the validation set (`CalibratedClassifierCV` with isotonic or sigmoid, whichever has the lower Brier score). Plot a reliability curve in notebook section 4.
- [x] `expected_profit(pd, threshold, lgd, margin, ead)` and `choose_threshold(...)` that picks the cut-off maximising expected profit on validation. Put the assumptions in `config.py` (e.g. `LGD = 0.60`, annual margin, EAD = credit limit).
- [x] `decide(pd) -> "approve" | "refer" | "decline"` with a refer band around the cut-off (in config). **Decided: ±2 points** (on Taiwan: ±5 would refer 37% of applicants, ±2 refers about 15%; on Home Credit ±2 refers 3.2%, see Step 5b).
- [x] Report final test-set metrics once, at the chosen threshold, including the confusion matrix.
- **Checks:** tests for `decide()` at band edges, and that the profit curve has a sensible maximum.

### Step 5: Prediction interface (`src/predict.py`) [done on Taiwan; redone in Step 5b]
- [x] `load_artifacts()` (cached by the app with `st.cache_resource`).
- [x] `predict(applicant: dict) -> {"pd": float, "decision": str, "reasons": list[str], "shap": dict}`, plus `predict_batch(df) -> DataFrame` (vectorised, no row loop).
- **Checks:** tests with a valid applicant dict, a missing field (clear error), and a batch of 100 rows.

### Step 5b: Switch to Home Credit (redo Steps 1 to 5 on the new data) [done]
Already done on the `feat/home-credit` branch: `src/config.py` (inputs, protected attributes, assumptions), `src/data.py`, `src/features.py`, the tests (47 passing on fake Home Credit-shaped data) and notebook sections 1 and 2 (old Taiwan findings removed, placeholders added). Still to do:
- [x] Get the data: accept the rules on the Kaggle page (log in, verify your phone), run `pip install -r requirements-dev.txt` (adds `kagglehub`), save a Kaggle API token to `~/.kaggle/access_token` once (command in the README Data section; `kagglehub.login()` in a terminal only lasts for that one Python process), then `python -m src.data`. It downloads only `application_train.csv` into `data/raw/` and prints the row count and default rate. Check `git status` does not list it.
- [x] Expect 307,511 applications and a default rate of about 8%.
- [x] Run notebook sections 0 to 2 and fill in both **Findings** blocks with the real numbers (default rate, external scores, missingness / thin files, affordability, fairness preview, strongest engineered features).
- [x] Retrain: `python -m src.train` (about 5 to 10 minutes; it overwrites the old Taiwan model, metadata and reports). Check the model file is under 100 MB before committing (GitHub's limit); if the random forest is too big, reduce its trees or depth.
- [x] Run notebook sections 3 and 4 and fill in their **Findings** blocks. Check the refer band share is still sensible (aim for 10–20%) and adjust `REFER_BAND` if not. **Decided (28 Oct): keep ±2.** On Home Credit it refers 3.2% (approve 92.4%, decline 4.3%); 10% would need ±7, which turns most declines into referrals. The 10–20% aim came from Taiwan's 22% default rate.
- [x] Update `README.md`: a **Data** section with the Kaggle link, the steps to get the file (own Kaggle account, accept the rules, save to `data/raw/`), that it is not in the repo and must not be shared, and that use for this course was approved by the lecturer.
- **Checks:** `pytest -q` and `ruff check .` pass; test AUC is close to validation AUC; mean test PD is close to the actual default rate.

### Step 6: Explainability (`src/explain.py`) [done]
- [x] SHAP `TreeExplainer` for tree models (or `LinearExplainer` if logistic regression wins), built once and cached. Course reference: topic 03.
- [x] `top_reasons(shap_row, k=3)` mapping features to plain-English phrases (`REASON_PHRASES`, e.g. `EXT_SOURCE_*` → "External credit scores", `credit_term` → "Length of the loan"). Phrases name the factor, not a direction, because several effects are U-shaped or reversed. `predict()` now fills `reasons` and `shap`.
- [x] Notebook section 5: global SHAP summary plot, most common decline reasons and one worked applicant.
- **Checks:** reasons only ever list features that increase risk for a decline. Test the phrase mapping covers every model feature.

### Step 7: Input validation (`src/validation.py`) [done]
- [x] `validate_applicant(dict, reference) -> list[str]`: errors naming the field, the allowed range and the value entered (`NUMERIC_RULES`: amounts ≥ 1, external scores 0–1, region rating 1–3, whole numbers for counts, repayment ≤ loan; categories must be ones seen in training). Optional fields (external scores, car age, occupation, goods price, bureau enquiries) may be blank. Only 1 of 61,502 real validation applicants fails (a missing repayment).
- [x] Out-of-range guard (selection bias, see Decisions): inputs outside the training range (from `models/reference_stats.json`) give a warning and force **refer** instead of approve; declines stay declines. **Decided (28 Oct): the full training min–max**, not the middle 99% (that flagged 7.8% of ordinary applicants, who defaulted less than average, and tripled the refer share). Flags about 0.01% of in-population applicants. `models/reference_stats.json` (ranges, deciles and category shares, no rows) is written by `python -m src.train`, or alone by `python -m src.validation`; Step 11 reuses it.
- [x] `validate_batch(df, reference) -> (valid_rows, errors)`: column and type checks, row-level errors, and `read_csv_upload()` with limits in config (`MAX_BATCH_ROWS` 10,000, `MAX_UPLOAD_MB` 5). 10,000 rows validate in about 0.4 s.
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
- [ ] Instructions state the scope: a second-stage screen for applicants who already pass the lender's basic policy rules (see Decisions: selection bias).
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
- [ ] Ship a small "shifted" sample batch for the demo (e.g. weaker external scores, shorter employment, younger customers): weaker than the approved population the model was trained on, to show monitoring catching the selection-bias risk.
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
