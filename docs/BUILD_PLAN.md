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
| Currency | **Decided (29 Oct): amounts are displayed in USD as a stated assumption.** Home Credit's data doesn't say which currency it uses; we show USD for readability and say so once on every page that shows money (a footnote), in the README and in the report. No conversion is applied: the numbers are the dataset's own. |
| App | Streamlit. `app.py` = screening page; extra pages in `pages/`. Deployed on Streamlit Community Cloud (fins5557-loan-screener.streamlit.app) with `LLM_BACKEND=gemini` (decided 29 Oct, see Step 8). The deployed app needs the trained model file but never the raw data; example applicants shown in the app must be invented or heavily altered, not copied rows from the Kaggle file. |
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

### Step 5c: Model v2: proper tuning and class-imbalance handling (keep v1 for comparison) [done]
**Why (29 Oct):** we compared our training with the Big Data exam (Problem 2, `Q2_validation.R`). Both avoid accuracy and handle the imbalance. But the exam **tuned** its imbalance correction (a `sampsize` grid for the class-balanced random forest) and tuned `mtry` by looping over a grid, while v1 fixed its class weights and used very small grids (forest `max_features` not tuned, XGBoost fixed at 400 trees). The exam also chose its threshold by **balanced accuracy** and reported TPR, TNR, precision and F1, while we choose by **profit**. v2 closes these gaps and keeps v1 so the report can show the improvement.

- [x] **Preserve v1 first** (before any retraining):
    - Annotated tags, pushed to GitHub: `v0-taiwan` on commit `5bc416d` (Taiwan pipeline, random forest, test AUC 0.786) and `v1-home-credit-baseline` on commit `d70cbd8` (Home Credit, XGBoost, test AUC 0.756, screening page built). Use `git tag -a <name> <commit> -m "<one line>"`, then `git push origin <name>`.
    - Copy the v1 artefacts into `archive/v1-home-credit-baseline/`: `models/metadata.json`, `models/reference_stats.json`, `models/model.joblib` (0.5 MB), and `reports/model_comparison.csv`, `profit_curve.csv`, `test_metrics.json`. Commit them as they are.
    - Create `docs/MODEL_HISTORY.md`: one row per version (v0, v1, then v2), with date, data, chosen model, how it was tuned, imbalance handling, validation/test AUC, Brier, cut-off, approval rate, default rate among approved, and a "what changed and why" column.
- [x] **Tuning** (`src/train.py`, settings in `src/config.py`): replace the fixed grids with `RandomizedSearchCV` (`N_ITER` in config, about 30 per tree model; logistic regression can stay a grid), still 3-fold stratified CV on the training split only, `random_state` fixed. Use multi-metric scoring `{"roc_auc", "average_precision"}` with `refit="roc_auc"`. Search spaces:
    - XGBoost: `max_depth` 2-6, `learning_rate` 0.02-0.2 (log-uniform), `n_estimators` 200-800, `min_child_weight` 1-20, `subsample` 0.6-1.0, `colsample_bytree` 0.5-1.0, and **`scale_pos_weight` in {1, sqrt(neg/pos), neg/pos}**: tuning the imbalance correction, as the exam tuned `sampsize`.
    - Random forest: **`max_features` in {"sqrt", 0.2, 0.33, 0.5}** (the exam's `mtry`), `max_depth` in {8, 12, 16, None}, `min_samples_leaf` in {20, 50, 100, 200}, **`class_weight` in {None, "balanced", "balanced_subsample"}**. Keep `n_estimators` 200 and `max_samples` 0.3 for speed, and keep the model file under 100 MB.
    - Logistic regression: `C` from 0.001 to 10 (log scale), `class_weight` in {None, "balanced"}.
    - Save every candidate's CV scores to `reports/tuning_results.csv` and the best settings to the metadata. **Ask Christian before running** (expect 15-30 minutes).
- [x] **Unchanged:** the model-choice rule (validation AUC with the 0.01 margin), calibration (best of sigmoid/isotonic on out-of-fold Brier score) and the profit-maximising cut-off. Calibration means a model trained without class weights is fine to use; record which won and say so.
- [x] **Imbalance metrics at the cut-off** (`src/metrics.py`, `src/threshold.py`): add TPR (recall), TNR (specificity), precision, F1 and PR-AUC (average precision) to the model comparison and the test report; keep the confusion matrix.
- [x] **Threshold comparison:** on the validation out-of-fold calibrated PDs, also find the cut-off that maximises **balanced accuracy** (the exam's method, Lecture 2 slide 57). Report both cut-offs side by side on the test split (cut-off, approval rate, TPR, TNR, precision, balanced accuracy, realised profit) in `reports/threshold_comparison.csv`. The app keeps the profit cut-off. Explain in the notebook: at an 8% default rate, balanced accuracy implicitly treats a missed defaulter as costing about 11 times a wrongly declined customer, while our economics (LGD 60%, margin 15%) say about 4 times.
- [x] **Notebook:** section 3 gets a "v1 vs v2" table and a short summary of the tuning results (which settings mattered, whether class weighting helped after calibration); section 4 gets the threshold comparison. Update the Findings blocks.
- [x] **Knock-on checks:** regenerate `models/reference_stats.json`; re-check that the six invented example applicants still land on their intended decisions (adjust the examples if not, keeping them invented); SHAP reasons and the app still work. Fill in the v2 row of `docs/MODEL_HISTORY.md`.
- **Checks:** `pytest -q` and `ruff check .` pass; CV, validation and test AUC are close; mean test PD is close to the default rate; model file under 100 MB; v1 artefacts in `archive/` unchanged.
- **Results (29 Oct, 25 min run):** all checks pass (133 tests; CV / validation / test AUC 0.762 / 0.757 / 0.757; mean test PD 8.2% vs 8.1% actual; model 1.2 MB; `archive/` unchanged; `reference_stats.json` regenerated identical; all six examples still on their decisions). XGBoost chosen again, with partial class weighting (`scale_pos_weight` 3.4). Tuning barely moved anything (test AUC 0.756 → 0.757, PR-AUC 0.231 → 0.236); class weighting made no difference after calibration (every setting within one CV standard deviation). The balanced-accuracy cut-off (8.7%) approves 68% and earns 12% less than the profit cut-off (21.0%). Details in notebook sections 3.2, 3.3 and 4.5 and `docs/MODEL_HISTORY.md`.
- **One deviation:** the model comparison reports TPR / TNR / precision / F1 at each model's own balanced-accuracy cut-off, not at 0.5, because an unweighted model's raw scores sit around 8% and would look as if they caught no defaulters at 0.5.

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

### Step 8: Decline letter (`src/llm.py`) [done]
- [x] `draft_letter(decision, reasons) -> {"text", "backend", "warning"}` that switches on `LLM_BACKEND`, for decline and refer letters (Code Bank W06-03).
- [x] Build `template` first: a fixed, well-written letter that fills in the reasons.
- [x] Then the AI backends behind the same function: `ollama` built with the standard library (no new package), tested on `llama3.2:3b` (about 15–30 s per letter on a laptop CPU). `gemini` added 29 Oct (`google-genai`, free tier) for the public demo, because Streamlit Cloud can't run Ollama and the demo should still show an LLM letter; same guardrails, 60 s timeout, no automatic retries, template fallback with a plain-English reason. **Decided (29 Oct): Gemini's free tier** (no card needed; prompts contain no applicant data). Tested on 6 letters: `gemini-3.1-flash-lite` passed all 6 (4-14 s), `gemini-3.5-flash` lost 2 to overload errors (503), `gemini-2.5-flash` no longer serves requests. So flash-lite is the main model and 3.5-flash the backup, tried once on 503/429 before the template. Claude or OpenAI would also work behind the same function but need paid API credit.
- [x] Guardrails: the prompt gets only the decision and the reasons (only phrases from `REASON_PHRASES`, so no free text or prompt injection), never protected attributes or the raw PD; refer prompts get no reasons. A timeout. On any error, fall back to the template and say so in the UI. Output checks reject letters with numbers, protected-attribute words, judged reasons ("too long") or no mention of the review right. **Decided (28 Oct): a model letter is a draft the analyst checks and edits, never sent automatically**: in testing, 1 in 6 letters judged a reason and was replaced, and softer wording still got through.
- **Checks:** tests that the template works with no key, and that a failing backend falls back to the template.

### Step 9: Screening page (`app.py`) and first deployment
- [x] Form (`st.form`) with grouped inputs, help text and a "Load example applicant" button (examples from a small committed sample file, e.g. `data/sample/sample_applicants.csv`; add `!data/sample/` to `.gitignore`). Done: 6 invented examples (approve, refer, decline, thin file, pensioner, out-of-range), each checked to land on its decision; the form asks in years and `src/applicant.py` converts to the model's day fields.
- [x] On submit: validate → spinner → PD, decision badge, SHAP waterfall, top reasons, out-of-range warnings → "Draft letter" button (decline/refer only) showing the letter in an editable text box labelled as a draft for the analyst to check, plus any fallback warning.
- [x] Instructions state the scope: a second-stage screen for applicants who already pass the lender's basic policy rules (see Decisions: selection bias).
- [x] States: `st.spinner` while working, `st.success` / `st.warning` / `st.error` for outcomes and problems.
- [x] Performance optimisation: cache the model and explainer with `st.cache_resource`, the sample data with `st.cache_data`. Time the first vs later loads with `time.perf_counter()` and record the numbers in `reports/performance.md` for the report. Result: reruns 0.157 s → 0.040 s (4× faster).
- [ ] Deploy to Streamlit Community Cloud (Python 3.12, entry point `app.py`). Add the live URL to the README.
- **Checks:** the app works locally and on the public URL in a private browser window. Invalid input shows an error instead of crashing. Local checks pass (`tests/test_app.py`: every example, the letter flow and an invalid input; also run in a clean environment with only `requirements.txt`); the public URL check is still to do after deployment.

### Step 10: Fairness (`src/fairness.py`) and the Model page [done]
- [x] Group metrics on the test set by sex, age band, family status and education: approval rate, good customers approved, defaulters approved, mean PD vs default rate. Disparate-impact ratio (approval rate vs the most-approved group), flagged below 0.8; parity and equalised-odds gaps flagged above 0.05 (Code Bank W04-06). Groups under 100 applicants are not judged, and their rates are removed from the published `reports/fairness.csv` (a rate over one or a few people is their own data). `src/reporting.py` (`python -m src.reporting`, also run by `src.train`) writes the fairness table and the chart data as aggregates, because the deployed app has no raw data. Notebook section 6 adds a proxy analysis.
- [x] `pages/2_Model_and_fairness.py`: model comparison table (and v1 vs v2), ROC and calibration plots, profit curve with both cut-offs, fairness table with flags.
- **Checks:** tests on a small hand-made example where the answer is known.
- **Results (29 Oct):** no group below the four-fifths rule (lowest: under-30s, 0.85); PDs accurate within every group (mean PD within 1.2 points of the default rate); but creditworthy under-30s are approved 86.9% vs 98.8% for over-60s. Most of the age effect comes through the external credit scores. Policy options are in the notebook Findings. **Decided (29 Oct): accept and document** (decisions follow accurately measured risk; the age-linked inputs have a clear credit rationale); the model is not changed.

### Step 11: Monitoring (`src/monitoring.py`) [done]
- [x] Save training reference statistics (deciles per feature) to `models/reference_stats.json` in `train.py`. Done: each numeric input also stores PSI bins (decile edges with ties removed, plus bins for missing and the "not employed" code) and their training shares; regenerate alone with `python -m src.validation`.
- [x] `compute_psi(expected, actual)` per feature (`psi_table()` for a whole batch; categories under 1% of training pooled, as the encoder pools them): < 0.1 stable, 0.1–0.25 watch, > 0.25 alert. Course reference: Code Bank W02-10, W02-14, W05-07.
- [x] Ship a small "shifted" sample batch for the demo (e.g. weaker external scores, shorter employment, younger customers): weaker than the approved population the model was trained on, to show monitoring catching the selection-bias risk. Done: `data/sample/monitoring_normal.csv` and `monitoring_shifted.csv`, 1,000 **synthetic** applicants each, drawn from the training summary statistics (`python -m src.monitoring`).
- [x] `pages/3_Monitoring.py`: upload a batch (or use the sample), show a PSI table and bar chart with traffic-light colours, a drill-down per input, and the batch's mean PD and decision mix against the test split.
- **Checks:** PSI of identical distributions is 0; the shifted sample triggers at least one alert.
- **Results (29 Oct):** both checks pass. Validation and test splits are stable on all 23 inputs (largest PSI 0.0005); a 1,000-applicant test sample reaches only 0.023. The normal synthetic batch is stable; the shifted one raises 8 alerts and lifts the mean PD from 7.6% to 11.3%. Limits of PSI (one input at a time, no view of accuracy) are in notebook section 7.

### Step 12: Batch scoring page [done]
- [x] `pages/1_Batch_scoring.py`: upload CSV → validate → score → preview → download results. Done: also the 6 invented examples as a demo file, an empty template and the example file to download; rows with errors listed by line in the file (and downloadable); results show PD, decision, top reasons for refers and declines (`predict_batch(..., with_reasons=True)`, SHAP only on those rows), guard warnings and expected loss. 10,000 rows score in 0.35 s, or 3 s with reasons.
- **Checks:** a bad CSV shows row-level errors; valid rows still score. Both pass (`tests/test_app.py`).

### Step 13: CI and code quality [done]
- [x] `.github/workflows/ci.yml` that runs `pip install -r requirements-dev.txt`, `ruff check .` and `pytest -q` on every pull request (Python 3.12). CI must not need the raw data or an API key: use small fixtures.
- [x] CI badge in the README.
- **Checks:** green tick on a pull request. Passed on PR #12 (29 Oct, 183 passed).

### Step 13b: Engineering polish (from the 29 Oct project review) [done]
**Why:** the review ("Project Review: Loan Default Screener", section 3) found the codebase well structured and tested, but missing the extras that make a repo look professionally run, and found formatting and labels handled differently on every page. This step is code and engineering only; the visual redesign (theme, navigation, charts) comes later. Work on branch `feat/engineering-polish`, small commits, tests and ruff after each part.

- [x] **Formatting module** `src/formatting.py`, used by every page, table, chart label and CSV export (no ad-hoc f-string formats left in `app.py` or `pages/`):
    - `money(x)` -> `"$450,000"`; `money_short(x)` -> `"$3.62bn"`, `"$128.4k"`; negatives as `"-$92,951"`. Currency symbol and code in `config.py` (`CURRENCY_SYMBOL = "$"`, `CURRENCY_CODE = "USD"`), plus `CURRENCY_NOTE` = "Amounts in USD (display assumption: Home Credit's data doesn't state its currency)."
    - `pct(x)` one decimal everywhere (`"0.5%"`, `"47.5%"`); `pct_points(x)` for gaps (`"+4.3 pts"`); `ratio(x)` two decimals; `score(x)` three decimals (AUC, Gini, KS, Brier).
    - `count(n, noun)` -> `"1 applicant"`, `"6 applicants"` (no "(s)").
    - `missing()` -> en dash for empty values; tables never show `None` or `nan`.
    - Unit tests for each function, including negatives, zero, NaN and very large numbers.
- [x] **Labels module** `src/labels.py`: one source for display text:
    - Decisions always capitalised: `Approve`, `Refer`, `Decline` (badges, tables, filter chips, batch CSV, monitoring). Keep the lower-case codes internally.
    - Model display names (`Logistic regression`, `Random forest`, `XGBoost`) used in legends, tables and the model-choice sentence (no `xgboost has the highest...`; build that sentence from display names, and fix `choose_model`'s reason text too).
    - Column headers for every table shown in the app (no `rule`, `group`, `example`, `pd`): e.g. `Rule`, `Group`, `Example applicant`, `PD`.
    - Fairness attribute and group names, monitoring status words (`Stable`, `Watch`, `Alert`).
    - A test that walks the app's tables and checks no raw column names or lower-case decisions are displayed (or at least that every displayed column has a label).
- [x] **Apply both modules** across `app.py` and `pages/`, and add the `CURRENCY_NOTE` as a caption on every page that shows money (Screening, Batch scoring, Model and fairness). Spell out PD and LGD at first use on each page.
- [x] **Friendly errors and logging:** never show raw exception text to users (`Details: {e}` in `app.py` and the batch page). Log the detail with Python's `logging` (module-level loggers, configured once) and show a short, actionable message. Same for scoring and letter failures.
- [x] **Decision log** `src/audit.py`: append one JSON line per assessed applicant to `logs/decisions.jsonl` (gitignored): timestamp, model version (`trained_at` + chosen model), inputs, PD, decision, reasons, warnings, letter backend. Controlled by `DECISION_LOG` in config (on locally, **off** on the public demo via Secrets). Tests with a temp path. Mention in the README as the audit trail a real lender would keep.
- [x] **Model card** `docs/MODEL_CARD.md`, generated by `python -m src.reporting model-card` (or a small script) from `models/metadata.json` and `reports/`: intended use and scope (second-stage screen), data and terms, inputs and excluded fields, protected attributes, training and tuning, metrics (validation and test), calibration, cut-off and assumptions, fairness summary, limitations, version history link, owners. Regenerated by `python -m src.train`. Link it from the README.
- [x] **Quality tooling** (ask Christian before installing; pin new packages in `requirements-dev.txt`):
    - `pytest-cov`: CI prints coverage; add the figure (or a badge) to the README. Aim for 85%+ on `src/`.
    - `pre-commit` with ruff (lint + format) and a secret scanner (e.g. `detect-secrets` or gitleaks); document `pre-commit install` in the README.
    - Optional: `mypy` on `src/` in CI (non-blocking at first). Skipped for now: mostly noise on pandas code.
- [x] **Repo hygiene:** README gets the live URL, the currency note, a coverage line and links to the model card and model history; fix any stale references (e.g. `LLM_BACKEND=template` for the public demo). Check `.gitignore` covers `logs/`.
- **Checks:** `pytest -q` and `ruff check .` pass in CI; coverage reported; no `None`/`nan`, lower-case decisions or raw column names visible on any page (click through all four pages locally); every money value shows `$`; the deployed app still works after merging.

### Step 13c: Final polish from the 4 Nov project review [done]
**Why:** the review against the brief, the rubric and the lecturer's in-class advice asked for the course's DDF framing, a "You are a ... Agent" prompt in the course style, and the fixed LLM input made visible for the demo video. Branch `feat/final-polish`.
- [x] Letter prompt rewritten as a role-based **Letter Agent** brief: ROLE, TASK, RULES, OUTPUT FORMAT, then INPUT (Code Bank W06-05, W04-07). Same rules and guardrails as before.
- [x] `run_checks()` returns each of the six guardrail checks; `draft_letter()` also returns the prompt sent and the checks.
- [x] Screening page: two panels under the letter, the agent's fixed input and the guardrail results.
- [x] `src/ddf.py`: the codebase mapped to the four DDF stations (ETL, Feature Engineering, Model Design, Implementation; previously numbered 1-5 in docstrings), shown as section 4 of the Model and fairness page and in the README. `tests/test_ddf.py` fails if a module has no station.
- [x] README: Letter Agent, DDF table, known issues and limitations, planned enhancements, AI tools.
- **Checks:** 305 tests pass, ruff clean. Gemini couldn't be reached from the test sandbox, so the live letter check is done on the deployed app.

### Step 14: Finish
- [ ] README complete: overview, live URL, video link, screenshots, setup, architecture, team, data citation, licence. Done 29 Oct except the live URL, video link, screenshots and teammates' names (placeholders).
- [x] Fresh clone + install + run works. Checked 29 Oct: a fresh clone of `main` with only `requirements.txt` runs all four pages without errors.
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
