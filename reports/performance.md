# Performance: caching the model in the app

**Optimisation:** `st.cache_resource` for the model, its metadata, the training reference ranges and the SHAP explainer (`get_artifacts()` in `app.py`), and `st.cache_data` for the example applicants (`get_examples()`).

**Why it's needed:** Streamlit reruns the whole of `app.py` on every interaction (each click, each form submit, loading an example). Without caching, every rerun would reload the model file from disk and rebuild the SHAP explainer. With caching, they are loaded once per server process and shared by all users and reruns.

## Measurements (28 Oct 2026)

Laptop (Windows 11, Python 3.12, CPU only), measured with Streamlit's `AppTest` (`streamlit.testing.v1`) and `time.perf_counter()`, median of 5 runs (20 for `predict()`).

| What | Time |
|---|---|
| Load the artefacts once (`load_artifacts()`: model, metadata, reference stats, SHAP explainer) | 0.16 s |
| First page load, cold cache (includes importing pandas, scikit-learn, XGBoost, SHAP) | 1.74 s |
| Page rerun **with** caching | **0.040 s** |
| Page rerun **without** caching (caches cleared before each run) | 0.157 s |
| Score one applicant: PD, decision, guard and SHAP reasons (`predict()`) | 71 ms |
| Draft a letter: template / local `llama3.2:3b` (Step 8) | < 1 ms / 15-30 s |

**Result:** caching makes each rerun about **4× faster** (0.157 s → 0.040 s). The absolute saving is modest (about 0.12 s per click) because the chosen XGBoost model is small (0.5 MB, depth-3 trees). These timings were measured with model v1; model v2 (29 Oct) is 1.2 MB with depth-5 trees, so loading it takes a little longer and caching saves slightly more. It grows with the model's size and the server's speed (Streamlit Community Cloud runs on shared CPUs), and it also stops every user's click from re-reading the model file.

The slow step a user can notice is the optional local language model letter (15-30 s), which is why the app shows a spinner with "This can take up to a minute" and falls back to the template on a 60 s timeout.
