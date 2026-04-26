# Bank Customer Churn — Exploratory Data Analysis

A modular, two-notebook EDA system for the **Churn Modelling** dataset (10 000 bank customers). The project separates reusable analysis machinery from narrative results, following a backend/frontend pattern common in production data science workflows.

---

## Repository Structure

```
.
├── eda.ipynb          # Narrative layer — results, outputs, commentary
├── eda_demo.ipynb     # Backend layer — all reusable functions and logic
└── README.md
```

### Design Philosophy

The two notebooks are intentionally split:

| Notebook | Role | What it contains |
|---|---|---|
| `eda_demo.ipynb` | **Backend / Library** | Every function definition, algorithm, and helper. No inline prose. |
| `eda.ipynb` | **Frontend / Report** | Loads the backend via `%run eda_demo.ipynb`, calls functions, and stores all rendered outputs with written commentary. |

This separation means `eda_demo.ipynb` can be imported by any downstream notebook (modelling, feature engineering, drift monitoring) without carrying along the rendered outputs from the analysis run.

---

## Dataset

**Source:** [Kaggle — Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling), loaded via `kagglehub`.

| Property | Value |
|---|---|
| Rows | 10 000 |
| Raw columns | 14 |
| Target column | `Exited` (binary: 1 = churned) |
| Churn rate | ~20.4% |
| Missing values | None |
| Duplicate rows | 0 |

### Column Classification

| Group | Columns |
|---|---|
| **Numeric** | `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary` |
| **Categorical** | `Geography`, `Gender` |
| **Boolean / Binary** | `HasCrCard`, `IsActiveMember`, `Exited` |
| **Identifiers (excluded)** | `RowNumber`, `CustomerId`, `Surname` |

---

## Notebook Walkthrough

### `eda.ipynb` — Results Layer

Cells are tagged with stable IDs so they can be referenced or re-run selectively.

| Cell ID | Section | What it does |
|---|---|---|
| `eda-load-backend` | Setup | `%run eda_demo.ipynb` — executes the backend and imports all functions. Prints balance distribution statistics as a smoke test. |
| `eda-setup-code` | Setup | Downloads the dataset via `kagglehub`, loads it into `df_model`, and defines shared column-group variables used throughout. |
| `eda-basic-viz` | Basic EDA | Calls `basic_data_visualization()` — first 5 rows, styled summary statistics, `df.info()`, target class proportions. |
| `eda-basic-quality` | Basic EDA | Calls `basic_data_quality_checks()` — duplicate rows, missing values, negative values, and categorical bar charts. |
| `eda-basic-quality` (extended) | Quality | Calls `a_bit_more_comprehensive_quality_checks()` — constant/near-constant columns, invalid-value range checks, cardinality audit. |
| `eda-anomaly-code` | Anomaly | Calls `build_summary_report()`, joins anomaly signals back onto the raw frame, runs PCA loading heatmap, top-anomaly table. |
| `eda-subgroup-code` | Subgroup | Calls `comprehensive_subgroup_analysis()` — churn rate and anomaly rate by segment, distribution plots, interaction analysis, relationship stability. |
| (drift cell) | Drift | Calls `run_drift_stability_checks()` comparing train vs test splits for PSI-based feature drift. |

### `eda_demo.ipynb` — Backend Layer

All function definitions live here. The file is structured in logical blocks:

1. **Imports** — full scientific Python stack plus ML libraries (scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Optuna, FastAPI).
2. **Column classification** — auto-detection of numeric, categorical, boolean, datetime, and identifier columns.
3. **Basic quality checks** — duplicates, missing values, negative values, range violations, cardinality.
4. **Association analysis** — numeric–numeric (Pearson + Spearman), categorical–categorical (Cramér's V + chi-square), numeric–categorical (correlation ratio + ANOVA), target associations, mutual information.
5. **Multicollinearity** — manual VIF computation, condition number, eigenvalue decomposition.
6. **Univariate outlier detection** — IQR fences, Z-score, extreme-value flags.
7. **Multivariate anomaly detection** — Isolation Forest + Local Outlier Factor (LOF), combined anomaly score, row suspicion score.
8. **Distribution checks** — normality tests, Benford's Law (first-digit and first-two-digit MAD), digit-preference / round-number heaping.
9. **Categorical quality** — rare levels, invalid labels, normalised entropy.
10. **Business rule checks** — configurable query-based or callable-based rule engine.
11. **PCA analysis** — standardised PCA, loading heatmap, scree plot, cumulative variance plot, biplot with anomaly-score colour.
12. **Subgroup analysis** — churn rate by segment, anomaly rate by segment, distribution plots, interaction tables, feature stability across groups, relationship stability.
13. **Drift / stability checks** — PSI, Wasserstein distance, KS test, target rate shift, unseen category detection between any two splits.
14. **Summary report builder** — `build_summary_report()` assembles all modules into one dict and produces all audit visualisations.

---

## Key Outputs

### Data Quality

- Zero missing values and zero duplicate rows confirmed across all 10 000 records.
- All numeric columns pass domain range validation (e.g., `CreditScore` in [350, 850], `Age` in [18, 92]).
- `Balance` is the only column flagged for Benford screening (wide enough range); first-digit MAD ≈ 0.112 — no strong evidence of fabrication.
- `NumOfProducts` has a repeated-value share of ~50.8% (values 1 and 2 dominate), flagged as near-constant by the suspicion scorer but expected for this domain.

### Anomaly Detection

Two unsupervised detectors run in parallel on the scaled numeric feature matrix:

| Detector | Contamination | Key output columns |
|---|---|---|
| Isolation Forest | 10% | `isolation_forest_score`, `isolation_forest_flag` |
| Local Outlier Factor | 10% | `lof_score`, `lof_flag` |

Both flags are combined into:
- `anomaly_vote_count` — how many detectors flagged the row (0, 1, or 2).
- `combined_anomaly_score` — weighted linear combination of normalised detector scores.
- `row_suspicion_score` — adds univariate outlier votes, rule violations, and rare/invalid categorical signals on top of the combined anomaly score.

Top suspicious rows share a consistent pattern: older customers (`Age` outlier votes), high anomaly scores from both detectors, and `Exited = 1`.

### PCA

Five principal components are extracted from the standardised numeric feature matrix.

| PC | Explained Variance | Dominant Loading |
|---|---|---|
| PC1 | 15.80% | `Balance` (−0.698), `NumOfProducts` (+0.698) — a product-vs-balance axis |
| PC2 | 13.21% | `Age` (+0.598), `Tenure` (−0.286) |
| PC3 | ~12% | `Tenure` (+0.607) |
| PC4 | ~11% | `CreditScore` (+0.955) |
| PC5 | ~10% | `Tenure` (+0.699) |

The first two PCs explain ~29% of total variance — adequate for exploratory visualisation but not a complete summary. The 80% cumulative-variance threshold requires approximately 7–8 components.

**Biplot interpretation:** `Balance` and `NumOfProducts` point in nearly opposite directions on PC1, indicating a strong negative association in this space. `Age` and `Exited` share a similar loading direction on PC2, consistent with the subgroup finding that churn rate is highest for ages 45–64. Anomalous points (light-coloured in the biplot) concentrate at the edges and within the thin `NumOfProducts >= 3` stripe visible in the upper-left quadrant.

### Subgroup Analysis

Key churn-rate findings:

| Segment | Churn Rate | Rate Ratio vs Overall |
|---|---|---|
| Age 45–54 | 50.6% | 2.48× |
| Age 55–64 | 48.3% | 2.37× |
| Geography: Germany | 32.4% | 1.59× |
| `IsActiveMember = 0` | 26.9% | 1.32× |
| Gender: Female | 25.1% | 1.23× |
| Age 25–34 | 8.5% | 0.42× |

Anomaly concentration follows a similar age pattern: the 65–99 and 55–64 age bands have the highest anomaly rates (29% and 27% respectively), roughly 3× the overall rate.

### Drift / Stability Check

PSI-based comparison of train vs test splits confirms no major distribution shift across features. Target prevalence (`Exited` rate) is stable across splits within 1–2 percentage points, consistent with a random stratified split.

---

## Environment and Dependencies

### Python Requirements

```
python >= 3.9
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
shap
optuna
scipy
fastapi
pydantic
uvicorn
joblib
kagglehub
ipywidgets
```

Install all at once:

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn \
    xgboost lightgbm catboost shap optuna scipy fastapi pydantic uvicorn \
    joblib kagglehub ipywidgets
```

### Running the Notebooks

1. Clone or download the repository.
2. Start JupyterLab or Jupyter Notebook from the project root.
3. Open `eda.ipynb` and run all cells top-to-bottom (`Kernel → Restart & Run All`).
   - `eda_demo.ipynb` does **not** need to be opened separately; it is executed automatically by the first cell of `eda.ipynb` via `%run eda_demo.ipynb`.
4. The dataset is downloaded automatically from Kaggle via `kagglehub` on first run. A Kaggle API key must be configured (`~/.kaggle/kaggle.json`).

---

## Function Reference

### Entry Points

```python
# Full audit report — runs all modules, produces all visualisations
report = build_summary_report(
    df,
    exclude_cols=None,           # list of columns to skip entirely
    invalid_value_rules=None,    # dict of domain range rules (defaults provided)
    rules=None,                  # list of business-rule dicts {name, query/callable}
    plot=True,
    thresholds=None,             # override AUDIT_DEFAULTS thresholds
)

# Basic visualisation only
basic_data_visualization(df)

# Basic quality check only
basic_data_quality_checks(df)

# Comprehensive quality check (ranges, cardinality, near-constants)
a_bit_more_comprehensive_quality_checks(df, invalid_value_rules=None)
```

### Column Classification

```python
meta = classify_columns(df, target_col="Exited", exclude_cols=["RowNumber"])
# Returns dict with keys:
#   numeric_cols, categorical_cols, boolean_cols, datetime_cols,
#   identifier_cols, excluded_cols, analysis_numeric_cols,
#   analysis_categorical_cols, target_col, target_type
```

### Association Analysis

```python
# Numeric–numeric (Pearson + Spearman matrices, ranked pairs)
results = numeric_numeric_associations(df, numeric_cols)

# Categorical–categorical (Cramér's V + chi-square p-value matrices, ranked pairs)
results = categorical_categorical_associations(df, cat_cols)

# Numeric–categorical (correlation ratio + ANOVA, pivot matrix, ranked pairs)
results = numeric_categorical_associations(df, numeric_cols, cat_cols)

# All features vs target (includes mutual information)
associations = target_associations(df, target_col, meta, include_mutual_info=True)

# Multicollinearity (VIF, condition number, eigenvalues, high-correlation pairs)
results = multicollinearity_checks(df, numeric_cols)
```

### Anomaly Detection

The multivariate anomaly module runs automatically inside `build_summary_report()`. The enriched row-level output can be extracted and joined back to the original frame:

```python
report = build_summary_report(df, ...)
row_summary = report["row_summary"]
df_enriched = df.join(row_summary, how="left")
```

Key columns added to `row_summary`:

| Column | Type | Meaning |
|---|---|---|
| `isolation_forest_score` | float | Anomaly score from Isolation Forest (higher = more anomalous) |
| `isolation_forest_flag` | bool | True if flagged by Isolation Forest |
| `lof_score` | float | LOF score (higher = more anomalous) |
| `lof_flag` | bool | True if flagged by LOF |
| `anomaly_vote_count` | int | Number of detectors that flagged this row (0–2) |
| `combined_anomaly_score` | float | Weighted combined anomaly score |
| `rule_violation_count` | int | Number of business rules violated |
| `suspicious_signal_count` | int | Total univariate + categorical warning signals |
| `row_suspicion_score` | float | Overall row-level risk score |

### Subgroup Analysis

```python
results = comprehensive_subgroup_analysis(
    df,
    target_col="Exited",
    segmentation_cols=["Gender", "Geography", "IsActiveMember", "HasCrCard"],
    anomaly_score_col="combined_anomaly_score",   # optional
    anomaly_flag_col="isolation_forest_flag",      # optional
    age_col="Age",
    min_support=30,
    numeric_feature_cols=None,   # auto-detected if None
    max_interactions=6,
)
# Returns dict with keys:
#   target_rates, anomaly_rates, stability, relationship_stability, segmentation_cols
```

### Drift Checks

```python
drift_report = run_drift_stability_checks(
    data_or_splits,              # DataFrame with a split column, or dict of DataFrames
    base_split="train",
    compare_split="test",
    target_col="Exited",
    numeric_cols=None,           # auto-detected if None
    categorical_cols=None,
    boolean_cols=None,
    bins=10,
    top_n_levels=10,
    include_plots=True,
)
# Returns: split_summary, feature_drift (PSI, Wasserstein, KS per feature),
#          categorical_level_drift, recommendations
```

### PCA Visualisation

```python
# Scree + cumulative variance chart (standalone, uses pre-computed dict)
plot_pca_variance_analysis(pca_explained_variance)
# pca_explained_variance: dict like {"PC1": 0.158, "PC2": 0.132, ...}
```

### Business Rule Engine

```python
rules = [
    {
        "name": "negative_balance",
        "query": "Balance < 0",
        "description": "Balance should never be negative.",
    },
    {
        "name": "impossible_products",
        "callable": lambda row: row["NumOfProducts"] > 4,
        "description": "NumOfProducts > 4 is outside the valid range.",
    },
]
rule_results = run_rule_checks(df, rules=rules)
# Returns: summary (per-rule failing counts and percentages),
#          row_flags (boolean DataFrame, one column per rule)
```

---

## Configuration

Key thresholds are centralised in the `AUDIT_DEFAULTS` dict inside `eda_demo.ipynb`. Override any threshold at runtime by passing a `thresholds` dict to `build_summary_report()`.

| Key | Default | Meaning |
|---|---|---|
| `missing_warn_threshold` | 0.05 | Flag columns with > 5% missing values |
| `near_constant_threshold` | 0.99 | Flag columns where the top value covers >= 99% of rows |
| `rare_level_threshold` | 0.01 | Categorical levels with < 1% share are treated as "rare" |
| `spike_threshold` | 0.80 | Top-level share above 80% triggers a categorical warning |
| `iqr_fence` | 3.0 | IQR multiplier for univariate outlier fences |
| `zscore_threshold` | 4.0 | Z-score threshold for outlier flagging |
| `anomaly_contamination` | 0.10 | Expected anomaly fraction for Isolation Forest and LOF |
| `plot_sample_size` | 2000 | Max rows sampled for heavy visualisations |
| `identifier_unique_ratio` | 0.95 | Unique-ratio threshold for auto-detecting identifier columns |

---

## Extending the Pipeline

### Adding a Custom Business Rule

```python
custom_rules = [
    {
        "name": "senior_zero_balance_churner",
        "callable": lambda row: row["Age"] > 60 and row["Balance"] == 0 and row["Exited"] == 1,
        "description": "Senior customers with zero balance who churned — possible data issue.",
    }
]
report = build_summary_report(df, rules=custom_rules)
```

### Changing Anomaly Contamination

```python
report = build_summary_report(df, thresholds={"anomaly_contamination": 0.05})
```

### Running Only the Subgroup Analysis

```python
%run eda_demo.ipynb

results = comprehensive_subgroup_analysis(
    df_model,
    target_col="Exited",
    segmentation_cols=["Geography", "Gender", "IsActiveMember"],
    anomaly_score_col="combined_anomaly_score",
)
```

### Applying the Backend to a New Dataset

The backend is dataset-agnostic. To run it on any new tabular classification dataset:

```python
%run eda_demo.ipynb

df_new = pd.read_csv("your_data.csv")

report = build_summary_report(
    df_new,
    exclude_cols=["id_column"],
    invalid_value_rules={
        "YourNumericCol": {"min": 0, "max": 1000},
    },
)
```

Column classification, outlier detection, anomaly scoring, and all visualisations will adapt automatically.

---

## Next Steps

The EDA phase establishes a clean, fully documented baseline. Recommended next steps in the modelling pipeline:

1. **Feature engineering** — age-band encoding, balance-zero indicator, interaction terms suggested by subgroup analysis (e.g., `Germany × Age > 44`).
2. **Preprocessing pipeline** — `ColumnTransformer` with `StandardScaler` for numerics and `OneHotEncoder` for `Geography` / `Gender`.
3. **Baseline model** — `DummyClassifier` (stratified) as minimum benchmark; primary metrics are ROC-AUC and Average Precision given class imbalance (~80/20).
4. **Model A** — Logistic Regression (interpretable baseline with regularisation).
5. **Model B** — Gradient Boosting (LightGBM / XGBoost / CatBoost).
6. **Model C** — Calibrated ensemble or stacking with SHAP-based feature importance.
7. **Imbalance handling** — evaluate SMOTE (already imported) vs class-weight adjustment vs threshold tuning.
8. **Hyperparameter tuning** — Optuna (already imported); use `StratifiedKFold` to preserve target balance.
9. **Model interpretation** — SHAP summary plots and partial dependence displays for the winning model.
10. **Deployment scaffold** — FastAPI endpoint skeleton is already imported in `eda_demo.ipynb`.

---

## Notes

- `EstimatedSalary` is auto-classified as an **identifier** by `classify_columns()` because its unique-value count approaches the row count. If salary should be included in modelling, pass it explicitly in `numeric_cols` or remove it from `exclude_cols`.
- 63.83% of customers have a non-zero `Balance`. The zero-balance group likely represents dormant accounts and may benefit from being treated as a separate binary indicator feature downstream.
- The Benford deviation for `Balance` (first-digit MAD ≈ 0.112) is elevated relative to the expected value under Benford's Law, but this is common for bounded financial data. It should not be interpreted as evidence of data manipulation without domain review.
- The `tqdm` warning about `IProgress not found` is benign — it occurs when `ipywidgets` is not installed or not enabled. It does not affect any output.
