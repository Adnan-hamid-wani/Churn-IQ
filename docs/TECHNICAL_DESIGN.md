# ChurnIQ technical design

## Components and build order

| Component | Responsibilities | Depends on |
| --- | --- | --- |
| Data and EDA | Pinned download, schema checks, analysis helpers, notebook | IBM sample |
| Training and evaluation | Split, preprocess, tune, select, evaluate, serialize | Data |
| Prediction and explanations | Validate single and batch inputs, score the bundle, explain a case | Model artifact, shared schema |
| Streamlit UI | Five pages, visualizations, feedback, session state, downloads | Data, metrics, prediction |
| Release | Tests, documentation, clean runtime and public smoke test | All components |

## Data contracts

Use the original IBM flat CSV. The source and checksum live in the data loader. Keep the raw CSV locally under `data/raw/`. The dashboard can obtain a verified copy at runtime when the local copy is absent. Cache the validated IBM sample and model bundle. Uploaded CSVs and their results stay scoped to the current session and are excluded from shared caches.

A single feature schema defines numeric fields, categories and their display labels. Use tenure, charges, contract, billing and service fields for the model. Gender, senior-citizen status, partner status and dependents are available for exploration. The ID is useful for file reconciliation. It and the target are excluded from model inputs. Parse numeric whitespace blanks as missing, treat infinity and bad numeric text as invalid, and validate finite nonnegative charges and whole-month tenure. Use valid service combinations in the synthetic example and the single-customer form.

## Model protocol

Use a seeded stratified 60/20/20 train, validation and test split. Tune logistic regression, a decision tree and a random forest with five-fold stratified grid search on the training split. Rank the best configurations by mean fold average precision. Use the same folds across models and a deterministic order for exact ties. Average precision is a weighted summary of the precision–recall curve. It differs from trapezoidal PR area.

The fitted pipeline contains a ColumnTransformer with median numeric imputation, most-frequent categorical imputation and one-hot encoding. Scale numeric inputs for logistic regression. Unknown categories can pass through the encoder safely. Shared input validation reports unrecognized categories in uploaded CSVs so users can fix typos.

Choose a classification threshold that maximizes F1 on validation predictions, preferring higher recall for exact ties. Keep the trained pipeline intact after setting the threshold. A lower threshold generally flags more churners and creates more false alarms. A higher threshold generally reduces false alarms and misses more churners. Report counts and metrics on the test partition after the model and threshold are fixed. Use a lower risk cutoff equal to half the classification threshold. Low is below that cutoff, Medium is between the cutoffs, and High begins at the classification threshold. These are demo labels relative to the validation-derived cutoff.

## Explanations and artifacts

Use held-out permutation importance with average precision for global inspection after model selection. Save mean and standard deviation across repeated shuffles. For a single case, score actual edits to the submitted feature vector and compare each edited prediction to the original. Replace related service fields together using real, common training profiles. Report signed differences and the replacement values. These are model responses to specific edits, with contextual limitations and possible feature dependence.

Write `models/churn_pipeline.joblib` with the fitted pipeline, threshold and explanation references. Write `models/metrics.json` with the artifact digest, CV results, validation thresholds, held-out results and permutation importance. Verify the project-supplied digest before loading Joblib and never load user-supplied model files. Bundle the small artifacts with pinned runtime dependencies. Relative paths resolve from the project root on Windows and Linux.

## Application and verification

Use Streamlit navigation with Overview as the default and native forms, filters and download controls. Use Plotly for app charts, and Matplotlib and Seaborn for the notebook. Light mode uses navy text, blue and teal data colors, consistent typography, and responsive Streamlit layout. Explicitly label charts and risk bands in text.

| Action | Command |
| --- | --- |
| Install | `python -m pip install -r requirements-dev.txt` |
| Download data | `python -m scripts.download_data` |
| Execute EDA | `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratory_data_analysis.ipynb` |
| Train | `python -m src.train` |
| Test | `python -m pytest -q` |
| Run | `python -m streamlit run app.py` |

Use descriptive names, type hints for public functions, and examples that reflect the code, such as `def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:`. Test schema errors, blank and unknown inputs, reproducibility, threshold behavior, artifact loading, and batch export. Verify all pages with app tests and desktop and mobile browser checks. Publish only from a repository that excludes raw data and credentials. Record any account action or deployment check still pending.

## References

- [IBM sample CSV and source code](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
- [Scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
