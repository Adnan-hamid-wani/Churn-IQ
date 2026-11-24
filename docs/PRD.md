# ChurnIQ product requirements

## Purpose

ChurnIQ is a public portfolio demo for exploring customer churn in IBM's fictional telecom dataset. Admissions reviewers should be able to explore the data, score a fictional customer, and inspect the model evidence. The project owner should be able to reproduce the training run and explain its decisions in an interview.

## Release scope

| Page | User needs | Acceptance criteria |
| --- | --- | --- |
| Overview | Understand the sample at a glance | Show customer count, churn rate, average monthly charges and tenure, churn distribution, contract-level churn rates and observations computed from source data. |
| Data Explorer | Investigate associations and inspect records | Filter by contract, internet service and churn outcome. Plot distributions by outcome, show service-level churn rates, and preview at most 50 rows with no customer identifiers. Demographics remain available for exploration. |
| Predict Churn | Score one fictional customer | Collect the shared service, billing and account-history inputs. Show churn probability, predicted class, risk category and signed, case-specific model responses with their reference values. Explain the statistical limits. |
| Model Performance | Understand selection and errors | Show cross-validation results for three models, selected model and threshold, holdout metrics and curves, confusion matrix, metric definitions and global feature importance. Label the split used for each result. |
| Batch Predictions | Score a CSV and export results | Provide a fictional example, validate the header and values, impute permitted blanks, preview predictions, and download identifiers, predicted classes and probabilities in input order. |

## Data and model requirements

- Use the commonly used 7,043-row IBM sample and verify the source, checksum and actual schema. Keep the raw dataset out of Git history.
- Exclude the customer identifier, target and all demographic attributes from training and inference. Share the schema across both prediction flows.
- Compare logistic regression, a decision tree and a random forest. Select by mean cross-validated average precision and choose the classification threshold on validation data.
- Generate all reported results from a seeded run. Report accuracy, precision, recall, F1, ROC-AUC, average precision, confusion matrix, ROC and precision–recall curves from the held-out test set for the selected model.
- Describe false positives, false negatives, limitations of probability scores and the limits of association-based explanations.
- Include an executed educational EDA notebook, a README, a beginner-friendly project explanation, and exactly 20 implementation-specific interview questions with answers.

## Release acceptance

The five pages run on Python 3.12 in a local browser and the prediction and batch flows work with the saved pipeline. Pytest and the notebook execute successfully. The public demo uses Streamlit Community Cloud, with no paid APIs, secrets, database or runtime model training. Record the public URL after verifying anonymous access and a real prediction. Milestone reviews are informational and implementation continues unless a major decision or blocker arises.
