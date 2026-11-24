# ChurnIQ interview preparation

## 1. What does ChurnIQ predict?
It estimates the probability that a telecom customer's label is churn using service, billing and account-history information. It then uses a validation-derived cutoff for the binary class.

## 2. What data did you use and how did you validate it?
I used IBM's 7,043-row, 21-column Telco sample at a pinned Git revision. The loader verifies a SHA-256 checksum and checks the expected fields, unique customer IDs and valid targets.

## 3. What did you learn from EDA?
The recorded churn rate was 26.5%. It was 42.7% for month-to-month contracts and 2.8% for two-year contracts. A difference in group rates describes the sample and can reflect other factors that vary between those groups.

## 4. Why did you omit some columns?
Customer IDs identify records and could create spurious patterns. The churn label is the answer, and demographics were reserved for analysis. The 15 selected features are shared by training and both inference paths.

## 5. How did you handle missing values?
The CSV had 11 whitespace-only TotalCharges values. They become missing numeric values. The pipeline learns numeric medians and categorical modes from training rows and applies them to later rows.

## 6. Why use one-hot encoding?
It gives each nominal category its own indicator. Coding contract types as arbitrary integers would introduce a numerical ordering into models that treat the codes as quantities.

## 7. What happens when a category is new?
The encoder uses `handle_unknown="ignore"` for resilience and produces zeros for that field's known indicators if it receives an unknown value. The public CSV validator flags unrecognized options so the user can resolve spelling or schema mismatches.

## 8. When did you scale numeric values?
I scaled the numeric branch of logistic regression after imputation so feature scale plays a smaller role in regularization. The tree models use feature ordering for splits and retain the original scales.

## 9. What did your train, validation and test splits do?
The stratified 60/20/20 split yielded 4,225, 1,409 and 1,409 rows. Training supplied the folds for model selection and fitting, validation selected the classification cutoff, and test estimated final performance.

## 10. Where could data leakage have happened?
Fitting the imputer or encoder before splitting would allow evaluation rows to inform preprocessing. The scikit-learn pipeline fits those steps inside each fold. The test partition also stays outside model and threshold selection.

## 11. Why compare these three models?
Logistic regression is an established baseline. A decision tree models interactions through branching and makes a single set of decisions easy to inspect. Random forest combines trees to improve stability.

## 12. Why use average precision as the main selection metric?
I wanted a metric that pays attention to the positive, less common churn class across thresholds. Average precision weights precision by gains in recall. Scikit-learn's calculation differs from a trapezoidal area under the precision–recall curve.

## 13. Which model won and how strong was the difference?
Random forest scored 0.6756 in mean five-fold average precision. Logistic regression scored 0.6673 and decision tree scored 0.6181. I selected by the predefined rule. The gap between the top models was modest.

## 14. How did you pick the probability cutoff?
I tried the score cutoffs available on the validation partition and picked the one that maximized churn F1, breaking ties toward higher recall. It was about 36.49%. The untouched test set supplied the final metrics.

## 15. What does a lower threshold change?
It flags more customers. In validation, moving from 50% to the chosen threshold caught another 91 churners and created 89 more false alerts. The right operating point depends on the team's costs and capacity.

## 16. How would you summarize test performance in an interview?
On 1,409 held-out records, the model reached 56.38% precision, 70.86% recall, F1 of 0.6280, ROC-AUC of 0.8427, and average precision of 0.6520. It caught 265 churners and missed 109.

## 17. How does permutation importance work here?
I shuffled one raw feature at a time eight times on the test set and measured each drop in average precision. This measures model reliance, which can be shared between correlated features.

## 18. How can you explain a single score with a random forest?
I take the customer's submitted inputs, swap a field group for observed training reference values, and rescore the edited profile. The difference is the model's response to the concrete edit. The app lists the values that were changed.

## 19. How do you get from training to a fast web demo?
The training script saves the preprocessing and model as one Joblib artifact with the threshold and reference values. Streamlit loads and caches the verified artifact. The batch flow validates once and passes the whole input frame to the same pipeline used by the single form.

## 20. What would you do before using it with real customers?
I would test on permissioned, chronological data, check probability calibration and group-wise errors, choose a cutoff based on outreach economics, and use an experiment to measure intervention impact.
