# How ChurnIQ works

## 1. The problem

Customer churn means that someone stops subscribing to a service. The IBM sample contains customer information and a churn label. Predicting whether that label is Yes or No makes this a binary classification problem. The dashboard shows a probability estimate and converts it to a class using a cutoff.

## 2. The three models

**Logistic regression** provides a familiar baseline. It learns a weighted combination of the input features and converts it to a probability. A **decision tree** asks a sequence of questions such as whether a customer's tenure is below a particular value. A **random forest** pools predictions from many trees fitted with randomness in their training rows and candidate features. That pooling typically makes it less sensitive to any single tree's decisions.

## 3. From CSV to model inputs

The data loader checks the original IBM CSV against a checksum and validates the fields. Whitespace in TotalCharges becomes a missing value. A shared schema orders the same 15 billing, service and account-history columns during training, single prediction and batch prediction. Identifiers, demographics and the target stay outside the model inputs.

A ColumnTransformer routes numeric and categorical values through separate branches. Numeric blanks receive medians learned from training. Categorical blanks receive the most frequent training category. One-hot encoding represents each known category with indicator columns. Logistic regression standardizes the numeric branch because differently scaled inputs can interact with its regularization. The tree models split values by order and can use their original scale.

## 4. How we prevent leakage

Leakage occurs when model development has access to information that would be unavailable at prediction time or when held-out data influences learned choices. ChurnIQ creates three disjoint, stratified partitions. Stratification keeps the share of churners similar in each partition. The training partition contains 4,225 rows, and validation and test each contain 1,409.

Grid search tries small sets of settings using five training folds. In each trial, the pipeline learns preprocessing and the model using four folds, then scores the remaining fold. Each fold gets a turn. The chosen configuration is fitted on the full training partition. Validation selects the cutoff after fitting, and the held-out test partition is used for final evaluation and inspection. The app serves that fitted pipeline.

## 5. How we picked the model and threshold

Average precision combines precision at different levels of recall into a ranking summary. It gives us a way to compare how well the models bring churners toward the top of the list. Random forest had mean cross-validated average precision of 0.6756, ahead of logistic regression at 0.6673 and decision tree at 0.6181. The difference between the top two was small.

Once selected, the model scored the validation partition. The cutoff of **36.49%** gave the highest positive-class F1. At that cutoff, validation caught 260 churners with 176 false alerts. At 50%, it would catch 169 and produce 87 false alerts. Which trade-off is desirable for a real company depends on the cost of outreach and the cost of missing a churner.

## 6. Reading the held-out metrics

- **Accuracy** counts all correct classifications. Predicting retained for everybody would already get about 73% accuracy here.
- **Precision** asks how many flagged customers actually churned. It was **56.38%**.
- **Recall** asks how many actual churners the model caught. It was **70.86%**.
- **F1** is the harmonic mean of precision and recall. It was **0.6280**.
- **ROC-AUC** judges whether churners generally rank above retained customers over different cutoffs. It was **0.8427**.
- **Average precision** was **0.6520**. The held-out churn prevalence of about **26.5%** is the reference line for the precision–recall curve.
- **Brier score** is the average squared error of estimated probabilities. It was **0.1364**.

The confusion matrix counted 830 correctly classified retained customers, 205 false alerts, 109 missed churners and 265 correctly flagged churners. A false alert can trigger unnecessary outreach. A missed churner is a lost opportunity for outreach.

## 7. From form to answer

Streamlit takes form or CSV inputs, validates them, and sends the feature frame to the saved pipeline. The pipeline repeats the same transformations used in training and returns the probability for the churn class. Scores at or above the validation cutoff receive a Yes class. The risk bands begin at 0%, half the cutoff and the cutoff itself. The app keeps uploaded file state within the session and exports only ID, class and probability.

## 8. How we explain scores

Global permutation importance assesses model reliance by shuffling one field at a time in the held-out data and recalculating average precision. A noticeable drop suggests the model used that field. Input correlations can split the importance between related features, and a shuffle can create combinations that seldom occur.

The individual explanation performs a different calculation. It scores one customer, swaps one group of values to a defined training reference, and scores the resulting profile again. The result is a signed change in percentage points. Related phone or internet fields move together to keep those service combinations consistent. Each number applies to that specific edit, and the changes can interact. They cannot establish what would happen to a real customer.

## 9. What could improve

The dataset covers a fictional provider and a single sample. A future study could use real, permissioned customer data, a chronological holdout, probability calibration, and a threshold tied to outreach costs or team capacity. A group-wise error and fairness analysis would help identify uneven effects, since demographic proxies can remain in service and billing fields. A controlled experiment would be needed to estimate whether a retention intervention actually reduces churn.
