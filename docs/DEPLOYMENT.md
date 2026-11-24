# Publish ChurnIQ

## Prepare the repository

Use Python 3.12 and run the commands in the README. Verify the saved Joblib file and metrics are included and `data/raw/` is ignored. GitHub and Streamlit require publisher accounts. The app itself needs no API key, cloud database or private data.

If publishing a fresh local checkout, create a public repository under your GitHub account and use the GitHub CLI:

```bash
gh auth login
git add .
git diff --cached --check
git commit -m "Build ChurnIQ portfolio application"
git branch -M main
gh repo create ChurnIQ --public --source . --remote origin --push
```

For an existing public repository, connect its URL as `origin` if needed, then run `git push -u origin main`. The README's MIT license applies to original ChurnIQ code. The separately attributed IBM sample downloads from the pinned public source at runtime when absent.

## Create the Streamlit Community Cloud app

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and connect the GitHub account that owns the repo. Complete account consent in your browser.
2. Select **Create app**, then **Yup, I have an app**. Enter the GitHub repository, branch `main`, and main file path `app.py`.
3. In **Advanced settings**, select Python **3.12**. Leave secrets empty. Choose an available subdomain if desired, then select **Deploy**.
4. Wait for the app to initialize. Check that the app is public in its sharing settings.
5. Run the public smoke test below from a signed-out or private browser window. Record the verified URL in the README.

The Community Cloud instance installs `requirements.txt` and loads the bundled model. If Overview reports a dataset issue, check the deployment logs for errors downloading the verified IBM sample. If the prediction pages report a model issue, check the published artifact pair and the installed scikit-learn version. [Official Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Public smoke test

- Open the deployed URL and confirm Overview has 7,043 customers and a 26.5% recorded churn rate.
- Open Data Explorer and filter to Month-to-month. Verify the count changes.
- Submit the default profile in Predict Churn and verify a probability, prediction and individual explanation.
- Open Model Performance and verify the random forest, ROC and precision–recall curves, and confusion matrix appear.
- Run the fictional example in Batch Predictions, download the output, and verify the two rows and the ID, class and probability fields.
