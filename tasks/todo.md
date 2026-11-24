# ChurnIQ implementation checklist

- [x] Write product requirements, technical design and implementation checklist.
- [x] Pin and validate the IBM CSV. Verify checksum, row count, target, duplicates, feature categories and missing values.
- [x] Execute the EDA notebook and record the observed churn rate and major associations.
- [x] Train and tune the three pipelines with identical stratified folds. Confirm the validation threshold and write real test results.
- [x] Save and load deployable artifacts. Verify fixed-seed training and artifact consistency.
- [x] Implement single and batch prediction, risk bands, CSV export and per-instance explanations. Test normal and invalid inputs.
- [x] Build the five Streamlit pages. Verify filters, submitted scores, charts, CSV preview and export in a browser.
- [x] Finish README, dataset attribution, MIT license, project explanation, 20 interview questions and screenshots.
- [x] Run the full test suite. Verify a clean launch with no raw data in Git and the committed model artifacts.
- [ ] Publish the GitHub repository and confirm that its tracked files match the locally verified build.
- [ ] Deploy to Streamlit Community Cloud. Confirm anonymous access, run the live smoke tests, and update the README with the verified demo URL.
