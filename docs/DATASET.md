# IBM Telco Customer Churn sample

IBM provides this sample in its archived Telco Customer Churn project. The CSV has 7,043 rows and 21 columns. The project uses the CSV at revision `d5371f5d83a446ad5673cbcca3b814b926491f8a`. IBM publishes the source repository under Apache 2.0. ChurnIQ's MIT license covers its original software and documentation. [IBM dataset at pinned revision](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv) · [IBM license](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/d5371f5d83a446ad5673cbcca3b814b926491f8a/LICENSE)

Run `python -m scripts.download_data` to save and verify the file. For a manual download, save the pinned raw CSV as `data/raw/Telco-Customer-Churn.csv` without editing it, then run the same command to verify it. The expected SHA-256 is `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. [Pinned raw CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv)

The raw CSV is excluded from Git. The Data Explorer omits identifiers. The downloadable batch example is fictional and is used only to demonstrate the prediction interface. The analysis, model and evaluation use IBM's sample.
