"""Download and check the IBM Telco Customer Churn sample."""

from src.data_loader import DATA_PATH, DATA_SHA256, ROOT, load_dataset


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Verified {len(dataset):,} rows and {len(dataset.columns)} columns.")
    print(f"File: {DATA_PATH.relative_to(ROOT)}")
    print(f"SHA-256: {DATA_SHA256}")
