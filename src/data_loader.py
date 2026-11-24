"""Fetch and validate the pinned IBM sample."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pandas as pd

from src.schema import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REVISION = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{REVISION}/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
EXPECTED_ROWS = 7043


def _verified(content: bytes) -> bytes:
    if sha256(content).hexdigest() != DATA_SHA256:
        raise ValueError("IBM dataset checksum differs from the pinned source. Use the download instructions.")
    return content


def download_dataset(path: Path = DATA_PATH) -> Path:
    """Verify an existing CSV or download it using an atomic file replacement."""
    if path.is_file():
        _verified(path.read_bytes())
        return path
    with urlopen(DATA_URL, timeout=20) as response:
        content = _verified(response.read(2_000_000))

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the verified sample and check the cleaned table before use."""
    content = _verified(download_dataset(path).read_bytes())
    raw = pd.read_csv(BytesIO(content), dtype="string", keep_default_na=False)
    clean = validate_dataset(raw)
    if len(clean) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} IBM rows, received {len(clean):,}.")
    return clean
