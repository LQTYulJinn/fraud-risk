"""Load the ULB credit-card fraud data and make a leakage-safe temporal split."""
import urllib.request
from pathlib import Path

import pandas as pd

VCOLS = [f"V{i}" for i in range(1, 29)]  # PCA components (anonymised by the data owner)
RAW_COLS = ["Time", *VCOLS, "Amount"]
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def download(path="data/creditcard_raw.csv") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        urllib.request.urlretrieve(URL, p)
    return p


def load(path="data/creditcard_raw.csv", dedupe=True) -> pd.DataFrame:
    """Exact duplicate rows are dropped first: duplicates spread across splits would leak labels."""
    df = pd.read_csv(download(path))
    if dedupe:
        df = df.drop_duplicates()
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train=0.6, val=0.2):
    """Split by time order, never shuffled: train < validation < test."""
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    tr, va, te = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min(), "temporal leakage"
    return tr, va, te
