"""Leakage-safe feature engineering: every statistic is fitted on the training split only."""
import numpy as np
import pandas as pd

from .data import VCOLS

ENGINEERED = ["log_amount", "hour_sin", "hour_cos", "is_small_amount", "is_round_amount",
              "z_max", "z_n_outliers", "v_norm"]


class FeatureBuilder:
    """Base features (V1..V28, Amount) plus engineered ones.

    The dataset has no card/account id, so velocity or per-customer features are impossible here;
    the engineered set is limited to amount/time patterns and per-row outlier statistics.
    """

    def __init__(self, engineered: bool = True):
        self.engineered = engineered

    def fit(self, df: pd.DataFrame):
        v = df[VCOLS]
        q = v.quantile([0.25, 0.75])
        iqr = (q.loc[0.75] - q.loc[0.25]).to_numpy()
        self.med_ = v.median().to_numpy()
        self.iqr_ = np.where(iqr == 0, 1.0, iqr)  # robust scale from TRAIN only
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[[*VCOLS, "Amount"]].copy()
        if not self.engineered:
            return x
        amount = df["Amount"].to_numpy()
        v = df[VCOLS].to_numpy()
        hour = (df["Time"].to_numpy() / 3600.0) % 24  # hour since the first transaction, not a calendar hour
        z = np.abs((v - self.med_) / self.iqr_)
        x["log_amount"] = np.log1p(amount)
        x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        x["is_small_amount"] = (amount <= 1.0).astype(int)  # tiny amounts are a known card-testing pattern
        x["is_round_amount"] = ((amount > 0) & (amount % 10 == 0)).astype(int)
        x["z_max"] = z.max(axis=1)
        x["z_n_outliers"] = (z > 4).sum(axis=1)
        x["v_norm"] = np.sqrt((v ** 2).sum(axis=1))
        return x
