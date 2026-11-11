"""Sparkov (Kaggle kartik2112/fraud-detection, CC0): simulated card transactions with card and merchant ids.

Every behavioural feature is built from the PAST of the same card only, so nothing leaks from the future.
Note: the `unix_time` column in this dataset is inconsistent with the date string (it is shifted by years),
so all time logic uses the parsed `trans_date_trans_time` instead.
"""
import numpy as np
import pandas as pd

CATS = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness", "home",
        "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"]
USECOLS = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "lat", "long", "merch_lat",
           "merch_long", "is_fraud"]
LABEL, AMOUNT = "is_fraud", "amt"


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load(root="data/sparkov", nrows=None) -> pd.DataFrame:
    parts = []
    for name, tag in (("fraudTrain", "train_file"), ("fraudTest", "test_file")):
        d = pd.read_csv(f"{root}/{name}.csv", usecols=USECOLS, nrows=nrows)
        d["file"] = tag
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    t = pd.to_datetime(df.pop("trans_date_trans_time"), format="%Y-%m-%d %H:%M:%S")
    # seconds since epoch from the DATE STRING (unix_time is unreliable). Cast the unit explicitly: pandas 3 may
    # parse to microseconds, and a hard-coded // 10**9 would silently give the wrong scale.
    df["ts"] = t.astype("datetime64[s]").astype("int64")
    df["hour"] = t.dt.hour.astype("int8")
    df["dow"] = t.dt.dayofweek.astype("int8")
    df["card_id"] = pd.factorize(df.pop("cc_num"))[0].astype("int32")
    df["merchant_id"] = pd.factorize(df.pop("merchant"))[0].astype("int32")
    df["cat_id"] = pd.Categorical(df.pop("category"), categories=CATS).codes.astype("int16")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return df


def split(df: pd.DataFrame, val_frac: float = 0.2):
    """Official split is already temporal (test file is later). The last `val_frac` of train-file time is validation."""
    is_test = df.file.eq("test_file").to_numpy()
    trv = df[~is_test]
    cut = trv.ts.quantile(1 - val_frac)
    tr, va, te = trv[trv.ts < cut], trv[trv.ts >= cut], df[is_test]
    assert tr.ts.max() <= va.ts.min() and va.ts.max() <= te.ts.min(), "temporal leakage"
    return tr, va, te


# ---------------------------------------------------------------------------- single-transaction features
def base_features(df) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    amt, h, dow = df.amt.to_numpy(float), df.hour.to_numpy(), df.dow.to_numpy()
    x["amt"] = amt
    x["log_amt"] = np.log1p(amt)
    x["hour_sin"], x["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    x["is_night"] = ((h >= 22) | (h <= 3)).astype(np.int8)
    x["dow_sin"], x["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    for i, c in enumerate(CATS):
        x[f"cat_{c}"] = (df.cat_id.to_numpy() == i).astype(np.int8)
    return x


def geo_features(df) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["dist_home_km"] = haversine(df.lat.to_numpy(float), df.long.to_numpy(float),
                                  df.merch_lat.to_numpy(float), df.merch_long.to_numpy(float))
    return x


# ---------------------------------------------------------------------------- per-card behaviour (past only)
def behaviour_features(df) -> pd.DataFrame:
    """Velocity, spending baseline, novelty and impossible-travel features. `df` must be sorted by time."""
    n = len(df)
    card, ts, amt = df.card_id.to_numpy(), df.ts.to_numpy(np.int64), df.amt.to_numpy(float)
    mlat, mlon = df.merch_lat.to_numpy(float), df.merch_long.to_numpy(float)
    cols = ["n_tx_1h", "n_tx_24h", "n_tx_7d", "hist_n_30d", "amt_mean_30d", "amt_std_30d", "secs_since_prev", "dist_prev_km"]
    out = {c: np.full(n, np.nan) for c in cols}
    order = np.argsort(card, kind="stable")  # stable: within a card the time order is preserved
    for idx in np.split(order, np.flatnonzero(np.diff(card[order])) + 1):
        t, a, i = ts[idx], amt[idx], np.arange(len(idx))
        for name, w in (("n_tx_1h", 3600), ("n_tx_24h", 86400), ("n_tx_7d", 7 * 86400)):
            out[name][idx] = i - np.searchsorted(t, t - w, side="left")  # earlier transactions inside the window
        l30 = np.searchsorted(t, t - 30 * 86400, side="left")
        c1, c2 = np.r_[0.0, np.cumsum(a)], np.r_[0.0, np.cumsum(a * a)]
        cnt = i - l30
        s1, s2 = c1[i] - c1[l30], c2[i] - c2[l30]
        safe = np.maximum(cnt, 1)
        mean = np.where(cnt > 0, s1 / safe, np.nan)
        var = np.where(cnt > 0, s2 / safe - (s1 / safe) ** 2, np.nan)
        out["hist_n_30d"][idx], out["amt_mean_30d"][idx] = cnt, mean
        out["amt_std_30d"][idx] = np.sqrt(np.maximum(var, 0))
        out["secs_since_prev"][idx] = np.r_[np.nan, np.diff(t)]
        la, lo = mlat[idx], mlon[idx]
        out["dist_prev_km"][idx] = np.r_[np.nan, haversine(la[:-1], lo[:-1], la[1:], lo[1:])]
    x = pd.DataFrame(out, index=df.index)
    mean = x.amt_mean_30d.fillna(pd.Series(amt, index=df.index))  # no history -> baseline is the amount itself (z = 0)
    std = x.amt_std_30d.fillna(0.0)
    x["amt_z_30d"] = ((amt - mean) / std.clip(lower=1.0)).clip(-10, 10)
    x["amt_mean_30d"], x["amt_std_30d"] = mean, std
    x["log_secs_since_prev"] = np.log1p(x.secs_since_prev.fillna(30 * 86400.0))
    x["dist_prev_km"] = x.dist_prev_km.fillna(0.0)
    x["speed_kmh"] = (x.dist_prev_km / np.maximum(x.secs_since_prev.fillna(30 * 86400.0) / 3600, 1 / 60)).clip(upper=2000)
    x["new_merchant"] = (df.groupby(["card_id", "merchant_id"]).cumcount() == 0).astype(np.int8)
    x["new_category"] = (df.groupby(["card_id", "cat_id"]).cumcount() == 0).astype(np.int8)
    return x.drop(columns=["secs_since_prev"])


# ---------------------------------------------------------------------------- label-delayed risk encoding
def delayed_risk(df, key: str, base_rate: float, delay_days: int = 7, prior: float = 20.0):
    """Smoothed fraud rate of `key` (merchant, category...) using only labels at least `delay_days` old.

    Fraud labels are not known immediately in real life (chargebacks arrive late), so a transaction may only use
    labels older than the delay. `base_rate` must come from the TRAINING split only.
    """
    n = len(df)
    k, ts, y = df[key].to_numpy(), df.ts.to_numpy(np.int64), df[LABEL].to_numpy(float)
    rate, support = np.full(n, base_rate), np.zeros(n)
    order = np.argsort(k, kind="stable")
    for idx in np.split(order, np.flatnonzero(np.diff(k[order])) + 1):
        t, cf = ts[idx], np.r_[0.0, np.cumsum(y[idx])]
        j = np.searchsorted(t, t - delay_days * 86400, side="right")  # labels older than the delay
        rate[idx] = (cf[j] + prior * base_rate) / (j + prior)
        support[idx] = j
    return rate, support


def risk_features(df, base_rate: float, delay_days: int = 7) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["merchant_risk"], sup = delayed_risk(df, "merchant_id", base_rate, delay_days)
    x["merchant_support"] = np.log1p(sup)
    x["category_risk"], _ = delayed_risk(df, "cat_id", base_rate, delay_days, prior=200.0)
    return x


# ---------------------------------------------------------------------------- feature sets
BASE = None  # filled lazily from base_features columns
GEO_COLS = ["dist_home_km"]
BEHAVIOUR_COLS = ["n_tx_1h", "n_tx_24h", "n_tx_7d", "hist_n_30d", "amt_mean_30d", "amt_std_30d", "amt_z_30d",
                  "log_secs_since_prev", "dist_prev_km", "speed_kmh", "new_merchant", "new_category"]
RISK_COLS = ["merchant_risk", "merchant_support", "category_risk"]
BASE_COLS = ["amt", "log_amt", "hour_sin", "hour_cos", "is_night", "dow_sin", "dow_cos", *[f"cat_{c}" for c in CATS]]


def build(df: pd.DataFrame, base_rate: float, delay_days: int = 7) -> pd.DataFrame:
    """All features for every row (float32 to save memory). Needs the full chronological frame."""
    parts = [df, base_features(df).drop(columns=["amt"]),  # `amt` already lives in df
             geo_features(df), behaviour_features(df), risk_features(df, base_rate, delay_days)]
    out = pd.concat(parts, axis=1)
    feat = BASE_COLS + GEO_COLS + BEHAVIOUR_COLS + RISK_COLS
    out[feat] = out[feat].astype("float32")
    return out


class ColumnSelector:
    """Feature 'builder' for FraudModel: the features are already in the frame, so just select columns."""

    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, df):
        return self

    def transform(self, df) -> pd.DataFrame:
        return df[self.cols]


class Selector:
    """Picklable factory (a lambda cannot be saved with the model): Selector(cols)() -> ColumnSelector."""

    def __init__(self, cols):
        self.cols = list(cols)

    def __call__(self):
        return ColumnSelector(self.cols)
