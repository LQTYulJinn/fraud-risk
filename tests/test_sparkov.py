"""Leakage and correctness tests for the per-card behavioural features (synthetic data, no network)."""
import numpy as np
import pandas as pd
import pytest

from fraudrisk import sparkov as S


def frame(ts, card=None, merchant=None, amt=None, fraud=None, cat=None):
    n = len(ts)
    return pd.DataFrame({
        "ts": np.asarray(ts, dtype="int64"),
        "card_id": np.zeros(n, "int32") if card is None else np.asarray(card, "int32"),
        "merchant_id": np.arange(n, dtype="int32") if merchant is None else np.asarray(merchant, "int32"),
        "cat_id": np.zeros(n, "int16") if cat is None else np.asarray(cat, "int16"),
        "amt": np.full(n, 10.0) if amt is None else np.asarray(amt, float),
        "lat": np.full(n, 40.0), "long": np.full(n, -74.0),
        "merch_lat": np.full(n, 40.0), "merch_long": np.full(n, -74.0),
        "is_fraud": np.zeros(n, int) if fraud is None else np.asarray(fraud, int),
        "hour": np.zeros(n, "int8"), "dow": np.zeros(n, "int8"),
    })


def synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, 90 * 86400, n))
    df = frame(ts, card=rng.integers(0, 30, n), merchant=rng.integers(0, 40, n), amt=rng.lognormal(3, 1, n),
               fraud=(rng.random(n) < 0.02).astype(int), cat=rng.integers(0, 14, n))
    df["merch_lat"] = 40 + rng.normal(0, 0.5, n)
    df["merch_long"] = -74 + rng.normal(0, 0.5, n)
    return df


def test_load_converts_timestamps_to_real_seconds(tmp_path):
    """Regression: a wrong time unit once made '1 hour' windows span weeks."""
    cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "lat", "long", "merch_lat",
            "merch_long", "is_fraud"]
    rows = [["2019-01-01 00:00:00", 1, "m", "travel", 5.0, 0, 0, 0, 0, 0],
            ["2019-01-01 01:00:00", 1, "m", "travel", 5.0, 0, 0, 0, 0, 0],
            ["2019-01-02 01:00:00", 1, "m", "travel", 5.0, 0, 0, 0, 0, 0]]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "fraudTrain.csv", index=False)
    pd.DataFrame(rows[:1], columns=cols).assign(trans_date_trans_time="2020-07-01 00:00:00").to_csv(
        tmp_path / "fraudTest.csv", index=False)
    df = S.load(str(tmp_path))
    assert df.ts.diff().dropna().iloc[:2].tolist() == [3600, 86400]
    assert df.ts.iloc[0] == 1546300800  # 2019-01-01 00:00:00 UTC
    assert S.behaviour_features(df).n_tx_1h.iloc[:3].tolist() == [0, 1, 0]


def test_haversine_one_degree_of_longitude_at_the_equator():
    assert S.haversine(0, 0, 0, 1) == pytest.approx(111.19, abs=0.1)


def test_window_counts_on_a_hand_checked_example():
    x = S.behaviour_features(frame([0, 1000, 5000, 100000]))
    assert x.n_tx_1h.tolist() == [0, 1, 0, 0]
    assert x.n_tx_24h.tolist() == [0, 1, 2, 0]
    assert x.new_merchant.tolist() == [1, 1, 1, 1]


def test_new_merchant_flag_turns_off_on_a_repeat_visit():
    x = S.behaviour_features(frame([0, 10, 20], merchant=[7, 8, 7]))
    assert x.new_merchant.tolist() == [1, 1, 0]


def test_behaviour_features_never_look_at_future_rows():
    df = synth()
    full = S.behaviour_features(df)
    k = 1500
    head = S.behaviour_features(df.iloc[:k].copy())  # pretend the future does not exist
    pd.testing.assert_frame_equal(full.iloc[:k].reset_index(drop=True), head.reset_index(drop=True), check_dtype=False)


def test_changing_future_rows_does_not_change_past_features():
    df = synth()
    a = S.behaviour_features(df)
    df2 = df.copy()
    df2.loc[2000:, "amt"] *= 50  # tamper with the future only
    b = S.behaviour_features(df2)
    pd.testing.assert_frame_equal(a.iloc[:2000], b.iloc[:2000], check_dtype=False)


def test_delayed_risk_ignores_labels_newer_than_the_delay():
    # key 0: one fraud on day 0; a later transaction on day 3 must NOT see it (delay 7), one on day 8 must.
    day = 86400
    df = frame([0, 3 * day, 8 * day], merchant=[0, 0, 0], fraud=[1, 0, 0])
    rate, support = S.delayed_risk(df, "merchant_id", base_rate=0.01, delay_days=7, prior=20.0)
    assert rate[0] == pytest.approx(0.01) and rate[1] == pytest.approx(0.01)  # no usable labels yet
    assert support[2] == 1 and rate[2] > 0.01  # the day-0 fraud is now old enough to use


def test_risk_features_are_causal_too():
    df = synth()
    a = S.risk_features(df, 0.02)
    df2 = df.copy()
    df2.loc[2500:, "is_fraud"] = 1  # flip future labels
    b = S.risk_features(df2, 0.02)
    pd.testing.assert_frame_equal(a.iloc[:2500], b.iloc[:2500], check_dtype=False)


def test_split_is_temporal_and_complete():
    df = synth()
    df["file"] = np.where(df.ts < df.ts.quantile(0.75), "train_file", "test_file")
    tr, va, te = S.split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.ts.max() <= va.ts.min() and va.ts.max() <= te.ts.min()


def test_full_feature_build_has_no_missing_values_and_stays_float32():
    df = synth()
    df["file"] = "train_file"
    out = S.build(df, base_rate=0.02)
    cols = S.BASE_COLS + S.GEO_COLS + S.BEHAVIOUR_COLS + S.RISK_COLS
    assert not out[cols].isna().any().any()
    assert (out[cols].dtypes == "float32").all()
