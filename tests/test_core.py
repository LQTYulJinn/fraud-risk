"""Offline tests on synthetic data (no network, a few seconds)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudrisk import data, metrics as M, serve
from fraudrisk.features import FeatureBuilder
from fraudrisk.model import FraudModel
from fraudrisk.scoring import FraudScorer


def synth(n=6000, frac=0.03, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=data.VCOLS)
    y = (rng.random(n) < frac).astype(int)
    df.loc[y == 1, ["V1", "V2"]] += 2.5  # frauds are separable but not trivially
    df["Amount"] = rng.lognormal(3, 1, n).round(2)
    df["Time"] = np.sort(rng.uniform(0, 172800, n))
    df["Class"] = y
    return df


def test_temporal_split_is_ordered_and_complete():
    df = synth()
    tr, va, te = data.temporal_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()


def test_split_refuses_unordered_input():
    df = synth().sample(frac=1.0, random_state=1).reset_index(drop=True)
    with pytest.raises(AssertionError):
        data.temporal_split(df)


def test_feature_stats_come_from_train_only():
    df = synth()
    tr, va, _ = data.temporal_split(df)
    fb = FeatureBuilder().fit(tr)
    before = fb.med_.copy()
    fb.transform(va)
    assert np.array_equal(before, fb.med_)  # transform never refits
    assert not np.array_equal(before, FeatureBuilder().fit(pd.concat([tr, va])).med_)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert M.psi(a, rng.normal(size=20000)) < 0.05
    assert M.psi(a, rng.normal(loc=1.5, size=20000)) > 0.25


def test_cost_threshold_beats_flagging_nothing_and_everything():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    score = np.where(y == 1, rng.beta(5, 2, 5000), rng.beta(2, 5, 5000))
    amount = rng.lognormal(4, 1, 5000)
    t = M.best_threshold(y, amount, score, review_cost=5.0)
    best = M.expected_cost(y, amount, score >= t, 5.0)
    assert best < M.expected_cost(y, amount, np.zeros(5000, bool), 5.0)
    assert best < M.expected_cost(y, amount, np.ones(5000, bool), 5.0)


def test_recall_at_budget_and_bootstrap_interval():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0] * 50)
    s = np.where(y == 1, 0.9, 0.1) + np.random.default_rng(0).normal(0, 0.01, len(y))
    assert M.recall_at_budget(y, s, 0.2) == 1.0
    lo, hi = M.bootstrap_ci(y, s, n=50)
    assert lo <= M.pr_auc(y, s) <= hi


@pytest.mark.parametrize("kind", ["logreg", "hgb", "iforest", "kmeans"])
def test_every_model_kind_beats_chance(kind):
    df = synth()
    tr, va, _ = data.temporal_split(df)
    m = FraudModel(kind=kind, engineered=True).fit(tr, tr.Class.to_numpy())
    s = m.raw_score(va)
    assert s.shape == (len(va),)
    assert M.pr_auc(va.Class.to_numpy(), s) > va.Class.mean()  # above the no-skill line


def test_stacked_anomaly_features_run_end_to_end():
    df = synth()
    tr, va, _ = data.temporal_split(df)
    m = FraudModel(kind="hgb", anomaly_features=True).fit(tr, tr.Class.to_numpy())
    assert np.isfinite(m.raw_score(va)).all()


def build_scorer():
    df = synth()
    tr, va, te = data.temporal_split(df)
    m = FraudModel(kind="hgb").fit(tr, tr.Class.to_numpy())
    sc = FraudScorer(m, 5.0).calibrate(va, va.Class.to_numpy())
    pva = sc.proba(va)
    sc.set_thresholds(M.best_threshold(va.Class.to_numpy(), va.Amount.to_numpy(), pva, 5.0))
    return sc, te


def test_bands_are_monotonic_in_fraud_rate():
    sc, te = build_scorer()
    p, band = sc.score(te)
    rate = {b: te.Class.to_numpy()[band == b].mean() for b in ("LOW", "MEDIUM", "HIGH") if (band == b).any()}
    vals = [rate[b] for b in ("LOW", "MEDIUM", "HIGH") if b in rate]
    assert vals == sorted(vals), rate
    assert ((p >= 0) & (p <= 1)).all()


def test_api_scores_and_validates(tmp_path, monkeypatch):
    import joblib
    sc, te = build_scorer()
    path = tmp_path / "model.joblib"
    joblib.dump(sc, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    serve.reset()
    c = TestClient(serve.app)
    assert c.get("/health").json()["status"] == "ok"
    row = te.drop(columns="Class").iloc[0].to_dict()
    r = c.post("/score", json=row)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1 and body["risk_band"] in ("LOW", "MEDIUM", "HIGH")
    bad = dict(row)
    bad.pop("V5")
    assert c.post("/score", json=bad).status_code == 422  # missing field
    neg = dict(row, Amount=-1.0)
    assert c.post("/score", json=neg).status_code == 422  # negative amount rejected


def test_api_reports_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    serve.reset()
    c = TestClient(serve.app)
    row = {k: 0.0 for k in ["Time", "Amount", *data.VCOLS]}
    assert c.post("/score", json=row).status_code == 503
