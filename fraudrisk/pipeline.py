"""Shared experiment pipeline for any fraud dataset.

Train every experiment on TRAIN, log to MLflow, pick the champion on VALIDATION PR-AUC, fit calibration and the
cost-optimal threshold on VALIDATION, and only then report TEST numbers.
"""
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_curve

from . import metrics as M
from .model import FraudModel
from .scoring import FraudScorer

REVIEW_COSTS = (1.0, 5.0, 20.0)  # ASSUMPTION: cost of one manual review in USD; the default is 5


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def run_pipeline(*, name, root, tr, va, te, label, amount, experiments, intro, review_cost=5.0, n_boot=200,
                 plot_models=()):
    root = Path(root)
    rep_dir, art_dir = root / "reports" / name, root / "artifacts" / name
    rep_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    ytr, yva, yte = (d[label].to_numpy() for d in (tr, va, te))
    a_va, a_te = va[amount].to_numpy(), te[amount].to_numpy()
    print(f"[{name}] train/val/test rows: {len(tr)}/{len(va)}/{len(te)} | frauds: {ytr.sum()}/{yva.sum()}/{yte.sum()}",
          flush=True)

    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(f"fraud-risk-{name}")

    rows, models = [], {}
    for exp_name, cfg in experiments:
        t0 = time.time()
        with mlflow.start_run(run_name=exp_name):
            m = FraudModel(**cfg).fit(tr, ytr)
            sva, ste = m.raw_score(va), m.raw_score(te)
            lo, hi = M.bootstrap_ci(yte, ste, n=n_boot)
            r = dict(model=exp_name, val_pr_auc=M.pr_auc(yva, sva), val_roc_auc=M.roc_auc(yva, sva),
                     test_pr_auc=M.pr_auc(yte, ste), test_pr_auc_lo=lo, test_pr_auc_hi=hi,
                     test_roc_auc=M.roc_auc(yte, ste),
                     test_recall_top_0p1pct=M.recall_at_budget(yte, ste, 0.001),
                     test_recall_top_1pct=M.recall_at_budget(yte, ste, 0.01), seconds=time.time() - t0)
            mlflow.log_params({k: str(v) for k, v in cfg.items() if k != "builder_factory"})
            mlflow.log_metrics({k: v for k, v in r.items() if k != "model"})
        rows.append(r)
        models[exp_name] = m
        print(f"[{name}] {exp_name:24s} val PR-AUC {r['val_pr_auc']:.4f} | test PR-AUC {r['test_pr_auc']:.4f} "
              f"[{lo:.3f}, {hi:.3f}] | {r['seconds']:.0f}s", flush=True)

    table = pd.DataFrame(rows)
    champ_name = table.sort_values("val_pr_auc", ascending=False).iloc[0]["model"]  # VALIDATION only
    champ = models[champ_name]
    print(f"[{name}] champion (selected on validation): {champ_name}", flush=True)

    # ---- calibration, threshold, bands: all fitted on VALIDATION -----------------------------------------
    scorer = FraudScorer(champ, review_cost).calibrate(va, yva)
    pva, pte = scorer.proba(va), scorer.proba(te)
    thr = M.best_threshold(yva, a_va, pva, review_cost)
    scorer.set_thresholds(thr)

    pr_, rc_, th_ = precision_recall_curve(yva, pva)
    f1_ = 2 * pr_[:-1] * rc_[:-1] / np.clip(pr_[:-1] + rc_[:-1], 1e-12, None)
    thr_f1 = float(th_[int(np.argmax(f1_))])

    cost_rows = []
    for rc in REVIEW_COSTS:
        t_rc = M.best_threshold(yva, a_va, pva, rc)
        c_model = M.expected_cost(yte, a_te, pte >= t_rc, rc)
        c_none = M.expected_cost(yte, a_te, np.zeros(len(yte), bool), rc)
        c_all = M.expected_cost(yte, a_te, np.ones(len(yte), bool), rc)
        p, r, f = M.precision_recall_f1(yte, pte >= t_rc)
        cost_rows.append(dict(review_cost_usd=rc, threshold=t_rc, precision=p, recall=r, f1=f,
                              alerts=int((pte >= t_rc).sum()), cost_model=c_model, cost_flag_nothing=c_none,
                              cost_flag_everything=c_all, saving_vs_nothing_pct=100 * (1 - c_model / c_none)))
    costs = pd.DataFrame(cost_rows)
    p_f1, r_f1, f1_test = M.precision_recall_f1(yte, pte >= thr_f1)

    band_te = scorer.band(pte)
    band_rows = []
    for b in ("LOW", "MEDIUM", "HIGH"):
        m_ = band_te == b
        band_rows.append(dict(band=b, transactions=int(m_.sum()), frauds=int(yte[m_].sum()),
                              fraud_rate_pct=100 * yte[m_].mean() if m_.any() else 0.0,
                              share_of_all_frauds_pct=100 * yte[m_].sum() / max(yte.sum(), 1)))
    bands = pd.DataFrame(band_rows)

    edges = [0, 0.001, 0.01, 0.1, 0.5, 1.0001]
    cal_rows = []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m_ = (pte >= lo_) & (pte < hi_)
        if m_.any():
            cal_rows.append(dict(predicted_range=f"[{lo_}, {min(hi_, 1)})", n=int(m_.sum()),
                                 mean_predicted=float(pte[m_].mean()), observed_fraud_rate=float(yte[m_].mean())))
    calib = pd.DataFrame(cal_rows)
    brier_cal = float(brier_score_loss(yte, pte))
    brier_raw = float(brier_score_loss(yte, np.clip(champ.raw_score(te), 0, 1))) if champ.is_probabilistic else float("nan")

    ftr, fte = champ.fb_.transform(tr), champ.fb_.transform(te)
    drift = pd.DataFrame({"feature": ftr.columns, "psi_train_vs_test": [M.psi(ftr[c], fte[c]) for c in ftr.columns]})
    drift = drift.sort_values("psi_train_vs_test", ascending=False).reset_index(drop=True)
    score_psi = M.psi(pva, pte)

    plt.figure(figsize=(6.5, 4.8))
    for nm in dict.fromkeys([*plot_models, champ_name]):
        if nm in models:
            s = models[nm].raw_score(te)
            p_, r_, _ = precision_recall_curve(yte, s)
            plt.plot(r_, p_, label=f"{nm} (PR-AUC {M.pr_auc(yte, s):.3f})")
    plt.axhline(yte.mean(), ls=":", c="gray", label=f"no-skill ({yte.mean():.4f})")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"{name}: test precision-recall")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(rep_dir / "pr_curve.png", dpi=130)
    plt.close()

    bundle = art_dir / "model.joblib"
    joblib.dump(scorer, bundle)
    champ_row = table[table.model == champ_name].iloc[0]
    report = f"""# Results: {name} (generated by the training script)

{intro}
Frauds in train/val/test: {int(ytr.sum())}/{int(yva.sum())}/{int(yte.sum())}.
Champion chosen on **validation** PR-AUC: `{champ_name}`. Test is used only for the numbers below.

## Model comparison
{md_table(table)}

`test_pr_auc_lo/hi` is a 95% stratified-bootstrap interval ({n_boot} resamples); with {int(yte.sum())} test frauds,
differences smaller than the interval are not evidence.

## Cost-optimal threshold (fitted on validation, applied to test)
Assumption: one manual review costs `review_cost_usd`; a missed fraud costs its amount.
{md_table(costs)}

F1-optimal threshold from validation, applied to test: precision {p_f1:.3f}, recall {r_f1:.3f}, F1 {f1_test:.3f}.

## Risk bands on test (HIGH >= {scorer.thr_high:.4f}, MEDIUM >= {scorer.thr_medium:.4f}; MEDIUM = 0.2 x HIGH is a heuristic, not tuned)
{md_table(bands)}

## Calibration on test (Brier calibrated {brier_cal:.5f}, raw {brier_raw:.5f})
{md_table(calib)}

## Drift (PSI, rule of thumb >0.25 = major shift). Score PSI validation -> test: {score_psi:.3f}
{md_table(drift.head(10))}
"""
    (rep_dir / "results.md").write_text(report, encoding="utf-8")
    table.to_csv(rep_dir / "models.csv", index=False)
    costs.to_csv(rep_dir / "costs.csv", index=False)
    with mlflow.start_run(run_name=f"champion_{champ_name}") as run_:
        mlflow.log_params({"champion": champ_name, "review_cost": review_cost, "thr_high": scorer.thr_high})
        mlflow.log_metrics({"val_pr_auc": float(champ_row.val_pr_auc), "test_pr_auc": float(champ_row.test_pr_auc),
                            "brier_calibrated": brier_cal, "score_psi": score_psi})
        mlflow.log_artifacts(str(rep_dir), "reports")
        try:
            # MLflow 3 registers *logged models*, not plain artifact folders
            # cloudpickle: MLflow 3's default (skops) refuses our own classes; fine for a local registry
            info = mlflow.sklearn.log_model(scorer, name="scorer", registered_model_name=f"fraud-risk-{name}",
                                            serialization_format="cloudpickle")
            ver = getattr(info, "registered_model_version", None)
            if ver:
                mlflow.MlflowClient().set_registered_model_alias(f"fraud-risk-{name}", "champion", str(ver))
            print(f"[{name}] registered fraud-risk-{name} version {ver}, alias champion", flush=True)
        except Exception as e:  # registry is a convenience; training results must not depend on it
            print(f"[{name}] MLflow registry skipped: {repr(e)[:300]}", flush=True)
    print(f"[{name}] wrote {rep_dir / 'results.md'} and {bundle}", flush=True)
    return table, costs, bands
