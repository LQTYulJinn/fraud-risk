"""python -m fraudrisk.train   -> ULB credit-card fraud (real transactions, anonymised PCA features)."""
import sys

from . import data
from .pipeline import run_pipeline

EXPERIMENTS = [
    ("logreg_base", dict(kind="logreg", engineered=False)),
    ("logreg_eng", dict(kind="logreg", engineered=True)),
    ("hgb_base", dict(kind="hgb", engineered=False)),
    ("hgb_eng", dict(kind="hgb", engineered=True)),
    ("hgb_eng_balanced", dict(kind="hgb", engineered=True, class_weight="balanced")),
    ("hgb_eng_anomaly", dict(kind="hgb", engineered=True, anomaly_features=True)),
    ("iforest_unsup", dict(kind="iforest", engineered=True)),
    ("kmeans_unsup", dict(kind="kmeans", engineered=True)),
]


def main(root=".", review_cost=5.0, n_boot=300):
    df = data.load(f"{root}/data/creditcard_raw.csv")
    tr, va, te = data.temporal_split(df)
    intro = (f"Data: ULB credit-card fraud, {len(df):,} transactions after dropping exact duplicates, 48 hours, "
             "anonymised PCA features (no card or account id, so no per-customer features are possible).\n"
             "Temporal split 60/20/20.")
    return run_pipeline(name="ulb", root=root, tr=tr, va=va, te=te, label="Class", amount="Amount",
                        experiments=EXPERIMENTS, intro=intro, review_cost=review_cost, n_boot=n_boot,
                        plot_models=("logreg_base", "hgb_base", "iforest_unsup"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
