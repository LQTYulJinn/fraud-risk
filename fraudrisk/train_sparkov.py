"""python -m fraudrisk.train_sparkov [--sample N]

Sparkov (Kaggle kartik2112/fraud-detection, CC0): SIMULATED card transactions with card and merchant ids.
Compares single-transaction features with per-card behavioural features and delayed merchant risk.
"""
import argparse
import sys

from . import sparkov as S
from .pipeline import run_pipeline


def experiments():
    base = S.BASE_COLS
    geo = base + S.GEO_COLS
    beh = geo + S.BEHAVIOUR_COLS
    risk = beh + S.RISK_COLS
    sel = S.Selector
    return [
        ("logreg_base", dict(kind="logreg", builder_factory=sel(base))),
        ("hgb_base", dict(kind="hgb", builder_factory=sel(base))),
        ("hgb_geo", dict(kind="hgb", builder_factory=sel(geo))),
        ("hgb_behaviour", dict(kind="hgb", builder_factory=sel(beh))),
        ("hgb_behaviour_risk", dict(kind="hgb", builder_factory=sel(risk))),
        ("logreg_behaviour_risk", dict(kind="logreg", builder_factory=sel(risk))),
        ("iforest_behaviour", dict(kind="iforest", builder_factory=sel(beh), n_trees=100)),
        ("kmeans_behaviour", dict(kind="kmeans", builder_factory=sel(beh), km_n_init=3)),
    ]


def main(root=".", sample=None, review_cost=5.0, n_boot=100):
    df = S.load(f"{root}/data/sparkov", nrows=sample)
    tr0, va0, te0 = S.split(df)
    full = S.build(df, base_rate=float(tr0.is_fraud.mean()))  # base rate from the TRAIN split only
    tr, va, te = full.loc[tr0.index], full.loc[va0.index], full.loc[te0.index]
    intro = (f"Data: Sparkov simulated card transactions (Kaggle, CC0), {len(df):,} rows, {df.card_id.nunique()} cards, "
             f"{df.merchant_id.nunique()} merchants. SIMULATED data, not real customers. Official split is temporal "
             "(test file is later); the last 20% of the train file by time is validation.\n"
             "Behavioural features use only the past of each card; merchant/category risk uses labels at least 7 days old.")
    name = "sparkov" if sample is None else f"sparkov_sample{sample}"
    return run_pipeline(name=name, root=root, tr=tr, va=va, te=te, label=S.LABEL, amount=S.AMOUNT,
                        experiments=experiments(), intro=intro, review_cost=review_cost, n_boot=n_boot,
                        plot_models=("logreg_base", "hgb_base", "hgb_behaviour_risk", "iforest_behaviour"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="rows to read from each file (smoke test)")
    main(sample=ap.parse_args().sample)
