"""Metrics for heavily imbalanced fraud data: PR-AUC, alert-budget metrics, cost-based threshold, drift (PSI)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pr_auc(y, s) -> float:
    """Average precision. With ~0.2% positives it is far more informative than ROC-AUC."""
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def precision_recall_f1(y, flag):
    y, flag = np.asarray(y).astype(bool), np.asarray(flag).astype(bool)
    tp = int((y & flag).sum())
    p = tp / max(int(flag.sum()), 1)
    r = tp / max(int(y.sum()), 1)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def recall_at_budget(y, s, budget: float) -> float:
    """Share of all frauds found when analysts can review only the top `budget` fraction of transactions."""
    y, s = np.asarray(y), np.asarray(s)
    k = max(int(np.ceil(budget * len(y))), 1)
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def expected_cost(y, amount, flag, review_cost: float) -> float:
    """Every flagged transaction costs one manual review; every missed fraud costs its amount."""
    y, flag, amount = np.asarray(y).astype(bool), np.asarray(flag).astype(bool), np.asarray(amount)
    return float(review_cost * flag.sum() + amount[y & ~flag].sum())


def best_threshold(y, amount, score, review_cost: float, n_grid: int = 2000) -> float:
    """Cost-minimising threshold. Fit it on VALIDATION only, then apply to test unchanged."""
    grid = np.unique(np.quantile(score, np.linspace(0.90, 1.0, n_grid)))
    costs = [expected_cost(y, amount, score >= t, review_cost) for t in grid]
    return float(grid[int(np.argmin(costs))])


def bootstrap_ci(y, s, fn=pr_auc, n: int = 500, seed: int = 0, alpha: float = 0.05):
    """Stratified bootstrap interval: the test split holds only ~70-80 frauds, so one number is not enough."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(fn(y[idx], s[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major shift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
