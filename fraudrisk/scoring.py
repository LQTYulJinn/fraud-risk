"""Risk scoring: calibrated fraud probability plus LOW / MEDIUM / HIGH bands.

Raw model scores are not probabilities (class weights and boosting distort them), so a Platt calibrator
is fitted on the VALIDATION split. Thresholds come from validation as well; the test split never decides anything.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

BANDS = ("LOW", "MEDIUM", "HIGH")


def _prep(model, score):
    s = np.asarray(score, float)
    if model.is_probabilistic:
        s = np.clip(s, 1e-9, 1 - 1e-9)
        return np.log(s / (1 - s)).reshape(-1, 1)  # logit
    return ((s - s.mean()) / (s.std() + 1e-9)).reshape(-1, 1)


class FraudScorer:
    def __init__(self, model, review_cost: float = 5.0):
        self.model, self.review_cost = model, review_cost

    def calibrate(self, df_val, y_val):
        raw = self.model.raw_score(df_val)
        self._mu, self._sd = raw.mean(), raw.std()  # for non-probabilistic scores
        self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(raw), y_val)
        return self

    def _x(self, raw):
        if self.model.is_probabilistic:
            return _prep(self.model, raw)
        return ((np.asarray(raw, float) - self._mu) / (self._sd + 1e-9)).reshape(-1, 1)

    def proba(self, df) -> np.ndarray:
        return self.calibrator_.predict_proba(self._x(self.model.raw_score(df)))[:, 1]

    def set_thresholds(self, thr_high: float, medium_ratio: float = 0.2):
        """HIGH = cost-optimal review threshold. MEDIUM = a documented heuristic (a fraction of it), not tuned."""
        self.thr_high, self.thr_medium = float(thr_high), float(thr_high * medium_ratio)
        return self

    def band(self, p) -> np.ndarray:
        p = np.asarray(p)
        return np.where(p >= self.thr_high, "HIGH", np.where(p >= self.thr_medium, "MEDIUM", "LOW"))

    def score(self, df):
        p = self.proba(df)
        return p, self.band(p)
