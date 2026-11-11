"""Supervised classifiers, unsupervised anomaly detection and clustering behind one small interface."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureBuilder

KINDS = ("logreg", "hgb", "iforest", "kmeans")


class FraudModel:
    """kind: logreg | hgb (supervised) | iforest (anomaly score) | kmeans (distance to nearest centroid).

    anomaly_features=True feeds the unsupervised scores into the gradient-boosting model as extra
    features (stacking); those scorers are fitted on the training split only.
    """

    def __init__(self, kind="hgb", engineered=True, anomaly_features=False, class_weight=None,
                 n_clusters=8, seed=42, builder_factory=None, n_trees=300, km_n_init=5):
        assert kind in KINDS, kind
        self.kind, self.engineered, self.anomaly_features = kind, engineered, anomaly_features
        self.class_weight, self.n_clusters, self.seed = class_weight, n_clusters, seed
        self.builder_factory = builder_factory  # callable -> object with fit(df) / transform(df); default = ULB builder
        self.n_trees, self.km_n_init = n_trees, km_n_init

    # -- unsupervised helpers -------------------------------------------------
    def _fit_unsup(self, x):
        self.scaler_ = StandardScaler().fit(x)
        xs = self.scaler_.transform(x)
        if self.kind in ("iforest", "hgb") and (self.kind == "iforest" or self.anomaly_features):
            self.iso_ = IsolationForest(n_estimators=self.n_trees, random_state=self.seed, n_jobs=-1).fit(xs)
        if self.kind == "kmeans" or self.anomaly_features:
            self.km_ = KMeans(n_clusters=self.n_clusters, n_init=self.km_n_init, random_state=self.seed).fit(xs)

    def _anomaly_cols(self, x):
        xs = self.scaler_.transform(x)
        x = x.copy()
        x["iso_score"] = -self.iso_.score_samples(xs)
        x["km_dist"] = self.km_.transform(xs).min(axis=1)
        return x

    # -- public API -------------------------------------------------------------
    def fit(self, df, y):
        self.fb_ = (self.builder_factory() if self.builder_factory else FeatureBuilder(self.engineered)).fit(df)
        x = self.fb_.transform(df)
        if self.kind == "logreg":
            self.est_ = make_pipeline(StandardScaler(), LogisticRegression(
                C=0.1, max_iter=2000, class_weight=self.class_weight or "balanced")).fit(x, y)
        elif self.kind == "hgb":
            if self.anomaly_features:
                self._fit_unsup(x)
                x = self._anomaly_cols(x)
            self.est_ = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                class_weight=self.class_weight, random_state=self.seed).fit(x, y)
        else:  # iforest / kmeans: labels are not used
            self._fit_unsup(x)
        return self

    def raw_score(self, df) -> np.ndarray:
        x = self.fb_.transform(df)
        if self.kind == "logreg":
            return self.est_.predict_proba(x)[:, 1]
        if self.kind == "hgb":
            if self.anomaly_features:
                x = self._anomaly_cols(x)
            return self.est_.predict_proba(x)[:, 1]
        xs = self.scaler_.transform(x)
        if self.kind == "iforest":
            return -self.iso_.score_samples(xs)
        return self.km_.transform(xs).min(axis=1)

    @property
    def is_probabilistic(self) -> bool:
        return self.kind in ("logreg", "hgb")
