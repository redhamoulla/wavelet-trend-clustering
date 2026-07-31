"""Scikit-learn compatible wavelet trend clustering estimator."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from ._features import WaveletFeatureExtractor


class WaveletKMeans(ClusterMixin, BaseEstimator):
    """Cluster aligned time series by their low-frequency wavelet content.

    Parameters mirror the method in Talbo et al. (2020), with reproducible
    defaults: Symlet 2, zero-padding, ``(c0, d0, d1)``, row-wise
    standardization and k-means++ initialization.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    wavelet : str, default="sym2"
        Discrete PyWavelets mother wavelet.
    n_detail_levels : int, default=2
        Number of coarsest detail coefficient bands to keep.
    mode : str, default="zero"
        Signal boundary-extension mode.
    standardize : bool, default=True
        Z-normalize each time series independently.
    random_state : int, RandomState instance or None, default=0
        Controls k-means++ initialization.
    n_init : int or "auto", default=20
        Number of k-means initializations.
    max_iter : int, default=300
        Maximum k-means iterations per run.
    tol : float, default=1e-4
        Relative convergence tolerance for k-means.
    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        wavelet: str = "sym2",
        n_detail_levels: int = 2,
        mode: str = "zero",
        standardize: bool = True,
        random_state=0,
        n_init=20,
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
    ) -> None:
        self.n_clusters = n_clusters
        self.wavelet = wavelet
        self.n_detail_levels = n_detail_levels
        self.mode = mode
        self.standardize = standardize
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Extract wavelet features and fit k-means."""
        del y
        self.feature_extractor_ = WaveletFeatureExtractor(
            wavelet=self.wavelet,
            n_detail_levels=self.n_detail_levels,
            mode=self.mode,
            standardize=self.standardize,
        )
        features = self.feature_extractor_.fit_transform(X)
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError("n_clusters must be an integer greater than zero.")
        if features.shape[0] < self.n_clusters:
            raise ValueError(
                "n_clusters cannot exceed the number of input time series."
            )

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm=self.algorithm,
        ).fit(features)
        self.labels_ = self.kmeans_.labels_
        self.cluster_centers_ = self.kmeans_.cluster_centers_
        self.inertia_ = self.kmeans_.inertia_
        self.n_iter_ = self.kmeans_.n_iter_
        self.n_features_in_ = self.feature_extractor_.n_features_in_
        self.n_wavelet_features_ = self.feature_extractor_.n_output_features_
        self.wavelet_features_ = features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign input series to the nearest fitted wavelet-space centroid."""
        features = self.extract_features(X)
        return self.kmeans_.predict(features)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return distances from each series to every cluster centroid."""
        features = self.extract_features(X)
        return self.kmeans_.transform(features)

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        """Return the negative k-means objective on ``X``."""
        del y
        features = self.extract_features(X)
        return self.kmeans_.score(features)

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Expose the low-frequency coefficients used for clustering."""
        check_is_fitted(self, "kmeans_")
        return self.feature_extractor_.transform(X)

    def reconstruct_trends(
        self, X: np.ndarray, *, original_scale: bool = False
    ) -> np.ndarray:
        """Reconstruct the trend retained by the clustering representation.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timestamps)
            Original aligned time series.
        original_scale : bool, default=False
            Restore each input series' mean and scale. By default the returned
            trends remain standardized, as in the paper's visualizations.
        """
        check_is_fitted(self, "kmeans_")
        features = self.feature_extractor_.transform(X)
        trends = self.feature_extractor_.inverse_transform(features)
        if original_scale and self.standardize:
            _, means, scales = self.feature_extractor_.normalize(X)
            trends = trends * scales + means
        return trends

    def get_feature_names_out(self) -> np.ndarray:
        """Return names of retained wavelet coefficients."""
        check_is_fitted(self, "kmeans_")
        return self.feature_extractor_.get_feature_names_out()
