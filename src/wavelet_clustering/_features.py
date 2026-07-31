"""Low-frequency discrete-wavelet features for aligned time series."""

from __future__ import annotations

import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class WaveletFeatureExtractor(TransformerMixin, BaseEstimator):
    """Extract the coarsest DWT approximation and detail coefficients.

    The implementation follows Talbo et al. (2020): each time series is
    standardized independently, decomposed at the maximum admissible level,
    and represented by ``c0, d0, ..., d(n_detail_levels-1)``. In PyWavelets'
    ordering these are the first ``1 + n_detail_levels`` arrays returned by
    :func:`pywt.wavedec`.

    Parameters
    ----------
    wavelet : str, default="sym2"
        Name of a discrete mother wavelet known to PyWavelets.
    n_detail_levels : int, default=2
        Number of coarsest detail bands to retain in addition to the coarsest
        approximation. The paper uses 2, corresponding to ``(c0, d0, d1)``.
    mode : str, default="zero"
        PyWavelets boundary-extension mode. The paper uses zero-padding.
    standardize : bool, default=True
        If true, z-normalize every series independently before the DWT.
    """

    def __init__(
        self,
        wavelet: str = "sym2",
        n_detail_levels: int = 2,
        mode: str = "zero",
        standardize: bool = True,
    ) -> None:
        self.wavelet = wavelet
        self.n_detail_levels = n_detail_levels
        self.mode = mode
        self.standardize = standardize

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """Validate the input shape and learn the coefficient layout."""
        del y
        X = self._validate_X(X)
        wavelet = self._validate_parameters()
        decomposition_level = pywt.dwt_max_level(X.shape[1], wavelet.dec_len)
        if decomposition_level < self.n_detail_levels:
            raise ValueError(
                "The time series are too short for n_detail_levels="
                f"{self.n_detail_levels} with wavelet={self.wavelet!r}; "
                f"the maximum decomposition level is {decomposition_level}."
            )

        template = pywt.wavedec(
            np.zeros(X.shape[1], dtype=float),
            wavelet,
            mode=self.mode,
            level=decomposition_level,
        )
        retained = template[: 1 + self.n_detail_levels]
        lengths = tuple(len(part) for part in retained)

        self.n_features_in_ = X.shape[1]
        self.decomposition_level_ = decomposition_level
        self.coefficient_lengths_ = lengths
        self._all_coefficient_lengths_ = tuple(len(part) for part in template)
        self.n_output_features_ = sum(lengths)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return low-frequency wavelet features for each input series."""
        check_is_fitted(self, "n_output_features_")
        X = self._validate_X(X, reset=False)
        X_normalized, _, _ = self._normalize(X)
        rows = [self._decompose(row) for row in X_normalized]
        return np.vstack(rows)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct standardized trends from retained coefficients.

        Fine-scale detail coefficients discarded by :meth:`transform` are set
        to zero. The result therefore contains only the trend information used
        by the clustering model.
        """
        check_is_fitted(self, "n_output_features_")
        features = check_array(X, ensure_2d=True, dtype=np.float64)
        if features.shape[1] != self.n_output_features_:
            raise ValueError(
                f"Expected {self.n_output_features_} coefficients, got "
                f"{features.shape[1]}."
            )

        reconstructed = []
        for row in features:
            retained = []
            start = 0
            for length in self.coefficient_lengths_:
                retained.append(row[start : start + length])
                start += length
            discarded = [
                np.zeros(length, dtype=float)
                for length in self._all_coefficient_lengths_[len(retained) :]
            ]
            signal = pywt.waverec(
                retained + discarded,
                self.wavelet,
                mode=self.mode,
            )
            reconstructed.append(signal[: self.n_features_in_])
        return np.vstack(reconstructed)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return names such as ``c0_0``, ``d0_0`` and ``d1_0``."""
        del input_features
        check_is_fitted(self, "coefficient_lengths_")
        names = []
        for band, length in enumerate(self.coefficient_lengths_):
            prefix = "c0" if band == 0 else f"d{band - 1}"
            names.extend(f"{prefix}_{index}" for index in range(length))
        return np.asarray(names, dtype=object)

    def normalize(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized series and their row-wise means and scales."""
        X = self._validate_X(X, reset=False)
        return self._normalize(X)

    def _decompose(self, row: np.ndarray) -> np.ndarray:
        coefficients = pywt.wavedec(
            row,
            self.wavelet,
            mode=self.mode,
            level=self.decomposition_level_,
        )
        return np.concatenate(coefficients[: 1 + self.n_detail_levels])

    def _normalize(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = X.mean(axis=1, keepdims=True)
        scales = X.std(axis=1, keepdims=True)
        safe_scales = np.where(scales == 0.0, 1.0, scales)
        if self.standardize:
            normalized = (X - means) / safe_scales
        else:
            normalized = X.copy()
            means = np.zeros_like(means)
            safe_scales = np.ones_like(safe_scales)
        return normalized, means, safe_scales

    def _validate_X(self, X: np.ndarray, *, reset: bool = True) -> np.ndarray:
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")
        if not reset and hasattr(self, "n_features_in_"):
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Expected {self.n_features_in_} timestamps, got {X.shape[1]}."
                )
        return X

    def _validate_parameters(self) -> pywt.Wavelet:
        if not isinstance(self.n_detail_levels, int) or self.n_detail_levels < 1:
            raise ValueError("n_detail_levels must be an integer greater than zero.")
        if self.mode not in pywt.Modes.modes:
            raise ValueError(
                f"Unknown extension mode {self.mode!r}. Choose from {pywt.Modes.modes}."
            )
        try:
            wavelet = pywt.Wavelet(self.wavelet)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"wavelet={self.wavelet!r} is not a discrete PyWavelets wavelet."
            ) from error
        return wavelet
