"""Diagnostics for comparing clustering stability across mother wavelets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._estimator import WaveletKMeans


@dataclass(frozen=True)
class ConsensusResult:
    """Cluster assignments and pairwise agreement across mother wavelets."""

    wavelets: tuple[str, ...]
    labels: np.ndarray
    agreement: np.ndarray


def wavelet_consensus(
    X: np.ndarray,
    *,
    wavelets: Sequence[str] = ("haar", "db2", "sym2", "coif1", "sym3"),
    n_clusters: int = 3,
    n_detail_levels: int = 2,
    random_state: int | None = 0,
    n_init: int = 20,
) -> ConsensusResult:
    """Fit several wavelets and compute label-permutation-invariant agreement.

    ``agreement[i, j]`` is the fraction of mother wavelets for which series
    ``i`` and ``j`` are assigned to the same cluster, reproducing the
    correlation-style stability matrix in the paper.
    """
    if not wavelets:
        raise ValueError("wavelets must contain at least one wavelet name.")
    labels = []
    for wavelet in wavelets:
        estimator = WaveletKMeans(
            n_clusters=n_clusters,
            wavelet=wavelet,
            n_detail_levels=n_detail_levels,
            random_state=random_state,
            n_init=n_init,
        )
        labels.append(estimator.fit_predict(X))
    label_matrix = np.vstack(labels)
    agreement = np.mean(
        label_matrix[:, :, np.newaxis] == label_matrix[:, np.newaxis, :], axis=0
    )
    return ConsensusResult(tuple(wavelets), label_matrix, agreement)
