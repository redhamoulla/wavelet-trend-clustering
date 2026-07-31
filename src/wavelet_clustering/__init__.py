"""Wavelet-based clustering for time-series trend detection."""

from ._diagnostics import ConsensusResult, wavelet_consensus
from ._estimator import WaveletKMeans
from ._features import WaveletFeatureExtractor

__all__ = [
    "ConsensusResult",
    "WaveletFeatureExtractor",
    "WaveletKMeans",
    "wavelet_consensus",
]
__version__ = "0.1.0"
