# Wavelet Trend Clustering

![From noisy time series to wavelet coefficients and trend clusters](docs/assets/overview.svg)

[![CI](https://github.com/redhamoulla/wavelet-trend-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/redhamoulla/wavelet-trend-clustering/actions/workflows/ci.yml)
[![Scientific validation](https://github.com/redhamoulla/wavelet-trend-clustering/actions/workflows/scientific-validation.yml/badge.svg)](https://github.com/redhamoulla/wavelet-trend-clustering/actions/workflows/scientific-validation.yml)

Cluster aligned time series by their coarse wavelet structure. The package
z-normalizes each series, retains the coarsest approximation and detail bands,
and fits k-means with k-means++ initialization. Reconstructed trends make the
representation easier to inspect.

Based on Talbo, Haddab, Aubert and Moulla (2020),
[*Wavelet-based clustering for time-series trend detection*](https://arxiv.org/abs/2011.12111).

**[Read the scientific manuscript →](docs/manuscript.md)**  
Theory, derivations, implementation choices, experiments, limitations and
references are collected in this single document.

## Repository guide

| Location | Contents |
|---|---|
| [Scientific manuscript](docs/manuscript.md) | Complete scientific presentation |
| [Python package](src/wavelet_clustering) | Feature extraction, clustering and reconstruction |
| [Examples](examples) | Runnable benchmark and validation protocol |
| [Recorded results](artifacts/validation) | CSV data, SVG figures, checks and environment manifest |
| [Tests](tests) | Estimator behavior and dimensionality regression tests |
| [Workflows](.github/workflows) | Tests, lint, package build and scientific validation |

## Install and use

Python 3.9 or newer:

```bash
git clone https://github.com/redhamoulla/wavelet-trend-clustering.git
cd wavelet-trend-clustering
python -m pip install -e ".[dev]"
```

Given a finite array `X` with shape `(n_series, n_timestamps)` and a common
time grid:

```python
from wavelet_clustering import WaveletKMeans, wavelet_consensus

model = WaveletKMeans(n_clusters=3, wavelet="sym2", random_state=0)
labels = model.fit_predict(X)
features = model.wavelet_features_
trends = model.reconstruct_trends(X)
trends_in_original_units = model.reconstruct_trends(X, original_scale=True)

# Compare partitions obtained with the default five wavelets.
agreement = wavelet_consensus(X).agreement
```

For new series on the same grid, use `model.predict(X_new)`.
`model.transform(X)` returns distances to centroids; use
`model.extract_features(X)` for the retained wavelet coefficients.
See the [API and usage notes](docs/manuscript.md#8-api-and-practical-use)
for reconstruction of cluster centers and interpretation.

## Reproduce the results

```bash
python examples/synthetic_benchmark.py --output-dir artifacts
python examples/scientific_validation.py --output-dir artifacts/validation
pytest
ruff check .
python -m build
```

The reference experiment contains 150 synthetic series of length 512. Raw
k-means and all five wavelet variants recover its three classes exactly.
The wavelet representations use 4–44 features. Silhouettes calculated in
different feature spaces measure different geometries; their increase is
**not evidence of a better recovered partition**.

The extended protocol includes a noise sweep, boundary and retained-level
comparisons, a discarded-information counterexample, reconstruction checks
and wavelet consensus. Its results and assumptions are discussed in the
[experimental section](docs/manuscript.md#7-reproducible-experiments).
The original retail data are not included, so these experiments validate
the public implementation on synthetic data.

## Citation and license

> V. Talbo, M. Haddab, D. Aubert, and R. Moulla (2020).  
> *Wavelet-based clustering for time-series trend detection*.  
> arXiv:2011.12111.

Machine-readable citation: [CITATION.cff](CITATION.cff).  
Code license: [BSD 3-Clause](LICENSE).
