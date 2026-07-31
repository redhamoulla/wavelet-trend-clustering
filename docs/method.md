# Method and implementation choices

This package implements the workflow described by Talbo, Haddab, Aubert and
Moulla in *Wavelet-based clustering for time-series trend detection* (2020).

For an input matrix with one aligned time series per row, the estimator:

1. z-normalizes each row independently;
2. computes the maximum admissible DWT level
   `floor(log2(N / (filter_length - 1)))` through `pywt.dwt_max_level`;
3. uses zero-padding at the signal boundaries;
4. retains the coarsest approximation and two coarsest detail bands,
   `(c0, d0, d1)`;
5. runs k-means with k-means++ initialization on the concatenated coefficients.

The defaults deliberately fix `random_state=0` and use 20 k-means restarts for
reproducibility. These are engineering choices; the article specifies
k-means++ but does not prescribe one random seed.

## Correspondence with PyWavelets

`pywt.wavedec` returns `[cA_J, cD_J, cD_(J-1), ..., cD_1]`. The paper numbers
the coarsest bands from zero, so `(c0, d0, d1)` corresponds to the first three
arrays returned by PyWavelets. For 846 timestamps, this produces the exact
feature counts reported in the article:

| Filter length | Example wavelet | Retained features |
|---:|---|---:|
| 2 | Haar | 8 |
| 4 | Symlet 2 | 21 |
| 6 | Symlet 3 | 40 |

The article's Table I reports 14 coefficients for filter-length-2 wavelets.
However, applying its stated maximum-level formula to 846 samples with
PyWavelets and zero-padding gives lengths `(2, 2, 4)`, hence 8 retained
coefficients. No valid DWT level yields 14 coefficients for all three retained
bands. The implementation follows the published formula and records this
discrepancy rather than special-casing Haar.

## Scope

The method assumes equal-length, time-aligned, finite, univariate series. It
does not impute missing values or align timestamps automatically. Those steps
are data-dependent and should happen before fitting.

Cluster identifiers are arbitrary. Use external domain knowledge or inspect
the reconstructed cluster trends before assigning semantic names such as
"increasing" or "seasonal".
