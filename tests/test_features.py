import numpy as np
import pytest

from wavelet_clustering import WaveletFeatureExtractor


@pytest.mark.parametrize(
    ("wavelet", "expected"),
    [("haar", 8), ("sym2", 21), ("sym3", 40)],
)
def test_paper_coefficient_counts(wavelet, expected):
    X = np.zeros((3, 846))
    features = WaveletFeatureExtractor(wavelet=wavelet).fit_transform(X)

    assert features.shape == (3, expected)


def test_feature_names_match_output_width():
    extractor = WaveletFeatureExtractor(wavelet="sym2").fit(np.ones((2, 128)))

    names = extractor.get_feature_names_out()

    assert len(names) == extractor.n_output_features_
    assert names[0].startswith("c0_")
    assert any(name.startswith("d1_") for name in names)


def test_inverse_transform_has_input_length_and_removes_fine_noise():
    rng = np.random.default_rng(3)
    time = np.linspace(0, 1, 256)
    clean = 3 * time
    noisy = clean + rng.normal(scale=0.8, size=time.size)
    extractor = WaveletFeatureExtractor().fit(noisy[None, :])

    normalized, _, _ = extractor.normalize(noisy[None, :])
    reconstructed = extractor.inverse_transform(extractor.transform(noisy[None, :]))

    assert reconstructed.shape == (1, noisy.size)
    assert np.mean((reconstructed[0] - normalized[0]) ** 2) > 0
    assert np.var(np.diff(reconstructed[0])) < np.var(np.diff(normalized[0]))


def test_rejects_wrong_width_and_non_finite_values():
    extractor = WaveletFeatureExtractor().fit(np.ones((2, 128)))

    with pytest.raises(ValueError, match="timestamps"):
        extractor.transform(np.ones((2, 127)))
    bad = np.ones((2, 128))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        extractor.transform(bad)
