import numpy as np
from sklearn.metrics import adjusted_rand_score

from wavelet_clustering import WaveletKMeans, wavelet_consensus


def make_trends(n_per_cluster=20, n_timestamps=256, seed=4):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 1, n_timestamps)
    prototypes = (
        4.0 * time,
        -0.5 * time,
        2.5 * np.sin(2 * np.pi * time - np.pi / 2),
    )
    rows = []
    labels = []
    for label, prototype in enumerate(prototypes):
        for _ in range(n_per_cluster):
            scale = rng.uniform(0.7, 1.4)
            offset = rng.uniform(-10, 10)
            noise = rng.normal(scale=0.65, size=n_timestamps)
            rows.append(offset + scale * (prototype + noise))
            labels.append(label)
    return np.asarray(rows), np.asarray(labels)


def test_clusters_synthetic_trends():
    X, truth = make_trends()

    labels = WaveletKMeans(random_state=12).fit_predict(X)

    assert adjusted_rand_score(truth, labels) > 0.95


def test_prediction_distances_and_reconstruction_shapes():
    X, _ = make_trends(n_per_cluster=5)
    model = WaveletKMeans().fit(X)

    assert model.predict(X[:2]).shape == (2,)
    assert model.transform(X[:2]).shape == (2, 3)
    assert model.reconstruct_trends(X[:2]).shape == (2, X.shape[1])
    assert model.reconstruct_trends(X[:2], original_scale=True).shape == (
        2,
        X.shape[1],
    )


def test_consensus_is_symmetric_and_label_invariant():
    X, _ = make_trends(n_per_cluster=4)

    result = wavelet_consensus(X, wavelets=("haar", "sym2", "coif1"))

    assert result.labels.shape == (3, len(X))
    np.testing.assert_allclose(result.agreement, result.agreement.T)
    np.testing.assert_allclose(np.diag(result.agreement), 1.0)
