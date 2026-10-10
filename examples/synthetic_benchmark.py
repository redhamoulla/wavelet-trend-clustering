"""Reproducible benchmark on increasing, flat and seasonal time series."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from wavelet_clustering import WaveletKMeans


def make_dataset(
    n_per_cluster: int = 50,
    n_timestamps: int = 512,
    random_state: int = 42,
    noise_scale: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build noisy series with amplitude/offset nuisance variation."""
    rng = np.random.default_rng(random_state)
    time = np.linspace(0, 1, n_timestamps)
    prototypes = {
        "increasing": 4.0 * time,
        "stagnating": -0.4 * time,
        "seasonal": 2.5 * np.sin(2 * np.pi * time - np.pi / 2),
    }
    X, y = [], []
    for label, prototype in enumerate(prototypes.values()):
        for _ in range(n_per_cluster):
            scale = rng.uniform(0.6, 1.5)
            offset = rng.uniform(-20, 20)
            local = rng.normal(scale=noise_scale, size=n_timestamps)
            high_frequency = 0.4 * np.sin(
                2 * np.pi * rng.integers(20, 50) * time + rng.uniform(0, 2 * np.pi)
            )
            X.append(offset + scale * (prototype + local + high_frequency))
            y.append(label)
    return np.asarray(X), np.asarray(y)


def run(output_dir: Path) -> list[dict[str, object]]:
    """Run raw and wavelet baselines and save tabular/visual results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X, truth = make_dataset()
    normalized = StandardScaler().fit_transform(X.T).T

    rows: list[dict[str, object]] = []
    raw = KMeans(n_clusters=3, n_init=20, random_state=0).fit(normalized)
    rows.append(
        {
            "method": "raw standardized samples",
            "features": normalized.shape[1],
            "adjusted_rand_index": adjusted_rand_score(truth, raw.labels_),
            "silhouette": silhouette_score(normalized, raw.labels_),
        }
    )

    models = {}
    for wavelet in ("haar", "db2", "sym2", "coif1", "sym3"):
        model = WaveletKMeans(wavelet=wavelet, random_state=0).fit(X)
        models[wavelet] = model
        rows.append(
            {
                "method": f"wavelet ({wavelet})",
                "features": model.n_wavelet_features_,
                "adjusted_rand_index": adjusted_rand_score(truth, model.labels_),
                "silhouette": silhouette_score(
                    model.wavelet_features_, model.labels_
                ),
            }
        )

    with (output_dir / "synthetic_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    model = models["sym2"]
    trends = model.reconstruct_trends(X)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    names = ("increasing", "stagnating", "seasonal")
    for label, (axis, name) in enumerate(zip(axes, names)):
        subset = trends[truth == label]
        axis.plot(subset.T, color="#4C78A8", alpha=0.12, linewidth=0.8)
        axis.plot(subset.mean(axis=0), color="#E45756", linewidth=2.5)
        axis.set_title(name.capitalize())
        axis.set_xlabel("Time")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Reconstructed standardized trend")
    figure.suptitle("Low-frequency trends retained by WaveletKMeans (sym2)")
    figure.tight_layout()
    figure.savefig(output_dir / "synthetic_trends.png", dpi=180)
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    rows = run(args.output_dir)
    for row in rows:
        print(
            f"{row['method']:<28} features={row['features']:<4} "
            f"ARI={row['adjusted_rand_index']:.3f} "
            f"silhouette={row['silhouette']:.3f}"
        )


if __name__ == "__main__":
    main()
