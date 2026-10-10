"""Reproduce the manuscript's algebraic checks and controlled experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from synthetic_benchmark import make_dataset

from wavelet_clustering import WaveletFeatureExtractor, WaveletKMeans, wavelet_consensus

ROOT = Path(__file__).resolve().parents[1]
WAVELETS = ("haar", "db2", "sym2", "coif1", "sym3")
SEEDS = tuple(range(5))
NOISE_LEVELS = (0.2, 0.8, 1.6, 3.2, 6.4, 12.8)


def write_table(out, name, rows):
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def standardized(X):
    means = X.mean(axis=1, keepdims=True)
    scales = X.std(axis=1, keepdims=True)
    return (X - means) / np.where(scales == 0, 1, scales)


def dimensions(out):
    rows = []
    for size in (512, 846):
        for wavelet in WAVELETS:
            extractor = WaveletFeatureExtractor(wavelet).fit(np.zeros((1, size)))
            lengths = extractor.coefficient_lengths_
            rows.append({
                "timestamps": size,
                "wavelet": wavelet,
                "filter_length": pywt.Wavelet(wavelet).dec_len,
                "level": extractor.decomposition_level_,
                "c0": lengths[0],
                "d0": lengths[1],
                "d1": lengths[2],
                "features": sum(lengths),
            })
    write_table(out, "dimensions", rows)
    return rows


def benchmark(out, X, truth):
    Z = standardized(X)
    rows = []
    for wavelet in ("raw", *WAVELETS):
        if wavelet == "raw":
            features = Z
            labels = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(Z)
        else:
            model = WaveletKMeans(wavelet=wavelet).fit(X)
            features, labels = model.wavelet_features_, model.labels_
        rows.append({
            "method": wavelet,
            "features": features.shape[1],
            "ari": adjusted_rand_score(truth, labels),
            "silhouette_feature_space": silhouette_score(features, labels),
            "silhouette_common_raw_space": silhouette_score(Z, labels),
        })
    write_table(out, "reference_benchmark", rows)
    return rows


def algebraic_checks(out, X):
    checks, metrics = {}, {}
    extractor = WaveletFeatureExtractor().fit(X)
    features = extractor.transform(X)
    rng = np.random.default_rng(20261010)
    scales = rng.uniform(0.3, 3, (len(X), 1))
    offsets = rng.uniform(-15, 15, (len(X), 1))
    transformed = extractor.transform(scales * X + offsets)
    metrics["positive_affine_max_error"] = float(
        np.max(np.abs(features - transformed))
    )
    checks["positive_affine_invariance"] = (
        metrics["positive_affine_max_error"] < 1e-10
    )
    constant = extractor.transform(np.full((2, X.shape[1]), 17.0))
    checks["constant_series_map_to_zero"] = bool(np.all(constant == 0))
    trends = extractor.inverse_transform(features)
    mean_trend = extractor.inverse_transform(features.mean(axis=0, keepdims=True))
    metrics["centroid_linearity_max_error"] = float(
        np.max(np.abs(mean_trend - trends.mean(axis=0)))
    )
    checks["reconstruction_commutes_with_average"] = (
        metrics["centroid_linearity_max_error"] < 1e-12
    )
    model = WaveletKMeans().fit(X)
    original = model.reconstruct_trends(X, original_scale=True)
    affine = model.reconstruct_trends(
        scales * X + offsets, original_scale=True
    )
    metrics["original_scale_max_error"] = float(
        np.max(np.abs(affine - (scales * original + offsets)))
    )
    checks["original_scale_reconstruction"] = (
        metrics["original_scale_max_error"] < 1e-10
    )
    # Two signals with identical coarse features but opposite fine details.
    size = 256
    coarse = np.r_[np.ones(size // 2), -np.ones(size // 2)]
    fine = np.tile([1.0, -1.0], size // 2)
    pair = np.vstack((coarse + fine, coarse - fine))
    pair_features = WaveletFeatureExtractor("haar").fit_transform(pair)
    metrics["discarded_detail_feature_distance"] = float(
        np.linalg.norm(pair_features[0] - pair_features[1])
    )
    metrics["discarded_detail_signal_distance"] = float(
        np.linalg.norm(standardized(pair)[0] - standardized(pair)[1])
    )
    checks["discarded_detail_collision"] = (
        metrics["discarded_detail_feature_distance"] < 1e-12
        and metrics["discarded_detail_signal_distance"] > 1
    )

    rows = []
    for wavelet, mode in (
        ("haar", "periodization"),
        ("sym2", "periodization"),
        ("sym2", "zero"),
        ("bior2.2", "zero"),
    ):
        ext = WaveletFeatureExtractor(
            wavelet=wavelet, mode=mode, standardize=False
        ).fit(np.eye(128))
        analysis = ext.transform(np.eye(128)).T
        synthesis = ext.inverse_transform(np.eye(ext.n_output_features_)).T
        projection = synthesis @ analysis
        row = {
            "wavelet": wavelet,
            "mode": mode,
            "features": ext.n_output_features_,
            "synthesis_gram_defect": float(np.linalg.norm(
                synthesis.T @ synthesis - np.eye(ext.n_output_features_)
            )),
            "idempotence_defect": float(np.linalg.norm(
                projection @ projection - projection
            )),
            "symmetry_defect": float(np.linalg.norm(projection - projection.T)),
        }
        rows.append(row)
        if mode == "periodization":
            checks[f"{wavelet}_orthogonal_periodic_case"] = (
                row["synthesis_gram_defect"] < 1e-8
                and row["idempotence_defect"] < 1e-8
                and row["symmetry_defect"] < 1e-8
            )
    write_table(out, "finite_geometry", rows)
    return checks, metrics


def sensitivity(out):
    noise_rows, option_rows = [], []
    for seed in SEEDS:
        for noise in NOISE_LEVELS:
            X, truth = make_dataset(random_state=seed, noise_scale=noise)
            for wavelet in ("raw", "haar", "sym2", "sym3"):
                if wavelet == "raw":
                    labels = KMeans(
                        n_clusters=3, n_init=20, random_state=0
                    ).fit_predict(standardized(X))
                else:
                    labels = WaveletKMeans(wavelet=wavelet).fit_predict(X)
                noise_rows.append({
                    "seed": seed, "noise_scale": noise, "method": wavelet,
                    "ari": adjusted_rand_score(truth, labels),
                })
        X, truth = make_dataset(random_state=seed, noise_scale=1.6)
        for mode in ("zero", "symmetric", "periodization"):
            for levels in (1, 2, 3, 4):
                model = WaveletKMeans(
                    mode=mode, n_detail_levels=levels
                ).fit(X)
                option_rows.append({
                    "seed": seed, "mode": mode, "detail_levels": levels,
                    "features": model.n_wavelet_features_,
                    "ari": adjusted_rand_score(truth, model.labels_),
                })
    write_table(out, "noise_sweep", noise_rows)
    write_table(out, "boundary_and_levels", option_rows)
    plt.rcParams.update({
        "svg.fonttype": "none", "svg.hashsalt": "wavelet-trend-clustering",
        "font.size": 10,
    })
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    aggregates = []
    for name in ("raw", "haar", "sym2", "sym3"):
        values = []
        for noise in NOISE_LEVELS:
            scores = [
                r["ari"] for r in noise_rows
                if r["method"] == name and r["noise_scale"] == noise
            ]
            values.append((np.mean(scores), np.std(scores, ddof=1)))
            aggregates.append({
                "method": name, "noise_scale": noise,
                "ari_mean": float(values[-1][0]),
                "ari_std": float(values[-1][1]),
                "ari_min": float(min(scores)), "ari_max": float(max(scores)),
            })
        values = np.asarray(values)
        ax.plot(NOISE_LEVELS, values[:, 0], "o-", label=name)
        ax.fill_between(
            NOISE_LEVELS, values[:, 0] - values[:, 1],
            values[:, 0] + values[:, 1], alpha=0.12,
        )
    ax.set(
        xlabel="Gaussian noise standard deviation",
        ylabel="Adjusted Rand index",
        title="Noise sensitivity: five dataset seeds",
        ylim=(-0.05, 1.06),
    )
    ax.grid(alpha=0.2)
    ax.legend(title="Shading: one standard deviation", loc="lower left")
    fig.savefig(out / "noise_sensitivity.svg", metadata={"Date": None})
    plt.close(fig)
    write_table(out, "noise_summary", aggregates)
    return aggregates


def interpretation(out, X, truth):
    result = wavelet_consensus(X)
    matrix = result.agreement
    metrics = {
        "consensus_min_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
        "consensus_symmetry_defect": float(np.max(np.abs(matrix - matrix.T))),
    }
    checks = {
        "consensus_unit_diagonal": bool(np.all(np.diag(matrix) == 1)),
        "consensus_symmetric": metrics["consensus_symmetry_defect"] < 1e-12,
        "consensus_positive_semidefinite": (
            metrics["consensus_min_eigenvalue"] > -1e-10
        ),
    }
    np.savetxt(out / "consensus_matrix.csv", matrix, delimiter=",")
    order = np.argsort(truth, kind="stable")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    im = ax.imshow(matrix[np.ix_(order, order)], vmin=0, vmax=1, cmap="Blues")
    ax.set(
        title="Co-clustering frequency across five wavelets",
        xlabel="Series sorted by known synthetic class", ylabel="Series",
    )
    fig.colorbar(im, ax=ax, label="Fraction of agreeing partitions")
    fig.savefig(out / "wavelet_consensus.svg", metadata={"Date": None})
    plt.close(fig)

    ext = WaveletFeatureExtractor().fit(X)
    atoms = ext.inverse_transform(np.eye(ext.n_output_features_))
    starts = np.cumsum((0, *ext.coefficient_lengths_))
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), layout="constrained")
    for band, ax in enumerate(axes):
        index = int(starts[band] + ext.coefficient_lengths_[band] // 2)
        ax.plot(atoms[index], color="#087f8c")
        ax.set(title=ext.get_feature_names_out()[index], ylabel="Unit response")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Timestamp index")
    fig.suptitle("Single-coefficient synthesis atoms (sym2, zero padding)")
    fig.savefig(out / "coefficient_atoms.svg", metadata={"Date": None})
    plt.close(fig)
    return checks, metrics


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    X, truth = make_dataset()
    dims = dimensions(out)
    baseline = benchmark(out, X, truth)
    checks, metrics = algebraic_checks(out, X)
    noise = sensitivity(out)
    more_checks, more_metrics = interpretation(out, X, truth)
    checks.update(more_checks)
    metrics.update(more_metrics)
    for wavelet, expected in (("haar", 8), ("sym2", 21), ("sym3", 40)):
        checks[f"{wavelet}_846_dimension"] = any(
            row["wavelet"] == wavelet and row["timestamps"] == 846
            and row["features"] == expected for row in dims
        )
    checks["reference_synthetic_labels"] = all(r["ari"] > 0.95 for r in baseline)
    summary = {
        "protocol": {
            "n_series": len(X), "timestamps": X.shape[1],
            "reference_dataset_seed": 42, "kmeans_seed": 0, "n_init": 20,
            "sweep_dataset_seeds": list(SEEDS), "noise_levels": NOISE_LEVELS,
        },
        "metrics": metrics, "reference": baseline, "noise_summary": noise,
        "checks": {key: bool(value) for key, value in checks.items()},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    sources = sorted((ROOT / "src").rglob("*.py"))
    sources += sorted((ROOT / "examples").glob("*.py"))
    manifest = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pywavelets": pywt.__version__, "scikit_learn": sklearn.__version__,
        "platform": platform.platform(), "commit": os.environ.get("GITHUB_SHA"),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    if not all(checks.values()):
        raise SystemExit("Failed: " + ", ".join(k for k, v in checks.items() if not v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "validation"
    )
    run(parser.parse_args().output_dir)
