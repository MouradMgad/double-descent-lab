import numpy as np
import pytest

from src.phase2_runner import (
    classify_capacity_curve,
    make_synthetic_dataset,
    prepare_diabetes_split,
    real_diabetes_external_validation,
)


def test_synthetic_targets_are_reproducible_and_dimension_aware():
    first = make_synthetic_dataset(
        "additive_sinusoid", input_dim=5, n_samples=60, noise_std=0.1, seed=17
    )
    second = make_synthetic_dataset(
        "additive_sinusoid", input_dim=5, n_samples=60, noise_std=0.1, seed=17
    )
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert first[0].shape[1] == 5


def test_synthetic_target_generates_distinct_signal_families():
    sinusoid = make_synthetic_dataset(
        "single_sinusoid", input_dim=2, n_samples=80, noise_std=0.0, seed=5
    )
    additive = make_synthetic_dataset(
        "additive_sinusoid", input_dim=2, n_samples=80, noise_std=0.0, seed=5
    )
    interaction = make_synthetic_dataset(
        "interaction", input_dim=2, n_samples=80, noise_std=0.0, seed=5
    )
    assert not np.allclose(sinusoid[2], additive[2])
    assert not np.allclose(sinusoid[2], interaction[2])
    assert not np.allclose(additive[2], interaction[2])


def test_interaction_target_requires_two_dimensions():
    with pytest.raises(ValueError, match="requires input_dim >= 2"):
        make_synthetic_dataset(
            "interaction", input_dim=1, n_samples=20, noise_std=0.1, seed=1
        )


def test_curve_classifier_requires_interpolation_and_ci_separation():
    summary = []
    for width, error, rank, interpolation_rate in (
        (4, 1.0, 5, 0.0),
        (7, 2.0, 8, 0.0),
        (8, 100.0, 8, 1.0),
        (9, 10.0, 8, 1.0),
        (16, 4.0, 8, 1.0),
    ):
        summary.append(
            {
                "target": "test",
                "input_dim": 1,
                "noise_std": 0.1,
                "solver": "ols",
                "hidden_width": width,
                "n_train": 8,
                "rank_mean": rank,
                "interpolates_mean": interpolation_rate,
                "test_mse_mean": error,
                "test_mse_ci95_low": error * 0.9,
                "test_mse_ci95_high": error * 1.1,
            }
        )
    result = classify_capacity_curve(summary, "test", 1, 0.1)
    assert result["peak_width"] == 8
    assert result["pointwise_peak_then_descent"]
    assert result["ci_separated_interpolation_peak"]


def test_diabetes_scaler_is_fit_only_on_training_partition():
    prepared = prepare_diabetes_split(seed=42)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train_raw, scaler_mean = prepared
    assert (len(y_train), len(y_val), len(y_test)) == (264, 89, 89)
    np.testing.assert_allclose(scaler_mean, X_train_raw.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(X_test_scaled.mean(axis=0), 0.0, atol=1e-3)
    assert X_val_scaled.shape[1] == X_test_scaled.shape[1] == 10


def test_diabetes_external_validation_selects_without_test_leakage():
    runs, summary = real_diabetes_external_validation(
        seeds=[42, 43],
        widths=[4, 8],
        ridge_alphas=[0.01],
    )
    assert len(runs) == 2
    assert len(summary) == 1
    assert all(run["dataset"] == "sklearn_diabetes" for run in runs)
    assert all(run["test_used_for_selection"] is False for run in runs)
    assert all(run["scaler_fit_rows"] == run["n_train"] for run in runs)
    assert all(run["scaler_train_mean_max_abs_error"] < 1e-12 for run in runs)
    assert all(np.isfinite(run["validation_mse"]) for run in runs)
    assert all(np.isfinite(run["test_mse"]) for run in runs)
    assert summary[0]["n_runs"] == 2


def test_diabetes_external_validation_rejects_empty_seed_list():
    with pytest.raises(ValueError, match="At least two split seeds"):
        real_diabetes_external_validation(seeds=[], widths=[4], ridge_alphas=[0.01])
