import numpy as np
import pytest

from src.experiment_runner import _summarize, polynomial_baseline_metrics
from src.random_feature_model import (
    RandomFeatureConfig,
    default_capacity_grid,
    evaluate_random_feature_regressor,
    fit_random_feature_regressor,
    generate_synthetic_regression_data,
    matrix_diagnostics,
    random_relu_features,
    sweep_random_feature_widths,
)


def test_data_generation_is_reproducible():
    first = generate_synthetic_regression_data(n_samples=80, noise_std=0.1, seed=7)
    second = generate_synthetic_regression_data(n_samples=80, noise_std=0.1, seed=7)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_default_capacity_grid_brackets_sample_count():
    widths = default_capacity_grid(160)
    assert min(widths) < 160
    assert max(widths) > 160
    assert len(widths) >= 10
    assert widths == sorted(set(widths))


def test_ols_matches_numpy_least_squares_solution():
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = np.cos(X[:, 0])
    model = fit_random_feature_regressor(X, y, hidden_width=8, seed=4, solver="ols")
    design = random_relu_features(X, model["W"], model["b"])
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(model["coefficients"], expected, rtol=1e-10, atol=1e-10)


def test_ridge_uses_augmented_least_squares_without_penalizing_intercept():
    X = np.linspace(-1, 1, 25).reshape(-1, 1)
    y = np.sin(X[:, 0])
    alpha = 0.25
    model = fit_random_feature_regressor(
        X, y, hidden_width=5, seed=2, solver="ridge", ridge_alpha=alpha
    )
    design = random_relu_features(X, model["W"], model["b"])
    penalty = np.eye(design.shape[1]) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    expected = np.linalg.lstsq(
        np.vstack((design, penalty)),
        np.concatenate((y, np.zeros(design.shape[1]))),
        rcond=None,
    )[0]
    np.testing.assert_allclose(model["coefficients"], expected, rtol=1e-10, atol=1e-10)


def test_rank_deficient_design_has_infinite_condition_number():
    design = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    diagnostics = matrix_diagnostics(design)
    assert diagnostics["rank"] == 1
    assert np.isinf(diagnostics["condition_number"])


def test_ill_conditioned_full_rank_design_reports_large_condition():
    design = np.diag([1.0, 1e-10])
    diagnostics = matrix_diagnostics(design)
    assert diagnostics["rank"] == 2
    assert diagnostics["condition_number"] == pytest.approx(1e10)


def test_interpolation_is_measured_not_inferred_from_width():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, size=(20, 1))
    y = rng.normal(size=20)
    result = evaluate_random_feature_regressor(
        X, X, y, y, hidden_width=30, seed=11, solver="ols"
    )
    assert result["n_features"] == 31
    assert result["interpolates"] == (result["train_mse"] <= result["interpolation_tolerance"])
    assert result["rank"] <= len(y)


def test_capacity_with_full_row_rank_reaches_interpolation():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.0, 1.0, size=(24, 1))
    y = rng.normal(size=24)
    result = evaluate_random_feature_regressor(
        X, X, y, y, hidden_width=len(y) - 1, seed=5, solver="ols"
    )
    assert result["rank"] == len(y)
    assert result["interpolates"]
    assert result["train_mse"] <= result["interpolation_tolerance"]


def test_sweep_repeats_deterministically_and_reports_solver_config():
    config = RandomFeatureConfig(
        n_samples=60,
        noise_std=0.1,
        seed=7,
        hidden_widths=[8, 16],
        solver="ridge",
        ridge_alpha=0.01,
    )
    first = sweep_random_feature_widths(config)
    second = sweep_random_feature_widths(config)
    assert first == second
    assert all(row["solver"] == "ridge" for row in first)
    assert all(row["ridge_alpha"] == 0.01 for row in first)


def test_summary_has_mean_std_and_student_t_confidence_interval():
    rows = [
        {"setting": "a", "mse": 1.0},
        {"setting": "a", "mse": 3.0},
        {"setting": "a", "mse": 5.0},
    ]
    summary = _summarize(rows, ["setting"], ["mse"])[0]
    assert summary["n_runs"] == 3
    assert summary["mse_mean"] == pytest.approx(3.0)
    assert summary["mse_std"] == pytest.approx(2.0)
    assert summary["mse_ci95_low"] < summary["mse_mean"] < summary["mse_ci95_high"]
    assert summary == _summarize(rows, ["setting"], ["mse"])[0]


def test_legendre_basis_is_better_conditioned_than_raw_monomials():
    raw = polynomial_baseline_metrics(
        n_samples=100, noise_std=0.0, seed=3, max_degree=15, representation="raw_monomial"
    )[-1]
    legendre = polynomial_baseline_metrics(
        n_samples=100, noise_std=0.0, seed=3, max_degree=15, representation="legendre"
    )[-1]
    assert legendre["condition_number"] < raw["condition_number"]


def test_invalid_solver_configuration_is_rejected():
    with pytest.raises(ValueError):
        RandomFeatureConfig(solver="unknown")
    with pytest.raises(ValueError):
        fit_random_feature_regressor(
            np.ones((4, 1)), np.ones(4), hidden_width=3, solver="ridge"
        )
