from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class RandomFeatureConfig:
    n_samples: int = 200
    noise_std: float = 0.1
    seed: int = 42
    hidden_widths: list[int] | None = None
    test_size: float = 0.2
    solver: str = "ols"
    ridge_alpha: float = 0.0
    interpolation_atol: float = 1e-10
    interpolation_rtol: float = 1e-8

    def __post_init__(self) -> None:
        if self.hidden_widths is None:
            self.hidden_widths = default_capacity_grid(int(self.n_samples * (1 - self.test_size)))
        if self.solver not in {"ols", "ridge"}:
            raise ValueError("solver must be 'ols' or 'ridge'")
        if self.solver == "ridge" and self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive when solver='ridge'")
        if self.n_samples < 3 or self.noise_std < 0:
            raise ValueError("n_samples must be >= 3 and noise_std must be non-negative")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")


def default_capacity_grid(n_train: int) -> list[int]:
    """Cover both sides of n_train, with extra resolution around the threshold."""
    if n_train < 2:
        raise ValueError("n_train must be at least 2")
    widths = {
        max(1, int(round(n_train * fraction)))
        for fraction in (0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25, 1.5, 2.0)
    }
    return sorted(widths)


def generate_synthetic_regression_data(
    n_samples: int = 200, noise_std: float = 0.1, seed: int = 42, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = np.sin(2 * np.pi * x).ravel() + rng.normal(0.0, noise_std, size=n_samples)
    return train_test_split(x, y, test_size=test_size, random_state=seed)


def random_relu_features(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    hidden = np.maximum(np.asarray(X) @ W + b, 0.0)
    return np.column_stack((np.ones(hidden.shape[0]), hidden))


def fit_random_feature_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_width: int,
    seed: int = 0,
    solver: str = "ols",
    ridge_alpha: float = 0.0,
) -> dict[str, Any]:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if X_train.ndim != 2 or X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train must be 2D and have the same number of rows as y_train")
    if hidden_width < 1:
        raise ValueError("hidden_width must be positive")
    if solver not in {"ols", "ridge"}:
        raise ValueError("solver must be 'ols' or 'ridge'")
    if solver == "ridge" and ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive when solver='ridge'")
    if solver == "ols" and ridge_alpha != 0:
        raise ValueError("ridge_alpha must be zero when solver='ols'")

    rng = np.random.default_rng(seed)
    if X_train.shape[1] == 1:
        # Quantile-spaced train-input knots prevent most random ReLUs from being
        # inactive or redundant in this one-dimensional spline experiment.
        slopes = rng.choice(np.array([-1.0, 1.0]), size=hidden_width)
        knots = np.quantile(
            X_train[:, 0], np.linspace(0.0, 1.0, hidden_width + 2)[1:-1]
        )
        W = slopes.reshape(1, -1)
        b = -slopes * knots
    else:
        W = rng.normal(0.0, 1.0, size=(X_train.shape[1], hidden_width))
        b = rng.uniform(-1.0, 1.0, size=(hidden_width,))
    feature_matrix = random_relu_features(X_train, W, b)

    if solver == "ols":
        coefficients, *_ = np.linalg.lstsq(feature_matrix, y_train, rcond=None)
    else:
        # Augmented least squares avoids forming the condition-number-squaring Gram matrix.
        penalty = np.zeros((feature_matrix.shape[1], feature_matrix.shape[1]))
        penalty[1:, 1:] = np.sqrt(ridge_alpha) * np.eye(feature_matrix.shape[1] - 1)
        augmented_design = np.vstack((feature_matrix, penalty))
        augmented_targets = np.concatenate((y_train, np.zeros(feature_matrix.shape[1])))
        coefficients, *_ = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)

    return {
        "W": W,
        "b": b,
        "coefficients": coefficients,
        "hidden_width": int(hidden_width),
        "solver": solver,
        "ridge_alpha": float(ridge_alpha),
        "fit_intercept": True,
    }


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, int | float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return {"rank": 0, "rank_tolerance": 0.0, "condition_number": float("inf")}
    tolerance = float(max(matrix.shape) * np.finfo(matrix.dtype).eps * singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    full_dimension = min(matrix.shape)
    condition = (
        float(singular_values[0] / singular_values[full_dimension - 1])
        if rank == full_dimension and full_dimension > 0
        else float("inf")
    )
    return {"rank": rank, "rank_tolerance": tolerance, "condition_number": condition}


def evaluate_random_feature_regressor(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    hidden_width: int,
    seed: int = 0,
    solver: str = "ols",
    ridge_alpha: float = 0.0,
    interpolation_atol: float = 1e-10,
    interpolation_rtol: float = 1e-8,
) -> dict[str, Any]:
    model = fit_random_feature_regressor(
        X_train,
        y_train,
        hidden_width,
        seed=seed,
        solver=solver,
        ridge_alpha=ridge_alpha,
    )
    Phi_train = random_relu_features(X_train, model["W"], model["b"])
    Phi_test = random_relu_features(X_test, model["W"], model["b"])
    train_prediction = Phi_train @ model["coefficients"]
    test_prediction = Phi_test @ model["coefficients"]
    train_mse = float(mean_squared_error(y_train, train_prediction))
    test_mse = float(mean_squared_error(y_test, test_prediction))
    diagnostics = matrix_diagnostics(Phi_train)
    target_variance = float(np.var(y_train))
    interpolation_tolerance = float(interpolation_atol + interpolation_rtol * target_variance)

    return {
        "hidden_width": int(hidden_width),
        "n_train": int(X_train.shape[0]),
        "n_features": int(Phi_train.shape[1]),
        "rank": diagnostics["rank"],
        "rank_tolerance": diagnostics["rank_tolerance"],
        "condition_number": diagnostics["condition_number"],
        "train_mse": train_mse,
        "test_mse": test_mse,
        "target_variance": target_variance,
        "interpolation_tolerance": interpolation_tolerance,
        "interpolates": bool(train_mse <= interpolation_tolerance),
        "solver": solver,
        "ridge_alpha": float(ridge_alpha),
        "model_seed": int(seed),
    }


def sweep_random_feature_widths(
    config: RandomFeatureConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or RandomFeatureConfig()
    X_train, X_test, y_train, y_test = generate_synthetic_regression_data(
        n_samples=config.n_samples,
        noise_std=config.noise_std,
        seed=config.seed,
        test_size=config.test_size,
    )
    return [
        evaluate_random_feature_regressor(
            X_train,
            X_test,
            y_train,
            y_test,
            hidden_width=width,
            seed=config.seed,
            solver=config.solver,
            ridge_alpha=config.ridge_alpha,
            interpolation_atol=config.interpolation_atol,
            interpolation_rtol=config.interpolation_rtol,
        )
        for width in config.hidden_widths
    ]

