from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.experiment_runner import _summarize, _write_csv
    from src.random_feature_model import (
        default_capacity_grid,
        evaluate_random_feature_regressor,
        fit_random_feature_regressor,
        matrix_diagnostics,
        random_relu_features,
    )
except ModuleNotFoundError:  # pragma: no cover
    from experiment_runner import _summarize, _write_csv
    from random_feature_model import (
        default_capacity_grid,
        evaluate_random_feature_regressor,
        fit_random_feature_regressor,
        matrix_diagnostics,
        random_relu_features,
    )


SYNTHETIC_CONDITIONS = [
    ("single_sinusoid", 1),
    ("single_sinusoid", 5),
    ("single_sinusoid", 20),
    ("additive_sinusoid", 5),
    ("additive_sinusoid", 20),
    ("interaction", 5),
    ("interaction", 20),
]


def make_synthetic_dataset(
    target: str,
    input_dim: int,
    n_samples: int,
    noise_std: float,
    seed: int,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target not in {"single_sinusoid", "additive_sinusoid", "interaction"}:
        raise ValueError(f"Unsupported synthetic target: {target}")
    if input_dim < 1 or n_samples < 3 or noise_std < 0:
        raise ValueError("input_dim and n_samples must be positive; noise_std must be non-negative")
    if target == "interaction" and input_dim < 2:
        raise ValueError("interaction target requires input_dim >= 2")

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, input_dim))
    if target == "single_sinusoid":
        signal = np.sin(2 * np.pi * X[:, 0])
    elif target == "additive_sinusoid":
        signal_dims = min(input_dim, 3)
        signal = np.sin(2 * np.pi * X[:, :signal_dims]).sum(axis=1) / np.sqrt(signal_dims)
    else:
        signal = np.sin(np.pi * X[:, 0] * X[:, 1])
    y = signal + rng.normal(0.0, noise_std, size=n_samples)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def phase2_synthetic_experiments(
    n_samples: int = 200,
    noise_levels: list[float] | None = None,
    repetitions: int = 30,
    base_seed: int = 2026,
    ridge_alpha: float = 1e-3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if noise_levels is None:
        noise_levels = [0.1, 0.5]
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2")

    rows = []
    n_train = n_samples - int(np.ceil(n_samples * 0.2))
    widths = default_capacity_grid(n_train)
    for target, input_dim in SYNTHETIC_CONDITIONS:
        for noise_std in noise_levels:
            for repetition in range(repetitions):
                data_seed = base_seed + repetition
                model_seed = data_seed + 2_000_003
                X_train, X_test, y_train, y_test = make_synthetic_dataset(
                    target, input_dim, n_samples, noise_std, data_seed
                )
                for solver, alpha in (("ols", 0.0), ("ridge", ridge_alpha)):
                    for width in widths:
                        metrics = evaluate_random_feature_regressor(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            hidden_width=width,
                            seed=model_seed,
                            solver=solver,
                            ridge_alpha=alpha,
                        )
                        rows.append(
                            {
                                "target": target,
                                "input_dim": input_dim,
                                "noise_std": noise_std,
                                "n_samples": n_samples,
                                "repetition": repetition,
                                "data_seed": data_seed,
                                "model_seed": model_seed,
                                **metrics,
                            }
                        )
    summary = _summarize(
        rows,
        [
            "target",
            "input_dim",
            "noise_std",
            "n_samples",
            "n_train",
            "solver",
            "ridge_alpha",
            "hidden_width",
        ],
        ["train_mse", "test_mse", "rank", "condition_number", "interpolates"],
    )
    return rows, summary


def classify_capacity_curve(
    summary_rows: list[dict[str, Any]],
    target: str,
    input_dim: int,
    noise_std: float,
    solver: str = "ols",
) -> dict[str, Any]:
    rows = sorted(
        (
            row
            for row in summary_rows
            if row["target"] == target
            and int(row["input_dim"]) == input_dim
            and float(row["noise_std"]) == noise_std
            and row["solver"] == solver
        ),
        key=lambda row: int(row["hidden_width"]),
    )
    if len(rows) < 3:
        raise ValueError("At least three capacity points are required to classify a curve")
    peak_index = max(range(len(rows)), key=lambda i: float(rows[i]["test_mse_mean"]))
    peak = rows[peak_index]
    has_lower = peak_index > 0
    has_higher = peak_index < len(rows) - 1
    full_row_rank = int(peak["rank_mean"]) == int(peak["n_train"])
    interpolates = float(peak["interpolates_mean"]) >= 0.8
    local_peak = has_lower and has_higher
    pointwise_descent = (
        local_peak
        and float(peak["test_mse_mean"]) > float(rows[peak_index - 1]["test_mse_mean"])
        and float(peak["test_mse_mean"]) > float(rows[peak_index + 1]["test_mse_mean"])
    )
    separated_interpolation_peak = (
        local_peak
        and full_row_rank
        and interpolates
        and float(peak["test_mse_ci95_low"]) > float(rows[peak_index - 1]["test_mse_ci95_high"])
        and float(peak["test_mse_ci95_low"]) > float(rows[peak_index + 1]["test_mse_ci95_high"])
    )
    return {
        "target": target,
        "input_dim": input_dim,
        "noise_std": noise_std,
        "solver": solver,
        "peak_width": int(peak["hidden_width"]),
        "peak_test_mse_mean": float(peak["test_mse_mean"]),
        "peak_test_mse_ci95_low": float(peak["test_mse_ci95_low"]),
        "peak_test_mse_ci95_high": float(peak["test_mse_ci95_high"]),
        "peak_rank_mean": float(peak["rank_mean"]),
        "peak_interpolation_rate": float(peak["interpolates_mean"]),
        "pointwise_peak_then_descent": bool(pointwise_descent),
        "ci_separated_interpolation_peak": bool(separated_interpolation_peak),
    }


def _predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return random_relu_features(X, model["W"], model["b"]) @ model["coefficients"]


def prepare_diabetes_split(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = load_diabetes()
    X, y = dataset.data.astype(np.float64), dataset.target.astype(np.float64)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=seed + 1
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train, scaler.mean_


def real_diabetes_external_validation(
    seeds: list[int] | None = None,
    widths: list[int] | None = None,
    ridge_alphas: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Repeated fixed test holdouts; scaler and hyperparameter selection use train/validation only."""
    seeds = [42, 43, 44, 45, 46] if seeds is None else seeds
    widths = [16, 32, 64, 128, 256] if widths is None else widths
    ridge_alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0] if ridge_alphas is None else ridge_alphas
    if len(seeds) < 2:
        raise ValueError("At least two split seeds are required for uncertainty estimates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("split seeds must be unique")
    if not widths or any(width < 1 for width in widths):
        raise ValueError("widths must be a non-empty list of positive integers")
    if not ridge_alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in ridge_alphas):
        raise ValueError("ridge_alphas must be positive")

    dataset = load_diabetes()
    X, y = dataset.data.astype(np.float64), dataset.target.astype(np.float64)
    rows = []
    for repetition, seed in enumerate(seeds):
        (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train,
            y_val,
            y_test,
            X_train,
            scaler_mean,
        ) = prepare_diabetes_split(seed)

        candidates = []
        model_seed = seed + 3_000_017
        for width in widths:
            for solver, alphas in (("ols", [0.0]), ("ridge", ridge_alphas)):
                for alpha in alphas:
                    model = fit_random_feature_regressor(
                        X_train_scaled,
                        y_train,
                        hidden_width=width,
                        seed=model_seed,
                        solver=solver,
                        ridge_alpha=alpha,
                    )
                    val_mse = float(mean_squared_error(y_val, _predict(model, X_val_scaled)))
                    candidates.append((val_mse, width, solver, alpha, model))

        validation_mse, selected_width, selected_solver, selected_alpha, selected_model = min(
            candidates, key=lambda item: item[0]
        )
        train_features = random_relu_features(
            X_train_scaled, selected_model["W"], selected_model["b"]
        )
        diagnostics = matrix_diagnostics(train_features)
        train_mse = float(
            mean_squared_error(y_train, _predict(selected_model, X_train_scaled))
        )
        test_mse = float(
            mean_squared_error(y_test, _predict(selected_model, X_test_scaled))
        )
        tolerance = float(1e-10 + 1e-8 * np.var(y_train))
        rows.append(
            {
                "dataset": "sklearn_diabetes",
                "repetition": repetition,
                "split_seed": seed,
                "model_seed": model_seed,
                "n_total": int(X.shape[0]),
                "input_dim": int(X.shape[1]),
                "n_train": int(len(y_train)),
                "n_validation": int(len(y_val)),
                "n_test": int(len(y_test)),
                "selected_width": int(selected_width),
                "selected_solver": selected_solver,
                "selected_ridge_alpha": float(selected_alpha),
                "validation_mse": validation_mse,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "n_features": int(train_features.shape[1]),
                "rank": diagnostics["rank"],
                "rank_tolerance": diagnostics["rank_tolerance"],
                "condition_number": diagnostics["condition_number"],
                "interpolation_tolerance": tolerance,
                "interpolates": bool(train_mse <= tolerance),
                "scaler_fit_rows": int(len(y_train)),
                "scaler_train_mean_max_abs_error": float(
                    np.max(np.abs(scaler_mean - X_train.mean(axis=0)))
                ),
                "test_used_for_selection": False,
            }
        )
    summary = _summarize(
        rows,
        ["dataset"],
        ["train_mse", "validation_mse", "test_mse", "rank", "condition_number", "interpolates"],
    )
    return rows, summary


def _plot_synthetic(
    summary: list[dict[str, Any]], classifications: list[dict[str, Any]], path: Path
) -> None:
    conditions = sorted(
        {(row["target"], int(row["input_dim"]), float(row["noise_std"])) for row in summary}
    )
    fig, axes = plt.subplots(
        len(conditions), 1, figsize=(10, max(5, 3.4 * len(conditions))), squeeze=False
    )
    for ax, condition in zip(axes[:, 0], conditions):
        train_ax = ax.twinx()
        target, input_dim, noise_std = condition
        for solver in ("ols", "ridge"):
            rows = sorted(
                (
                    row for row in summary
                    if row["target"] == target
                    and int(row["input_dim"]) == input_dim
                    and float(row["noise_std"]) == noise_std
                    and row["solver"] == solver
                ),
                key=lambda row: int(row["hidden_width"]),
            )
            widths = np.asarray([row["hidden_width"] for row in rows])
            mean = np.asarray([row["test_mse_mean"] for row in rows])
            low = np.asarray([row["test_mse_ci95_low"] for row in rows])
            high = np.asarray([row["test_mse_ci95_high"] for row in rows])
            ax.plot(widths, mean, "o-", label=f"{solver} test")
            ax.fill_between(widths, np.maximum(low, 1e-14), high, alpha=0.16)
            if solver == "ols":
                train = np.asarray([row["train_mse_mean"] for row in rows])
                train_ax.plot(
                    widths, np.maximum(train, 1e-30), "s--", color="black", alpha=0.7,
                    label="OLS train",
                )
                ax.axvline(int(rows[0]["n_train"]), color="grey", linestyle=":")
        classification = next(
            row for row in classifications
            if row["target"] == target
            and int(row["input_dim"]) == input_dim
            and float(row["noise_std"]) == noise_std
        )
        ax.set_title(
            f"{target}, d={input_dim}, noise={noise_std:g}; "
            f"peak/descent={classification['pointwise_peak_then_descent']}, "
            f"CI-separated={classification['ci_separated_interpolation_peak']}"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        train_ax.set_yscale("log")
        ax.set_xlabel("ReLU hidden width")
        ax.set_ylabel("Test MSE (95% bootstrap CI)")
        train_ax.set_ylabel("OLS training MSE")
        ax.grid(True, linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        train_handles, train_labels = train_ax.get_legend_handles_labels()
        ax.legend(handles + train_handles, labels + train_labels, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_external(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([row["test_mse"] for row in rows], tick_labels=["Test MSE"])
    ax.set_ylabel("MSE")
    ax.set_title("Diabetes external validation: held-out test MSE\n(selection performed on validation split)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 robustness and external validation.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--external-repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 2 or args.external_repetitions < 2:
        parser.error("synthetic and external repetitions must both be at least 2")
    if args.n_samples < 3 or any(not np.isfinite(noise) or noise < 0 for noise in args.noise_levels):
        parser.error("n-samples must be at least 3 and noise levels must be finite and non-negative")
    if not np.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0:
        parser.error("ridge-alpha must be finite and positive")

    root = Path(__file__).resolve().parent.parent
    artifacts_dir = root / "artifacts" / "phase2"
    figures_dir = root / "figures" / "phase2"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "synthetic": {
            "n_samples": args.n_samples,
            "noise_levels": args.noise_levels,
            "target_and_dimensions": SYNTHETIC_CONDITIONS,
            "repetitions": args.repetitions,
            "base_seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
            "seed_derivation": "data_seed=base_seed+repetition; model_seed=data_seed+2000003",
            "capacity_grid": "Phase 1 default_capacity_grid around n_train",
        },
        "real_dataset": {
            "name": "scikit-learn diabetes",
            "repetitions": args.external_repetitions,
            "split_seeds": list(range(42, 42 + args.external_repetitions)),
            "split": "20% held-out test; remaining 80% split 75/25 into train/validation (60/20/20 overall)",
            "preprocessing": "StandardScaler fit only on training partition",
            "selection": "validation MSE selects width, solver, and ridge alpha; test is evaluated once",
            "candidate_widths": [16, 32, 64, 128, 256],
            "ridge_alphas": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        },
        "bootstrap_seed": 20261013,
        "confidence_interval": "95% percentile bootstrap confidence interval for mean; 5000 resamples",
    }
    (artifacts_dir / "phase2_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    synthetic_runs, synthetic_summary = phase2_synthetic_experiments(
        n_samples=args.n_samples,
        noise_levels=args.noise_levels,
        repetitions=args.repetitions,
        base_seed=args.seed,
        ridge_alpha=args.ridge_alpha,
    )
    classifications = [
        classify_capacity_curve(synthetic_summary, target, input_dim, noise)
        for target, input_dim in SYNTHETIC_CONDITIONS
        for noise in args.noise_levels
    ]
    _write_csv(synthetic_runs, artifacts_dir / "synthetic_runs.csv")
    _write_csv(synthetic_summary, artifacts_dir / "synthetic_summary.csv")
    _write_csv(classifications, artifacts_dir / "synthetic_curve_classification.csv")
    _plot_synthetic(
        synthetic_summary,
        classifications,
        figures_dir / "synthetic_robustness_curves.png",
    )

    external_runs, external_summary = real_diabetes_external_validation(
        seeds=list(range(42, 42 + args.external_repetitions))
    )
    _write_csv(external_runs, artifacts_dir / "diabetes_external_runs.csv")
    _write_csv(external_summary, artifacts_dir / "diabetes_external_summary.csv")
    _plot_external(external_runs, figures_dir / "diabetes_external_test_mse.png")
    print(f"Phase 2 results saved under {artifacts_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
