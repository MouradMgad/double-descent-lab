from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.random_feature_model import (
        default_capacity_grid,
        evaluate_random_feature_regressor,
        generate_synthetic_regression_data,
        matrix_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover
    from random_feature_model import (
        default_capacity_grid,
        evaluate_random_feature_regressor,
        generate_synthetic_regression_data,
        matrix_diagnostics,
    )


def _design_matrix(x: np.ndarray, degree: int, representation: str) -> np.ndarray:
    if representation == "raw_monomial":
        return PolynomialFeatures(degree, include_bias=True).fit_transform(x)
    if representation == "legendre":
        return np.polynomial.legendre.legvander(x.ravel(), degree)
    raise ValueError("representation must be 'raw_monomial' or 'legendre'")


def _fit_linear(
    design: np.ndarray, targets: np.ndarray, solver: str, ridge_alpha: float
) -> np.ndarray:
    if solver == "ols":
        return np.linalg.lstsq(design, targets, rcond=None)[0]
    if solver != "ridge":
        raise ValueError("solver must be 'ols' or 'ridge'")
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive for ridge")
    penalty = np.zeros((design.shape[1], design.shape[1]))
    penalty[1:, 1:] = np.sqrt(ridge_alpha) * np.eye(design.shape[1] - 1)
    augmented_design = np.vstack((design, penalty))
    augmented_targets = np.concatenate((targets, np.zeros(design.shape[1])))
    return np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)[0]


def polynomial_baseline_metrics(
    n_samples: int = 200,
    noise_std: float = 0.1,
    seed: int = 42,
    max_degree: int = 40,
    representation: str = "legendre",
    solver: str = "ols",
    ridge_alpha: float = 0.0,
) -> list[dict[str, Any]]:
    X_train, X_test, y_train, y_test = generate_synthetic_regression_data(
        n_samples=n_samples, noise_std=noise_std, seed=seed
    )
    metrics = []
    for degree in range(1, max_degree + 1):
        train_design = _design_matrix(X_train, degree, representation)
        test_design = _design_matrix(X_test, degree, representation)
        coefficients = _fit_linear(train_design, y_train, solver, ridge_alpha)
        diagnostics = matrix_diagnostics(train_design)
        train_mse = float(mean_squared_error(y_train, train_design @ coefficients))
        test_mse = float(mean_squared_error(y_test, test_design @ coefficients))
        target_variance = float(np.var(y_train))
        interpolation_tolerance = 1e-10 + 1e-8 * target_variance
        metrics.append(
            {
                "degree": degree,
                "representation": representation,
                "solver": solver,
                "ridge_alpha": float(ridge_alpha),
                "n_train": int(train_design.shape[0]),
                "n_features": int(train_design.shape[1]),
                "rank": diagnostics["rank"],
                "rank_tolerance": diagnostics["rank_tolerance"],
                "condition_number": diagnostics["condition_number"],
                "train_mse": train_mse,
                "test_mse": test_mse,
                "interpolation_tolerance": interpolation_tolerance,
                "interpolates": train_mse <= interpolation_tolerance,
                "seed": int(seed),
                "noise_std": float(noise_std),
                "n_samples": int(n_samples),
            }
        )
    return metrics


def _write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: list[dict[str, Any]], group_fields: list[str], value_fields: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    summaries = []
    bootstrap_rng = np.random.default_rng(20261013)
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        summary: dict[str, Any] = dict(zip(group_fields, key))
        count = len(group)
        summary["n_runs"] = count
        for field in value_fields:
            all_values = np.asarray([row[field] for row in group], dtype=float)
            values = all_values[np.isfinite(all_values)]
            summary[f"{field}_n_finite"] = int(values.size)
            summary[f"{field}_infinite_runs"] = int(np.isinf(all_values).sum())
            if values.size == 0:
                summary[f"{field}_mean"] = float("nan")
                summary[f"{field}_std"] = float("nan")
                summary[f"{field}_ci95_low"] = float("nan")
                summary[f"{field}_ci95_high"] = float("nan")
                continue
            finite_count = values.size
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if finite_count > 1 else 0.0
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = std
            if finite_count > 1:
                indices = bootstrap_rng.integers(
                    0, finite_count, size=(5000, finite_count)
                )
                bootstrap_means = values[indices].mean(axis=1)
                low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            else:
                low = high = mean
            summary[f"{field}_ci95_low"] = float(low)
            summary[f"{field}_ci95_high"] = float(high)
        summaries.append(summary)
    return summaries


def _random_feature_experiments(
    sample_sizes: list[int],
    noise_levels: list[float],
    repetitions: int,
    base_seed: int,
    ridge_alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = []
    for n_samples in sample_sizes:
        n_train = n_samples - int(np.ceil(n_samples * 0.2))
        widths = default_capacity_grid(n_train)
        for noise_std in noise_levels:
            for repetition in range(repetitions):
                data_seed = base_seed + repetition
                model_seed = data_seed + 1_000_003
                X_train, X_test, y_train, y_test = generate_synthetic_regression_data(
                    n_samples=n_samples, noise_std=noise_std, seed=data_seed
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
                        raw_rows.append(
                            {
                                "n_samples": n_samples,
                                "noise_std": noise_std,
                                "repetition": repetition,
                                "data_seed": data_seed,
                                "model_seed": model_seed,
                                **metrics,
                            }
                        )

    summaries = _summarize(
        raw_rows,
        ["n_samples", "noise_std", "solver", "ridge_alpha", "hidden_width"],
        ["train_mse", "test_mse", "rank", "condition_number", "interpolates"],
    )
    return raw_rows, summaries


def _polynomial_experiments(
    n_samples: int,
    noise_std: float,
    repetitions: int,
    base_seed: int,
    max_degree: int,
    ridge_alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for repetition in range(repetitions):
        seed = base_seed + repetition
        for representation in ("raw_monomial", "legendre"):
            for solver, alpha in (("ols", 0.0), ("ridge", ridge_alpha)):
                rows.extend(
                    {
                        "repetition": repetition,
                        **result,
                    }
                    for result in polynomial_baseline_metrics(
                        n_samples=n_samples,
                        noise_std=noise_std,
                        seed=seed,
                        max_degree=max_degree,
                        representation=representation,
                        solver=solver,
                        ridge_alpha=alpha,
                    )
                )
    summaries = _summarize(
        rows,
        ["n_samples", "noise_std", "representation", "solver", "ridge_alpha", "degree"],
        ["train_mse", "test_mse", "rank", "condition_number", "interpolates"],
    )
    return rows, summaries


def _plot_random_feature_summary(summaries: list[dict[str, Any]], path: Path) -> None:
    sample_noise_pairs = sorted({(row["n_samples"], row["noise_std"]) for row in summaries})
    fig, axes = plt.subplots(
        len(sample_noise_pairs), 1, figsize=(10, max(5, 4 * len(sample_noise_pairs))), squeeze=False
    )
    for ax, (n_samples, noise_std) in zip(axes[:, 0], sample_noise_pairs):
        for solver in ("ols", "ridge"):
            rows = sorted(
                (
                    row
                    for row in summaries
                    if row["n_samples"] == n_samples
                    and row["noise_std"] == noise_std
                    and row["solver"] == solver
                ),
                key=lambda row: row["hidden_width"],
            )
            if not rows:
                continue
            widths = np.asarray([row["hidden_width"] for row in rows])
            means = np.asarray([row["test_mse_mean"] for row in rows])
            lower = np.asarray([row["test_mse_ci95_low"] for row in rows])
            upper = np.asarray([row["test_mse_ci95_high"] for row in rows])
            ax.plot(widths, means, "o-", label=f"{solver} test MSE")
            ax.fill_between(widths, np.maximum(lower, 1e-14), upper, where=lower > 0, alpha=0.18)
        train_rows = sorted(
            (
                row
                for row in summaries
                if row["n_samples"] == n_samples
                and row["noise_std"] == noise_std
                and row["solver"] == "ols"
            ),
            key=lambda row: row["hidden_width"],
        )
        widths = np.asarray([row["hidden_width"] for row in train_rows])
        means = np.asarray([row["train_mse_mean"] for row in train_rows])
        lower = np.asarray([row["train_mse_ci95_low"] for row in train_rows])
        upper = np.asarray([row["train_mse_ci95_high"] for row in train_rows])
        ax.plot(widths, means, "s--", color="black", label="OLS train MSE")
        ax.fill_between(widths, np.maximum(lower, 1e-14), upper, color="black", alpha=0.1)
        ax.axvline(
            int(round(n_samples * 0.8)) - 1,
            color="grey",
            linestyle=":",
            label="n_features = n_train",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"n={n_samples}, noise σ={noise_std:g}")
        ax.set_xlabel("Hidden ReLU feature width")
        ax.set_ylabel("MSE (95% bootstrap CI)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_polynomial_summary(summaries: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for representation, solver in (
        ("raw_monomial", "ols"),
        ("legendre", "ols"),
        ("raw_monomial", "ridge"),
        ("legendre", "ridge"),
    ):
        rows = sorted(
            (
                row
                for row in summaries
                if row["representation"] == representation and row["solver"] == solver
            ),
            key=lambda row: row["degree"],
        )
        if not rows:
            continue
        degrees = np.asarray([row["degree"] for row in rows])
        mean = np.asarray([row["test_mse_mean"] for row in rows])
        low = np.asarray([row["test_mse_ci95_low"] for row in rows])
        high = np.asarray([row["test_mse_ci95_high"] for row in rows])
        label = f"{representation.replace('_', ' ')} / {solver}"
        ax.plot(degrees, mean, label=label)
        ax.fill_between(degrees, np.maximum(low, 1e-14), high, where=low > 0, alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Test MSE (95% bootstrap CI)")
    ax.set_title("Polynomial basis and regularization comparison")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated controlled double-descent experiments.")
    parser.add_argument("--n-samples", nargs="+", type=int, default=[200, 500])
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.1, 0.5])
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-degree", type=int, default=40)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 to report uncertainty")

    root = Path(__file__).resolve().parent.parent
    artifacts_dir = root / "artifacts"
    figures_dir = root / "figures"
    artifacts_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    config = {
        "n_samples": args.n_samples,
        "noise_levels": args.noise_levels,
        "test_fraction": 0.2,
        "repetitions": args.repetitions,
        "base_seed": args.seed,
        "max_degree": args.max_degree,
        "random_feature_capacity_grid": "25%-200% of n_train; denser near n_train",
        "one_dimensional_feature_initialization": "random +/-1 slopes with fixed empirical train-input quantile knots",
        "seed_derivation": "data_seed=base_seed+repetition; model_seed=data_seed+1000003",
        "solvers": ["OLS (SVD least squares)", f"Ridge (augmented least squares, alpha={args.ridge_alpha})"],
        "interpolation_rule": "training MSE <= 1e-10 + 1e-8 * training-target variance",
        "rank_rule": "singular values > max(shape) * machine_epsilon * largest_singular_value",
        "ci95": "95% percentile bootstrap confidence interval for the mean (5000 resamples)",
        "bootstrap_seed": 20261013,
    }
    (artifacts_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    rf_raw, rf_summary = _random_feature_experiments(
        args.n_samples, args.noise_levels, args.repetitions, args.seed, args.ridge_alpha
    )
    _write_csv(rf_raw, artifacts_dir / "random_feature_runs.csv")
    _write_csv(rf_summary, artifacts_dir / "random_feature_summary.csv")
    paired: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in rf_raw:
        key = (
            row["n_samples"],
            row["noise_std"],
            row["hidden_width"],
            row["data_seed"],
        )
        paired[key][row["solver"]] = float(row["test_mse"])
    paired_rows = [
        {
            "n_samples": key[0],
            "noise_std": key[1],
            "hidden_width": key[2],
            "data_seed": key[3],
            "ridge_minus_ols_test_mse": solvers["ridge"] - solvers["ols"],
        }
        for key, solvers in paired.items()
        if "ols" in solvers and "ridge" in solvers
    ]
    paired_summary = _summarize(
        paired_rows,
        ["n_samples", "noise_std", "hidden_width"],
        ["ridge_minus_ols_test_mse"],
    )
    _write_csv(paired_rows, artifacts_dir / "random_feature_paired_runs.csv")
    _write_csv(paired_summary, artifacts_dir / "random_feature_paired_summary.csv")

    polynomial_n = args.n_samples[0]
    polynomial_noise = args.noise_levels[0]
    poly_raw, poly_summary = _polynomial_experiments(
        polynomial_n,
        polynomial_noise,
        args.repetitions,
        args.seed,
        args.max_degree,
        args.ridge_alpha,
    )
    _write_csv(poly_raw, artifacts_dir / "polynomial_runs.csv")
    _write_csv(poly_summary, artifacts_dir / "polynomial_summary.csv")

    _plot_random_feature_summary(rf_summary, figures_dir / "random_feature_repeated_results.png")
    _plot_polynomial_summary(poly_summary, figures_dir / "polynomial_basis_regularization.png")
    print(f"Saved repeated-run metrics and configuration to {artifacts_dir}")
    print(f"Saved uncertainty plots to {figures_dir}")


if __name__ == "__main__":
    main()
