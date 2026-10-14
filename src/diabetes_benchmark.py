from __future__ import annotations

import csv
import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.random_feature_model import fit_random_feature_regressor, random_relu_features
except ModuleNotFoundError:  # pragma: no cover
    from random_feature_model import fit_random_feature_regressor, random_relu_features


DEFAULT_SEED = 20261014
DEFAULT_REPEATS = 10
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 5
DEFAULT_WIDTHS = (16, 32, 64, 128, 256)
DEFAULT_RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
BOOTSTRAP_RESAMPLES = 10_000


def _rf_predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return random_relu_features(X, model["W"], model["b"]) @ model["coefficients"]


def _scaled_rf_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    width: int,
    solver: str,
    alpha: float,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    model = fit_random_feature_regressor(
        X_train_scaled,
        y_train,
        hidden_width=width,
        seed=seed,
        solver=solver,
        ridge_alpha=alpha,
    )
    return _rf_predict(model, X_eval_scaled)


def _candidate_seed(seed: int, repeat: int, outer_fold: int, inner_fold: int) -> int:
    return int(seed + repeat * 1_000_003 + outer_fold * 10_007 + inner_fold * 101)


def _inner_cv_selection(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    repeat: int,
    outer_fold: int,
    seed: int,
    widths: tuple[int, ...],
    ridge_alphas: tuple[float, ...],
) -> tuple[dict[str, Any], float, dict[tuple[str, int, float], float]]:
    scores: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for inner_fold, (fit_indices, validation_indices) in enumerate(splits):
        X_fit, X_validation = X[fit_indices], X[validation_indices]
        y_fit, y_validation = y[fit_indices], y[validation_indices]

        for width in widths:
            for solver, alphas in (("ols", (0.0,)), ("ridge", ridge_alphas)):
                for alpha in alphas:
                    prediction = _scaled_rf_fit_predict(
                        X_fit,
                        y_fit,
                        X_validation,
                        width,
                        solver,
                        alpha,
                        _candidate_seed(seed, repeat, outer_fold, inner_fold),
                    )
                    scores[(solver, width, alpha)].append(
                        float(mean_squared_error(y_validation, prediction))
                    )

    candidate_means = {key: float(np.mean(values)) for key, values in scores.items()}
    selected_key = min(candidate_means, key=lambda key: (candidate_means[key], key))
    selected = {
        "solver": selected_key[0],
        "width": selected_key[1],
        "alpha": selected_key[2],
        "inner_cv_mse": candidate_means[selected_key],
        "selection": "inner_cv_mse",
    }
    return selected, candidate_means[selected_key], candidate_means


def repeated_nested_diabetes_cv(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    repeats: int = DEFAULT_REPEATS,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate baselines and tuned random features with repeated nested CV.

    All transformations and model selection are confined to outer-training rows.
    The same outer folds are used for every model.
    """
    if (X is None) != (y is None):
        raise ValueError("X and y must either both be provided or both be omitted")
    if X is None and y is None:
        dataset = load_diabetes(scaled=False)
        X = dataset.data.astype(np.float64)
        y = dataset.target.astype(np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2D and y must be 1D with matching rows")
    if repeats < 2 or outer_folds < 2 or inner_folds < 2:
        raise ValueError("repeats, outer_folds, and inner_folds must each be at least 2")
    if outer_folds > len(y) or inner_folds > len(y) - int(np.ceil(len(y) / outer_folds)):
        raise ValueError("fold counts exceed the available training observations")
    if not widths or any(width < 1 for width in widths):
        raise ValueError("widths must be non-empty positive integers")
    if not ridge_alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in ridge_alphas):
        raise ValueError("ridge_alphas must be non-empty, finite, and positive")

    rows: list[dict[str, Any]] = []
    model_names = ("mean_prediction", "linear_regression", "ridge", "random_features")
    for repeat in range(repeats):
        outer = KFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=seed + repeat,
        )
        for outer_fold, (train_indices, test_indices) in enumerate(outer.split(X)):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            inner = KFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=seed + repeat * 1009 + outer_fold,
            )
            inner_splits = list(inner.split(X_train))
            selected, selected_inner_mse, _ = _inner_cv_selection(
                X_train,
                y_train,
                inner_splits,
                repeat,
                outer_fold,
                seed,
                widths,
                ridge_alphas,
            )

            dummy = DummyRegressor(strategy="mean")
            dummy.fit(X_train, y_train)
            linear = make_pipeline(StandardScaler(), LinearRegression())
            linear.fit(X_train, y_train)
            ridge_scores = {
                alpha: _ridge_inner_score(X_train, y_train, inner_splits, alpha)
                for alpha in ridge_alphas
            }
            selected_ridge_alpha = min(ridge_scores, key=ridge_scores.get)
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=selected_ridge_alpha))
            ridge.fit(X_train, y_train)
            final_scaler = StandardScaler()
            X_train_scaled = final_scaler.fit_transform(X_train)
            X_test_scaled = final_scaler.transform(X_test)
            rf_seed = _candidate_seed(seed, repeat, outer_fold, inner_folds)
            rf_model = fit_random_feature_regressor(
                X_train_scaled,
                y_train,
                hidden_width=int(selected["width"]),
                seed=rf_seed,
                solver=str(selected["solver"]),
                ridge_alpha=float(selected["alpha"]),
            )

            predictions = {
                "mean_prediction": dummy.predict(X_test),
                "linear_regression": linear.predict(X_test),
                "ridge": ridge.predict(X_test),
                "random_features": _rf_predict(rf_model, X_test_scaled),
            }
            outer_ids = json.dumps([int(index) for index in test_indices])
            for model_name in model_names:
                prediction = predictions[model_name]
                mse = float(mean_squared_error(y_test, prediction))
                if model_name == "random_features":
                    selected_params = {
                        "width": int(selected["width"]),
                        "solver": selected["solver"],
                        "ridge_alpha": float(selected["alpha"]),
                    }
                    inner_mse = float(selected_inner_mse)
                elif model_name == "ridge":
                    selected_params = {"ridge_alpha": float(selected_ridge_alpha)}
                    inner_mse = float(ridge_scores[selected_ridge_alpha])
                else:
                    selected_params = {}
                    inner_mse = float("nan")
                rows.append(
                    {
                        "dataset": "sklearn_diabetes_raw",
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "n_train": int(len(train_indices)),
                        "n_test": int(len(test_indices)),
                        "outer_test_indices": outer_ids,
                        "test_mse": mse,
                        "test_rmse": float(np.sqrt(mse)),
                        "selected_params": json.dumps(selected_params, sort_keys=True),
                        "inner_cv_mse": inner_mse,
                        "inner_folds": inner_folds,
                        "scaler_fit_rows": int(len(train_indices))
                        if model_name in {"linear_regression", "ridge", "random_features"}
                        else 0,
                        "outer_test_used_for_selection": False,
                    }
                )
    summaries = _summarize_models(rows, repeats, seed)
    paired = _paired_differences(rows, repeats, seed)
    return rows, summaries, paired


def _ridge_inner_score(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
) -> float:
    fold_scores = []
    for fit_indices, validation_indices in splits:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[fit_indices], y[fit_indices])
        fold_scores.append(
            mean_squared_error(y[validation_indices], model.predict(X[validation_indices]))
        )
    return float(np.mean(fold_scores))


def _cluster_ci(values_by_repeat: list[float], seed: int) -> tuple[float, float]:
    values = np.asarray(values_by_repeat, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _summarize_models(
    rows: list[dict[str, Any]], repeats: int, seed: int
) -> list[dict[str, Any]]:
    summaries = []
    for model in ("mean_prediction", "linear_regression", "ridge", "random_features"):
        group = [row for row in rows if row["model"] == model]
        summary: dict[str, Any] = {
            "dataset": "sklearn_diabetes_raw",
            "model": model,
            "n_repeats": repeats,
            "n_outer_folds": len(group),
        }
        for metric in ("test_mse", "test_rmse"):
            per_repeat = [
                float(np.mean([row[metric] for row in group if row["repeat"] == repeat]))
                for repeat in range(repeats)
            ]
            low, high = _cluster_ci(per_repeat, seed + sum(model.encode("utf-8")))
            all_folds = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = float(np.mean(per_repeat))
            summary[f"{metric}_std_outer_folds"] = float(np.std(all_folds, ddof=1))
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
        summaries.append(summary)
    return summaries


def _paired_differences(
    rows: list[dict[str, Any]], repeats: int, seed: int
) -> list[dict[str, Any]]:
    by_split = {
        (row["repeat"], row["outer_fold"], row["model"]): row
        for row in rows
    }
    output = []
    for baseline in ("mean_prediction", "linear_regression", "ridge"):
        result: dict[str, Any] = {
            "dataset": "sklearn_diabetes_raw",
            "difference": "random_features - baseline",
            "baseline": baseline,
            "n_repeats": repeats,
            "n_paired_outer_folds": repeats
            * len({row["outer_fold"] for row in rows}),
        }
        for metric in ("test_mse", "test_rmse"):
            per_repeat = []
            for repeat in range(repeats):
                fold_diffs = []
                for outer_fold in sorted({row["outer_fold"] for row in rows}):
                    rf = by_split[(repeat, outer_fold, "random_features")][metric]
                    base = by_split[(repeat, outer_fold, baseline)][metric]
                    fold_diffs.append(float(rf - base))
                per_repeat.append(float(np.mean(fold_diffs)))
            low, high = _cluster_ci(
                per_repeat, seed + 177 + sum(baseline.encode("utf-8"))
            )
            result[f"{metric}_mean_difference"] = float(np.mean(per_repeat))
            result[f"{metric}_ci95_low"] = low
            result[f"{metric}_ci95_high"] = high
        output.append(result)
    return output


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty results to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_benchmark(
    summaries: list[dict[str, Any]], path: Path
) -> None:
    labels = ["Mean", "Linear", "Ridge", "Random features"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summaries))
    means = np.asarray([row["test_mse_mean"] for row in summaries])
    lows = np.asarray([row["test_mse_ci95_low"] for row in summaries])
    highs = np.asarray([row["test_mse_ci95_high"] for row in summaries])
    ax.bar(x, means, color=["#999999", "#4C78A8", "#72B7B2", "#F58518"])
    ax.errorbar(x, means, yerr=np.vstack((means - lows, highs - means)), fmt="none", color="black", capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Outer-fold MSE (95% repeated-CV interval)")
    ax.set_title("Diabetes regression: repeated nested cross-validation")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def audit_synthetic_claims(
    runs_path: Path,
    *,
    bootstrap_seed: int = 20261015,
    bootstrap_resamples: int = 10_000,
) -> list[dict[str, Any]]:
    with runs_path.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    if not runs:
        raise ValueError(f"No saved synthetic runs found in {runs_path}")

    grouped: dict[tuple[str, int, float, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        key = (
            row["target"],
            int(row["input_dim"]),
            float(row["noise_std"]),
            row["solver"],
            int(row["hidden_width"]),
        )
        grouped[key].append(row)

    conditions = sorted({key[:3] for key in grouped if key[3] == "ols"})
    audit_rows = []
    for condition_index, (target, input_dim, noise_std) in enumerate(conditions):
        curve = []
        condition_rows = [
            row
            for key, group in grouped.items()
            if key[:4] == (target, input_dim, noise_std, "ols")
            for row in group
        ]
        flag_disagreements = sum(
            row["interpolates"].lower()
            != str(
                float(row["train_mse"]) <= float(row["interpolation_tolerance"])
            ).lower()
            for row in condition_rows
        )
        tolerance_disagreements = sum(
            not np.isclose(
                float(row["interpolation_tolerance"]),
                1e-10 + 1e-8 * float(row["target_variance"]),
                rtol=1e-12,
                atol=1e-20,
            )
            for row in condition_rows
        )
        for key, group in grouped.items():
            if key[:4] == (target, input_dim, noise_std, "ols"):
                curve.append(
                    (key[4], float(np.mean([float(row["test_mse"]) for row in group])))
                )
        curve.sort()
        peak_index = max(range(len(curve)), key=lambda index: curve[index][1])
        if peak_index == 0 or peak_index == len(curve) - 1:
            raise ValueError(f"Peak is not interior for {target}, d={input_dim}, noise={noise_std}")
        peak_width, peak_mean = curve[peak_index]
        left_width, left_mean = curve[peak_index - 1]
        right_width, right_mean = curve[peak_index + 1]
        base_key = (target, input_dim, noise_std, "ols")
        peak_rows = {int(row["repetition"]): row for row in grouped[(*base_key, peak_width)]}
        left_rows = {int(row["repetition"]): row for row in grouped[(*base_key, left_width)]}
        right_rows = {int(row["repetition"]): row for row in grouped[(*base_key, right_width)]}
        repetitions = sorted(set(peak_rows) & set(left_rows) & set(right_rows))
        if len(repetitions) < 2:
            raise ValueError("At least two paired repetitions are required for claim auditing")

        paired_intervals = []
        for neighbor_rows in (left_rows, right_rows):
            differences = np.asarray(
                [
                    float(peak_rows[rep]["test_mse"])
                    - float(neighbor_rows[rep]["test_mse"])
                    for rep in repetitions
                ]
            )
            rng = np.random.default_rng(bootstrap_seed + condition_index * 2 + len(paired_intervals))
            indices = rng.integers(
                0, len(differences), size=(bootstrap_resamples, len(differences))
            )
            bootstrap_means = differences[indices].mean(axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            paired_intervals.append((float(np.mean(differences)), float(low), float(high)))

        n_train_values = {int(row["n_train"]) for row in peak_rows.values()}
        rank_consistent = all(
            int(row["rank"]) == int(row["n_train"]) for row in peak_rows.values()
        )
        tolerance_consistent = all(
            row["interpolates"].lower() == str(
                float(row["train_mse"]) <= float(row["interpolation_tolerance"])
            ).lower()
            for row in peak_rows.values()
        )
        interpolation_rate = float(
            np.mean([row["interpolates"].lower() == "true" for row in peak_rows.values()])
        )
        pointwise_peak_then_descent = (
            peak_mean > left_mean and peak_mean > right_mean
        )
        paired_interpolation_peak = (
            len(n_train_values) == 1
            and rank_consistent
            and tolerance_consistent
            and interpolation_rate >= 0.8
            and all(interval[1] > 0 for interval in paired_intervals)
        )
        audit_rows.append(
            {
                "target": target,
                "input_dim": input_dim,
                "noise_std": noise_std,
                "n_repetitions": len(repetitions),
                "peak_width": peak_width,
                "left_neighbor_width": left_width,
                "right_neighbor_width": right_width,
                "peak_test_mse_mean": peak_mean,
                "left_neighbor_test_mse_mean": left_mean,
                "right_neighbor_test_mse_mean": right_mean,
                "n_train": next(iter(n_train_values)),
                "all_ols_run_interpolation_flags_match_tolerance": flag_disagreements == 0,
                "n_ols_interpolation_flag_disagreements": int(flag_disagreements),
                "all_ols_tolerances_match_variance_rule": tolerance_disagreements == 0,
                "n_ols_tolerance_disagreements": int(tolerance_disagreements),
                "all_peak_runs_full_row_rank": rank_consistent,
                "interpolation_flag_matches_tolerance": tolerance_consistent,
                "peak_interpolation_rate": interpolation_rate,
                "peak_minus_left_mean": paired_intervals[0][0],
                "peak_minus_left_ci95_low": paired_intervals[0][1],
                "peak_minus_left_ci95_high": paired_intervals[0][2],
                "peak_minus_right_mean": paired_intervals[1][0],
                "peak_minus_right_ci95_low": paired_intervals[1][1],
                "peak_minus_right_ci95_high": paired_intervals[1][2],
                "pointwise_peak_then_descent": pointwise_peak_then_descent,
                "paired_interpolation_peak_supported": paired_interpolation_peak,
                "post_selection_and_multiplicity_adjusted": False,
            }
        )
    return audit_rows


def run_benchmark(
    artifact_dir: Path,
    figure_dir: Path,
    *,
    repeats: int = DEFAULT_REPEATS,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
) -> None:
    root = Path(__file__).resolve().parent.parent
    config = {
        "dataset": "scikit-learn diabetes with load_diabetes(scaled=False)",
        "evaluation": f"{repeats} repeats of {outer_folds}-fold outer CV; {inner_folds}-fold inner CV for Ridge and random-feature selection",
        "seed": DEFAULT_SEED,
        "outer_repeats": repeats,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "outer_split_seed": "seed + repeat_index",
        "inner_split_seed": "seed + repeat_index * 1009 + outer_fold_index",
        "baseline_models": ["mean_prediction", "linear_regression", "ridge"],
        "random_feature_widths": list(widths),
        "ridge_alphas": list(ridge_alphas),
        "random_feature_selection": "minimum mean inner-fold MSE over width, OLS/Ridge solver, and ridge alpha",
        "preprocessing": "StandardScaler fitted only on each inner or outer training fold",
        "uncertainty": "95% percentile bootstrap interval across 10 repetition-level means (10,000 resamples)",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paired_differences": "random_features minus each baseline, paired by identical outer fold; bootstrap clustered by repeat",
        "synthetic_audit": {
            "source": "artifacts/phase2/synthetic_runs.csv",
            "method": "paired bootstrap of peak-minus-adjacent-capacity test-MSE differences",
            "bootstrap_seed": 20261015,
            "bootstrap_resamples": 10000,
            "multiplicity_adjustment": False,
        },
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "release_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    rows, summaries, paired = repeated_nested_diabetes_cv(
        repeats=repeats,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        widths=widths,
        ridge_alphas=ridge_alphas,
        seed=DEFAULT_SEED,
    )
    _write_csv(rows, artifact_dir / "diabetes_nested_cv_runs.csv")
    _write_csv(summaries, artifact_dir / "diabetes_nested_cv_summary.csv")
    _write_csv(paired, artifact_dir / "diabetes_nested_cv_paired_differences.csv")
    audit = audit_synthetic_claims(root / "artifacts" / "phase2" / "synthetic_runs.csv")
    _write_csv(audit, artifact_dir / "synthetic_claim_audit.csv")
    _plot_benchmark(summaries, figure_dir / "diabetes_nested_cv_mse.png")
    print(f"Final release results saved under {artifact_dir} and {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the diabetes nested-CV benchmark.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small temporary benchmark for CI without touching saved results.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="double-descent-smoke-") as temp_dir:
            temp_root = Path(temp_dir)
            run_benchmark(
                temp_root / "artifacts",
                temp_root / "figures",
                repeats=2,
                outer_folds=3,
                inner_folds=2,
                widths=(8, 16),
                ridge_alphas=(0.1, 1.0),
            )
        print("Temporary smoke-test outputs were removed.")
        return
    run_benchmark(root / "artifacts" / "final_release", root / "figures" / "final_release")


if __name__ == "__main__":
    main()
