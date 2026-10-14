import csv

import numpy as np

from src.diabetes_benchmark import audit_synthetic_claims, repeated_nested_diabetes_cv


def test_repeated_nested_cv_uses_identical_outer_splits_without_test_selection():
    rng = np.random.default_rng(81)
    X = rng.normal(size=(48, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=len(X))
    runs, summaries, paired = repeated_nested_diabetes_cv(
        X,
        y,
        repeats=2,
        outer_folds=3,
        inner_folds=3,
        widths=(3, 6),
        ridge_alphas=(0.01, 0.1),
        seed=19,
    )

    assert len(runs) == 2 * 3 * 4
    assert {row["model"] for row in runs} == {
        "mean_prediction",
        "linear_regression",
        "ridge",
        "random_features",
    }
    for repeat in range(2):
        for outer_fold in range(3):
            fold = [
                row
                for row in runs
                if row["repeat"] == repeat and row["outer_fold"] == outer_fold
            ]
            assert len({row["outer_test_indices"] for row in fold}) == 1
            assert all(row["outer_test_used_for_selection"] is False for row in fold)
            assert all(row["n_train"] + row["n_test"] == len(X) for row in fold)
            assert all(
                row["scaler_fit_rows"] == row["n_train"]
                for row in fold
                if row["model"] != "mean_prediction"
            )

    assert len(summaries) == 4
    assert len(paired) == 3
    assert all(np.isfinite(row["test_rmse_mean"]) for row in summaries)
    assert all(
        row["test_mse_ci95_low"] <= row["test_mse_mean"] <= row["test_mse_ci95_high"]
        for row in summaries
    )
    assert all(row["n_paired_outer_folds"] == 6 for row in paired)
    assert all(
        np.isfinite(row["test_mse_mean_difference"])
        and np.isfinite(row["test_rmse_mean_difference"])
        for row in paired
    )


def test_nested_cv_rejects_mismatched_data_and_invalid_fold_counts():
    X = np.ones((12, 2))
    y = np.ones(11)
    try:
        repeated_nested_diabetes_cv(X, y, repeats=2, outer_folds=3, inner_folds=2)
    except ValueError as error:
        assert "matching rows" in str(error)
    else:
        raise AssertionError("mismatched X and y should be rejected")


def test_synthetic_claim_audit_uses_paired_seed_differences(tmp_path):
    path = tmp_path / "runs.csv"
    fields = [
        "target",
        "input_dim",
        "noise_std",
        "solver",
        "hidden_width",
        "repetition",
        "n_train",
        "rank",
        "train_mse",
        "target_variance",
        "interpolation_tolerance",
        "interpolates",
        "test_mse",
    ]
    rows = []
    for repetition in range(12):
        for width, test_mse, train_mse, rank, interpolates in (
            (4, 2.0, 0.2, 5, False),
            (8, 100.0 + repetition, 1e-15, 8, True),
            (12, 3.0, 1e-15, 8, True),
        ):
            rows.append(
                {
                    "target": "fixture",
                    "input_dim": 2,
                    "noise_std": 0.1,
                    "solver": "ols",
                    "hidden_width": width,
                    "repetition": repetition,
                    "n_train": 8,
                    "rank": rank,
                    "train_mse": train_mse,
                    "target_variance": 0.0,
                    "interpolation_tolerance": 1e-10,
                    "interpolates": interpolates,
                    "test_mse": test_mse,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    [result] = audit_synthetic_claims(path, bootstrap_seed=3, bootstrap_resamples=1000)
    assert result["pointwise_peak_then_descent"]
    assert result["all_peak_runs_full_row_rank"]
    assert result["interpolation_flag_matches_tolerance"]
    assert result["peak_interpolation_rate"] == 1.0
    assert result["paired_interpolation_peak_supported"]
    assert result["peak_minus_left_ci95_low"] > 0
    assert result["peak_minus_right_ci95_low"] > 0
