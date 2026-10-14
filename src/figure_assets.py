from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
PHASE1_RUNS = ROOT / "artifacts" / "random_feature_runs.csv"
PHASE1_SUMMARY = ROOT / "artifacts" / "random_feature_summary.csv"
BENCHMARK_SUMMARY = ROOT / "artifacts" / "final_release" / "diabetes_nested_cv_summary.csv"
OUTPUT_DIR = ROOT / "assets" / "figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment output is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _phase1_condition(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row for row in rows
        if int(row["n_samples"]) == 200 and float(row["noise_std"]) == 0.1
    ]
    if not selected:
        raise ValueError("Phase 1 results do not contain n_samples=200, noise_std=0.1")
    return selected


def plot_synthetic_overview(summary_path: Path, output_path: Path) -> None:
    rows = _phase1_condition(_read_csv(summary_path))
    run_rows = _phase1_condition(_read_csv(PHASE1_RUNS))
    fig, (test_ax, train_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    styles = {"ols": ("#1769aa", "OLS"), "ridge": ("#d97706", "Ridge (α=0.001)")}
    n_samples = int(rows[0]["n_samples"])
    n_train = n_samples - int(np.ceil(n_samples * 0.2))
    for solver, (color, label) in styles.items():
        curve = sorted(
            (row for row in rows if row["solver"] == solver),
            key=lambda row: int(row["hidden_width"]),
        )
        widths = np.asarray([int(row["hidden_width"]) for row in curve])
        test_mean = np.asarray([float(row["test_mse_mean"]) for row in curve])
        lower = np.asarray([float(row["test_mse_ci95_low"]) for row in curve])
        upper = np.asarray([float(row["test_mse_ci95_high"]) for row in curve])
        train_mean = np.asarray([float(row["train_mse_mean"]) for row in curve])
        test_ax.plot(widths, test_mean, marker="o", markersize=4, linewidth=2, color=color, label=label)
        test_ax.fill_between(widths, lower, upper, color=color, alpha=0.16, linewidth=0)
        train_ax.plot(widths, train_mean, marker="o", markersize=4, linewidth=2, color=color, label=label)
    tolerance = float(np.median([float(row["interpolation_tolerance"]) for row in run_rows]))
    test_ax.axvline(
        n_train,
        color="#4b5563",
        linestyle="--",
        linewidth=1.3,
        label=f"Interpolation reference (n_train={n_train})",
    )
    train_ax.axvline(n_train, color="#9ca3af", linestyle=":")
    train_ax.axhline(
        tolerance,
        color="#4b5563",
        linestyle="--",
        linewidth=1.3,
        label="Median interpolation tolerance",
    )
    test_ax.set_yscale("log")
    train_ax.set_yscale("log")
    test_ax.set_ylabel("Test MSE (95% bootstrap CI)")
    train_ax.set_xlabel("Random ReLU hidden-feature width")
    train_ax.set_ylabel("Training MSE")
    test_ax.set_title("Double Descent Lab | Synthetic capacity sweep")
    test_ax.text(
        0.02,
        0.03,
        "n=200  ·  noise σ=0.1  ·  30 seeds",
        transform=test_ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    test_ax.grid(True, which="both", linestyle=":", alpha=0.35)
    train_ax.grid(True, which="both", linestyle=":", alpha=0.35)
    test_ax.legend(frameon=False, loc="upper left")
    train_ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, metadata={"Software": "Double Descent Lab"})
    plt.close(fig)


def plot_interpolation_diagnostics(runs_path: Path, output_path: Path) -> None:
    rows = _phase1_condition(_read_csv(runs_path))
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["solver"] == "ols":
            grouped[int(row["hidden_width"])].append(row)
    widths = sorted(grouped)
    n_samples = int(rows[0]["n_samples"])
    n_train = n_samples - int(np.ceil(n_samples * 0.2))
    mean_rank = [np.mean([int(row["rank"]) for row in grouped[width]]) for width in widths]
    interpolation_rate = [
        np.mean([row["interpolates"].lower() == "true" for row in grouped[width]])
        for width in widths
    ]
    condition_numbers = [
        np.mean(
            [
                float(row["condition_number"])
                for row in grouped[width]
                if np.isfinite(float(row["condition_number"]))
            ]
        )
        for width in widths
    ]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    rank_ax, error_ax = axes
    rank_ax.plot(widths, mean_rank, "o-", color="#1769aa", linewidth=2, markersize=4)
    rank_ax.axhline(n_train, color="#4b5563", linestyle="--", label=f"n_train={n_train}")
    rank_ax.axvline(n_train, color="#9ca3af", linestyle=":")
    rank_ax.set_ylabel("Numerical rank of training design")
    rank_ax.set_title("Double Descent Lab | Interpolation diagnostics (OLS, 30 seeds)")
    rank_ax.grid(True, linestyle=":", alpha=0.35)
    rank_ax.legend(frameon=False, loc="lower right")

    error_ax.plot(widths, condition_numbers, "o-", color="#a23b72", linewidth=2, markersize=4)
    error_ax.axvline(n_train, color="#9ca3af", linestyle=":")
    error_ax.set_yscale("log")
    error_ax.set_xlabel("Random ReLU hidden-feature width")
    error_ax.set_ylabel("Condition number (finite runs)")
    error_ax.grid(True, which="both", linestyle=":", alpha=0.35)
    rate_ax = error_ax.twinx()
    rate_ax.plot(
        widths,
        interpolation_rate,
        "s:",
        color="#16845b",
        markersize=4,
        linewidth=1.5,
        label="Interpolating run fraction",
    )
    rate_ax.set_ylim(-0.05, 1.05)
    rate_ax.set_ylabel("Interpolating runs", color="#16845b")
    rate_ax.tick_params(axis="y", colors="#16845b")
    lines, labels = error_ax.get_legend_handles_labels()
    rate_lines, rate_labels = rate_ax.get_legend_handles_labels()
    error_ax.legend(lines + rate_lines, labels + rate_labels, frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, metadata={"Software": "Double Descent Lab"})
    plt.close(fig)


def plot_diabetes_benchmark(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    order = ("mean_prediction", "linear_regression", "ridge", "random_features")
    labels = ("Mean predictor", "Linear regression", "Ridge", "Random features")
    palette = ("#7b8794", "#1769aa", "#16845b", "#d97706")
    by_name = {row["model"]: row for row in rows}
    missing = set(order) - set(by_name)
    if missing:
        raise ValueError(f"Benchmark summary lacks model rows: {sorted(missing)}")
    means = np.asarray([float(by_name[name]["test_mse_mean"]) for name in order])
    lower = np.asarray([float(by_name[name]["test_mse_ci95_low"]) for name in order])
    upper = np.asarray([float(by_name[name]["test_mse_ci95_high"]) for name in order])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.errorbar(
        x,
        means,
        yerr=np.vstack((means - lower, upper - means)),
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=2,
        color="#263238",
    )
    for index, color in enumerate(palette):
        ax.scatter(x[index], means[index], s=78, color=color, zorder=3)
        ax.annotate(f"{means[index]:,.0f}", (x[index], means[index]), xytext=(8, 5), textcoords="offset points", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Outer-fold test MSE")
    ax.set_title("Double Descent Lab | Diabetes benchmark")
    ax.text(
        0.02,
        0.96,
        "10 repeats × 5 outer folds; 95% intervals bootstrap repeat-level means",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#4b5563",
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, metadata={"Software": "Double Descent Lab"})
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_synthetic_overview(
        PHASE1_SUMMARY, OUTPUT_DIR / "double-descent-overview.png"
    )
    plot_interpolation_diagnostics(
        PHASE1_RUNS, OUTPUT_DIR / "interpolation-diagnostics.png"
    )
    plot_diabetes_benchmark(
        BENCHMARK_SUMMARY, OUTPUT_DIR / "diabetes-benchmark.png"
    )
    print(f"Canonical figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
