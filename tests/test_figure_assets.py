from pathlib import Path

from src.figure_assets import (
    BENCHMARK_SUMMARY,
    OUTPUT_DIR,
    PHASE1_RUNS,
    PHASE1_SUMMARY,
    plot_diabetes_benchmark,
    plot_interpolation_diagnostics,
    plot_synthetic_overview,
)


def test_saved_result_files_exist_for_reproducible_canonical_figures():
    assert PHASE1_RUNS.is_file()
    assert PHASE1_SUMMARY.is_file()
    assert BENCHMARK_SUMMARY.is_file()


def test_canonical_figures_render_from_saved_csvs(tmp_path: Path):
    outputs = (
        tmp_path / "double-descent-overview.png",
        tmp_path / "interpolation-diagnostics.png",
        tmp_path / "diabetes-benchmark.png",
    )
    plot_synthetic_overview(PHASE1_SUMMARY, outputs[0])
    plot_interpolation_diagnostics(PHASE1_RUNS, outputs[1])
    plot_diabetes_benchmark(BENCHMARK_SUMMARY, outputs[2])
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in outputs)
    assert OUTPUT_DIR.parent.name == "assets"
    assert OUTPUT_DIR.name == "figures"
