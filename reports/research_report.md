# Double Descent Lab
*A hands-on study of model capacity, noise, and regularization.*

## Phase 1: interpolation and numerical stability

## Research question

How do model capacity, observation noise, and ridge regularization affect generalization around the interpolation threshold in a one-dimensional ReLU feature-regression experiment?

## Experimental setup

The input is sampled uniformly from `[-1, 1]`; targets follow `sin(2πx) + ε` with Gaussian noise standard deviation `0.1` or `0.5`. Each data seed generates its own train/test split; the same split and same fixed feature map are used for OLS and ridge. The selected sample sizes are 200, 500, and 1,000 observations, with 30 independent data seeds per configuration.

To make interpolation measurable in this one-dimensional setting, feature slopes are random signs and the ReLU knots are empirical quantiles of training inputs. Knot placement uses only training inputs, not labels or test data. This produces a calibrated spline-like ReLU feature family; results should not be generalized to every iid random-feature initialization.

The capacity grid samples from 25% to 200% of the number of training rows and becomes denser around that count. Each width `m` has `m` ReLU features plus one unpenalized intercept. OLS uses SVD-based `numpy.linalg.lstsq`. Ridge (`alpha=0.001`) uses augmented least squares and leaves the intercept unpenalized. The numerical rank tolerance is `max(matrix.shape) * machine_epsilon * largest_singular_value`; the condition number is infinite when numerical rank is below `min(matrix.shape)`. Interpolation requires `train_mse <= 1e-10 + 1e-8 * Var(y_train)`.

The polynomial comparison fits the same intercept-inclusive models using raw monomial and Legendre bases, with OLS and ridge. It is limited to the 200-sample, noise-0.1 configuration and degrees 1–40. Summary confidence intervals are 95% percentile bootstrap intervals for the mean, based on 5,000 resamples of independent seed-level results. Raw rows and settings are preserved in `artifacts/`.

## Numerical problems found and addressed

- The prior random ReLU initialization gave very low effective rank because many one-dimensional units were inactive or redundant. Width up to 2,048 had failed to interpolate 160 training rows. Empirical-quantile knots were introduced so model capacity can be measured against actual rank and training error.
- The previous ridge implementation formed `Phi.T @ Phi`, squaring the condition number. The current implementation instead solves an augmented least-squares problem.
- The old polynomial curve used raw monomials without an intercept and showed rank loss and extreme conditioning. The comparison now includes an intercept consistently, adds a Legendre basis, reports rank/conditioning, and includes ridge.
- The prior reports treated feature width as if it established interpolation. The new output records feature count, numerical rank, condition number, a scale-aware interpolation tolerance, and the observed interpolation flag.

## Results

The final random-feature run produced 4,680 seed-level rows: 3 sample sizes × 2 noise levels × 30 seeds × 13 capacities × 2 solvers. Every OLS run at hidden width `n_train` reached full row rank and met the interpolation tolerance (100% of runs in each condition). At that capacity, mean training MSE ranged from `4.75e-21` (`n=200`, σ=0.1) to `3.92e-14` (`n=1,000`, σ=0.5). Ridge did not interpolate.

The table compares an underparameterized capacity (`0.75 × n_train`), hidden width `n_train` (there are `n_train + 1` design columns including the intercept), twice the training-set size, the maximum observed OLS test-MSE point on the grid, and ridge at width `n_train`. Entries are mean test MSE with a 95% bootstrap interval for the mean.

| Samples | Noise σ | OLS under (`0.75 n_train`) | OLS at `n_train` | OLS at `2 n_train` | OLS peak (width) | Ridge at `n_train` |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.1 | 0.057 [0.037, 0.083] | 697 [18, 1,919] | 10.8 [0.52, 29.5] | 697 (160) | 0.0110 [0.0104, 0.0117] |
| 200 | 0.5 | 1.43 [0.92, 2.04] | 17,432 [468, 47,968] | 269 [13, 732] | 17,432 (160) | 0.276 [0.259, 0.292] |
| 500 | 0.1 | 0.050 [0.035, 0.072] | 13,024 [96, 38,099] | 27.7 [0.98, 77.1] | 13,024 (400) | 0.0113 [0.0105, 0.0120] |
| 500 | 0.5 | 1.26 [0.87, 1.82] | 325,587 [2,313, 951,864] | 693 [25, 1,932] | 325,587 (400) | 0.282 [0.264, 0.300] |
| 1,000 | 0.1 | 0.044 [0.037, 0.052] | 2,298 [162, 6,369] | 241 [3.64, 690] | 7,836 (840) | 0.0107 [0.0103, 0.0110] |
| 1,000 | 0.5 | 1.10 [0.93, 1.29] | 57,449 [3,957, 159,278] | 6,028 [94.7, 17,267] | 195,895 (840) | 0.267 [0.257, 0.276] |

The previous grid point (width `0.95 × n_train`) did not interpolate; width `n_train` reached full row rank and interpolated in all 30 seeds for each condition. These results support an empirical interpolation-associated test-error peak followed by a substantial descending branch for unregularized least squares in this calibrated feature family. Increasing noise strongly increases the peak. However, the overparameterized test error remains well above the underparameterized value in all six settings, and some confidence intervals at high capacity are wide. This is a peak-and-partial-descent pattern, not a return to the original low-error regime. Fixed-alpha ridge largely removes the peak but also avoids interpolation; it is a regularized comparison, not a tuned optimum.

The design is numerically sensitive even at full row rank: mean condition numbers at hidden width `n_train` were approximately `5.88e7`, `8.15e8`, and `5.06e9` for 200, 500, and 1,000 samples, respectively. The trend and stable training fits are measured with SVD-based least squares, but the large condition numbers prevent a strong claim that numerical sensitivity is irrelevant. These findings are evidence for a sharp generalization peak near interpolation in this controlled setup, not a clean demonstration of a universal double-descent law.

The polynomial comparison reinforces that caution. At degree 40, raw monomials had numerical rank 37.5 on average out of 41 columns and an infinite condition number in all 30 seeds; OLS test MSE averaged 222, while ridge reduced it to 0.024. Legendre features had full rank and mean condition number about `3.79e4`, but OLS test MSE averaged 6,924; ridge reduced it to 0.100. A better-conditioned basis does not by itself prevent high-variance fits. Ridge results are basis-dependent because the same coefficient penalty has different meanings under different feature scalings. The raw-monomial tail is additionally affected by rank truncation and numerical conditioning, so neither polynomial curve is presented as independent evidence of double descent.

The saved tables contain 4,800 polynomial seed-level rows (30 seeds × 40 degrees × 2 bases × 2 solvers), 156 random-feature summary groups, 160 polynomial summary groups, and 78 seed-paired ridge-minus-OLS comparisons. Every plotted point is derived from these summaries.

![Phase 1 repeated-seed capacity sweep and 95% bootstrap confidence intervals](../assets/figures/double-descent-overview.png)

![Phase 1 training-rank, conditioning, and observed interpolation fraction from saved run-level results](../assets/figures/interpolation-diagnostics.png)

## Interpretation and limitations

The experiment is synthetic and one-dimensional. Training-quantile knot placement is a deliberate capacity calibration and differs from standard iid random-feature theory. Ridge alpha is fixed rather than tuned on a separate validation set; the ridge curve demonstrates regularization sensitivity, not an optimized model. Percentile bootstrap intervals characterize variability across the configured random seeds and are not universal guarantees. High-degree polynomial results remain sensitive to basis conditioning and finite precision.

The conclusions should be restricted to the configurations measured here. They do not establish that double descent must occur for other feature distributions, target functions, sample sizes, or real datasets.

## Reproduction

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python src\experiment_runner.py --n-samples 200 500 1000 --noise-levels 0.1 0.5 --repetitions 30 --seed 42 --max-degree 40 --ridge-alpha 0.001
```

## Phase 2: robustness and external validation

### Method

Phase 1 code, outputs, and figures are retained as the original baseline. Phase 2 is implemented separately in `src/phase2_runner.py`; its generated files are isolated under `artifacts/phase2/` and `figures/phase2/`.

The synthetic study uses 200 observations per condition (160 training and 40 test), 30 data/model seed pairs, noise standard deviations 0.1 and 0.5, and the Phase 1 capacity grid. It compares three targets: a single-coordinate sinusoid, a normalized additive sinusoid over up to three coordinates, and a two-coordinate sinusoidal interaction. The tested input dimensions are 1, 5, and 20 where defined (seven target/dimension combinations in total). OLS and fixed-alpha ridge (`alpha=0.001`) use the existing random-ReLU model, stable least-squares solvers, rank/condition diagnostics, and interpolation tolerance from Phase 1. The one-dimensional empirical-quantile knot initialization is retained; dimensions above one use the existing Gaussian random weights and uniform biases. Those feature families are not identical, so cross-dimension differences are conditional on the initialization protocol as well as on dimension.

Each summary reports seed-level mean and sample standard deviation, plus a 95% percentile-bootstrap confidence interval for the mean (5,000 resamples). For OLS, a curve is counted as having a pointwise peak followed by descent when its maximum test-MSE point is interior and exceeds both immediate neighbors. The saved classification's stricter flag additionally requires full row rank, interpolation in at least 80% of runs at the peak, and non-overlapping marginal peak/neighbor confidence intervals. That last condition does not account for the paired seeds and should not be treated as a paired test; a seed-matched bootstrap audit is reported below.

The original Phase 2 external run used five repeated 60/20/20 train/validation/test partitions (264/89/89 rows), selecting a random-feature model on validation MSE. Those files are retained unchanged as historical results. During release review, the default `load_diabetes()` behavior was identified as scaling features using the full dataset before the train-only scaler was fit. The new benchmark below avoids this preprocessing leakage by loading raw features with `load_diabetes(scaled=False)` and putting scaling inside each inner/outer training fold. The five-split result is therefore not used for the final model comparison.

### Synthetic results

At width 120, mean numerical rank was about 121 and no OLS run interpolated. At width 160 (= the 160 training rows; 161 design columns including the intercept), all 30 OLS runs in every target/dimension/noise condition reached rank 160 and met the scale-aware interpolation tolerance. Across these 420 OLS fits, mean training MSE at width 160 was `8.28e-22`. The actual rank and training error—not width alone—establish interpolation here.

All 14 OLS curves had an interior pointwise maximum at width 160 followed by lower test MSE at the next capacity. The table reports the width-160 mean and its 95% bootstrap interval. Under the saved marginal-CI-overlap flag, 12/14 configurations passed and the two one-dimensional single-sinusoid configurations did not. This saved flag is not a paired statistical comparison: each seed uses the same data and test partition at neighboring capacities.

| Target | Input dim | Test MSE peak, noise 0.1 (95% CI) | Test MSE peak, noise 0.5 (95% CI) | CI-separated interpolation peak |
|---|---:|---:|---:|---|
| Single sinusoid | 1 | 201 [24, 493] | 5,034 [577, 12,289] | No (both noise levels) |
| Single sinusoid | 5 | 330 [71, 661] | 547 [169, 1,100] | Yes (both) |
| Single sinusoid | 20 | 90.7 [50.6, 147] | 178 [94.9, 296] | Yes (both) |
| Additive sinusoid | 5 | 158 [75, 261] | 330 [151, 564] | Yes (both) |
| Additive sinusoid | 20 | 122 [59, 205] | 234 [102, 412] | Yes (both) |
| Interaction | 5 | 33.6 [13.4, 60.6] | 272 [109, 522] | Yes (both) |
| Interaction | 20 | 75.4 [39.6, 122] | 126 [63.1, 212] | Yes (both) |

The width-160 mean design condition number varied substantially with input dimension: approximately `2.09e7` for dimension 1, `4.31e4` for dimension 5, and `4.34e3` for dimension 20. The OLS interpolation fits are therefore accompanied by pronounced sensitivity to conditioning, especially in the calibrated one-dimensional feature map. Higher observation noise increased the measured peak in every target/dimension comparison. Descent beyond the peak was present in all tested curves, but it did not imply low test error: for example, the one-dimensional single-sinusoid test MSE at width 320 remained far above its underparameterized value. Other target/dimension configurations recovered more strongly. Fixed-alpha ridge did not interpolate at width 160 (mean training MSE `0.0173`) and serves as a regularized reference, not a validation-tuned optimum.

The final audit recomputed each capacity mean from the 10,920 saved run-level rows and verified all 14 peaks against their immediate lower- and higher-capacity neighbors. It found zero disagreement between the stored OLS interpolation flags and `train_mse <= interpolation_tolerance`, zero deviation from `1e-10 + 1e-8 × target_variance`, and full row rank (`160/160`) with interpolation in `30/30` runs at every width-160 peak. For each condition, it then bootstrapped the paired per-seed differences between peak test MSE and each immediate neighbor (10,000 resamples). Both 95% intervals were entirely above zero in all 14 conditions, including the two one-dimensional cases that failed the original non-overlapping-marginal-CI rule. On this paired, local criterion, all 14 conditions support an interpolation-associated peak followed by descent.

This paired result is exploratory: the peak capacity was selected as the maximum from the same capacity grid being assessed, the adjacent comparisons are not corrected for multiple testing, and the response depends strongly on conditioning and feature initialization. It is stronger evidence than a pointwise plot, but it is not a confirmatory test or a universal double-descent claim. The tested conditions are exactly the three synthetic target definitions, dimensions, two noise levels, capacity grid, and random-feature initialization protocols described above.

These results show an interpolation-associated peak and descending branch for the particular finite synthetic grid and feature-generation protocols tested. They do not establish a universal double-descent effect: target, dimension, representation, capacity grid, and numerical conditioning all matter, and the one-dimensional intervals are especially broad.

The curated README figure presents a representative Phase 1 condition. The full Phase 2 multi-condition plot remains available at `figures/phase2/synthetic_robustness_curves.png`; it contains all target/dimension/noise combinations and is less suitable as a compact landing-page image.

### Final real-data benchmark: repeated nested cross-validation

The release benchmark compares constant mean prediction, ordinary linear regression, ridge regression, and the existing random-feature model on the scikit-learn diabetes data (442 observations, 10 predictors). It uses `load_diabetes(scaled=False)` to avoid the dataset loader's global feature scaling. Evaluation is 10 repetitions of shuffled 5-fold outer cross-validation (50 paired outer test folds); all four models receive exactly the same folds. Linear regression and ridge use `StandardScaler` pipelines. The random-feature pipeline fits its scaler only on each fit partition and uses the existing stable OLS/augmented-least-squares ridge solver.

Hyperparameters are selected only within each outer training partition using shuffled 5-fold inner CV and mean validation MSE. Ridge searches alphas `1e-4`, `1e-3`, `1e-2`, `1e-1`, and `1.0`. Random features search widths 16, 32, 64, 128, and 256, OLS or ridge readouts, and the same ridge-alpha grid. The outer test fold is scored only after selection and final refitting on the full outer training fold. The mean and linear models have no tuned hyperparameters.

The following means average all outer fold scores. Standard deviations are across the 50 outer-fold scores; the 95% percentile-bootstrap intervals resample the 10 repetition-level means (10,000 resamples), keeping folds within a repetition clustered.

| Model | MSE mean (outer-fold SD) | MSE 95% interval | RMSE mean (outer-fold SD) | RMSE 95% interval |
|---|---:|---:|---:|---:|
| Mean prediction | 5,958 (628) | [5,948, 5,969] | 77.08 (4.07) | [77.02, 77.14] |
| Linear regression | 3,015 (381) | [2,995, 3,036] | 54.80 (3.48) | [54.58, 55.02] |
| Ridge (inner-CV alpha) | 3,015 (381) | [2,996, 3,037] | 54.81 (3.47) | [54.59, 55.03] |
| Random features (inner-CV selection) | 3,278 (477) | [3,217, 3,335] | 57.11 (4.08) | [56.60, 57.60] |

Paired differences are random-feature error minus baseline error on the same outer fold; intervals use a bootstrap clustered by repetition.

| Comparison | MSE difference (95% interval) | RMSE difference (95% interval) |
|---|---:|---:|
| Random features − mean prediction | −2,680 [−2,738, −2,626] | −19.97 [−20.47, −19.49] |
| Random features − linear regression | +263 [+185, +333] | +2.31 [+1.63, +2.92] |
| Random features − ridge | +263 [+186, +335] | +2.31 [+1.64, +2.93] |

The random-feature model substantially improves on the constant-mean baseline but does not improve on either linear baseline in this protocol; the paired intervals indicate higher error than linear regression and ridge. The linear and ridge averages are nearly identical. These findings do not support a claim that random features are superior for this real dataset. This is one external dataset; repeated folds reuse observations, so the intervals quantify split/repetition variability conditional on this dataset and protocol, not uncertainty across independent datasets.

![Diabetes benchmark outer-fold MSE and 95% repeated-CV intervals](../assets/figures/diabetes-benchmark.png)

### Final release artifacts, limitations, and reproduction

The original Phase 1 and Phase 2 result files remain unchanged. New outputs are isolated under `artifacts/final_release/`: `release_config.json`, `diabetes_nested_cv_runs.csv`, `diabetes_nested_cv_summary.csv`, `diabetes_nested_cv_paired_differences.csv`, and `synthetic_claim_audit.csv`. The benchmark plot is `assets/figures/diabetes-benchmark.png`; the benchmark runner also retains its phase output at `figures/final_release/diabetes_nested_cv_mse.png`. The audit file includes all 14 conditions, paired peak-minus-neighbor intervals, and rank/interpolation consistency checks.

The synthetic evidence applies only to the tested synthetic targets, data sizes, noise levels, capacities, and feature initialization. Paired peak comparisons are exploratory and unadjusted for post-selection or multiplicity. The real-data benchmark covers only one small regression dataset; repeated cross-validation does not create independent datasets. Random-feature search uses a finite candidate grid, and neither these predictive results nor the synthetic curves establish universal behavior.

From the repository root, reproduce the final benchmark and audit, then run the complete tests:

```powershell
python src\diabetes_benchmark.py
python -m pytest -q
```

The benchmark writes to `artifacts/final_release/` and `figures/final_release/`. It reads the preserved `artifacts/phase2/synthetic_runs.csv` for the synthetic claim audit and does not regenerate or overwrite Phase 1/Phase 2 outputs. `python src\figure_assets.py` renders the canonical README images under `assets/figures/` from saved Phase 1 and final-release CSVs.

The preserved Phase 2 configuration, run-level CSVs, summaries, and original curve classification remain under `artifacts/phase2/`; phase-specific plots remain under `figures/phase2/`. The old five-split diabetes evaluation is retained for provenance but is not the primary external benchmark.

From the repository root, reproduce the synthetic study, final benchmark, curated figures, and tests:

```powershell
python src\phase2_runner.py
python src\diabetes_benchmark.py
python src\figure_assets.py
python -m pytest -q
```
