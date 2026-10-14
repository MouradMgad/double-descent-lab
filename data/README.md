# Data provenance

The real-data benchmark uses the diabetes regression dataset distributed by scikit-learn through `sklearn.datasets.load_diabetes(scaled=False)`. The dataset is packaged with scikit-learn, so the experiment does not download or commit a separate data file.

The benchmark deliberately requests the unscaled predictors and fits `StandardScaler` only on training folds. Consult the dataset and scikit-learn documentation for upstream provenance and licensing details; this repository does not claim ownership of the dataset.
