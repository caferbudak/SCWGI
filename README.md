# SCWGI — Soft-Cluster Weighted Gaussian Imputer

Code, raw results and figure scripts accompanying the article *"SCWGI: …"* (Terzioğlu & Budak).

SCWGI imputes missing values in numeric tabular data and returns, for every imputed cell, a point estimate and a
calibrated prediction interval. It has no gradient-trained component: every setting is chosen by masked
cross-validation inside the training set, and imputation uses closed-form conditional-Gaussian computations.

1. **Atom indicators.** A feature with at least 11 distinct values whose modal value has relative frequency ≥ 5 %
   (e.g. zero-inflated clinical measurements) gets an indicator column whose mask is tied to the feature's mask.
2. **Data-adaptive marginal.** Each dataset chooses, by internal masked CV, between a linear (z-score) space and an
   empirical Gaussian-copula space (Gauss–Hermite back-transformation, 32 nodes).
3. **Soft-cluster weighted conditional Gaussian.** FCM-initialised EM Gaussian mixtures; cluster covariances shrunk
   towards the Ledoit–Wolf estimate, Σ̃ = (1 − λ)Σ̂ + λΣ_LW + εI; cluster weights from the likelihood of the *observed*
   coordinates only; Cholesky conditional means and variances; the conditional variance follows the law of total
   variance. Mixtures with K ∈ {1, 2, 3, 4, 6, 8} are averaged with weights w_K ∝ exp(−(e_K − e_min)/(τ e_min)).
4. **Conformal uncertainty.** Split and Mondrian (tertiles of the model's conditional s.d.) conformal intervals on the
   standardised score |y − x̂| / s.

Two inference modes are provided:
- **SCWGI (inductive):** fitted on fully observed training rows only; never sees the rows it imputes.
- **SCWGI-T (transductive):** identical training; at inference the mixture components are re-estimated by
  missing-data EM on the training rows *plus the observed cells* of the rows being imputed. Hidden values are never used.

## Installation

```bash
pip install numpy scipy scikit-learn torch pandas
pip install -e .
```

## Usage

```python
from scwgi import SCWGIImputer, ConformalCalibrator

imp = SCWGIImputer(random_state=42).fit(X_train)        # fully observed training rows (z-scored)
X_hat = imp.transform(X_missing, M)                      # M: 1 = observed, 0 = missing
X_hat, s = imp.transform_dist(X_missing, M)              # conditional mean and conditional s.d.

cal = ConformalCalibrator(alpha=0.10).fit(imp, X_cal, rate=0.2)
x_hat, lower, upper, s = cal.intervals(imp, X_missing, M, mondrian=True)
```

## Reproducing the article

`experiments/README.md` describes the protocol, the data files (included under `data/`), and which script produces each
table and figure. All raw per-seed results used in the article are in `results/` (see `results/README.md`).

`tests/test_regression.py` checks that the packaged implementation reproduces the research code that produced the
article's results (linear and copula marginals, atom indicators, transductive mode).
