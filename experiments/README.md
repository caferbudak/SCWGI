# Reproducing the article

All scripts are run from the repository root (`python experiments/<script>.py`). Every script uses the same
common protocol:

- random 70 % / 30 % train–test split (seed 0), z-scaling with training statistics;
- missingness applied to the **test** set only, mask seeds 42, 142, 242, 342, 442;
- MCAR (each cell hidden independently), MAR (half of the columns are always-observed references; missingness in the
  other columns increases monotonically with a randomly chosen reference column), MNAR (values below the ρ-th
  percentile of each column are hidden; deterministic);
- MAE, RMSE and R² computed on the hidden cells only, in z-scale, pooled over cells.

| Dataset | Conditions |
|---|---|
| Breast Cancer (WDBC) | MCAR 10/20/30 % |
| Gold Price | MCAR 5/10/15 % |
| CWRU bearing (25 window features) | MCAR 10 % |
| Diabetes (Pima) | MAR 10/20 % |
| Fertility | MCAR and MNAR 10/20/30 % |

## Data

All data files used in the article are included in this repository:

```
data/tabular/diabetes.csv              Pima Indians Diabetes (UCI / Kaggle)
data/tabular/fertility_Diagnosis.csv   UCI Fertility (Gil et al., 2012)
data/tabular/FINAL_USO.csv             Kaggle "Gold Price Prediction Dataset" (sid321axn)
data/cwru/*.csv                        CWRU Bearing Data Center, drive-end accelerometer (DE_time), 12 kHz, 1 hp:
                                       Time_Normal_1_098, B007_1_123, B014_1_190, B021_1_227, IR007_1_110,
                                       IR014_1_175, IR021_1_214, OR007_6_1_136, OR014_6_1_202, OR021_6_1_239
```

Breast Cancer (WDBC) is loaded from scikit-learn. `data/tabular/feature_time_48k_2048_load_1.csv` is kept from an
earlier version of this repository and is not used by the experiments.

## Scripts

Run from the repository root (`python experiments/<script>.py`).

| Script | Produces |
|---|---|
| `run_scwgi.py --mode inductive` | `results/scwgi/SCWGI_raw.csv` |
| `run_scwgi.py --mode transductive` | `results/scwgi/SCWGI-T_raw.csv` |
| `run_baselines_lib.py` | Mean, EM, MICE, MissForest, GAIN, SoftImpute, MIWAE (HyperImpute library v0.1.17) → `results/competitors/` |
| `run_hyperimpute.py` | HyperImpute → `results/competitors/` |
| `run_deep_baselines.py` | TabDDPM, SAINT → `results/competitors/deep_raw.csv` |
| `run_ablation.py` | component ablation → `results/ablation/` |
| `run_calibration.py` | conformal coverage and width → `results/calibration_raw.csv` |
| `make_results.py` | Table 4 (R²), MAE/RMSE tables, ranks, Friedman and Holm–Wilcoxon tests, timings → `results/summary/` |
| `make_figures.py` | Figures 2–4 → `results/figures/` |
| `make_figure1.py` | Figure 1 (schematic) → `results/figures/` |

Library baselines and HyperImpute are fitted on the training rows stacked with the masked test rows (these methods are
transductive by design); SCWGI, TabDDPM and SAINT are trained on the training rows only. SCWGI-T gives SCWGI the same
information as the transductive baselines. The Sinkhorn plugin was started but not completed (about 4 minutes per
condition even on Diabetes); its partial file is kept for transparency and is excluded from all summaries.

### Environment notes

The HyperImpute library (v0.1.17) requires scikit-learn < 1.6 and pandas < 3. Its EM plugin contains the expression
`np.where(np.isnan(sum(X.T)) is False)[0]`, which always returned an empty index under NumPy 1.x and raises under
NumPy ≥ 2; we replaced it with `np.array([], dtype=int)`, preserving the original behaviour (diagonal initial
covariance). The Sinkhorn plugin additionally needs `geomloss`.
