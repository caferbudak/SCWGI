# Raw results

Every file holds one row per (dataset, mechanism, missing rate, mask seed) unless stated otherwise. Metrics are computed
on the hidden test cells only, in z-scale, pooled over cells.

| Path | Content |
|---|---|
| `scwgi/SCWGI_raw.csv` | SCWGI (inductive): MAE, RMSE, R², fit + impute time, selected marginal, K with the largest weight |
| `scwgi/SCWGI-T_raw.csv` | SCWGI-T (transductive) |
| `competitors/<Method>_raw.csv` | HyperImpute-library baselines (Mean = `Ortalama`, EM, MICE, MissForest, GAIN, SoftImpute, MIWAE) |
| `competitors/HyperImpute_trainplus_*_raw.csv` | HyperImpute (fitted on training rows + masked test rows), two parts |
| `competitors/deep_raw.csv` | TabDDPM and SAINT (trained on training rows only) |
| `competitors/Sinkhorn_raw.csv` | Sinkhorn — incomplete (4 of 75 runs), excluded from every summary |
| `ablation/<variant>_raw.csv` | component ablation (`-kopula`, `-atom`, `-ort`, `-küme`, `-büzülme`, `FCM-ağırlık`) |
| `calibration_raw.csv` | 90 % interval coverage and width: raw Gaussian, split conformal, Mondrian; per s-tertile coverage; bias |
| `summary/` | tables produced by `experiments/make_results.py` |
| `figures/` | figures produced by `experiments/make_figures.py` and `experiments/make_figure1.py` |
