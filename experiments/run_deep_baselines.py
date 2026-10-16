"""TabDDPM ve SAINT — ortak protokol (yazarların karşılaştırma koşusu; yalnız eğitim kümesiyle eğitilir).
Model sarmalayıcıları: deep_baselines/models.py; yapılandırmalar özgün koşu betiğiyle (main.py) aynıdır.
Çıktı: results/competitors/deep_raw.csv"""
import sys, time, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "deep_baselines"))
from data import EXPERIMENT_CONFIG, load_dataset
from scwgi.masking import get_train_test, introduce_missing, compute_metrics
from models import TabDDPMImputer, SAINTImputer

SMALL_DATASET_THRESHOLD = 200
MODEL_CONFIGS = {
    "HyperImpute": {
        "class":  None,
        "params": {"optimizer": "hyperimpute"},
    },
    "TabDDPM": {
        "class":  TabDDPMImputer,
        "params_default": {
            "T":          500,
            "hidden":     256,
            "n_layers":   4,
            "epochs":     300,
            "batch_size": 256,
            "lr":         1e-3,
            "n_samples":  5,
        },
        "params_small": {   # n < SMALL_DATASET_THRESHOLD
            "T":          300,
            "hidden":     128,
            "n_layers":   3,
            "epochs":     200,
            "batch_size": 64,
            "lr":         1e-3,
            "n_samples":  5,
        },
    },
    "SAINT": {
        "class":  SAINTImputer,
        "params_default": {
            "dim":        64,
            "depth":      3,
            "heads":      4,
            "epochs":     300,
            "batch_size": 128,
            "lr":         3e-4,
        },
        "params_small": {
            "dim":        32,
            "depth":      2,
            "heads":      4,
            "epochs":     200,
            "batch_size": 32,
            "lr":         3e-4,
        },
    },
}


def _get_model_params(model_name: str, n_train: int) -> dict:
    cfg = MODEL_CONFIGS[model_name]
    if "params" in cfg:
        return cfg["params"]
    key = "params_small" if n_train < SMALL_DATASET_THRESHOLD else "params_default"
    return cfg[key]


rows = []
for ds in EXPERIMENT_CONFIG:
    Xtr, Xte, _ = get_train_test(load_dataset(ds), 0.70, seed=0)
    for name, cls in (("TabDDPM", TabDDPMImputer), ("SAINT", SAINTImputer)):
        for s in [42, 142, 242, 342, 442]:
            for mech in EXPERIMENT_CONFIG[ds]["mechanisms"]:
                for rate in EXPERIMENT_CONFIG[ds]["rates"]:
                    Xm, M = introduce_missing(Xte, rate, mechanism=mech, seed=s)
                    p = _get_model_params(name, len(Xtr)).copy(); p["random_state"] = s
                    t0 = time.time(); Xi = cls(**p).fit_transform(Xtr, Xm, M, verbose=False)
                    rows.append({"Dataset": ds, "Model": name, "Mechanism": mech, "Rate": f"%{int(rate * 100)}", "Seed": s,
                                 **compute_metrics(Xte, Xi, M), "Time_sec": round(time.time() - t0, 1)})
                    print(rows[-1], flush=True)
Path("results/competitors").mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv("results/competitors/deep_raw.csv", index=False)
