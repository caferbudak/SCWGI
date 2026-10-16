"""Konformal belirsizlik (makale Bölüm 2.6): model eğitim satırlarının %85'inde uydurulur, %15'i kalibrasyon kümesidir;
5 MCAR + 5 MAR kalibrasyon maskesi; test maskelerinde %90 kapsama ve genişlik (ham Gauss, bölünmüş, Mondrian),
ayrıca s'nin kalibrasyon üçte-birliklerine göre grup kapsamaları. Çıktı: results/calibration_raw.csv"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXPERIMENT_CONFIG, load_dataset
from scwgi import SCWGIImputer, ConformalCalibrator
from scwgi.masking import get_train_test, introduce_missing

Z90 = 1.6449; rows = []; GROUPS = ("düşük", "orta", "yüksek")
for ds in EXPERIMENT_CONFIG:
    Xtr, Xte, _ = get_train_test(load_dataset(ds), 0.70, seed=0)
    for s in [42, 142, 242, 342, 442]:
        idx = np.random.default_rng(s + 500).permutation(len(Xtr)); nc = int(round(0.15 * len(Xtr)))
        imp = SCWGIImputer(random_state=s).fit(Xtr[idx[nc:]])
        cal = ConformalCalibrator(alpha=0.10).fit(imp, Xtr[idx[:nc]], float(np.mean(EXPERIMENT_CONFIG[ds]["rates"])), seed=s)
        for mech in EXPERIMENT_CONFIG[ds]["mechanisms"]:
            for rate in EXPERIMENT_CONFIG[ds]["rates"]:
                Xm, M = introduce_missing(Xte, rate, mechanism=mech, seed=s); h = M == 0
                mu, sd = imp.transform_dist(Xm, M); y, yh, ys = Xte[h], mu[h], sd[h]; err = np.abs(y - yh); g = cal.group(ys)
                r = {"Dataset": ds, "Mechanism": mech, "Rate": f"%{int(round(rate * 100))}", "Seed": s, "n_cells": int(h.sum()),
                     "bias": float(np.mean(yh - y)), "cov90_raw": float(np.mean(err <= Z90 * ys)), "width90_raw": float(np.mean(2 * Z90 * ys))}
                for key, q in (("conf", np.full_like(ys, cal.q_)), ("mond", cal.qm_[g])):
                    r[f"cov90_{key}"] = float(np.mean(err <= q * ys)); r[f"width90_{key}"] = float(np.mean(2 * q * ys))
                    for gi, gn in enumerate(GROUPS):
                        sel = g == gi; r[f"cov90_{key}_{gn}"] = float(np.mean(err[sel] <= q[sel] * ys[sel])) if sel.any() else np.nan
                rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        Path("results").mkdir(exist_ok=True); pd.DataFrame(rows).to_csv("results/calibration_raw.csv", index=False)
