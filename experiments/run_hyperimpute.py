"""
run_hyperimpute_trainplus.py — HyperImpute'un eğitim kümesini de kullanarak yeniden koşulması (ortak protokol).

Önceki koşuda (main.py → models.HyperImputer.fit_transform_with_mask) HyperImpute yalnız maskeli TEST matrisinde
uyduruluyordu; eğitim kümesi kullanılmıyordu. HyperImpute kütüphanesi ayrı fit/transform desteklemez (plugin._transform
= model.fit_transform(X)); bu nedenle eğitim satırları (tam gözlenmiş) ile maskeli test satırları ALT ALTA birleştirilip
tek matris olarak uydurulur ve yalnız test satırlarındaki gizli hücreler değerlendirilir. Böylece HyperImpute hem eğitim
verisinden hem de (yöntemin doğası gereği) test satırlarının gözlenen hücrelerinden yararlanır.

Protokol diğer modellerle birebir aynı: get_train_test(0.70, seed=0) (z-ölçek) → introduce_missing(X_test, oran,
mekanizma, tohum), tohumlar 42, 142, 242, 342, 442 → compute_metrics (z-ölçek, gizli hücreler, havuzlanmış).
Varsayılan "hyperimpute" eklentisi; model tohumu = maske tohumu.
Çıktı: results/competitors/HyperImpute_trainplus_raw.csv
"""
import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from data import EXPERIMENT_CONFIG, load_dataset
from scwgi.masking import get_train_test, introduce_missing, compute_metrics

SEEDS = [42, 142, 242, 342, 442]


def impute(Xtr, Xte, M, seed):
    from hyperimpute.plugins.imputers import Imputers
    Xq = Xte.astype(float).copy(); Xq[M == 0] = np.nan
    full = pd.DataFrame(np.vstack([Xtr, Xq]))
    out = Imputers().get("hyperimpute", random_state=seed).fit_transform(full)
    out = np.asarray(out, float)[len(Xtr):]
    return np.where(M == 1, Xte, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(EXPERIMENT_CONFIG))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    od = Path("results/competitors"); od.mkdir(parents=True, exist_ok=True)
    fn = od / f"HyperImpute_trainplus_{a.tag}_raw.csv"
    rows = pd.read_csv(fn).to_dict("records") if fn.exists() else []
    done = {(r["Dataset"], r["Mechanism"], r["Rate"], int(r["Seed"])) for r in rows}
    for ds in a.datasets:
        X = load_dataset(ds); Xtr, Xte, _ = get_train_test(X, 0.70, seed=0)
        for s in a.seeds:
            for mech in EXPERIMENT_CONFIG[ds]["mechanisms"]:
                for rate in EXPERIMENT_CONFIG[ds]["rates"]:
                    if (ds, mech, f"%{int(round(rate * 100))}", s) in done: continue
                    _, M = introduce_missing(Xte, rate, mechanism=mech, seed=s)
                    t0 = time.time(); Xi = impute(Xtr, Xte, M, s)
                    rows.append({"Dataset": ds, "Model": "HyperImpute", "Protocol": "train+test", "Mechanism": mech,
                                 "Rate": f"%{int(round(rate * 100))}", "Seed": s, **compute_metrics(Xte, Xi, M),
                                 "Time_sec": round(time.time() - t0, 1)})
                    print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(fn, index=False)
