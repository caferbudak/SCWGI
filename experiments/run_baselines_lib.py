"""
run_baselines_lib.py — Klasik ve derin doldurma yöntemlerinin, HyperImpute kütüphanesindeki (v0.1.17) referans
uygulamalarıyla ORTAK protokolde koşulması.

Yöntemler (kütüphane eklenti adı → makaledeki ad):
    mean → Ortalama, EM → EM (çok değişkenli normal, beklenti-maksimizasyonu), mice → MICE, missforest → MissForest,
    gain → GAIN, softimpute → SoftImpute, sinkhorn → Sinkhorn (optimal taşıma), miwae → MIWAE
Varsayılan hiperparametreler kullanılır; model tohumu = maske tohumu.

Protokol (diğer tüm modellerle aynı): get_train_test(0.70, seed=0) (z-ölçek) → introduce_missing(X_test, oran, mekanizma,
tohum), tohumlar 42, 142, 242, 342, 442 → compute_metrics (z-ölçek, gizli hücreler, havuzlanmış).
Bu eklentilerin bir kısmı (SoftImpute, Sinkhorn) yalnız transdüktif çalışır; tutarlılık için TÜM kütüphane yöntemleri
HyperImpute ile aynı biçimde, tam gözlenmiş eğitim satırları + maskeli test satırları ALT ALTA birleştirilerek uydurulur;
yalnız test satırlarının gizli hücreleri değerlendirilir. (SCWGI, TabDDPM ve SAINT ise yalnız eğitim kümesiyle eğitilir.)
Çıktı: results/competitors/<Yöntem>_<etiket>_raw.csv
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
NAMES = {"mean": "Ortalama", "EM": "EM", "mice": "MICE", "missforest": "MissForest", "gain": "GAIN",
         "softimpute": "SoftImpute", "sinkhorn": "Sinkhorn", "miwae": "MIWAE"}


def impute(plugin_name, Xtr, Xte, M, seed):
    import torch
    from hyperimpute.plugins.imputers import Imputers
    np.random.seed(seed); torch.manual_seed(seed)
    Xq = Xte.astype(float).copy(); Xq[M == 0] = np.nan
    full = pd.DataFrame(np.vstack([Xtr, Xq]))
    try:
        plug = Imputers().get(plugin_name, random_state=seed)
    except TypeError:
        plug = Imputers().get(plugin_name)
    out = np.asarray(plug.fit_transform(full), float)[len(Xtr):]
    return np.where(M == 1, Xte, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(NAMES))
    ap.add_argument("--datasets", nargs="+", default=list(EXPERIMENT_CONFIG))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--tag", default="all")
    a = ap.parse_args()
    od = Path("results/competitors"); od.mkdir(parents=True, exist_ok=True)
    data = {ds: get_train_test(load_dataset(ds), 0.70, seed=0)[:2] for ds in a.datasets}
    for meth in a.methods:
        fn = od / f"{NAMES[meth]}_{a.tag}_raw.csv"
        rows = pd.read_csv(fn).to_dict("records") if fn.exists() else []
        done = {(r["Dataset"], r["Mechanism"], r["Rate"], int(r["Seed"])) for r in rows}
        for ds in a.datasets:
            Xtr, Xte = data[ds]
            for s in a.seeds:
                for mech in EXPERIMENT_CONFIG[ds]["mechanisms"]:
                    for rate in EXPERIMENT_CONFIG[ds]["rates"]:
                        if (ds, mech, f"%{int(round(rate * 100))}", s) in done: continue
                        _, M = introduce_missing(Xte, rate, mechanism=mech, seed=s)
                        t0 = time.time(); Xi = impute(meth, Xtr, Xte, M, s)
                        rows.append({"Dataset": ds, "Model": NAMES[meth], "Protocol": "train+test", "Mechanism": mech,
                                     "Rate": f"%{int(round(rate * 100))}", "Seed": s, **compute_metrics(Xte, Xi, M),
                                     "Time_sec": round(time.time() - t0, 1)})
                        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(fn, index=False)
