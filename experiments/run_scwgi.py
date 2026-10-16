"""SCWGI (endüktif) ve SCWGI-T (transdüktif) — ortak protokol.
Protokol: rastgele %70/%30 bölme (tohum 0), eğitim istatistikleriyle z-ölçekleme; eksiklik yalnız test kümesine,
tohumlar 42, 142, 242, 342, 442; metrikler gizli hücrelerde, z-ölçekte, havuzlanmış.
Kullanım: python experiments/run_scwgi.py --mode inductive|transductive [--datasets ...]
Çıktı: results/scwgi/SCWGI_raw.csv ya da results/scwgi/SCWGI-T_raw.csv (yeniden başlatılabilir)"""
import argparse, sys, time, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXPERIMENT_CONFIG, load_dataset
from scwgi import SCWGIImputer, SCWGITImputer
from scwgi.masking import get_train_test, introduce_missing, compute_metrics

SEEDS = [42, 142, 242, 342, 442]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["inductive", "transductive"], default="inductive")
ap.add_argument("--datasets", nargs="+", default=list(EXPERIMENT_CONFIG))
a = ap.parse_args()
make, name = (SCWGIImputer, "SCWGI") if a.mode == "inductive" else (SCWGITImputer, "SCWGI-T")
od = Path("results/scwgi"); od.mkdir(parents=True, exist_ok=True); fn = od / f"{name}_raw.csv"
rows = pd.read_csv(fn).to_dict("records") if fn.exists() else []
done = {(r["Dataset"], int(r["Seed"])) for r in rows}
for ds in a.datasets:
    Xtr, Xte, _ = get_train_test(load_dataset(ds), 0.70, seed=0)
    for s in SEEDS:
        if (ds, s) in done: continue
        t0 = time.time(); m = make(random_state=s).fit(Xtr); tf = time.time() - t0
        for mech in EXPERIMENT_CONFIG[ds]["mechanisms"]:
            for rate in EXPERIMENT_CONFIG[ds]["rates"]:
                Xm, M = introduce_missing(Xte, rate, mechanism=mech, seed=s); t1 = time.time()
                rows.append({"Dataset": ds, "Model": name, "Mechanism": mech, "Rate": f"%{int(round(rate * 100))}", "Seed": s,
                             **compute_metrics(Xte, m.transform(Xm, M), M), "Time_sec": round(tf + time.time() - t1, 1),
                             "marginal": m.marg, "k_star": m.k_star})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(fn, index=False)
