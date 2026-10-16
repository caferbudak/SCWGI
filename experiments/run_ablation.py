"""SCWGI bileşen ablasyonu (makale Bölüm 2.8.4 ve 3.5): her varyant tek bir bileşeni değiştirir.
Marjinal, her (veri seti, tohum) için SCWGI'nin seçtiği değere sabitlenir (results/scwgi/SCWGI_raw.csv).
Varyantlar: -kopula (doğrusal marjinal), -atom, -ort (tek K), -küme (K = 1), -büzülme (λ = 0), FCM-ağırlık.
Çıktı: results/ablation/<varyant>_raw.csv"""
import argparse, functools, sys, time, warnings
from pathlib import Path
import pandas as pd, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXPERIMENT_CONFIG, load_dataset
import scwgi.model as SM, scwgi._core as SC
from scwgi import SCWGI, AugWrap
from scwgi.masking import get_train_test, introduce_missing, compute_metrics

_sab, _comp = SM.select_and_build, SC.GaussPrior._comp


def fcm_comp(self, x, m):
    ce, var_k, _ = _comp(self, x, m)
    d2 = ((x[:, None, :] - self.mu[None]) ** 2).sum(-1).clamp_min(1e-12)        # eksikler z-ortalama (0) ile dolu
    return ce, var_k, torch.log_softmax(-(1.0 / (1.5 - 1.0)) * torch.log(d2), -1)  # FCM üyeliği, m = 1.5


def make(v, s, marg):
    SM.select_and_build, SC.GaussPrior._comp = _sab, _comp
    if v == "-atom": return SCWGI(random_state=s, marginal=marg)
    if v == "-kopula": return AugWrap(SCWGI, random_state=s, marginal="linear")
    if v == "-ort": SM.select_and_build = functools.partial(_sab, tau=1e-9)
    if v == "-büzülme": SM.select_and_build = functools.partial(_sab, lams=(0.0,))
    if v == "FCM-ağırlık": SC.GaussPrior._comp = fcm_comp
    return AugWrap(SCWGI, random_state=s, marginal=marg, **({"ks": (1,)} if v == "-küme" else {}))


ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=["-kopula", "-atom", "-ort", "-küme", "-büzülme", "FCM-ağırlık"])
a = ap.parse_args()
MARG = pd.read_csv("results/scwgi/SCWGI_raw.csv").groupby(["Dataset", "Seed"]).marginal.first().to_dict()
od = Path("results/ablation"); od.mkdir(parents=True, exist_ok=True)
for v in a.variants:
    rows = []
    for ds in EXPERIMENT_CONFIG:
        Xtr, Xte, _ = get_train_test(load_dataset(ds), 0.70, seed=0)
        for s in [42, 142, 242, 342, 442]:
            t0 = time.time(); m = make(v, s, MARG[(ds, s)]).fit(Xtr); tf = time.time() - t0
            for mech in EXPERIMENT_CONFIG[ds]["mechanisms"]:
                for rate in EXPERIMENT_CONFIG[ds]["rates"]:
                    Xm, M = introduce_missing(Xte, rate, mechanism=mech, seed=s)
                    rows.append({"Dataset": ds, "Model": v, "Mechanism": mech, "Rate": f"%{int(round(rate * 100))}", "Seed": s,
                                 **compute_metrics(Xte, m.transform(Xm, M), M), "Time_sec": round(tf, 1)})
            print(rows[-1], flush=True)
    SM.select_and_build, SC.GaussPrior._comp = _sab, _comp
    pd.DataFrame(rows).to_csv(od / f"{v}_raw.csv", index=False)
