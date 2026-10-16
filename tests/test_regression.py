"""Paketlenmiş SCWGI'nin, makaledeki sonuçları üreten araştırma koduyla aynı çıktıyı verdiğini doğrular.
Referans çıktılar (tests/reference/*.npz) araştırma kodundan aynı tohumlarla üretilmiştir; doğrusal ve kopula
marjinali, atom göstergeleri (Diyabet) ve transdüktif kip (SCWGI-T) ayrı ayrı sınanır.
Çalıştırma: python -m pytest tests/  (ya da python tests/test_regression.py)"""
from pathlib import Path
import numpy as np
from scwgi import SCWGIImputer, SCWGITImputer

REF = Path(__file__).resolve().parent / "reference"
CASES = [("fertility_mcar10_seed42.npz", SCWGIImputer, 42), ("diabetes_mar10_seed42.npz", SCWGIImputer, 42),
         ("breastcancer_mcar10_seed42.npz", SCWGIImputer, 42),
         ("fertility_mcar20_seed142_transductive.npz", SCWGITImputer, 142)]


def _check(fname, make, seed, tol=1e-6):
    z = np.load(REF / fname)
    m = make(random_state=seed).fit(z["Xtr"])
    assert m.marg == str(z["marg"]), (fname, m.marg, str(z["marg"]))
    mu, sd = m.transform_dist(z["Xm"], z["M"])
    for name, a in (("mu", mu), ("sd", sd)):
        assert np.abs(a - z[name]).max() <= tol, (fname, name, float(np.abs(a - z[name]).max()))


def test_fertility(): _check(*CASES[0])
def test_diabetes_atoms(): _check(*CASES[1])
def test_breastcancer_copula(): _check(*CASES[2])
def test_transductive(): _check(*CASES[3])


if __name__ == "__main__":
    for c in CASES:
        _check(*c); print("OK", c[0])
