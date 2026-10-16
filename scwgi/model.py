"""SCWGI: atom göstergeli giriş + veri uyarlamalı marjinal + yumuşak küme ağırlıklı (çok çözünürlüklü) koşullu Gauss kolu.

Gradyan tabanlı eğitim yoktur; tüm ayarlar eğitim kümesi içindeki maskeli çapraz doğrulamayla seçilir.
Bu sınıf, makaledeki sonuçları üreten araştırma kodundaki Gauss-çekirdeği yapılandırmasının
sadeleştirilmiş eşdeğeridir; tests/test_regression.py iki uygulamanın aynı çıktıyı verdiğini doğrular.
"""
from __future__ import annotations
import numpy as np
import torch
from ._core import T, mix
from .gauss import select_and_build
from .copula import EmpiricalCopula, CopulaGauss


class LinearGauss:
    """Doğrusal (z-ölçek) marjinalde Gauss kolu."""
    def __init__(self, gp):
        self.gp = gp; self.K = gp.K

    @torch.no_grad()
    def inputs(self, X, M, Zfull=None):
        tx, tm = T(X * M), T(M)
        ce, vk, logr = self.gp.components(tx, tm); P, LV, R = mix(tx, tm, ce, vk, logr)
        return dict(tz=tx, tm=tm, tx=tx, Pz=P, LVz=LV, R=R, Px=P, LVx=LV)


class SCWGI:
    """Yumuşak Küme Ağırlıklı Gauss Doldurucu (Soft-Cluster Weighted Gaussian Imputer).

    marginal : "auto" (iç CV ile doğrusal/kopula seçimi), "linear" ya da "copula"
    ks       : çok çözünürlüklü model ortalamasındaki küme sayıları (K > 1 yalnız 0.8 n ≥ 10 K ise)
    tie      : {ek sütun: asıl sütun} — ek sütunların (atom göstergeleri) maskesi asıl sütunun maskesine bağlanır
    n_eval   : doğrulama hatası yalnız ilk n_eval sütunda ölçülür (atom göstergeleri hariç)
    """

    def __init__(self, marginal="auto", ks=(1, 2, 3, 4, 6, 8), fuzzifier=1.5, random_state=42, tie=None, n_eval=None):
        self.marginal, self.ks, self.fuzzifier, self.random_state, self.tie, self.n_eval = marginal, ks, fuzzifier, random_state, tie, n_eval

    def _hide(self, g, shape, rate):
        h = (g.random(shape) < rate).astype(float)
        for a_, o_ in (self.tie or {}).items(): h[:, a_] = h[:, o_]
        return h

    def _build(self, rows, X, marg):
        if marg == "copula":
            gp, info = select_and_build(self.Zfit[rows], self.random_state, self.fuzzifier, "em", self.ks_eff,
                                        ridge_cv=True, min_per_cluster=0, tie=self.tie, n_eval=self.n_eval)
            return CopulaGauss(self.cop, gp), info
        gp, info = select_and_build(X[rows], self.random_state, self.fuzzifier, "em", self.ks_eff,
                                    ridge_cv=True, min_per_cluster=0, tie=self.tie, n_eval=self.n_eval)
        return LinearGauss(gp), info

    def _choose_marginal(self, X):
        """Eğitim içi maskeli CV (3 tekrar, %20 satır, %20 hücre): doğrusal vs kopula, özgün uzayda MSE."""
        rng = np.random.default_rng(self.random_state + 7); n, d = X.shape; err = {"linear": [], "copula": []}
        for _ in range(3):
            idx = rng.permutation(n); nv = max(2, int(0.2 * n)); va, tr = idx[:nv], idx[nv:]
            hide = self._hide(rng, (nv, d), 0.2); mv = 1 - hide; ne = self.n_eval or d
            for marg in err:
                br, _ = self._build(tr, X, marg)
                P = br.inputs(X[va] * mv, mv, self.Zobs[va] if marg == "copula" else None)["Px"].cpu().numpy()
                err[marg].append(((P - X[va])[:, :ne] ** 2 * hide[:, :ne]).sum() / max(hide[:, :ne].sum(), 1))
        self.marg_cv = {k: float(np.mean(v)) for k, v in err.items()}
        return min(self.marg_cv, key=self.marg_cv.get)

    def fit(self, X):
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        X = np.asarray(X, float); n, d = X.shape
        self.cop = EmpiricalCopula().fit(X)
        self.Zfit = self.cop.latent_train(X, self.random_state); self.Zobs = self.cop.to_latent(X, np.ones_like(X))
        self.ks_eff = tuple(k for k in self.ks if k == 1 or int(0.8 * n) >= 10 * k)
        self.marg = self._choose_marginal(X) if self.marginal == "auto" else self.marginal
        self.br, self.info = self._build(np.arange(n), X, self.marg)
        self.k_star = max(self.info, key=lambda k: self.info[k]["w"])
        return self

    def transform_dist(self, X, M):
        """Dönüş: (koşullu ortalama x̂, koşullu standart sapma s) — özgün (z) ölçekte.
        Gözlenen hücrelerde x̂ = gözlenen değer ve s = 0."""
        X = np.asarray(X, float).copy(); M = np.asarray(M, float); X[M == 0] = 0.0
        mu, sd = [], []
        for s in range(0, len(X), 256):
            I = self.br.inputs(X[s:s + 256], M[s:s + 256], None)
            mu.append(I["Px"]); sd.append(torch.exp(0.5 * I["LVx"]))
        cat = lambda L: torch.cat(L).cpu().numpy()
        return np.where(M == 1, X, cat(mu)), np.where(M == 1, 0.0, cat(sd))

    def transform(self, X, M):
        return self.transform_dist(X, M)[0]

    def fit_transform(self, Xtr, Xm, M):
        return self.fit(Xtr).transform(Xm, M)


class AugWrap:
    """Atom göstergeli genişletme: en az `min_unique` farklı değer alan ve en sık değerinin göreli frekansı en az
    `atom_min_frac` olan her sütuna bir gösterge sütunu ekler; temel modeli genişletilmiş veride çalıştırır."""
    def __init__(self, base_cls, atom_min_frac=0.05, min_unique=11, random_state=42, **kw):
        self.base_cls, self.amf, self.mu_, self.random_state, self.kw = base_cls, atom_min_frac, min_unique, random_state, kw

    def _aug(self, X, M):
        I = np.stack([np.isclose(X[:, j], a) for j, a in self.atoms], 1).astype(float) if self.atoms else np.zeros((len(X), 0))
        I = (I - self.im) / self.isd if self.atoms else I
        Ma = np.concatenate([M, M[:, [j for j, _ in self.atoms]]], 1)
        return np.concatenate([X, I], 1) * Ma, Ma

    def fit(self, X):
        X = np.asarray(X, float); n, d = X.shape; self.d = d; self.atoms = []
        for j in range(d):
            u, c = np.unique(np.round(X[:, j], 8), return_counts=True)
            if len(u) >= self.mu_ and c.max() / n >= self.amf: self.atoms.append((j, u[c.argmax()]))
        if self.atoms:
            I = np.stack([np.isclose(X[:, j], a) for j, a in self.atoms], 1).astype(float)
            self.im, self.isd = I.mean(0), I.std(0) + 1e-12
        Xa, _ = self._aug(X, np.ones_like(X))
        tie = {d + i: j for i, (j, _) in enumerate(self.atoms)} or None
        self.m = self.base_cls(random_state=self.random_state, tie=tie, n_eval=d if self.atoms else None, **self.kw).fit(Xa)
        self.k_star, self.marg = self.m.k_star, self.m.marg
        return self

    def transform_dist(self, X, M):
        """(x̂, s) — yalnız asıl sütunlar."""
        X = np.asarray(X, float).copy(); M = np.asarray(M, float); X[M == 0] = 0.0
        Xa, Ma = self._aug(X, M)
        mu, sd = self.m.transform_dist(Xa, Ma)
        return np.where(M == 1, X, mu[:, :self.d]), np.where(M == 1, 0.0, sd[:, :self.d])

    def transform(self, X, M):
        return self.transform_dist(X, M)[0]

    def fit_transform(self, Xtr, Xm, M):
        return self.fit(Xtr).transform(Xm, M)
