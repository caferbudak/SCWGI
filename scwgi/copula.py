"""Ampirik Gauss kopulası (Gauss–Hermite geri dönüşümlü) ve kopula uzayındaki Gauss kolu."""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.stats import norm
from ._core import DEVICE, T, mix

_T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


def torch_interp(x, xp, fp):
    """Doğrusal enterpolasyon (xp artan, uçlarda kırpılır). x: (...), xp/fp: (n,)."""
    x = x.clamp(xp[0], xp[-1])
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    w = torch.where(x1 > x0, (x - x0) / (x1 - x0), torch.zeros_like(x))
    return y0 + w * (y1 - y0)


class EmpiricalCopula:
    def __init__(self, n_quad=32, atom_min_frac=0.05, min_unique=11):
        self.n_quad, self.atom_min_frac, self.min_unique = n_quad, atom_min_frac, min_unique

    def fit(self, X):
        X = np.asarray(X, float); n, d = X.shape; self.n, self.d = n, d
        self.xs = [np.sort(X[:, j]) for j in range(d)]
        self.lv = (np.arange(n) + 0.5) / n                                   # kantil seviyeleri
        # benzersiz değer → orta-kantil (bağlar için)
        self.uval, self.umid, self.atom = [], [], []
        for j in range(d):
            u, cnt = np.unique(self.xs[j], return_counts=True)
            cum = np.cumsum(cnt); start = cum - cnt
            self.uval.append(u); self.umid.append((start + cum) / 2 / n)
            k = cnt.argmax(); self.atom.append(u[k] if cnt[k] / n >= self.atom_min_frac else np.nan)
        # ayrık özellikler (az sayıda benzersiz değer): kopula yerine doğrusal (standart) marjinal
        self.lin = np.array([len(u) < self.min_unique for u in self.uval])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.t_lin, self.t_mu, self.t_sd = torch.as_tensor(self.lin, device=DEVICE), _T(self.mu), _T(self.sd)
        # doğrusal kuyruklar: (z, x) düzleminde uçtaki %10'luk dilimin eğimi → aralık dışı ileri/geri dönüşüm
        self.zk = norm.ppf(self.lv); k = max(2, int(0.1 * n))
        self.blo = np.array([max((xs[k] - xs[0]) / (self.zk[k] - self.zk[0]), 1e-6) for xs in self.xs])
        self.bhi = np.array([max((xs[-1] - xs[-k - 1]) / (self.zk[-1] - self.zk[-k - 1]), 1e-6) for xs in self.xs])
        self.t_zk, self.t_blo, self.t_bhi = _T(self.zk), _T(self.blo), _T(self.bhi)
        # torch tabloları (geri dönüşüm için)
        self.t_lv = _T(self.lv); self.t_xs = [_T(s) for s in self.xs]
        t, w = np.polynomial.hermite_e.hermegauss(self.n_quad)
        self.qt, self.qw = _T(t), _T(w / w.sum())
        return self

    def latent_train(self, X, seed=0):
        """Uydurma için: bağlar rastgele kırılmış sıralardan normal skorlar."""
        X = np.asarray(X, float); rng = np.random.default_rng(seed); n, d = X.shape; Z = np.empty_like(X)
        for j in range(d):
            r = np.lexsort((rng.random(n), X[:, j]))                         # değer, sonra rastgele
            u = np.empty(n); u[r] = (np.arange(n) + 0.5) / n
            Z[:, j] = norm.ppf(u) if not self.lin[j] else (X[:, j] - self.mu[j]) / self.sd[j]
        return Z

    def to_latent(self, X, M):
        """Çıkarım için: gözlenen değer → orta-kantil normal skoru (eksik hücreler 0)."""
        X = np.asarray(X, float); M = np.asarray(M, float); Z = np.zeros_like(X); lo, hi = 0.5 / self.n, 1 - 0.5 / self.n
        for j in range(self.d):
            if self.lin[j]:
                Z[:, j] = (X[:, j] - self.mu[j]) / self.sd[j]; continue
            x = X[:, j]; u = np.interp(x, self.uval[j], self.umid[j], left=lo, right=hi)
            z = norm.ppf(np.clip(u, lo, hi))
            z = np.where(x > self.xs[j][-1], self.zk[-1] + (x - self.xs[j][-1]) / self.bhi[j], z)
            z = np.where(x < self.xs[j][0], self.zk[0] - (self.xs[j][0] - x) / self.blo[j], z)
            Z[:, j] = z
        return np.where(M > 0, Z, 0.0)

    def atom_indicator(self, X, M):
        A = np.zeros_like(np.asarray(X, float))
        for j in range(self.d):
            if not np.isnan(self.atom[j]): A[:, j] = np.isclose(X[:, j], self.atom[j]) * M[:, j]
        return A

    @torch.no_grad()
    def decode(self, m, v):
        """m, v: (..., d) gizil ortalama/varyans → orijinal uzayda E[x] ve Var[x] (Gauss–Hermite)."""
        s = v.clamp_min(1e-8).sqrt()
        z = m[..., None] + s[..., None] * self.qt                            # (..., d, q)
        cols = []
        for j in range(self.d):
            zj = z[..., j, :]; xj = torch_interp(zj, self.t_zk, self.t_xs[j])
            xj = torch.where(zj > self.t_zk[-1], self.t_xs[j][-1] + (zj - self.t_zk[-1]) * self.t_bhi[j], xj)
            xj = torch.where(zj < self.t_zk[0], self.t_xs[j][0] - (self.t_zk[0] - zj) * self.t_blo[j], xj)
            cols.append(xj)
        x = torch.stack(cols, -2)
        Ex = (x * self.qw).sum(-1); Ex2 = (x ** 2 * self.qw).sum(-1)
        Vx = (Ex2 - Ex ** 2).clamp_min(1e-8)
        # doğrusal marjinaller: kapalı form
        Ex = torch.where(self.t_lin, self.t_mu + self.t_sd * m, Ex); Vx = torch.where(self.t_lin, self.t_sd ** 2 * v, Vx)
        return Ex, Vx


class CopulaGauss:
    def __init__(self, cop, gp):
        self.cop, self.gp = cop, gp; self.K = gp.K

    @torch.no_grad()
    def inputs(self, X, M, Zfull=None):
        """X: orijinal z-ölçek (eksikler 0), M maske. Dönüş: dict (torch)."""
        Z = (Zfull * M) if Zfull is not None else self.cop.to_latent(X, M)
        tz, tm, tx = T(Z), T(M), T(X * M)
        ce, vk, logr = self.gp.components(tz, tm)                           # gizil (B,K,d)
        Pz, LVz, R = mix(tz, tm, ce, vk, logr)
        Ex, Vx = self.cop.decode(ce, vk)                                     # bileşen başına orijinal uzay
        Rr = logr.softmax(-1)[..., None]
        Px = (Rr * Ex).sum(1); Vxx = (Rr * (Vx + Ex ** 2)).sum(1) - Px ** 2
        Px = torch.where(tm > 0, tx, Px); LVx = torch.log(torch.where(tm > 0, torch.full_like(Vxx, 1e-6), Vxx.clamp_min(1e-6)))
        A = T(self.cop.atom_indicator(X, M))
        return dict(tz=tz, tm=tm, tx=tx, Pz=Pz, LVz=LVz, R=R, Px=Px, LVx=LVx, A=A)
