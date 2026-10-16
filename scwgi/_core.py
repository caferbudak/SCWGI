"""Ortak yardımcılar: cihaz, tensör dönüşümü, FCM, küme-koşullu Gauss bileşenleri ve karışım."""
from __future__ import annotations
import math
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fcm_fit(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 300,
            tol: float = 1e-5, seed: int = 42):
    """Standart FCM (Bezdek). Döndürür: V (c×d) merkezler, U (n×c) üyelikler (satır toplamı 1)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, c))
    U /= U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        Um = U ** m
        V = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)
        U_new = fcm_memberships(X, V, m)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    return V, U


def fcm_memberships(X: np.ndarray, V: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Dondurulmuş merkezlerden üyelik: u_ik = 1 / Σ_j (d_ik / d_ij)^{2/(m-1)}."""
    d2 = ((X[:, None, :] - V[None, :, :]) ** 2).sum(-1) + 1e-12      # (n, c)
    p = 1.0 / (m - 1.0)
    inv = d2 ** (-p)
    return inv / inv.sum(axis=1, keepdims=True)


T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


class GaussPrior:
    def __init__(self, k, lam, fuzzifier=1.5, seed=0, eps=1e-3, method="fcm"):
        self.k, self.lam, self.m, self.seed, self.eps, self.method = k, lam, fuzzifier, seed, eps, method

    def fit(self, X):
        n, d = X.shape
        Sg = LedoitWolf().fit(X).covariance_
        if self.k == 1:
            mu = X.mean(0, keepdims=True); S = Sg[None]; pi = np.ones(1)
        else:
            V, U = fcm_fit(X, self.k, m=self.m, seed=self.seed)          # (n,k)
            if self.method == "em":   # FCM ile başlatılan EM (olabilirlik-optimal yumuşak kümeler)
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(self.k, covariance_type="full", reg_covar=1e-3, means_init=V, max_iter=200,
                                     random_state=self.seed).fit(X)
                mu, S, pi = gm.means_, gm.covariances_, gm.weights_
            else:
                w = U / U.sum(0, keepdims=True)
                mu = w.T @ X                                              # (k,d)
                S = np.stack([((X - mu[c]) * w[:, c:c + 1]).T @ (X - mu[c]) for c in range(self.k)])
                pi = U.mean(0)
            S = (1 - self.lam) * S + self.lam * Sg[None]
        S = S + self.eps * np.eye(d)[None]
        self.mu, self.S, self.logpi = T(mu), T(S), T(np.log(pi + 1e-12))
        P = torch.linalg.inv(self.S)
        dg = torch.sqrt(torch.diagonal(P, dim1=-2, dim2=-1))
        rho = -P / (dg[:, :, None] * dg[:, None, :])
        self.rho = rho * (1 - torch.eye(d, device=DEVICE))[None]           # (k,d,d) kısmi korelasyon, köşegen 0
        return self

    @torch.no_grad()
    def components(self, x, m, chunk=256):
        """Bileşen düzeyi: ce_k (B,K,d), var_k (B,K,d), logr (B,K) (model içinde normalize log-sorumluluk)."""
        outs = [self._comp(x[s:s + chunk], m[s:s + chunk]) for s in range(0, len(x), chunk)]
        return tuple(torch.cat(t) for t in zip(*outs))

    def _comp(self, x, m):
        mk = m[:, None, :]                                                # (B,1,d)
        r = (x[:, None, :] - self.mu[None]) * mk                          # (B,K,d)
        mo = mk[..., :, None] * mk[..., None, :]                          # (B,1,d,d)
        St = self.S[None] * mo + torch.diag_embed(1 - m)[:, None]          # (B,K,d,d)  Σ_oo ⊕ I_mm
        L = torch.linalg.cholesky(St)
        y = torch.cholesky_solve(r[..., None], L)[..., 0]                # Σ_oo⁻¹ r_o (eksiklerde 0)
        ce = self.mu[None] + (self.S[None] @ y[..., None])[..., 0]        # (B,K,d)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        logp = self.logpi[None] - 0.5 * ((r * y).sum(-1) + logdet)
        SDo = self.S[None] * mk[..., None, :]                             # Σ D_o
        Z = torch.cholesky_solve(SDo.transpose(-1, -2), L)                # Σ̃⁻¹ D_o Σ
        var_k = torch.diagonal(self.S[None] - SDo @ Z, dim1=-2, dim2=-1).clamp_min(1e-6)
        return ce, var_k, torch.log_softmax(logp, -1)

    @torch.no_grad()
    def __call__(self, x, m):
        """Döndürür: karışım ortalaması (B,d), log-varyans (B,d), ağırlıklar R (B,K)."""
        ce, var_k, logr = self.components(x, m)
        return mix(x, m, ce, var_k, logr)


def mix(x, m, ce, var_k, logr):
    R = logr.softmax(-1)
    mean = (R[..., None] * ce).sum(1)
    var = (R[..., None] * (var_k + ce ** 2)).sum(1) - mean ** 2
    mean = torch.where(m > 0, x, mean)
    var = torch.where(m > 0, torch.full_like(var, 1e-6), var.clamp_min(1e-6))
    return mean, torch.log(var), R


class MultiPrior:
    """Çok çözünürlüklü önsel: farklı k (ve λ) modellerinin CV-skoruyla ağırlıklı ortalaması (BMA benzeri).
    Eşdeğer olarak tüm bileşenlerin tek bir büyük karışımı: log w = log w_model + log r_(k|model)."""
    def __init__(self, models, logw):
        self.models, self.logw = models, logw
        self.rho = torch.cat([g.rho for g in models])
        self.K = int(sum(g.mu.shape[0] for g in models))

    @torch.no_grad()
    def components(self, x, m):
        parts = [g.components(x, m) for g in self.models]
        ce = torch.cat([p[0] for p in parts], 1); vk = torch.cat([p[1] for p in parts], 1)
        logr = torch.cat([p[2] + lw for p, lw in zip(parts, self.logw)], 1)
        return ce, vk, logr

    @torch.no_grad()
    def __call__(self, x, m):
        return mix(x, m, *self.components(x, m))
