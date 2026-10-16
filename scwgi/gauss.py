"""Yumuşak küme ağırlıklı koşullu Gauss kolu: küme momentleri, büzülme, maskeli CV ve çok çözünürlüklü karışım."""
from __future__ import annotations
import math
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from ._core import DEVICE, T, fcm_fit, GaussPrior, MultiPrior, mix

def cluster_moments(X, k, fuzzifier=1.5, seed=0, method="em"):
    Sg = LedoitWolf().fit(X).covariance_
    if k == 1:
        return X.mean(0, keepdims=True), Sg[None].copy(), np.ones(1), Sg
    if method == "kmeans":   # sert K-means bölümlemesi; küme içi örnek momentleri
        from sklearn.cluster import KMeans
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X); d = X.shape[1]
        mu = np.stack([X[lab == c].mean(0) for c in range(k)])
        S = np.stack([np.cov(X[lab == c].T, bias=True).reshape(d, d) if (lab == c).sum() > 1 else Sg for c in range(k)])
        return mu, S, np.bincount(lab, minlength=k) / len(X), Sg
    V, U = fcm_fit(X, k, m=fuzzifier, seed=seed)
    if method == "em":
        gm = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, means_init=V, max_iter=200, random_state=seed).fit(X)
        return gm.means_, gm.covariances_, gm.weights_, Sg
    w = U / U.sum(0, keepdims=True); mu = w.T @ X
    S = np.stack([((X - mu[c]) * w[:, c:c + 1]).T @ (X - mu[c]) for c in range(k)])
    return mu, S, U.mean(0), Sg


def prior_from_moments(mom, lam, eps):
    mu, S, pi, Sg = mom; d = mu.shape[1]
    S2 = ((1 - lam) * S + lam * Sg[None]) if len(pi) > 1 else S.copy()
    S2 = S2 + eps * np.eye(d)[None]
    gp = GaussPrior(len(pi), lam, eps=eps)
    gp.mu, gp.S, gp.logpi = T(mu), T(S2), T(np.log(pi + 1e-12))
    P = torch.linalg.inv(gp.S); dg = torch.sqrt(torch.diagonal(P, dim1=-2, dim2=-1))
    gp.rho = (-P / (dg[:, :, None] * dg[:, None, :])) * (1 - torch.eye(d, device=DEVICE))[None]
    return gp


def select_and_build(X, seed=0, fuzzifier=1.5, method="em", ks=(1, 2, 3, 4, 6, 8), lams=(0.0, 0.25, 0.5, 0.75),
                     epss=(1e-3, 1e-2, 1e-1, 3e-1), rate=0.2, reps=3, min_per_cluster=10, tau=0.02, ridge_cv=True,
                     tie=None, n_eval=None):
    """Her k için (λ, ε) maskeli CV ile; k'lar arası CV-ağırlıklı karışım (MultiPrior).
    tie: ek sütunların (ör. atom göstergeleri) maskesini bağlı oldukları sütunun maskesine eşitler {ek: asıl};
    n_eval: CV hatası yalnız ilk n_eval sütunda ölçülür."""
    rng = np.random.default_rng(seed + 99); n, d = X.shape
    folds = []
    for _ in range(reps):
        idx = rng.permutation(n); nv = max(2, int(round(0.2 * n)))
        hide = (rng.random((nv, d)) < rate).astype(float)
        for a_, o_ in (tie or {}).items(): hide[:, a_] = hide[:, o_]
        folds.append((idx[nv:], idx[:nv], hide))
    epss = epss if ridge_cv else (1e-3,)
    best = {}
    for k in ks:
        if k > 1 and int(0.8 * n) < min_per_cluster * k: continue
        moms = [cluster_moments(X[tr], k, fuzzifier, seed, method) for tr, _, _ in folds]
        for lam in (lams if k > 1 else (0.0,)):
            for eps in epss:
                errs = []
                for mom, (tr, va, hide) in zip(moms, folds):
                    mv = 1 - hide
                    ce = prior_from_moments(mom, lam, eps)(T(X[va] * mv), T(mv))[0].cpu().numpy()
                    ne = n_eval or d
                    errs.append(((ce - X[va])[:, :ne] ** 2 * hide[:, :ne]).sum() / max(hide[:, :ne].sum(), 1))
                e = float(np.mean(errs))
                if k not in best or e < best[k][2]: best[k] = (lam, eps, e)
    emin = min(v[2] for v in best.values()); ks_ = sorted(best)
    lw = np.array([-(best[k][2] - emin) / (tau * emin) for k in ks_]); lw -= np.log(np.exp(lw).sum())
    models = [prior_from_moments(cluster_moments(X, k, fuzzifier, seed, method), best[k][0], best[k][1]) for k in ks_]
    info = {k: dict(lam=best[k][0], eps=best[k][1], cv=best[k][2], w=float(np.exp(l))) for k, l in zip(ks_, lw)}
    return MultiPrior(models, [float(l) for l in lw]), info
