"""SCWGI-T: transdüktif kip — Gauss kolu, eğitim satırları + doldurulacak satırların yalnız GÖZLENEN hücreleriyle
eksik-veri EM'i ile yeniden kestirilir (gizli değerler kullanılmaz). Marjinal türü, (λ, ε) ve model ağırlıkları w_K
eğitimde seçilen değerlerde kalır."""
from __future__ import annotations
import numpy as np
import torch
from ._core import T, MultiPrior
from .gauss import cluster_moments, prior_from_moments
from .copula import CopulaGauss
from .model import SCWGI, LinearGauss


def cond_stats(gp, x, m, chunk=256):
    """x, m: torch (B,d) (eksikler 0). Dönüş: x̂ (B,K,d), C (B,K,d,d), log r (B,K) — normalize edilmiş."""
    outs = []
    for s in range(0, len(x), chunk):
        xb, mb = x[s:s + chunk], m[s:s + chunk]
        mk = mb[:, None, :]
        r = (xb[:, None, :] - gp.mu[None]) * mk
        mo = mk[..., :, None] * mk[..., None, :]
        St = gp.S[None] * mo + torch.diag_embed(1 - mb)[:, None]
        L = torch.linalg.cholesky(St)
        y = torch.cholesky_solve(r[..., None], L)[..., 0]
        ce = gp.mu[None] + (gp.S[None] @ y[..., None])[..., 0]
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        logp = gp.logpi[None] - 0.5 * ((r * y).sum(-1) + logdet + mb.sum(-1, keepdim=True) * np.log(2 * np.pi))
        SDo = gp.S[None] * mk[..., None, :]
        Z = torch.cholesky_solve(SDo.transpose(-1, -2), L)
        C = gp.S[None] - SDo @ Z
        mm = (1 - mk)                                                  # yalnız eksik blok
        C = C * mm[..., :, None] * mm[..., None, :]
        ce = torch.where(mk > 0, xb[:, None, :], ce)
        outs.append((ce, C, torch.log_softmax(logp, -1)))
    return tuple(torch.cat(t) for t in zip(*outs))


def em_raw(mu, S, pi, Xtr, Xq, Mq, iters=20, reg=1e-3):
    """Ham GMM momentleri (μ, Σ, π) üzerinde eksik-veri EM'i; E-adımında Σ + reg·I (sklearn reg_covar ile aynı).
    Xq boşsa ve başlangıç yakınsamış bir eğitim-EM çözümüyse, sabit noktada kalır (kontrol)."""
    xa = torch.cat([T(Xtr), T(np.asarray(Xq) * Mq)]); ma = torch.cat([torch.ones_like(T(Xtr)), T(Mq)])
    K, d = mu.shape; I = torch.eye(d, device=xa.device)
    mu, S, logpi = T(mu), T(S), T(np.log(np.asarray(pi) + 1e-12))

    class _G: pass
    g = _G()
    for _ in range(iters):
        g.mu, g.S, g.logpi = mu, S + reg * I[None], logpi
        ce, C, logr = cond_stats(g, xa, ma)
        R = logr.exp(); Nk = R.sum(0).clamp_min(1e-8)
        mu = (R[..., None] * ce).sum(0) / Nk[:, None]
        dev = ce - mu[None]
        S = (R[..., None, None] * (dev[..., :, None] * dev[..., None, :] + C)).sum(0) / Nk[:, None, None]
        logpi = torch.log(Nk / Nk.sum())
    return mu.cpu().numpy(), S.cpu().numpy(), logpi.exp().cpu().numpy()


def transductive_rebuild(info, Xtr, Xq, Mq, seed, fuzzifier=1.5, method="em", iters=20):
    """MultiPrior'ı endüktif kurulumla AYNI tahminciyle, eğitim + doldurulacak satırların gözlenen hücreleriyle yeniden kurar.
    k>1: ham GMM momentleri (cluster_moments) → eksik-veri EM → (1−λ)Σ + λΣ_LW + εI (bir kez).
    k=1: eksik-veri EM ile ortalama/kovaryans → eğitimdeki Ledoit-Wolf büzülme oranıyla ölçekli birime büzülme → + εI."""
    from sklearn.covariance import LedoitWolf
    ks = sorted(info); models = []; lw = []
    lwm = LedoitWolf().fit(Xtr); delta = lwm.shrinkage_; Sg = lwm.covariance_
    for k in ks:
        lam, eps = info[k]["lam"], info[k]["eps"]
        if k == 1:
            mu0, S0 = Xtr.mean(0, keepdims=True), np.cov(Xtr.T, bias=True)[None]
            mu, S, pi = em_raw(mu0, S0, np.ones(1), Xtr, Xq, Mq, iters)
            Sk = S[0]; S1 = (1 - delta) * Sk + delta * np.trace(Sk) / Sk.shape[0] * np.eye(Sk.shape[0])
            gp = prior_from_moments((mu, S1[None], np.ones(1), Sg), 0.0, eps)
        else:
            mu0, S0, pi0, _ = cluster_moments(Xtr, k, fuzzifier, seed, method)
            mu, S, pi = em_raw(mu0, S0, pi0, Xtr, Xq, Mq, iters)
            gp = prior_from_moments((mu, S, pi, Sg), lam, eps)
        models.append(gp); lw.append(np.log(info[k]["w"] + 1e-300))
    lw = np.array(lw); lw -= np.log(np.exp(lw).sum())
    return MultiPrior(models, [float(v) for v in lw])


class SCWGIT(SCWGI):
    def __init__(self, em_iters=20, transductive=True, **kw):
        super().__init__(**kw); self.em_iters, self.transductive = em_iters, transductive

    def fit(self, X):
        super().fit(X); self._Xfit = np.asarray(X, float)
        return self

    def _rebuild(self, Xq, Mq):
        if self.marg == "copula":
            gp = transductive_rebuild(self.info, self.Zfit, self.cop.to_latent(Xq, Mq), Mq, self.random_state,
                                      self.fuzzifier, "em", self.em_iters)
            return CopulaGauss(self.cop, gp)
        gp = transductive_rebuild(self.info, self._Xfit, Xq * Mq, Mq, self.random_state, self.fuzzifier, "em", self.em_iters)
        return LinearGauss(gp)

    def transform_dist(self, X, M):
        X = np.asarray(X, float).copy(); M = np.asarray(M, float); X[M == 0] = 0.0
        Xq, Mq = (X, M) if self.transductive else (X[:0], M[:0])
        saved = self.br
        try:
            self.br = self._rebuild(Xq, Mq)
            return super().transform_dist(X, M)
        finally:
            self.br = saved
