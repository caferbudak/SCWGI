"""Bölünmüş ve Mondrian konformal aralıklar (standartlaştırılmış skor |y − x̂| / s).

Makaledeki tasarım (Bölüm 2.6):
  - Model eğitim satırlarının %85'inde uydurulur; kalan %15 kalibrasyon kümesidir.
  - Kalibrasyon satırlarında 5 MCAR + 5 MAR maskesiyle (oran: veri setinin ortalama eksiklik oranı) hücreler gizlenir.
  - Skor S = |y − x̂| / s ; bölünmüş konformal kantil q = S_(⌈(n+1)(1−α)⌉).
  - Mondrian: s'nin kalibrasyon üçte-birlik sınırlarına göre üç grupta ayrı kantiller.
"""
import numpy as np
from .masking import introduce_missing


def conf_q(scores, alpha=0.10):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(scores)[min(k, n) - 1] if n else np.nan


class ConformalCalibrator:
    def __init__(self, alpha=0.10, n_masks=10, mechanisms=("MCAR", "MAR")):
        self.alpha, self.n_masks, self.mechanisms = alpha, n_masks, mechanisms

    def fit(self, imputer, X_cal, rate, seed=42):
        """imputer: uydurulmuş SCWGI (transform_dist sağlayan); X_cal: tam gözlenmiş kalibrasyon satırları."""
        sc, cal_s = [], []
        half = self.n_masks // len(self.mechanisms)
        for j in range(self.n_masks):
            mech = self.mechanisms[min(j // half, len(self.mechanisms) - 1)]
            Mc = introduce_missing(X_cal, rate, mechanism=mech, seed=10_000 * seed + j)[1]; hc = Mc == 0
            mu, sd = imputer.transform_dist(X_cal * Mc, Mc)
            sc.append(np.abs(X_cal[hc] - mu[hc]) / sd[hc]); cal_s.append(sd[hc])
        s = np.concatenate(sc); cs = np.concatenate(cal_s)
        self.q_ = conf_q(s, self.alpha)
        self.edges_ = np.quantile(cs, [1 / 3, 2 / 3]); g = np.digitize(cs, self.edges_)
        self.qm_ = np.array([conf_q(s[g == i], self.alpha) for i in range(3)])
        return self

    def group(self, sd):
        """Mondrian grubu: 0 = düşük, 1 = orta, 2 = yüksek belirsizlik (kalibrasyon üçte-birlik sınırları)."""
        return np.digitize(sd, self.edges_)

    def intervals(self, imputer, X, M, mondrian=True):
        """Dönüş: (x̂, alt sınır, üst sınır, s); gözlenen hücrelerde aralık genişliği 0."""
        mu, sd = imputer.transform_dist(X, M)
        q = self.qm_[self.group(sd)] if mondrian else self.q_
        lo, hi = mu - q * sd, mu + q * sd
        return mu, np.where(M == 1, mu, lo), np.where(M == 1, mu, hi), sd
