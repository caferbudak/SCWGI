"""SCWGI — Soft-Cluster Weighted Gaussian Imputer (Yumuşak Küme Ağırlıklı Gauss Doldurucu).

Kullanım:
    from scwgi import SCWGIImputer, ConformalCalibrator
    imp = SCWGIImputer(random_state=42).fit(X_train)       # tam gözlenmiş eğitim verisi (z-ölçek önerilir)
    X_hat = imp.transform(X_missing, M)                     # M: 1 = gözlenen, 0 = eksik
    X_hat, s = imp.transform_dist(X_missing, M)             # koşullu ortalama ve koşullu standart sapma
"""
from .model import SCWGI, AugWrap
from .transductive import SCWGIT
from .conformal import ConformalCalibrator


def SCWGIImputer(random_state=42, **kw):
    """Makaledeki yapılandırma: atom göstergeli genişletme + SCWGI (otomatik marjinal, çok çözünürlüklü küme-Gauss)."""
    return AugWrap(SCWGI, random_state=random_state, **kw)


def SCWGITImputer(random_state=42, **kw):
    """Transdüktif kip (SCWGI-T): eğitim SCWGI ile aynı; çıkarımda bileşenler doldurulacak satırların gözlenen
    hücreleriyle eksik-veri EM'i ile yeniden kestirilir."""
    return AugWrap(SCWGIT, random_state=random_state, **kw)


__all__ = ["SCWGIImputer", "SCWGITImputer", "SCWGI", "SCWGIT", "AugWrap", "ConformalCalibrator"]
__version__ = "1.0.0"
