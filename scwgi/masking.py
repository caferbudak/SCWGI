"""Ortak deney protokolü: bölme + z-ölçekleme, eksiklik mekanizmaları (MCAR/MAR/MNAR), metrikler."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_train_test(X: np.ndarray, train_ratio: float = 0.70, seed: int = 42):
    """Veriyi train/test olarak böl, StandardScaler uygula."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(len(X) * train_ratio)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])
    return X_train, X_test, scaler


def introduce_missing(X, rate, mechanism="MCAR", seed=42):
    rng  = np.random.default_rng(seed)
    mask = np.ones(X.shape, dtype=np.float32)

    if mechanism == "MCAR":
        mask[rng.random(X.shape) < rate] = 0.0

    elif mechanism == "MAR":
        n, d = X.shape
        obs_cols = rng.choice(d, size=max(1, d // 2), replace=False)
        for col in range(d):
            if col not in obs_cols:
                ref_col = rng.choice(obs_cols)
                p = rate * (1.0 + X[:, ref_col])
                p = np.clip(p / (p.max() + 1e-8), 0.0, 1.0) * rate * 2
                mask[:, col][rng.random(n) < np.clip(p, 0, 1)] = 0.0

    elif mechanism == "MNAR":
        thresholds = np.percentile(X, rate * 100, axis=0)
        mask[X < thresholds] = 0.0

    else:
        raise ValueError(f"Bilinmeyen mekanizma: {mechanism}")

    X_miss = X.copy()
    X_miss[mask == 0.0] = 0.0
    return X_miss, mask


def compute_metrics(X_true, X_imputed, mask):
    missing_locs = mask == 0.0
    if missing_locs.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    y_true = X_true[missing_locs]
    y_pred = X_imputed[missing_locs]
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }
