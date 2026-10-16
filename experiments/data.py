"""Veri yükleyiciler ve deney koşulları (makaledeki ortak protokol). Veri dosyaları ../data altında beklenir
(bkz. experiments/README.md: indirme bağlantıları). Meme Kanseri (WDBC) scikit-learn ile gelir."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from scipy.stats import kurtosis, skew

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TABULAR_DIR = DATA_DIR / "tabular"
CWRU_DIR = DATA_DIR / "cwru"

DATASETS = {
    "Diabetes":    "tabular/diabetes.csv",
    "BreastCancer": None,                          # sklearn'den otomatik
    "Fertility":   "tabular/fertility_Diagnosis.csv",
    "GoldPrice":   "tabular/FINAL_USO.csv",
    "CWRU":        "cwru",                         # CSV klasörü
}

FERTILITY_COLS = [
    "Season", "Age", "ChildDisease", "Accident", "Surgical",
    "HighFever", "Alcohol", "Smoking", "SittingHours"
]

CWRU_CSV_FILES = {
    "Time_Normal_1_098.csv": "Normal",
    "B007_1_123.csv":        "Ball_007",
    "B014_1_190.csv":        "Ball_014",
    "B021_1_227.csv":        "Ball_021",
    "IR007_1_110.csv":       "IR_007",
    "IR014_1_175.csv":       "IR_014",
    "IR021_1_214.csv":       "IR_021",
    "OR007_6_1_136.csv":     "OR_007",
    "OR014_6_1_202.csv":     "OR_014",
    "OR021_6_1_239.csv":     "OR_021",
}

EXPERIMENT_CONFIG = {
    "Diabetes": {
        "mechanisms": ["MAR"],
        "rates":      [0.10, 0.20],
        "metric":     "mae_rmse",
    },
    "BreastCancer": {
        "mechanisms": ["MCAR"],
        "rates":      [0.10, 0.20, 0.30],
        "metric":     "mae_rmse",
    },
    "Fertility": {
        "mechanisms": ["MCAR", "MNAR"],
        "rates":      [0.10, 0.20, 0.30],
        "metric":     "mae_rmse",
    },
    "GoldPrice": {
        "mechanisms": ["MCAR"],
        "rates":      [0.05, 0.10, 0.15],
        "metric":     "r2",
    },
    "CWRU": {
        "mechanisms": ["MCAR"],
        "rates":      [0.10],
        "metric":     "r2",
    },
}

def _load_cwru_csv(cwru_dir: Path) -> np.ndarray:
    """
    CWRU CSV dosyalarını yükle, sinyal istatistiklerini çıkar.
    Her CSV: tek sütun ham sinyal (DE_time kanalı, MAT'tan dönüştürülmüş).
    Her pencere için: RMS, Kurtosis, Skewness, Peak, CrestFactor + 20 FFT bileşeni
    """
    rows = []
    window_size = 2048

    for csv_file, label in CWRU_CSV_FILES.items():
        csv_path = cwru_dir / csv_file
        if not csv_path.exists():
            print(f"  [UYARI] {csv_file} bulunamadı, atlanıyor.")
            continue

        signal = pd.read_csv(csv_path, header=None).values.flatten().astype(float)

        # Kaydırmalı pencere ile özellik çıkarımı
        for start in range(0, len(signal) - window_size, window_size):
            seg  = signal[start:start + window_size]
            rms  = float(np.sqrt(np.mean(seg**2)))
            kurt = float(kurtosis(seg))
            sk   = float(skew(seg))
            peak = float(np.max(np.abs(seg)))
            cf   = float(peak / (rms + 1e-8))
            fft_mag = np.abs(np.fft.rfft(seg))[:20].tolist()
            rows.append([rms, kurt, sk, peak, cf] + fft_mag)

    if not rows:
        raise ValueError(
            "CWRU CSV dosyaları okunamadı veya klasör boş.\n"
            f"Beklenen konum: {cwru_dir.resolve()}"
        )

    return np.array(rows, dtype=float)

def load_dataset(name: str) -> np.ndarray:
    """
    Veri setini yükle ve sadece sayısal özellikleri döndür (hedef hariç).

    Args:
        name: 'Diabetes', 'BreastCancer', 'Fertility', 'GoldPrice', 'CWRU'

    Returns:
        X: (n_samples, n_features) float array
    """
    if name == "BreastCancer":
        return load_breast_cancer().data.astype(float)

    if name == "CWRU":
        return _load_cwru_csv(CWRU_DIR)

    path = DATA_DIR / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' bulunamadı.\n"
            f"Beklenen konum: {path.resolve()}\n"
            f"Klasör yapısı: data/tabular/ ve data/cwru/"
        )

    if name == "Diabetes":
        df = pd.read_csv(path)
        return df.drop(columns=["Outcome"], errors="ignore").values.astype(float)

    elif name == "Fertility":
        df = pd.read_csv(path)
        # Sütun başlıkları varsa kullan, yoksa ilk 9 sütunu al
        if all(c in df.columns for c in FERTILITY_COLS):
            return df[FERTILITY_COLS].values.astype(float)
        else:
            return df.iloc[:, :9].values.astype(float)

    elif name == "GoldPrice":
        df = pd.read_csv(path)
        num_cols = df.select_dtypes(include=np.number).columns
        return df[num_cols].ffill().bfill().values.astype(float)

    raise ValueError(f"Bilinmeyen veri seti: {name}")
