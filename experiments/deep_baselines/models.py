"""
models.py
---------
Üç imputation modeli:
  1. HyperImpute  – Jarrett vd. (2022), hyperimpute paketi
  2. TabDDPM      – Kotelnikov vd. (2023), DDPM tabanlı koşullu imputation
  3. SAINT        – Somepalli vd. (2021), col + row attention Transformer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Ortak yardımcılar ─────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════
#  1. HyperImpute
# ══════════════════════════════════════════════════════════════════════════

class HyperImputer:
    """
    HyperImpute wrapper (Jarrett vd., 2022).
    Kaynak: https://github.com/vanderschaarlab/hyperimpute
    Kurulum: pip install hyperimpute
    """

    def __init__(self, optimizer: str = "hyperimpute", random_state: int = 42):
        """
        Args:
            optimizer: 'hyperimpute', 'missforest', 'mice', 'gain', 'softimpute', ...
            random_state: Seed
        """
        self.optimizer    = optimizer
        self.random_state = random_state
        self._plugin      = None

    def fit_transform(self, X_train: np.ndarray, X_test_missing: np.ndarray) -> np.ndarray:
        """
        X_train üzerinde fit et, X_test_missing'i doldur.
        HyperImpute train+test'i birleştirerek çalıştırır.

        Returns:
            X_imputed: X_test_missing ile aynı boyutta, eksikler doldurulmuş
        """
        try:
            from hyperimpute.plugins.imputers import Imputers
            import pandas as pd
        except ImportError:
            raise ImportError(
                "HyperImpute kurulu değil.\n"
                "Kurmak için: pip install hyperimpute"
            )

        plugin = Imputers().get(self.optimizer)

        # Test verisindeki 0.0 → np.nan dönüşümü (HyperImpute NaN bekler)
        X_test_nan = X_test_missing.copy().astype(float)
        # mask=0 olan konumlar 0 ile doldurulmuştu; orijinal maskeyi bilmiyoruz
        # Bu yüzden caller X_test'i mask ile birlikte geçirir (mask_obs argümanı)
        df_test = pd.DataFrame(X_test_nan)

        fitted = plugin.fit_transform(df_test)
        return fitted.values

    def fit_transform_with_mask(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        mask_obs: np.ndarray
    ) -> np.ndarray:
        """
        mask_obs: 1=gözlemlenen, 0=eksik
        Eksik konumları NaN'a çevir, impute et.
        """
        try:
            from hyperimpute.plugins.imputers import Imputers
            import pandas as pd
        except ImportError:
            raise ImportError("pip install hyperimpute")

        plugin = Imputers().get(self.optimizer)

        X_nan = X_test.copy().astype(float)
        X_nan[mask_obs == 0.0] = np.nan

        df = pd.DataFrame(X_nan)
        imputed = plugin.fit_transform(df)
        return imputed.values


# ══════════════════════════════════════════════════════════════════════════
#  2. TabDDPM
# ══════════════════════════════════════════════════════════════════════════

def _cosine_beta_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    steps         = torch.arange(T + 1, dtype=torch.float64)
    alphas_bar    = torch.cos(((steps / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_bar    = alphas_bar / alphas_bar[0]
    betas         = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return torch.clamp(betas, 1e-5, 0.999).float()


class _DDPMScheduler:
    def __init__(self, T: int = 500):
        self.T          = T
        self.betas      = _cosine_beta_schedule(T).to(DEVICE)
        self.alphas     = 1 - self.betas
        self.alpha_bar  = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.alpha_bar[t].view(-1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def p_sample(self, model, xt, t_int, mask=None, x_obs=None):
        t_tensor = torch.full((xt.size(0),), t_int, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            eps_pred = model(xt, t_tensor)

        ab      = self.alpha_bar[t_int]
        ab_prev = self.alpha_bar[t_int - 1] if t_int > 0 else torch.tensor(1.0, device=DEVICE)
        x0_pred = (xt - (1 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-8)
        x0_pred = x0_pred.clamp(-5, 5)

        coef1 = ab_prev.sqrt() * self.betas[t_int] / (1 - ab)
        coef2 = self.alphas[t_int].sqrt() * (1 - ab_prev) / (1 - ab)
        mean  = coef1 * x0_pred + coef2 * xt

        sigma = ((1 - ab_prev) / (1 - ab) * self.betas[t_int]).clamp(min=0).sqrt()
        noise = torch.randn_like(xt) if t_int > 0 else torch.zeros_like(xt)
        xt_prev = mean + sigma * noise

        # Replacement Method: gözlemlenenler yeniden yerleştirilir
        if mask is not None and x_obs is not None:
            t_prev = max(t_int - 1, 0)
            t_prev_tensor = torch.full((xt.size(0),), t_prev, device=DEVICE, dtype=torch.long)
            noisy_obs, _ = self.q_sample(x_obs, t_prev_tensor)
            xt_prev = mask * noisy_obs + (1 - mask) * xt_prev

        return xt_prev


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _ResBlock(nn.Module):
    def __init__(self, dim: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1    = nn.Linear(dim, dim)
        self.fc2    = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(t_dim, dim)
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)
        self.act    = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x))) + self.t_proj(t_emb)
        h = self.drop(self.act(self.norm2(self.fc2(h))))
        return x + h


class _TabDDPMNet(nn.Module):
    """Denoising ağı: MLP + sinusoidal zaman gömme + ResBlock"""

    def __init__(self, d_in: int, hidden: int = 256, n_layers: int = 4, t_dim: int = 128):
        super().__init__()
        self.t_emb      = _SinusoidalEmbedding(t_dim)
        self.input_proj = nn.Linear(d_in, hidden)
        self.blocks     = nn.ModuleList([_ResBlock(hidden, t_dim) for _ in range(n_layers)])
        self.out        = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d_in))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_emb(t)
        h     = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out(h)


class TabDDPMImputer:
    """
    TabDDPM tabanlı imputation (Kotelnikov vd., 2023).
    Koşullu üretim: Replacement Method ile eksik sütunlar reverse diffusion ile yeniden üretilir.

    Referans: https://arxiv.org/abs/2209.15421
    """

    def __init__(
        self,
        T: int         = 500,
        hidden: int    = 256,
        n_layers: int  = 4,
        t_dim: int     = 128,
        epochs: int    = 300,
        batch_size: int = 256,
        lr: float      = 1e-3,
        n_samples: int = 5,
        random_state: int = 42,
    ):
        self.T            = T
        self.hidden       = hidden
        self.n_layers     = n_layers
        self.t_dim        = t_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.n_samples    = n_samples
        self.random_state = random_state
        self._model       = None
        self._scheduler   = None

    def fit(self, X_train: np.ndarray, verbose: bool = True) -> "TabDDPMImputer":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._scheduler = _DDPMScheduler(self.T)
        self._model     = _TabDDPMNet(
            d_in=X_train.shape[1],
            hidden=self.hidden,
            n_layers=self.n_layers,
            t_dim=self.t_dim
        ).to(DEVICE)

        opt   = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Xt    = _to_tensor(X_train)
        loader = DataLoader(TensorDataset(Xt), batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for ep in range(self.epochs):
            total = 0.0
            for (xb,) in loader:
                t     = torch.randint(0, self.T, (xb.size(0),), device=DEVICE)
                xt, noise = self._scheduler.q_sample(xb, t)
                loss  = F.mse_loss(self._model(xt, t), noise)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            sch.step()
            if verbose and (ep + 1) % 100 == 0:
                print(f"    [TabDDPM] Epoch {ep+1}/{self.epochs} | Loss: {total/len(loader):.4f}")

        return self

    def transform(self, X_missing: np.ndarray, mask_obs: np.ndarray) -> np.ndarray:
        """
        Args:
            X_missing: Eksik konumlar 0.0 ile doldurulmuş (n, d)
            mask_obs:  1=gözlemlenen, 0=eksik (n, d)

        Returns:
            X_imputed: (n, d) doldurulmuş array
        """
        assert self._model is not None, "Önce fit() çağrılmalı."
        self._model.eval()

        Xm   = _to_tensor(X_missing)
        Mo   = _to_tensor(mask_obs)
        preds = []

        for _ in range(self.n_samples):
            xt = Mo * Xm + (1 - Mo) * torch.randn_like(Xm)
            for t in reversed(range(self.T)):
                xt = self._scheduler.p_sample(self._model, xt, t, mask=Mo, x_obs=Xm)
            preds.append(xt.cpu().numpy())

        X_imp = np.mean(preds, axis=0)
        # Gözlemlenen değerleri koru
        X_result = X_missing.copy()
        X_result[mask_obs == 0.0] = X_imp[mask_obs == 0.0]
        return X_result

    def fit_transform(
        self,
        X_train: np.ndarray,
        X_test_missing: np.ndarray,
        mask_obs: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        self.fit(X_train, verbose=verbose)
        return self.transform(X_test_missing, mask_obs)


# ══════════════════════════════════════════════════════════════════════════
#  3. SAINT
# ══════════════════════════════════════════════════════════════════════════

class _MHA(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.out   = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.heads
        q, k, v = [
            t.view(B, N, h, D // h).transpose(1, 2)
            for t in self.qkv(x).chunk(3, dim=-1)
        ]
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)


class _SAINTBlock(nn.Module):
    """
    SAINT Transformer bloğu:
    Col-Attention (feature-wise) + Row-Attention (sample-wise) + FFN
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, ff_mult: int = 2):
        super().__init__()
        self.norm_col = nn.LayerNorm(dim)
        self.col_attn = _MHA(dim, heads, dropout)

        self.norm_row = nn.LayerNorm(dim)
        self.row_attn = _MHA(dim, heads, dropout)

        self.norm_ff  = nn.LayerNorm(dim)
        self.ff       = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_features, dim)
        x = x + self.col_attn(self.norm_col(x))

        # Row attention: örnekler arası dikkat
        # (B, N, D) → (N, B, D) → MHA → (N, B, D) → (B, N, D)
        xt = x.permute(1, 0, 2)                        # (N, B, D)
        xt = xt + self.row_attn(self.norm_row(xt))
        x  = xt.permute(1, 0, 2)                       # (B, N, D)

        x = x + self.ff(self.norm_ff(x))
        return x


class _SAINTNet(nn.Module):
    """SAINT imputation ağı"""

    def __init__(
        self,
        n_features: int,
        dim: int       = 64,
        depth: int     = 3,
        heads: int     = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features

        # Her feature için bağımsız skaler → dim projeksiyonu
        self.feature_proj = nn.ModuleList([nn.Linear(1, dim) for _ in range(n_features)])
        # Öğrenilebilir MASK token
        self.mask_token   = nn.Parameter(torch.randn(dim) * 0.02)

        self.blocks = nn.ModuleList([
            _SAINTBlock(dim, heads, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Her feature için bağımsız regresyon kafası
        self.heads_out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1)
            )
            for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, F) — eksik konumlar 0.0
            mask: (B, F) — 1=gözlemlenen, 0=eksik
        Returns:
            pred: (B, F) — tüm konumlar için tahmin
        """
        B, F = x.shape
        tokens = []
        for i in range(F):
            feat_tok = self.feature_proj[i](x[:, i:i+1])          # (B, dim)
            m_i      = mask[:, i:i+1]                              # (B, 1)
            tok      = m_i * feat_tok + (1 - m_i) * self.mask_token
            tokens.append(tok.unsqueeze(1))                        # (B, 1, dim)

        h = torch.cat(tokens, dim=1)    # (B, F, dim)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)               # (B, F, dim)

        # Her feature için bağımsız çıkış
        preds = torch.cat([self.heads_out[i](h[:, i, :]) for i in range(F)], dim=1)
        return preds  # (B, F)


class SAINTImputer:
    """
    SAINT tabanlı imputation (Somepalli vd., 2021).
    Self-supervised mask-token pretraining + regresyon kafası.

    Referans: https://arxiv.org/abs/2106.01342
    """

    def __init__(
        self,
        dim: int           = 64,
        depth: int         = 3,
        heads: int         = 4,
        dropout: float     = 0.1,
        mask_ratio: float  = 0.15,  # self-supervised eğitim mask oranı
        epochs: int        = 300,
        batch_size: int    = 128,
        lr: float          = 3e-4,
        random_state: int  = 42,
    ):
        self.dim          = dim
        self.depth        = depth
        self.heads        = heads
        self.dropout      = dropout
        self.mask_ratio   = mask_ratio
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state
        self._model       = None

    def fit(self, X_train: np.ndarray, verbose: bool = True) -> "SAINTImputer":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_features  = X_train.shape[1]
        self._model = _SAINTNet(
            n_features=n_features,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dropout=self.dropout,
        ).to(DEVICE)

        opt    = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Xt     = _to_tensor(X_train)
        loader = DataLoader(TensorDataset(Xt), batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for ep in range(self.epochs):
            total = 0.0
            for (xb,) in loader:
                # Self-supervised: rastgele mask ile eğit
                m    = (torch.rand_like(xb) > self.mask_ratio).float()
                pred = self._model(m * xb, m)
                # Sadece maskelenmiş konumlarda kayıp
                loss = ((pred - xb) ** 2 * (1 - m)).sum() / ((1 - m).sum() + 1e-8)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            sch.step()
            if verbose and (ep + 1) % 100 == 0:
                print(f"    [SAINT] Epoch {ep+1}/{self.epochs} | Loss: {total/len(loader):.4f}")

        return self

    def transform(self, X_missing: np.ndarray, mask_obs: np.ndarray) -> np.ndarray:
        """
        Args:
            X_missing: (n, d) eksik konumlar 0.0
            mask_obs:  (n, d) 1=gözlemlenen, 0=eksik

        Returns:
            X_imputed: (n, d)
        """
        assert self._model is not None, "Önce fit() çağrılmalı."
        self._model.eval()

        with torch.no_grad():
            pred = self._model(
                _to_tensor(X_missing),
                _to_tensor(mask_obs)
            ).cpu().numpy()

        X_result = X_missing.copy()
        X_result[mask_obs == 0.0] = pred[mask_obs == 0.0]
        return X_result

    def fit_transform(
        self,
        X_train: np.ndarray,
        X_test_missing: np.ndarray,
        mask_obs: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        self.fit(X_train, verbose=verbose)
        return self.transform(X_test_missing, mask_obs)
