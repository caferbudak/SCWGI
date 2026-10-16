"""make_figure1.py — Şekil 1: SCWGI'nin yapısı (şematik). Çıktı: results/figures/sekil1_yapi.png (300 dpi)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

O = Path("results/figures"); O.mkdir(parents=True, exist_ok=True)
INK, INK2, LINE = "#0b0b0b", "#52514e", "#9a9993"
FILL = {"io": "#f3f2ee", "prep": "#eef3fb", "core": "#dbe8f8", "cal": "#ddf3ea", "cv": "#ffffff"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(7.4, 3.1)); ax.set_xlim(0, 100); ax.set_ylim(0, 41.5); ax.axis("off")


def box(x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=FILL[kind], ec=LINE if kind != "cv" else INK2,
                                lw=0.9, ls="-" if kind != "cv" else (0, (3, 2))))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.8, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.3, body, ha="center", va="top", fontsize=6.9, color=INK2, linespacing=1.35)
    return (x, y, w, h)


def arrow(p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2, ls=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


W, H, Y1, Y2 = 21.5, 14, 26, 3
b1 = box(1, Y1, W, H, "Girdi", r"gözlenen $x_o$, maske $m$" + "\n(eğitim istatistikleriyle\nz-ölçek)", "io")
b2 = box(26, Y1, W, H, "Atom göstergeleri", r"$a_{ij}=\mathbb{1}[x_{ij}=\alpha_j]$" + "\nmaske asıl değişkene bağlı\n(Denk. 1)", "prep")
b3 = box(51, Y1, W, H, "Marjinal dönüşüm", "doğrusal  |  ampirik kopula\n" + r"$\Phi^{-1}(\hat F_j(x))$" + ", Gauss–Hermite\ngeri dönüşüm (Denk. 2)", "prep")
b4 = box(76, Y1, 23, H, "Gauss karışım bankası", r"$K\in\{1,2,3,4,6,8\}$" + "\nFCM → EM bileşenleri\n" + r"$\tilde\Sigma=(1-\lambda)\hat\Sigma+\lambda\Sigma_{LW}+\varepsilon I$", "core")
b5 = box(76, Y2, 23, H, "Sorumluluklar", r"$r_c(x_o)\propto\pi_c\,\mathcal{N}(x_o;\mu_{c,o},\tilde\Sigma_{c,oo})$" + "\n× model ağırlığı " + r"$w_K$" + "\n(Denk. 4)", "core")
b6 = box(51, Y2, W, H, "Koşullu Gauss", r"$m_c,\ V_c$" + " (Cholesky)\n" + r"$\hat x=\sum w_K r_c m_c$" + "\n" + r"$s^2$" + ": toplam varyans (Denk. 5–6)", "core")
b7 = box(26, Y2, W, H, "Konformal kalibrasyon", r"$S=|y-\hat x|/s$" + "\nbölünmüş  |  Mondrian\n(" + r"$s$" + " üçte-birlikleri)", "cal")
b8 = box(1, Y2, W, H, "Çıktı", r"$\hat x$" + " ve %90 aralık\n" + r"$\hat x \pm q\,s$", "io")
mid = lambda b, side: {"r": (b[0] + b[2] + 0.4, b[1] + b[3] / 2), "l": (b[0] - 0.4, b[1] + b[3] / 2),
                       "b": (b[0] + b[2] / 2, b[1] - 0.4), "t": (b[0] + b[2] / 2, b[1] + b[3] + 0.4)}[side]
arrow(mid(b1, "r"), mid(b2, "l")); arrow(mid(b2, "r"), mid(b3, "l")); arrow(mid(b3, "r"), mid(b4, "l"))
arrow(mid(b4, "b"), mid(b5, "t")); arrow(mid(b5, "l"), mid(b6, "r")); arrow(mid(b6, "l"), mid(b7, "r")); arrow(mid(b7, "l"), mid(b8, "r"))
ax.text(50, 21.5, "Eğitim kümesi içi maskeli çapraz doğrulama: marjinal türü, her K için (λ, ε) ve " + r"$w_K\propto\exp(-(e_K-e_{min})/(\tau e_{min}))$",
        ha="center", va="center", fontsize=6.4, color=INK2, style="italic", linespacing=1.5)
fig.subplots_adjust(0.005, 0.01, 0.995, 0.99); fig.savefig(O / "sekil1_yapi.png"); plt.close(fig); print("ok")
