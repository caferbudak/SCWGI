"""make_figures.py — Makale şekilleri (300 dpi PNG, yayın için açık zemin).
Şekil 2: 15 koşul üzerinde ortalama R² sırası (düşük = iyi), yöntem başına.
Şekil 3: bileşen ablasyonu — tam SCWGI'ye göre eşleştirilmiş ΔR², veri seti bazında.
Şekil 4: %90 aralık kapsaması — ham Gauss, bölünmüş konformal, Mondrian; veri seti bazında ve s üçte-birliklerine göre.
Renk: SCWGI ailesi kategorik 1–2. slot (mavi, turuncu); diğer yöntemler nötr gri (vurgu ilkesi).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

R = Path("results"); O = R / "figures"; O.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
DS_TR = {"BreastCancer": "Meme Kanseri", "GoldPrice": "Altın", "CWRU": "CWRU", "Diabetes": "Diyabet", "Fertility": "Fertilite"}


def fig_ranks():
    R2 = pd.read_csv(R / "summary/R2_kosul_tam.csv", index_col=[0, 1, 2])   # yuvarlanmamış ortalamalar (sıra bağı yapay oluşmasın)
    rank = R2.rank(1, ascending=False).mean().sort_values(ascending=False)
    # Düz çubuk: yalnızca eğitim kümesinde eğitilen (endüktif) yöntemler; taralı: transdüktif yöntemler
    from matplotlib.patches import Patch
    inductive = {"SCWGI", "SAINT", "TabDDPM"}
    fig, ax = plt.subplots(figsize=(5.4, 0.32 * len(rank) + 1.2))
    plt.rcParams["hatch.linewidth"] = 0.8
    for i, (m, v) in enumerate(rank.items()):
        col = BLUE if m == "SCWGI" else ORANGE if m == "SCWGI-T" else GRAY
        ind = m in inductive
        ax.barh(i, v, height=0.62, color=col if ind else "white", edgecolor=col, linewidth=1.0, hatch=None if ind else "////")
        ax.text(v + 0.08, i, f"{v:.2f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(rank))); ax.set_yticklabels(rank.index)
    ax.set_xlabel("Ortalama sıra (R², 15 koşul; düşük = daha iyi)"); ax.set_xlim(0, rank.max() + 1.2)
    ax.xaxis.grid(True, color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    handles = [Patch(facecolor=GRAY, edgecolor=GRAY, label="Endüktif (yalnızca eğitim kümesinde eğitilen)"),
               Patch(facecolor="white", edgecolor=GRAY, hatch="////", label="Transdüktif (doldurulacak satırları da kullanan)")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=8, bbox_to_anchor=(0.55, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(O / "sekil2_ortalama_sira.png"); plt.close(fig)


def fig_ablation():
    """Eşleştirilmiş fark: her (koşul, tohum) için varyant R² − SCWGI R²; MNAR hariç. Nokta = ortalama, çizgi = ±1 SE."""
    K = ["Dataset", "Mechanism", "Rate", "Seed"]
    base = pd.read_csv(R / "scwgi/SCWGI_raw.csv").set_index(K).R2
    rows = {}
    for v, lab in (("-kopula", "− kopula (doğrusal marjinal)"), ("-atom", "− atom göstergesi"), ("-ort", "− model ortalaması (tek K)"),
                   ("-küme", "− kümeleme (K = 1)"), ("-büzülme", "− kovaryans büzülmesi"), ("FCM-ağırlık", "FCM üyelik ağırlığı")):
        p = R / f"ablation/{v}_raw.csv"
        if p.exists(): rows[lab] = pd.read_csv(p)
    order = ["BreastCancer", "GoldPrice", "CWRU", "Diabetes", "Fertility"]
    M, S, recs = {}, {}, []
    for lab, d in rows.items():
        dd = (d.set_index(K).R2 - base).dropna().reset_index()
        dd = dd[dd.Mechanism != "MNAR"]
        g = dd.groupby("Dataset").R2
        M[lab] = g.mean(); S[lab] = g.std(ddof=1) / np.sqrt(g.count())
        for ds, v in g: recs.append({"Varyant": lab, "Veri seti": ds, "ΔR²_ort": v.mean(), "SE": v.std(ddof=1) / np.sqrt(len(v)), "n": len(v)})
    Mn, Sn = pd.DataFrame(M).T[order], pd.DataFrame(S).T[order]
    fig, axes = plt.subplots(1, len(order), figsize=(7.4, 0.32 * len(Mn) + 1.1), sharey=True)
    for ax, ds in zip(axes, order):
        v, e = Mn[ds].values * 100, Sn[ds].values * 100; y = np.arange(len(v))   # ΔR² × 100 (yüzde puan): eksen etiketleri sade kalır
        ax.axvline(0, color=INK2, lw=0.8)
        for yi, vi, ei in zip(y, v, e):
            c = GRAY if abs(vi) < 5e-2 else (BLUE if vi > 0 else ORANGE)
            ax.plot([vi - ei, vi + ei], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
            ax.plot(vi, yi, "o", ms=4.5, color=c, mec="white", mew=0.6)
        ax.set_title(DS_TR[ds], fontsize=9, color=INK); ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
        ax.xaxis.grid(True, color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(Mn))); axes[0].set_yticklabels(Mn.index, fontsize=8); axes[0].invert_yaxis()
    fig.supxlabel("ΔR² × 100 (tam SCWGI'ye göre, eşleştirilmiş; MCAR/MAR koşulları; ±1 SE)", fontsize=8)
    fig.tight_layout(); fig.savefig(O / "sekil3_ablasyon.png"); plt.close(fig)
    pd.DataFrame(recs).round(4).to_csv(O / "sekil3_veri.csv", index=False)


def fig_calibration():
    c = pd.read_csv(R / "calibration_raw.csv")
    c = c[c.Mechanism != "MNAR"]
    g = c.groupby("Dataset")[["cov90_raw", "cov90_conf", "cov90_mond"]].mean()
    t = c.groupby("Dataset")[["cov90_conf_yüksek", "cov90_mond_yüksek"]].mean()
    order = ["BreastCancer", "GoldPrice", "CWRU", "Diabetes", "Fertility"]; g, t = g.loc[order], t.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    x = np.arange(len(order)); w = 0.22
    # Nokta grafiği (kapsama oranı sıfırdan başlamayan eksende çubukla gösterilmez); 0.90 hedef çizgisi
    for i, (col, lab, colr, mk) in enumerate((("cov90_raw", "Ham Gauss aralığı", GRAY, "s"), ("cov90_conf", "Bölünmüş konformal", BLUE, "o"),
                                              ("cov90_mond", "Mondrian konformal", AQUA, "D"))):
        axes[0].plot(x + (i - 1) * w, g[col], mk, ms=6, color=colr, label=lab, mec="white", mew=0.8)
    for i, (col, colr, mk) in enumerate((("cov90_conf_yüksek", BLUE, "o"), ("cov90_mond_yüksek", AQUA, "D"))):
        axes[1].plot(x + (i - 0.5) * w, t[col], mk, ms=6, color=colr, mec="white", mew=0.8)
    for ax, ttl in zip(axes, ("Tüm hücreler", "En belirsiz üçte-bir (yüksek s)")):
        ax.axhline(0.90, color=INK, lw=0.8, ls="--"); ax.set_xticks(x); ax.set_xticklabels([DS_TR[d] for d in order], fontsize=8)
        ax.set_ylim(0.74, 1.0); ax.set_xlim(-0.6, len(order) - 0.4); ax.set_title(ttl, fontsize=9, color=INK)
        ax.yaxis.grid(True, color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("%90 aralık kapsaması")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(O / "sekil4_kalibrasyon.png"); plt.close(fig)
    pd.concat([g, t], axis=1).round(3).to_csv(O / "sekil4_veri.csv")


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["ranks", "ablation", "calibration"]):
        {"ranks": fig_ranks, "ablation": fig_ablation, "calibration": fig_calibration}[name]()
        print("ok", name)
