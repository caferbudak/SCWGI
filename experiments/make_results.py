"""
make_results.py — SCWGI ile ortak protokoldeki tüm sonuçları birleştirir (makale Tablo 4–6).
Koşul bazında ortalama ± std, birincilik sayıları, ortalama sıralar, Friedman testi, SCWGI'ye (ve SCWGI-T'ye) karşı
Holm düzeltmeli Wilcoxon işaretli sıra testleri, endüktif/transdüktif grup analizleri, MNAR-MAE ve süre özetleri.
Kaynaklar:
    results/scwgi/SCWGI_raw.csv, SCWGI-T_raw.csv           run_scwgi.py
    results/competitors/*_raw.csv                          run_baselines_lib.py, run_hyperimpute.py, run_deep_baselines.py
Çıktı: results/summary/
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

R = Path("results"); O = R / "summary"; O.mkdir(parents=True, exist_ok=True)
COLS = ["Dataset", "Model", "Mechanism", "Rate", "Seed", "MAE", "RMSE", "R2", "Time_sec"]
ORDER = [("BreastCancer", "MCAR", "%10"), ("BreastCancer", "MCAR", "%20"), ("BreastCancer", "MCAR", "%30"),
         ("GoldPrice", "MCAR", "%5"), ("GoldPrice", "MCAR", "%10"), ("GoldPrice", "MCAR", "%15"),
         ("CWRU", "MCAR", "%10"), ("Diabetes", "MAR", "%10"), ("Diabetes", "MAR", "%20"),
         ("Fertility", "MCAR", "%10"), ("Fertility", "MCAR", "%20"), ("Fertility", "MCAR", "%30"),
         ("Fertility", "MNAR", "%10"), ("Fertility", "MNAR", "%20"), ("Fertility", "MNAR", "%30")]
INDUCTIVE = ["SCWGI", "SAINT", "TabDDPM"]
DS_TR = {"BreastCancer": "Meme Kanseri", "GoldPrice": "Altın", "CWRU": "CWRU", "Diabetes": "Diyabet", "Fertility": "Fertilite"}


def load():
    files = glob.glob(str(R / "scwgi/*_raw.csv")) + glob.glob(str(R / "competitors/*_raw.csv"))
    d = pd.concat([pd.read_csv(f).reindex(columns=COLS) for f in files], ignore_index=True)
    return d.drop_duplicates(["Dataset", "Model", "Mechanism", "Rate", "Seed"], keep="last")


def holm(res):
    res = res.sort_values("p").reset_index(drop=True); k = len(res); adj, run = [], 0.0
    for i, p in enumerate(res.p.values):
        run = max(run, min(1.0, (k - i) * p)); adj.append(run)
    res["p_Holm"] = adj
    return res


def wilcoxon_vs(R2, me, skip):
    res = []
    for m in R2.columns:
        if m in skip or m == me: continue
        w = stats.wilcoxon(R2[me], R2[m], alternative="two-sided", zero_method="wilcox")
        res.append({"Rakip": m, f"{me} daha iyi (koşul)": int((R2[me] > R2[m]).sum()),
                    "Ort. ΔR²": round(float((R2[me] - R2[m]).mean()), 4), "p": w.pvalue})
    return holm(pd.DataFrame(res))


def table_md(T, fmt="{:.3f}", best="max"):
    cols = list(T.columns); lines = ["| Koşul | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for (ds, mech, rate), row in T.iterrows():
        b = row.max() if best == "max" else row.min()
        cells = [("**" + fmt.format(v) + "**") if np.isclose(v, b) else fmt.format(v) for v in row.values]
        lines.append(f"| {DS_TR[ds]} {mech} {rate} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    d = load()
    cnt = d.groupby("Model").size(); complete = cnt[cnt == 75].index.tolist()
    print("Tamamlanan (75):", complete, "\nEksik:", cnt[cnt < 75].to_dict())
    d = d[d.Model.isin(complete)]
    g = d.groupby(["Dataset", "Mechanism", "Rate", "Model"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"]); g.to_csv(O / "kosul_ortalama_std.csv")
    mean = d.groupby(["Dataset", "Mechanism", "Rate", "Model"])[["MAE", "RMSE", "R2"]].mean()
    R2 = mean.R2.unstack("Model").loc[ORDER]; MAE = mean.MAE.unstack("Model").loc[ORDER]; RMSE = mean.RMSE.unstack("Model").loc[ORDER]
    order_cols = [c for c in ["SCWGI", "SCWGI-T", "EM", "SAINT", "HyperImpute", "TabDDPM", "MissForest", "GAIN", "SoftImpute",
                              "MICE", "MIWAE", "Ortalama"] if c in R2.columns]
    R2, MAE, RMSE = R2[order_cols], MAE[order_cols], RMSE[order_cols]
    models = list(R2.columns)
    rank = R2.rank(1, ascending=False).mean().sort_values()
    fr = stats.friedmanchisquare(*[R2[m].values for m in models])
    win_r2 = R2.idxmax(1).value_counts().reindex(models, fill_value=0)
    win_mae = MAE.idxmin(1).value_counts().reindex(models, fill_value=0)
    res = wilcoxon_vs(R2, "SCWGI", {"SCWGI-T"})
    res_nomnar = wilcoxon_vs(R2.iloc[:12], "SCWGI", {"SCWGI-T"})
    out = [f"Friedman χ² = {fr.statistic:.2f}, p = {fr.pvalue:.2e} (k = {len(models)}, N = 15)",
           "Ortalama sıra (R²):\n" + rank.round(2).to_string(),
           "R² birincilik (koşul):\n" + win_r2.to_string(), "MAE birincilik (koşul):\n" + win_mae.to_string(),
           "SCWGI'ye karşı Wilcoxon + Holm (15 koşul):\n" + res.to_string(index=False),
           "SCWGI'ye karşı Wilcoxon + Holm (MNAR hariç 12 koşul):\n" + res_nomnar.to_string(index=False)]
    ind = [m for m in INDUCTIVE if m in R2.columns]
    out.append("Endüktif grup birincilik:\n" + R2[ind].idxmax(1).value_counts().to_string())
    out.append("Endüktif grup ortalama sıra:\n" + R2[ind].rank(1, ascending=False).mean().round(2).to_string())
    if "SCWGI-T" in R2.columns:
        trans = [m for m in models if m not in ("SCWGI", "SAINT", "TabDDPM")]
        out.append("Transdüktif grup ortalama sıra:\n" + R2[trans].rank(1, ascending=False).mean().sort_values().round(2).to_string())
        out.append("Transdüktif grup birincilik:\n" + R2[trans].idxmax(1).value_counts().to_string())
        rt = wilcoxon_vs(R2[trans], "SCWGI-T", set()); rt.to_csv(O / "wilcoxon_holm_T.csv", index=False)
        out.append("SCWGI-T'ye karşı Wilcoxon + Holm (transdüktif grup):\n" + rt.to_string(index=False))
        K = ["Dataset", "Mechanism", "Rate", "Seed"]
        a = d[d.Model == "SCWGI"].set_index(K).R2; t = d[d.Model == "SCWGI-T"].set_index(K).R2
        dd = (t - a).reset_index(); dd["Grup"] = dd.Dataset + " " + dd.Mechanism
        sm = dd.groupby("Grup").R2.agg(["mean", lambda v: v.std(ddof=1) / np.sqrt(len(v)), "count"]); sm.columns = ["ΔR²", "SE", "n"]
        out.append("SCWGI-T − SCWGI (eşleştirilmiş, tohum düzeyi):\n" + sm.round(4).to_string())
    piv = d.pivot_table(index=["Dataset", "Mechanism", "Rate", "Seed"], columns="Model", values="R2")
    out.append("Tohum düzeyinde R² birincilik (75):\n" + piv.idxmax(1).value_counts().to_string())
    mn = MAE.iloc[12:]; out.append("MNAR MAE (düşük = iyi):\n" + mn.T.round(3).to_string())
    tm = d.groupby(["Model", "Dataset"]).Time_sec.median().unstack("Dataset"); tm.to_csv(O / "sure_medyan.csv")
    out.append("Medyan süre (s):\n" + tm.round(1).to_string())
    R2.round(3).to_csv(O / "R2_kosul.csv"); R2.to_csv(O / "R2_kosul_tam.csv"); MAE.round(4).to_csv(O / "MAE_kosul.csv")
    RMSE.round(4).to_csv(O / "RMSE_kosul.csv"); res.to_csv(O / "wilcoxon_holm.csv", index=False)
    (O / "tablo4_R2.md").write_text(table_md(R2)); (O / "tablo_MAE.md").write_text(table_md(MAE, "{:.3f}", "min"))
    txt = "\n\n".join(out); (O / "ozet.md").write_text(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
