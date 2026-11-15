"""Figures 2 et 3 sur la base unique de 30 seeds (CTGAN : 10 seeds, etiquete)."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
F = json.load(open("final_results.json")); S = sorted(F, key=int)
B = {str(int(x.split("_")[-1][:-5])): json.load(open(x)) for x in glob.glob("sens/seed_*.json")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"; blue, orange, grey = "#2a78d6", "#eb6834", "#9a9893"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def strip(ax, groups, cols, names, ylab):
    r = np.random.default_rng(0)
    for i, (v, c) in enumerate(zip(groups, cols)):
        v = np.asarray(v); ax.bar(i, v.mean(), 0.6, color=c, alpha=.25, edgecolor=c)
        ax.scatter(i + r.uniform(-.15, .15, len(v)), v, s=12, color=c, zorder=3)
        ax.errorbar(i, v.mean(), v.std(ddof=1), color=INK, capsize=3, zorder=4)
        ax.text(i, v.mean() + v.std(ddof=1) + .01, f"{v.mean():.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right"); ax.set_ylabel(ylab); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
g = lambda m, gen, *p: [(lambda x: [x := x[k] for k in p][-1])(F[s][m][gen]) for s in S]
fig, ax = plt.subplots(figsize=(7, 3.6))
names = ["SR-DMLR", "Flat VAE", "Concatenation", "SMOTE-only", "CORAL", "Unsup.\nshared-private", "CTGAN\n(10 seeds)"]
grp = [g("SR-DMLR", "gauss", "STRT", "LR"), g("Flat-VAE", "gauss", "STRT", "LR"), [B[s]["Concatenation"] for s in S], [B[s]["SMOTE-only"] for s in S],
       [B[s]["CORAL"] for s in S], [B[s]["Unsup"] for s in S], [json.load(open(f"../rerun/sens/seed_{s}.json"))["CTGAN"] for s in S if int(s) < 10]]
strip(ax, grp, [blue, orange] + [grey] * 5, names, "STRT AUROC (LR)")
real = np.mean([B[s]["Real"] for s in S]); ax.axhline(real, ls="--", color=INK2, lw=1)
ax.text(6.4, real + .005, f"real features ({real:.3f})", ha="right", fontsize=7.5, color=INK2); ax.axhline(.5, ls=":", color=INK2, lw=1); ax.set_ylim(.35, .8)
ax.set_title("Synthetic-to-real transfer on held-out ABIDE-I participants (30 seeds)", loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig("figures_final/Fig2_utility.png", dpi=300); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
pn = ["SR-DMLR\nGaussian", "SR-DMLR\nSMOTE", "Flat VAE\nGaussian", "Flat VAE\nSMOTE"]; cols = [blue, "#9cc0ea", orange, "#f3b597"]
for ax, k, yl, ref, lim in [(axs[0], "NND_ratio_vs_real", "NND ratio (synthetic / real)", 1, (0, 1.2)), (axs[1], "MIA_AUROC", "Membership inference AUROC (DCR)", .5, (.3, .72))]:
    strip(ax, [g(m, gen, "privacy_clinical", k) for m in ("SR-DMLR", "Flat-VAE") for gen in ("gauss", "smote")], cols, pn, yl)
    ax.axhline(ref, ls="--", color=INK2, lw=1); ax.set_ylim(*lim)
axs[0].set_title("(A) Proximity to training records", loc="left", fontsize=9); axs[1].set_title("(B) Membership inference", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig("figures_final/Fig3_privacy.png", dpi=300); plt.close(fig); print("ok")
