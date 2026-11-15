"""Figure 7 : LOSO sans harmonisation, difference d'AUROC par site (SR-DMLR moins SMOTE-only / concatenation)."""
import json, glob, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
L = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("loso/*.json")}
a = lambda k, m: roc_auc_score(L[k]["y"], L[k]["none"][m])
rows = sorted(((k, a(k, "SR-DMLR") - a(k, "SMOTE-only"), a(k, "SR-DMLR") - a(k, "Concatenation"), len(L[k]["y"])) for k in L), key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(3.9, 4.2), dpi=300); y = np.arange(len(rows))
ax.scatter([r[1] for r in rows], y, s=[r[3] * 0.9 + 8 for r in rows], color="#2c6e99", label="vs SMOTE", zorder=3)
ax.scatter([r[2] for r in rows], y, s=[r[3] * 0.9 + 8 for r in rows], facecolors="none", edgecolors="#c0392b", linewidths=1, label="vs concatenation", zorder=3)
ax.axvline(0, color="black", lw=0.6); ax.set_yticks(y); ax.set_yticklabels([f"{r[0]} (n={r[3]})" for r in rows], fontsize=6.5)
ax.set_xlabel("AUROC difference,\nSR-DMLR minus baseline", fontsize=8); ax.tick_params(axis="x", labelsize=8)
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("figures_final/Fig7_loso_sites.png", dpi=300); print("ok")
