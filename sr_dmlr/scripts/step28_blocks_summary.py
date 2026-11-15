"""STEP 28 : synthese des taches par bloc (30 seeds) : moyennes, baisse apparente au retrait d'un bloc
(IC95 bootstrap sur seeds), Wilcoxon apparie, Holm sur 4 taches x 3 retraits = 12 tests.
Usage : python step28_blocks_summary.py <dossier_seeds> <sortie.json>"""
import sys, json, numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
src, out = sys.argv[1], sys.argv[2]
D = [json.load(open(f"{src}/seed_{s}.json")) for s in range(30)]
T = ["diagnosis", "emotion", "condition_HPE_vs_LPE", "condition_3class"]
C = ["all", "z_c only", "z_b only", "z_x only", "without z_c", "without z_b", "without z_x"]
rng = np.random.default_rng(0); S = {"n_test": D[0]["n_test"]}; P = []
for t in T:
    S[t] = {c: round(float(np.mean([d[t][c] for d in D])), 3) for c in C}
    S[t]["ref"] = round(float(np.mean([d["real_reference"][t] for d in D])), 3)
    for b in ["z_c", "z_b", "z_x"]:
        x = np.array([d[t]["all"] - d[t][f"without {b}"] for d in D])
        bs = x[rng.integers(0, 30, (10000, 30))].mean(1)
        S[t][f"without {b}_drop"] = [round(float(x.mean()), 3), [round(float(v), 3) for v in np.percentile(bs, [2.5, 97.5])]]
        P.append((t, b, wilcoxon(x).pvalue if np.any(x != 0) else 1.0))
h = multipletests([p for *_, p in P], method="holm")[1]
for (t, b, _), q in zip(P, h): S[t][f"without {b}_pholm"] = float(f"{q:.2g}")
json.dump(S, open(out, "w"), indent=1); print(json.dumps({t: {k: S[t][k] for k in ['all','z_b only','z_x only','ref']} for t in T}, indent=0))
