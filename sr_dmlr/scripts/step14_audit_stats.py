"""STEP 14 : synthese de l'audit (30 seeds) : moyennes, IC95 bootstrap, difference appariee bloc - profil,
Wilcoxon avec Holm sur la famille des comparaisons bloc vs profil."""
import json, glob, numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
A = [json.load(open(f)) for f in sorted(glob.glob("audit/seed_*.json"))]
rng = np.random.default_rng(0)
def ci(v): b = rng.choice(v, (10000, len(v))).mean(1); return [round(float(np.percentile(b, q)), 3) for q in (2.5, 97.5)]
def g(key, m): return np.array([a[key][m] for a in A])
out = {"n_seeds": len(A), "metrics": {}, "block_vs_full": {}}
for gen in ("gauss", "smote"):
    for sp in ("clinical", "full"):
        out["metrics"][f"{gen}|{sp}"] = {m: {"mean": round(float(g(f"{gen}|{sp}", m).mean()), 3), "ci95": ci(g(f"{gen}|{sp}", m))}
                                       for m in ("NND_ratio", "DCR", "kNN5", "DOMIAS")}
    out["metrics"][f"{gen}|share_bx"] = {"mean": round(float(np.mean([a[f"{gen}|share_bx"] for a in A])), 3),
                                         "ci95": ci(np.array([a[f"{gen}|share_bx"] for a in A]))}
ps, keys = [], []
for gen in ("gauss", "smote"):
    for m in ("NND_ratio", "DCR", "kNN5", "DOMIAS"):
        c, f = g(f"{gen}|clinical", m), g(f"{gen}|full", m); dlt = c - f
        p = float(wilcoxon(c, f).pvalue); ps.append(p); keys.append((gen, m))
        out["block_vs_full"][f"{gen}|{m}"] = {"diff_clinical_minus_full": round(float(dlt.mean()), 3), "ci95": ci(dlt),
                                              "p_raw": float(f"{p:.2e}"), "seeds_clinical_worse": int((dlt < 0).sum() if m == "NND_ratio" else (dlt > 0).sum())}
for (gen, m), q in zip(keys, multipletests(ps, method="holm")[1]): out["block_vs_full"][f"{gen}|{m}"]["p_holm"] = float(f"{q:.2e}")
# SMOTE vs gaussien, bloc clinique
for m in ("NND_ratio", "DCR", "kNN5", "DOMIAS"):
    a, b = g("smote|clinical", m), g("gauss|clinical", m)
    out[f"smote_minus_gauss_clinical|{m}"] = {"diff": round(float((a - b).mean()), 3), "ci95": ci(a - b), "p": float(f"{wilcoxon(a, b).pvalue:.2e}")}
json.dump(out, open("audit_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
