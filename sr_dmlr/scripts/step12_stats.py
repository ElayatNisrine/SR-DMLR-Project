"""STEP 12 : statistiques finales.
- Comparaison principale (30 seeds, sens/) : difference appariee moyenne, IC95 bootstrap, Wilcoxon,
  correction de Holm sur la famille des 5 baselines ; meme chose sur les 10 premieres seeds (coherence).
- Ablations (10 seeds, final_results.json) : Holm sur la famille des 5 variantes.
- Blocs (blocks/) : moyenne et IC95 par tache et configuration ; Holm sur les comparaisons 'all' vs 'retrait du bloc'.
"""
import json, glob, numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
rng = np.random.default_rng(0)
def ci(v):
    b = rng.choice(v, (10000, len(v))).mean(1); return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
def family(ref, others):
    rows = {}; ps = []
    for k, v in others.items():
        dlt = ref - v; p = float(wilcoxon(ref, v).pvalue); ps.append(p)
        rows[k] = {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4), "diff": round(float(dlt.mean()), 4),
                   "diff_ci95": ci(dlt), "p_raw": round(p, 4), "wins": int((dlt > 0).sum()), "n": len(v)}
    adj = multipletests(ps, method="holm")[1]
    for k, a in zip(rows, adj): rows[k]["p_holm"] = round(float(a), 4)
    return rows
out = {}
S = {int(f.split("_")[-1][:-5]): json.load(open(f)) for f in glob.glob("sens/seed_*.json")}
for tag, seeds in [("30_seeds", sorted(S)), ("first_10_seeds", [s for s in sorted(S) if s < 10])]:
    g = lambda k: np.array([S[x][k] for x in seeds], float)
    ref = g("SR-DMLR")
    out[f"main_{tag}"] = {"SR-DMLR": {"mean": round(float(ref.mean()), 4), "sd": round(float(ref.std(ddof=1)), 4), "ci95": ci(ref), "n": len(ref)},
                          "baselines": family(ref, {k: g(k) for k in ("Concatenation", "SMOTE-only", "CORAL", "Unsup")}
                                              | ({"CTGAN": g("CTGAN")} if all(S[x]["CTGAN"] is not None for x in seeds) else {})),
                          "flat_vae": family(ref, {"Flat-VAE": g("Flat-VAE")}),
                          "real_features": family(ref, {"Real": g("Real")}),
                          "privacy_clinical": {m: [round(float(np.mean([S[x]["SR_priv_clin"][m] for x in seeds])), 4),
                                                   round(float(np.std([S[x]["SR_priv_clin"][m] for x in seeds], ddof=1)), 4)]
                                               for m in ("NND_ratio_vs_real", "MIA_AUROC")},
                          "confinement": {k: [round(float(g(k).mean()), 4), round(float(g(k).std(ddof=1)), 4)]
                                          for k in ("SR-DMLR_real_first8", "SR-DMLR_real_last16", "Flat-VAE_real_first8", "Flat-VAE_real_last16")}}
F = json.load(open("final_results.json")); s10 = [str(s) for s in range(10)]
fg = lambda m: np.array([F[s][m]["gauss"]["STRT"]["LR"] for s in s10])
ab = {"no_MMD": fg("no_MMD"), "no_Lsev": fg("no_Lsev"), "Flat-VAE": fg("Flat-VAE"),
      "SMOTE sampler": np.array([F[s]["SR-DMLR"]["smote"]["STRT"]["LR"] for s in s10]), "no diagnostic supervision": fg("unsup_sharedprivate")}
out["ablations_10_seeds"] = family(fg("SR-DMLR"), ab)
B = [json.load(open(f)) for f in sorted(glob.glob("blocks/seed_*.json"))]
if B:
    bl = {"n_seeds": len(B), "n_test": B[0]["n_test"]}; ps, keys = [], []
    for t in ("diagnosis", "emotion", "condition_HPE_vs_LPE", "condition_3class"):
        bl[t] = {}
        for c in B[0][t]:
            v = np.array([b[t][c] for b in B]); bl[t][c] = {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4), "ci95": ci(v)}
        bl[t]["real_reference"] = round(float(np.mean([b["real_reference"][t] for b in B])), 4)
        a = np.array([b[t]["all"] for b in B])
        for c in ("without z_c", "without z_b", "without z_x"):
            v = np.array([b[t][c] for b in B]); p = float(wilcoxon(a, v).pvalue) if np.any(a != v) else 1.0
            bl[t][c]["drop_vs_all"] = round(float((a - v).mean()), 4); bl[t][c]["drop_ci95"] = ci(a - v); bl[t][c]["p_raw"] = round(p, 4)
            ps.append(p); keys.append((t, c))
    for (t, c), q in zip(keys, multipletests(ps, method="holm")[1]): bl[t][c]["p_holm"] = round(float(q), 4)
    out["blocks"] = bl
json.dump(out, open("stats_final.json", "w"), indent=1); print(json.dumps(out, indent=1))
