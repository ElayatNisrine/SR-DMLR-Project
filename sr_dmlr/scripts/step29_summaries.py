"""STEP 29 : syntheses des analyses dependant du sampler SMOTE (dilution, audit bilateral, harmonisation des sites,
VAE plat et LOSO). Moyenne, IC95 bootstrap sur seeds/plis (10 000), Wilcoxon apparie, Holm la ou indique.
Usage : python step29_summaries.py <dossier> ; ecrit dilution_summary.json, audit_twosided.json,
siteharm_summary.json, step24_25_summary.json dans <dossier>."""
import sys, os, json, glob, numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
os.chdir(sys.argv[1]); R = lambda: np.random.default_rng(0)
def ci(x, B=10000):
    x = np.asarray(x, float); b = x[R().integers(0, len(x), (B, len(x)))].mean(1); return [round(float(v), 3) for v in np.percentile(b, [2.5, 97.5])]
def mci(x): return [round(float(np.mean(x)), 3), ci(x)]
def wp(x):
    x = np.asarray(x, float); return float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
A3 = ("DCR", "kNN5", "DOMIAS")
# dilution
D = [json.load(open(f)) for f in sorted(glob.glob("dilution/seed_*.json"))]
json.dump({k: {a: mci([d[k][a] for d in D]) for a in A3} for k in D[0]}, open("dilution_summary.json", "w"), indent=1)
# audit bilateral
Au = [json.load(open(f)) for f in sorted(glob.glob("audit/seed_*.json"))]; out = {}
for g in ("smote", "gauss"):
    for sp in ("clinical", "full"):
        for a in A3:
            x = np.array([d[f"{g}|{sp}"][a] for d in Au]); m = float(x.mean())
            out[f"{g}|{sp}|{a}"] = {"mean": round(m, 3), "ci": ci(x), "p_vs_0.5_two_sided": float(f"{wp(x - 0.5):.2g}"),
                                    "dev_from_chance": round(abs(m - 0.5), 3), "reversed_auroc": round(1 - m, 3)}
    for a in A3:
        x = np.array([abs(d[f"{g}|clinical"][a] - .5) - abs(d[f"{g}|full"][a] - .5) for d in Au])
        out[f"{g}|absdev_clin_minus_full|{a}"] = {"diff": round(float(x.mean()), 3), "ci": ci(x), "p": float(f"{wp(x):.2g}")}
json.dump(out, open("audit_twosided.json", "w"), indent=1)
# harmonisation des sites
M = ("SR-DMLR", "Real", "SMOTE-only", "Concatenation"); sh = {}
for kind in ("random", "site"):
    F = [json.load(open(f)) for f in sorted(glob.glob(f"siteharm/{kind}_*.json"))]; sh[kind] = {"n": len(F)}
    for h in ("none", "center", "zscore"):
        r = {m: round(float(np.mean([f[h][m] for f in F])), 3) for m in M}
        for m in ("SMOTE-only", "Concatenation", "Real"):
            x = [f[h]["SR-DMLR"] - f[h][m] for f in F]; r[f"SR-{m}"] = [round(float(np.mean(x)), 3), ci(x), round(wp(x), 4)]
        if h != "none":
            x = [f[h]["SR-DMLR"] - f["none"]["SR-DMLR"] for f in F]; r["SR_vs_none"] = [round(float(np.mean(x)), 3), ci(x), round(wp(x), 4)]
        sh[kind][h] = r
json.dump(sh, open("siteharm_summary.json", "w"), indent=1)
# VAE plat + LOSO
FA = [json.load(open(f)) for f in sorted(glob.glob("flat_audit/seed_*.json"))]; s = {"n_flat": len(FA)}
for g in ("gauss", "smote"):
    for a in A3:
        x = [f[g][a] for f in FA]; s[f"flat|{g}|{a}"] = [round(float(np.mean(x)), 3), *ci(x)]
        for sp in ("clinical", "full"):
            x = [d[f"{g}|{sp}"][a] for d in Au]; s[f"SR|{g}|{sp}|{a}"] = [round(float(np.mean(x)), 3), *ci(x)]
L = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob("loso/*.json"))}
ML = ("SR-DMLR", "Flat-VAE", "SMOTE-only", "Concatenation", "Real")
def pooled_auc(h, m, idx):
    y = np.concatenate([L[k]["y"] for k in L]); p = np.concatenate([L[k][h][m] for k in L]); return roc_auc_score(y[idx], p[idx])
for h in ("none", "center"):
    y = np.concatenate([L[k]["y"] for k in L]); n = len(y); B = R().integers(0, n, (2000, n))
    for m in ML:
        site = [roc_auc_score(L[k]["y"], L[k][h][m]) for k in L]
        boot = [pooled_auc(h, m, b) for b in B]
        s[f"loso|{h}|{m}"] = {"site_mean": round(float(np.mean(site)), 3), "site_median": round(float(np.median(site)), 3),
                              "pooled": [round(pooled_auc(h, m, np.arange(n)), 3), *[round(float(v), 3) for v in np.percentile(boot, [2.5, 97.5])]]}
cmps = []
for h in ("none", "center"):
    y = np.concatenate([L[k]["y"] for k in L]); n = len(y); B = R().integers(0, n, (2000, n))
    for m in ("SMOTE-only", "Concatenation", "Flat-VAE"):
        dlt = [roc_auc_score(L[k]["y"], L[k][h]["SR-DMLR"]) - roc_auc_score(L[k]["y"], L[k][h][m]) for k in L]
        pb = [pooled_auc(h, "SR-DMLR", b) - pooled_auc(h, m, b) for b in B]
        key = f"cmp|{h}|SR-DMLR-{m}"
        s[key] = {"mean_site_delta": round(float(np.mean(dlt)), 3), "wins": int(sum(v > 0 for v in dlt)), "p": wp(dlt),
                  "pooled_delta": [round(pooled_auc(h, "SR-DMLR", np.arange(n)) - pooled_auc(h, m, np.arange(n)), 3), *[round(float(v), 3) for v in np.percentile(pb, [2.5, 97.5])]]}
        cmps.append(key)
ph = multipletests([s[k]["p"] for k in cmps], method="holm")[1]
for k, q in zip(cmps, ph): s[k]["p_holm"] = round(float(q), 4)
json.dump(s, open("step24_25_summary.json", "w"), indent=1); print("ok")
