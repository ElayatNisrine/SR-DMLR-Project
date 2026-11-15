"""STEP 19 : base unique de 30 seeds. SR-DMLR et variantes : final_results_30.json ; baselines : sens/ (memes splits) ;
sensibilite : sens16/ ; controles : controls/ ; sites : sites/."""
import json, glob, numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
rng = np.random.default_rng(0)
def ci(v): b = rng.choice(v, (10000, len(v))).mean(1); return [round(float(np.percentile(b, q)), 3) for q in (2.5, 97.5)]
def ms(v): return {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3), "ci95": ci(np.asarray(v))}
def fam(ref, d, sign=1):
    out, ps = {}, []
    for k, v in d.items():
        dl = sign * (v - ref) if sign == -1 else (v - ref); p = float(wilcoxon(ref, v).pvalue) if np.any(ref != v) else 1.0; ps.append(p)
        out[k] = {**ms(v), "diff_variant_minus_ref": round(float((v - ref).mean()), 3), "diff_ci95": ci(v - ref), "p_raw": round(p, 4),
                  "n_ref_higher": int((ref > v).sum())}
    for k, q in zip(out, multipletests(ps, method="holm")[1]): out[k]["p_holm"] = float(f"{q:.2g}")
    return out
F = json.load(open("final_results.json")); S = sorted(F, key=int); assert len(S) == 30
f = lambda m, g, *p: np.array([(lambda x: [x := x[k] for k in p][-1])(F[s][m][g]) for s in S], float)
ref = f("SR-DMLR", "gauss", "STRT", "LR")
B = {int(x.split("_")[-1][:-5]): json.load(open(x)) for x in glob.glob("sens/seed_*.json")}
b = lambda k: np.array([B[int(s)][k] for s in S], float)
out = {"n_seeds": 30, "SR-DMLR": {**ms(ref), "RF": ms(f("SR-DMLR", "gauss", "STRT", "RF"))}}
out["baselines"] = fam(ref, {"Concatenation": b("Concatenation"), "SMOTE-only": b("SMOTE-only"), "CORAL": b("CORAL"), "Unsup shared-private": b("Unsup")})
s10 = [s for s in S if int(s) < 10]; B_old = {int(x.split("_")[-1][:-5]): json.load(open(x)) for x in glob.glob("../rerun/sens/seed_*.json")}; c10 = np.array([B_old[int(s)]["CTGAN"] for s in s10]); r10 = np.array([F[s]["SR-DMLR"]["gauss"]["STRT"]["LR"] for s in s10])
out["CTGAN_10"] = {**ms(c10), "diff_SR_minus": round(float((r10 - c10).mean()), 3), "diff_ci95": ci(r10 - c10), "p": round(float(wilcoxon(r10, c10).pvalue), 4)}
out["real"] = fam(ref, {"Real features": b("Real")})
out["ablations"] = fam(ref, {"No MMD": f("no_MMD", "gauss", "STRT", "LR"), "No severity": f("no_Lsev", "gauss", "STRT", "LR"),
                             "Flat VAE": f("Flat-VAE", "gauss", "STRT", "LR"), "SMOTE sampler": f("SR-DMLR", "smote", "STRT", "LR"),
                             "No diagnostic supervision": f("unsup_sharedprivate", "gauss", "STRT", "LR")})
out["zc_only"] = ms(f("SR-DMLR", "gauss", "STRT_zc_only")); out["bx_zeroed"] = ms(f("SR-DMLR", "gauss", "STRT_bx_zeroed"))
out["bx_from_pools"] = ms(f("SR-DMLR", "gauss", "STRT_bx_from_pools"))
out["TRTR_latent"] = ms(f("SR-DMLR", "gauss", "TRTR", "LR")); out["TRTS"] = ms(f("SR-DMLR", "gauss", "TRTS", "LR")); out["STST"] = ms(f("SR-DMLR", "gauss", "STST", "LR"))
out["privacy"] = {f"{m}|{g}|{sp}": {k: ms(f(m, g, sp, k)) for k in ("NND_ratio_vs_real", "MIA_AUROC")}
                  for m in ("SR-DMLR", "Flat-VAE") for g in ("gauss", "smote") for sp in ("privacy_clinical", "privacy_24d")}
out["fidelity"] = {k: ms(f("SR-DMLR", "gauss", k)) for k in ("KS_rejected_raw", "KS_rejected_BH", "MMD_zc_synth_vs_test", "MMD_zc_train_vs_test")}
out["fidelity"]["W1"] = [ms([F[s]["SR-DMLR"]["gauss"]["W1_by_block"][i] for s in S]) for i in range(3)]
X = {int(x.split("_")[-1][:-5]): json.load(open(x)) for x in glob.glob("sens16/seed_*.json")}; x = lambda k: np.array([X[int(s)][k] for s in S], float)
out["sensitivity"] = {"base_model_retrained": ms(x("base")), **fam(x("base"), {"PCA32": x("PCA32"), "PCA128": x("PCA128"), "Engagnition 50%": x("physio50")})}
out["source_balacc"] = ms(x("zc_source_balacc")); out["severity"] = {k: ms(x(k)) for k in ("sev_MAE", "sev_rho", "ridge_MAE", "mean_MAE")}
C = [json.load(open(x)) for x in sorted(glob.glob("controls/seed_*.json"))]
out["controls"] = {k: {a: ms([c[k][a] for c in C]) for a in ("DCR", "kNN5", "DOMIAS")} for k in C[0]}
T = [json.load(open(x)) for x in sorted(glob.glob("sites/*.json"))]
tk = ("SR-DMLR", "Flat-VAE", "Concatenation", "SMOTE-only", "Real"); tv = {k: np.array([t[k] for t in T]) for k in tk}
out["sites"] = {"n_folds": len(T), "n_test_mean": round(float(np.mean([t["n_test"] for t in T])), 1),
                **{k: ms(v) for k, v in tv.items()},
                "SR_minus": {k: {"diff": round(float((tv["SR-DMLR"] - tv[k]).mean()), 3), "ci95": ci(tv["SR-DMLR"] - tv[k]),
                                 "p": round(float(wilcoxon(tv["SR-DMLR"], tv[k]).pvalue), 4), "SR_higher": int((tv["SR-DMLR"] > tv[k]).sum())}
                             for k in ("Flat-VAE", "Concatenation", "SMOTE-only", "Real")}}
json.dump(out, open("unified30.json", "w"), indent=1); print(json.dumps(out, indent=1))
