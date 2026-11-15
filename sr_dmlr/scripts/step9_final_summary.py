"""STEP 9 : tableau final (generateur gaussien) + tests de Wilcoxon apparies, 10 seeds."""
import json, numpy as np
from scipy.stats import wilcoxon
F = json.load(open("final_results.json")); B = json.load(open("results.json"))["per_seed"]
V = json.load(open("revision_results.json")); Z = json.load(open("zdim_results.json"))["per_seed"]
S = [str(s) for s in range(10)]
def f(m, g, *p):
    out = []
    for s in S:
        x = F[s][m][g]
        for k in p: x = x[k]
        out.append(x)
    return np.array(out, float)
ref = f("SR-DMLR", "gauss", "STRT", "LR")
bs = json.load(open("results.json"))["summary"]
rows = {
 "SR-DMLR": ref,
 "SR-DMLR (RF)": f("SR-DMLR", "gauss", "STRT", "RF"),
 "no_MMD": f("no_MMD", "gauss", "STRT", "LR"), "no_Lsev": f("no_Lsev", "gauss", "STRT", "LR"),
 "Flat-VAE": f("Flat-VAE", "gauss", "STRT", "LR"),
 "SR-DMLR SMOTE generator": f("SR-DMLR", "smote", "STRT", "LR"),
 "unsup shared-private": f("unsup_sharedprivate", "gauss", "STRT", "LR"),
 "zc only": f("SR-DMLR", "gauss", "STRT_zc_only"),
 "zb zx zeroed": f("SR-DMLR", "gauss", "STRT_bx_zeroed"),
 "SMOTE-only": np.array(bs["SMOTE-only_STRT"]["LR"]["per_seed"]),
 "Concatenation": np.array(bs["Concatenation_STRT"]["LR"]["per_seed"]),
 "CORAL": np.array([V[s]["base_CORAL"]["STRT"] for s in S]),
 "CTGAN": np.array([V[s]["base_CTGAN"]["STRT"] for s in S]),
 "Real PCA64 (TRTR)": np.array(bs["Brain-only_real(TRTR)"]["LR"]["per_seed"]),
}
out = {"STRT": {}}
for k, v in rows.items():
    e = {"mean": round(v.mean(), 3), "sd": round(v.std(ddof=1), 3)}
    if k != "SR-DMLR" and "RF" not in k:
        e["diff"] = round((ref - v).mean(), 3); e["p"] = round(float(wilcoxon(ref, v).pvalue), 4)
        e["wins"] = int((ref > v).sum())
    out["STRT"][k] = e
pv = lambda m, g, b, k: (round(f(m, g, b, k).mean(), 3), round(f(m, g, b, k).std(ddof=1), 3))
out["privacy"] = {f"{m}|{g}|{b}": {k: pv(m, g, b, k) for k in ("NND_ratio_vs_real", "MIA_AUROC")}
                  for m in ("SR-DMLR", "Flat-VAE") for g in ("gauss", "smote") for b in ("privacy_clinical", "privacy_24d")}
out["fidelity"] = {k: round(float(f("SR-DMLR", "gauss", k).mean()), 4) for k in ("KS_rejected_raw", "KS_rejected_BH", "MMD_zc_synth_vs_test", "MMD_zc_train_vs_test")}
out["fidelity"]["W1"] = np.round(np.mean([F[s]["SR-DMLR"]["gauss"]["W1_by_block"] for s in S], 0), 3).tolist()
out["TRTS"] = round(f("SR-DMLR", "gauss", "TRTS", "LR").mean(), 3); out["STST"] = round(f("SR-DMLR", "gauss", "STST", "LR").mean(), 3)
out["TRTR_latent"] = round(f("SR-DMLR", "gauss", "TRTR", "LR").mean(), 3)
out["zdim"] = {k: {p: {m: round(float(np.mean([Z[s][k][p][m] for s in S])), 3) for m in ("STRT", "MIA")} for p in ("val", "test")} for k in Z["0"]}
out["zdim_p_SR8_vs_Flat8_val"] = round(float(wilcoxon([Z[s]["SR_ZD8"]["val"]["STRT"] for s in S], [Z[s]["Flat_ZD8"]["val"]["STRT"] for s in S]).pvalue), 4)
json.dump(out, open("final_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
