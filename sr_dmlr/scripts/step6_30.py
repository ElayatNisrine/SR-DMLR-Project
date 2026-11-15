"""
STEP 6 : evaluation finale avec generateur gaussien par classe (choix retenu apres
constat de la fuite de SMOTE sur le bloc clinique, cf. step5).

Pour chaque seed et chaque modele (full, no_MMD, no_Lsev, Flat-VAE, unsup) :
  generateurs : 'gauss' (principal) et 'smote' (ablation)
  STRT LR/RF, TRTS, STST
  confidentialite sur le bloc clinique (z_c ; latent complet pour Flat-VAE) ET sur 24 dims
  fidelite (KS+BH, W1, MMD) pour le modele principal
Les baselines sans latent (SMOTE-only, Concatenation, CORAL, CTGAN) proviennent de
step2/step4 (memes splits).
"""
import os, json, time
import numpy as np, torch
import step4_revision as R, step2_pipeline as P
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.stats.multitest import multipletests

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
ZD, N = P.ZD, P.N_SYNTH


def gauss_sample(Z, y, n, seed):
    rng = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    out = [rng.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k)
           for c, k in ((1, n1), (0, n - n1))]
    return np.r_[out[0], out[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)


def evaluate(L, d, seed, gen, flat=False):
    ytr, yte = d["ytr"], d["yte"]
    rng = np.random.default_rng(seed)
    if flat:
        S, ys = (gauss_sample if gen == "gauss" else P.smote_sample)(L["a_tr"], ytr, N, seed)
        clin_S, clin_tr, clin_te = S, L["a_tr"], L["a_te"]
    else:
        zc, ys = (gauss_sample if gen == "gauss" else P.smote_sample)(L["a_tr"][:, :ZD], ytr, N, seed)
        S = np.c_[zc, L["e"][rng.integers(len(L["e"]), size=N), ZD:2 * ZD],
                  L["g"][rng.integers(len(L["g"]), size=N), 2 * ZD:]]
        clin_S, clin_tr, clin_te = zc, L["a_tr"][:, :ZD], L["a_te"][:, :ZD]
    r = {"STRT": P.clf_auc(S, ys, L["a_te"], yte, seed),
         "TRTR": P.clf_auc(L["a_tr"], ytr, L["a_te"], yte, seed),
         "TRTS": P.clf_auc(L["a_tr"], ytr, S, ys, seed),
         "privacy_clinical": P.privacy(clin_S, clin_tr, clin_te),
         "privacy_24d": P.privacy(S, L["a_tr"], L["a_te"])}
    st = [P.clf_auc(S[a], ys[a], S[b], ys[b], seed)
          for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(S, ys)]
    r["STST"] = {c: float(np.mean([x[c] for x in st])) for c in ("LR", "RF")}
    if not flat:
        r["STRT_zc_only"] = P.clf_auc(S[:, :ZD], ys, L["a_te"][:, :ZD], yte, seed)["LR"]
        refs = [L["a_tr"][:, j] for j in range(ZD)] + [L["e"][:, j] for j in range(ZD, 2 * ZD)] \
             + [L["g"][:, j] for j in range(2 * ZD, 3 * ZD)]
        pv = np.array([ks_2samp(S[:, j], refs[j]).pvalue for j in range(3 * ZD)])
        r["KS_rejected_raw"] = int((pv < 0.05).sum())
        r["KS_rejected_BH"] = int(multipletests(pv, 0.05, "fdr_bh")[0].sum())
        r["W1_by_block"] = [float(np.mean([wasserstein_distance(S[:, j], refs[j]) for j in rr]))
                            for rr in (range(ZD), range(ZD, 2 * ZD), range(2 * ZD, 3 * ZD))]
        r["MMD_zc_synth_vs_test"] = R.mmd_np(zc, L["a_te"][:, :ZD])
        r["MMD_zc_train_vs_test"] = R.mmd_np(L["a_tr"][:, :ZD], L["a_te"][:, :ZD])
        rr = np.random.default_rng(seed + 11)
        Te_pool = np.c_[L["a_te"][:, :ZD], L["e"][rr.integers(len(L["e"]), size=len(yte)), ZD:2 * ZD],
                        L["g"][rr.integers(len(L["g"]), size=len(yte)), 2 * ZD:]]
        r["STRT_bx_zeroed"] = P.clf_auc(np.c_[zc, np.zeros((N, 2 * ZD))], ys,
                                        np.c_[L["a_te"][:, :ZD], np.zeros((len(yte), 2 * ZD))], yte, seed)["LR"]
        r["STRT_bx_from_pools"] = P.clf_auc(S, ys, Te_pool, yte, seed)["LR"]
    return r


OUTF = os.environ.get("OUTF", "final_results.json")
runs = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
for seed in SEEDS:
    if str(seed) in runs: continue
    t0 = time.time(); d = R.prep(seed); res = {}
    for name, kw in [("SR-DMLR", {}), ("no_MMD", {"lam_mmd": 0.0}), ("no_Lsev", {"lam_sev": 0.0}),
                     ("Flat-VAE", {"flat": True}), ("unsup_sharedprivate", {"supervised": False})]:
        _, L = R.train(d, seed, **kw)
        res[name] = {g: evaluate(L, d, seed, g, flat=kw.get("flat", False)) for g in ("gauss", "smote")}
    res["seconds"] = round(time.time() - t0, 1)
    runs[str(seed)] = res
    print(f"seed {seed} ok ({res['seconds']} s) gauss STRT={res['SR-DMLR']['gauss']['STRT']['LR']:.3f} "
          f"clinMIA={res['SR-DMLR']['gauss']['privacy_clinical']['MIA_AUROC']:.3f}", flush=True)
    json.dump(runs, open(OUTF, "w"), indent=1)
print("done")
