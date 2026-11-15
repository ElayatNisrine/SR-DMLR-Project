"""
STEP 5 : la protection vient-elle du bloc clinique ou seulement des blocs tires d'autres pools ?
Confidentialite (NND ratio, MIA) calculee sur z_c seul, pour SR-DMLR et ses variantes.
"""
import os, json
import numpy as np
import step4_revision as R, step2_pipeline as P

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
ZD, N = P.ZD, P.N_SYNTH
out = {}
for seed in SEEDS:
    d = R.prep(seed); ytr = d["ytr"]
    _, L = R.train(d, seed)
    zc_tr, zc_te = L["a_tr"][:, :ZD], L["a_te"][:, :ZD]
    S, ys = R.synth(L, ytr, seed)                               # SR-DMLR (SMOTE dans z_c)
    rng = np.random.default_rng(seed + 3); n1 = int(round(N * ytr.mean()))
    zc_g = np.r_[rng.multivariate_normal(zc_tr[ytr == 1].mean(0), np.cov(zc_tr[ytr == 1].T), n1),
                 rng.multivariate_normal(zc_tr[ytr == 0].mean(0), np.cov(zc_tr[ytr == 0].T), N - n1)]
    S_full_intra, _ = P.smote_sample(L["a_tr"], ytr, N, seed)   # tous les blocs du meme sujet
    out[seed] = {
        "SRDMLR_full24": P.privacy(S, L["a_tr"], L["a_te"]),
        "SRDMLR_zc_only": P.privacy(S[:, :ZD], zc_tr, zc_te),
        "Gaussian_zc_only": P.privacy(zc_g, zc_tr, zc_te),
        "NoInterSubject_zc_only": P.privacy(S_full_intra[:, :ZD], zc_tr, zc_te),
        "STRT_gaussian_zc": P.clf_auc(zc_g, np.r_[np.ones(n1), np.zeros(N - n1)].astype(int),
                                      zc_te, d["yte"], seed)["LR"],
    }
    print(seed, {k: (round(v["NND_ratio_vs_real"], 3), round(v["MIA_AUROC"], 3)) if isinstance(v, dict) else round(v, 3)
                 for k, v in out[seed].items()}, flush=True)
json.dump(out, open("privacy_zc.json", "w"), indent=1)
agg = {}
for k in out[SEEDS[0]]:
    if isinstance(out[SEEDS[0]][k], dict):
        agg[k] = {m: [round(float(np.mean([out[s][k][m] for s in SEEDS])), 4),
                      round(float(np.std([out[s][k][m] for s in SEEDS], ddof=1)), 4)]
                  for m in ("NND_ratio_vs_real", "MIA_AUROC")}
    else:
        agg[k] = [round(float(np.mean([out[s][k] for s in SEEDS])), 4),
                  round(float(np.std([out[s][k] for s in SEEDS], ddof=1)), 4)]
json.dump(agg, open("privacy_zc_summary.json", "w"), indent=1)
print(json.dumps(agg, indent=1))
