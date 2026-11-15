"""STEP 13 : robustesse de l'audit par bloc contre l'audit du profil complet (30 seeds).

Protocole d'attaque (aucun modele d'attaque n'est entraine sur les membres) :
  membres      : tirage aleatoire de participants ABIDE-I d'entrainement (meme taille que les non-membres)
  non-membres  : moitie A du test ABIDE-I (jamais vue par l'encodeur ni par le generateur)
  reference    : moitie B du test ABIDE-I, donnee a l'attaquant DOMIAS comme echantillon de la population ;
                 elle n'est jamais evaluee.
  attaques     : DCR (distance au plus proche synthetique), kNN-5 (distance moyenne aux 5 plus proches),
                 DOMIAS (van Breugel et al. 2023) : log p_synth(x) - log p_ref(x), densites par KDE gaussien.
  espaces      : bloc clinique z_c (8 d) et profil complet (24 d).
  generateurs  : gaussien par classe et SMOTE, sur le meme modele entraine.
Sortie : audit/seed_<s>.json
"""
import os, json, numpy as np
import step4_revision as R, step2_pipeline as P
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import roc_auc_score
from pathlib import Path
ZD, N = P.ZD, P.N_SYNTH
OUT = Path("audit"); OUT.mkdir(exist_ok=True)
def gauss(Z, y, n, seed):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    o = [r.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k) for c, k in ((1, n1), (0, n - n1))]
    return np.r_[o[0], o[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)
def kde(X):
    bw = X.std(0).mean() * len(X) ** (-1 / (X.shape[1] + 4))       # regle de Scott
    return KernelDensity(bandwidth=bw).fit(X)
def attacks(S, M, NM, REF):
    lab = np.r_[np.ones(len(M)), np.zeros(len(NM))]; Q = np.r_[M, NM]
    nn = NearestNeighbors(n_neighbors=5).fit(S); d = nn.kneighbors(Q)[0]
    dom = kde(S).score_samples(Q) - kde(REF).score_samples(Q)
    return {"DCR": float(roc_auc_score(lab, -d[:, 0])), "kNN5": float(roc_auc_score(lab, -d.mean(1))),
            "DOMIAS": float(roc_auc_score(lab, dom))}
for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
    f = OUT / f"seed_{s}.json"
    if f.exists(): continue
    d = R.prep(s); ytr = d["ytr"]; _, L = R.train(d, s)
    rng = np.random.default_rng(s + 50)
    te = rng.permutation(len(d["yte"])); nm, ref = te[: len(te) // 2], te[len(te) // 2:]
    mem = rng.choice(len(ytr), len(nm), replace=False)
    r2 = np.random.default_rng(s); ib = r2.integers(len(L["e"]), size=N); ix = r2.integers(len(L["g"]), size=N)
    res = {}
    for gen in ("gauss", "smote"):
        zc, _ = gauss(L["a_tr"][:, :ZD], ytr, N, s) if gen == "gauss" else P.smote_sample(L["a_tr"][:, :ZD], ytr, N, s)
        S = np.c_[zc, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
        for space, k in (("clinical", slice(0, ZD)), ("full", slice(None))):
            Tr, Te = L["a_tr"][:, k], L["a_te"][:, k]
            res[f"{gen}|{space}"] = {**attacks(S[:, k], Tr[mem], Te[nm], Te[ref]),
                                     "NND_ratio": P.privacy(S[:, k], Tr, Te)["NND_ratio_vs_real"]}
        # mecanisme : part de la distance au plus proche synthetique (profil complet) due a z_b et z_x
        Q = L["a_tr"][mem]; j = NearestNeighbors(n_neighbors=1).fit(S).kneighbors(Q)[1][:, 0]
        diff2 = (Q - S[j]) ** 2
        res[f"{gen}|share_bx"] = float((diff2[:, ZD:].sum(1) / diff2.sum(1)).mean())
    f.write_text(json.dumps(res)); print(s, {k: (round(v["DCR"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in res.items()}, flush=True)
