"""STEP 20 : le masquage vient-il de la dimension des blocs ajoutes ou de leur origine ?
Pour chaque seed (30), cohorte SMOTE (cas de fuite), attaques DCR / kNN-5 / DOMIAS sur :
  clinical          : z_c seul (8 d)
  full_raw          : profil complet tel quel (24 d)
  full_std          : profil complet, chaque dimension standardisee sur les donnees reelles d'entrainement
  full_blockeq      : standardisation + poids 1/sqrt(dim du bloc) : chaque bloc pese autant
  full_w0.5/0.25    : poids des blocs z_b, z_x reduit (apres standardisation)
  same_origin       : MEME dimension (24 d) mais z_b, z_x du meme participant ABIDE (SMOTE sur le latent
                      complet ABIDE : aucun bloc d'une autre population). Si la fuite reste visible ici sur
                      24 d, le masquage tient a l'origine des blocs et non a la dimension.
Protocole membres / non-membres / reference identique a step13."""
import os, json, numpy as np
import step4_revision as R, step2_pipeline as P
from step13_audit import attacks
from pathlib import Path
ZD, N = P.ZD, P.N_SYNTH
OUT = Path("dilution"); OUT.mkdir(exist_ok=True)
for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
    f = OUT / f"seed_{s}.json"
    if f.exists():
        continue
    d = R.prep(s); ytr = d["ytr"]; _, L = R.train(d, s)
    rng = np.random.default_rng(s + 50); te = rng.permutation(len(d["yte"]))
    nm, ref = te[: len(te) // 2], te[len(te) // 2:]
    mem = rng.choice(len(ytr), len(nm), replace=False)
    r2 = np.random.default_rng(s); ib = r2.integers(len(L["e"]), size=N); ix = r2.integers(len(L["g"]), size=N)
    zc, _ = P.smote_sample(L["a_tr"][:, :ZD], ytr, N, s)
    S = np.c_[zc, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
    Tr, Te = L["a_tr"], L["a_te"]; mu, sd = Tr.mean(0), Tr.std(0) + 1e-8

    def run(W, SS=S):
        g = lambda X: (X - mu) / sd * W
        return attacks(g(SS), g(Tr[mem]), g(Te[nm]), g(Te[ref]))

    res = {"clinical": attacks(S[:, :ZD], Tr[mem, :ZD], Te[nm, :ZD], Te[ref, :ZD]),
           "full_raw": attacks(S, Tr[mem], Te[nm], Te[ref]),
           "full_std": run(np.ones(3 * ZD)),
           "full_blockeq": run(np.r_[np.full(ZD, 1 / np.sqrt(ZD)), np.full(2 * ZD, 1 / np.sqrt(2 * ZD))])}
    for w in (0.5, 0.25):
        res[f"full_w{w}"] = run(np.r_[np.ones(ZD), np.full(2 * ZD, w)])
    S_same, _ = P.smote_sample(L["a_tr"], ytr, N, s)
    res["same_origin"] = attacks(S_same, Tr[mem], Te[nm], Te[ref])
    f.write_text(json.dumps(res))
    print(s, {k: round(v["DCR"], 3) for k, v in res.items()}, flush=True)
