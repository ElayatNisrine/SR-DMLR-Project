"""STEP 23 : harmonisation des sites ABIDE-I avant PCA.
Methode : standardisation par site (chaque connectivite est centree et reduite avec la moyenne et l'ecart-type
de son site), sans utiliser le diagnostic. Elle est transductive : un site jamais vu en entrainement est
normalise avec ses propres participants, ce qui ne demande que des donnees non etiquetees de ce site.
Variante plus stricte : centrage seul (sans reduction).
Evaluation : (a) splits aleatoires, 30 seeds ; (b) splits groupes par site, 15 plis (identiques a step18).
Methodes : SR-DMLR (gaussien), concatenation, SMOTE-only, reference reelle ; avec et sans harmonisation."""
import os, json, numpy as np
import step4_revision as R, step2_pipeline as P
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
A, E, G, ZD, N = P.A, P.E, P.G, P.ZD, P.N_SYNTH
sites = A["site"]; y_all, ados = A["y"], A["ados"]
OUT = Path("siteharm"); OUT.mkdir(exist_ok=True)

def harmonise(X, mode):
    if mode == "none":
        return X
    Z = np.empty_like(X)
    for s_ in np.unique(sites):
        i = sites == s_; mu = X[i].mean(0)
        Z[i] = (X[i] - mu) / (X[i].std(0) + 1e-6) if mode == "zscore" else X[i] - mu
    return Z

XH = {m: harmonise(A["X"], m) for m in ("none", "center", "zscore")}

def gauss(Z, y, n, seed):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    o = [r.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k) for c, k in ((1, n1), (0, n - n1))]
    return np.r_[o[0], o[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)

def run(tr, te, s, mode):
    X = XH[mode]
    pca = PCA(64, random_state=s).fit(X[tr]); sc = StandardScaler().fit(pca.transform(X[tr]))
    Xtr = sc.transform(pca.transform(X[tr])).astype(np.float32); Xte = sc.transform(pca.transform(X[te])).astype(np.float32)
    Xe = StandardScaler().fit_transform(E["X"]).astype(np.float32); Xg = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
    d = dict(Xa_tr=Xtr, Xa_te=Xte, Xe=Xe, Xg=Xg, ytr=y_all[tr], yte=y_all[te], ados_tr=ados[tr], ados_te=ados[te])
    lr = lambda *a: P.clf_auc(*a, s)["LR"]
    _, L = R.train(d, s); rng = np.random.default_rng(s)
    zc, ys = gauss(L["a_tr"][:, :ZD], d["ytr"], N, s)
    S = np.c_[zc, L["e"][rng.integers(len(L["e"]), size=N), ZD:2 * ZD], L["g"][rng.integers(len(L["g"]), size=N), 2 * ZD:]]
    r = {"SR-DMLR": lr(S, ys, L["a_te"], d["yte"]), "Real": lr(Xtr, d["ytr"], Xte, d["yte"])}
    Sm, ysm = P.smote_sample(Xtr, d["ytr"], N, s); r["SMOTE-only"] = lr(Sm, ysm, Xte, d["yte"])
    rr = np.random.default_rng(s + 100)
    r["Concatenation"] = lr(np.c_[Sm, Xe[rr.integers(len(Xe), size=N)], Xg[rr.integers(len(Xg), size=N)]], ysm,
                            np.c_[Xte, Xe[rr.integers(len(Xe), size=len(te))], Xg[rr.integers(len(Xg), size=len(te))]], d["yte"])
    return r

jobs = []
for s in range(30):
    tr, te = train_test_split(np.arange(len(y_all)), test_size=0.3, stratify=y_all, random_state=s)
    jobs.append((f"random_s{s}", tr, te, s))
us = np.unique(sites)
for rep in range(3):
    perm = np.random.default_rng(rep).permutation(us); fold_of = {x: i % 5 for i, x in enumerate(perm)}
    for k in range(5):
        te = np.where([fold_of[x] == k for x in sites])[0]; tr = np.setdiff1d(np.arange(len(sites)), te)
        jobs.append((f"site_rep{rep}_fold{k}", tr, te, 100 * rep + k))
only = os.environ.get("ONLY")
for name, tr, te, s in jobs:
    if only and not name.startswith(only):
        continue
    f = OUT / f"{name}.json"
    if f.exists():
        continue
    res = {m: run(tr, te, s, m) for m in ("none", "center", "zscore")}
    f.write_text(json.dumps(res)); print(name, {m: round(v["SR-DMLR"], 3) for m, v in res.items()}, flush=True)
