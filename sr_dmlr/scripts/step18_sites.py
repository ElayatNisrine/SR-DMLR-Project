"""STEP 18 : generalisation entre sites ABIDE-I. GroupKFold (5 plis) par site, repete 3 fois avec des
attributions de sites differentes : aucun site n'est a la fois en entrainement et en test."""
import json, numpy as np, torch
import step4_revision as R, step2_pipeline as P
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
A, E, G, ZD, N = P.A, P.E, P.G, P.ZD, P.N_SYNTH
sites = A["site"]; us = np.unique(sites); out = Path("sites"); out.mkdir(exist_ok=True)
def gauss(Z, y, n, seed):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    o = [r.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k) for c, k in ((1, n1), (0, n - n1))]
    return np.r_[o[0], o[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)
lr = lambda *a: P.clf_auc(*a)["LR"]
for rep in range(3):
    perm = np.random.default_rng(rep).permutation(us); fold_of = {s_: i % 5 for i, s_ in enumerate(perm)}
    for k in range(5):
        f = out / f"rep{rep}_fold{k}.json"
        if f.exists(): continue
        s = 100 * rep + k; te = np.where([fold_of[x] == k for x in sites])[0]; tr = np.setdiff1d(np.arange(len(sites)), te)
        y, ados = A["y"], A["ados"]
        pca = PCA(64, random_state=s).fit(A["X"][tr]); sc = StandardScaler().fit(pca.transform(A["X"][tr]))
        Xtr = sc.transform(pca.transform(A["X"][tr])).astype(np.float32); Xte = sc.transform(pca.transform(A["X"][te])).astype(np.float32)
        Xe = StandardScaler().fit_transform(E["X"]).astype(np.float32); Xg = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
        d = dict(Xa_tr=Xtr, Xa_te=Xte, Xe=Xe, Xg=Xg, ytr=y[tr], yte=y[te], ados_tr=ados[tr], ados_te=ados[te])
        r = {"test_sites": sorted(set(sites[te])), "n_test": int(len(te))}
        rng = np.random.default_rng(s)
        for name, kw in (("SR-DMLR", {}), ("Flat-VAE", {"flat": True})):
            _, L = R.train(d, s, **kw)
            if kw:
                S, ys = gauss(L["a_tr"], d["ytr"], N, s)
            else:
                zc, ys = gauss(L["a_tr"][:, :ZD], d["ytr"], N, s)
                S = np.c_[zc, L["e"][rng.integers(len(L["e"]), size=N), ZD:2 * ZD], L["g"][rng.integers(len(L["g"]), size=N), 2 * ZD:]]
            r[name] = lr(S, ys, L["a_te"], d["yte"], s)
        r["Real"] = lr(Xtr, d["ytr"], Xte, d["yte"], s)
        Sm, ysm = P.smote_sample(Xtr, d["ytr"], N, s); r["SMOTE-only"] = lr(Sm, ysm, Xte, d["yte"], s)
        rr = np.random.default_rng(s + 100)
        r["Concatenation"] = lr(np.c_[Sm, Xe[rr.integers(len(Xe), size=N)], Xg[rr.integers(len(Xg), size=N)]], ysm,
                                np.c_[Xte, Xe[rr.integers(len(Xe), size=len(te))], Xg[rr.integers(len(Xg), size=len(te))]], d["yte"], s)
        f.write_text(json.dumps(r)); print(rep, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items() if a != "test_sites"}, flush=True)
