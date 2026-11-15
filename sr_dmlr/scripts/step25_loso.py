import json, numpy as np
import step4_revision as R, step2_pipeline as P
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
A, E, G, ZD, N = P.A, P.E, P.G, P.ZD, P.N_SYNTH
sites, y_all, ados = A["site"], A["y"], A["ados"]
OUT = Path("loso"); OUT.mkdir(exist_ok=True)
def harm(X, mode):
    if mode == "none": return X
    Z = X.copy()
    for u in np.unique(sites): i = sites == u; Z[i] = X[i] - X[i].mean(0)
    return Z
XH = {m: harm(A["X"], m) for m in ("none", "center")}
def gauss(Z, y, n, seed):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    o = [r.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k) for c, k in ((1, n1), (0, n - n1))]
    return np.r_[o[0], o[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)
def prob(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); c = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    return c.predict_proba(sc.transform(Xte))[:, 1].tolist()
for si, u in enumerate(np.unique(sites)):
    f = OUT / f"{u}.json"
    if f.exists(): continue
    te = np.where(sites == u)[0]; tr = np.where(sites != u)[0]; s = si; res = {"idx": te.tolist(), "y": y_all[te].tolist()}
    for mode in ("none", "center"):
        X = XH[mode]; pca = PCA(64, random_state=s).fit(X[tr]); sc = StandardScaler().fit(pca.transform(X[tr]))
        Xtr = sc.transform(pca.transform(X[tr])).astype(np.float32); Xte = sc.transform(pca.transform(X[te])).astype(np.float32)
        Xe = StandardScaler().fit_transform(E["X"]).astype(np.float32); Xg = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
        d = dict(Xa_tr=Xtr, Xa_te=Xte, Xe=Xe, Xg=Xg, ytr=y_all[tr], yte=y_all[te], ados_tr=ados[tr], ados_te=ados[te])
        r = {}; rng = np.random.default_rng(s)
        _, L = R.train(d, s); zc, ys = gauss(L["a_tr"][:, :ZD], d["ytr"], N, s)
        S = np.c_[zc, L["e"][rng.integers(len(L["e"]), size=N), ZD:2 * ZD], L["g"][rng.integers(len(L["g"]), size=N), 2 * ZD:]]
        r["SR-DMLR"] = prob(S, ys, L["a_te"])
        _, Lf = R.train(d, s, flat=True); Sf, yf = gauss(Lf["a_tr"], d["ytr"], N, s); r["Flat-VAE"] = prob(Sf, yf, Lf["a_te"])
        r["Real"] = prob(Xtr, d["ytr"], Xte)
        Sm, ysm = P.smote_sample(Xtr, d["ytr"], N, s); r["SMOTE-only"] = prob(Sm, ysm, Xte)
        rr = np.random.default_rng(s + 100)
        r["Concatenation"] = prob(np.c_[Sm, Xe[rr.integers(len(Xe), size=N)], Xg[rr.integers(len(Xg), size=N)]], ysm,
                                  np.c_[Xte, Xe[rr.integers(len(Xe), size=len(te))], Xg[rr.integers(len(Xg), size=len(te))]])
        res[mode] = r
    f.write_text(json.dumps(res)); print(u, flush=True)
