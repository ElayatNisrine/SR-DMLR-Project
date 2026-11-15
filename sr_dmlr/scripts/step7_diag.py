"""STEP 7 : ou se perd l'utilite entre donnees reelles (0.731) et cohorte synthetique (0.687) ?"""
import json, numpy as np
import step4_revision as R, step2_pipeline as P
from sklearn.covariance import LedoitWolf
ZD = P.ZD
def gauss(Z, y, n, seed, shrink=False):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean())); out = []
    for c, k in ((1, n1), (0, n - n1)):
        cov = LedoitWolf().fit(Z[y == c]).covariance_ if shrink else np.cov(Z[y == c].T)
        out.append(r.multivariate_normal(Z[y == c].mean(0), cov, k))
    return np.r_[out[0], out[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)
res = {}
for s in range(10):
    d = R.prep(s); ytr, yte = d["ytr"], d["yte"]; a = lambda *x: P.clf_auc(*x, s)["LR"]
    r = {"real_PCA64_TRTR": a(d["Xa_tr"], ytr, d["Xa_te"], yte)}
    S, ys = gauss(d["Xa_tr"], ytr, 250, s, True); r["gauss_PCA64_noVAE_N250"] = a(S, ys, d["Xa_te"], yte)
    S, ys = gauss(d["Xa_tr"], ytr, 2000, s, True); r["gauss_PCA64_noVAE_N2000"] = a(S, ys, d["Xa_te"], yte)
    for name, kw in [("SR", {}), ("Flat", {"flat": True})]:
        _, L = R.train(d, s, **kw); k = slice(None) if kw else slice(0, ZD)
        Ztr, Zte = L["a_tr"][:, k], L["a_te"][:, k]
        r[f"{name}_latent_TRTR"] = a(Ztr, ytr, Zte, yte)
        for n in (250, 2000):
            S, ys = gauss(Ztr, ytr, n, s); r[f"{name}_gauss_N{n}"] = a(S, ys, Zte, yte)
    res[s] = r; print(s, {k: round(v, 3) for k, v in r.items()}, flush=True)
agg = {k: [round(float(np.mean([res[s][k] for s in res])), 4), round(float(np.std([res[s][k] for s in res], ddof=1)), 4)] for k in res[0]}
json.dump({"per_seed": res, "summary": agg}, open("diag_results.json", "w"), indent=1)
print(json.dumps(agg, indent=1))
