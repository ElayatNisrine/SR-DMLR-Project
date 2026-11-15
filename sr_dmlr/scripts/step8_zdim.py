"""STEP 8 : taille du latent clinique (ZD = 8, 16, 32), SR-DMLR et Flat-VAE.
Selection sur validation interne (20 % du train), test ABIDE jamais vu pour la selection."""
import json, numpy as np
import step4_revision as R, step2_pipeline as P
from sklearn.model_selection import train_test_split
def gauss(Z, y, n, seed):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    out = [r.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k) for c, k in ((1, n1), (0, n - n1))]
    return np.r_[out[0], out[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)
def run(d, s, zd, flat, Xtr, ytr, ados, Xev, yev):
    P.ZD = R.ZD = zd
    dd = dict(d, Xa_tr=Xtr, ytr=ytr, ados_tr=ados, Xa_te=Xev)
    _, L = R.train(dd, s, **({"flat": True} if flat else {}))
    k = slice(None) if flat else slice(0, zd)
    Zt, Ze = L["a_tr"][:, k], L["a_te"][:, k]
    S, ys = gauss(Zt, ytr, 250, s)
    zc = S if flat else S[:, :zd]
    pr = P.privacy(zc, Zt, Ze)
    return {"STRT": P.clf_auc(S, ys, Ze, yev, s)["LR"], "MIA": pr["MIA_AUROC"], "NND": pr["NND_ratio_vs_real"]}
res = {}
for s in range(10):
    d = R.prep(s); r = {}
    itr, iva = train_test_split(np.arange(len(d["ytr"])), test_size=0.2, stratify=d["ytr"], random_state=s)
    for zd in (8, 16, 32):
        for flat in (False, True):
            name = f"{'Flat' if flat else 'SR'}_ZD{zd}"
            r[name] = {"val": run(d, s, zd, flat, d["Xa_tr"][itr], d["ytr"][itr], d["ados_tr"][itr], d["Xa_tr"][iva], d["ytr"][iva]),
                       "test": run(d, s, zd, flat, d["Xa_tr"], d["ytr"], d["ados_tr"], d["Xa_te"], d["yte"])}
    res[s] = r; print(s, {k: (round(v["val"]["STRT"], 3), round(v["test"]["STRT"], 3)) for k, v in r.items()}, flush=True)
    json.dump(res, open("zdim_results.json", "w"), indent=1)
agg = {k: {p: {m: round(float(np.mean([res[s][k][p][m] for s in res])), 4) for m in ("STRT", "MIA", "NND")}
           for p in ("val", "test")} for k in res[0]}
json.dump({"per_seed": res, "summary": agg}, open("zdim_results.json", "w"), indent=1)
print(json.dumps(agg, indent=1))
