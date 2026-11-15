"""STEP 10 : sensibilite au nombre de seeds (30) pour la comparaison principale + confinement du diagnostic.
Memes splits et memes definitions que step2/step4/step6. Sortie : sens/seed_<s>.json"""
import os, json, time, numpy as np, torch
import step4_revision as R, step2_pipeline as P
from sklearn.decomposition import PCA
from pathlib import Path
ZD, N = P.ZD, P.N_SYNTH
OUT = Path("sens"); OUT.mkdir(exist_ok=True)
def gauss(Z, y, n, seed):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    o = [r.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k) for c, k in ((1, n1), (0, n - n1))]
    return np.r_[o[0], o[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)
def lr(*a): return P.clf_auc(*a)["LR"]
for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
    f = OUT / f"seed_{s}.json"
    if f.exists(): continue
    t0 = time.time(); d = R.prep(s); ytr, yte = d["ytr"], d["yte"]; r = {}
    rng = np.random.default_rng(s)
    for name, kw in [("SR-DMLR", {}), ("Flat-VAE", {"flat": True}), ("Unsup", {"supervised": False})]:
        _, L = R.train(d, s, **kw)
        if kw.get("flat"):
            S, ys = gauss(L["a_tr"], ytr, N, s); Te = L["a_te"]
        else:
            zc, ys = gauss(L["a_tr"][:, :ZD], ytr, N, s)
            S = np.c_[zc, L["e"][rng.integers(len(L["e"]), size=N), ZD:2 * ZD], L["g"][rng.integers(len(L["g"]), size=N), 2 * ZD:]]
            Te = L["a_te"]
        r[name] = lr(S, ys, Te, yte, s); print("  ", name, round(time.time()-t0), "s", flush=True)
        if name == "SR-DMLR":
            r["SR_priv_clin"] = P.privacy(zc, L["a_tr"][:, :ZD], L["a_te"][:, :ZD])
        if name != "Unsup":   # confinement : ou se trouve le diagnostic dans les latents reels ABIDE
            r[f"{name}_real_first8"] = lr(L["a_tr"][:, :ZD], ytr, L["a_te"][:, :ZD], yte, s)
            r[f"{name}_real_last16"] = lr(L["a_tr"][:, ZD:], ytr, L["a_te"][:, ZD:], yte, s)
    r["Real"] = lr(d["Xa_tr"], ytr, d["Xa_te"], yte, s)
    S, ys = P.smote_sample(d["Xa_tr"], ytr, N, s); r["SMOTE-only"] = lr(S, ys, d["Xa_te"], yte, s)
    rr = np.random.default_rng(s + 100); Xe, Xg = d["Xe"], d["Xg"]
    S2 = np.c_[S, Xe[rr.integers(len(Xe), size=N)], Xg[rr.integers(len(Xg), size=N)]]
    Te2 = np.c_[d["Xa_te"], Xe[rr.integers(len(Xe), size=len(yte))], Xg[rr.integers(len(Xg), size=len(yte))]]
    r["Concatenation"] = lr(S2, ys, Te2, yte, s)
    def coral(Xs, Xt):
        cs = np.cov(Xs.T) + np.eye(Xs.shape[1]); ct = np.cov(Xt.T) + np.eye(Xt.shape[1])
        return (Xs - Xs.mean(0)) @ np.linalg.inv(np.linalg.cholesky(cs)).T @ np.linalg.cholesky(ct).T + Xt.mean(0)
    Za_tr = PCA(ZD, random_state=s).fit(d["Xa_tr"]).transform(d["Xa_tr"])
    Ze_c = coral(PCA(ZD, random_state=s).fit_transform(Xe), Za_tr); Zg_c = coral(PCA(ZD, random_state=s).fit_transform(Xg), Za_tr)
    cr = np.random.default_rng(s)
    S_c = np.c_[S, Ze_c[cr.integers(len(Ze_c), size=N)], Zg_c[cr.integers(len(Zg_c), size=N)]]
    Te_c = np.c_[d["Xa_te"], Ze_c[cr.integers(len(Ze_c), size=len(yte))], Zg_c[cr.integers(len(Zg_c), size=len(yte))]]
    r["CORAL"] = lr(S_c, ys, Te_c, yte, s)
    print("  before CTGAN", round(time.time()-t0), "s", flush=True)
    if os.environ.get("SKIP_CTGAN_ALL") == "1" or (s >= 10 and os.environ.get("SKIP_CTGAN_AFTER_10", "1") == "1"):
        r["CTGAN"] = None; r["seconds"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(r)); print(s, "done", r["seconds"], flush=True); continue
    from ctgan import CTGAN; import pandas as pd
    torch.manual_seed(s); np.random.seed(s)
    df = pd.DataFrame(d["Xa_tr"]); df.columns = [f"f{i}" for i in range(df.shape[1])]; df["y"] = ytr
    gan = CTGAN(epochs=300, batch_size=100, verbose=False, cuda=False); gan.fit(df, discrete_columns=["y"])
    sm = gan.sample(N); yg = sm["y"].values.astype(int)
    r["CTGAN"] = lr(sm.drop(columns="y").values, yg, d["Xa_te"], yte, s) if len(set(yg)) == 2 else 0.5
    r["seconds"] = round(time.time() - t0, 1)
    f.write_text(json.dumps(r)); print(s, {k: (round(v, 3) if isinstance(v, float) else "") for k, v in r.items()}, flush=True)
