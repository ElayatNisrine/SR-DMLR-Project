import os, json, numpy as np
import step4_revision as R, step2_pipeline as P
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import roc_auc_score
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
from pathlib import Path
ZD, N = P.ZD, P.N_SYNTH
OUT = Path("flat_audit"); OUT.mkdir(exist_ok=True)
for s in range(30):
    f = OUT / f"seed_{s}.json"
    if f.exists(): continue
    d = R.prep(s); ytr = d["ytr"]; _, L = R.train(d, s, flat=True)
    rng = np.random.default_rng(s + 50)
    te = rng.permutation(len(d["yte"])); nm, ref = te[: len(te) // 2], te[len(te) // 2:]
    mem = rng.choice(len(ytr), len(nm), replace=False)
    Tr, Te = L["a_tr"], L["a_te"]; res = {}
    for gen in ("gauss", "smote"):
        S, _ = gauss(Tr, ytr, N, s) if gen == "gauss" else P.smote_sample(Tr, ytr, N, s)
        res[gen] = {**attacks(S, Tr[mem], Te[nm], Te[ref]), "NND_ratio": P.privacy(S, Tr, Te)["NND_ratio_vs_real"]}
    f.write_text(json.dumps(res)); print(s, {g: round(v["DOMIAS"], 3) for g, v in res.items()}, flush=True)
