"""STEP 16 : analyses de sensibilite et diagnostics sur 30 seeds (generateur gaussien)."""
import os, json, numpy as np, torch
import step4_revision as R, step2_pipeline as P
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from pathlib import Path
ZD, N = P.ZD, P.N_SYNTH; OUT = Path("sens16"); OUT.mkdir(exist_ok=True)
def gauss(Z, y, n, seed):
    r = np.random.default_rng(seed + 3); n1 = int(round(n * y.mean()))
    o = [r.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k) for c, k in ((1, n1), (0, n - n1))]
    return np.r_[o[0], o[1]], np.r_[np.ones(n1), np.zeros(n - n1)].astype(int)
def strt(d, L, s):
    zc, ys = gauss(L["a_tr"][:, :ZD], d["ytr"], N, s); rng = np.random.default_rng(s)
    S = np.c_[zc, L["e"][rng.integers(len(L["e"]), size=N), ZD:2 * ZD], L["g"][rng.integers(len(L["g"]), size=N), 2 * ZD:]]
    return P.clf_auc(S, ys, L["a_te"], d["yte"], s)["LR"]
for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
    f = OUT / f"seed_{s}.json"
    if f.exists(): continue
    d = R.prep(s); ytr, yte = d["ytr"], d["yte"]; m, L = R.train(d, s); r = {"base": strt(d, L, s)}
    Z = np.r_[L["a_tr"][:, :ZD], L["e"][:, :ZD], L["g"][:, :ZD]]
    lab = np.r_[np.zeros(len(L["a_tr"])), np.ones(len(L["e"])), 2 * np.ones(len(L["g"]))]
    r["zc_source_balacc"] = float(cross_val_score(LogisticRegression(max_iter=3000), StandardScaler().fit_transform(Z), lab,
                                                  cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="balanced_accuracy").mean())
    with torch.no_grad(): sev = m.sev(torch.as_tensor(L["a_te"][:, :ZD])).squeeze(1).numpy() * 30
    ok_te = np.isfinite(d["ados_te"]) & (yte == 1); ok_tr = np.isfinite(d["ados_tr"]) & (ytr == 1)
    r["sev_MAE"] = float(np.abs(sev[ok_te] - d["ados_te"][ok_te]).mean()); r["sev_rho"] = float(spearmanr(sev[ok_te], d["ados_te"][ok_te]).correlation)
    pr = Ridge(10.0).fit(d["Xa_tr"][ok_tr], d["ados_tr"][ok_tr]).predict(d["Xa_te"][ok_te])
    r["ridge_MAE"] = float(np.abs(pr - d["ados_te"][ok_te]).mean()); r["mean_MAE"] = float(np.abs(d["ados_tr"][ok_tr].mean() - d["ados_te"][ok_te]).mean())
    for k in (32, 128):
        dk = R.prep(s, n_pca=k); _, Lk = R.train(dk, s); r[f"PCA{k}"] = strt(dk, Lk, s)
    dh = R.prep(s, eng_frac=0.5); _, Lh = R.train(dh, s); r["physio50"] = strt(dh, Lh, s)
    f.write_text(json.dumps(r)); print(s, {k: round(v, 3) for k, v in r.items()}, flush=True)
