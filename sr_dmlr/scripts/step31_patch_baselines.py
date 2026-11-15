"""STEP 31 : recalcule les baselines qui utilisent SMOTE (SMOTE-only et Concatenation) avec le sampler du
step2_pipeline courant, sur les memes splits, et recopie toutes les autres valeurs des fichiers d'origine
(ces baselines n'entrainent aucun modele : le calcul est deterministe).
Usage : python step31_patch_baselines.py <dossier_source> <dossier_sortie>"""
import sys, os, json, glob, numpy as np
from pathlib import Path
SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
import step4_revision as R, step2_pipeline as P
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
A, E, G, ZD, N = P.A, P.E, P.G, P.ZD, P.N_SYNTH
sites, y_all, ados = A["site"], A["y"], A["ados"]
lr2 = lambda *a: P.clf_auc(*a)["LR"]
Xe0 = StandardScaler().fit_transform(E["X"]).astype(np.float32); Xg0 = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
def concat_parts(Sm, Xte, n_te, s, Xe, Xg):
    rr = np.random.default_rng(s + 100)
    return np.c_[Sm, Xe[rr.integers(len(Xe), size=N)], Xg[rr.integers(len(Xg), size=N)]], np.c_[Xte, Xe[rr.integers(len(Xe), size=n_te)], Xg[rr.integers(len(Xg), size=n_te)]]
def pca_split(X, tr, te, s):
    pca = PCA(64, random_state=s).fit(X[tr]); sc = StandardScaler().fit(pca.transform(X[tr]))
    return sc.transform(pca.transform(X[tr])).astype(np.float32), sc.transform(pca.transform(X[te])).astype(np.float32)
# sens (step10)
os.makedirs(DST / "sens", exist_ok=True)
for f in sorted(glob.glob(str(SRC / "sens" / "seed_*.json"))):
    s = int(f.split("_")[-1][:-5]); r = json.load(open(f)); d = R.prep(s); ytr, yte = d["ytr"], d["yte"]
    S, ys = P.smote_sample(d["Xa_tr"], ytr, N, s); r["SMOTE-only"] = lr2(S, ys, d["Xa_te"], yte, s)
    rr = np.random.default_rng(s + 100); Xe, Xg = d["Xe"], d["Xg"]
    S2 = np.c_[S, Xe[rr.integers(len(Xe), size=N)], Xg[rr.integers(len(Xg), size=N)]]
    Te2 = np.c_[d["Xa_te"], Xe[rr.integers(len(Xe), size=len(yte))], Xg[rr.integers(len(Xg), size=len(yte))]]
    r["Concatenation"] = lr2(S2, ys, Te2, yte, s); json.dump(r, open(DST / "sens" / os.path.basename(f), "w"))
print("sens ok", flush=True)
# sites (step18)
os.makedirs(DST / "sites", exist_ok=True); us = np.unique(sites)
for rep in range(3):
    perm = np.random.default_rng(rep).permutation(us); fold_of = {x: i % 5 for i, x in enumerate(perm)}
    for k in range(5):
        r = json.load(open(SRC / "sites" / f"rep{rep}_fold{k}.json"))
        s = 100 * rep + k; te = np.where([fold_of[x] == k for x in sites])[0]; tr = np.setdiff1d(np.arange(len(sites)), te)
        Xtr, Xte = pca_split(A["X"], tr, te, s)
        Sm, ysm = P.smote_sample(Xtr, y_all[tr], N, s); r["SMOTE-only"] = lr2(Sm, ysm, Xte, y_all[te], s)
        a, b = concat_parts(Sm, Xte, len(te), s, Xe0, Xg0); r["Concatenation"] = lr2(a, ysm, b, y_all[te], s)
        json.dump(r, open(DST / "sites" / f"rep{rep}_fold{k}.json", "w"))
print("sites ok", flush=True)
# siteharm (step23)
def harmonise(X, mode):
    if mode == "none": return X
    Z = np.empty_like(X)
    for s_ in np.unique(sites):
        i = sites == s_; mu = X[i].mean(0); Z[i] = (X[i] - mu) / (X[i].std(0) + 1e-6) if mode == "zscore" else X[i] - mu
    return Z
XH = {m: harmonise(A["X"], m) for m in ("none", "center", "zscore")}
jobs = []
for s in range(30):
    tr, te = train_test_split(np.arange(len(y_all)), test_size=0.3, stratify=y_all, random_state=s); jobs.append((f"random_s{s}", tr, te, s))
for rep in range(3):
    perm = np.random.default_rng(rep).permutation(us); fold_of = {x: i % 5 for i, x in enumerate(perm)}
    for k in range(5):
        te = np.where([fold_of[x] == k for x in sites])[0]; tr = np.setdiff1d(np.arange(len(sites)), te); jobs.append((f"site_rep{rep}_fold{k}", tr, te, 100 * rep + k))
os.makedirs(DST / "siteharm", exist_ok=True)
for name, tr, te, s in jobs:
    res = json.load(open(SRC / "siteharm" / f"{name}.json"))
    for m in ("none", "center", "zscore"):
        Xtr, Xte = pca_split(XH[m], tr, te, s); lr = lambda *a: P.clf_auc(*a, s)["LR"]
        Sm, ysm = P.smote_sample(Xtr, y_all[tr], N, s); res[m]["SMOTE-only"] = lr(Sm, ysm, Xte, y_all[te])
        a, b = concat_parts(Sm, Xte, len(te), s, Xe0, Xg0); res[m]["Concatenation"] = lr(a, ysm, b, y_all[te])
    json.dump(res, open(DST / "siteharm" / f"{name}.json", "w"))
print("siteharm ok", flush=True)
# loso (step25)
def prob(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); c = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    return c.predict_proba(sc.transform(Xte))[:, 1].tolist()
XL = {"none": A["X"], "center": harmonise(A["X"], "center")}
os.makedirs(DST / "loso", exist_ok=True)
for si, u in enumerate(np.unique(sites)):
    res = json.load(open(SRC / "loso" / f"{u}.json")); te = np.where(sites == u)[0]; tr = np.where(sites != u)[0]; s = si
    for mode in ("none", "center"):
        Xtr, Xte = pca_split(XL[mode], tr, te, s)
        Sm, ysm = P.smote_sample(Xtr, y_all[tr], N, s); res[mode]["SMOTE-only"] = prob(Sm, ysm, Xte)
        a, b = concat_parts(Sm, Xte, len(te), s, Xe0, Xg0); res[mode]["Concatenation"] = prob(a, ysm, b)
    json.dump(res, open(DST / "loso" / f"{u}.json", "w"))
print("loso ok", flush=True)
