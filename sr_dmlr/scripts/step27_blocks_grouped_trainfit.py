"""STEP 27 : utilite des blocs separes dans un pipeline aval (provenance controlee, 30 seeds).
Pretraitement ajuste sur la partie entrainement uniquement. Par seed : FaceNet brut -> StandardScaler + PCA(64) + StandardScaler ajustes sur e_tr seulement ;
Engagnition -> StandardScaler ajuste sur g_tr seulement (masque applique). ABIDE 70/30 ; ASD Emotions 2/3-1/3 par groupes de quasi-doublons, stratifie par emotion ;
Engagnition 70/30 stratifie par condition. Taches aval entrainees sur la cohorte synthetique, testees sur des sujets reels jamais vus.
Sortie : blocks_trainfit/seed_<s>.json"""
import os, json, numpy as np
import step4_revision as R, step2_pipeline as P
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from pathlib import Path
ZD, N = P.ZD, P.N_SYNTH
OUT = Path("blocks_trainfit"); OUT.mkdir(exist_ok=True)
F = np.load("features/emotion_facenet.npz", allow_pickle=True); FX = F["X"]; emo = F["emotion"]
G = P.G; cond = G["condition"]
DUP = np.load("features/emotion_dupgroups.npy")
BL = {"z_c": slice(0, ZD), "z_b": slice(ZD, 2 * ZD), "z_x": slice(2 * ZD, 3 * ZD)}
def keep(X, blocks):
    Y = np.zeros_like(X)
    for b in blocks: Y[:, BL[b]] = X[:, BL[b]]
    return Y
def auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr); c = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    p = c.predict_proba(sc.transform(Xte))
    return float(roc_auc_score(yte, p[:, 1])) if p.shape[1] == 2 else float(roc_auc_score(yte, p, multi_class="ovr", average="macro"))
for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
    f = OUT / f"seed_{s}.json"
    if f.exists(): continue
    d = R.prep(s); ytr, yte = d["ytr"], d["yte"]
    e_tr, e_te = next(StratifiedGroupKFold(3, shuffle=True, random_state=s).split(np.zeros(len(emo)), emo, DUP))
    g_tr, g_te = train_test_split(np.arange(len(cond)), test_size=0.3, stratify=cond, random_state=s)
    # pretraitement ajuste sur train seulement
    s1 = StandardScaler().fit(FX[e_tr]); pca = PCA(64, random_state=0).fit(s1.transform(FX[e_tr]))
    s2 = StandardScaler().fit(pca.transform(s1.transform(FX[e_tr])))
    fe = lambda X: s2.transform(pca.transform(s1.transform(X))).astype(np.float32)
    Xe_tr, Xe_te = fe(FX[e_tr]), fe(FX[e_te])
    sg = StandardScaler().fit(G["X"][g_tr])
    Xg_tr = (sg.transform(G["X"][g_tr]) * G["mask"][g_tr]).astype(np.float32)
    Xg_te = (sg.transform(G["X"][g_te]) * G["mask"][g_te]).astype(np.float32)
    dd = dict(d, Xe=Xe_tr, Xg=Xg_tr)
    m, L = R.train(dd, s)
    Le_te = P.latents(m, Xe_te, "emotion"); Lg_te = P.latents(m, Xg_te, "engagement")
    rng = np.random.default_rng(s + 3); n1 = int(round(N * ytr.mean())); Z = L["a_tr"][:, :ZD]
    zc = np.r_[rng.multivariate_normal(Z[ytr == 1].mean(0), np.cov(Z[ytr == 1].T), n1),
               rng.multivariate_normal(Z[ytr == 0].mean(0), np.cov(Z[ytr == 0].T), N - n1)]
    ys = np.r_[np.ones(n1), np.zeros(N - n1)].astype(int)
    r2 = np.random.default_rng(s); ib = r2.integers(len(e_tr), size=N); ix = r2.integers(len(g_tr), size=N)
    S = np.c_[zc, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
    s_emo, s_cond = emo[e_tr][ib], cond[g_tr][ix]
    hl = np.isin(s_cond, ["HPE", "LPE"]); hl_te = np.isin(cond[g_te], ["HPE", "LPE"])
    tasks = {"diagnosis": (S, ys, L["a_te"], yte),
             "emotion": (S, s_emo, Le_te, emo[e_te]),
             "condition_HPE_vs_LPE": (S[hl], (s_cond[hl] == "HPE").astype(int), Lg_te[hl_te], (cond[g_te][hl_te] == "HPE").astype(int)),
             "condition_3class": (S, s_cond, Lg_te, cond[g_te])}
    configs = {"all": ["z_c", "z_b", "z_x"], "z_c only": ["z_c"], "z_b only": ["z_b"], "z_x only": ["z_x"],
               "without z_c": ["z_b", "z_x"], "without z_b": ["z_c", "z_x"], "without z_x": ["z_c", "z_b"]}
    res = {t: {c: auc(keep(a, b), y1, keep(te, b), y2) for c, b in configs.items()} for t, (a, y1, te, y2) in tasks.items()}
    hltr = np.isin(cond[g_tr], ["HPE", "LPE"])
    res["real_reference"] = {
        "diagnosis": auc(d["Xa_tr"], ytr, d["Xa_te"], yte),
        "emotion": auc(Xe_tr, emo[e_tr], Xe_te, emo[e_te]),
        "condition_HPE_vs_LPE": auc(Xg_tr[hltr], (cond[g_tr][hltr] == "HPE").astype(int), Xg_te[hl_te], (cond[g_te][hl_te] == "HPE").astype(int)),
        "condition_3class": auc(Xg_tr, cond[g_tr], Xg_te, cond[g_te])}
    res["n_test"] = {"diagnosis": int(len(yte)), "emotion": int(len(e_te)), "condition_HPE_vs_LPE": int(hl_te.sum()), "condition_3class": int(len(g_te))}
    f.write_text(json.dumps(res)); print(s, {t: round(v["all"], 3) for t, v in res.items() if isinstance(v, dict) and "all" in v}, flush=True)
