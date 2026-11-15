"""
STEP 4 : experiences demandees par les reviewers (Pattern Recognition, rejet).

Pour chaque seed (meme split que step2) :
  [R1] STRT : trois representations du test reel
       (a) latent complet de l'encodeur (protocole principal)
       (b) blocs z_b et z_x mis a zero (train et test)
       (c) blocs z_b et z_x du test tires des pools reels Emotions / Engagnition
  [R2] Baselines supplementaires (meme split, meme PCA, meme N=250)
       CTGAN sur les features ABIDE ; generateur gaussien conditionnel dans z_c ;
       alignement CORAL non apparie ; VAE partage/prive non supervise + MMD
  [R3] Ablations : sans SMOTE (gaussienne par classe), sans reconstruction
       inter-sujets (z_b, z_x issus du meme sujet ABIDE), PCA 32 / 128
  [R4] Sensibilite au pool physiologique : Engagnition sous-echantillonne a 50 %,
       incertitude bootstrap des features z_x synthetiques
  [R5] Fidelite : KS par dimension avec correction BH, Wasserstein-1 par bloc,
       MMD synthetique vs test reel compare a MMD train reel vs test reel
  [R6] Severite : MAE et Spearman sur les ASD test avec ADOS, vs regression Ridge
  [R7] Identifiabilite de z_c : precision d'un classifieur predisant la source
       depuis z_c (avec / sans MMD), variance par source (effondrement)
"""
import os, json, time
from pathlib import Path
import numpy as np, torch
import step2_pipeline as P
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr
from statsmodels.stats.multitest import multipletests

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
A, E, G, ZD, N = P.A, P.E, P.G, P.ZD, P.N_SYNTH
T = lambda a: torch.as_tensor(a, dtype=torch.float32)


def mmd_np(a, b):
    a, b = T(a), T(b)
    return float(P.mmd(a, b))


def prep(seed, n_pca=64, eng_frac=1.0):
    y, ados = A["y"], A["ados"]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    pca = PCA(n_pca, random_state=seed).fit(A["X"][tr])
    sc = StandardScaler().fit(pca.transform(A["X"][tr]))
    Xa_tr = sc.transform(pca.transform(A["X"][tr])).astype(np.float32)
    Xa_te = sc.transform(pca.transform(A["X"][te])).astype(np.float32)
    Xe = StandardScaler().fit_transform(E["X"]).astype(np.float32)
    Xg_all = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
    rng = np.random.default_rng(seed + 7)
    keep = np.sort(rng.choice(len(Xg_all), int(round(eng_frac * len(Xg_all))), replace=False))
    Xg = Xg_all[keep]
    return dict(tr=tr, te=te, Xa_tr=Xa_tr, Xa_te=Xa_te, Xe=Xe, Xg=Xg,
                ytr=y[tr], yte=y[te], ados_tr=ados[tr], ados_te=ados[te])


def train(d, seed, **kw):
    sup = kw.pop("supervised", True)
    data = {"abide": dict(X=T(d["Xa_tr"]), y=T(d["ytr"]) if sup else None,
                          sev=T(d["ados_tr"]) if sup else None),
            "emotion": dict(X=T(d["Xe"]), y=None, sev=None),
            "engagement": dict(X=T(d["Xg"]), y=None, sev=None)}
    m = P.train_model(data, seed, **kw)
    return m, {k: P.latents(m, d[x], s) for k, x, s in
               [("a_tr", "Xa_tr", "abide"), ("a_te", "Xa_te", "abide"),
                ("e", "Xe", "emotion"), ("g", "Xg", "engagement")]}


def synth(L, ytr, seed, zc=None, ys=None):
    rng = np.random.default_rng(seed)
    if zc is None:
        zc, ys = P.smote_sample(L["a_tr"][:, :ZD], ytr, N, seed)
    return np.c_[zc, L["e"][rng.integers(len(L["e"]), size=N), ZD:2 * ZD],
                 L["g"][rng.integers(len(L["g"]), size=N), 2 * ZD:]], ys


def strt_lr(S, ys, Te, yte, seed):
    return P.clf_auc(S, ys, Te, yte, seed)["LR"]


def run(seed):
    t0 = time.time(); r = {}
    d = prep(seed)
    ytr, yte = d["ytr"], d["yte"]

    # ---------------- modele principal
    model, L = train(d, seed)
    S, ys = synth(L, ytr, seed)
    rng = np.random.default_rng(seed + 11)

    # [R1] variantes de representation du test
    Te_full = L["a_te"]
    Te_zero = np.c_[L["a_te"][:, :ZD], np.zeros((len(yte), 2 * ZD))]
    S_zero = np.c_[S[:, :ZD], np.zeros((N, 2 * ZD))]
    Te_pool = np.c_[L["a_te"][:, :ZD], L["e"][rng.integers(len(L["e"]), size=len(yte)), ZD:2 * ZD],
                    L["g"][rng.integers(len(L["g"]), size=len(yte)), 2 * ZD:]]
    r["STRT_encoder_latent"] = strt_lr(S, ys, Te_full, yte, seed)
    r["STRT_bx_zeroed"] = strt_lr(S_zero, ys, Te_zero, yte, seed)
    r["STRT_bx_from_pools"] = strt_lr(S, ys, Te_pool, yte, seed)
    r["STRT_zc_only"] = strt_lr(S[:, :ZD], ys, L["a_te"][:, :ZD], yte, seed)

    # [R7] identifiabilite de z_c
    def source_acc(Lx):
        Z = np.r_[Lx["a_tr"][:, :ZD], Lx["e"][:, :ZD], Lx["g"][:, :ZD]]
        lab = np.r_[np.zeros(len(Lx["a_tr"])), np.ones(len(Lx["e"])), 2 * np.ones(len(Lx["g"]))]
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        return float(cross_val_score(LogisticRegression(max_iter=3000), StandardScaler().fit_transform(Z),
                                     lab, cv=cv, scoring="balanced_accuracy").mean())
    r["zc_source_balacc_with_MMD"] = source_acc(L)
    r["zc_std_by_source"] = {k: float(L[k][:, :ZD].std(0).mean()) for k in ("a_tr", "e", "g")}

    # [R5] fidelite
    refs = [L["a_tr"][:, j] for j in range(ZD)] + [L["e"][:, j] for j in range(ZD, 2 * ZD)] \
         + [L["g"][:, j] for j in range(2 * ZD, 3 * ZD)]
    pv = np.array([ks_2samp(S[:, j], refs[j]).pvalue for j in range(3 * ZD)])
    rej = multipletests(pv, alpha=0.05, method="fdr_bh")[0]
    r["KS_rejected_raw"] = int((pv < 0.05).sum()); r["KS_rejected_BH"] = int(rej.sum())
    r["KS_rejected_BH_by_block"] = [int(rej[:ZD].sum()), int(rej[ZD:2 * ZD].sum()), int(rej[2 * ZD:].sum())]
    r["W1_by_block"] = [float(np.mean([wasserstein_distance(S[:, j], refs[j]) for j in rng_])) for rng_ in
                        (range(ZD), range(ZD, 2 * ZD), range(2 * ZD, 3 * ZD))]
    r["MMD_zc_synth_vs_test"] = mmd_np(S[:, :ZD], L["a_te"][:, :ZD])
    r["MMD_zc_train_vs_test"] = mmd_np(L["a_tr"][:, :ZD], L["a_te"][:, :ZD])

    # [R6] severite
    with torch.no_grad():
        sev_te = model.sev(T(L["a_te"][:, :ZD])).squeeze(1).numpy() * 30
    ok_te = np.isfinite(d["ados_te"]) & (yte == 1)
    ok_tr = np.isfinite(d["ados_tr"]) & (ytr == 1)
    r["sev_MAE"] = float(np.abs(sev_te[ok_te] - d["ados_te"][ok_te]).mean())
    r["sev_spearman"] = float(spearmanr(sev_te[ok_te], d["ados_te"][ok_te]).correlation)
    rid = Ridge(10.0).fit(d["Xa_tr"][ok_tr], d["ados_tr"][ok_tr])
    pr = rid.predict(d["Xa_te"][ok_te])
    r["ridge_MAE"] = float(np.abs(pr - d["ados_te"][ok_te]).mean())
    r["ridge_spearman"] = float(spearmanr(pr, d["ados_te"][ok_te]).correlation)
    r["mean_predictor_MAE"] = float(np.abs(d["ados_tr"][ok_tr].mean() - d["ados_te"][ok_te]).mean())
    r["n_sev_test"] = int(ok_te.sum())

    # ---------------- [R3] ablations
    # sans SMOTE : gaussienne par classe dans z_c
    g_rng = np.random.default_rng(seed + 3)
    zc_tr = L["a_tr"][:, :ZD]; n1 = int(round(N * ytr.mean()))
    zc_g = np.r_[g_rng.multivariate_normal(zc_tr[ytr == 1].mean(0), np.cov(zc_tr[ytr == 1].T), n1),
                 g_rng.multivariate_normal(zc_tr[ytr == 0].mean(0), np.cov(zc_tr[ytr == 0].T), N - n1)]
    ys_g = np.r_[np.ones(n1), np.zeros(N - n1)].astype(int)
    S_g, _ = synth(L, ytr, seed, zc_g, ys_g)
    r["abl_no_SMOTE_gaussian"] = {"STRT": strt_lr(S_g, ys_g, Te_full, yte, seed),
                                  **P.privacy(S_g, L["a_tr"], L["a_te"])}
    # sans reconstruction inter-sujets : z_b, z_x du meme sujet ABIDE
    S_intra, ys_i = P.smote_sample(L["a_tr"], ytr, N, seed)
    r["abl_no_intersubject"] = {"STRT": strt_lr(S_intra, ys_i, Te_full, yte, seed),
                                **P.privacy(S_intra, L["a_tr"], L["a_te"])}
    r["SRDMLR_privacy"] = P.privacy(S, L["a_tr"], L["a_te"])
    # PCA 32 et 128
    for k in (32, 128):
        dk = prep(seed, n_pca=k); _, Lk = train(dk, seed)
        Sk, ysk = synth(Lk, dk["ytr"], seed)
        r[f"abl_PCA{k}"] = strt_lr(Sk, ysk, Lk["a_te"], dk["yte"], seed)

    # ---------------- [R4] sensibilite physiologique
    dh = prep(seed, eng_frac=0.5); _, Lh = train(dh, seed)
    Sh, ysh = synth(Lh, dh["ytr"], seed)
    r["physio50_STRT"] = strt_lr(Sh, ysh, Lh["a_te"], dh["yte"], seed)
    boots = np.array([L["g"][np.random.default_rng(seed * 1000 + b).integers(len(L["g"]), size=len(L["g"])),
                             2 * ZD:].mean(0) for b in range(150)])
    r["physio_zx_bootstrap_sd_mean"] = float(boots.std(0).mean())
    r["physio_zx_pool_sd_mean"] = float(L["g"][:, 2 * ZD:].std(0).mean())

    # ---------------- [R2] baselines
    # VAE partage/prive non supervise + MMD (pas de tete clinique pendant l'entrainement)
    _, Lu = train(d, seed, supervised=False)
    Su, ysu = synth(Lu, ytr, seed)
    r["base_unsup_sharedprivate"] = {"STRT": strt_lr(Su, ysu, Lu["a_te"], yte, seed),
                                     **P.privacy(Su, Lu["a_tr"], Lu["a_te"])}
    r["zc_source_balacc_unsup"] = source_acc(Lu)
    # CORAL non apparie : PCA 8 par source, covariances alignees sur ABIDE
    def coral(Xs, Xt):
        cs = np.cov(Xs.T) + np.eye(Xs.shape[1]); ct = np.cov(Xt.T) + np.eye(Xt.shape[1])
        ws = np.linalg.inv(np.linalg.cholesky(cs)).T; wt = np.linalg.cholesky(ct).T
        return (Xs - Xs.mean(0)) @ ws @ wt + Xt.mean(0)
    pa = PCA(ZD, random_state=seed).fit(d["Xa_tr"])
    Za_tr, Za_te = pa.transform(d["Xa_tr"]), pa.transform(d["Xa_te"])
    Ze_c = coral(PCA(ZD, random_state=seed).fit_transform(d["Xe"]), Za_tr)
    Zg_c = coral(PCA(ZD, random_state=seed).fit_transform(d["Xg"]), Za_tr)
    # bloc ABIDE : les 64 features (l'espace PCA a 8 dimensions ne porte pas le diagnostic)
    zc_s, ys_c = P.smote_sample(d["Xa_tr"], ytr, N, seed)
    cr = np.random.default_rng(seed)
    S_c = np.c_[zc_s, Ze_c[cr.integers(len(Ze_c), size=N)], Zg_c[cr.integers(len(Zg_c), size=N)]]
    Te_c = np.c_[d["Xa_te"], Ze_c[cr.integers(len(Ze_c), size=len(yte))], Zg_c[cr.integers(len(Zg_c), size=len(yte))]]
    Tr_c = np.c_[d["Xa_tr"], Ze_c[cr.integers(len(Ze_c), size=len(ytr))], Zg_c[cr.integers(len(Zg_c), size=len(ytr))]]
    r["base_CORAL"] = {"STRT": strt_lr(S_c, ys_c, Te_c, yte, seed), **P.privacy(S_c, Tr_c, Te_c)}
    # CTGAN sur les features ABIDE (64 PCA + label)
    from ctgan import CTGAN
    import pandas as pd
    torch.manual_seed(seed); np.random.seed(seed)
    df = pd.DataFrame(d["Xa_tr"]); df.columns = [f"f{i}" for i in range(df.shape[1])]; df["y"] = ytr
    gan = CTGAN(epochs=300, batch_size=100, verbose=False, cuda=False); gan.fit(df, discrete_columns=["y"])
    sm = gan.sample(N); Sg, ysg = sm.drop(columns="y").values, sm["y"].values.astype(int)
    if len(np.unique(ysg)) == 2:
        r["base_CTGAN"] = {"STRT": strt_lr(Sg, ysg, d["Xa_te"], yte, seed),
                           **P.privacy(Sg, d["Xa_tr"], d["Xa_te"])}
    r["seconds"] = round(time.time() - t0, 1)
    return r


if __name__ == "__main__":
    runs = {}
    for s in SEEDS:
        runs[s] = run(s)
        print(f"seed {s} ok ({runs[s]['seconds']} s) STRT={runs[s]['STRT_encoder_latent']:.3f}", flush=True)
        Path("revision_results.json").write_text(json.dumps(runs, indent=1))
    print("done")
