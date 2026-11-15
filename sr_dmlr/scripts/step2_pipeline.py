"""
STEP 2 : encodeur tripartite corrige + generation + evaluation + baselines

Corrections par rapport au pipeline d'origine :
  [C1] etiquettes gardees PAR MODALITE (plus de collate qui melange les labels)
  [C2] diagnostic et severite uniquement pour ABIDE (Emotions/Engagnition n'en ont pas)
  [C3] la severite n'est jamais une entree de l'encodeur
  [C4] split train/test ABIDE fait AVANT tout (PCA, encodeur, SMOTE) : pas de fuite
  [C5] chaque chiffre est calcule ici et ecrit dans results.json (rien a la main)
  [C6] tests bloquants : si un test echoue, le script s'arrete

Usage : python step2_pipeline.py   (lit features/, ecrit results.json)
"""
import json, hashlib, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

torch.set_num_threads(2)
import os
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "150"))
N_SYNTH, K_SMOTE, ZD = 250, 9, 8
FEAT = Path("features")

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

A = np.load(FEAT / "abide.npz", allow_pickle=True)
if os.environ.get("EMO", "facenet") == "facenet":
    # embeddings FaceNet (step22) reduits a 64 composantes par PCA non supervisee sur toutes les images
    from sklearn.decomposition import PCA as _PCA
    _f = np.load(FEAT / "emotion_facenet.npz", allow_pickle=True)
    _Xs = (_f["X"] - _f["X"].mean(0)) / (_f["X"].std(0) + 1e-8)
    class _E(dict):
        files = ["X", "emotion"]
    E = _E(X=_PCA(64, random_state=0).fit_transform(_Xs).astype(np.float32), emotion=_f["emotion"])
else:
    E = np.load(FEAT / "emotion.npz", allow_pickle=True)
G = np.load(FEAT / "engagement.npz", allow_pickle=True)

# ============================================================ tests de donnees
def check(cond, msg):
    if not cond:
        raise SystemExit(f"TEST ECHOUE : {msg}")
check(set(np.unique(A["y"])) == {0, 1}, "ABIDE doit avoir ASD et TD")
check("y" not in E.files and "diagnosis" not in E.files, "Emotions ne doit pas avoir de diagnostic")
check("y" not in G.files and "diagnosis" not in G.files, "Engagnition ne doit pas avoir de diagnostic")
check(len(set(A["sub_id"])) == len(A["sub_id"]), "sujet ABIDE duplique")


# ============================================================ modele
class Tripartite(nn.Module):
    """Projecteurs par modalite -> tronc partage -> z_c, z_b, z_x (ou z plat)."""
    def __init__(self, dims, flat=False):
        super().__init__()
        self.flat = flat
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, 128), nn.LayerNorm(128), nn.ReLU())
                                   for m, d in dims.items()})
        self.trunk = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        nz = 3 * ZD
        self.mu, self.lv = nn.Linear(64, nz), nn.Linear(64, nz)
        self.dec = nn.ModuleDict({m: nn.Sequential(nn.Linear(nz, 128), nn.ReLU(), nn.Linear(128, d))
                                  for m, d in dims.items()})
        hin = nz if flat else ZD        # flat : les tetes cliniques voient tout le latent
        self.diag = nn.Sequential(nn.Linear(hin, 16), nn.ReLU(), nn.Linear(16, 1))
        self.sev = nn.Sequential(nn.Linear(hin, 16), nn.ReLU(), nn.Linear(16, 1))

    def encode(self, x, m):
        h = self.trunk(self.proj[m](x))
        return self.mu(h), self.lv(h)

    def clin(self, z):
        return z if self.flat else z[:, :ZD]


def mmd(a, b):
    if len(a) < 2 or len(b) < 2: return a.new_zeros(())
    def k(x, y): return torch.exp(-torch.cdist(x, y) ** 2 / (2 * x.shape[1]))
    return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()


def train_model(data, seed, flat=False, lam_mmd=1.0, lam_sev=1.0, epochs=None):
    """
    data[m] = dict(X=tensor, y=tensor or None, sev=tensor or None)
    Chaque modalite est echantillonnee separement : les etiquettes ne quittent
    jamais leurs lignes (correction [C1]).
    """
    epochs = epochs or EPOCHS
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Tripartite({m: d["X"].shape[1] for m, d in data.items()}, flat)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    bs = 64
    for ep in range(epochs):
        steps = max(len(d["X"]) for d in data.values()) // bs + 1
        for _ in range(steps):
            loss, zc = 0.0, {}
            for m, d in data.items():
                idx = torch.as_tensor(rng.choice(len(d["X"]), min(bs, len(d["X"])), replace=False))
                x = d["X"][idx]
                mu, lv = model.encode(x, m)
                z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
                loss = loss + F.mse_loss(model.dec[m](z), x)
                loss = loss + 1e-3 * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean())
                zc[m] = model.clin(mu)
                if d["y"] is not None:                       # ABIDE uniquement
                    yy = d["y"][idx]
                    loss = loss + F.binary_cross_entropy_with_logits(model.diag(zc[m]).squeeze(1), yy)
                if d["sev"] is not None and lam_sev > 0:
                    s = d["sev"][idx]; ok = torch.isfinite(s)
                    if ok.any():
                        loss = loss + lam_sev * F.mse_loss(model.sev(zc[m][ok]).squeeze(1), s[ok] / 30)
            if lam_mmd > 0 and not flat:
                ms = list(zc)
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        loss = loss + lam_mmd * mmd(zc[ms[i]], zc[ms[j]])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def latents(model, X, m):
    return model.encode(torch.as_tensor(X, dtype=torch.float32), m)[0].numpy()


# ============================================================ evaluation
def clf_auc(Xtr, ytr, Xte, yte, seed):
    out = {}
    for name, c in [("LR", LogisticRegression(max_iter=3000, C=1.0)),
                    ("RF", RandomForestClassifier(200, random_state=seed, n_jobs=2))]:
        sc = StandardScaler().fit(Xtr)
        c.fit(sc.transform(Xtr), ytr)
        out[name] = float(roc_auc_score(yte, c.predict_proba(sc.transform(Xte))[:, 1]))
    return out


def smote_sample(Z, y, n, seed):
    """SMOTE : chaque classe est triplee par interpolation entre voisins (k = K_SMOTE), puis n points sont tires
    UNIQUEMENT parmi les points nouvellement crees, en gardant la proportion de classes d'origine.
    Aucun enregistrement d'entrainement n'est recopie."""
    rng = np.random.default_rng(seed)
    c = np.bincount(y); k = min(K_SMOTE, c.min() - 1)
    Zs, ys = SMOTE(sampling_strategy={0: 3 * int(c[0]), 1: 3 * int(c[1])}, k_neighbors=k, random_state=seed).fit_resample(Z, y)
    new = np.arange(len(Z), len(Zs))
    n1 = int(round(n * y.mean()))
    new1, new0 = new[ys[new] == 1], new[ys[new] == 0]
    idx = np.r_[rng.choice(new1, n1, replace=n1 > len(new1)), rng.choice(new0, n - n1, replace=(n - n1) > len(new0))]
    return Zs[idx], ys[idx]


def privacy(S, Rtr, Rte):
    """NND et MIA par distance : AUROC ~0.5 = les synthetiques ne trahissent pas les membres."""
    nn_ = NearestNeighbors(n_neighbors=1).fit(S)
    d_tr = nn_.kneighbors(Rtr)[0][:, 0]; d_te = nn_.kneighbors(Rte)[0][:, 0]
    mia = roc_auc_score(np.r_[np.ones(len(d_tr)), np.zeros(len(d_te))], -np.r_[d_tr, d_te])
    nnd = NearestNeighbors(n_neighbors=1).fit(Rtr).kneighbors(S)[0][:, 0]
    rr = NearestNeighbors(n_neighbors=2).fit(Rtr).kneighbors(Rtr)[0][:, 1]
    return {"MIA_AUROC": float(mia), "NND_mean": float(nnd.mean()),
            "NND_ratio_vs_real": float(nnd.mean() / rr.mean())}


def sev_attack(S, sev, seed):
    y = (sev >= np.median(sev)).astype(int)
    b = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(S, y):
        m = RandomForestClassifier(200, random_state=seed, n_jobs=2).fit(S[tr], y[tr])
        b.append(balanced_accuracy_score(y[te], m.predict(S[te])))
    return {"bal_acc": float(np.mean(b)), "baseline": 0.5}


# ============================================================ une repetition
def run_seed(seed):
    t0 = time.time()
    y_all, ados = A["y"], A["ados"]
    idx = np.arange(len(y_all))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y_all, random_state=seed)
    check(len(set(A["sub_id"][tr]) & set(A["sub_id"][te])) == 0, "fuite train/test ABIDE")

    pca = PCA(64, random_state=seed).fit(A["X"][tr])                   # [C4] fit sur train
    sc_a = StandardScaler().fit(pca.transform(A["X"][tr]))
    Xa_tr = sc_a.transform(pca.transform(A["X"][tr])).astype(np.float32)
    Xa_te = sc_a.transform(pca.transform(A["X"][te])).astype(np.float32)
    Xe = StandardScaler().fit_transform(E["X"]).astype(np.float32)
    Xg = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
    ytr, yte = y_all[tr], y_all[te]

    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    data = {"abide": dict(X=T(Xa_tr), y=T(ytr), sev=T(ados[tr])),
            "emotion": dict(X=T(Xe), y=None, sev=None),
            "engagement": dict(X=T(Xg), y=None, sev=None)}
    check(all(d["X"].shape[1] in (64, 26, 14) for d in data.values()), "entree inattendue (severite ?)")

    res = {}

    def eval_tripartite(name, **kw):
        model = train_model(data, seed, **kw)
        Za_tr, Za_te = latents(model, Xa_tr, "abide"), latents(model, Xa_te, "abide")
        Ze, Zg = latents(model, Xe, "emotion"), latents(model, Xg, "engagement")
        r = {"z_b_std": float(Ze[:, ZD:2 * ZD].std()) if not kw.get("flat") else None}
        r["TRTR"] = clf_auc(Za_tr, ytr, Za_te, yte, seed)
        rng = np.random.default_rng(seed)
        if kw.get("flat"):
            S, ys = smote_sample(Za_tr, ytr, N_SYNTH, seed)
            Rtr, Rte = Za_tr, Za_te
        else:
            Zc, ys = smote_sample(Za_tr[:, :ZD], ytr, N_SYNTH, seed)
            S = np.c_[Zc, Ze[rng.integers(len(Ze), size=N_SYNTH), ZD:2 * ZD],
                      Zg[rng.integers(len(Zg), size=N_SYNTH), 2 * ZD:]]
            Rtr, Rte = Za_tr, Za_te
            r["STRT_zc_only"] = clf_auc(S[:, :ZD], ys, Za_te[:, :ZD], yte, seed)
        r["STRT"] = clf_auc(S, ys, Rte, yte, seed)
        r["TRTS"] = clf_auc(Rtr, ytr, S, ys, seed)
        st = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(S, ys):
            st.append(clf_auc(S[a], ys[a], S[b], ys[b], seed))
        r["STST"] = {c: float(np.mean([x[c] for x in st])) for c in ("LR", "RF")}
        r["privacy"] = privacy(S, Rtr, Rte)
        # fidelite : KS par dimension, synthetique vs pool reel correspondant
        from scipy.stats import ks_2samp
        if kw.get("flat"):
            refs = [Za_tr[:, j] for j in range(S.shape[1])]
        else:
            refs = [Za_tr[:, j] for j in range(ZD)] + [Ze[:, j] for j in range(ZD, 2 * ZD)] \
                 + [Zg[:, j] for j in range(2 * ZD, 3 * ZD)]
        r["KS_pass_frac"] = float(np.mean([ks_2samp(S[:, j], refs[j]).pvalue > 0.05
                                           for j in range(S.shape[1])]))
        # gradient clinique : |corr(PC1(z_c), ADOS)| chez les ASD d'entrainement avec ADOS
        okg = np.isfinite(ados[tr]) & (ytr == 1)
        Zg_c = Za_tr[okg] if kw.get("flat") else Za_tr[okg, :ZD]
        pc1 = PCA(1).fit_transform(Zg_c)[:, 0]
        r["clinical_gradient_abs_rho"] = float(abs(np.corrcoef(pc1, ados[tr][okg])[0, 1]))
        with torch.no_grad():
            sev_s = model.sev(T(S if kw.get("flat") else S[:, :ZD])).squeeze(1).numpy() * 30
            sev_te = model.sev(T(Za_te if kw.get("flat") else Za_te[:, :ZD])).squeeze(1).numpy() * 30
        ok = np.isfinite(ados[te]) & (yte == 1)
        r["sev_pred_vs_ADOS_test_r"] = float(np.corrcoef(sev_te[ok], ados[te][ok])[0, 1]) if ok.sum() > 5 else None
        r["severity_attack"] = sev_attack(S, sev_s, seed)
        r["synth_sev_range"] = [float(sev_s.min()), float(sev_s.max())]
        return r

    res["SR-DMLR"] = eval_tripartite("full")
    res["no_MMD"] = eval_tripartite("nommd", lam_mmd=0.0)
    res["no_Lsev"] = eval_tripartite("nosev", lam_sev=0.0)
    res["Flat-VAE+Lsev"] = eval_tripartite("flat", flat=True)

    # ------------------------------------------------ baselines sans latent
    res["Brain-only_real(TRTR)"] = clf_auc(Xa_tr, ytr, Xa_te, yte, seed)
    S, ys = smote_sample(Xa_tr, ytr, N_SYNTH, seed)
    res["SMOTE-only_STRT"] = clf_auc(S, ys, Xa_te, yte, seed)
    res["SMOTE-only_privacy"] = privacy(S, Xa_tr, Xa_te)
    res["SMOTE-only_TRTS"] = clf_auc(Xa_tr, ytr, S, ys, seed)
    rng = np.random.default_rng(seed + 100)
    S2 = np.c_[S, Xe[rng.integers(len(Xe), size=N_SYNTH)], Xg[rng.integers(len(Xg), size=N_SYNTH)]]
    Te2 = np.c_[Xa_te, Xe[rng.integers(len(Xe), size=len(te))], Xg[rng.integers(len(Xg), size=len(te))]]
    res["Concatenation_STRT"] = clf_auc(S2, ys, Te2, yte, seed)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ============================================================ main
if __name__ == "__main__":
    runs = {}
    for s in SEEDS:
        runs[s] = run_seed(s)
        print(f"seed {s} ok ({runs[s]['seconds']} s)  SR-DMLR STRT LR="
              f"{runs[s]['SR-DMLR']['STRT']['LR']:.3f}", flush=True)

    def agg(get):
        v = np.array([get(runs[s]) for s in SEEDS], float)
        return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4),
                "per_seed": [round(float(x), 4) for x in v]}

    summary = {}
    for m in ["SR-DMLR", "no_MMD", "no_Lsev", "Flat-VAE+Lsev"]:
        summary[m] = {f"{k}_{c}": agg(lambda r, k=k, c=c, m=m: r[m][k][c])
                      for k in ["TRTR", "TRTS", "STRT", "STST"] for c in ["LR", "RF"]}
        summary[m]["KS_pass_frac"] = agg(lambda r, m=m: r[m]["KS_pass_frac"])
        summary[m]["clinical_gradient_abs_rho"] = agg(lambda r, m=m: r[m]["clinical_gradient_abs_rho"])
        summary[m]["NND_mean"] = agg(lambda r, m=m: r[m]["privacy"]["NND_mean"])
        if m != "Flat-VAE+Lsev":
            summary[m]["STRT_zc_only_LR"] = agg(lambda r, m=m: r[m]["STRT_zc_only"]["LR"])
            summary[m]["z_b_std"] = agg(lambda r, m=m: r[m]["z_b_std"])
        summary[m]["MIA_AUROC"] = agg(lambda r, m=m: r[m]["privacy"]["MIA_AUROC"])
        summary[m]["NND_ratio"] = agg(lambda r, m=m: r[m]["privacy"]["NND_ratio_vs_real"])
        summary[m]["severity_attack_balacc"] = agg(lambda r, m=m: r[m]["severity_attack"]["bal_acc"])
        vals = [runs[s][m]["sev_pred_vs_ADOS_test_r"] for s in SEEDS]
        if None not in vals:
            summary[m]["sevpred_vs_ADOS_r_test"] = agg(lambda r, m=m: r[m]["sev_pred_vs_ADOS_test_r"])
    for b in ["Brain-only_real(TRTR)", "SMOTE-only_STRT", "Concatenation_STRT"]:
        summary[b] = {c: agg(lambda r, b=b, c=c: r[b][c]) for c in ["LR", "RF"]}
    summary["SMOTE-only_privacy"] = {k: agg(lambda r, k=k: r["SMOTE-only_privacy"][k])
                                     for k in ["MIA_AUROC", "NND_mean", "NND_ratio_vs_real"]}
    summary["SMOTE-only_TRTS"] = {c: agg(lambda r, c=c: r["SMOTE-only_TRTS"][c]) for c in ["LR", "RF"]}

    # tests apparies (Wilcoxon) contre SR-DMLR, STRT LR
    from scipy.stats import wilcoxon, levene
    ref = np.array(summary["SR-DMLR"]["STRT_LR"]["per_seed"])
    tests = {}
    for name, vals in [("SMOTE-only", summary["SMOTE-only_STRT"]["LR"]["per_seed"]),
                       ("Concatenation", summary["Concatenation_STRT"]["LR"]["per_seed"]),
                       ("Flat-VAE+Lsev", summary["Flat-VAE+Lsev"]["STRT_LR"]["per_seed"]),
                       ("no_MMD", summary["no_MMD"]["STRT_LR"]["per_seed"]),
                       ("no_Lsev", summary["no_Lsev"]["STRT_LR"]["per_seed"]),
                       ("zc_only (brain-only synthesis)", summary["SR-DMLR"]["STRT_zc_only_LR"]["per_seed"])]:
        v = np.array(vals)
        tests[name] = {"mean_diff_SRDMLR_minus": round(float((ref - v).mean()), 4),
                       "wilcoxon_p": round(float(wilcoxon(ref, v).pvalue), 4)}
    g1 = np.array(summary["SR-DMLR"]["clinical_gradient_abs_rho"]["per_seed"])
    g2 = np.array(summary["Flat-VAE+Lsev"]["clinical_gradient_abs_rho"]["per_seed"])
    tests["clinical_gradient_variance_ratio_Flat_over_SRDMLR"] = round(float(g2.var(ddof=1) / g1.var(ddof=1)), 3)
    tests["clinical_gradient_levene_p"] = round(float(levene(g1, g2).pvalue), 4)
    summary["paired_tests"] = tests

    out = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "inputs_sha256": {p.name: sha(p) for p in FEAT.glob("*.npz")},
        "protocol": {"seeds": SEEDS, "abide_split": "70/30 stratifie, par seed",
                     "N_synth": N_SYNTH, "SMOTE_k": K_SMOTE, "latent": "3 x 8",
                     "labels": "diagnostic et severite (ADOS) : ABIDE uniquement"},
        "counts": {"ABIDE_ASD": int(A["y"].sum()), "ABIDE_TD": int((A["y"] == 0).sum()),
                   "ABIDE_ADOS": int(np.isfinite(A["ados"]).sum()),
                   "Emotion_images": int(len(E["X"])), "Engagnition_participants": int(len(G["X"]))},
        "summary": summary, "per_seed": runs,
    }
    Path("results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=1))
