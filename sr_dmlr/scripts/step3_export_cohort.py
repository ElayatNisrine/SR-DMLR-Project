"""
STEP 3 : export de la cohorte synthetique SR-DMLR (N=250), meme pipeline que step2 (seed 0).

Sorties (cohort/):
  sr_dmlr_synthetic_cohort.csv   24 dimensions latentes + diagnostic + provenance
  datasheet.json                 description, protocole, metriques de la seed, limites
Aucune severite n'est exportee : elle ne se transfere pas aux donnees reelles (r ~ 0.05).
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import step2_pipeline as P
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 0
A, E, G, ZD = P.A, P.E, P.G, P.ZD
y, ados = A["y"], A["ados"]
tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
pca = PCA(64, random_state=SEED).fit(A["X"][tr])
sc = StandardScaler().fit(pca.transform(A["X"][tr]))
Xa_tr = sc.transform(pca.transform(A["X"][tr])).astype(np.float32)
Xa_te = sc.transform(pca.transform(A["X"][te])).astype(np.float32)
Xe = StandardScaler().fit_transform(E["X"]).astype(np.float32)
Xg = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
T = lambda a: torch.as_tensor(a, dtype=torch.float32)
data = {"abide": dict(X=T(Xa_tr), y=T(y[tr]), sev=T(ados[tr])),
        "emotion": dict(X=T(Xe), y=None, sev=None),
        "engagement": dict(X=T(Xg), y=None, sev=None)}

model = P.train_model(data, SEED)
Za_tr, Za_te = P.latents(model, Xa_tr, "abide"), P.latents(model, Xa_te, "abide")
Ze, Zg = P.latents(model, Xe, "emotion"), P.latents(model, Xg, "engagement")

# generateur gaussien par classe (SMOTE fait fuiter les sujets d'entrainement, cf. step5)
_r = np.random.default_rng(SEED + 3); _n1 = int(round(P.N_SYNTH * y[tr].mean())); _z = Za_tr[:, :ZD]; _y = y[tr]
Zc = np.r_[_r.multivariate_normal(_z[_y == 1].mean(0), np.cov(_z[_y == 1].T), _n1),
           _r.multivariate_normal(_z[_y == 0].mean(0), np.cov(_z[_y == 0].T), P.N_SYNTH - _n1)]
ys = np.r_[np.ones(_n1), np.zeros(P.N_SYNTH - _n1)].astype(int)
rng = np.random.default_rng(SEED)
ib, ix = rng.integers(len(Ze), size=P.N_SYNTH), rng.integers(len(Zg), size=P.N_SYNTH)
S = np.c_[Zc, Ze[ib, ZD:2 * ZD], Zg[ix, 2 * ZD:]]

# verification : memes metriques que dans l'article pour cette seed
strt = P.clf_auc(S, ys, Za_te, y[te], SEED)
priv = P.privacy(S, Za_tr, Za_te)
priv_c = P.privacy(Zc, Za_tr[:, :ZD], Za_te[:, :ZD])
P.check(priv_c["NND_ratio_vs_real"] > 0.8, f"bloc clinique trop proche de l'entrainement ({priv_c})")

out = Path("cohort"); out.mkdir(exist_ok=True)
cols = [f"z_c_{i+1}" for i in range(ZD)] + [f"z_b_{i+1}" for i in range(ZD)] + [f"z_x_{i+1}" for i in range(ZD)]
df = pd.DataFrame(S.round(5), columns=cols)
df.insert(0, "participant_id", [f"SYN{i:04d}" for i in range(len(df))])
df["diagnosis"] = ys.astype(int)  # 1 = ASD, 0 = TD (herite de z_c, source ABIDE-I)
df["z_b_source"] = "ASD Emotions"
df["z_x_source"] = "Engagnition (" + pd.Series(G["condition"][ix]).values + ")"
df.to_csv(out / "sr_dmlr_synthetic_cohort.csv", index=False)

sheet = {
    "name": "SR-DMLR synthetic zero-overlap ASD cohort",
    "created": time.strftime("%Y-%m-%d"),
    "n_participants": int(len(df)), "n_ASD": int(ys.sum()), "n_TD": int((ys == 0).sum()),
    "space": "24-dimensional latent space of the SR-DMLR encoder (not original features)",
    "blocks": {"z_c (8)": "clinical, sampled from class-conditional Gaussians fitted on ABIDE-I training latents; carries the diagnosis",
               "z_b (8)": "behavioural, drawn from ASD Emotions latents, independent of diagnosis",
               "z_x (8)": "contextual, drawn from Engagnition latents, independent of diagnosis"},
    "protocol": "seed 0; ABIDE-I 70/30 split before PCA, scaling, training and SMOTE; 150 epochs",
    "checks_this_seed": {"STRT_AUROC": strt, "privacy_24d": priv, "privacy_clinical_block": priv_c},
    "generator": "class-conditional Gaussian in z_c (SMOTE rejected: leaks training participants)",
    "intended_use": ["benchmarking pipelines without access to real records",
                     "studying utility-privacy trade-offs in zero-overlap synthesis"],
    "not_suitable_for": ["learning links between facial or physiological variation and diagnosis",
                         "any severity-dependent modelling (no valid severity is provided)",
                         "clinical conclusions"],
    "inputs_sha256": {p.name: P.sha(p) for p in Path("features").glob("*.npz")},
}
(out / "datasheet.json").write_text(json.dumps(sheet, indent=2))
print(df.shape, "ASD", int(ys.sum()), "TD", int((ys == 0).sum()), "STRT", strt, "clin", priv_c)
