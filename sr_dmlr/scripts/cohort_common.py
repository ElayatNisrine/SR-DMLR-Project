"""Code partage par step3_export_cohort.py (auteurs) et rebuild_cohort.py (utilisateurs).
Entraine le VAE tripartite (seed 0, meme protocole que l'article) et renvoie les latents des trois sources."""
import hashlib
import numpy as np, torch
import step2_pipeline as P
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED, ZD, DEC = 0, P.ZD, 5          # graine, dimension d'un bloc, arrondi des latents publies


def encode_sources(seed=SEED):
    A, E, G = P.A, P.E, P.G
    y, ados = A["y"], A["ados"]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    pca = PCA(64, random_state=seed).fit(A["X"][tr])
    sc = StandardScaler().fit(pca.transform(A["X"][tr]))
    Xa_tr = sc.transform(pca.transform(A["X"][tr])).astype(np.float32)
    Xa_te = sc.transform(pca.transform(A["X"][te])).astype(np.float32)
    Xe = StandardScaler().fit_transform(E["X"]).astype(np.float32)
    Xg = StandardScaler().fit_transform(G["X"]).astype(np.float32) * G["mask"]
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    data = {"abide": dict(X=T(Xa_tr), y=T(y[tr]), sev=T(ados[tr])),
            "emotion": dict(X=T(Xe), y=None, sev=None),
            "engagement": dict(X=T(Xg), y=None, sev=None)}
    model = P.train_model(data, seed)
    return dict(Za_tr=P.latents(model, Xa_tr, "abide"), Za_te=P.latents(model, Xa_te, "abide"),
                Ze=P.latents(model, Xe, "emotion"), Zg=P.latents(model, Xg, "engagement"),
                y_tr=y[tr], y_te=y[te])


def profile_digest(S):
    """SHA-256 des profils complets (24 dimensions, arrondis a DEC decimales, ordre des lignes du manifeste)."""
    txt = "\n".join(",".join(f"{v:.{DEC}f}" for v in row) for row in np.round(S, DEC))
    return hashlib.sha256(txt.encode()).hexdigest()
