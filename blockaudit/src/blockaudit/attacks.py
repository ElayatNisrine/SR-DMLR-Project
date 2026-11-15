"""Attaques d'appartenance sans entraînement de modèle d'attaque.

Toutes renvoient l'AUROC de la séparation membres (étiquette 1) / non-membres (0) :
0,5 = aucun signal ; > 0,5 = les membres sont plus proches / plus denses sous le synthétique.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity, NearestNeighbors


def _labels(members, nonmembers):
    return np.r_[np.ones(len(members)), np.zeros(len(nonmembers))], np.r_[members, nonmembers]


def _kde(X):
    bw = X.std(0).mean() * len(X) ** (-1 / (X.shape[1] + 4))  # règle de Scott
    return KernelDensity(bandwidth=bw).fit(X)


def dcr_auroc(synthetic, members, nonmembers):
    """Distance au plus proche enregistrement synthétique (DCR)."""
    y, Q = _labels(members, nonmembers)
    d = NearestNeighbors(n_neighbors=1).fit(synthetic).kneighbors(Q)[0][:, 0]
    return float(roc_auc_score(y, -d))


def knn_auroc(synthetic, members, nonmembers, k=5):
    """Distance moyenne aux k plus proches enregistrements synthétiques."""
    y, Q = _labels(members, nonmembers)
    d = NearestNeighbors(n_neighbors=k).fit(synthetic).kneighbors(Q)[0].mean(1)
    return float(roc_auc_score(y, -d))


def domias_auroc(synthetic, members, nonmembers, reference):
    """DOMIAS (van Breugel et al., 2023) : log p_synth(x) - log p_ref(x), densités par KDE."""
    y, Q = _labels(members, nonmembers)
    s = _kde(synthetic).score_samples(Q) - _kde(reference).score_samples(Q)
    return float(roc_auc_score(y, s))


def run_attacks(synthetic, members, nonmembers, reference):
    return {"DCR": dcr_auroc(synthetic, members, nonmembers),
            "kNN5": knn_auroc(synthetic, members, nonmembers),
            "DOMIAS": domias_auroc(synthetic, members, nonmembers, reference)}
