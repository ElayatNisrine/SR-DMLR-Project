"""Audit par bloc et comparaison avec l'audit du profil complet."""
from dataclasses import dataclass, field
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .attacks import run_attacks


@dataclass
class BlockAudit:
    blocks: dict                 # nom -> AUROC par attaque, pour chaque bloc
    full: dict                   # AUROC par attaque, profil complet
    foreign_distance_share: float  # part de la distance au plus proche synthétique due aux blocs non sensibles
    sensitive: list = field(default_factory=list)

    def masking(self, block, attack="DOMIAS"):
        """Écart signé AUROC(bloc) - AUROC(profil complet) (inclut la direction). Voir piag() pour la mesure de l'article."""
        return self.blocks[block][attack] - self.full[attack]

    def piag(self, block, attack="DOMIAS"):
        """Provenance-induced audit gap : |AUROC(bloc)-0.5| - |AUROC(complet)-0.5|.

        Compare les écarts absolus au hasard, donc ne confond pas un AUROC complet < 0.5
        (signal inversé) avec une absence de signal. Positif = signal détectable perdu dans l'audit global.
        """
        return abs(self.blocks[block][attack] - 0.5) - abs(self.full[attack] - 0.5)

    def masked_fraction(self, block, attack="DOMIAS"):
        """100 * PIAG / |AUROC(bloc)-0.5| ; None si le signal du bloc est nul ou si le PIAG n'est pas positif."""
        sb = abs(self.blocks[block][attack] - 0.5)
        g = self.piag(block, attack)
        return None if sb == 0 or g <= 0 else 100.0 * g / sb

    def report(self, threshold=0.05):
        """Rapport par bloc sensible et par attaque, aligné sur piag().

        Un masquage est signalé quand le signal du bloc |AUROC-0.5| atteint `threshold` et que le PIAG est
        positif (le profil complet s'écarte moins du hasard que le bloc). Un AUROC complet < 0.5 dont l'écart
        absolu dépasse celui du bloc n'est donc pas signalé comme masquage.
        """
        lines = []
        for b in self.sensitive:
            for a, v in self.blocks[b].items():
                f = self.full[a]; g = self.piag(b, a)
                flag = "  <-- fuite masquée par l'audit global" if abs(v - 0.5) >= threshold and g > 0 else ""
                fm = self.masked_fraction(b, a)
                fm_txt = f" fraction masquée={fm:.0f}%" if fm is not None else ""
                lines.append(f"{b:>12} | {a:6} bloc={v:.3f} complet={f:.3f} PIAG={g:+.3f}{fm_txt}{flag}")
        lines.append(f"part de distance due aux blocs non sensibles : {self.foreign_distance_share:.1%}")
        return "\n".join(lines)


def audit_blocks(synthetic, members, nonmembers, reference, blocks, sensitive=None):
    """Lance les attaques sur chaque bloc et sur le profil complet.

    synthetic, members, nonmembers, reference : tableaux (n, d) dans le même espace assemblé.
    blocks : dict nom -> slice (ou liste d'indices) des colonnes de chaque bloc.
    sensitive : noms des blocs issus de la population à protéger (par défaut : tous).
    """
    sensitive = list(blocks) if sensitive is None else list(sensitive)
    res = {b: run_attacks(synthetic[:, s], members[:, s], nonmembers[:, s], reference[:, s])
           for b, s in blocks.items()}
    full = run_attacks(synthetic, members, nonmembers, reference)
    idx = np.zeros(synthetic.shape[1], bool)
    for b in sensitive:
        idx[np.arange(synthetic.shape[1])[blocks[b]]] = True
    j = NearestNeighbors(n_neighbors=1).fit(synthetic).kneighbors(members)[1][:, 0]
    d2 = (members - synthetic[j]) ** 2
    share = float((d2[:, ~idx].sum(1) / d2.sum(1)).mean()) if (~idx).any() else 0.0
    return BlockAudit(res, full, share, sensitive)
