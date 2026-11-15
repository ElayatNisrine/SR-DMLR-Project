"""blockaudit : audit de confidentialité par bloc pour des données synthétiques assemblées.

Idée : quand un enregistrement synthétique assemble un bloc généré à partir de données sensibles
et des blocs provenant d'autres populations, un audit sur le profil complet peut masquer une fuite
que l'audit du bloc sensible seul révèle. blockaudit lance les mêmes attaques d'appartenance
sur chaque bloc et sur le profil complet, et signale l'écart.
"""
from .attacks import dcr_auroc, knn_auroc, domias_auroc, run_attacks
from .audit import BlockAudit, audit_blocks

__all__ = ["dcr_auroc", "knn_auroc", "domias_auroc", "run_attacks", "BlockAudit", "audit_blocks"]
__version__ = "0.2.1"
