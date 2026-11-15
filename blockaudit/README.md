# blockaudit

Block-wise membership-inference auditing for synthetic records assembled from blocks of different provenance.

When a synthetic record concatenates a block generated from a sensitive population with blocks drawn
from other populations, an audit run on the complete record can hide leakage that an audit of the
sensitive block reveals: the foreign blocks dominate the distances and dilute the signal.
`blockaudit` runs the same attacks (DCR, kNN-5, DOMIAS) on every block and on the complete record,
and reports the gap.

## Install
    pip install -e .

## Use
```python
from blockaudit import audit_blocks
a = audit_blocks(synthetic, members, nonmembers, reference,
                 blocks={"clinical": slice(0, 8), "behavioural": slice(8, 16), "contextual": slice(16, 24)},
                 sensitive=["clinical"])
print(a.report())                     # AUROC per attack, block versus complete record
a.masking("clinical", "DOMIAS")       # positive = the complete-record audit underestimates leakage
```
All arrays share the assembled column space. `members` are training records of the sensitive population,
`nonmembers` held-out records, `reference` a further held-out sample used only by DOMIAS.
Real audited records must carry foreign blocks drawn independently of the synthetic ones.

## Attacks
- DCR: distance to the closest synthetic record.
- kNN-5: mean distance to the five closest synthetic records.
- DOMIAS (van Breugel et al., 2023): log density under the synthetic data minus log density under the reference, Gaussian KDE with Scott's rule.
AUROC 0.5 = no signal; above 0.5 = members are closer to, or denser under, the synthetic data.

## Recommendation
Audit every block derived from a sensitive source on its own, with at least one distance-based and one
density-based attack. A complete-record audit alone is not sufficient for assembled records.

License: MIT.

## PIAG (v0.2.0)
`BlockAudit.piag(block, attack)` renvoie le *provenance-induced audit gap* défini dans l'article :
|AUROC(bloc) − 0.5| − |AUROC(profil complet) − 0.5|. `masked_fraction()` renvoie 100 × PIAG / |AUROC(bloc) − 0.5|
(None si le PIAG n'est pas positif). `masking()` reste l'écart signé AUROC(bloc) − AUROC(complet), qui inclut la direction
de l'attaque et ne doit pas être confondu avec le PIAG.
Depuis la v0.2.1, `report()` signale un masquage seulement si |AUROC(bloc) − 0.5| ≥ 0.05 et PIAG > 0 ; il affiche le PIAG et la fraction masquée.
