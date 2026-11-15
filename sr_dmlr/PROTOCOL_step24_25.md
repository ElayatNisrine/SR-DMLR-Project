# Protocole fixe AVANT execution (2026-10-24)
## step24 : audit du VAE plat (30 seeds, memes splits, memes attaques que step13)
- VAE plat (flat=True), latent 24 d sans blocs. Generateurs : gaussien par classe et SMOTE sur le latent ABIDE complet.
- Attaques DCR, kNN-5, DOMIAS ; memes membres / non-membres / reference que step13.
- Rapport : moyenne, IC bootstrap 95 %, comparaison descriptive avec SR-DMLR (bloc clinique et profil complet).
## step25 : leave-one-site-out (20 sites ABIDE-I)
- Un pli par site ; entrainement sur les 19 autres. Seed du pli = indice du site.
- Methodes : SR-DMLR (gaussien), VAE plat (gaussien), SMOTE-only, Concatenation, Reel.
- Deux traitements des sites, tous deux rapportes : aucun ; centrage par site (transductif, sans label).
- Critere principal : AUROC par site, sites contenant les deux classes (20/20) ; test de Wilcoxon apparie sur les 20 sites,
  SR-DMLR contre SMOTE et SR-DMLR contre Concatenation et SR-DMLR contre VAE plat ; Holm sur les 3 comparaisons x 2 traitements = 6.
- Secondaire : AUROC poolee (predictions de tous les plis concatenees), IC bootstrap 2000 participants ; difference appariee bootstrap.
- Aucun autre test ni variante ne sera ajoute apres lecture des resultats.
