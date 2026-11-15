# Plan pré-enregistré : audit par bloc sur un second domaine (hors autisme)
Figé le : voir horodatage et SHA-256 ci-dessous, AVANT tout chargement de données ou calcul d'attaque.

## But
Tester si le masquage observé dans SR-DMLR (l'audit du profil complet cache une fuite que l'audit du bloc sensible révèle)
se reproduit hors des données autisme, sur des données médicales tabulaires publiques assemblées par blocs.

## Données (choisies par règle, avant tout calcul)
Règle : jeux PMLB médicaux, classification binaire, n >= 450, variables numériques.
Retenus : pima (diabète, n=768), wdbc (cancer du sein, n=569), saheart (cardiopathie, n=462). Trois populations sans participant commun.
Assemblage par rotation : pour chaque jeu "sensible", les deux autres jeux fournissent les blocs étrangers.
Chaque bloc : standardisation + ACP à d = min(8, nb variables) dimensions. Bloc sensible ajusté sur l'entraînement uniquement ; blocs étrangers ajustés sur leur pool complet.

## Protocole (identique en nature à SR-DMLR, step13)
- 30 seeds (0..29). Split stratifié 70/30 du jeu sensible. Test coupé en deux moitiés : non-membres / référence DOMIAS.
- Membres : tirage aléatoire dans l'entraînement, même effectif que les non-membres.
- Générateurs du bloc sensible, ajustés sur l'entraînement : SMOTE (k=5) et gaussienne par classe ; N = 250 enregistrements.
- Enregistrement synthétique = [bloc sensible généré ; blocs étrangers rééchantillonnés dans leurs pools].
  Enregistrement réel audité = [bloc sensible réel ; blocs étrangers tirés indépendamment dans les mêmes pools].
- Attaques : DCR, kNN-5, DOMIAS (KDE gaussien, règle de Scott). Espaces : bloc sensible seul, profil complet.
- Contrôle négatif : générateur ajusté sur la moitié référence (aucun membre) ; AUROC attendue ~0,5.

## Test confirmatoire
Générateur SMOTE. Pour chaque jeu (3) et chaque attaque (3) : différence appariée par seed AUROC(bloc) - AUROC(profil complet),
Wilcoxon bilatéral, n = 30. Famille Holm de 9 tests, alpha = 0,05.
Critère « masquage reproduit » pour un jeu : DCR et DOMIAS significatifs après Holm avec différence > 0.
Résultat principal rapporté : nombre de jeux (sur 3) où le masquage est reproduit, quel qu'il soit.

## Descriptif (sans test)
Générateur gaussien ; contrôle négatif ; part de la distance au plus proche synthétique due aux blocs étrangers.

## Règles
Aucun changement de jeux, de dimensions, de générateurs ou d'attaques après lecture. Toute correction technique est datée dans LOG.md.
