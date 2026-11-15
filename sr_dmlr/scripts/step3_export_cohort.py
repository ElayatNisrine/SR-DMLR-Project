"""
STEP 3 : export de la cohorte synthetique SR-DMLR (N=250, seed 0), generateur gaussien par classe (choix de l'article).

Sorties :
  cohort_manifest/sr_dmlr_cohort_manifest.csv   publie : bloc clinique genere z_c (8), diagnostic synthetique,
                                                identifiants des enregistrements sources de z_b et z_x
  cohort_manifest/datasheet.json                publie : description, protocole, limites, empreinte SHA-256
  cohort/sr_dmlr_synthetic_cohort.csv           NON publie (.gitignore) : profils complets en 24 dimensions

Les blocs z_b et z_x derivent d'enregistrements individuels : ils ne sont pas publies et sont reconstruits
localement par rebuild_cohort.py, qui verifie l'empreinte.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import step2_pipeline as P
from cohort_common import SEED, ZD, DEC, encode_sources, profile_digest

P.check(P.E.get("files") is not None, "relancer step22_face_features.py : les chemins des images sont requis")
L = encode_sources(SEED)
Za_tr, Ze, Zg, ytr = L["Za_tr"], L["Ze"], L["Zg"], L["y_tr"]

# bloc clinique : gaussienne par classe ajustee sur les latents d'entrainement ABIDE-I
r = np.random.default_rng(SEED + 3); n1 = int(round(P.N_SYNTH * ytr.mean())); z = Za_tr[:, :ZD]
Zc = np.r_[r.multivariate_normal(z[ytr == 1].mean(0), np.cov(z[ytr == 1].T), n1),
           r.multivariate_normal(z[ytr == 0].mean(0), np.cov(z[ytr == 0].T), P.N_SYNTH - n1)]
ys = np.r_[np.ones(n1), np.zeros(P.N_SYNTH - n1)].astype(int)
rng = np.random.default_rng(SEED)
ib, ix = rng.integers(len(Ze), size=P.N_SYNTH), rng.integers(len(Zg), size=P.N_SYNTH)
S = np.c_[np.round(Zc, DEC), Ze[ib, ZD:2 * ZD], Zg[ix, 2 * ZD:]]

# verifications de la seed (memes mesures que l'article)
strt = P.clf_auc(S, ys, L["Za_te"], L["y_te"], SEED)
priv_c = P.privacy(Zc, Za_tr[:, :ZD], L["Za_te"][:, :ZD])
P.check(priv_c["NND_ratio_vs_real"] > 0.8, f"bloc clinique trop proche de l'entrainement ({priv_c})")

ids = [f"SYN{i:04d}" for i in range(P.N_SYNTH)]
man = pd.DataFrame(np.round(Zc, DEC), columns=[f"z_c_{i+1}" for i in range(ZD)])
man.insert(0, "profile_id", ids)
man["synthetic_diagnosis"] = ys          # 1 = ASD, 0 = TD ; porte par z_c genere, pas par les sources de z_b, z_x
man["z_b_source_image"] = np.asarray(P.E["files"])[ib]            # chemin relatif sous raw/emotion/
man["z_x_source_participant"] = np.asarray(P.G["pid"])[ix]
man["z_x_source_condition"] = np.asarray(P.G["condition"])[ix]
pub = Path("cohort_manifest"); pub.mkdir(exist_ok=True)
man.to_csv(pub / "sr_dmlr_cohort_manifest.csv", index=False, float_format=f"%.{DEC}f")

loc = Path("cohort"); loc.mkdir(exist_ok=True)
cols = [f"z_{b}_{i+1}" for b in "cbx" for i in range(ZD)]
full = pd.DataFrame(np.round(S, DEC), columns=cols); full.insert(0, "profile_id", ids); full["synthetic_diagnosis"] = ys
full.to_csv(loc / "sr_dmlr_synthetic_cohort.csv", index=False, float_format=f"%.{DEC}f")

sheet = {
    "name": "SR-DMLR synthetic zero-overlap ASD cohort (reconstruction manifest)",
    "created": time.strftime("%Y-%m-%d"),
    "n_profiles": int(P.N_SYNTH), "n_ASD": int(ys.sum()), "n_TD": int((ys == 0).sum()),
    "space": "24-dimensional latent space of the SR-DMLR encoder (3 blocks x 8), not original features",
    "released": "generated clinical block z_c, synthetic diagnosis, source identifiers of z_b and z_x",
    "not_released": "z_b and z_x (derived from individual source records); rebuilt locally by rebuild_cohort.py",
    "blocks": {"z_c": "sampled from class-conditional Gaussians fitted on ABIDE-I training latents; carries the diagnosis",
               "z_b": "latent code of one ASD Emotions image, independent of the diagnosis by construction",
               "z_x": "latent code of one Engagnition participant, independent of the diagnosis by construction"},
    "label_warning": "synthetic_diagnosis refers to the generated clinical block only; it is not a diagnosis "
                     "of the source participants identified in the provenance columns",
    "protocol": "seed 0; ABIDE-I 70/30 split before PCA, scaling and training; 150 epochs; class-conditional Gaussian sampler",
    "profiles_sha256": profile_digest(S), "rounding_decimals": DEC,
    "reproducibility": "the digest is matched when retraining reproduces the encoder exactly; on other hardware or "
                       "library versions training may not be bit-identical, and z_b, z_x then follow the same "
                       "procedure but not the same values",
    "checks_this_seed": {"STRT_AUROC": strt, "privacy_clinical_block": priv_c},
    "intended_use": ["benchmarking multimodal pipelines under controlled provenance",
                     "negative control for cross-modal diagnostic claims",
                     "privacy audits of block-wise synthetic data"],
    "not_suitable_for": ["learning links between facial or physiological variation and diagnosis",
                         "severity modelling", "estimating performance on real participants", "clinical conclusions"],
    "no_formal_privacy_guarantee": True,
}
(pub / "datasheet.json").write_text(json.dumps(sheet, indent=2))
print("manifest", man.shape, "ASD", int(ys.sum()), "TD", int((ys == 0).sum()), "STRT", strt, "sha", sheet["profiles_sha256"][:16])
