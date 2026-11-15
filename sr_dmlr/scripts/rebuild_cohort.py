"""
Reconstruit les profils complets (24 dimensions) de la cohorte SR-DMLR a partir du manifeste publie et des
copies locales des trois sources, puis verifie l'empreinte SHA-256 donnee dans la datasheet.

Prerequis (depuis sr_dmlr/) :
  python scripts/step1_features.py raw
  python scripts/step22_face_features.py
Usage :
  python scripts/rebuild_cohort.py            -> cohort/sr_dmlr_synthetic_cohort_rebuilt.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import step2_pipeline as P
from cohort_common import SEED, ZD, DEC, encode_sources, profile_digest

pub = Path("cohort_manifest")
man = pd.read_csv(pub / "sr_dmlr_cohort_manifest.csv")
sheet = json.loads((pub / "datasheet.json").read_text())
P.check(P.E.get("files") is not None, "relancer step22_face_features.py : les chemins des images sont requis")

files = {f: i for i, f in enumerate(np.asarray(P.E["files"]))}
pids = {p: i for i, p in enumerate(np.asarray(P.G["pid"]))}
miss_b = sorted(set(man["z_b_source_image"]) - set(files)); miss_x = sorted(set(man["z_x_source_participant"]) - set(pids))
if miss_b or miss_x:
    sys.exit(f"enregistrements sources absents de la copie locale : {len(miss_b)} images, {len(miss_x)} participants "
             f"(ex. {miss_b[:3]} {miss_x[:3]})")

L = encode_sources(SEED)                                   # meme entrainement que step3 (seed 0)
ib = man["z_b_source_image"].map(files).to_numpy(); ix = man["z_x_source_participant"].map(pids).to_numpy()
Zc = man[[f"z_c_{i+1}" for i in range(ZD)]].to_numpy()
S = np.c_[Zc, L["Ze"][ib, ZD:2 * ZD], L["Zg"][ix, 2 * ZD:]]

out = Path("cohort"); out.mkdir(exist_ok=True)
cols = [f"z_{b}_{i+1}" for b in "cbx" for i in range(ZD)]
full = pd.DataFrame(np.round(S, DEC), columns=cols); full.insert(0, "profile_id", man["profile_id"])
full["synthetic_diagnosis"] = man["synthetic_diagnosis"]
full.to_csv(out / "sr_dmlr_synthetic_cohort_rebuilt.csv", index=False, float_format=f"%.{DEC}f")

d = profile_digest(S)
if d == sheet["profiles_sha256"]:
    print("OK : profils reconstruits identiques a la cohorte publiee (SHA-256", d[:16] + "...)")
else:
    print("ATTENTION : empreinte differente. Le manifeste et la procedure sont respectes, mais l'entrainement "
          "n'est pas bit a bit identique dans cet environnement ; z_b et z_x suivent la meme procedure, "
          "pas les memes valeurs.")
    sys.exit(1)
