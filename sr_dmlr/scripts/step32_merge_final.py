"""STEP 32 : final_results.json pour step19 / figures : resultats d'origine (rerun_v2) pour le sampler gaussien et
les variantes (sortie de step6_30.py), entrees 'smote' de SR-DMLR et du VAE plat remplacees par la relance step30 (sampler corrige)."""
import json, glob
F = json.load(open("final_results.json"))      # ecrit par step6_30.py
for f in glob.glob("step6_new/seed_*.json"):
    s = f.split("_")[-1][:-5]; n = json.load(open(f))
    for m in ("SR-DMLR", "Flat-VAE"): F[s][m]["smote"] = n[m]["smote"]
json.dump(F, open("final_results.json", "w"), indent=1); print("merged", len(F))
