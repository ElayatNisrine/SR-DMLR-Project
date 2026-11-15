"""STEP 17 : controles de validite des attaques (30 seeds).
(a) Controle negatif : pseudo-membres = moitie A1 du test, non-membres = moitie A2, reference = moitie B.
    Aucun pseudo-membre n'a ete vu par le modele : toute AUROC differente de 0.5 serait un artefact.
(b) Controle encodeur : cohorte gaussienne ajustee sur B1 (participants jamais vus), reference DOMIAS = B2 ;
    membres (train) contre non-membres (A). Une AUROC > 0.5 indique que les codes des membres different
    de ceux des non-membres a cause de l'encodeur, independamment du generateur."""
import os, json, numpy as np
import step4_revision as R, step2_pipeline as P
from step13_audit import attacks, gauss
from pathlib import Path
ZD, N = P.ZD, P.N_SYNTH; OUT = Path("controls"); OUT.mkdir(exist_ok=True)
for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
    f = OUT / f"seed_{s}.json"
    if f.exists(): continue
    d = R.prep(s); ytr, yte = d["ytr"], d["yte"]; _, L = R.train(d, s)
    rng = np.random.default_rng(s + 50); te = rng.permutation(len(yte)); A, B = te[: len(te) // 2], te[len(te) // 2:]
    A1, A2 = A[: len(A) // 2], A[len(A) // 2:]; B1, B2 = B[: len(B) // 2], B[len(B) // 2:]
    mem = rng.choice(len(ytr), len(A), replace=False)
    r2 = np.random.default_rng(s); ib = r2.integers(len(L["e"]), size=N); ix = r2.integers(len(L["g"]), size=N)
    res = {}
    for gen in ("gauss", "smote"):
        zc, _ = gauss(L["a_tr"][:, :ZD], ytr, N, s) if gen == "gauss" else P.smote_sample(L["a_tr"][:, :ZD], ytr, N, s)
        S = np.c_[zc, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
        for sp, k in (("clinical", slice(0, ZD)), ("full", slice(None))):
            Te = L["a_te"][:, k]; res[f"negctrl|{gen}|{sp}"] = attacks(S[:, k], Te[A1], Te[A2], Te[B])
    zc_u, _ = gauss(L["a_te"][B1, :ZD], yte[B1], N, s)                 # generateur sans aucun membre
    Su = np.c_[zc_u, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
    for sp, k in (("clinical", slice(0, ZD)), ("full", slice(None))):
        Tr, Te = L["a_tr"][:, k], L["a_te"][:, k]; res[f"encoder|{sp}"] = attacks(Su[:, k], Tr[mem], Te[A], Te[B2])
    f.write_text(json.dumps(res)); print(s, {k: round(v["DOMIAS"], 3) for k, v in res.items()}, flush=True)
