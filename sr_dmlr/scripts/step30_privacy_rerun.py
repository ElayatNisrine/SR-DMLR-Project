"""STEP 30 : relance, avec le sampler SMOTE corrige, de toutes les analyses de confidentialite qui dependent de SMOTE.
Par seed : SR-DMLR entraine UNE fois, puis audit (corps de step13), controles (step17), dilution (step20) et
evaluation step6 (STRT, confidentialite) pour gauss et SMOTE ; VAE plat entraine une fois : audit plat (step24) et
evaluation step6. Ecrit les memes formats de fichiers que les scripts d'origine. Usage : SEEDS=0,1,... python step30_privacy_rerun.py"""
import os, json, numpy as np
import step4_revision as R, step2_pipeline as P
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
ns = {}; src = open("step6_30.py").read(); exec(src[:src.index("OUTF =")], ns); evaluate = ns["evaluate"]
ns13 = {}; s13 = open("step13_audit.py").read(); exec(s13[:s13.index("for s in")], ns13); attacks, gauss = ns13["attacks"], ns13["gauss"]
ZD, N = P.ZD, P.N_SYNTH
for d_ in ("audit", "controls", "dilution", "flat_audit", "step6_new"): Path(d_).mkdir(exist_ok=True)
for s in [int(x) for x in os.environ["SEEDS"].split(",")]:
    if Path(f"step6_new/seed_{s}.json").exists(): continue
    d = R.prep(s); ytr, yte = d["ytr"], d["yte"]; _, L = R.train(d, s)
    # --- audit (step13)
    rng = np.random.default_rng(s + 50); te = rng.permutation(len(yte)); nm, ref = te[: len(te) // 2], te[len(te) // 2:]
    mem = rng.choice(len(ytr), len(nm), replace=False)
    r2 = np.random.default_rng(s); ib = r2.integers(len(L["e"]), size=N); ix = r2.integers(len(L["g"]), size=N)
    res = {}
    for gen in ("gauss", "smote"):
        zc, _ = gauss(L["a_tr"][:, :ZD], ytr, N, s) if gen == "gauss" else P.smote_sample(L["a_tr"][:, :ZD], ytr, N, s)
        S = np.c_[zc, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
        for space, k in (("clinical", slice(0, ZD)), ("full", slice(None))):
            Tr, Te = L["a_tr"][:, k], L["a_te"][:, k]
            res[f"{gen}|{space}"] = {**attacks(S[:, k], Tr[mem], Te[nm], Te[ref]), "NND_ratio": P.privacy(S[:, k], Tr, Te)["NND_ratio_vs_real"]}
        Q = L["a_tr"][mem]; j = NearestNeighbors(n_neighbors=1).fit(S).kneighbors(Q)[1][:, 0]; diff2 = (Q - S[j]) ** 2
        res[f"{gen}|share_bx"] = float((diff2[:, ZD:].sum(1) / diff2.sum(1)).mean())
    Path(f"audit/seed_{s}.json").write_text(json.dumps(res))
    # --- controles (step17)
    rng = np.random.default_rng(s + 50); te = rng.permutation(len(yte)); A_, B_ = te[: len(te) // 2], te[len(te) // 2:]
    A1, A2 = A_[: len(A_) // 2], A_[len(A_) // 2:]; B1, B2 = B_[: len(B_) // 2], B_[len(B_) // 2:]
    memc = rng.choice(len(ytr), len(A_), replace=False)
    r2 = np.random.default_rng(s); ib = r2.integers(len(L["e"]), size=N); ix = r2.integers(len(L["g"]), size=N)
    rc = {}
    for gen in ("gauss", "smote"):
        zc, _ = gauss(L["a_tr"][:, :ZD], ytr, N, s) if gen == "gauss" else P.smote_sample(L["a_tr"][:, :ZD], ytr, N, s)
        S = np.c_[zc, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
        for sp, k in (("clinical", slice(0, ZD)), ("full", slice(None))):
            Te = L["a_te"][:, k]; rc[f"negctrl|{gen}|{sp}"] = attacks(S[:, k], Te[A1], Te[A2], Te[B_])
    zc_u, _ = gauss(L["a_te"][B1, :ZD], yte[B1], N, s); Su = np.c_[zc_u, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
    for sp, k in (("clinical", slice(0, ZD)), ("full", slice(None))):
        Tr, Te = L["a_tr"][:, k], L["a_te"][:, k]; rc[f"encoder|{sp}"] = attacks(Su[:, k], Tr[memc], Te[A_], Te[B2])
    Path(f"controls/seed_{s}.json").write_text(json.dumps(rc))
    # --- dilution (step20)
    rng = np.random.default_rng(s + 50); te = rng.permutation(len(yte)); nm, ref = te[: len(te) // 2], te[len(te) // 2:]
    mem = rng.choice(len(ytr), len(nm), replace=False)
    r2 = np.random.default_rng(s); ib = r2.integers(len(L["e"]), size=N); ix = r2.integers(len(L["g"]), size=N)
    zc, _ = P.smote_sample(L["a_tr"][:, :ZD], ytr, N, s); S = np.c_[zc, L["e"][ib, ZD:2 * ZD], L["g"][ix, 2 * ZD:]]
    Tr, Te = L["a_tr"], L["a_te"]; mu, sd = Tr.mean(0), Tr.std(0) + 1e-8
    def run(W, SS=S):
        g = lambda X: (X - mu) / sd * W
        return attacks(g(SS), g(Tr[mem]), g(Te[nm]), g(Te[ref]))
    rd = {"clinical": attacks(S[:, :ZD], Tr[mem, :ZD], Te[nm, :ZD], Te[ref, :ZD]), "full_raw": attacks(S, Tr[mem], Te[nm], Te[ref]),
          "full_std": run(np.ones(3 * ZD)), "full_blockeq": run(np.r_[np.full(ZD, 1 / np.sqrt(ZD)), np.full(2 * ZD, 1 / np.sqrt(2 * ZD))])}
    for w in (0.5, 0.25): rd[f"full_w{w}"] = run(np.r_[np.ones(ZD), np.full(2 * ZD, w)])
    S_same, _ = P.smote_sample(L["a_tr"], ytr, N, s); rd["same_origin"] = attacks(S_same, Tr[mem], Te[nm], Te[ref])
    Path(f"dilution/seed_{s}.json").write_text(json.dumps(rd))
    # --- evaluation step6, SR-DMLR
    r6 = {"SR-DMLR": {g: evaluate(L, d, s, g) for g in ("gauss", "smote")}}
    # --- VAE plat : audit (step24) + evaluation step6
    _, Lf = R.train(d, s, flat=True)
    rng = np.random.default_rng(s + 50); te = rng.permutation(len(yte)); nm, ref = te[: len(te) // 2], te[len(te) // 2:]
    mem = rng.choice(len(ytr), len(nm), replace=False); Tr, Te = Lf["a_tr"], Lf["a_te"]; rf = {}
    for gen in ("gauss", "smote"):
        S, _ = gauss(Tr, ytr, N, s) if gen == "gauss" else P.smote_sample(Tr, ytr, N, s)
        rf[gen] = {**attacks(S, Tr[mem], Te[nm], Te[ref]), "NND_ratio": P.privacy(S, Tr, Te)["NND_ratio_vs_real"]}
    Path(f"flat_audit/seed_{s}.json").write_text(json.dumps(rf))
    r6["Flat-VAE"] = {g: evaluate(Lf, d, s, g, flat=True) for g in ("gauss", "smote")}
    Path(f"step6_new/seed_{s}.json").write_text(json.dumps(r6)); print(s, "ok", flush=True)
