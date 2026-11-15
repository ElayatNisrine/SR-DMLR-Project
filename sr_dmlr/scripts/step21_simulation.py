"""STEP 21 : etude par simulation du masquage (independante des donnees ASD).

Population : bloc sensible z_c ~ N(0, I_8) avec deux classes decalees ; blocs etrangers de dimension d_f, k blocs.
Membres (n=300) et non-membres (n=150), reference DOMIAS (n=150) tires de la meme population.
Generateurs du bloc sensible, ajustes sur les membres : SMOTE (fuite), gaussienne par classe (fuite faible).
Enregistrement synthetique = [bloc sensible genere ; k blocs etrangers tires d'une population independante].
Les enregistrements reels portent leurs propres blocs etrangers, independants de ceux du synthetique.
Controle meme origine : les blocs etrangers des membres sont copies dans le synthetique (SMOTE sur le
profil complet des membres), a dimension egale.
Attaques : DCR, kNN-5, DOMIAS (memes definitions que step13). 50 repetitions par configuration."""
import json, itertools, numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

def kde(X):
    return KernelDensity(bandwidth=X.std(0).mean() * len(X) ** (-1 / (X.shape[1] + 4))).fit(X)

def attacks(S, M, NM, REF):
    lab = np.r_[np.ones(len(M)), np.zeros(len(NM))]; Q = np.r_[M, NM]
    d = NearestNeighbors(n_neighbors=5).fit(S).kneighbors(Q)[0]
    dom = kde(S).score_samples(Q) - kde(REF).score_samples(Q)
    return {"DCR": roc_auc_score(lab, -d[:, 0]), "kNN5": roc_auc_score(lab, -d.mean(1)), "DOMIAS": roc_auc_score(lab, dom)}

def population(n, rng, dc=8):
    y = rng.integers(0, 2, n); z = rng.standard_normal((n, dc)); z[:, 0] += 1.0 * y
    return z, y

def gen(Z, y, n, rng, kind):
    if kind == "smote":
        cnt = np.bincount(y); Zs, ys = SMOTE(sampling_strategy={0: 3 * cnt[0], 1: 3 * cnt[1]}, k_neighbors=5,
                                             random_state=int(rng.integers(1e9))).fit_resample(Z, y)
        new = np.arange(len(Z), len(Zs)); pick = rng.choice(new, n, replace=len(new) < n)
        return Zs[pick]
    out = []
    for c in (0, 1):
        k = int(round(n * (y == c).mean())) if c == 0 else n - len(out[0])
        out.append(rng.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k))
    return np.r_[out[0], out[1]]

res = []
for kind, k, df, rep in itertools.product(("smote", "gauss"), (0, 1, 2, 4), (4, 8, 16), range(50)):
    if k == 0 and df != 8:
        continue
    rng = np.random.default_rng(10_000 * rep + 100 * k + df + (0 if kind == "smote" else 7))
    Zm, ym = population(300, rng); Zn, _ = population(150, rng); Zr, _ = population(150, rng)
    F = lambda n: rng.standard_normal((n, k * df))
    Sc = gen(Zm, ym, 250, rng, kind)
    S = np.c_[Sc, F(250)]; M = np.c_[Zm, F(300)]; NM = np.c_[Zn, F(150)]; RF = np.c_[Zr, F(150)]
    r = {"kind": kind, "k": k, "df": df, "rep": rep,
         "clinical": attacks(Sc, Zm, Zn, Zr), "full": attacks(S, M, NM, RF)}
    if kind == "smote" and k > 0:                          # meme origine : blocs etrangers des membres eux-memes
        cnt = np.bincount(ym); Ss = SMOTE(sampling_strategy={0: 3 * cnt[0], 1: 3 * cnt[1]}, k_neighbors=5,
                                          random_state=rep).fit_resample(M, ym)
        new = np.arange(len(M), len(Ss[0])); S_same = Ss[0][rng.choice(new, 250, replace=len(new) < 250)]
        r["same_origin"] = attacks(S_same, M, NM, RF)
    res.append(r)

json.dump(res, open("simulation_results.json", "w"))
summ = {}
for kind, k, df in sorted({(r["kind"], r["k"], r["df"]) for r in res}):
    R = [r for r in res if (r["kind"], r["k"], r["df"]) == (kind, k, df)]
    key = f"{kind}|k={k}|df={df}|foreign_dims={k * df}"
    summ[key] = {sp: {a: [round(float(np.mean([r[sp][a] for r in R])), 3),
                          round(float(np.percentile([np.mean(np.random.default_rng(0).choice([r[sp][a] for r in R], len(R))) for _ in range(2000)], 2.5)), 3),
                          round(float(np.percentile([np.mean(np.random.default_rng(1).choice([r[sp][a] for r in R], len(R))) for _ in range(2000)], 97.5)), 3)]
                      for a in ("DCR", "kNN5", "DOMIAS")}
                 for sp in ("clinical", "full", "same_origin") if sp in R[0]}
json.dump(summ, open("simulation_summary.json", "w"), indent=1)
for k_, v in summ.items():
    print(k_, {sp: v[sp]["DCR"][0] for sp in v}, {sp: v[sp]["DOMIAS"][0] for sp in v})
