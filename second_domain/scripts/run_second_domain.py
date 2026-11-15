"""Exécute le plan PLAN_second_domain.md (SHA-256 e4645280...). Aucune lecture de résultat avant la fin."""
import json, numpy as np, pmlb
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from blockaudit import audit_blocks

OUT = Path("results"); OUT.mkdir(exist_ok=True)
NAMES = ["pima", "wdbc", "saheart"]; N = 250
data = {}
import pandas as pd
FILES = {"pima": "_deprecated_pima", "wdbc": "breast_cancer_wisconsin_diagnostic", "saheart": "saheart"}
for n in NAMES:
    df = pd.read_csv(f"cache_{FILES[n]}.tsv.gz", sep="\t")
    data[n] = (df.drop(columns="target").to_numpy(float), df["target"].to_numpy().astype(int))

def enc_fit(X):
    d = min(8, X.shape[1]); sc = StandardScaler().fit(X); p = PCA(d, random_state=0).fit(sc.transform(X))
    return lambda Z: p.transform(sc.transform(Z))

def gauss(Z, y, n, rng):
    out, lab = [], []
    for c in (0, 1):
        k = int(round(n * (y == c).mean())) if c == 0 else n - len(out[0])
        out.append(rng.multivariate_normal(Z[y == c].mean(0), np.cov(Z[y == c].T), k)); lab += [c] * k
    return np.r_[out[0], out[1]]

def smote(Z, y, n, seed):
    cnt = np.bincount(y); Zs, _ = SMOTE(sampling_strategy={0: 3 * cnt[0], 1: 3 * cnt[1]}, k_neighbors=5,
                                         random_state=seed).fit_resample(Z, y)
    new = np.arange(len(Z), len(Zs)); return Zs[np.random.default_rng(seed).choice(new, n, replace=len(new) < n)]

res = []
for sens in NAMES:
    X, y = data[sens]; foreign = [f for f in NAMES if f != sens]
    pools = {f: enc_fit(data[f][0])(data[f][0]) for f in foreign}
    for s in range(30):
        rng = np.random.default_rng(s)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=s)
        e = enc_fit(X[tr]); Ztr, Zte = e(X[tr]), e(X[te])
        te_p = rng.permutation(len(te)); nm, ref = te_p[: len(te) // 2], te_p[len(te) // 2:]
        mem = rng.choice(len(tr), len(nm), replace=False)
        d = Ztr.shape[1]
        blocks = {"sensitive": slice(0, d)}; off = d
        for f in foreign:
            k = pools[f].shape[1]; blocks[f] = slice(off, off + k); off += k
        F = lambda n: np.concatenate([pools[f][rng.integers(len(pools[f]), size=n)] for f in foreign], 1)
        M, NM, R = np.c_[Ztr[mem], F(len(mem))], np.c_[Zte[nm], F(len(nm))], np.c_[Zte[ref], F(len(ref))]
        row = {"dataset": sens, "seed": s, "d_sensitive": d, "d_full": off}
        gens = {"smote": smote(Ztr, y[tr], N, s), "gauss": gauss(Ztr, y[tr], N, rng),
                "negative": gauss(Zte[ref], y[te][ref], N, rng)}
        for g, Sc in gens.items():
            a = audit_blocks(np.c_[Sc, F(N)], M, NM, R, blocks, sensitive=["sensitive"])
            row[g] = {"sensitive": a.blocks["sensitive"], "full": a.full, "foreign_share": a.foreign_distance_share}
        res.append(row)
    print(sens, "done", flush=True)
json.dump(res, open(OUT / "raw_results.json", "w"))

# analyse pré-enregistrée
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
summ, tests = {}, []
for sens in NAMES:
    R_ = [r for r in res if r["dataset"] == sens]
    for g in ("smote", "gauss", "negative"):
        for att in ("DCR", "kNN5", "DOMIAS"):
            b = np.array([r[g]["sensitive"][att] for r in R_]); f = np.array([r[g]["full"][att] for r in R_])
            summ[f"{sens}|{g}|{att}"] = {"sensitive": round(b.mean(), 3), "full": round(f.mean(), 3), "diff": round((b - f).mean(), 3)}
            if g == "smote":
                tests.append((f"{sens}|{att}", wilcoxon(b - f).pvalue))
    summ[f"{sens}|smote|foreign_share"] = round(float(np.mean([r["smote"]["foreign_share"] for r in R_])), 3)
ph = multipletests([p for _, p in tests], method="holm")[1]
summ["confirmatory"] = {k: {"p": float(p), "p_holm": float(h)} for (k, p), h in zip(tests, ph)}
rep = {}
for sens in NAMES:
    rep[sens] = all(summ["confirmatory"][f"{sens}|{a}"]["p_holm"] < 0.05 and summ[f"{sens}|smote|{a}"]["diff"] > 0 for a in ("DCR", "DOMIAS"))
summ["masking_reproduced"] = rep; summ["n_reproduced"] = int(sum(rep.values()))
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
print("analyse terminee")
