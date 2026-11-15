"""Analyse de sensibilite POST HOC (demandee en relecture, 2026-10-31) : Pima avec zeros impossibles traites comme manquants.
Variables : glucose, pression diastolique, epaisseur cutanee, insuline, IMC ; imputation par la mediane du split d'entrainement.
Tout le reste est identique a run_second_domain.py (meme seeds, memes blocs etrangers, memes attaques)."""
import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon
from blockaudit import audit_blocks
FILES={"pima":"_deprecated_pima","wdbc":"breast_cancer_wisconsin_diagnostic","saheart":"saheart"}
D={n:pd.read_csv(f"cache_{f}.tsv.gz",sep="\t") for n,f in FILES.items()}
ZERO=['plasma glucose','Diastolic blood pressure','Triceps skin fold thickness','2-Hour serum insulin','Body mass index']
def enc_fit(X):
    d=min(8,X.shape[1]); sc=StandardScaler().fit(X); p=PCA(d,random_state=0).fit(sc.transform(X)); return lambda Z:p.transform(sc.transform(Z))
def smote(Z,y,n,seed):
    c=np.bincount(y); Zs,_=SMOTE(sampling_strategy={0:3*c[0],1:3*c[1]},k_neighbors=5,random_state=seed).fit_resample(Z,y)
    new=np.arange(len(Z),len(Zs)); return Zs[np.random.default_rng(seed).choice(new,n,replace=len(new)<n)]
df=D["pima"].copy(); y=df.pop("target").to_numpy().astype(int); df[ZERO]=df[ZERO].replace(0,np.nan)
foreign=["wdbc","saheart"]; pools={}
for f in foreign:
    X=D[f].drop(columns="target").to_numpy(float); pools[f]=enc_fit(X)(X)
rows=[]
for s in range(30):
    rng=np.random.default_rng(s)
    tr,te=train_test_split(np.arange(len(y)),test_size=0.3,stratify=y,random_state=s)
    med=df.iloc[tr].median(); X=df.fillna(med).to_numpy(float)
    e=enc_fit(X[tr]); Ztr,Zte=e(X[tr]),e(X[te])
    tp=rng.permutation(len(te)); nm,ref=tp[:len(te)//2],tp[len(te)//2:]; mem=rng.choice(len(tr),len(nm),replace=False)
    d=Ztr.shape[1]; blocks={"sensitive":slice(0,d)}; off=d
    for f in foreign: k=pools[f].shape[1]; blocks[f]=slice(off,off+k); off+=k
    F=lambda n: np.concatenate([pools[f][rng.integers(len(pools[f]),size=n)] for f in foreign],1)
    M,NM,R=np.c_[Ztr[mem],F(len(mem))],np.c_[Zte[nm],F(len(nm))],np.c_[Zte[ref],F(len(ref))]
    a=audit_blocks(np.c_[smote(Ztr,y[tr],250,s),F(250)],M,NM,R,blocks,sensitive=["sensitive"])
    rows.append({"sensitive":a.blocks["sensitive"],"full":a.full})
out={}
for att in ["DCR","kNN5","DOMIAS"]:
    b=np.array([r["sensitive"][att] for r in rows]); f=np.array([r["full"][att] for r in rows])
    out[att]={"sensitive":round(b.mean(),3),"full":round(f.mean(),3),"diff":round((b-f).mean(),3),"p":float(wilcoxon(b-f).pvalue)}
json.dump(out,open("results/sensitivity_pima_missing.json","w"),indent=1); json.dump(rows,open("results/sensitivity_pima_missing_perseed.json","w")); print(json.dumps(out,indent=1))
