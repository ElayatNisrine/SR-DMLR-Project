"""STEP 26 : provenance-induced audit gap (PIAG) a partir des audits par seed (audit/seed_*.json).
PIAG = |AUC_bloc-0.5| - |AUC_full-0.5| par seed ; bootstrap 10 000 sur les seeds ; Wilcoxon + Holm (6 paires)."""
import json,numpy as np
from scipy.stats import wilcoxon
rng=np.random.default_rng(2026)
D=[json.load(open(f'audit/seed_{s}.json')) for s in range(30)]
res={};ps=[]
for q in ['smote','gauss']:
  for a in ['DCR','kNN5','DOMIAS']:
    b=np.array([d[f'{q}|clinical'][a] for d in D]); f=np.array([d[f'{q}|full'][a] for d in D])
    Sb=abs(b-.5); Sf=abs(f-.5); P=Sb-Sf
    idx=rng.integers(0,30,(10000,30)); bp=P[idx].mean(1); bfr=100*bp/Sb[idx].mean(1)
    res[(q,a)]=dict(Ab=b.mean(),Af=f.mean(),Sb=Sb.mean(),Sf=Sf.mean(),P=P.mean(),ci=np.percentile(bp,[2.5,97.5]),
                    fr=100*P.mean()/Sb.mean(),frci=np.percentile(bfr,[2.5,97.5]),p=wilcoxon(P).pvalue)
    ps.append(((q,a),res[(q,a)]['p']))
ps.sort(key=lambda x:x[1]); prev=0
for i,(k,p) in enumerate(ps):
  prev=min(1,max(prev,(len(ps)-i)*p)); res[k]['holm']=prev
json.dump({f'{k[0]}|{k[1]}':{kk:np.asarray(v).tolist() for kk,v in r.items()} for k,r in res.items()},open('piag_results.json','w'),indent=1)
for k,r in res.items(): print(k,round(r['P'],4),np.round(r['ci'],4),round(r['fr'],1),np.round(r['frci'],1),r['holm'])
