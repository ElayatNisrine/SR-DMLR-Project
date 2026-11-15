"""
STEP 1 : extraction des features depuis raw/ (aucune etiquette inventee)

Sorties (features/):
  abide.npz       X_conn (N, 19900) Fisher-z upper triangle, y (0=TD, 1=ASD),
                  ados (NaN si absent), site, sub_id
  emotion.npz     X (1199, 26), emotion
  engagement.npz  X (57, 14), mask (57, 14), condition, pid

Regles :
  - diagnostic ABIDE = dossier (ASD / TDC), verifie contre DX_GROUP du CSV
  - Emotions et Engagnition : AUCUN diagnostic (les donnees n'en fournissent pas)
"""
import sys, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
OUT = Path("features"); OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- ABIDE
pheno = pd.read_csv(RAW / "abide/Phenotypic_V1_0b_preprocessed1.csv")
pheno = pheno.set_index("FILE_ID")

X, y, ados, site, sid, n_bad = [], [], [], [], [], 0
for label, folder in [(1, "ASD"), (0, "TDC")]:
    for f in sorted((RAW / "abide" / folder).glob("*.1D")):
        fid = f.name.replace("_rois_cc200.1D", "")
        ts = np.loadtxt(f, comments="#")
        if ts.ndim != 2 or ts.shape[0] < 50:
            n_bad += 1; continue
        ts = ts[:, ts.std(0) > 0] if False else ts
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(ts.T)
        c = np.nan_to_num(c)
        iu = np.triu_indices(c.shape[0], 1)
        z = np.arctanh(np.clip(c[iu], -0.999, 0.999)).astype(np.float32)
        # verification du label contre le CSV
        assert fid in pheno.index, f"{fid} absent du CSV"
        dx = int(pheno.loc[fid, "DX_GROUP"])
        assert (dx == 1) == (label == 1), f"Label dossier != DX_GROUP pour {fid}"
        a = pheno.loc[fid, "ADOS_TOTAL"]
        X.append(z); y.append(label); site.append(pheno.loc[fid, "SITE_ID"])
        ados.append(float(a) if pd.notna(a) and float(a) > 0 else np.nan)
        sid.append(fid)
X = np.stack(X)
np.savez_compressed(OUT / "abide.npz", X=X, y=np.array(y), ados=np.array(ados),
                    site=np.array(site), sub_id=np.array(sid))
print(f"ABIDE : {len(y)} sujets  ASD={sum(y)}  TD={len(y)-sum(y)}  "
      f"ADOS dispo={np.isfinite(ados).sum()}  rejetes={n_bad}  dim={X.shape[1]}")

# ---------------------------------------------------------------- EMOTION
def img_feats(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), np.float32) / 255
    a = a.transpose(2, 0, 1)
    f = []
    for ch in a: f += [ch.mean(), ch.std()]
    f += [a.mean(), a.std()]
    g = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
    h, _ = np.histogram(g, 10, (0, 1)); f += list(h / h.sum())
    p, _ = np.histogram(g, 256, (0, 1)); p = p / p.sum() + 1e-10
    f.append(-(p * np.log2(p)).sum())
    for i in range(7):
        r, cc = (i // 4) * 16, (i % 4) * 16
        f.append(g[r:r + 16, cc:cc + 16].var())
    return np.array(f, np.float32)

EX, EL = [], []
for emo in ["anger", "fear", "joy", "Natural", "sadness", "surprise"]:
    for p in sorted((RAW / "emotion" / emo).glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            EX.append(img_feats(p)); EL.append(emo)
EX = np.stack(EX); assert EX.shape[1] == 26
np.savez_compressed(OUT / "emotion.npz", X=EX, emotion=np.array(EL))
print(f"EMOTION : {len(EX)} images, dim=26, diagnostic=AUCUN")

# ---------------------------------------------------------------- ENGAGNITION
NAMES = ["gsr_mean", "gsr_std", "gsr_peaks", "acc_mean", "acc_std", "tmp_mean",
         "tmp_slope", "gaze_on_rate", "gaze_switches", "perf_rate",
         "perf_n", "eng_mean", "eng_std", "eng_drops"]

def eng_feats(d):
    f = dict.fromkeys(NAMES, np.nan)
    g = pd.read_csv(d / "E4GsrData.csv")["GSR"].dropna().values
    if len(g) > 60:
        b, a = signal.butter(4, 0.5, fs=4, btype="low")
        gf = signal.filtfilt(b, a, g)
        ph = gf - signal.savgol_filter(gf, 51, 3)
        f.update(gsr_mean=gf.mean(), gsr_std=gf.std(),
                 gsr_peaks=len(signal.find_peaks(ph)[0]) / (len(g) / 4 / 60))
    acc = pd.read_csv(d / "E4AccData.csv")
    svm = np.sqrt((acc[["Acc_X", "Acc_Y", "Acc_Z"]] ** 2).sum(1))
    f.update(acc_mean=svm.mean(), acc_std=svm.std())
    t = pd.read_csv(d / "E4TmpData.csv")["ST"].dropna().values
    f.update(tmp_mean=t.mean(), tmp_slope=np.polyfit(np.arange(len(t)), t, 1)[0])
    if (d / "GazeData.csv").exists():
        gz = pd.read_csv(d / "GazeData.csv")["Gaze"].values
        f.update(gaze_on_rate=gz.mean(), gaze_switches=np.abs(np.diff(gz)).sum() / len(gz))
    if (d / "PerformanceData.csv").exists():
        pf = pd.read_csv(d / "PerformanceData.csv")["Performance"].values
        f.update(perf_rate=pf.mean(), perf_n=len(pf))
    if (d / "EngagementData.csv").exists():
        e = pd.read_csv(d / "EngagementData.csv")["Engagement"].values
        f.update(eng_mean=e.mean(), eng_std=e.std(), eng_drops=(np.diff(e) < 0).sum() / len(e))
    return np.array([f[n] for n in NAMES], np.float64)

GX, cond, pid = [], [], []
for c in ["Baseline condition", "HPE condition", "LPE condition"]:
    for d in sorted((RAW / "engagnition" / c).glob("P*")):
        GX.append(eng_feats(d)); cond.append(c.split()[0]); pid.append(d.name)
GX = np.stack(GX); M = np.isfinite(GX)
assert len(set(pid)) == len(pid), "participant en double"
np.savez_compressed(OUT / "engagement.npz", X=np.nan_to_num(GX).astype(np.float32),
                    mask=M.astype(np.float32), condition=np.array(cond),
                    pid=np.array(pid), names=np.array(NAMES))
print(f"ENGAGNITION : {len(GX)} participants uniques, dim=14, "
      f"features presentes={M.mean():.0%} (Baseline sans gaze/perf/engagement), diagnostic=AUCUN")
