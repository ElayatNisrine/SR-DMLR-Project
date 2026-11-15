# Source data

No source record and no feature derived from an individual record is distributed here.
Obtain each dataset from its provider, under its own terms of use, and place it under `sr_dmlr/raw/`:

| Folder | Dataset | Provider | Expected content |
|---|---|---|---|
| `raw/abide/` | ABIDE-I, Preprocessed Connectomes Project (CC200, `filt_global`) | http://preprocessed-connectomes-project.org/abide/ | `Phenotypic_V1_0b_preprocessed1.csv` and one folder per diagnostic group with the `*.1D` ROI time series |
| `raw/emotion/` | ASD Emotions (Talaat, 2023) | Kaggle | one folder per emotion: `anger`, `fear`, `joy`, `Natural`, `sadness`, `surprise` |
| `raw/engagnition/` | Engagnition (Kim et al., Scientific Data 11:299, 2024, doi:10.1038/s41597-024-03132-3) | link given in the data descriptor | one folder per condition containing the `P*` participant folders |

Then run, from `sr_dmlr/`:

```
python scripts/step1_features.py raw      # features/abide.npz, emotion.npz, engagement.npz
python scripts/step22_face_features.py    # features/emotion_facenet.npz (MTCNN + FaceNet/VGGFace2)
```

The `features/` folder created by these steps contains individual-level derived data and must not be redistributed.
