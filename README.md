# SR-DMLR: code and results

Code, aggregated results, and the `blockaudit` package for the article
"Source-Resolved Disentangled Multimodal Representation Learning (SR-DMLR) for Autism Research:
Privacy Auditing in Zero-Overlap Synthetic Cohorts" (submitted to IEEE Access).

## Contents
- `sr_dmlr/scripts/` : full pipeline (feature extraction, tripartite VAE, synthesis, evaluation, audits, controls).
- `sr_dmlr/results/` : aggregated per-seed results (JSON) and logs that produced the values of the article.
  No source record, no individual feature, no per-participant prediction and no synthetic cohort is included.
  The leave-one-site-out results are given as per-site AUROC (`loso_site_auroc.json`).
- `blockaudit/` : block-wise versus full-profile membership-inference audit (`audit_blocks`, `BlockAudit.piag`).
- `second_domain/` : PMLB replication (protocol, SHA-256 of the protocol, execution log, scripts, results).
- `sr_dmlr/results/cohort_datasheet.json` : datasheet of the synthetic cohort (250 profiles, seed 0).
- `DATASET_INSTRUCTIONS.md` : how to obtain the three source datasets.

## Environment
Python 3.13; `pip install -r requirements.txt`; `pip install -e blockaudit`. Experiments ran on CPU (two threads);
one complete seed takes about 140 s. Hyperparameters are fixed in `step2_pipeline.py` (`train_model`:
KL weight 1e-3, loss weights 1, Adam 1e-3, 150 epochs, RBF bandwidth 2*d).

## Running
From `sr_dmlr/` (with the data under `sr_dmlr/raw/`, see `DATASET_INSTRUCTIONS.md`):

```
python scripts/step1_features.py raw
python scripts/step22_face_features.py
SEEDS=$(seq -s, 0 29) python scripts/step6_30.py            # main SR-DMLR runs (30 seeds)
SEEDS=$(seq -s, 0 29) python scripts/step10_sensitivity.py  # baselines on the same splits
python scripts/step19_unified_stats.py                      # utility table and tests
SEEDS=$(seq -s, 0 29) python scripts/step13_audit.py         # block vs full-profile audit
python scripts/step14_audit_stats.py
python scripts/step26_piag.py                                # provenance-induced audit gap
SEEDS=$(seq -s, 0 29) python scripts/step17_controls.py      # negative and encoder controls
python scripts/step20_dilution.py                            # dimensionality vs provenance
python scripts/step21_simulation.py                          # simulation
SEEDS=$(seq -s, 0 29) python scripts/step27_blocks_grouped_trainfit.py   # per-block downstream tasks
python scripts/step28_blocks_summary.py blocks_trainfit blocks_trainfit_summary.json
python scripts/step18_sites.py; python scripts/step23_siteharm.py; python scripts/step25_loso.py
python scripts/step24_flat_audit.py; python scripts/step8_zdim.py
```

The per-block downstream evaluation is produced by `step27_blocks_grouped_trainfit.py`, which fits all preprocessing
of the facial and physiological features on the training part of each seed only.

## SMOTE sampler
`step2_pipeline.smote_sample` triples each class by interpolation and draws the cohort only from newly created points,
so no training record is copied. All SMOTE-dependent analyses were run with this sampler: `step30_privacy_rerun.py`
(audit, controls, dilution, flat-VAE audit, SMOTE evaluation; one SR-DMLR and one flat-VAE training per seed),
`step31_patch_baselines.py` (SMOTE-only and concatenation baselines, which train no model, on the original splits),
`step32_merge_final.py`, then `step19_unified_stats.py`, `step14_audit_stats.py`, `step26_piag.py`, `step29_summaries.py`,
`make_figures_30.py` and `make_fig7_loso.py`.

## Synthetic cohort
The cohort is not redistributed, because its behavioural and contextual blocks derive from individual source records.
It is regenerated from locally obtained sources with `python scripts/step3_export_cohort.py` (seed 0), which writes
`cohort/sr_dmlr_synthetic_cohort.csv` and its datasheet; `cohort/` is excluded from version control. The diagnosis
column refers to the generated clinical block only, not to the source participants.

## Second domain
`second_domain/PLAN_second_domain.md` was fixed before data loading; `PLAN.sha256` holds its hash. The hash
certifies that the plan was not altered, not the date at which it was written. Deviations are dated in `LOG.md`.

## Scripts not used for the article
`step2_pipeline_v1_descriptors.py`, `step5_privacy_zc.py`, `step6_final.py`, `step7_diag.py`, `step9_final_summary.py`,
`step12_stats.py` and `step16_sens30.py` are earlier development steps kept for transparency. `step4_revision.py` is not
run directly but is imported by several scripts.

## License
MIT (see `LICENSE`).
