# Cohort manifest

`sr_dmlr_cohort_manifest.csv` and `datasheet.json` are produced by `scripts/step3_export_cohort.py` (seed 0).
They contain the generated clinical blocks, the synthetic diagnosis, the source identifiers of the behavioural and
contextual blocks, and the SHA-256 digest of the complete profiles. Use `scripts/rebuild_cohort.py` to rebuild the
complete profiles from locally obtained source data.
