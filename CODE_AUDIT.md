# Static code audit — public-code preparation

Scope: the 16 Python scripts copied unchanged from the archived code-only V19 package.

Checks performed:
- Python syntax compilation with the current execution environment.
- Static scan for common absolute-path patterns (`C:\`, `/Users/`, `/home/`, `/mnt/data`).
- Static scan for simple hard-coded credential patterns (`api_key`, `token`, `password`, `secret`).
- No raw data files are included in this release.

These checks are not a full security audit, license audit of every dependency, or clean-room reproduction of the manuscript. A clean-room analytical reproduction remains a separate requirement.

| Script | Bytes | Syntax | Absolute-path scan | Credential scan |
|---|---:|---|---|---|
| `V10_OIL_CONTENT_BENCHMARK.py` | 8,081 | PASS | No absolute path pattern | No embedded-secret pattern |
| `V11_robust_loss.py` | 5,990 | PASS | No absolute path pattern | No embedded-secret pattern |
| `V13_COUNTRY_CALENDAR_BOOTSTRAP.py` | 2,488 | PASS | No absolute path pattern | No embedded-secret pattern |
| `V13_FORECAST_HORIZONS.py` | 8,751 | PASS | No absolute path pattern | No embedded-secret pattern |
| `V15_ECONOMIC_SCALE.py` | 2,006 | PASS | No absolute path pattern | No embedded-secret pattern |
| `V15_OFFICIAL_CPI_2024_AUDIT.py` | 3,667 | PASS | No absolute path pattern | No embedded-secret pattern |
| `V15_VERIFY_FROZEN_RESULTS.py` | 3,176 | PASS | No absolute path pattern | No embedded-secret pattern |
| `VERIFY_V13_RESULTS.py` | 1,890 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v7_fe_contrast_reestimate.py` | 2,992 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v7_fe_replication_base.py` | 25,204 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v7_freeze_sensitivities_figures.py` | 4,430 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v7_official_cpi_audit.py` | 3,945 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v7_run_ucdp_deterministic.py` | 3,524 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v7_temporal_recheck.py` | 8,267 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v7_ucdp_deterministic_pipeline.py` | 65,866 | PASS | No absolute path pattern | No embedded-secret pattern |
| `v9_diagnostics.py` | 9,231 | PASS | No absolute path pattern | No embedded-secret pattern |
