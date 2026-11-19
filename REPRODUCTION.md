# Reproduction instructions

## 1. Obtain the input data

The scripts expect user-supplied input files. They are intentionally not bundled here.

At minimum, the core panel is expected to be provided locally as:

`PAPER GEOPOLITICA_.dta`

The deterministic UCDP pipeline additionally requires the official UCDP Conflict Termination Dataset v4-2024 CSV and, for historical comparison, the archived 22-episode crosswalk. Obtain source data directly from the relevant provider and retain the source version, retrieval date, attribution, and applicable terms.

## 2. Install Python dependencies

```bash
python -m pip install -r code/requirements.txt
```

## 3. Core analysis and audit commands

Run from the repository root. Replace the example input paths with the locations of the files you have lawfully obtained.

```bash
python code/v7_official_cpi_audit.py --panel "/path/to/PAPER GEOPOLITICA_.dta" --output "reproduction/check_cpi"

python code/v7_fe_contrast_reestimate.py --panel "/path/to/PAPER GEOPOLITICA_.dta" --output "reproduction/check_fe"

python code/v7_run_ucdp_deterministic.py --panel "/path/to/PAPER GEOPOLITICA_.dta" --ucdp "/path/to/UCDPConflictTerminationDataset_v4_2024_Conflict.csv" --legacy-22 "/path/to/V6_UCDP_22_original_episode_crosswalk.csv" --output "reproduction/check_ucdp"

python code/v7_temporal_recheck.py --input "/path/to/PAPER GEOPOLITICA_.dta" --output "reproduction/check_forecast" --figures "reproduction/figures"
```

Other scripts in `code/` perform additional diagnostics, predictive-loss sensitivity, horizon analysis, and frozen-result checks. Inspect each script's `--help` and docstring before execution because some require archived prediction files or derived outputs in addition to the master panel.

## 4. Preserve outputs

Write new outputs to a new `reproduction/` directory. Do not overwrite archived or frozen result files. Record:
- input file names and SHA-256 hashes;
- source version and retrieval date;
- Python and package versions;
- command-line arguments;
- execution logs;
- generated tables and figures;
- any warnings or failed checks.

## 5. What this release does not certify

The commands above are entry points, not a claim that one command recreates the full Word/PDF manuscript. A full clean-room rebuild of every manuscript table and figure remains a separate final reproducibility gate.
