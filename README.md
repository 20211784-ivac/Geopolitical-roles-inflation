# Paper 1 — AI-Classified Geopolitical Actor Roles and Inflation

**Manuscript:** *Do AI-Classified Geopolitical Actor Roles Improve Inflation Assessment? A Source-Audited Global Panel and Out-of-Time Incremental Information Test.*

This repository contains the analysis and audit scripts associated with Paper 1. It does **not** contain the merged country-month source panel, raw AI-GPR data, raw CPI source files, or other third-party datasets. Those inputs must be obtained from their original providers, subject to their terms.

## Authors

1. Ivanna Cubas-Cubas — National Agrarian University; Semillero de Investigación en Economía de los Recursos Naturales y del Ambiente (SIERNA).
2. José Dávila-Garcia — ESAN University.
3. Alissa Bravo Guillén — Ministry of Economy and Finance of Peru; National University of San Marcos.
4. Carlos Orihuela Romero — National Agrarian University; Semillero de Investigación en Economía de los Recursos Naturales y del Ambiente (SIERNA).

Author order and affiliations are reproduced from the author information supplied for the manuscript. Corresponding author, email addresses, ORCID identifiers, CRediT roles, funding, and competing-interest declarations must be completed and approved by all authors before formal journal submission.

## Scope and research status

The scripts cover:
- source-screened fixed-effects estimation and common-scale Direct–Spillover contrasts;
- retrospective chronological prediction checks;
- CPI source-identity checks for selected 2024 observations;
- UCDP conflict-event screening and descriptive event-stack diagnostics;
- predictive-loss, influence, bootstrap, and horizon sensitivity analyses;
- frozen-result verification and economic-unit translation.

The preferred empirical sample reported in the manuscript contains 141 countries and 33,780 fixed-effects observations. The repository does not claim that the archived AI-GPR panel is a real-time vintage, that the UCDP 22-episode selection rule has been fully recovered, or that full 2000–2025 CPI histories for Gabon, the United Kingdom, and the United States have been reconstructed.

## Repository layout

- `code/` — Python analysis and audit scripts.
- `README.md` — project overview and limitations.
- `REPRODUCTION.md` — input requirements and execution order.
- `DATA_SOURCES_AND_REUSE.md` — source attribution and redistribution boundary.
- `CITATION.cff` — citation metadata.
- `LICENSE_PENDING.md` — reuse rights are not granted until all authors approve a license.
- `CODE_AUDIT.md` — static code checks performed for this release.
- `SHA256SUMS.txt` — integrity hashes for released files.

## Software

Python 3.11 or later is recommended. Install dependencies with:

```bash
python -m pip install -r code/requirements.txt
```

The scripts were syntax-compiled as part of the release-preparation audit. This check does not constitute a full clean-room reproduction of every manuscript table and figure.

## Important limitations

1. Historical AI-GPR extraction timestamp/provider vintage is not independently verified.
2. CPI checks validate selected official 2024 source identities; they do not reconstruct all excluded national CPI histories.
3. The original upstream selection rule for the 22 UCDP episodes remains unavailable. The transparent 30-episode screen is a separate analysis.
4. The retrospective prediction exercise is not a real-time forecast evaluation.
5. Some sensitivity analyses are post hoc and are labelled as such in the manuscript and supplement.

## Version

Code source archive: `CODIGO_ANALISIS_GEOPOLITICA_V19.zip` (project archive).  
Manuscript editorial baseline: V28.  
This public-code preparation does not modify the empirical scripts or results.

## Contact

The corresponding author and contact email will be added after author confirmation.
