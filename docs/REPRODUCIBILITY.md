# Reproducibility workflow

Run from the repository root:

```bash
pip install -r requirements.txt
python src/run_all.py
```

The stages are:

1. `01_build_components.py` — reconstructs the nine normalized component scores and checks them against the frozen component panel.
2. `02_primary_fiui.py` — rebuilds the revised eight-component analysis matrix, Entropy/CRITIC/PCA/Equal weights, primary FIUI, descriptives and robustness tests.
3. `03_content_validity.py` — computes CVR, I-CVI, S-CVI/Ave and Cronbach alpha.
4. `04_nomological_and_sector_tests.py` — computes market-based criterion correlations and financial/nonfinancial diagnostics.
5. `05_round2_diagnostics.py` — regenerates Round-2 missingness and confirmatory-distribution diagnostics and preserves the frozen sensitivity/decomposition result tables.
6. `99_validate_reported_results.py` — compares regenerated results with the exact manuscript snapshots.

A successful run ends with all validation rows marked `True` in `outputs/99_validation_report.csv`.

## Rebuilding from the raw archives

The exact original extraction scripts are retained under `src/provenance/`. They document the historical CODAL/TSETMC data-construction pipeline. Because the raw archives contain source files collected at different times and the original scripts contain historical path assumptions, they are provided primarily for provenance. The analysis-ready `financial_panel_all_1392_1401.csv` is the stable starting point for reproducing the published article results.
