# Data dictionary

## `data/processed/financial_panel_all_1392_1401.csv`

Analysis-ready firm-year panel assembled from CODAL financial statements and TSETMC market data. Key field groups include:

- identifiers: `symbol`, `company_name`, `ins_code`, `isin`, `fiscal_year`;
- CODAL report metadata: announcement IDs, reporting/publication dates, audit/consolidation flags and source links;
- financial-statement items: assets, liabilities, equity, revenue, net income, operating cash flow and related items;
- derived accounting ratios and growth measures;
- market variables: annual raw return, annualized volatility, trading value/volume, Amihud illiquidity, shares outstanding and year-end market capitalization;
- data-quality fields: extraction item count and parser/matching scores.

## `data/processed/company_year_components_and_legacy_indices.csv`

Contains the nine original normalized FIUI components, the legacy weighting-method indices, and firm-year ranking outputs. The revised paper uses these component scores as the starting empirical component panel.

## `data/survey/expert_cvr_cvi_deidentified.csv`

150 rows = 15 experts × 10 theoretical components. Fields contain expert code, broad background characteristics, essential/useful/not-essential judgment, and 1–4 relevance/clarity/simplicity ratings.

## `data/survey/expert_questionnaire_36item_deidentified.csv`

15 expert records. Contains four broad respondent-background columns followed by the 36 Likert items used for the main Cronbach-alpha calculation. Timestamps and free-text comments are excluded.

## `data/metadata/sector_classification.csv`

Company-level TSE sector code/name with an indicator for bank/insurance exclusion.

## `data/metadata/banks_insurance_symbols.csv`

The 17 firms classified in TSE sector 57 (banks and credit institutions) or 66 (insurance/pension excluding social security) for the sector robustness test.

## `reported_results/`

Frozen numerical snapshots used in the manuscript/rebuttal. They are intentionally versioned so a future library update cannot silently alter reported values.
