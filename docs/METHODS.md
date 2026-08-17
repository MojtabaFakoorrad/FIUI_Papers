# Methods implemented in the repository

## 1. Component construction

The nine original component scores are constructed at the firm-year level. Raw measures are winsorized within fiscal year at the 1st and 99th percentiles and min-max scaled so that larger scores indicate greater information usefulness.

- **Predictive value:** inverse absolute next-year net-income growth proxy.
- **Confirmatory value:** agreement between current net-income growth and annual stock return.
- **Faithful representation:** inverse absolute total accruals scaled by average assets.
- **Neutrality:** original duplicated accrual proxy; retained only in the legacy nine-component dataset.
- **Structured-data completeness:** number of standardized financial-statement items extracted.
- **Extraction confidence:** average parser/matching confidence.
- **Accounting-return mapping stability:** inverse residual error from a rolling within-firm mapping of ROA on stock return.
- **Timeliness:** inverse reporting/publication lag.
- **Conservatism:** negative signed accrual ratio transformed to the common positive orientation.

These are empirical proxies and should not be interpreted as complete semantic measurements of the corresponding IASB concepts.

## 2. Eligibility and missing values

A firm-year enters the index analysis if at least five of the original nine component scores are directly observed. The revised primary index then removes neutrality and uses eight dimensions. The exact analysis matrix used in the final revised paper is reconstructed in `common.paper_locked_matrix()`.

## 3. Weighting

- **Entropy:** weights statistical dispersion/information content.
- **CRITIC:** weights dispersion and non-redundancy based on inter-component correlations.
- **Primary:** normalized arithmetic mean of Entropy and CRITIC.
- **Equal:** one-eighth per component, used as a benchmark.
- **PCA:** absolute loading contribution of retained principal components reaching at least 80% cumulative explained variance, used as a benchmark.

## 4. Robustness

The code reproduces:

- original 9-component vs revised 8-component rank agreement;
- equal-weight benchmark;
- PCA benchmark;
- complete-case recalibration;
- re-estimation after excluding 17 banks and insurance firms;
- component correlation matrix;
- missingness by component;
- confirmatory-value distribution diagnostics.

## 5. Expert validation

The CVR uses Lawshe's formula with N=15. I-CVI is the proportion of experts assigning 3 or 4 on relevance, clarity, or simplicity. The scale-level CVI is the average of component I-CVIs. Cronbach alpha is computed from the 36 main questionnaire items.

## 6. Nomological tests

Spearman correlations relate the primary FIUI to current- and next-year Amihud illiquidity, annualized return volatility, and mean daily trading value. These tests are diagnostic and the manuscript does not claim successful external validity.
