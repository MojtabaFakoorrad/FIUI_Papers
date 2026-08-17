from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata, t
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

COMPONENTS_9 = [
    "predictive_value_score",
    "confirmatory_value_score",
    "faithful_representation_score",
    "neutrality_score",
    "completeness_score",
    "understandability_score",
    "comparability_score",
    "timeliness_score",
    "conservatism_score",
]

COMPONENTS_8 = [c for c in COMPONENTS_9 if c != "neutrality_score"]

DISPLAY_NAMES = {
    "predictive_value_score": "Predictive value",
    "confirmatory_value_score": "Confirmatory value",
    "faithful_representation_score": "Faithful representation",
    "completeness_score": "Structured-data completeness",
    "understandability_score": "Extraction confidence",
    "comparability_score": "Accounting-return mapping stability",
    "timeliness_score": "Timeliness",
    "conservatism_score": "Conservatism",
}

# Paper-lock value used in the final revised analysis snapshot when the last fiscal
# year (1401) had no directly observable next-year predictive-value score.
# Keeping this constant makes the GitHub repository reproduce the exact reported
# Entropy/CRITIC/PCA weights and Table 5 year statistics in the revised paper.
PREDICTIVE_LAST_YEAR_FALLBACK = 0.8826104529535383


def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    return w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)


def entropy_weights(A):
    A = np.asarray(A, dtype=float)
    P = A / np.maximum(A.sum(axis=0, keepdims=True), 1e-12)
    k = 1.0 / math.log(A.shape[0])
    ent = -k * np.sum(np.where(P > 0, P * np.log(np.maximum(P, 1e-300)), 0.0), axis=0)
    return normalize_weights(1.0 - ent)


def critic_weights(A):
    A = np.asarray(A, dtype=float)
    sd = A.std(axis=0, ddof=1)
    R = np.nan_to_num(np.corrcoef(A, rowvar=False), nan=0.0)
    contrast = np.sum(1.0 - R, axis=1)
    return normalize_weights(sd * contrast)


def pca_weights(A, explained_threshold=0.80):
    A = np.asarray(A, dtype=float)
    sd = A.std(axis=0, ddof=1)
    Z = (A - A.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    model = PCA().fit(Z)
    cum = np.cumsum(model.explained_variance_ratio_)
    k = int(np.searchsorted(cum, explained_threshold) + 1)
    loadings = model.components_[:k, :].T * np.sqrt(model.explained_variance_[:k])
    contrib = (np.abs(loadings) * model.explained_variance_ratio_[:k]).sum(axis=1)
    return normalize_weights(contrib), k, float(cum[k - 1])


def rank_desc(w):
    return rankdata(-np.asarray(w), method="average")


def load_components():
    p = DATA / "processed" / "company_year_components_and_legacy_indices.csv"
    return pd.read_csv(p)


def eligible_mask(df):
    # Exact original eligibility criterion: at least five of the nine empirical
    # component scores were directly available before imputation.
    return df[COMPONENTS_9].notna().sum(axis=1) >= 5


def paper_locked_matrix(df):
    """Return the exact 8-component analysis matrix used in the revised paper.

    Missing observations are first filled with the fiscal-year median computed
    from the full 2,940-row component panel. Remaining missing values are filled
    with the component-wide median, except the final-year predictive component,
    for which the historical paper-lock fallback above is retained to reproduce
    the exact final manuscript snapshot.
    """
    elig = eligible_mask(df)
    d = df.loc[elig].copy()
    X = d[COMPONENTS_8].copy()
    for c in COMPONENTS_8:
        year_med = df.groupby("fiscal_year")[c].transform("median")
        X[c] = X[c].fillna(year_med.loc[d.index])
        if c == "predictive_value_score":
            X[c] = X[c].fillna(PREDICTIVE_LAST_YEAR_FALLBACK)
        else:
            X[c] = X[c].fillna(df[c].median())
    if X.isna().any().any():
        raise ValueError("Paper analysis matrix still contains missing values.")
    return d, X


def primary_weights(A):
    w_e = entropy_weights(A)
    w_c = critic_weights(A)
    w_p = normalize_weights((w_e + w_c) / 2.0)
    return w_e, w_c, w_p


def spearman(x, y):
    r = spearmanr(x, y, nan_policy="omit")
    return float(r.statistic), float(r.pvalue)


def partial_spearman_formula(x, y, z):
    """Classic partial Spearman from the three pairwise Spearman coefficients."""
    tmp = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(tmp) < 5:
        return len(tmp), np.nan, np.nan
    rxy = spearmanr(tmp.x, tmp.y).statistic
    rxz = spearmanr(tmp.x, tmp.z).statistic
    ryz = spearmanr(tmp.y, tmp.z).statistic
    denom = math.sqrt(max((1-rxz**2)*(1-ryz**2), 1e-15))
    r = (rxy - rxz*ryz) / denom
    dfree = len(tmp)-3
    tv = r * math.sqrt(dfree / max(1-r*r, 1e-15))
    p = 2*t.sf(abs(tv), df=dfree)
    return len(tmp), float(r), float(p)
