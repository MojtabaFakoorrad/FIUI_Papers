"""Rebuild the nine empirical FIUI component scores from the processed firm-year panel.

This is the article-facing version of the original component-construction script.
It preserves the historical 1st/99th percentile year-wise winsorization and
min-max scaling used in the dissertation pipeline.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from common import ROOT, DATA, OUT, COMPONENTS_9

INPUT = DATA / "processed" / "financial_panel_all_1392_1401.csv"
OUTPUT = OUT / "01_rebuilt_components.csv"


def pdate(s):
    if pd.isna(s):
        return None
    t = str(s).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")).replace("-", "/").strip()
    parts = t.split("/")
    if len(parts) < 3:
        return None
    try:
        return tuple(map(int, parts[:3]))
    except Exception:
        return None


def jalali_to_jdn(y, m, d):
    epbase = y - (474 if y >= 0 else 473)
    epyear = 474 + (epbase % 2820)
    mdays = (m - 1) * 31 if m <= 7 else (m - 1) * 30 + 6
    return d + mdays + ((epyear * 682 - 110) // 2816) + (epyear - 1) * 365 + (epbase // 2820) * 1029983 + 1948319


def day_diff(a, b):
    if not a or not b:
        return np.nan
    return float(jalali_to_jdn(*b) - jalali_to_jdn(*a))


def winsor_bounds(values, p=0.01):
    values = sorted(float(x) for x in values if pd.notna(x))
    if not values:
        return 0.0, 0.0
    lo = values[max(0, min(len(values)-1, int((len(values)-1)*p)))]
    hi = values[max(0, min(len(values)-1, int((len(values)-1)*(1-p))))]
    return lo, hi


def scale(s, lo, hi, inverse=False):
    if hi <= lo:
        out = pd.Series(0.5, index=s.index)
        out[s.isna()] = np.nan
        return out
    out = (s.clip(lo, hi) - lo) / (hi-lo)
    if inverse:
        out = 1.0 - out
    return out


df = pd.read_csv(INPUT).sort_values(["symbol", "fiscal_year"]).copy()
# Accrual-based quantities
avg_assets = np.where(df["lag_total_assets"].notna() & (df["lag_total_assets"] != 0),
                      (df["total_assets"] + df["lag_total_assets"]) / 2.0,
                      df["total_assets"])
accrual = df["net_income"] - df["operating_cash_flow"]
df["_accrual_ratio"] = np.abs(accrual / avg_assets)
df["_signed_accrual_ratio"] = accrual / avg_assets
# Reporting lag
pairs = zip(df["date_title"], df["date_publish"])
df["_publication_lag"] = [day_diff(pdate(a), pdate(b)) for a, b in pairs]
# Predictive / confirmatory
next_growth = df.groupby("symbol")["net_income_growth"].shift(-1)
next_year = df.groupby("symbol")["fiscal_year"].shift(-1)
next_growth = next_growth.where(next_year.eq(df["fiscal_year"] + 1))
df["_next_growth_abs"] = next_growth.abs()
scale_confirm = df["net_income_growth"].abs() + df["annual_stock_return_raw"].abs() + 1e-9
df["_confirmation_raw"] = 1.0 - (df["net_income_growth"] - df["annual_stock_return_raw"]).abs() / scale_confirm
df.loc[df["net_income_growth"].isna() | df["annual_stock_return_raw"].isna(), "_confirmation_raw"] = np.nan
# Pipeline proxies
df["_completeness_raw"] = df["extracted_item_count"]
df["_understandability_raw"] = df["mean_match_score"]
df["_conservatism_raw"] = -df["_signed_accrual_ratio"]
# Within-firm rolling accounting-return mapping error
df["_mapping_error"] = np.nan
for symbol, g in df.groupby("symbol", sort=False):
    positions = list(g.index)
    for j, idx in enumerate(positions):
        hist = g.iloc[max(0, j-3):j+1][["annual_stock_return_raw", "roa_average_assets"]].dropna()
        if len(hist) < 3:
            continue
        x = hist["annual_stock_return_raw"].to_numpy(float)
        y = hist["roa_average_assets"].to_numpy(float)
        den = np.sum((x-x.mean())**2)
        if den <= 0:
            continue
        beta = np.sum((x-x.mean())*(y-y.mean()))/den
        alpha = y.mean()-beta*x.mean()
        df.loc[idx, "_mapping_error"] = np.mean(np.abs(y-(alpha+beta*x)))

specs = [
    ("_next_growth_abs", "predictive_value_score", True),
    ("_confirmation_raw", "confirmatory_value_score", False),
    ("_accrual_ratio", "faithful_representation_score", True),
    ("_accrual_ratio", "neutrality_score", True),
    ("_completeness_raw", "completeness_score", False),
    ("_understandability_raw", "understandability_score", False),
    ("_mapping_error", "comparability_score", True),
    ("_publication_lag", "timeliness_score", True),
    ("_conservatism_raw", "conservatism_score", False),
]
for year, idx in df.groupby("fiscal_year").groups.items():
    for raw, score, inv in specs:
        vals = df.loc[idx, raw].dropna().tolist()
        lo, hi = winsor_bounds(vals, 0.01)
        df.loc[idx, score] = scale(df.loc[idx, raw], lo, hi, inv)

keep = ["symbol", "fiscal_year", "company_name", "announcement_id"] + COMPONENTS_9
out = df[keep].copy()
out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

# Validation against the frozen component file used in the paper.
frozen = pd.read_csv(DATA / "processed" / "company_year_components_and_legacy_indices.csv")
check = out.merge(frozen[["symbol", "fiscal_year"] + COMPONENTS_9], on=["symbol", "fiscal_year"], suffixes=("_new", "_frozen"))
max_diff = 0.0
for c in COMPONENTS_9:
    a, b = check[f"{c}_new"], check[f"{c}_frozen"]
    diff = (a-b).abs().dropna()
    if len(diff):
        max_diff = max(max_diff, float(diff.max()))
print(f"Wrote {OUTPUT}")
print(f"Maximum absolute difference vs frozen component scores: {max_diff:.12g}")
