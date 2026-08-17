"""Reproduce the primary eight-component FIUI and Round-1 robustness tables."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (
    ROOT, DATA, OUT, COMPONENTS_8, DISPLAY_NAMES, load_components,
    paper_locked_matrix, entropy_weights, critic_weights, pca_weights,
    primary_weights, rank_desc,
)

reported = ROOT / "reported_results" / "round1"
df = load_components()
d, X = paper_locked_matrix(df)
A = X.to_numpy(float)
wE, wC, wP = primary_weights(A)
wEq = np.ones(len(COMPONENTS_8))/len(COMPONENTS_8)
wPCA, pca_k, pca_cum = pca_weights(A)

score_primary = 100*(A@wP)
score_equal = 100*(A@wEq)
score_pca = 100*(A@wPCA)
score_original9 = d["index_FinalRobust"].to_numpy(float)*100

# Save the exact paper analysis matrix for easy independent replication.
analysis = d[["symbol", "fiscal_year", "company_name"]].copy()
for c in COMPONENTS_8:
    analysis[c] = X[c].to_numpy()
analysis["FIUI_primary_8"] = score_primary
analysis["FIUI_equal_8"] = score_equal
analysis["FIUI_PCA_8"] = score_pca
analysis["FIUI_original_9"] = score_original9
analysis.to_csv(OUT / "02_paper_analysis_matrix_8.csv", index=False, encoding="utf-8-sig")

# Weights
rows=[]
for i,c in enumerate(COMPONENTS_8):
    rows.append({
        "component": DISPLAY_NAMES[c],
        "Entropy": wE[i], "CRITIC": wC[i], "Primary": wP[i],
        "Equal": wEq[i], "PCA": wPCA[i], "rank": rank_desc(wP)[i],
    })
wdf=pd.DataFrame(rows).sort_values("rank")
wdf.to_csv(OUT/"02_revised_weights_8.csv", index=False)

# Index descriptives and year statistics
s=pd.Series(score_primary)
desc=pd.DataFrame([{
    "n":len(s),"mean":s.mean(),"sd":s.std(ddof=1),"median":s.median(),
    "p5":s.quantile(.05),"p25":s.quantile(.25),"p75":s.quantile(.75),"p95":s.quantile(.95),
    "min":s.min(),"max":s.max(),"skewness":s.skew(),
}])
desc.to_csv(OUT/"02_index_descriptive_8.csv",index=False)
year=d.assign(FIUI=score_primary).groupby("fiscal_year").FIUI.agg(["count","mean","std","median"]).reset_index()
year.to_csv(OUT/"02_year_stats_8.csv",index=False)

# Robustness comparisons
rob=[]
def add(label, x, y):
    m=np.isfinite(x)&np.isfinite(y)
    rob.append([label,int(m.sum()),float(spearmanr(np.asarray(x)[m],np.asarray(y)[m]).statistic)])
add("Primary 8-component vs original 9-component",score_primary,score_original9)
add("Primary 8-component vs equal-weighted 8-component",score_primary,score_equal)
add("Primary 8-component vs PCA-weighted 8-component",score_primary,score_pca)
# Complete cases, re-estimated weights
complete=d[COMPONENTS_8].notna().all(axis=1).to_numpy()
Ac=d.loc[d[COMPONENTS_8].notna().all(axis=1), COMPONENTS_8].to_numpy(float)
wc=(entropy_weights(Ac)+critic_weights(Ac))/2
add("Complete-case recalibrated vs primary weights (complete cases only)",score_primary[complete],100*(Ac@wc))
# Financial/nonfinancial robustness
sectors=pd.read_csv(DATA/"metadata"/"banks_insurance_symbols.csv")
financial=set(sectors["symbol"].astype(str))
non=~d["symbol"].astype(str).isin(financial).to_numpy()
An=A[non]
wn=(entropy_weights(An)+critic_weights(An))/2
add("Nonfinancial recalibrated vs full-sample weights (nonfinancial rows)",score_primary[non],100*(An@wn))
robdf=pd.DataFrame(rob,columns=["comparison","n","spearman_rho"])
robdf.to_csv(OUT/"02_robustness_summary.csv",index=False)

# Sector distributions
sector_group=np.where(d["symbol"].astype(str).isin(financial),"Financial (banks and insurance)","Nonfinancial")
sector_df=pd.DataFrame({"group":sector_group,"score":score_primary}).groupby("group").score.agg(["count","mean","std","median"]).reset_index().rename(columns={"count":"n"})
sector_df.to_csv(OUT/"02_sector_distribution.csv",index=False)

# Component Spearman matrix on directly observed scores before imputation.
original9=d[[
    "predictive_value_score","confirmatory_value_score","faithful_representation_score","neutrality_score",
    "completeness_score","understandability_score","comparability_score","timeliness_score","conservatism_score"
]]
original9.corr(method="spearman").round(4).to_csv(OUT/"02_component_spearman_9.csv")

meta={
    "eligible8":len(d),"complete8":int(complete.sum()),"pca_components":pca_k,"pca_cumvar":pca_cum,
    "financial_firms":len(financial),"nonfinancial_firms":d.loc[non,"symbol"].nunique(),
    "weight_rank_rho_full_nonfin":float(spearmanr(wP,wn).statistic),
    "score_rho_nonfin":robdf.loc[robdf.comparison.str.startswith("Nonfinancial"),"spearman_rho"].iloc[0],
    "complete_score_rho":robdf.loc[robdf.comparison.str.startswith("Complete-case"),"spearman_rho"].iloc[0],
}
(OUT/"02_stats.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

# Exact reported-result checks
checks={
    "weights": (wdf[["component","Entropy","CRITIC","Primary","Equal","PCA"]].reset_index(drop=True), pd.read_csv(reported/"revised_weights_8.csv")[["component","Entropy","CRITIC","Primary","Equal","PCA"]].reset_index(drop=True)),
    "year": (year, pd.read_csv(reported/"year_stats_8.csv")),
    "robustness": (robdf, pd.read_csv(reported/"robustness_summary.csv")),
}
for name,(a,b) in checks.items():
    # numeric tolerant comparison, text exact where practical
    for c in set(a.columns).intersection(b.columns):
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            md=float(np.nanmax(np.abs(a[c].to_numpy(float)-b[c].to_numpy(float))))
            if md>1e-9:
                raise AssertionError(f"{name}:{c} differs from reported snapshot by {md}")
print("Primary FIUI reproduction matches the reported Round-1 revision snapshot.")
print(wdf.to_string(index=False))
