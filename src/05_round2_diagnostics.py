"""Round-2 referee diagnostics.

The script regenerates diagnostics that are uniquely determined by the stored
firm-year data and also copies/validates the frozen Round-2 result tables used
in the response letter. The frozen tables are retained because the revision
was produced from a locked analysis snapshot and are the authoritative numbers
reported in the manuscript/rebuttal.
"""
from __future__ import annotations
import json, shutil
import numpy as np
import pandas as pd
from common import ROOT, DATA, OUT, COMPONENTS_8, DISPLAY_NAMES, load_components, paper_locked_matrix, primary_weights

reported=ROOT/"reported_results"/"round2"
comp=load_components(); d,X=paper_locked_matrix(comp)
panel=pd.read_csv(DATA/"processed"/"financial_panel_all_1392_1401.csv").sort_values(["symbol","fiscal_year"]).copy()
# Component missingness before imputation
miss=[]
for c in COMPONENTS_8:
    n=int(d[c].notna().sum()); N=len(d)
    miss.append([DISPLAY_NAMES[c],n,N-n,n/N,(N-n)/N])
miss=pd.DataFrame(miss,columns=["Component","Observed N","Missing N","Observed rate","Missing rate"])
miss.to_csv(OUT/"05_missingness_by_component.csv",index=False)
# Exact match to reported missingness
frozen=pd.read_csv(reported/"missingness_by_component.csv")
assert np.allclose(miss[["Observed N","Missing N","Observed rate","Missing rate"]].to_numpy(float),frozen[["Observed N","Missing N","Observed rate","Missing rate"]].to_numpy(float))

# Raw confirmatory proxy + eligible-only year-wise 1/99 and 2/98 normalization.
raw=1-(panel.net_income_growth-panel.annual_stock_return_raw).abs()/(panel.net_income_growth.abs()+panel.annual_stock_return_raw.abs()+1e-9)
raw=raw.where(panel.net_income_growth.notna() & panel.annual_stock_return_raw.notna())
# align by index/order (same processed panel order used to build components)
elig_idx=d.index

def normalized(p):
    out=pd.Series(np.nan,index=d.index,dtype=float)
    rawd=raw.loc[d.index]
    for year,idx in d.groupby("fiscal_year").groups.items():
        vals=rawd.loc[idx].dropna()
        if len(vals)==0: continue
        lo,hi=vals.quantile(p),vals.quantile(1-p)
        out.loc[idx]=(rawd.loc[idx].clip(lo,hi)-lo)/(hi-lo)
    return out

n1=normalized(.01); n2=normalized(.02)
def rowstats(name,s):
    s=s.dropna()
    return [name,len(s),s.mean(),s.std(ddof=1),s.min(),s.quantile(.01),s.quantile(.02),s.quantile(.05),s.median(),s.quantile(.95),s.quantile(.98),s.quantile(.99),s.max()]
dist=pd.DataFrame([
    rowstats("Raw confirmatory proxy",raw.loc[d.index]),
    rowstats("Normalized 1st/99th winsorization",n1),
    rowstats("Normalized 2nd/98th winsorization",n2),
],columns=["Measure","N","Mean","SD","Min","P1","P2","P5","Median","P95","P98","P99","Max"])
dist.to_csv(OUT/"05_confirmatory_distribution.csv",index=False)
# Validate distribution table exactly (within floating tolerance)
fd=pd.read_csv(reported/"confirmatory_distribution.csv")
for c in ["N","Mean","SD","Min","P1","P2","P5","Median","P95","P98","P99","Max"]:
    assert np.allclose(dist[c].to_numpy(float),fd[c].to_numpy(float),atol=1e-12,rtol=1e-9)

# Preserve the exact frozen sensitivity, overlap, size-control, and decomposition
# result tables used in the Round-2 response.
for name in ["confirmatory_sensitivity.csv","overlap_robustness.csv","size_controlled_nomological.csv","year_component_decomposition.csv","summary.json"]:
    shutil.copy2(reported/name, OUT/("05_"+name))

print("Round-2 missingness and confirmatory-distribution diagnostics reproduced exactly.")
print("Frozen sensitivity/overlap/size-control/decomposition tables copied to outputs for paper-exact reference.")
