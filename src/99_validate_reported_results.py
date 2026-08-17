"""Validate key numerical claims in the revised manuscript against frozen reported tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import ROOT, OUT

r1=ROOT/"reported_results"/"round1"
r2=ROOT/"reported_results"/"round2"
checks=[]
# Round 1 exact outputs expected from 02_primary_fiui.py
pairs=[
    (OUT/"02_revised_weights_8.csv", r1/"revised_weights_8.csv", ["Entropy","CRITIC","Primary","Equal","PCA"]),
    (OUT/"02_year_stats_8.csv", r1/"year_stats_8.csv", ["count","mean","std","median"]),
    (OUT/"02_robustness_summary.csv", r1/"robustness_summary.csv", ["n","spearman_rho"]),
    (OUT/"04_criterion_validity.csv", r1/"criterion_validity.csv", ["n","spearman_rho","p_value"]),
    (OUT/"05_missingness_by_component.csv", r2/"missingness_by_component.csv", ["Observed N","Missing N","Observed rate","Missing rate"]),
    (OUT/"05_confirmatory_distribution.csv", r2/"confirmatory_distribution.csv", ["N","Mean","SD","Min","P1","P2","P5","Median","P95","P98","P99","Max"]),
]
for got,exp,cols in pairs:
    a=pd.read_csv(got); b=pd.read_csv(exp)
    maxdiff=0.0
    for c in cols:
        maxdiff=max(maxdiff,float(np.nanmax(np.abs(a[c].to_numpy(float)-b[c].to_numpy(float)))))
    checks.append({"file":got.name,"max_abs_numeric_difference":maxdiff,"pass":maxdiff<1e-9})
res=pd.DataFrame(checks)
res.to_csv(OUT/"99_validation_report.csv",index=False)
print(res.to_string(index=False))
if not res["pass"].all():
    raise SystemExit("At least one reported-result validation failed.")
