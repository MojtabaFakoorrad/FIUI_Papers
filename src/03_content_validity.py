"""Reproduce the expert CVR/CVI table and 36-item Cronbach alpha."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import DATA, OUT

# CVR/CVI data: one row per expert x theoretical component.
df=pd.read_csv(DATA/"survey"/"expert_cvr_cvi_deidentified.csv")
N=df["کد خبره"].nunique()
rows=[]
for component,g in df.groupby("مؤلفه", sort=False):
    ne=(g["ضرورت"].astype(str).str.strip()=="ضروری").sum()
    cvr=(ne-N/2)/(N/2)
    rows.append({
        "component_fa":component,
        "essential_votes":int(ne),
        "CVR":float(cvr),
        "relevance_I_CVI":float((pd.to_numeric(g["ارتباط"],errors="coerce")>=3).mean()),
        "clarity_I_CVI":float((pd.to_numeric(g["وضوح"],errors="coerce")>=3).mean()),
        "simplicity_I_CVI":float((pd.to_numeric(g["سادگی"],errors="coerce")>=3).mean()),
    })
res=pd.DataFrame(rows)
res.to_csv(OUT/"03_content_validity_by_component.csv",index=False,encoding="utf-8-sig")
summary={
    "N_experts":N,
    "S_CVI_Ave_relevance":res.relevance_I_CVI.mean(),
    "S_CVI_Ave_clarity":res.clarity_I_CVI.mean(),
    "S_CVI_Ave_simplicity":res.simplicity_I_CVI.mean(),
    "S_CVI_Ave_overall":res[["relevance_I_CVI","clarity_I_CVI","simplicity_I_CVI"]].to_numpy().mean(),
}
# Cronbach alpha, 36 primary Likert items; first four columns are demographics.
q=pd.read_csv(DATA/"survey"/"expert_questionnaire_36item_deidentified.csv")
items=q.iloc[:,4:].apply(pd.to_numeric,errors="coerce")
k=items.shape[1]
alpha=k/(k-1)*(1-items.var(axis=0,ddof=1).sum()/items.sum(axis=1).var(ddof=1))
summary["Cronbach_alpha_36_items"]=float(alpha)
pd.DataFrame([summary]).to_csv(OUT/"03_expert_validation_summary.csv",index=False)
print(pd.DataFrame([summary]).to_string(index=False))
print(res.to_string(index=False))
