"""Nomological-network diagnostics and financial/nonfinancial robustness."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from common import DATA, OUT, load_components, paper_locked_matrix, primary_weights

comp=load_components(); d,X=paper_locked_matrix(comp)
A=X.to_numpy(float); _,_,w=primary_weights(A); d=d.copy(); d["FIUI_primary_8"]=100*(A@w)
panel=pd.read_csv(DATA/"processed"/"financial_panel_all_1392_1401.csv")
market_cols=["symbol","fiscal_year","amihud_illiquidity_raw","annualized_volatility_raw","mean_daily_trade_value","market_cap_year_end_raw"]
m=d[["symbol","fiscal_year","FIUI_primary_8"]].merge(panel[market_cols],on=["symbol","fiscal_year"],how="left")
# next-year market outcomes
nxt=panel[market_cols].copy(); nxt["fiscal_year"]-=1
nxt=nxt.rename(columns={c:"next_"+c for c in market_cols if c not in ["symbol","fiscal_year"]})
m=m.merge(nxt,on=["symbol","fiscal_year"],how="left")
criteria=[
    ("Current-year Amihud illiquidity","negative","amihud_illiquidity_raw"),
    ("Current-year return volatility","negative","annualized_volatility_raw"),
    ("Current-year mean daily trade value","positive","mean_daily_trade_value"),
    ("Next-year Amihud illiquidity","negative","next_amihud_illiquidity_raw"),
    ("Next-year return volatility","negative","next_annualized_volatility_raw"),
    ("Next-year mean daily trade value","positive","next_mean_daily_trade_value"),
]
rows=[]
for label,expected,c in criteria:
    tmp=m[["FIUI_primary_8",c]].dropna()
    r=spearmanr(tmp.iloc[:,0],tmp.iloc[:,1])
    rows.append([label,expected,len(tmp),float(r.statistic),float(r.pvalue)])
res=pd.DataFrame(rows,columns=["criterion","expected_sign","n","spearman_rho","p_value"])
res.to_csv(OUT/"04_criterion_validity.csv",index=False)
print(res.to_string(index=False))

# Financial/nonfinancial level test.
sector=pd.read_csv(DATA/"metadata"/"banks_insurance_symbols.csv")
financial=set(sector.symbol.astype(str))
d["group"]=np.where(d.symbol.astype(str).isin(financial),"Financial","Nonfinancial")
f=d.loc[d.group=="Financial","FIUI_primary_8"] if "FIUI_primary_8" in d else pd.Series(dtype=float)
# score lives in m, join group for test
m["group"]=np.where(m.symbol.astype(str).isin(financial),"Financial","Nonfinancial")
stat=mannwhitneyu(m.loc[m.group=="Financial","FIUI_primary_8"],m.loc[m.group=="Nonfinancial","FIUI_primary_8"],alternative="two-sided")
pd.DataFrame([{"mann_whitney_U":stat.statistic,"p_value":stat.pvalue}]).to_csv(OUT/"04_financial_nonfinancial_level_test.csv",index=False)
