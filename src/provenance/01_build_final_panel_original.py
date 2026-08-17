from pathlib import Path
import pandas as pd, numpy as np

root=Path('/mnt/data/work_thesis2')
fin_path=root/'fast_output_v2/metadata/financial_items_long.csv'
manifest_path=root/'manifest/codal_file_manifest.csv'
market_path=root/'market_output/market_annual_1392_1401.csv'
out=root/'final_output'; out.mkdir(exist_ok=True)

fin=pd.read_csv(fin_path,encoding='utf-8-sig')
man=pd.read_csv(manifest_path,encoding='utf-8-sig',dtype={'announcement_id':str,'ins_code':str})
market=pd.read_csv(market_path,encoding='utf-8-sig',dtype={'ins_code':str})

# Canonical selected report per symbol-year (latest correction/publication candidate).
selected=man[man['selected_latest_candidate'].astype(str).str.lower().isin(['true','1'])].copy()
meta_cols=['flattened_name','announcement_id','symbol','fiscal_year','requested_company_name','ins_code','isin','date_title','date_send','date_publish','time_publish','title','is_correction','original_filename','link','link_excel','link_pdf']
selected=selected[meta_cols]
merged=fin.merge(selected,left_on='source_name',right_on='flattened_name',how='inner',suffixes=('_extract',''))
# Trust the archive/BrsAPI symbol-year, not free-text inference.
merged['symbol']=merged['symbol'].fillna(merged['symbol_extract'])
merged['fiscal_year']=merged['fiscal_year'].fillna(merged['fiscal_year_extract'])
merged=merged[(merged.fiscal_year>=1392)&(merged.fiscal_year<=1401)].copy()

# Keep best match for each item within selected report.
merged=merged.sort_values(['symbol','fiscal_year','item_code','match_score'],ascending=[True,True,True,False])
merged=merged.drop_duplicates(['symbol','fiscal_year','item_code'],keep='first')
merged['value_million_rial']=pd.to_numeric(merged['current_value_rial'],errors='coerce')/1_000_000

id_cols=['symbol','fiscal_year']
wide=merged.pivot(index=id_cols,columns='item_code',values='value_million_rial').reset_index()
wide.columns.name=None

# Report metadata and extraction quality.
q=merged.groupby(id_cols).agg(
    extracted_item_count=('item_code','nunique'),
    mean_match_score=('match_score','mean'),
    minimum_match_score=('match_score','min'),
    report_unit=('unit_name','first'),
    audited=('audited','max'),
    consolidated=('consolidated','max'),
).reset_index()
report=selected.rename(columns={'requested_company_name':'company_name'})
report=report[['symbol','fiscal_year','company_name','announcement_id','ins_code','isin','date_title','date_send','date_publish','time_publish','title','is_correction','original_filename','link','link_excel','link_pdf']]
panel=report.merge(wide,on=id_cols,how='left').merge(q,on=id_cols,how='left')

# Remove exact duplicate report rows if any remain after manifest selection.
panel=panel.sort_values(['symbol','fiscal_year','is_correction','date_publish','time_publish']).drop_duplicates(id_cols,keep='last')

# Coerce financial variables.
financial_cols=[c for c in wide.columns if c not in id_cols]
for c in financial_cols:
    panel[c]=pd.to_numeric(panel[c],errors='coerce')

# Lags and average denominators.
panel=panel.sort_values(['symbol','fiscal_year'])
for c in ['total_assets','total_equity','revenue','net_income']:
    if c in panel:
        panel[f'lag_{c}']=panel.groupby('symbol')[c].shift(1)
        panel[f'{c}_growth']=panel.groupby('symbol')[c].pct_change(fill_method=None)

def div(a,b):
    return a / b.replace(0,np.nan)

if {'net_income','total_assets'}.issubset(panel):
    avg=(panel.total_assets+panel.lag_total_assets)/2
    panel['roa_average_assets']=div(panel.net_income,avg)
    panel['roa_ending_assets']=div(panel.net_income,panel.total_assets)
if {'net_income','total_equity'}.issubset(panel):
    avg=(panel.total_equity+panel.lag_total_equity)/2
    panel['roe_average_equity']=div(panel.net_income,avg)
    panel['roe_ending_equity']=div(panel.net_income,panel.total_equity)
if {'current_assets','current_liabilities'}.issubset(panel):
    panel['current_ratio']=div(panel.current_assets,panel.current_liabilities)
if {'current_assets','inventory','current_liabilities'}.issubset(panel):
    panel['quick_ratio']=div(panel.current_assets-panel.inventory,panel.current_liabilities)
if {'total_liabilities','total_assets'}.issubset(panel):
    panel['debt_ratio']=div(panel.total_liabilities,panel.total_assets)
if {'total_liabilities','total_equity'}.issubset(panel):
    panel['debt_to_equity']=div(panel.total_liabilities,panel.total_equity)
if {'gross_profit','revenue'}.issubset(panel): panel['gross_margin']=div(panel.gross_profit,panel.revenue)
if {'operating_profit','revenue'}.issubset(panel): panel['operating_margin']=div(panel.operating_profit,panel.revenue)
if {'net_income','revenue'}.issubset(panel): panel['net_margin']=div(panel.net_income,panel.revenue)
if {'revenue','total_assets'}.issubset(panel):
    avg=(panel.total_assets+panel.lag_total_assets)/2
    panel['asset_turnover']=div(panel.revenue,avg)
if {'operating_cash_flow','total_assets'}.issubset(panel): panel['operating_cashflow_to_assets']=div(panel.operating_cash_flow,panel.total_assets)
if {'operating_cash_flow','current_liabilities'}.issubset(panel): panel['operating_cashflow_to_current_liabilities']=div(panel.operating_cash_flow,panel.current_liabilities)

# Merge annual market metrics.
market['fiscal_year']=pd.to_numeric(market['fiscal_year'],errors='coerce').astype('Int64')
panel=panel.merge(market.drop(columns=['company_name','isin','ins_code'],errors='ignore'),on=id_cols,how='left')
panel=panel.sort_values(['symbol','fiscal_year'])
panel['next_year_stock_return_raw']=panel.groupby('symbol')['annual_stock_return_raw'].shift(-1)
panel['market_available']=panel['trading_days'].notna()
core=['revenue','net_income','total_assets','total_equity','total_liabilities']
existing=[c for c in core if c in panel]
panel['complete_core_financials']=panel[existing].notna().all(axis=1) if existing else False
panel['modeling_row_contemporaneous']=panel['complete_core_financials'] & panel['annual_stock_return_raw'].notna()
panel['modeling_row_predictive']=panel['complete_core_financials'] & panel['next_year_stock_return_raw'].notna()

# Clean infs and implausible ratio artifacts; retain flags rather than silently deleting amounts.
panel=panel.replace([np.inf,-np.inf],np.nan)
ratio_cols=[c for c in panel.columns if c in ['roa_average_assets','roa_ending_assets','roe_average_equity','roe_ending_equity','current_ratio','quick_ratio','debt_ratio','debt_to_equity','gross_margin','operating_margin','net_margin','asset_turnover','operating_cashflow_to_assets','operating_cashflow_to_current_liabilities'] or c.endswith('_growth')]
panel['ratio_outlier_flag']=False
for c in ratio_cols:
    panel['ratio_outlier_flag'] |= panel[c].abs().gt(100).fillna(False)

panel.to_csv(out/'financial_panel_all_1392_1401.csv',index=False,encoding='utf-8-sig')
model_cont=panel[panel.modeling_row_contemporaneous].copy()
model_pred=panel[panel.modeling_row_predictive].copy()
model_cont.to_csv(out/'modeling_dataset_contemporaneous.csv',index=False,encoding='utf-8-sig')
model_pred.to_csv(out/'modeling_dataset_predictive.csv',index=False,encoding='utf-8-sig')
merged.to_csv(out/'financial_items_selected_long.csv',index=False,encoding='utf-8-sig')

coverage=panel.groupby('fiscal_year').agg(
    financial_reports=('symbol','size'),
    companies=('symbol','nunique'),
    core_complete=('complete_core_financials','sum'),
    market_matched=('market_available','sum'),
    contemporaneous_model_rows=('modeling_row_contemporaneous','sum'),
    predictive_model_rows=('modeling_row_predictive','sum'),
).reset_index()
coverage.to_csv(out/'coverage_by_year.csv',index=False,encoding='utf-8-sig')

summary=pd.DataFrame([
 {'metric':'Codal files supplied','value':len(man)},
 {'metric':'Canonical company-year reports','value':len(panel)},
 {'metric':'Unique Codal symbols','value':panel.symbol.nunique()},
 {'metric':'Selected financial line items','value':len(merged)},
 {'metric':'Core-complete company-years','value':int(panel.complete_core_financials.sum())},
 {'metric':'Market-matched company-years','value':int(panel.market_available.sum())},
 {'metric':'Contemporaneous modeling rows','value':len(model_cont)},
 {'metric':'Predictive modeling rows','value':len(model_pred)},
])
summary.to_csv(out/'pipeline_summary.csv',index=False,encoding='utf-8-sig')
print(summary.to_string(index=False))
print('\nCoverage:\n',coverage.to_string(index=False))
