from pathlib import Path
import pandas as pd, numpy as np, json, math
from datetime import datetime
import jdatetime

root=Path('/mnt/data/work_thesis2/tsetmc/research_data_tsetmc_only_v2_6')
price_dir=root/'exports'/'market_second_prices'
raw_dir=root/'raw'/'tsetmc'
out=Path('/mnt/data/work_thesis2/market_output'); out.mkdir(exist_ok=True)
rows=[]; errors=[]

def greg_to_jy(v):
    try:
        s=str(int(v)); d=datetime.strptime(s,'%Y%m%d').date(); return jdatetime.date.fromgregorian(date=d).year
    except Exception: return None

def shares_at(symbol, d_int):
    p=raw_dir/symbol/'share_changes.json'
    if not p.exists(): return None
    try:
        arr=json.load(open(p,encoding='utf-8')).get('instrumentShareChange',[])
        arr=sorted(arr,key=lambda x:int(x.get('dEven') or 0))
        # Start with oldest known old shares, apply changes up to target date
        if not arr: return None
        shares=float(arr[0].get('numberOfShareOld') or arr[0].get('numberOfShareNew') or 0)
        for x in arr:
            if int(x.get('dEven') or 0)<=d_int:
                shares=float(x.get('numberOfShareNew') or shares)
        return shares if shares>0 else None
    except Exception: return None

files=sorted(price_dir.glob('*.csv'))
for i,p in enumerate(files,1):
    symbol=p.stem
    try:
        df=pd.read_csv(p,encoding='utf-8-sig')
        if df.empty: continue
        df['dEven']=pd.to_numeric(df['dEven'],errors='coerce').astype('Int64')
        df=df.dropna(subset=['dEven','pClosing']).copy()
        df['jyear']=df['dEven'].map(greg_to_jy)
        df=df[df['jyear'].between(1392,1401)].copy()
        if df.empty: continue
        df=df.sort_values('dEven')
        df['daily_return']=pd.to_numeric(df['pClosing'],errors='coerce').pct_change()
        df['trade_value']=pd.to_numeric(df['qTotCap'],errors='coerce')
        df['volume']=pd.to_numeric(df['qTotTran5J'],errors='coerce')
        df['trades']=pd.to_numeric(df['zTotTran'],errors='coerce')
        for y,g in df.groupby('jyear'):
            g=g.sort_values('dEven')
            first=g.iloc[0]; last=g.iloc[-1]
            first_close=float(first['pClosing']); last_close=float(last['pClosing'])
            ret=last_close/first_close-1 if first_close else np.nan
            dr=g['daily_return'].replace([np.inf,-np.inf],np.nan).dropna()
            annual_vol=float(dr.std(ddof=1)*math.sqrt(252)) if len(dr)>1 else np.nan
            end_date=int(last['dEven'])
            shares=shares_at(symbol,end_date)
            market_cap=last_close*shares if shares else np.nan
            value=g['trade_value'].sum(min_count=1)
            amihud=(dr.abs()/g.loc[dr.index,'trade_value'].replace(0,np.nan)).mean() if len(dr) else np.nan
            rows.append({
                'symbol':symbol,
                'company_name':str(last.get('company_name','')),
                'isin':str(last.get('isin','')),
                'ins_code':str(last.get('ins_code','')),
                'fiscal_year':int(y),
                'trading_days':int(len(g)),
                'first_trade_date_gregorian':int(first['dEven']),
                'last_trade_date_gregorian':end_date,
                'first_closing_price':first_close,
                'last_closing_price':last_close,
                'annual_stock_return_raw':ret,
                'mean_closing_price':float(pd.to_numeric(g['pClosing'],errors='coerce').mean()),
                'annualized_volatility_raw':annual_vol,
                'total_trade_value':float(value) if pd.notna(value) else np.nan,
                'total_volume':float(g['volume'].sum(min_count=1)),
                'total_trades':float(g['trades'].sum(min_count=1)),
                'mean_daily_trade_value':float(g['trade_value'].mean()),
                'amihud_illiquidity_raw':float(amihud) if pd.notna(amihud) else np.nan,
                'shares_outstanding_year_end':shares,
                'market_cap_year_end_raw':market_cap,
                'return_adjustment_status':'provisional_unadjusted'
            })
    except Exception as e:
        errors.append({'symbol':symbol,'error':f'{type(e).__name__}: {e}'})
    if i%25==0: print(i,'/',len(files),flush=True)

annual=pd.DataFrame(rows).sort_values(['symbol','fiscal_year'])
annual.to_csv(out/'market_annual_1392_1401.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(errors).to_csv(out/'market_errors.csv',index=False,encoding='utf-8-sig')
coverage=annual.groupby('fiscal_year').agg(companies=('symbol','nunique'),observations=('symbol','size'),median_trading_days=('trading_days','median')).reset_index()
coverage.to_csv(out/'market_coverage.csv',index=False,encoding='utf-8-sig')
print('annual rows',len(annual),'symbols',annual.symbol.nunique(),'errors',len(errors))
print(coverage.to_string(index=False))
