from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'final_output' / 'financial_panel_all_1392_1401.csv'
OUTDIR = ROOT / 'phase2_usefulness_index'
OUTDIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    'predictive_value_score': 1.0,
    'confirmatory_value_score': 1.0,
    'faithful_representation_score': 1.0,
    'neutrality_score': 1.0,
    'completeness_score': 1.0,
    'understandability_score': 1.0,
    'comparability_score': 1.0,
    'timeliness_score': 1.0,
    'conservatism_score': 1.0,
}


def fnum(v: Any) -> float | None:
    try:
        if v is None or str(v).strip() == '':
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def pdate(s: str | None) -> tuple[int,int,int] | None:
    if not s: return None
    t = str(s).translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹','0123456789')).replace('-','/').strip()
    parts = t.split('/')
    if len(parts) < 3: return None
    try: return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception: return None


def jalali_to_jdn(y:int,m:int,d:int)->int:
    epbase = y - (474 if y >= 0 else 473)
    epyear = 474 + (epbase % 2820)
    mdays = (m - 1) * 31 if m <= 7 else (m - 1) * 30 + 6
    return d + mdays + ((epyear * 682 - 110) // 2816) + (epyear - 1) * 365 + (epbase // 2820) * 1029983 + (1948320 - 1)


def day_diff(a: tuple[int,int,int] | None, b: tuple[int,int,int] | None) -> float | None:
    if not a or not b: return None
    try: return float(jalali_to_jdn(*b) - jalali_to_jdn(*a))
    except Exception: return None


def winsor_bounds(vals:list[float], p=0.01)->tuple[float,float]:
    vals=sorted(vals)
    if not vals: return (0.0,0.0)
    lo=vals[max(0,min(len(vals)-1,int((len(vals)-1)*p)))]
    hi=vals[max(0,min(len(vals)-1,int((len(vals)-1)*(1-p))))]
    return lo,hi


def minmax(x:float|None, lo:float, hi:float, inverse=False)->float|None:
    if x is None: return None
    if hi <= lo: return 0.5
    z=(min(max(x,lo),hi)-lo)/(hi-lo)
    return 1-z if inverse else z

with INPUT.open(encoding='utf-8-sig', newline='') as f:
    rows=list(csv.DictReader(f))

# Sort by company-year and create lags/leads.
rows.sort(key=lambda r:(r['symbol'], int(r['fiscal_year'])))
by_symbol=defaultdict(list)
for r in rows: by_symbol[r['symbol']].append(r)

for sym, rs in by_symbol.items():
    for i,r in enumerate(rs):
        ni=fnum(r.get('net_income')); ocf=fnum(r.get('operating_cash_flow')); assets=fnum(r.get('total_assets'))
        lag_assets=fnum(r.get('lag_total_assets'))
        avg_assets=(assets+lag_assets)/2 if assets is not None and lag_assets not in (None,0) else assets
        accruals=(ni-ocf) if ni is not None and ocf is not None else None
        r['_accrual_ratio']=abs(accruals/avg_assets) if accruals is not None and avg_assets not in (None,0) else None
        r['_signed_accrual_ratio']=(accruals/avg_assets) if accruals is not None and avg_assets not in (None,0) else None
        r['_publication_lag']=day_diff(pdate(r.get('date_title')), pdate(r.get('date_publish')))
        r['_earnings_change']=fnum(r.get('net_income_growth'))
        r['_return']=fnum(r.get('annual_stock_return_raw'))
        r['_match_quality']=fnum(r.get('mean_match_score'))
        r['_item_count']=fnum(r.get('extracted_item_count'))
        r['_audited']=1.0 if str(r.get('audited')).lower()=='true' else 0.0
        # Predictability: inverse absolute next-year earnings growth, available only when next observation exists.
        next_growth=fnum(rs[i+1].get('net_income_growth')) if i+1<len(rs) and int(rs[i+1]['fiscal_year'])==int(r['fiscal_year'])+1 else None
        r['_next_growth_abs']=abs(next_growth) if next_growth is not None else None
        # Confirmatory value: agreement between earnings change and stock return.
        eg=r['_earnings_change']; ret=r['_return']
        if eg is None or ret is None:
            r['_confirmation_raw']=None
        else:
            scale=abs(eg)+abs(ret)+1e-9
            r['_confirmation_raw']=1.0-abs(eg-ret)/scale
        # Comparability proxy: stability of accounting mapping over rolling observations.
        hist=rs[max(0,i-3):i+1]
        pairs=[(fnum(x.get('annual_stock_return_raw')), fnum(x.get('roa_average_assets'))) for x in hist]
        pairs=[p for p in pairs if p[0] is not None and p[1] is not None]
        if len(pairs)>=3:
            xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
            mx=statistics.mean(xs); my=statistics.mean(ys)
            den=sum((x-mx)**2 for x in xs)
            if den>0:
                beta=sum((x-mx)*(y-my) for x,y in pairs)/den
                alpha=my-beta*mx
                resid=[y-(alpha+beta*x) for x,y in pairs]
                r['_mapping_error']=statistics.mean(abs(e) for e in resid)
            else: r['_mapping_error']=None
        else: r['_mapping_error']=None
        # Understandability proxy: machine readability / mapping confidence.
        r['_understandability_raw']=r['_match_quality']
        # Completeness proxy: number of standardized items successfully extracted.
        r['_completeness_raw']=r['_item_count']
        # Conservatism proxy: more negative accruals score higher; transformed later.
        r['_conservatism_raw']=(-r['_signed_accrual_ratio']) if r['_signed_accrual_ratio'] is not None else None

# Year-wise normalization avoids inflation/format drift across years.
raw_to_score = {
    '_next_growth_abs': ('predictive_value_score', True),
    '_confirmation_raw': ('confirmatory_value_score', False),
    '_accrual_ratio': ('faithful_representation_score', True),
    '_accrual_ratio': ('neutrality_score', True),
    '_completeness_raw': ('completeness_score', False),
    '_understandability_raw': ('understandability_score', False),
    '_mapping_error': ('comparability_score', True),
    '_publication_lag': ('timeliness_score', True),
    '_conservatism_raw': ('conservatism_score', False),
}
# faithful representation and neutrality deliberately use same quantitative proxy at baseline.

by_year=defaultdict(list)
for r in rows: by_year[r['fiscal_year']].append(r)

for year, rs in by_year.items():
    specs=[
        ('_next_growth_abs','predictive_value_score',True),
        ('_confirmation_raw','confirmatory_value_score',False),
        ('_accrual_ratio','faithful_representation_score',True),
        ('_accrual_ratio','neutrality_score',True),
        ('_completeness_raw','completeness_score',False),
        ('_understandability_raw','understandability_score',False),
        ('_mapping_error','comparability_score',True),
        ('_publication_lag','timeliness_score',True),
        ('_conservatism_raw','conservatism_score',False),
    ]
    for raw, score, inv in specs:
        vals=[r[raw] for r in rs if r.get(raw) is not None]
        lo,hi=winsor_bounds(vals,0.01)
        for r in rs: r[score]=minmax(r.get(raw),lo,hi,inv)

# Composite index as weighted mean over available standardized indicators.
for r in rows:
    num=den=0.0; used=0
    for col,w in WEIGHTS.items():
        v=r.get(col)
        if v is not None:
            num += w*v; den += w; used += 1
    r['usefulness_index_weighted_mean']=num/den if den else None
    r['usefulness_index_0_100']=100*r['usefulness_index_weighted_mean'] if den else None
    r['indicator_count_used']=used
    r['index_status']='baseline_provisional' if used>=5 else 'insufficient_indicators'

fields=['symbol','fiscal_year','company_name','announcement_id','date_title','date_publish','audited',
        'predictive_value_score','confirmatory_value_score','faithful_representation_score','neutrality_score',
        'completeness_score','understandability_score','comparability_score','timeliness_score','conservatism_score',
        'usefulness_index_weighted_mean','usefulness_index_0_100','indicator_count_used','index_status',
        'annual_stock_return_raw','next_year_stock_return_raw','net_income','operating_cash_flow','total_assets',
        'extracted_item_count','mean_match_score','link']

out=OUTDIR/'financial_information_usefulness_index_baseline.csv'
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k) for k in fields} for r in rows])

with (OUTDIR/'weights_baseline.json').open('w',encoding='utf-8') as f:
    json.dump({'method':'weighted_mean_available_indicators','normalization':'year-wise winsorized min-max','weights':WEIGHTS,
               'note':'Equal weights are only the baseline. Replace with weights estimated by KH, PSO, and Cultural Algorithm.'},f,ensure_ascii=False,indent=2)

indicator_rows=[
 ['predictive_value_score','ارزش پیش‌بینی‌کنندگی','معکوس نوسان رشد سود سال بعد','provisional','annual data proxy'],
 ['confirmatory_value_score','ارزش تأییدکنندگی','همسویی تغییر سود و بازده سهام','provisional','raw return currently unadjusted'],
 ['faithful_representation_score','بیان صادقانه','معکوس قدرمطلق اقلام تعهدی کل / متوسط دارایی','provisional','audit-report enrichment recommended'],
 ['neutrality_score','بی‌طرفی','معکوس قدرمطلق اقلام تعهدی کل / متوسط دارایی','provisional','Modified Jones preferred later'],
 ['completeness_score','کامل بودن','تعداد اقلام استاندارد استخراج‌شده','provisional','auditor non-disclosure count preferred'],
 ['understandability_score','قابل فهم بودن','میانگین اطمینان تطبیق اقلام در parser','provisional','text/readability data preferred'],
 ['comparability_score','قابل مقایسه بودن','ثبات نگاشت بازده به ROA در پنجره 4 ساله','provisional','De Franco quarterly/industry model preferred'],
 ['timeliness_score','به‌موقع بودن','معکوس فاصله پایان سال مالی تا انتشار گزارش','near-final proxy','proposal mentions AGM date; publication date used temporarily'],
 ['conservatism_score','محافظه‌کاری','منفی اقلام تعهدی کل / متوسط دارایی','provisional','Basu asymmetric timeliness preferred'],
]
with (OUTDIR/'indicator_dictionary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['variable','persian_name','calculation','status','limitation']); w.writerows(indicator_rows)

# Coverage summary.
summary=[]
for y,rs in sorted(by_year.items(), key=lambda x:int(x[0])):
    valid=[r for r in rs if r['index_status']=='baseline_provisional']
    summary.append([y,len(rs),len(valid), round(100*len(valid)/len(rs),2) if rs else 0,
                    round(statistics.mean([r['usefulness_index_0_100'] for r in valid]),4) if valid else None])
with (OUTDIR/'index_coverage_by_year.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['fiscal_year','company_year_rows','valid_index_rows','coverage_percent','mean_index_0_100']); w.writerows(summary)

print(out)
print('rows',len(rows),'valid',sum(r['index_status']=='baseline_provisional' for r in rows))
