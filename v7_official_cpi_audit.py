"""Check selected panel CPI values against official external reference tables.
Official reference values manually transcribed from: ONS Dec 2024 CPIH/CPI table 1;
Statistics Finland Jan, Feb, Nov, Dec 2024 CPI 2015=100;
Gabon DGS Dec 2024 national IHPC 2018=100; BLS USA Dec 2024 12mo CPI-U.
The external values are NOT full-panel replacements; this program verifies only the listed months.
"""
from pathlib import Path
import pandas as pd, numpy as np, json, argparse
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='V7 official source-month CPI checks; no full-series imputation')
parser.add_argument('--panel',default=str(R/'data'/'PAPER GEOPOLITICA_.dta'))
parser.add_argument('--output',default=str(R/'tables'))
args=parser.parse_args()
OUT=Path(args.output); OUT.mkdir(parents=True,exist_ok=True)
panel=pd.read_stata(args.panel,convert_categoricals=False)
ons_cpih=[130.0,130.8,131.6,132.2,132.7,133.0,132.9,133.4,133.5,134.3,134.6,135.1]
ons_cpi=[131.5,132.3,133.0,133.5,133.9,134.1,133.8,134.3,134.2,135.0,135.1,135.6]
records=[]
for iso in ['GBR','USA']:
 for m in range(1,13):
  v=panel.loc[(panel.iso3==iso)&(panel.year==2024)&(panel.month==m)].iloc[0]
  records.append({'iso3':iso,'month':f'2024-{m:02d}','panel_hcpi':float(v.hcpi),'official_index':ons_cpih[m-1],
    'official_series':'UK CPIH 2015=100 (ONS 2024-12 table 1)','index_abs_diff':abs(v.hcpi-ons_cpih[m-1]),
    'interpretation':'GBR index exactly matches UK CPIH' if iso=='GBR' else 'USA value exactly equals UK CPIH, not US CPI'})
for month,val in [(1,121.9),(2,122.5),(11,122.5),(12,122.4)]:
 for iso in ['FIN','GAB']:
  v=panel.loc[(panel.iso3==iso)&(panel.year==2024)&(panel.month==month)].iloc[0]
  records.append({'iso3':iso,'month':f'2024-{month:02d}','panel_hcpi':float(v.hcpi),'official_index':val,
    'official_series':'Finnish CPI 2015=100 (Statistics Finland 2024 individual releases)',
    'index_abs_diff':abs(float(v.hcpi)-val),'interpretation':'FIN matches Finnish CPI to 0.1 rounding' if iso=='FIN' else 'GAB matches FIN improperly'})
# Actual Dec 2024 variation (12-month percentage, comparability immune to rebasing).
for iso,source,realrate in [('GAB','Gabon DGS national IHPC December 2024 (base 2018)',0.9),('USA','US BLS CPI-U Dec 2024 annual change',2.9),('GBR','UK ONS CPI Dec 2024 annual change',2.5),('FIN','Statistics Finland CPI December 2024 annual change',0.7)]:
 v=panel.loc[(panel.iso3==iso)&(panel.year==2024)&(panel.month==12)].iloc[0]
 records.append({'iso3':iso,'month':'2024-12 (YoY)','panel_hcpi':float(v.headline_yoy),'official_index':realrate,'official_series':source,
 'index_abs_diff':abs(float(v.headline_yoy)-realrate),'interpretation':'annual inflation vs official: source comparability check'})
df=pd.DataFrame(records)
(OUT/'V7_CPI_official_source_monthly_checks.csv').write_text(df.to_csv(index=False),encoding='utf-8')
# Pair duplicates check all months (code tolerant to NaNs)
def dup(a,b):
 da=panel[panel.iso3==a].sort_values('mdate'); db=panel[panel.iso3==b].sort_values('mdate');
 return int(((da.hcpi.values==db.hcpi.values)&pd.notna(da.hcpi.values)&pd.notna(db.hcpi.values)).sum())
flags={'FIN_GAB_exact_303_months':dup('FIN','GAB'), 'GBR_USA_exact_303_months':dup('GBR','USA'),
 'GBR_matches_OFFICIAL_UK_CPIH_2024_months':int((df[(df.iso3=='GBR') & ~df.month.str.contains('YoY')].index_abs_diff<1e-8).sum()),
 'GBR_matches_OFFICIAL_UK_CPI_2024_months':sum(abs(panel[(panel.iso3=='GBR')&(panel.year==2024)].sort_values('month').hcpi.values[i]-ons_cpi[i])<1e-8 for i in range(12)),
 'FIN_official_months_matched_to_one_decimal':int((df[(df.iso3=='FIN') & ~df.month.str.contains('YoY')].index_abs_diff<=0.051).sum()),
 'data_source_provenance':'ONS Dec 2024 Table 1; Statistics Finland Jan Feb Nov Dec 2024; Gabon DGS IHPC Dec 2024; US BLS 2024-12 release'}
(OUT/'V7_CPI_source_audit_summary.json').write_text(json.dumps(flags,indent=2,default=int),encoding='utf-8')
print(flags)
