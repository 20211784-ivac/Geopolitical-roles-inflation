#!/usr/bin/env python3
"""Independent source-identity audit against official posted 2024 monthly national index tables.
Official public source tables accessed 2026-11-16. No automated network required.
Sources:
  FIN: Statistics Finland CPI 2015=100 table, https://stat.fi/en/statistics/khi
  GBR: ONS Consumer Price Inflation UK, December 2024, Table 1
        https://www.ons.gov.uk/economy/inflationandpriceindices/bulletins/consumerpriceinflation/december2024/previous/v1
  USA: BLS CPI-U Dec 2024 12-month change = 2.9%
        https://www.bls.gov/news.release/archives/cpi_01152025.htm
The Finnish figures are published to two decimals; ONS figures to one decimal.
No unsupported full-historical validation or interpolation is claimed.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--out',required=True);args=p.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
d=pd.read_stata(args.panel,convert_categoricals=False)
assert len(d)==44238 and d.iso3.nunique()==146
# Checked against the 2024 historical table currently displayed by Statistics Finland (CPI 2015=100)
fin=[121.85,122.50,122.36,122.39,122.25,122.23,122.49,121.89,122.26,122.92,122.47,122.39]
# ONS 2024 year-month index table (CPIH and CPI, both 2015=100)
uk_cpih=[130.0,130.8,131.6,132.2,132.7,133.0,132.9,133.4,133.5,134.3,134.6,135.1]
uk_cpi=[131.5,132.3,133.0,133.5,133.9,134.1,133.8,134.3,134.2,135.0,135.1,135.6]
rows=[]
for i in range(12):
 mo=i+1
 for iso,off,kind,rounding in [('FIN',fin[i],'Finnish national CPI 2015=100',2),('GBR',uk_cpih[i],'UK CPIH 2015=100',1)]:
  xx=d.loc[(d.iso3==iso)&(d.year==2024)&(d.month==mo)].iloc[0]
  actual=float(xx.hcpi); match=round(actual,rounding)==round(float(off),rounding)
  rows.append({'iso3':iso,'year':2024,'month':mo,'official_series':kind,'official_index':off,
   'supplied_panel_hcpi':actual,'matches_official_to_reported_precision':match})
r=pd.DataFrame(rows).sort_values(['iso3','month']);r.to_csv(out/'V15_OFFICIAL_2024_INDEX_24_CHECKS.csv',index=False)
assert r.matches_official_to_reported_precision.all()
# UK HCPI matches CPIH not CPI for all twelve months, at published one-decimal precision.
uk=d[(d.iso3=='GBR')&(d.year==2024)].sort_values('month');assert len(uk)==12
assert all(round(float(v),1)!=round(float(o),1) for v,o in zip(uk.hcpi,uk_cpi))
# Duplicate histories are exact full paths, 303 months.
for aa,bb in [('FIN','GAB'),('GBR','USA')]:
 a=d[d.iso3==aa].sort_values('mdate');b=d[d.iso3==bb].sort_values('mdate')
 assert len(a)==len(b)==303
 assert (a.mdate.to_numpy()==b.mdate.to_numpy()).all()
 assert (a.hcpi.to_numpy()==b.hcpi.to_numpy()).all()
usa_panel=float(d.loc[(d.iso3=='USA')&(d.year==2024)&(d.month==12),'headline_yoy'].iloc[0])
assert abs(usa_panel-3.5249)<0.005
output={'status':'PASSED','finland_2024_exact_index_matches':12,'uk_2024_cpih_matches':12,'uk_2024_cpi_matches':0,
 'fin_gabon_duplicate_months':303,'gbr_usa_duplicate_months':303,
 'us_dec_2024_panel_yoy_pct':usa_panel,'us_dec_2024_official_bls_yoy_pct':2.9,
 'methodological_limit':'Monthly 2024 source-identity checks do not validate the whole 2000-2025 CPI history nor establish how misassignment arose.',
 'sources':{'finland':'Statistics Finland CPI 2015=100 published annual monthly table for 2024, https://stat.fi/en/statistics/khi',
 'uk':'ONS Consumer Price Inflation, December 2024, Table 1, CPIH and CPI',
 'usa':'US BLS CPI-U December 2024 (release January 15, 2025)'}}
(out/'V15_CPI_AUDIT_SUMMARY.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
