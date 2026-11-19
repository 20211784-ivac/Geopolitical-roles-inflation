#!/usr/bin/env python3
"""Economic-unit translation of existing audited FE role contrast; no new regression.
The median news intensity is computed only on the preferred FE complete-case sample.
10-percentage-point shift in Direct share at fixed T is hypothetical regression-scale
interpretation, not a policy intervention or identified effect.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
d=pd.read_stata(a.panel,convert_categoricals=False)
d=d[~d.iso3.isin(['GAB','GBR','USA','XKX','WSM'])].sort_values(['iso3','mdate']).copy()
raw=(d.gpr_ai_init+d.gpr_ai_resp).to_numpy(float);sp=d.gpr_ai_spill.to_numpy(float)
q=np.divide(raw,raw+sp,out=np.zeros_like(raw),where=raw+sp>0)
d['T']=np.log1p(d.gpr_ai_all);d['q']=q;d['D']=d['T']*d['q'];d['S']=d['T']-d['D']
for v in ['headline_yoy','D','S']:
 for lag in (1,2):d[f'L{lag}_{v}']=d.groupby('iso3')[v].shift(lag)
samp=d.dropna(subset=['headline_yoy','D','S','L1_D','L2_D','L1_S','L2_S','L1_headline_yoy','L2_headline_yoy'])
assert len(samp)==33780 and samp.iso3.nunique()==141
T=float(samp['T'].median());delta=0.0964116730480407;ci=[-0.0188,0.2116]
scale=0.1*T
v={'n':len(samp),'countries':141,'zero_total_news_share':float((samp['T']==0).mean()),
'median_total_ai_gpr':float(samp.gpr_ai_all.median()),'median_log_total_ai_gpr':T,
'conditional_direct_share_change':0.10,'role_difference_per_unit':delta,
'model_implied_pp_at_median_intensity_10ppt_direct_share_shift':float(delta*scale),
'ci95_pp_from_reported_rounded_CI_at_10ppt_shift':[float(x*scale) for x in ci],
'interpretation':'This is algebraic coefficient-unit translation at sample-median T, not a feasible causal intervention, effect of physical conflict, or independent new estimate.'}
(out/'V15_ECONOMIC_SCALE.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
