"""Re-estimate 140-country legacy and 141-country source-checked common-scale FE contrast.
Includes exact decomposition check and sensitivity excluding countries with ever |YoY|>100.
Uses the existing implementation from user's earlier audited script; not a new estimator.
"""
import json, sys, os, argparse
from pathlib import Path
import importlib.util
import pandas as pd, numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
parser=argparse.ArgumentParser(description='V7: reproduce the original 140-country and corrected 141-country FE contrast')
parser.add_argument('--panel',default=str(BASE/'data'/'PAPER GEOPOLITICA_.dta'))
parser.add_argument('--output',default=str(BASE/'tables'))
args=parser.parse_args()
OUT=Path(args.output); OUT.mkdir(exist_ok=True,parents=True)
spec=importlib.util.spec_from_file_location('base',HERE/'v7_fe_replication_base.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.DATA=Path(args.panel)
DF=mod.load_and_derive()
# AI intensity formula may lose T when all role counts are zero. Audit and report rather than silence.
failed=(DF['log_total']-DF['wdirect']-DF['wspill']).abs()
print('n role-total mismatch >1e-8:',int((failed>1e-8).sum()),'max difference:',float(failed.max()))
reg=['headline_yoy_l1','headline_yoy_l2','wdirect','wspill','wdirect_l1','wspill_l1','wdirect_l2','wspill_l2']
legacy_exclude=['FIN','GAB','GBR','USA','XKX','WSM']
corrected_exclude=['GAB','GBR','USA','XKX','WSM']
results=[]
for label, ex in [('V6_legacy_140',legacy_exclude),('V7_corrected_141',corrected_exclude)]:
  d=DF.loc[~DF.iso3.isin(ex)].copy()
  for sensitivity in ['main','drop_countries_with_any_annual_infl_gt100','drop_observations_annual_infl_gt100']:
    dd=d
    if sensitivity=='drop_countries_with_any_annual_infl_gt100':
      high=d.groupby('iso3').headline_yoy.transform(lambda x: (x.abs()>100).any())
      dd=d.loc[~high].copy()
    elif sensitivity=='drop_observations_annual_infl_gt100':
      dd=d.loc[(d.headline_yoy.abs()<=100)| d.headline_yoy.isna()].copy()
    print('estimating',label,sensitivity,'rows',len(dd),flush=True)
    m=mod.fit(dd,'headline_yoy',reg,label+'_'+sensitivity,hac=True)
    c=mod.contrast(m,{'wdirect':1,'wspill':-1})
    cdk=mod.contrast(m,{'wdirect':1,'wspill':-1},use_dk=True)
    result={'sample':label,'spec':sensitivity,'n':m['n'],'countries':m['G'],'months':m['T'],
      'beta_direct':m['coef']['wdirect'],'beta_spill':m['coef']['wspill'],'contrast':c['coef'],
      'SE_two_way_cluster':c['se'],'p_two_way_cluster':c['p'],'CI_low':c['low95'],'CI_high':c['high95'],
      'p_DK12':cdk['p'],'SE_DK12':cdk['se']}
    results.append(result); print(result,flush=True)

out=OUT/'V7_common_scale_FE_reestimate.csv'
pd.DataFrame(results).to_csv(out,index=False)
(OUT/'V7_common_scale_FE_sample_comparison.json').write_text(json.dumps({'identity_mismatch_observations':int((failed>1e-8).sum()),'identity_max':float(failed.max()),'table':str(out)},indent=2))
print('saved',out)
