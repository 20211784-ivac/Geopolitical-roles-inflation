#!/usr/bin/env python3
"""V11: fixed-forecast, paired two-dimensional resampling of held-out losses.

Source is the archival V7 prediction file delivered inside V10; models are NOT refitted.
Resamples countries with replacement and overlapping, shared calendar-month blocks;
conditional exploratory uncertainty only, not an exact nested-model hypothesis test.
"""
from pathlib import Path
import numpy as np, pandas as pd, hashlib, json, argparse
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);p.add_argument('--replications',type=int,default=3000);args=p.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
seed=20261116
file=Path(args.predictions)
rawsha=hashlib.sha256(file.read_bytes()).hexdigest()
df=pd.read_csv(file)
assert not df.duplicated(['target','timing','iso3','mdate']).any(), 'Duplicate prediction keys'
assert set(df['target'])=={'yoy','mom'}
assert set(df['timing'])=={'contemporaneous','lagged_inputs'}
results=[];percountry=[];delta_bycountry=[];bootrows=[]
for tt in ['yoy','mom']:
 for tm in ['contemporaneous','lagged_inputs']:
  dat=df[(df.target==tt)&(df.timing==tm)].copy()
  countries=sorted(dat.iso3.unique());months=sorted(dat.mdate.unique());
  C=len(countries);T=len(months);N=len(dat)
  assert (C,T)==(141,147)
  assert np.allclose(dat['se_Total'],(dat['y']-dat['p_Total'])**2)
  assert np.allclose(dat['se_Roles'],(dat['y']-dat['p_Roles'])**2)
  assert np.allclose(dat['ae_Total'],np.abs(dat['y']-dat['p_Total']))
  assert np.allclose(dat['ae_Roles'],np.abs(dat['y']-dat['p_Roles']))
  dat['drmse']=dat['se_Roles']-dat['se_Total']
  dat['dmae']=dat['ae_Roles']-dat['ae_Total']
  cs=dat.groupby('iso3').agg(n=('y','size'), mse_total=('se_Total','mean'),mse_roles=('se_Roles','mean'),mae_total=('ae_Total','mean'),mae_roles=('ae_Roles','mean'),lossdiff=('drmse','mean')).reset_index()
  cs['country_rmse_rel_pct']=100*(np.sqrt(cs.mse_roles/cs.mse_total)-1)
  cs['country_mae_rel_pct']=100*(cs.mae_roles/cs.mae_total-1)
  cs['target']=tt;cs['timing']=tm
  percountry.append(cs)
  pooled=100*(np.sqrt(dat.se_Roles.mean()/dat.se_Total.mean())-1)
  poolmae=100*(dat.ae_Roles.mean()/dat.ae_Total.mean()-1)
  countrybalanced=100*(np.sqrt(cs.mse_roles.mean()/cs.mse_total.mean())-1)
  typical=float(cs.country_rmse_rel_pct.median())
  typical_mae=float(cs.country_mae_rel_pct.median())
  results.append(dict(target=tt,timing=tm,N=N,countries=C,months=T,pooled_dRMSE_pct=pooled,pooled_dMAE_pct=poolmae,country_balanced_dRMSE_pct=countrybalanced,median_country_dRMSE_pct=typical,median_country_dMAE_pct=typical_mae,mean_country_dRMSE_pct=cs.country_rmse_rel_pct.mean(),mean_country_dMAE_pct=cs.country_mae_rel_pct.mean(),countries_roles_lower_MSE=int((cs.mse_roles<cs.mse_total).sum()),countries_roles_lower_MAE=int((cs.mae_roles<cs.mae_total).sum()),countries_with_test_n_lt24=int((cs.n<24).sum())))
  ix=pd.MultiIndex.from_product([countries,months],names=['iso3','mdate']); gr=dat.set_index(['iso3','mdate']).reindex(ix)
  mk=gr.se_Total.notna().to_numpy().reshape(C,T)
  mats=[gr[col].fillna(0).to_numpy().reshape(C,T) for col in ['se_Total','se_Roles','ae_Total','ae_Roles']]
  m1,m2,a1,a2=mats
  assert np.isclose(pooled,100*(np.sqrt(m2.sum()/m1.sum())-1),atol=1e-8)
  print(tt,tm,N,C,T,'pooled',pooled,'balanced',countrybalanced,'med',typical,'improved_count',int((cs.mse_roles<cs.mse_total).sum()),flush=True)
  for block in (12,24):
   rng=np.random.default_rng(seed+('yoy'==tt)*100000+('lagged_inputs'==tm)*10000+block)
   for mode in ('calendar_only','country_only','country_calendar'):
    diffr=[];diffa=[]
    for b in range(args.replications):
     wc=np.bincount(rng.integers(0,C,C),minlength=C) if mode!='calendar_only' else np.ones(C,dtype=int)
     if mode=='country_only':wt=np.ones(T,dtype=int)
     else:
      ks=int(np.ceil(T/block));starts=rng.integers(0,T-block+1,size=ks)
      periods=np.concatenate([np.arange(s,s+block) for s in starts])[:T]
      wt=np.bincount(periods,minlength=T)
     v1=wc@m1@wt;v2=wc@m2@wt;a1s=wc@a1@wt;a2s=wc@a2@wt
     if v1>0 and a1s>0:
      diffr.append(100*(np.sqrt(v2/v1)-1));diffa.append(100*(a2s/a1s-1))
    diffr=np.asarray(diffr);diffa=np.asarray(diffa)
    for name,v in [('RMSE',diffr),('MAE',diffa)]:
     bootrows.append(dict(target=tt,timing=tm,mode=mode,block_months=block,metric=name,B=len(v),estimate_pct=pooled if name=='RMSE' else poolmae,ci_low_pct=float(np.quantile(v,.025)),ci_high_pct=float(np.quantile(v,.975)),replicate_improvement_fraction=float(np.mean(v<0)),bootstrap_median_pct=float(np.median(v))))
  if tt=='yoy' and tm=='contemporaneous':
   cd=cs[['iso3','n','mse_total','mse_roles','mae_total','mae_roles','country_rmse_rel_pct','country_mae_rel_pct','lossdiff']].copy()
   cd.sort_values('country_rmse_rel_pct').to_csv(out/'V11_country_rmse_changes.csv',index=False)
R=pd.DataFrame(results);B=pd.DataFrame(bootrows)
R.to_csv(out/'V11_equal_country_loss_summary.csv',index=False)
B.to_csv(out/'V11_paired_country_calendar_bootstrap.csv',index=False)
pd.concat(percountry,ignore_index=True).to_csv(out/'V11_country_outcomes_by_scenario.csv',index=False)
(out/'V11_analysis_manifest.json').write_text(json.dumps({'prediction_source_sha256':rawsha,'seed':seed,'resampling':'Paired with-replacement countries; overlapping consecutive common-calendar-month blocks; fixed model predictions. Months chosen within 147-month evaluation horizon and entire country panel retained at sampled calendar dates. Country-only and month-only comparisons also included. Non-representative non-IID countries; no new data vintages or model-fitting uncertainty.','interpretation':'post hoc conditional sensitivity; no exact probability statements about full real-time forecasting performance','B':args.replications},indent=2))
print('\nRESULTS\n',R.to_string(index=False))
print('\nBOOTSTRAPS same-month yoy\n',B[(B.target=='yoy')&(B.timing=='contemporaneous')].to_string(index=False))
