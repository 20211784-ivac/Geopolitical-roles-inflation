"""Country x calendar-block sensitivity of fixed V13 held-out predictions.
Conditional, post-hoc: countries resampled with replacement and 12/24-month
synchronous calendar blocks; does not account for model re-estimation or news vintages.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);p.add_argument('--reps',type=int,default=1500);a=p.parse_args()
out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
D=pd.read_csv(a.predictions)
rows=[];summ=[]
for (h,ar),d in D.groupby(['horizon','ar_lags']):
 countries=sorted(d.iso3.unique());months=sorted(d.mdate.unique());C=len(countries);T=len(months)
 ix=pd.MultiIndex.from_product([countries,months],names=['iso3','mdate'])
 V=d.set_index(['iso3','mdate']).reindex(ix)
 mats=[V[k].fillna(0).to_numpy().reshape(C,T) for k in ['se_M1_Total','se_M2_Roles','ae_M1_Total','ae_M2_Roles']]
 s1,s2,a1,a2=mats
 S=d.groupby('iso3').agg(mse_total=('se_M1_Total','mean'),mse_roles=('se_M2_Roles','mean'),mae_total=('ae_M1_Total','mean'),mae_roles=('ae_M2_Roles','mean'))
 S['d_pct']=100*(np.sqrt(S.mse_roles/S.mse_total)-1)
 summ.append(dict(horizon=h,ar_lags=ar,countries=C,months=T,pooled_delta_rmse_pct=100*(np.sqrt(s2.sum()/s1.sum())-1),
  country_balanced_rmse_pct=100*(np.sqrt(S.mse_roles.mean()/S.mse_total.mean())-1),
  median_country_delta_rmse_pct=float(S.d_pct.median()),countries_better=int((S.d_pct<0).sum())))
 for block in (12,24):
  rng=np.random.default_rng(20261116+1000*h+10*ar+block)
  R=np.zeros((a.reps,2))
  for b in range(a.reps):
   wc=np.bincount(rng.integers(C,size=C),minlength=C)
   ix=[]
   while len(ix)<T:
    start=int(rng.integers(0,T-block+1));ix.extend(range(start,start+block))
   wt=np.bincount(np.asarray(ix[:T]),minlength=T)
   ss1,ss2,aa1,aa2=(float(wc@A@wt) for A in mats)
   R[b]=[100*(np.sqrt(ss2/ss1)-1),100*(aa2/aa1-1)]
  z=np.quantile(R,[.025,.975],axis=0)
  rows.append(dict(horizon=h,ar_lags=ar,block_months=block,B=a.reps,
   rmse_ci_low=z[0,0],rmse_ci_high=z[1,0],mae_ci_low=z[0,1],mae_ci_high=z[1,1],
   frac_bootstrap_roles_better=float((R[:,0]<0).mean())))
  print('h',h,'AR',ar,'block',block,'RMSE interval',z[:,0].round(4),'MAE interval',z[:,1].round(4),flush=True)
pd.DataFrame(rows).to_csv(out/'V13_country_calendar_bootstrap.csv',index=False)
pd.DataFrame(summ).to_csv(out/'V13_country_balanced_summary.csv',index=False)
print('COMPLETE')
