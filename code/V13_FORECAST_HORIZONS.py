#!/usr/bin/env python3
"""Horizon 1/3/6 retrospective forecasts of annual CPI inflation, independently audited.
Data inputs: original PAPER GEOPOLITICA_.dta, 2000m1--2025m3; no energy data imputed.
Definition: target y_i,t, information set includes y_i,t-h, lagged global AI-GPR,
country total news or direct/spillover decomposition at months t-h and t-h-1.
Fold-specific estimation closes before the earliest target's information month.
The index has been revised ex post and CPI release vintages are unavailable.
Accordingly: NOT an operational real-time forecast, no causal claims.
This code sets all horizons, AR lag variants and inference diagnostics in advance.
"""
import argparse, hashlib, json, warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.preprocessing import OneHotEncoder

parser=argparse.ArgumentParser()
parser.add_argument('--panel',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
FOLDS=[('2013-2015',2013,2015),('2016-2018',2016,2018),('2019-2021',2019,2021),('2022-2023',2022,2023),('2024-2025m3',2024,2025)]
HORIZONS=(1,3,6)
AR_LAGS=(2,12)
B=2000
SEED=20261116
EXCL=['GAB','GBR','USA','XKX','WSM']

def sha256(path):
 h=hashlib.sha256();f=open(path,'rb')
 for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 f.close();return h.hexdigest()

def prepare(p):
 d=pd.read_stata(p,convert_categoricals=False)
 assert len(d)==44238 and d.iso3.nunique()==146
 d=d[~d.iso3.isin(EXCL)].sort_values(['iso3','mdate']).copy()
 assert len(d)==141*303 and d.iso3.nunique()==141
 assert (d.groupby('iso3').mdate.diff().dropna()==1).all()
 x=(d.gpr_ai_init+d.gpr_ai_resp).to_numpy(float);s=d.gpr_ai_spill.to_numpy(float)
 share=np.divide(x,x+s,out=np.zeros_like(x),where=x+s>0)
 d['T']=np.log1p(d.gpr_ai_all);d['D']=d['T']*share;d['S']=d['T']-d['D'];d['G']=np.log1p(d.gpr_global_ai)
 assert np.allclose(d["D"]+d["S"],d["T"],atol=1e-11)
 for v in ['headline_yoy','T','D','S','G']:
  for k in range(1,19):
   d[f'L{k}_{v}']=d.groupby('iso3')[v].shift(k)
 return d

d=prepare(args.panel)
rows=[];folds=[];allpred=[];boots=[];extreme=[]
for horizon in HORIZONS:
 for ar in AR_LAGS:
  # The same y and lag sample is used for M0, M1, and M2 in this scenario.
  base=[f'L{k}_headline_yoy' for k in range(horizon,horizon+ar)]+[f'L{horizon}_G']
  total=[f'L{k}_T' for k in (horizon,horizon+1)]
  roles=[f'L{k}_{v}' for k in (horizon,horizon+1) for v in ['D','S']]
  feats=list(dict.fromkeys(base+total+roles))
  valid=d[['iso3','month','mdate','year','headline_yoy']+feats].dropna().copy()
  predictions=[]
  for label,start,end in FOLDS:
   # Earliest observed outcome and its information date determine the estimation cutoff.
   # This prevents inadvertent estimation on future outcomes unavailable when the first
   # test prediction in a fold must be issued.
   test=valid[valid.year.between(start,end)].copy()
   first_target=(start-1960)*12   # Stata Jan1960=0; Jan2013=636
   cutoff=first_target-horizon
   train=valid[valid.mdate<=cutoff].copy()
   valid_ids=train.groupby('iso3').size();elig=valid_ids[valid_ids>=24].index
   train=train[train.iso3.isin(elig)].copy()
   test=test[test.iso3.isin(elig)].copy()
   if train.empty or test.empty: continue
   assert train.mdate.max()<=cutoff
   assert test.mdate.min()>=first_target
   # One-hot country & target season; target season is calendar-known at forecast origin.
   enc=OneHotEncoder(drop='first',sparse_output=True,handle_unknown='ignore',dtype=float)
   Ctr=enc.fit_transform(train[['iso3','month']]);Cte=enc.transform(test[['iso3','month']])
   Itr=sparse.csr_matrix(np.ones((len(train),1)));Ite=sparse.csr_matrix(np.ones((len(test),1)))
   preds={}
   for model,extra in [('M0_ARglobal',[]),('M1_Total',total),('M2_Roles',roles)]:
    fs=base+extra
    mu=train[fs].mean().to_numpy(float);sd=train[fs].std(ddof=0).to_numpy(float)
    sd[~np.isfinite(sd)|(sd<1e-10)]=1
    Atr=(train[fs].to_numpy(float)-mu)/sd;Ate=(test[fs].to_numpy(float)-mu)/sd
    Xtr=sparse.hstack([Itr,Ctr,sparse.csr_matrix(Atr)],format='csr')
    Xte=sparse.hstack([Ite,Cte,sparse.csr_matrix(Ate)],format='csr')
    fit=lsqr(Xtr,train.headline_yoy.to_numpy(float),atol=1e-10,btol=1e-10,iter_lim=3500)
    if fit[1] not in (1,2):warnings.warn(f'LSQR convergence: h{horizon} AR{ar} {label} {model}={fit[1]}')
    preds[model]=np.asarray(Xte@fit[0]).ravel()
   p=pd.DataFrame({'horizon':horizon,'ar_lags':ar,'fold':label,
     'iso3':test.iso3.to_numpy(),'mdate':test.mdate.to_numpy(),
     'year':test.year.to_numpy(),'y':test.headline_yoy.to_numpy(float),
     'estimation_cutoff_mdate':cutoff})
   for m,v in preds.items():
    p['p_'+m]=v;p['se_'+m]=(p.y-v)**2;p['ae_'+m]=abs(p.y-v)
    folds.append({'horizon':horizon,'ar_lags':ar,'fold':label,'model':m,'n':len(test),'train_end_mdate':cutoff,
      'rmse':float(np.sqrt(p['se_'+m].mean())),'mae':float(p['ae_'+m].mean())})
   predictions.append(p)
  P=pd.concat(predictions,ignore_index=True)
  assert not P.duplicated(['iso3','mdate']).any()
  assert np.isfinite(P[['y','p_M0_ARglobal','p_M1_Total','p_M2_Roles']].to_numpy()).all()
  allpred.append(P)
  row={'horizon':horizon,'ar_lags':ar,'N':len(P),'countries':P.iso3.nunique(),'months':P.mdate.nunique()}
  for m in ['M0_ARglobal','M1_Total','M2_Roles']:
   row['rmse_'+m]=float(np.sqrt(P['se_'+m].mean()))
   row['mae_'+m]=float(P['ae_'+m].mean())
  row['delta_rmse_roles_total_pct']=100*(row['rmse_M2_Roles']/row['rmse_M1_Total']-1)
  row['delta_mae_roles_total_pct']=100*(row['mae_M2_Roles']/row['mae_M1_Total']-1)
  row['delta_rmse_roles_base_pct']=100*(row['rmse_M2_Roles']/row['rmse_M0_ARglobal']-1)
  C=P.groupby('iso3').agg(mse_total=('se_M1_Total','mean'),mse_roles=('se_M2_Roles','mean'))
  row['countries_with_lower_mse_roles']=int((C.mse_roles<C.mse_total).sum())
  # Future outcomes at >100 are removed only to test error influence; not model refitted.
  Q=P[P.y.abs()<=100]
  extreme.append({'horizon':horizon,'ar_lags':ar,'N_all':len(P),'N_nonextreme':len(Q),
       'countries_with_extremes':int(P[P.y.abs()>100].iso3.nunique()),
       'pct_rmse_roles_total_full':row['delta_rmse_roles_total_pct'],
       'pct_rmse_roles_total_nonextreme':100*(np.sqrt(Q.se_M2_Roles.mean()/Q.se_M1_Total.mean())-1)})
  # Cross-country synchronous calendar blocks (resamples all countries for a month).
  M=P.groupby('mdate').agg(n=('y','size'),sseT=('se_M1_Total','sum'),sseR=('se_M2_Roles','sum'),saeT=('ae_M1_Total','sum'),saeR=('ae_M2_Roles','sum')).reset_index()
  mat=M[['n','sseT','sseR','saeT','saeR']].to_numpy(float)
  rng=np.random.default_rng(SEED+100*horizon+ar)
  for block in (12,24):
   R=np.zeros((B,2));months=len(M)
   for b in range(B):
    ix=[]
    while len(ix)<months:
     start_idx=int(rng.integers(0,months-block+1));ix.extend(range(start_idx,start_idx+block))
    N,sseT,sseR,saeT,saeR=mat[np.array(ix[:months])].sum(axis=0)
    R[b,0]=100*(np.sqrt(sseR/sseT)-1)
    R[b,1]=100*(saeR/saeT-1)
   ci=np.quantile(R,[.025,.975],axis=0)
   boots.append({'horizon':horizon,'ar_lags':ar,'block_months':block,'B':B,
        'rmse_ci_low':float(ci[0,0]),'rmse_ci_high':float(ci[1,0]),
        'mae_ci_low':float(ci[0,1]),'mae_ci_high':float(ci[1,1])})
  rows.append(row)
  print('H',horizon,'AR',ar,'N',len(P),'countries',P.iso3.nunique(),
   'RMSE base,total,roles:',*(round(row['rmse_'+m],6) for m in ['M0_ARglobal','M1_Total','M2_Roles']),
   'roles-total %',round(row['delta_rmse_roles_total_pct'],5),
   'MAE%',round(row['delta_mae_roles_total_pct'],5),flush=True)

pd.DataFrame(rows).to_csv(OUT/'V13_horizon_performance.csv',index=False)
pd.DataFrame(folds).to_csv(OUT/'V13_horizon_performance_folds.csv',index=False)
pd.DataFrame(boots).to_csv(OUT/'V13_horizon_uncertainty.csv',index=False)
pd.DataFrame(extreme).to_csv(OUT/'V13_horizon_outlier_sensitivity.csv',index=False)
pd.concat(allpred,ignore_index=True).to_csv(OUT/'V13_horizon_predictions.csv.gz',index=False,compression='gzip')
config={'data_sha256':sha256(args.panel),'N_input':44238,'N_after_cpi_screen':len(d),
 'excluded_cpi':['GAB','GBR','USA','XKX','WSM'],'horizons':list(HORIZONS),'ar_lags':list(AR_LAGS),
 'folds':FOLDS,'calendar_block_bootstrap_reps':B,'seed':SEED,
 'note':'The month-year origin is target month - horizon. Estimation ends before the earliest origin in each test fold; all target and predictor dates respect this ordering.',
 'vintage_limit':'Retrospective frozen AI-GPR and ex-post CPI observations; no release-time archived vintages. Neither actual real-time prediction nor identified causal effects.'}
(OUT/'V13_method_and_data_manifest.json').write_text(json.dumps(config,indent=2))
print('COMPLETE',OUT)
