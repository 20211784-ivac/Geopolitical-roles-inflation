"""V7 source-audited sequential one-month-ahead and contemporaneous nowcast robustness.
Uses input source PAPER GEOPOLITICA_.dta without downloads.
NO real-time vintage assertion; all inputs are the frozen retrospective country-month index.
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('--input',default=str(Path(__file__).resolve().parents[1]/'data'/'PAPER GEOPOLITICA_.dta'))
parser.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'tables'))
parser.add_argument('--figures',default=str(Path(__file__).resolve().parents[1]/'figures'))
a=parser.parse_args()
OUT=Path(a.output); FIG=Path(a.figures);OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
d=pd.read_stata(a.input,convert_categoricals=False)
assert d.shape[0]==44238
excluded=['GAB','GBR','USA','XKX','WSM']
d=d.loc[~d.iso3.isin(excluded)].sort_values(['iso3','mdate']).copy()
rawdirect=d.gpr_ai_init+d.gpr_ai_resp
rawspill=d.gpr_ai_spill
share=np.divide(rawdirect.to_numpy(),(rawdirect+rawspill).to_numpy(),out=np.zeros(len(d)),where=(rawdirect+rawspill).to_numpy()>0)
d['T']=np.log1p(d.gpr_ai_all);d['D']=d['T'].to_numpy()*share
d['S']=d['T'].to_numpy()*(1-share)
d['G']=np.log1p(d.gpr_global_ai)
for t in ['T','D','S','G','headline_yoy','headline_mom']:
 for lag in [1,2]:d[f'L{lag}_{t}']=d.groupby('iso3')[t].shift(lag)
assert np.max(np.abs(d['D']+d['S']-d['T']))<1e-10
assert (d.groupby('iso3').mdate.diff().dropna()==1).all()
FOLDS=[('2013-2015',2012,2013,2015),('2016-2018',2015,2016,2018),('2019-2021',2018,2019,2021),('2022-2023',2021,2022,2023),('2024-2025m3',2023,2024,2025)]
scenarios=[
 ('yoy','contemporaneous', 'headline_yoy', ['L1_headline_yoy','L2_headline_yoy','G'], ['T','L1_T','L2_T'], ['D','S','L1_D','L1_S','L2_D','L2_S']),
 ('mom','contemporaneous', 'headline_mom', ['L1_headline_mom','L2_headline_mom','G'], ['T','L1_T','L2_T'], ['D','S','L1_D','L1_S','L2_D','L2_S']),
 ('yoy','lagged_inputs', 'headline_yoy', ['L1_headline_yoy','L2_headline_yoy','L1_G'], ['L1_T','L2_T'], ['L1_D','L1_S','L2_D','L2_S']),
 ('mom','lagged_inputs', 'headline_mom', ['L1_headline_mom','L2_headline_mom','L1_G'], ['L1_T','L2_T'], ['L1_D','L1_S','L2_D','L2_S']),
]
RESULTS=[];FOLDROWS=[];ALLPRED=[];HAC=[]; BOOT=[]
for target,timing,y,base,tot,roles in scenarios:
 feats=list(dict.fromkeys(base+tot+roles))
 panel=d.loc[d[[y]+feats].notna().all(axis=1)].copy()
 predictions=[]
 for label,end,ystart,yend in FOLDS:
  tr=panel[panel.year<=end].copy();te=panel[(panel.year>=ystart)&(panel.year<=yend)].copy()
  cts=tr.groupby('iso3').size(); ok=cts[cts>=24].index
  tr=tr[tr.iso3.isin(ok)];te=te[te.iso3.isin(ok)]
  if not len(tr) or not len(te):continue
  enc=OneHotEncoder(drop='first',sparse_output=True,handle_unknown='ignore',dtype=np.float64)
  Ctr=enc.fit_transform(tr[['iso3','month']]);Cte=enc.transform(te[['iso3','month']]);
  consttr=sparse.csr_matrix(np.ones((len(tr),1)));constte=sparse.csr_matrix(np.ones((len(te),1)))
  pred={}
  for model,extras in [('AR+global',[]),('Total',tot),('Roles',roles)]:
   f=base+extras
   mu=tr[f].mean().to_numpy(float);sd=tr[f].std(ddof=0).to_numpy(float)
   sd[sd<1e-8]=1
   Xtr=sparse.hstack([consttr,Ctr,sparse.csr_matrix((tr[f].to_numpy(float)-mu)/sd)],format='csr')
   Xte=sparse.hstack([constte,Cte,sparse.csr_matrix((te[f].to_numpy(float)-mu)/sd)],format='csr')
   result=lsqr(Xtr,tr[y].to_numpy(float),atol=1e-10,btol=1e-10,iter_lim=3000)
   if result[1] not in [1,2]:warnings.warn(f'LSQR issue {target} {timing} {label} {model} {result[1]}')
   pred[model]=np.asarray(Xte@result[0]).ravel()
  p=pd.DataFrame({'iso3':te.iso3.to_numpy(),'mdate':te.mdate.to_numpy(),'fold':label,'year':te.year.to_numpy(),'y':te[y].to_numpy(float)})
  for model in ['AR+global','Total','Roles']:
   p[f'p_{model}']=pred[model];p[f'se_{model}']=(p.y-p[f'p_{model}'])**2;p[f'ae_{model}']=abs(p.y-p[f'p_{model}'])
   FOLDROWS.append({'target':target,'timing':timing,'fold':label,'model':model,'n':len(p),'rmse':float(np.sqrt(p[f'se_{model}'].mean())), 'mae':float(p[f'ae_{model}'].mean())})
  predictions.append(p)
 P=pd.concat(predictions,ignore_index=True)
 P.insert(0,'target',target);P.insert(1,'timing',timing)
 ALLPRED.append(P)
 r={'target':target,'timing':timing,'n':len(P),'countries':P.iso3.nunique(),'months':P.mdate.nunique()}
 for model in ['AR+global','Total','Roles']:
  r[f'rmse_{model}']=float(np.sqrt(P[f'se_{model}'].mean()))
  r[f'mae_{model}']=float(P[f'ae_{model}'].mean())
 r['pct_rmse_roles_vs_total']=100*(r['rmse_Roles']/r['rmse_Total']-1)
 r['pct_mae_roles_vs_total']=100*(r['mae_Roles']/r['mae_Total']-1)
 r['pct_rmse_roles_vs_AR']=100*(r['rmse_Roles']/r['rmse_AR+global']-1)
 r['pct_rmse_total_vs_AR']=100*(r['rmse_Total']/r['rmse_AR+global']-1)
 r['pct_countries_roles_lower_mse']=100*P.groupby('iso3').apply(lambda g:g['se_Roles'].mean()<g['se_Total'].mean(),include_groups=False).mean()
 M=P.groupby('mdate',sort=True).agg(n=('y','size'),sseT=('se_Total','sum'),sseR=('se_Roles','sum'),sseA=('se_AR+global','sum'),saeT=('ae_Total','sum'),saeR=('ae_Roles','sum')).reset_index()
 M['diff_mse_mean']=(M.sseR-M.sseT)/M.n
 for lag in [6,12,24]:
  est=sm.OLS(M.diff_mse_mean.to_numpy(float),np.ones((len(M),1))).fit(cov_type='HAC',cov_kwds={'maxlags':lag,'use_correction':True})
  HAC.append({'target':target,'timing':timing,'lags':lag,'mse_diff_equal_month':float(est.params[0]), 'se':float(est.bse[0]),'p':float(est.pvalues[0]), 'n_months':len(M)})
 # Calendar-block bootstrap. Each month carries all countries, retaining within-month correlation.
 mat=M[['n','sseT','sseR','sseA','saeT','saeR']].to_numpy(float)
 rng=np.random.default_rng(20261116)
 for block in [12,24]:
  B=2000;val=np.empty((B,2));nt=len(mat)
  for j in range(B):
   idx=[]
   while len(idx)<nt:
    i=int(rng.integers(0,nt-block+1));idx.extend(range(i,i+block))
   N,st,sr,sa,at,ar=mat[np.asarray(idx[:nt])].sum(axis=0)
   val[j,0]=100*(np.sqrt(sr/N)/np.sqrt(st/N)-1)
   val[j,1]=100*((ar/N)/(at/N)-1)
  lo,hi=np.quantile(val[:,0],[.025,.975]);lo1,hi1=np.quantile(val[:,1],[.025,.975])
  BOOT.append({'target':target,'timing':timing,'block_months':block,'B':B,'rmse_rel_pct_obs':r['pct_rmse_roles_vs_total'], 'rmse_ci_low_pct':float(lo),'rmse_ci_high_pct':float(hi), 'mae_rel_pct_obs':r['pct_mae_roles_vs_total'],'mae_ci_low_pct':float(lo1),'mae_ci_high_pct':float(hi1)})
 RESULTS.append(r)
 print(target,timing,'n',len(P),'RMSE',r['rmse_AR+global'],r['rmse_Total'],r['rmse_Roles'],'pct R-T',r['pct_rmse_roles_vs_total'],'pct R-AR',r['pct_rmse_roles_vs_AR'],flush=True)

RES=pd.DataFrame(RESULTS);RES.to_csv(OUT/'V7_temporal_performance_summary.csv',index=False)
pd.DataFrame(FOLDROWS).to_csv(OUT/'V7_temporal_performance_by_fold.csv',index=False)
pd.DataFrame(HAC).to_csv(OUT/'V7_temporal_paired_loss_HAC.csv',index=False)
pd.DataFrame(BOOT).to_csv(OUT/'V7_temporal_bootstrap_blocks.csv',index=False)
pd.concat(ALLPRED,ignore_index=True).to_csv(OUT/'V7_temporal_predictions.csv.gz',index=False,compression='gzip')

fig,ax=plt.subplots(figsize=(9.5,5.4))
for i,row in RES.iterrows():
 ax.scatter(row.pct_rmse_roles_vs_total,3-i,marker='o',s=70)
ax.axvline(0,ls='--',lw=1)
ax.set_yticks(range(4),['Monthly inflation, 1-month-ahead','YoY inflation, 1-month-ahead','Monthly inflation, same-month','YoY inflation, same-month'])
ax.set_xlabel('Change in RMSE: roles vs total country AI-GPR (%)  [negative = better]')
ax.set_title('Does role decomposition add information across timing and outcomes?')
ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(FIG/'V7_Figure1_Timing_and_Outcome_Robustness.png',dpi=260,bbox_inches='tight');plt.close(fig)
summary={'excluded_reasons_country_cpi_comparability_iso':excluded,'folds':FOLDS,'n_scenarios':len(RESULTS),'scenarios':RESULTS,'caveat':'Retrospective vintages; lagged_inputs is pseudo-real-time sequential forecasts, not archived vintage forecasts.'}
(OUT/'V7_temporal_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('DONE',OUT)
