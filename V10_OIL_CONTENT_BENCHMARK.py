#!/usr/bin/env python3
"""Exploratory oil-content AI-GPR benchmark using source archived panel only.
Re-estimates all models at each pre-defined V7 origin. No full sample future rows train.
Not external structural energy exposure, not real-time vintages, not causal.
"""
from pathlib import Path
import argparse,json,warnings
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('--input',required=True)
parser.add_argument('--out',required=True)
a=parser.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
d=pd.read_stata(a.input,convert_categoricals=False)
assert len(d)==44238 and d.iso3.nunique()==146
excl=['GAB','GBR','USA','XKX','WSM'];d=d.loc[~d.iso3.isin(excl)].sort_values(['iso3','mdate']).copy()
direct=d.gpr_ai_init + d.gpr_ai_resp
sumroles=direct+d.gpr_ai_spill
q=np.divide(direct.to_numpy(float),sumroles.to_numpy(float),out=np.zeros(len(d)),where=sumroles.to_numpy()>0)
d['T']=np.log1p(d.gpr_ai_all);d['D']=d.T.values if False else d['T'].to_numpy()*q
d['S']=d['T']-d['D']; d['G']=np.log1p(d.gpr_global_ai);d['O']=np.log1p(d.gpr_oil)
assert np.max(np.abs(d['D']+d['S']-d['T']))<1e-10
assert d.groupby('iso3').mdate.diff().dropna().eq(1).all()
for v in ['T','D','S','G','O','headline_yoy','headline_mom']:
 for lag in [1,2]:d['L'+str(lag)+'_'+v]=d.groupby('iso3')[v].shift(lag)
FOLDS=[('2013–15',2012,2013,2015),('2016–18',2015,2016,2018),('2019–21',2018,2019,2021),('2022–23',2021,2022,2023),('2024–25m3',2023,2024,2025)]
scenarios=[('YoY','Same-month','headline_yoy',['L1_headline_yoy','L2_headline_yoy','G'],['T','L1_T','L2_T'],['D','S','L1_D','L1_S','L2_D','L2_S'],'O'),('MoM','Same-month','headline_mom',['L1_headline_mom','L2_headline_mom','G'],['T','L1_T','L2_T'],['D','S','L1_D','L1_S','L2_D','L2_S'],'O'),('YoY','Lag-only','headline_yoy',['L1_headline_yoy','L2_headline_yoy','L1_G'],['L1_T','L2_T'],['L1_D','L1_S','L2_D','L2_S'],'L1_O'),('MoM','Lag-only','headline_mom',['L1_headline_mom','L2_headline_mom','L1_G'],['L1_T','L2_T'],['L1_D','L1_S','L2_D','L2_S'],'L1_O')]
modelnames=['ARglobal','Total','Roles','OilTotal','OilRoles']
results=[];foldrows=[];predrows=[];bootstrap=[]
for target,timing,y,base,tot,roles,oil in scenarios:
 allfeat=list(dict.fromkeys(base+tot+roles+[oil]));panel=d.loc[d[[y]+allfeat].notna().all(axis=1)].copy()
 pp=[]
 for label,end,sy,ey in FOLDS:
  tr=panel[panel.year<=end].copy();te=panel[panel.year.between(sy,ey)].copy()
  ok=tr.groupby('iso3').size();ok=ok[ok>=24].index;tr=tr[tr.iso3.isin(ok)];te=te[te.iso3.isin(ok)]
  assert len(tr)>10000 and len(te)>1000
  enc=OneHotEncoder(drop='first',sparse_output=True,handle_unknown='ignore',dtype=np.float64)
  Ctr=enc.fit_transform(tr[['iso3','month']]);Cte=enc.transform(te[['iso3','month']]);onestr=sparse.csr_matrix(np.ones((len(tr),1)));oneste=sparse.csr_matrix(np.ones((len(te),1)))
  variants=[('ARglobal',base),('Total',base+tot),('Roles',base+roles),('OilTotal',base+tot+[oil]),('OilRoles',base+roles+[oil])]
  row=pd.DataFrame({'iso3':te.iso3.to_numpy(),'mdate':te.mdate.to_numpy(),'fold':label,'year':te.year.to_numpy(),'y':te[y].to_numpy(float)})
  for model,ff in variants:
   mu=tr[ff].mean().to_numpy(float);sd=tr[ff].std(ddof=0).to_numpy(float);sd[sd<1e-8]=1
   xtr=sparse.hstack([onestr,Ctr,sparse.csr_matrix((tr[ff].to_numpy(float)-mu)/sd)],format='csr')
   xte=sparse.hstack([oneste,Cte,sparse.csr_matrix((te[ff].to_numpy(float)-mu)/sd)],format='csr')
   est=lsqr(xtr,tr[y].to_numpy(float),atol=1e-10,btol=1e-10,iter_lim=3000)
   if est[1] not in [1,2]:warnings.warn(f'LSQR stop status={est[1]} {target} {timing} {label} {model}')
   pred=np.asarray(xte@est[0]).ravel();row['p_'+model]=pred;row['se_'+model]=(row.y-pred)**2;row['ae_'+model]=abs(row.y-pred)
   foldrows.append({'target':target,'timing':timing,'fold':label,'model':model,'N':len(te),'rmse':np.sqrt(np.mean(row['se_'+model])),'mae':np.mean(row['ae_'+model])})
  pp.append(row)
 P=pd.concat(pp,ignore_index=True);P.insert(0,'target',target);P.insert(1,'timing',timing)
 if target=='YoY':predrows.append(P)
 res={'target':target,'timing':timing,'N':len(P),'countries':P.iso3.nunique(),'months':P.mdate.nunique()}
 for model in modelnames:
  res['RMSE_'+model]=np.sqrt(P['se_'+model].mean());res['MAE_'+model]=P['ae_'+model].mean()
 for a1,b1,tag in [('Roles','Total','Roles-v-Total'),('OilTotal','Total','OilTotal-v-Total'),('OilRoles','OilTotal','OilRoles-v-OilTotal'),('OilRoles','Roles','OilRoles-v-Roles'),('OilTotal','ARglobal','OilTotal-v-ARglobal')]:
  res['rmse_pct_'+tag]=100*(res['RMSE_'+a1]/res['RMSE_'+b1]-1)
  res['mae_pct_'+tag]=100*(res['MAE_'+a1]/res['MAE_'+b1]-1)
 # loss uncertainty: calendar-month moving blocks, same countries in month, frozen prediction diagnostics
 mon=P.groupby('mdate',sort=True).agg(N=('y','size'),**{f'sse_{m}':('se_'+m,'sum') for m in modelnames},**{f'sae_{m}':('ae_'+m,'sum') for m in modelnames}).reset_index()
 T=len(mon);rng=np.random.default_rng(20261116)
 for a1,b1,tag in [('Roles','Total','Roles-v-Total'),('OilTotal','Total','OilTotal-v-Total'),('OilRoles','OilTotal','OilRoles-v-OilTotal'),('OilRoles','Roles','OilRoles-v-Roles')]:
  for block in [12,24]:
   B=2500;vals=np.empty((B,2));arr=mon[['N','sse_'+a1,'sse_'+b1,'sae_'+a1,'sae_'+b1]].to_numpy(float)
   for j in range(B):
    pieces=[]
    while len(pieces)<T:
     i=int(rng.integers(0,T-block+1));pieces.extend(range(i,i+block))
    N,sseA,sseB,saeA,saeB=arr[np.array(pieces[:T])].sum(axis=0)
    vals[j,0]=100*(np.sqrt(sseA/sseB)-1);vals[j,1]=100*(saeA/saeB-1)
   interval=np.quantile(vals,[.025,.975],axis=0)
   bootstrap.append({'target':target,'timing':timing,'contrast':tag,'block_months':block,'B':B,'observed_pct_rmse':res['rmse_pct_'+tag],'rmse_low':interval[0,0],'rmse_high':interval[1,0],'observed_pct_mae':res['mae_pct_'+tag],'mae_low':interval[0,1],'mae_high':interval[1,1]})
 results.append(res);print('SCENARIO',target,timing,'N',len(P),'RMSE',*[round(res['RMSE_'+m],7) for m in modelnames],flush=True)
res=pd.DataFrame(results);res.to_csv(out/'V10_oil_comparator_performance.csv',index=False)
pd.DataFrame(foldrows).to_csv(out/'V10_oil_comparator_by_fold.csv',index=False)
pd.DataFrame(bootstrap).to_csv(out/'V10_oil_comparator_bootstrap.csv',index=False)
pd.concat(predrows,ignore_index=True).to_csv(out/'V10_oil_comparator_yoy_predictions.csv.gz',compression='gzip',index=False)
summary={'title':'V10 oil-risk-news conditioning benchmark, post hoc','models':modelnames,'dates':FOLDS,'source':'Archival supplied 44,238-row .dta; gpr_oil is monthly country-invariant oil GPR index','estimation':'Each calendar fold refit using past data only; training-only scaling, same-country testing','caveats':['Not physical energy import dependency; not World Bank series','gpr_oil historical availability is retrospective; no vintage-safe real-time claim','post hoc content comparator; no causal or structural mechanism','block bootstrap fixes fitted predictions']}
(out/'V10_oil_comparator_methodology.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
fig,ax=plt.subplots(figsize=(8.8,4.3))
rows=res[(res.target=='YoY')]
xs=np.arange(2);w=.32
for j,(k,lab) in enumerate([('rmse_pct_OilTotal-v-Total','Oil feature vs total'),('rmse_pct_OilRoles-v-OilTotal','Roles on top of oil')]):
 ax.bar(xs+(j-.5)*w,rows[k].to_numpy(),width=w,label=lab)
ax.axhline(0,lw=.8,color='0.35');ax.set_xticks(xs,['Same-month','Lag-only']);ax.set_ylabel('Relative RMSE change (%)');ax.set_title('Exploratory oil-related news vs semantic roles, annual inflation');ax.legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(out/'V10_oil_information_comparison.png',dpi=240);plt.close(fig)
print('\n'.join([f'{r.target}, {r.timing}: roles-v-total {r["rmse_pct_Roles-v-Total"]:.6f}%, oil-v-total {r["rmse_pct_OilTotal-v-Total"]:.6f}%, roles+oil-v-oil {r["rmse_pct_OilRoles-v-OilTotal"]:.6f}%' for _,r in res.iterrows()]))
