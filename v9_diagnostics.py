#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post hoc, descriptive loss-concentration and influence audit for V7 frozen
141-country retrospective predictions. This code neither refits a model nor
creates/imputes energy observations. Reproduction: python v9_diagnostics.py
--predictions V7_temporal_predictions.csv.gz --out output_directory.
"""
import argparse, json, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--predictions',required=True)
p.add_argument('--out',required=True)
p.add_argument('--B',type=int,default=4000)
p.add_argument('--seed',type=int,default=20261116)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(a.predictions)
assert D.shape[0]==77806, 'Frozen prediction file count differs from provenance inventory'
for mod in ['AR+global','Total','Roles']:
 assert np.allclose(D[f'se_{mod}'],(D.y-D[f'p_{mod}'])**2,rtol=1e-7,atol=1e-7)
 assert np.allclose(D[f'ae_{mod}'],(D.y-D[f'p_{mod}']).abs(),rtol=1e-7,atol=1e-7)
assert D[['target','timing','iso3','mdate']].duplicated().sum()==0
P=D[(D.target=='yoy')&(D.timing=='contemporaneous')].copy().reset_index(drop=True)
assert (len(P),P.iso3.nunique(),P.mdate.nunique())==(18945,141,147)
P['diff_sq']=P.se_Roles-P.se_Total
P['diff_abs']=P.ae_Roles-P.ae_Total

def summarize(frame,name):
 if len(frame)==0:return None
 t=np.sqrt(frame.se_Total.mean());r=np.sqrt(frame.se_Roles.mean())
 ac=frame.ae_Total.mean();ar=frame.ae_Roles.mean()
 return {'sample':name,'N':len(frame),'countries':frame.iso3.nunique(),
  'months':frame.mdate.nunique(),'rmse_total':t,'rmse_roles':r,
  'relative_rmse_change_pct':100*(r/t-1),
  'mae_total':ac,'mae_roles':ar,'relative_mae_change_pct':100*(ar/ac-1),
  'total_loss_diff_sum':frame.diff_sq.sum(),'mean_loss_diff':frame.diff_sq.mean()}

subsets={
 'All frozen holdouts':P,
 'Exclude |YoY| > 100 (ex post)':P[P.y.abs()<=100],
 'Exclude Lebanon (ex post)':P[P.iso3!='LBN'],
 'Exclude Argentina (ex post)':P[P.iso3!='ARG'],
 'Exclude Lebanon and Argentina (ex post)':P[~P.iso3.isin(['LBN','ARG'])],
 'Exclude Lebanon, Argentina, Venezuela (ex post)':P[~P.iso3.isin(['LBN','ARG','VEN'])]
}
metric=pd.DataFrame([summarize(v,k) for k,v in subsets.items()]);metric.to_csv(out/'V9_sample_influence_metrics.csv',index=False,float_format='%.12g')
hi=P[P.y.abs()>100]
assert len(hi)==77 and set(hi.iso3)=={'LBN','ARG','VEN'}
parts=[{'group':'All holdout country-months','N':len(P),'share_of_holdouts_pct':100,'sum_squared_loss_difference':P.diff_sq.sum()},
 {'group':'|annual inflation| <= 100%','N':len(P)-len(hi),'share_of_holdouts_pct':100*(len(P)-len(hi))/len(P),'sum_squared_loss_difference':P.loc[P.y.abs()<=100,'diff_sq'].sum()},
 {'group':'|annual inflation| > 100%','N':len(hi),'share_of_holdouts_pct':100*len(hi)/len(P),'sum_squared_loss_difference':hi.diff_sq.sum()}]
pd.DataFrame(parts).to_csv(out/'V9_loss_concentration.csv',index=False,float_format='%.12g')
country=hi.groupby('iso3',as_index=False).agg(extreme_N=('y','size'),max_abs_yoy=('y',lambda x: x.abs().max()),extreme_diff_sum=('diff_sq','sum'))
country.to_csv(out/'V9_extreme_country_inventory.csv',index=False,float_format='%.12g')
leave=pd.DataFrame([summarize(P[P.iso3!=c],f'omit_{c}')|{'omitted_country':c} for c in sorted(P.iso3.unique())])
leave.to_csv(out/'V9_leave_one_country_out.csv',index=False,float_format='%.12g')
print('LOOCV positive changes (roles worse):', (leave.relative_rmse_change_pct>0).sum(), 'minimum',leave.relative_rmse_change_pct.min(),'maximum',leave.relative_rmse_change_pct.max())

cnt=P.groupby('iso3').agg(n=('iso3','size'),loss_difference_mean=('diff_sq','mean'),loss_difference_sum=('diff_sq','sum')).reset_index()
mon=P.groupby('mdate').agg(n=('mdate','size'),loss_difference_mean=('diff_sq','mean'),loss_difference_sum=('diff_sq','sum')).reset_index()
cnt.to_csv(out/'V9_country_level_loss.csv',index=False)
mon.to_csv(out/'V9_calendar_month_loss.csv',index=False)

# Paired, synchronous calendar-month block resampling of frozen losses;
# no re-estimation or protection against index release-vintage selection.
# A block draw comprises whole calendar months across all available countries.
rng=np.random.default_rng(a.seed)
def block_ci(d,block,B):
 months=np.sort(P.mdate.unique())
 ix=pd.Index(months);G=d.groupby('mdate').agg(sq_total=('se_Total','sum'),sq_roles=('se_Roles','sum'),abs_total=('ae_Total','sum'),abs_roles=('ae_Roles','sum'),n=('mdate','size')).reindex(ix,fill_value=0)
 arr=G[['sq_total','sq_roles','abs_total','abs_roles','n']].to_numpy(float)
 assert np.sum(arr[:,-1])==len(d)
 m=len(ix); blocks=int(np.ceil(m/block))
 gains=[];maes=[]
 for j in range(B):
  starts=rng.integers(0,m-block+1,size=blocks)
  idx=np.concatenate([np.arange(s,s+block) for s in starts])[:m]
  sums=arr[idx].sum(axis=0)
  gains.append(100*(np.sqrt(sums[1]/sums[0])-1))
  maes.append(100*(sums[3]/sums[2]-1))
 point=summarize(d,'tmp')
 return {'relative_rmse_change_pct':point['relative_rmse_change_pct'],
 'bootstrap_95_ci_low_pct':np.percentile(gains,2.5),
 'bootstrap_95_ci_high_pct':np.percentile(gains,97.5),
 'relative_mae_change_pct':point['relative_mae_change_pct'],
 'mae_bootstrap_95_ci_low_pct':np.percentile(maes,2.5),
 'mae_bootstrap_95_ci_high_pct':np.percentile(maes,97.5)}

res=[]
for sample in ['All frozen holdouts','Exclude |YoY| > 100 (ex post)','Exclude Lebanon (ex post)']:
 for b in (12,24):
  row=block_ci(subsets[sample],b,a.B)
  row.update(sample=sample,block_months=b,replications=a.B,seed=a.seed,estimation='frozen predictions / no refit')
  res.append(row)
ci=pd.DataFrame(res);ci.to_csv(out/'V9_calendar_block_CIs.csv',index=False,float_format='%.12g')

# Figure 3: arithmetic source of marginal squared-loss result.
fig,ax=plt.subplots(figsize=(8.6,4.3))
names=['All 18,945 held-out observations','18,868 observations with |YoY| <= 100%','77 observations with |YoY| > 100%']
vals=[parts[i]['sum_squared_loss_difference'] for i in range(3)]
y=np.arange(3)
bars=ax.barh(y,vals,height=.55,color=['#454545','#8b8b8b','#b4b4b4'],edgecolor='#303030',linewidth=.6)
ax.axvline(0,color='#454545',lw=1)
for bar,v in zip(bars,vals):
 ax.text(v+(0.7 if v>=0 else -.7),bar.get_y()+bar.get_height()/2, f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=10)
ax.set_yticks(y,names);ax.invert_yaxis(); ax.set_xlim(-48,38)
ax.set_xlabel('Sum of squared-error differences: Roles minus Total (pp²)')
ax.text(.0,1.04,'Negative = role classification reduces squared prediction loss',transform=ax.transAxes,fontsize=9)
ax.spines[['top','right','left']].set_visible(False)
ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(out/'Figure3_Tail_Loss_Decomposition.png',dpi=190,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.6,4.6))
s=leave.sort_values('relative_rmse_change_pct'); x=np.arange(len(s))
ax.plot(x,s.relative_rmse_change_pct,'o',markersize=2.8,alpha=.65,color='#6b6b6b')
ax.axhline(0,color='#454545',lw=1);ax.axhline(metric.iloc[0].relative_rmse_change_pct,color='#454545',ls='--',lw=.9)
for label in ['LBN','ARG','VEN']:
 df=s[s.omitted_country==label]
 xj=int(s.index.get_loc(df.index[0])) if False else int(np.flatnonzero(s.omitted_country.values==label)[0])
 v=float(df.relative_rmse_change_pct.iloc[0]);ax.scatter([xj],[v],s=26,marker='D',color='#242424');ax.annotate(f'Omit {label}: {v:+.4f}%',(xj,v),xytext=(8,5 if label!='VEN' else -15),textcoords='offset points',fontsize=9)
ax.set_ylim(min(-.025,s.relative_rmse_change_pct.min()-.005),max(.075,s.relative_rmse_change_pct.max()+.005))
ax.set_xlim(-4,len(s)+5);ax.set_xticks([]);ax.set_xlabel('141 leave-one-country-out exclusions, sorted by relative RMSE change')
ax.set_ylabel('Roles versus Total: ΔRMSE (%)')
ax.spines[['right','top']].set_visible(False)
ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(out/'SupplementFigure_Country_Influence.png',dpi=190,bbox_inches='tight');plt.close(fig)

manifest={'source_file':str(Path(a.predictions).name), 'source_sha256':hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest(),
 'filters':'target=yoy, timing=contemporaneous, 141 countries, 147 dates, 18945 held-out rows',
 'count_all':len(P),'count_abs_yoy_gt_100':len(hi),'extreme_countries':country.to_dict(orient='records'),
 'countries_better_mse':int((cnt.loss_difference_mean<0).sum()),'months_better_mse':int((mon.loss_difference_mean<0).sum()),
 'all_role_vs_total_pct':float(metric.iloc[0].relative_rmse_change_pct),
 'bootstrap_seed':a.seed,'bootstrap_B':a.B,'bootstrap_assumption':'calendar-month moving blocks resample all countries in each month, fixed predictions; descriptive not nested-model formal test',
 'energy_interaction_estimated':False,
 'source_vintage_caveat':'the archived AI-GPR extraction timestamp is unknown; data revised during 2026',
 'all_outputs':[z.name for z in out.glob('*')]}
(out/'V9_diagnostic_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(metric.to_string(index=False))
print(ci.to_string(index=False))
print('Extreme country counts: ',country.to_dict(orient='records'))
