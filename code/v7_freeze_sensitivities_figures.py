"""Recompute V7 ex-post sensitivity statistics from frozen held-out predictions,
with no refits and no post-hoc claim of formal forecast equivalence."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
T=R/'tables'; F=R/'figures'
P=pd.read_csv(T/'V7_temporal_predictions.csv.gz')
d=P[(P.target=='yoy')&(P.timing=='contemporaneous')].copy()

def relative(g):
 a=np.sqrt(g.se_Roles.mean());b=np.sqrt(g.se_Total.mean())
 return {'n':len(g),'rmse_roles':a,'rmse_total':b,'delta_RMSE_pct':100*(a/b-1),
         'delta_MAE_pct':100*(g.ae_Roles.mean()/g.ae_Total.mean()-1)}
sub={
 'All held-out months':d,
 'Exclude |YoY| > 100% (ex post)':d[d.y.abs()<=100],
 '2013–2019':d[d.year<=2019],
 '2020–2025':d[d.year>=2020],
 '2022–2025':d[d.year>=2022],
 'Exclude 2022–2023 fold':d[d.fold!='2022-2023']
}
rows=[]
for label,x in sub.items():rows.append({'evaluation':label, **relative(x)})
sp=pd.DataFrame(rows);sp.to_csv(T/'V7_contemporaneous_YoY_expost_sensitivity.csv',index=False)
print(sp.to_string(index=False))

summary=pd.read_csv(T/'V7_temporal_performance_summary.csv')
boots=pd.read_csv(T/'V7_temporal_bootstrap_blocks.csv')
fold=pd.read_csv(T/'V7_temporal_performance_by_fold.csv')
print('fold fields',fold.columns.tolist())
# Main manuscript figure 1, per-fold (contemporaneous annual)
f=fold[(fold.target=='yoy')&(fold.timing=='contemporaneous')].pivot(index='fold',columns='model',values='rmse')
f['pct_rmse_roles_vs_total']=100*(f['Roles']/f['Total']-1)
print(f.to_string())
fig,ax=plt.subplots(figsize=(8.0,4.1))
ax.plot(range(len(f)),f['pct_rmse_roles_vs_total'].to_numpy(),marker='o',linestyle='none')
ax.axhline(0,color='gray',linestyle='--',linewidth=.9)
ax.set_xticks(range(len(f)),f.index.tolist(),rotation=15,ha='right')
ax.set_ylabel('Relative RMSE change: roles vs total (%)')
ax.set_title('V7: expanding-origin holdouts, 141 countries')
ax.grid(axis='y',alpha=.25)
fig.tight_layout();fig.savefig(F/'V7_Figure1_Roles_by_Fold.png',dpi=240,bbox_inches='tight');plt.close(fig)

# Main manuscript fig 2, block CI lengths 12 and 24
df=boots[(boots.target=='yoy')&(boots.timing=='contemporaneous')].sort_values('block_months')
fig,ax=plt.subplots(figsize=(7.6,4.2))
x=np.arange(len(df));y=df.rmse_rel_pct_obs.to_numpy();lo=df.rmse_ci_low_pct.to_numpy();hi=df.rmse_ci_high_pct.to_numpy()
ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=5)
ax.axhline(0,color='gray',linestyle='--',linewidth=.9)
ax.set_xticks(x,[f'{int(v)}-month blocks' for v in df.block_months])
ax.set_ylabel('Relative RMSE change (%)')
ax.set_title('V7: calendar-block bootstrap uncertainty (2,000 replicates)')
ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'V7_Figure2_Roles_Block_CIs.png',dpi=240,bbox_inches='tight');plt.close(fig)

# Main manuscript fig 3, four scenarios from V7 summary
lbl=[]; vals=[]
for _,r in summary.iterrows():
 name=('Annual YoY' if r.target=='yoy' else 'Monthly MoM')+' · '+('same-month' if r.timing=='contemporaneous' else 'lag-only')
 lbl.append(name);vals.append(r.pct_rmse_roles_vs_total)
fig,ax=plt.subplots(figsize=(7.9,4.3))
ax.plot(range(4),vals,marker='s',linestyle='none')
ax.axhline(0,color='gray',linestyle='--',linewidth=.9)
ax.set_xticks(range(4),lbl,rotation=15,ha='right')
ax.set_ylabel('Relative RMSE change: roles vs total (%)')
ax.set_title('V7: outcome and news-timing sensitivity')
ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(F/'V7_Figure3_Four_Timing_Scenarios.png',dpi=240,bbox_inches='tight');plt.close(fig)

# UCDP 30 episode effect distribution: without invalid independent-episode CI
cum=pd.read_csv(T/'V7_UCDP_episode_cumulative.csv')
x=cum[cum.H==6].sort_values('cumulative_response_pct')
fig,ax=plt.subplots(figsize=(7.7,6.0))
ax.plot(x.cumulative_response_pct.to_numpy(),np.arange(len(x)),marker='o',linestyle='none')
ax.axvline(0,color='gray',linestyle='--',linewidth=.9)
ax.axvline(x.cumulative_response_pct.mean(),color='gray',linestyle=':',linewidth=.9)
ax.set_yticks(np.arange(len(x)),x.episode_id.astype(str),fontsize=7)
ax.set_xlabel('6-month cumulative treated-minus-control inflation gap (pp)')
ax.set_ylabel('UCDP episode ID (30 episodes)')
ax.set_title('Transparent UCDP screen: heterogeneous descriptive responses')
ax.grid(axis='x',alpha=.20);fig.tight_layout();fig.savefig(F/'V7_Figure4_Transparent_UCDP_30_Dispersion.png',dpi=250,bbox_inches='tight');plt.close(fig)
