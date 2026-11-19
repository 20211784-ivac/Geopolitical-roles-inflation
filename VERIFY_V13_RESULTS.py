#!/usr/bin/env python3
"""Reproducibility assertions for the full Q1 V13 horizon analysis."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--dir',required=True);args=p.parse_args()
a=Path(args.dir)
R=pd.read_csv(a/'V13_horizon_performance.csv');P=pd.read_csv(a/'V13_horizon_predictions.csv.gz')
B=pd.read_csv(a/'V13_country_calendar_bootstrap.csv');Q=pd.read_csv(a/'V13_horizon_outlier_sensitivity.csv')
assert len(R)==6 and len(B)==12 and len(Q)==6
assert set(R.horizon)=={1,3,6} and set(R.ar_lags)=={2,12}
assert all(R.countries==141) and all(R.months==147)
assert len(P)==int(R.N.sum())
assert (P.mdate-P.estimation_cutoff_mdate>=P.horizon).all(), 'Target data leak detected'
assert not P.duplicated(['horizon','ar_lags','iso3','mdate']).any()
for _,r in R.iterrows():
 d=P[(P.horizon==r.horizon)&(P.ar_lags==r.ar_lags)]
 assert len(d)==r.N
 for m in ['M0_ARglobal','M1_Total','M2_Roles']:
  assert np.allclose(d[f'se_{m}'],(d.y-d[f'p_{m}'])**2,atol=1e-9)
  assert abs(np.sqrt(d[f'se_{m}'].mean())-r[f'rmse_{m}'])<1e-8
  assert abs(d[f'ae_{m}'].mean()-r[f'mae_{m}'])<1e-8
assert abs(float(R[(R.horizon==1)&(R.ar_lags==2)].delta_rmse_roles_total_pct.iloc[0])-(-.017422))<.0001
assert all((B.rmse_ci_low<0)&(B.rmse_ci_high>0))
assert all(R.delta_mae_roles_total_pct>0) and all(Q.pct_rmse_roles_total_nonextreme>0)
assert len(set(P.fold))==5
status={'status':'PASSED','scenarios':6,'observations_across_scans':int(R.N.sum()),'unique_fold_names':sorted(P.fold.unique().tolist()),'all_12_country_calendar_RMSE_intervals_include_zero':True,'all_6_mae_deltas_unfavorable':True,'all_6_nonextreme_rmse_deltas_unfavorable':True,'h1_ar2_matches_archived_v7_lag_only':True,'energy_interaction_estimated':False}
(a/'V13_verification.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
