#!/usr/bin/env python3
"""Fail-closed numeric assertions for the V15 submission folder.
Does not refit historical predictors; checks exact source panel and frozen holdout losses.
"""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--root',required=True,help='Root of package containing inputs and analysis');p.add_argument('--out',required=True);a=p.parse_args()
R=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
panel=R/'inputs/PAPER GEOPOLITICA_.dta';holdout=R/'analysis/V7_temporal_predictions.csv.gz'
if not panel.exists() or not holdout.exists():raise FileNotFoundError('Source files required: inputs/PAPER GEOPOLITICA_.dta and analysis/V7_temporal_predictions.csv.gz')
D=pd.read_stata(panel,convert_categoricals=False);assert D.shape[0]==44238 and D.iso3.nunique()==146 and D.mdate.nunique()==303
assert not D.duplicated(['iso3','mdate']).any()
for x,y in [('FIN','GAB'),('GBR','USA')]:
 A=D[D.iso3==x].sort_values('mdate');B=D[D.iso3==y].sort_values('mdate')
 assert len(A)==len(B)==303 and np.array_equal(A.hcpi.to_numpy(),B.hcpi.to_numpy())
P=pd.read_csv(holdout)
q=P[(P.target=='yoy')&(P.timing=='contemporaneous')].copy()
assert len(q)==18945 and q.iso3.nunique()==141 and q.mdate.nunique()==147
result={'input_panel_rows':44238,'input_countries':146,'input_months':303,'heldout_n':len(q),'heldout_countries':int(q.iso3.nunique()),'heldout_months':int(q.mdate.nunique())}
for m in ['AR+global','Total','Roles']:
 result['rmse_'+m]=float(np.sqrt(q['se_'+m].mean()))
 result['mae_'+m]=float(q['ae_'+m].mean())
result['relative_rmse_roles_total_pct']=100*(result['rmse_Roles']/result['rmse_Total']-1)
result['relative_mae_roles_total_pct']=100*(result['mae_Roles']/result['mae_Total']-1)
result['extreme_yoy_abs_gt_100']=int((q.y.abs()>100).sum())
assert abs(result['relative_rmse_roles_total_pct']+0.0086569871382558)<1e-8
assert abs(result['relative_mae_roles_total_pct']-0.1411573890982)<1e-8
assert result['extreme_yoy_abs_gt_100']==77
assert result['rmse_AR+global']<result['rmse_Total']
# Read newly independently checked 2024 national index observations
c=pd.read_csv(out/'V15_OFFICIAL_2024_INDEX_24_CHECKS.csv')
assert len(c)==24 and c.matches_official_to_reported_precision.all()
scale=json.loads((out/'V15_ECONOMIC_SCALE.json').read_text())
assert scale['n']==33780 and abs(scale['model_implied_pp_at_median_intensity_10ppt_direct_share_shift']-.00103217439)<1e-10
result['source_2024_national_index_matches']=24
result['median_log_country_total_ai_gpr']=scale['median_log_total_ai_gpr']
result['illustrative_10ppt_role_share_change_at_median_T_pp']=scale['model_implied_pp_at_median_intensity_10ppt_direct_share_shift']
result['panel_sha256']=digest(panel);result['predictions_sha256']=digest(holdout)
result['energy_interaction_estimated']=False
result['status']='PASSED'
(out/'V15_MASTER_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
