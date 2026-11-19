from pathlib import Path
import importlib.util, json, numpy as np, pandas as pd, argparse
BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='UCDP source-driven all-eligible onset screen, NOT original 22 event-selection reconstruction')
parser.add_argument('--panel',default=str(BASE/'data'/'PAPER GEOPOLITICA_.dta'))
parser.add_argument('--ucdp',default=str(BASE/'data'/'UCDPConflictTerminationDataset_v4_2024_Conflict.csv'))
parser.add_argument('--legacy-22',default=str(BASE/'data'/'V6_UCDP_22_original_episode_crosswalk.csv'))
parser.add_argument('--output',default=str(BASE/'tables'))
args=parser.parse_args()
OUT=Path(args.output); OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('v7_ucdp_pipeline',str(Path(__file__).with_name('v7_ucdp_deterministic_pipeline.py')))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.OUT=OUT
panel=mod.panel_prepare(Path(args.panel))
map_one,panel_iso=mod.load_country_maps(panel)
raw=mod.read_csv_flexible(Path(args.ucdp))
raw,episodes,unmapped=mod.build_episode_metadata(raw,map_one,panel_iso)
selected=episodes.loc[episodes.event_date.notna() & episodes.start_precision.notna() & (episodes.start_precision<=3) & episodes.type_of_conflict.isin([2,4]) & episodes.event_date.dt.year.between(2000,2023)].copy()
ss,a,dd,_=mod.build_stacks(panel,episodes,episodes,date_source='threshold',treat_secondary_at_onset=False)
# Exact filter criteria are stated. No original 22 ID list is used for selecting this sample.
a=a.copy();a['episode_id']=a.episode_id.astype(str)
assert a.episode_id.nunique()==ss.episode_id.nunique()
a.to_csv(OUT/'V7_UCDP_transparent_eligibility_final.csv',index=False)
dd.to_csv(OUT/'V7_UCDP_transparent_eligibility_excluded.csv',index=False)
sel=selected[['episode_id','event_date','start_precision','type_of_conflict','location']].copy();sel.episode_id=sel.episode_id.astype(str)
sel['passes_full_window_and_controls']=sel.episode_id.isin(set(a.episode_id))
sel.to_csv(OUT/'V7_UCDP_full_candidate_screen.csv',index=False)
monthly,raw_paths,cum=mod.episode_level_effects(ss)
for t,df in [('V7_UCDP_episode_monthly.csv',monthly),('V7_UCDP_episode_cumulative.csv',cum)]:df.to_csv(OUT/t,index=False)
print('cum-cols',cum.columns.tolist())
vals=cum.loc[cum.H==6,'cumulative_response_pct']
orig=pd.read_csv(args.legacy_22)
original=set(orig['stack'].astype(str))
N_orig_reproduced=len(original&set(a.episode_id)); nnew=len(set(a.episode_id)-original)
report={
'candidate_n_before_coverage_controls':int(len(selected)),
'usable_n_after_full_25month_and_min5_controls':int(a.episode_id.nunique()),
'original_22_overlap_n':N_orig_reproduced,
'additional_episodes_beyond_original_22':nnew,
'dropped_n':len(selected)-len(a),
'stack_observations':len(ss),
'countries_unique':ss.iso3.nunique(),
'treated_country_episodes':int(a.n_treated.sum()),
'cum_0_6_equal_episode_mean_pp':float(vals.mean()),
'cum_0_6_median_pp':float(vals.median()),
'cum_0_6_positive_episode_count':int((vals>0).sum()),
'criterion':'UCDP conflict-level v4-2024; type 2/4; start_date2, start_prec2 <=3; onset 2000-2023; exact 25 monthly CPI observations on each country; >=5 controls; exclude current secondaries and countries in another type-2/4 conflict-year touching the window; drop GAB GBR USA XKX WSM, retain source-confirmed FIN; all eligible, not 22 Figure-selected.'
}
(OUT/'V7_UCDP_transparent_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2))
