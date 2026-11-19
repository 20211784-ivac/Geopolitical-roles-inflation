#!/usr/bin/env python3
"""Reproducible reanalysis of country-month AI-GPR and inflation panel.

Data: user-supplied PAPER GEOPOLITICA_.dta. No UCDP event file provided.
Methods: country x month fixed-effects within OLS, country x time
multiway cluster CR1; Driscoll-Kraay (12 month) inference; lead-conditioned
local projections of cumulative headline CPI changes. These are conditional
associations, not identified structural causal effects.
"""
from __future__ import annotations
import argparse, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source' / 'PAPER GEOPOLITICA_.dta'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
PALETTE = {'direct':'#1f497d','spill':'#bf5c3b','neutral':'#7e8796','forest':'#25775e'}


def abse(arr, groups, trend=False, time_values=None, tol=3e-10, maxiter=250):
    """Absorb country + month-year FE by alternating orthogonal projections.
    Optionally absorb a country-linear-time trend jointly with country intercepts.
    """
    A = np.asarray(arr, dtype=np.float64).copy()
    if A.ndim == 1: A = A[:,None]
    n = A.shape[0]
    matrices = []
    for g in groups:
        code, names = pd.factorize(np.asarray(g),sort=False)
        assert (code>=0).all()
        G = int(code.max()+1)
        membership = csr_matrix((np.ones(n),(code,np.arange(n))), shape=(G,n))
        counts = np.asarray(membership.sum(axis=1)).ravel()
        matrices.append((code,membership,counts))
    if trend:
        assert len(groups)>=2 and time_values is not None
        country_code,mem_country,cnt_country=matrices[0]
        t = np.asarray(time_values,dtype=np.float64)
        tcenter = t - (mem_country @ t)[country_code]/cnt_country[country_code]
        tcenter/=12.0
        den = np.asarray(mem_country @ (tcenter*tcenter)).ravel()
        den = np.maximum(den,1.e-12)
    for itr in range(maxiter):
        old = A.copy()
        for jj,(code,mem,counts) in enumerate(matrices):
            if trend and jj==0:
                means=(mem @ A)/counts[:,None]
                A -= means[code]
                slopes=(mem @ (A*tcenter[:,None]))/den[:,None]
                A -= slopes[code]*tcenter[:,None]
            else:
                means=(mem @ A)/counts[:,None]
                A -= means[code]
        if np.max(np.abs(A-old)) < tol: break
    else:
        raise RuntimeError('FE not converged')
    return A


def cluster_cov(X,u, grp, dfk=True):
    codes,_ = pd.factorize(np.asarray(grp),sort=False)
    G = codes.max()+1; n,k=X.shape
    S = np.zeros((G,k),dtype=np.float64)
    np.add.at(S,codes, X*u[:,None])
    M=S.T@S
    if dfk and G>1: M *= G/(G-1)*((n-1)/(n-k))
    return M


def dk_cov(X,u,time_idx,bandwidth=12):
    ts,_=pd.factorize(np.asarray(time_idx),sort=True)
    T=ts.max()+1; n,k=X.shape
    S=np.zeros((T,k),dtype=np.float64)
    np.add.at(S,ts,X*u[:,None])
    meat=S.T@S
    for lag in range(1,min(bandwidth,T-1)+1):
        weight=1-lag/(bandwidth+1)
        m=S[lag:].T@S[:-lag]
        meat += weight*(m+m.T)
    meat *= T/(T-1)
    return meat


def fit(df, outcome, regressors, name, trend=False, hac=False):
    cc=[outcome]+regressors+['iso3','mdate']
    d=df[cc].dropna().copy()
    n=len(d); G=d.iso3.nunique(); T=d.mdate.nunique()
    if n < 500 or G < 20: raise ValueError('Insufficient sample '+name)
    raw=d[[outcome]+regressors].to_numpy(dtype=float)
    residualized=abse(raw,[d.iso3.to_numpy(), d.mdate.to_numpy()], trend, d.mdate.to_numpy())
    y=residualized[:,0]; X=residualized[:,1:]
    XX=X.T@X
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    u=y-X@beta
    bread=np.linalg.pinv(XX,rcond=1e-13)
    cell=d.iso3.astype(str)+'__'+d.mdate.astype(str)
    V = bread @(cluster_cov(X,u,d.iso3)+cluster_cov(X,u,d.mdate)-cluster_cov(X,u,cell))@bread
    V=(V+V.T)/2
    se=np.sqrt(np.maximum(np.diag(V),0.0))
    dof=min(G-1,T-1)
    test=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    pval=2*st.t.sf(np.abs(test),dof)
    result={'model':name,'n':n,'G':G,'T':T,'dof':int(dof),'outcome':outcome,'regressors':regressors,
            'coef':dict(zip(regressors,map(float,beta))),'se':dict(zip(regressors,map(float,se))),
            'p':dict(zip(regressors,map(float,pval))), 'v':V.tolist(),
            'r2_within':float(1-np.sum(u*u)/np.sum(y*y)),
            'hac':None}
    if hac:
        VDK = bread @ dk_cov(X,u,d.mdate,bandwidth=12) @ bread
        sedk=np.sqrt(np.maximum(np.diag(VDK),0.0))
        dk_p=2*st.t.sf(np.abs(beta/sedk),T-1)
        result['hac']={'se':dict(zip(regressors,map(float,sedk))),
                       'p':dict(zip(regressors,map(float,dk_p))), 'v':VDK.tolist()}
    return result


def contrast(result, weights, use_dk=False):
    keys=result['regressors']; w=np.array([weights.get(k,0.0) for k in keys]); b=np.array([result['coef'][k] for k in keys]);
    V=np.array(result['hac']['v'] if use_dk else result['v']);
    estimate=float(w@b); SE=float(np.sqrt(max(w@V@w,0.0)))
    df=result['T']-1 if use_dk else result['dof']
    p=float(2*st.t.sf(abs(estimate/SE),df)) if SE>0 else np.nan
    critical=float(st.t.ppf(.975,df))
    return {'coef':estimate,'se':SE,'p':p,'low95':estimate-critical*SE,'high95':estimate+critical*SE}


def lag(d, var, k):
    # source file covers exactly consecutive 303 months for every country
    return d.groupby('iso3',sort=False)[var].shift(k)


def load_and_derive():
    df=pd.read_stata(DATA, convert_categoricals=False).sort_values(['iso3','mdate']).reset_index(drop=True)
    assert len(df)==44238 and df.iso3.nunique()==146
    assert not df.duplicated(['iso3','mdate']).any()
    df['direct'] = df.gpr_ai_init + df.gpr_ai_resp
    df['sum_roles'] = df.direct + df.gpr_ai_spill
    df['all_gap']=(df.gpr_ai_all-df.sum_roles).abs()
    # Log-intensity weighted by source-provided role fraction: 1 common log unit.
    share=np.divide(df.direct,df.sum_roles,out=np.zeros(len(df)),where=df.sum_roles>0)
    df['direct_share']=share
    df['log_total']=np.log1p(df.gpr_ai_all)
    df['wdirect']=df.log_total*share
    df['wspill']=df.log_total*(1-share)
    df.loc[df.sum_roles==0,'wspill']=0
    df['log_direct']=np.log1p(df.direct)
    df['log_spill']=np.log1p(df.gpr_ai_spill)
    for v in ['direct','gpr_ai_spill','gpr_ai_init','gpr_ai_resp']:
        sd=df.groupby('iso3')[v].transform('std')
        avg=df.groupby('iso3')[v].transform('mean')
        df[v+'_z']=(df[v]-avg)/sd
    df['z_direct']=df.direct_z
    df['z_spill']=df.gpr_ai_spill_z
    # Recompute CPI monthly log changes to avoid uncertainty over source units.
    df['pi_m']=100*np.log(df.hcpi/lag(df,'hcpi',1))
    df['pi_food_m']=100*np.log(df.fcpi/lag(df,'fcpi',1))
    df['pi_yoy']=100*np.log(df.hcpi/lag(df,'hcpi',12))
    # Original headline_yoy uses 100*log? test deviations in audit.
    df['pi_yoy_orig']=df.headline_yoy
    for v in ['headline_yoy','pi_m','pi_yoy','food_yoy','z_direct','z_spill','wdirect','wspill','log_direct','log_spill']:
        for j in range(1,13):
            df[v+'_l'+str(j)]=lag(df,v,j)
    for v in ['hcpi','z_direct','z_spill','wdirect','wspill']:
        for j in range(1,13):
            df[v+'_f'+str(j)]=lag(df,v,-j)
    for h in range(13):
        p=df.hcpi if h==0 else df['hcpi_f'+str(h)]
        df['cum_price_h'+str(h)]=100*np.log(p/df.hcpi.groupby(df.iso3).shift(1))
    return df


def audit(df):
    dup_pairs=[]
    pivot=df.pivot(index='mdate',columns='iso3',values='hcpi')
    for i, a in enumerate(pivot.columns):
        for b in pivot.columns[i+1:]:
            x,y=pivot[a],pivot[b]
            overlap=x.notna() & y.notna()
            if overlap.sum()>=150 and x.notna().sum()==y.notna().sum() and (x[overlap].to_numpy()==y[overlap].to_numpy()).all() and (x.notna().to_numpy()==y.notna().to_numpy()).all():
                dup_pairs.append({'countries':[a,b],'observations':int(overlap.sum())})
    # Report both pooled and within-country autocorrelation on the preferred
    # 140-country CPI-clean sample. The within-country statistic is the relevant
    # descriptive diagnostic for a country-month panel; pooled correlations can
    # mix within- and between-country variation.
    dacf=df.loc[~df.iso3.isin(['XKX','WSM','USA','GBR','FIN','GAB'])].copy()
    acf_pooled={}
    acf_within_mean={}
    acf_within_median={}
    acf_within_p25={}
    acf_within_p75={}
    for v in ['wdirect','wspill','headline_yoy','pi_m']:
        pooled=[]
        for k in range(1,13):
            pooled.append(float(dacf[v].corr(dacf[v+'_l'+str(k)])))
        acf_pooled[v]=pooled
        country_corrs=[]
        for iso,g in dacf.groupby('iso3',sort=False):
            row=[]
            for k in range(1,13):
                a=g[v]; b=g[v+'_l'+str(k)]
                ok=a.notna() & b.notna()
                if ok.sum()>=30 and a[ok].std()>0 and b[ok].std()>0:
                    row.append(float(a[ok].corr(b[ok])))
                else:
                    row.append(np.nan)
            country_corrs.append(row)
        arr=np.asarray(country_corrs,dtype=float)
        acf_within_mean[v]=np.nanmean(arr,axis=0).tolist()
        acf_within_median[v]=np.nanmedian(arr,axis=0).tolist()
        acf_within_p25[v]=np.nanquantile(arr,.25,axis=0).tolist()
        acf_within_p75[v]=np.nanquantile(arr,.75,axis=0).tolist()
    acf_diff={}
    for v in ['wdirect','wspill']:
        dv=dacf.groupby('iso3',sort=False)[v].diff()
        vals=[]
        for k in [1,2,3,6,12]:
            corr=[]
            for iso,idx in dacf.groupby('iso3',sort=False).groups.items():
                z=dv.loc[idx]
                lagz=z.shift(k)
                ok=z.notna() & lagz.notna()
                if ok.sum()>=30 and z[ok].std()>0 and lagz[ok].std()>0:
                    corr.append(float(z[ok].corr(lagz[ok])))
            vals.append({'lag':k,'mean':float(np.mean(corr)),'median':float(np.median(corr)),'countries':len(corr)})
        acf_diff[v]=vals
    yoy_log_diff=(df.headline_yoy-df.pi_yoy).abs().dropna()
    eligible=df[['headline_yoy','z_direct','z_spill']+[x+'_l'+str(j) for x in ['headline_yoy','z_direct','z_spill'] for j in [1,2]]].notna().all(axis=1)
    counts=df.loc[eligible].groupby('iso3').size().sort_values()
    vals={'master_observations':len(df),'master_countries':df.iso3.nunique(),'months':df.mdate.nunique(),
          'eligible_original_n_all':int(eligible.sum()),'eligible_original_countries_all':int(df.loc[eligible,'iso3'].nunique()),
          'eligible_original_without_XKX':int((eligible & (df.iso3!='XKX')).sum()),
          'eligible_original_without_XKX_countries':int(df.loc[eligible & (df.iso3!='XKX'),'iso3'].nunique()),
          'excluded_country_observations_XKX':int(counts.get('XKX',0)),
          'max_absolute_role_total_discrepancy':float(df.all_gap.max()),
          'same_hcpi_country_series':dup_pairs,'acf_sample_countries':int(dacf.iso3.nunique()),
          'acf_pooled':acf_pooled,'acf_within_country_mean':acf_within_mean,
          'acf_within_country_median':acf_within_median,'acf_within_country_p25':acf_within_p25,
          'acf_within_country_p75':acf_within_p75,'acf_first_differences':acf_diff,
          'acf_within_country_mean':acf_within_mean,'acf_within_country_median':acf_within_median,
          'acf_within_country_p25':acf_within_p25,'acf_within_country_p75':acf_within_p75,
          'headline_yoy_vs_100log_cpi_max_abs_diff':float(yoy_log_diff.max()),
          'headline_yoy_vs_100log_cpi_median_abs_diff':float(yoy_log_diff.median()),
          'sample_main_sum':int(df.sample_main.sum()),
          'sample_main_countries':int(df.loc[df.sample_main==1,'iso3'].nunique()),
          'source':DATA.name, 'exact_data_received': True, 'ucdp_event_microdata_received':False}
    (OUT/'audit.json').write_text(json.dumps(vals,indent=2,ensure_ascii=False),encoding='utf8')
    return vals


def model_suite(df):
    dref=df.loc[~df.iso3.isin(['XKX','WSM'])].copy() # recover source's common 144-country z sample
    d=dref.loc[~dref.iso3.isin(['USA','GBR','FIN','GAB'])].copy() # primary: exclude audited duplicate CPI pairs
    model=[]; cts=[]
    def calc(outcome,exp,lagY=2,lagX=2,name='model',trend=False,where=None,hac=False,dataset=None):
        ds=d if dataset is None else dataset
        ds=ds if where is None else ds.loc[where(ds)].copy()
        reg=[outcome+'_l'+str(i) for i in range(1,lagY+1)]
        reg+=exp+[v+'_l'+str(i) for i in range(1,lagX+1) for v in exp]
        m=fit(ds,outcome,reg,name,trend=trend,hac=hac)
        model.append(m)
        p0=contrast(m,{exp[0]:1,exp[1]:-1})
        cts.append({'model':name,'contrast':'current_direct_minus_spill',**p0})
        w={exp[0]:1,exp[1]:-1}
        for i in range(1,lagX+1): w.update({exp[0]+'_l'+str(i):1,exp[1]+'_l'+str(i):-1})
        cts.append({'model':name,'contrast':'distributed_lag_sum_direct_minus_spill',**contrast(m,w)})
        if hac:
            cts.append({'model':name,'contrast':'current_direct_minus_spill_DK12',**contrast(m,{exp[0]:1,exp[1]:-1},True)})
        return m
    primary=calc('headline_yoy',['wdirect','wspill'],2,2,'A_primary_equal_log_intensity',hac=True)
    rep=calc('headline_yoy',['z_direct','z_spill'],2,2,'B_replication_country_z',hac=True,dataset=dref)
    calc('headline_yoy',['wdirect','wspill'],2,2,'C_with_duplicate_HCPI_sources',dataset=dref)
    calc('headline_yoy',['wdirect','wspill'],2,2,'D_exclude_COVID_inflation_2020_2022',where=lambda x:~((x.year>=2020)&(x.year<=2022)))
    calc('headline_yoy',['wdirect','wspill'],2,2,'E_country_linear_trends',trend=True)
    calc('headline_yoy',['wdirect','wspill'],2,2,'F_exclude_extreme_inflation_countries',where=lambda x:~x.iso3.isin(x.loc[x.headline_yoy.abs()>100,'iso3'].unique()))
    calc('headline_yoy',['wdirect','wspill'],2,2,'G_long_coverage_at_least_180',where=lambda x:x.iso3.isin(x.loc[x.headline_yoy.notna()].groupby('iso3').size().loc[lambda x:x>=180].index))
    # winsorization on outcome 1-99% uses transformed Y, shift lags on common data
    dw=d.copy();q1,q99=dw.headline_yoy.quantile([.01,.99]);dw['headline_yoy_w']=dw.headline_yoy.clip(q1,q99)
    for i in range(1,3):dw['headline_yoy_w_l'+str(i)]=lag(dw,'headline_yoy_w',i)
    rr=['headline_yoy_w_l1','headline_yoy_w_l2','wdirect','wspill']+[v+'_l'+str(i) for i in (1,2) for v in ['wdirect','wspill']]
    m=fit(dw,'headline_yoy_w',rr,'H_winsorize_outcome_1_99');model.append(m)
    cts.append({'model':m['model'],'contrast':'current_direct_minus_spill',**contrast(m,{'wdirect':1,'wspill':-1})})
    calc('pi_m',['wdirect','wspill'],3,3,'I_monthly_log_inflation')
    calc('pi_m',['z_direct','z_spill'],3,3,'J_monthly_log_inflation_country_z')
    calc('pi_m',['log_direct','log_spill'],3,3,'K_monthly_log_inflation_logroles')
    calc('headline_yoy',['log_direct','log_spill'],2,2,'L_annual_logroles')
    # Reserve 2020+ split: unstable GPR role classification historical comparability
    calc('headline_yoy',['wdirect','wspill'],2,2,'M_pre2020',where=lambda x:x.year<=2019)
    calc('headline_yoy',['wdirect','wspill'],2,2,'N_2020_onwards',where=lambda x:x.year>=2020)
    # Future role-exposure leads placebo: nonzero coefficients signal serial feedback/anticipation,
    # but insignificance cannot validate exogeneity.
    rr=['headline_yoy_l1','headline_yoy_l2','wdirect','wspill','wdirect_l1','wspill_l1','wdirect_l2','wspill_l2',
        'wdirect_f1','wspill_f1','wdirect_f2','wspill_f2']
    pm=fit(d,'headline_yoy',rr,'P_future_exposure_placebo')
    pvars=['wdirect_f1','wspill_f1','wdirect_f2','wspill_f2']
    bp=np.array([pm['coef'][v] for v in pvars])
    indices=[rr.index(v) for v in pvars]
    VV=np.array(pm['v'])[np.ix_(indices,indices)]
    chi2=float(bp@np.linalg.pinv(VV)@bp)
    p_joint=float(st.chi2.sf(chi2,len(pvars)))
    diagnostic={'test':'Future 1-2 month roles jointly zero, conditional on current and 2 lagged role and inflation variables',
                'n':pm['n'],'countries':pm['G'],'chi2':chi2,'df':len(pvars), 'p_joint_asymptotic':p_joint,
                'individual_coefs':{v:{'b':pm['coef'][v],'se':pm['se'][v],'p':pm['p'][v]} for v in pvars},
                'interpretation':'Outcome-feedback/anticipation diagnostic only; cannot prove exogeneity.'}
    (OUT/'placebo.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf8')
    model.append(pm)
    (OUT/'models.json').write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(cts).to_csv(OUT/'contrasts.csv',index=False)
    summary=[]
    for m in model:
        primaryvars=[v for v in ['wdirect','wspill','z_direct','z_spill','log_direct','log_spill'] if v in m['regressors']]
        r={'model':m['model'],'n':m['n'],'countries':m['G'],'months':m['T'],'within_r2':m['r2_within']}
        for v in primaryvars:r[v+'_coef']=m['coef'][v];r[v+'_se']=m['se'][v];r[v+'_p']=m['p'][v]
        ct=[x for x in cts if x['model']==m['model'] and x['contrast']=='current_direct_minus_spill']
        if ct:r['role_premium']=ct[0]['coef'];r['premium_p']=ct[0]['p'];r['premium_low95']=ct[0]['low95'];r['premium_high95']=ct[0]['high95']
        summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'model_summary.csv',index=False)
    return model,cts


def local_projections(df):
    d=df.loc[~df.iso3.isin(['XKX','WSM','USA','GBR','FIN','GAB'])].copy()
    # constant sample across h with complete price paths 0..12 and 3 inflation lags
    core=['cum_price_h'+str(h) for h in range(13)] + ['pi_m_l'+str(j) for j in (1,2,3)]
    base=['wdirect','wspill']+[v+'_l'+str(j) for j in range(1,4) for v in ['wdirect','wspill']]
    allf=[v+'_f'+str(h) for h in range(1,13) for v in ['wdirect','wspill']]
    common=d[core+base+allf].notna().all(axis=1)
    d=d.loc[common].copy()
    print('LP common sample', len(d), 'countries',d.iso3.nunique(),flush=True)
    results=[]
    for h in range(13):
        y='cum_price_h'+str(h)
        reg=['wdirect','wspill'] + [v+'_l'+str(j) for j in range(1,4) for v in ['wdirect','wspill']] + ['pi_m_l'+str(j) for j in (1,2,3)]
        a=fit(d,y,reg,f'LP_standard_h{h}')
        reg_aug=reg+[v+'_f'+str(k) for k in range(1,h+1) for v in ['wdirect','wspill']]
        b=fit(d,y,reg_aug,f'LP_future_exposure_conditioned_h{h}')
        for model,prefix in [(a,'standard_total_persistence'),(b,'future_exposure_conditioned')]:
            for role in ['wdirect','wspill']:
                bcoef=model['coef'][role];se=model['se'][role]
                crit=float(st.t.ppf(.975,model['dof']))
                cb=float(st.t.ppf(1-.05/(2*13),model['dof']))
                results.append({'h':h,'method':prefix,'role':role,'coef':bcoef,'se':se,'p':model['p'][role],
                                'low95':bcoef-crit*se,'high95':bcoef+crit*se,
                                'low95_bonf':bcoef-cb*se,'high95_bonf':bcoef+cb*se,
                                'n':model['n'],'G':model['G']})
        print('h=',h,'std',round(a['coef']['wdirect'],4),'aug',round(b['coef']['wdirect'],4),flush=True)
    R=pd.DataFrame(results)
    R.to_csv(OUT/'local_projections.csv',index=False)
    return R


def figures(audit,models,cts,lp):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(8,3.8))
    for v,label,color in [('wdirect','Direct exposure (common scale)',PALETTE['direct']),('wspill','Spillover exposure (common scale)',PALETTE['spill']),('headline_yoy','Headline YoY inflation',PALETTE['neutral']),('pi_m','Monthly inflation',PALETTE['forest'])]:
        mean=np.asarray(audit['acf_within_country_mean'][v],dtype=float)
        med=np.asarray(audit['acf_within_country_median'][v],dtype=float)
        p25=np.asarray(audit['acf_within_country_p25'][v],dtype=float)
        p75=np.asarray(audit['acf_within_country_p75'][v],dtype=float)
        ax.plot(range(1,13),mean,marker='o',markersize=3,label=label,color=color,lw=1.8)
        ax.fill_between(range(1,13),p25,p75,color=color,alpha=.08,lw=0)
    ax.axhline(0,color='#999',lw=.7);ax.set(xlabel='Lag (months)',ylabel='Mean within-country autocorrelation',xticks=[1,2,3,4,6,9,12],ylim=(-.5,1.03))
    ax.legend(ncol=2,loc='upper right',frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'figure_1_persistence.png',dpi=230,bbox_inches='tight');plt.close(fig)
    df=pd.DataFrame([{**{'model':m['model']},**{q:m['coef'].get(q,np.nan) for q in ['wdirect','wspill','z_direct','z_spill']}} for m in models])
    use=[m for m in models if m['model'] in ['A_primary_equal_log_intensity','C_with_duplicate_HCPI_sources','D_exclude_COVID_inflation_2020_2022','E_country_linear_trends','F_exclude_extreme_inflation_countries','G_long_coverage_at_least_180','H_winsorize_outcome_1_99','M_pre2020','N_2020_onwards']]
    labels=['Baseline: shared scale','Include duplicate CPI','Exclude 2020-2022','Country-specific trends','Exclude hyperinflation countries','Countries >=180 months','Winsorize inflation 1/99','2000-2019 only','2020-2025 only']
    fig,ax=plt.subplots(figsize=(8,5.6))
    for j,m in enumerate(use):
        y=len(use)-1-j
        for offset, v, color in [(-.16,'wdirect',PALETTE['direct']),(.16,'wspill',PALETTE['spill'])]:
            ci=st.t.ppf(.975,m['dof'])*m['se'][v]
            ax.errorbar(m['coef'][v],y+offset,xerr=ci,fmt='o',markersize=4.2,capthick=.9,capsize=2,lw=1.1,color=color)
    ax.axvline(0,color='#6b7280',lw=1,ls=':');ax.set_yticks(range(len(labels)));ax.set_yticklabels(list(reversed(labels)),fontsize=8.4)
    ax.set(xlabel='Coefficient per unit of total log AI-GPR exposure',title='Role-weighted inflation associations: robustness')
    ax.plot([],[],color=PALETTE['direct'],marker='o',lw=0,label='Direct')
    ax.plot([],[],color=PALETTE['spill'],marker='o',lw=0,label='Spillover')
    ax.legend(frameon=False,ncol=2,loc='lower right')
    fig.tight_layout();fig.savefig(OUT/'figure_2_robustness.png',dpi=240,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    a=lp[lp.role=='wdirect']
    for meth,color,lbl in [('standard_total_persistence',PALETTE['direct'],'Standard LP: future exposure evolves'),('future_exposure_conditioned',PALETTE['spill'],'Future-exposure-conditioned LP')]:
        p=a[a.method==meth].sort_values('h')
        ax.plot(p.h,p.coef,marker='o',markersize=3.5,color=color,label=lbl,lw=1.7)
        ax.fill_between(p.h.to_numpy(),p.low95_bonf.to_numpy(),p.high95_bonf.to_numpy(),color=color,alpha=.13)
    ax.axhline(0,ls=':',lw=1,color='#676d77')
    ax.set(xlabel='Months after exposure at t',ylabel='Cumulative CPI log change (× 100)',xticks=[0,1,2,3,4,6,8,10,12],title='Direct role: dynamic conditional associations')
    ax.legend(loc='upper left',frameon=False,fontsize=8)
    fig.text(.5,.01,'Shading: Bonferroni 95% family-wise bands (13 horizons); not causal impulse responses.',ha='center',fontsize=7.7)
    fig.tight_layout(rect=[0,.03,1,1]);fig.savefig(OUT/'figure_3_lp_direct.png',dpi=250,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    p=lp[lp.role=='wspill']
    for meth,color,lbl in [('standard_total_persistence',PALETTE['direct'],'Standard LP'),('future_exposure_conditioned',PALETTE['spill'],'Future-exposure-conditioned LP')]:
        r=p[p.method==meth].sort_values('h')
        ax.plot(r.h,r.coef,marker='o',markersize=3.5,color=color,label=lbl,lw=1.7)
        ax.fill_between(r.h.to_numpy(),r.low95_bonf.to_numpy(),r.high95_bonf.to_numpy(),color=color,alpha=.13)
    ax.axhline(0,ls=':',lw=1,color='#676d77')
    ax.set(xlabel='Months after exposure at t',ylabel='Cumulative CPI log change (× 100)',xticks=[0,1,2,3,4,6,8,10,12],title='Spillover role: dynamic conditional associations')
    ax.legend(loc='upper left',frameon=False,fontsize=8)
    fig.text(.5,.01,'Shading: Bonferroni 95% family-wise bands (13 horizons); not causal impulse responses.',ha='center',fontsize=7.7)
    fig.tight_layout(rect=[0,.03,1,1]);fig.savefig(OUT/'figure_4_lp_spillover.png',dpi=250,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT / 'source' / 'PAPER GEOPOLITICA_.dta'));args=p.parse_args()
    global DATA
    DATA=Path(args.data)
    print('load',DATA,flush=True)
    df=load_and_derive();print('audit',flush=True)
    audit_out=audit(df);print('audit summary', {k:audit_out[k] for k in ['eligible_original_n_all','eligible_original_without_XKX','same_hcpi_country_series','headline_yoy_vs_100log_cpi_median_abs_diff']},flush=True)
    print('models',flush=True); models,contrasts=model_suite(df)
    for mm in models:
        pp=[p for p in contrasts if p['model']==mm['model'] and p['contrast']=='current_direct_minus_spill']
        v1=mm['regressors'][2] if len(mm['regressors'])>2 else ''
        print(mm['model'],'n',mm['n'],'p diff',round(pp[0]['p'],4) if pp else '',flush=True)
    print('LP',flush=True);lp=local_projections(df)
    print('figures',flush=True); figures(audit_out,models,contrasts,lp)
    print('DONE',flush=True)

if __name__=='__main__': main()
