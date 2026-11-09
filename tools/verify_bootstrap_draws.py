from pathlib import Path
import csv,io,json,statistics,zipfile
import numpy as np
R=Path(__file__).resolve().parents[1];draws={};checks=[]
for base in ('h7','v6'):
    d=R/'rq3_adapters/v32_matched_init'/base/'summary'
    b=(d/'bootstrap_draws.npz').read_bytes()
    arrays=np.load(io.BytesIO(b),allow_pickle=False);draws[base]={k:arrays[k] for k in arrays.files}
    rows=list(csv.DictReader((d/'paired_bootstrap.csv').open(encoding='utf-8-sig')))
    main=[]
    for x in rows:
        delta=np.mean([arrays[f"draw|{x['left']}|{s}|{x['metric']}"]-arrays[f"draw|{x['right']}|{s}|{x['metric']}"] for s in (42,43,44)],axis=0)
        lo,hi=np.percentile(delta,[2.5,97.5])*100
        pvalue=float(min(1.,2*min(np.mean(delta<=0),np.mean(delta>=0))))
        c=dict(base=base,left=x['left'],right=x['right'],metric=x['metric'],n_draws=len(delta),ci_max_abs_error=max(abs(float(lo)-float(x['ci95_low_pp'])),abs(float(hi)-float(x['ci95_high_pp']))),p_abs_error=abs(pvalue-float(x['p_two_sided'])))
        checks.append(c)
        if x['p_holm']:main.append((pvalue,float(x['p_holm']),c))
    previous=0.;m=len(main)
    for rank,(pvalue,reported,c) in enumerate(sorted(main,key=lambda x:x[0])):
        previous=max(previous,min(1.,(m-rank)*pvalue));c['holm_abs_error']=abs(previous-reported)
cross=[]
p=R/'rq3_adapters/v32_matched_init/cross_base/cross_base_interaction.csv'
for x in csv.DictReader(p.open(encoding='utf-8-sig')):
    arm=x['arm'];metric=x['metric'];delta=[]
    for s in (42,43,44):delta.append((draws['h7'][f'draw|{arm}|{s}|{metric}']-draws['h7'][f'draw|b0|{s}|{metric}'])-(draws['v6'][f'draw|{arm}|{s}|{metric}']-draws['v6'][f'draw|b0|{s}|{metric}']))
    a=np.mean(delta,axis=0);lo,hi=np.percentile(a,[2.5,97.5])*100;pvalue=float(min(1.,2*min(np.mean(a<=0),np.mean(a>=0))))
    cross.append(dict(arm=arm,metric=metric,ci_max_abs_error=max(abs(float(lo)-float(x['ci95_low_pp'])),abs(float(hi)-float(x['ci95_high_pp']))),p_abs_error=abs(pvalue-float(x['p_two_sided']))))
out={'source':'archived bootstrap draws, not regenerated resampling','numpy_version':np.__version__,'contrast_checks':checks,'cross_base_checks':cross}
(R/'checks/bootstrap_draw_reconciliation.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'contrast_intervals':len(checks),'cross_base_intervals':len(cross),'maximum_CI_error':max(x['ci_max_abs_error'] for x in checks+cross),'maximum_p_error':max(x['p_abs_error'] for x in checks+cross),'maximum_Holm_error':max(x.get('holm_abs_error',0) for x in checks)}))
