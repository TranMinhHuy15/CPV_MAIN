from pathlib import Path
import csv,gzip,hashlib,io,json,statistics,zipfile
R=Path(__file__).resolve().parents[1]

def js(p):
    if not p.exists() and p.with_name(p.name+'.gz').exists():return json.loads(gzip.decompress(p.with_name(p.name+'.gz').read_bytes()).decode('utf-8-sig'))
    return json.loads(p.read_text(encoding='utf-8-sig'))
def wr(rel,obj):
    p=R/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def csvout(rel,rows):
    p=R/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(rows,model,source):
    out=[]
    for k in rows[0]:
        if all(isinstance(row.get(k),(float,int)) for row in rows):
            a=[float(row[k]) for row in rows];out.append(dict(model=model,metric=k,n=len(a),mean=statistics.mean(a),sample_sd=statistics.stdev(a),source=source))
    return out

solution=list(csv.DictReader((R/'rq1_reference/riskprop_v6/code/metadata_snapshot/solution.csv').open(encoding='utf-8-sig')))
def binary_metrics(labels,scores):
    pairs=sorted(zip(scores,labels),reverse=True);pos=sum(labels);neg=len(labels)-pos
    if not pos or not neg:raise ValueError('Need both classes')
    tp=fp=0;roc=[(0.,0.)];ap=0.;i=0
    while i<len(pairs):
        j=i;old=tp
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:
            tp+=pairs[j][1];fp+=1-pairs[j][1];j+=1
        ap+=(tp-old)/pos*tp/(tp+fp);roc.append((fp/neg,tp/pos));i=j
    def area(limit):
        total=0.
        for (x0,y0),(x1,y1) in zip(roc,roc[1:]):
            if x0>=limit:break
            right=min(x1,limit)
            yr=y1 if x1<=limit else y0+(y1-y0)*(right-x0)/(x1-x0)
            total+=(right-x0)*(y0+yr)/2
        return total
    raw=area(.1)
    return {'AP':ap,'AUC':area(1.),'raw':raw/.1,'std':.5*(1+(raw-.005)/.095)}

def scalar(pred):
    mapping={str(p['id']).zfill(5):float(p['score']) for p in pred}
    assert len(mapping)==1344
    out={};splits={}
    for scope in ('Combined','Public','Private'):
        groups=[]
        for g,lead in enumerate((.5,1.,1.5)):
            selected=[p for p in solution if int(p['group'])==g and (scope=='Combined' or p['Usage']==scope)]
            y=[int(p['target']) for p in selected];s=[mapping[p['id'].zfill(5)] for p in selected]
            groups.append(dict(lead_time_s=lead,**binary_metrics(y,s)))
        v={'mAP':statistics.mean(q['AP'] for q in groups),'mAUC_grouped':statistics.mean(q['AUC'] for q in groups),'mPAUC_raw_divided_by_0.1':statistics.mean(q['raw'] for q in groups),'mPAUC_sklearn_standardized':statistics.mean(q['std'] for q in groups)}
        for q in groups:
            lead=q['lead_time_s'];v[f'AP@{lead:.1f}s']=q['AP'];v[f'AUC_full@{lead:.1f}s']=q['AUC'];v[f'pAUC_raw_div_0.1@{lead:.1f}s']=q['raw']
        splits[scope]=v
        if scope=='Combined':out.update(v)
    return out,splits

rq1=[];rq1runs=[];replay=[]
for model in ('RiskProp V6','TOP V2','AdaLEA V4'):
    values=[]
    for seed in (42,43,44):
        if model=='RiskProp V6':p=R/(f'rq1_reference/riskprop_v6/evidence/seed_{seed}/run/test_predictions.json' if seed!=44 else 'rq1_reference/riskprop_v6_seed44_rerun/evidence/workspace/run/test_predictions.json')
        elif model=='TOP V2':p=R/f'rq1_reference/top_v2/evidence/seed_{seed}/run/test_predictions.json'
        else:p=R/f'rq1_reference/adalea_v4/evidence/seed_{seed}/test_predictions.json'
        pred=js(p);v,splits=scalar(pred);values.append(v)
        report=p.with_name('official_test_metrics.json');original=js(report)
        if 'Public' in original:
            delta=max(abs(original[scope][k]-splits[scope][k]) for scope in ('Public','Private') for k in ['mAP','mAUC_grouped','mPAUC_raw_divided_by_0.1'])
            replay.append(dict(model=model,seed=seed,max_original_scalar_delta=delta))
        for k,val in v.items():rq1runs.append(dict(model=model,seed=seed,metric=k,value=val,source=p.relative_to(R).as_posix()))
    rq1+=stats(values,model,'rq1_reference/summary/RQ1_per_seed.csv')
csvout('rq1_reference/summary/RQ1_table.csv',rq1);csvout('rq1_reference/summary/RQ1_per_seed.csv',rq1runs)

rq2=[];rq2runs=[]
for h in (7,13,26):
    vals=[]
    for seed in (42,43,44):
        p=R/(f'rq2_context_length/h{h}/evidence/H{h}/sources/output/medium_h{h}/seed_{seed}/run/rq2_test_report.json' if h!=26 else f'rq2_context_length/h26/evidence/seed_{seed}/run/rq2_test_report.json')
        x=js(p);v=x['summary_metrics'];vals.append(v)
        for k,value in v.items():rq2runs.append(dict(model=f'H{h}',seed=seed,metric=k,value=value,checkpoint=x['selected_checkpoint'],source=p.relative_to(R).as_posix()))
    rq2+=stats(vals,f'H{h}','rq2_context_length/summary/RQ2_per_seed.csv')
csvout('rq2_context_length/summary/RQ2_table.csv',rq2);csvout('rq2_context_length/summary/RQ2_per_seed.csv',rq2runs)

rq3=[];contrasts=[];rounding=[];verification=[]
for base in ('h7','v6'):
    d=R/'rq3_adapters/v32_matched_init'/base/'summary'
    rows=list(csv.DictReader((d/'all_models_all28_metrics.csv').open(encoding='utf-8-sig')))
    baseline='H7' if base=='h7' else 'V6';base_rows={int(x['seed']):x for x in rows if x['model']==baseline}
    bs=list(csv.DictReader((d/'paired_bootstrap.csv').open(encoding='utf-8-sig')))
    for x in bs:
        target=baseline if x['right']=='b0' else x['right'];metric='mAUC_grouped' if x['metric']=='mAUC' else x['metric']
        left={int(a['seed']):float(a[metric]) for a in rows if a['model']==x['left']};right={int(a['seed']):float(a[metric]) for a in rows if a['model']==target}
        delta=statistics.mean(left[s]-right[s] for s in (42,43,44))*100
        verification.append(dict(base=base,left=x['left'],right=x['right'],metric=metric,error=abs(delta-float(x['delta_pp']))))
        contrasts.append(dict(x,source=(d/'paired_bootstrap.csv').relative_to(R).as_posix()))
    for arm in sorted(set(x['model'] for x in rows)):
        vals=[{k:float(v) for k,v in x.items() if k not in ('model','seed')} for x in rows if x['model']==arm]
        rq3+= [dict(base=base,**x) for x in stats(vals,arm,(d/'all_models_all28_metrics.csv').relative_to(R).as_posix())]
csvout('rq3_adapters/v32_matched_init/summary/RQ3_all28_mean_sd.csv',rq3)
csvout('rq3_adapters/v32_matched_init/summary/RQ3_contrasts.csv',contrasts)
scalar_checks=[]
for base in ('h7','v6'):
    d=R/'rq3_adapters/v32_matched_init'/base
    expected=list(csv.DictReader((d/'summary/all_models_all28_metrics.csv').open(encoding='utf-8-sig')))
    for p in (d/'evidence').rglob('test_predictions.json'):
        parts=p.parts
        if 'runs' not in parts:continue
        arm=parts[parts.index('runs')+1];seed=int(next(x[5:] for x in parts if x.startswith('seed_') and x[5:].isdigit()))
        model=('H7' if base=='h7' else 'V6') if arm=='b0' else arm
        ref=next(x for x in expected if x['model']==model and int(x['seed'])==seed)
        measured,_=scalar(js(p))
        scalar_checks.append(dict(base=base,model=model,seed=seed,metrics=len(measured),max_abs_error=max(abs(value-float(ref[k])) for k,value in measured.items()),source=p.relative_to(R).as_posix()))
checks={'rq1_scalar_replay':replay,'rq3_point_estimate_checks':verification,'rq3_per_video_scalar_replay':scalar_checks,'note':'No archived code executed. Per-video scalar scores replayed independently. Bootstrap confidence intervals separately checked against stored bootstrap draws, not fresh resampling.'}
wr('checks/numeric_verification.json',checks)
for table in (rq1,rq2):
    for row in table:
        if row['metric'] in ('mAP','mAUC_grouped','AP@1.5s'):print(json.dumps(row))
print('RQ3_POINT_MAX_ERROR',max(x['error'] for x in verification))
print('RQ3_SCALAR_REPLAYS',len(scalar_checks),'MAX_ERROR',max(x['max_abs_error'] for x in scalar_checks))
