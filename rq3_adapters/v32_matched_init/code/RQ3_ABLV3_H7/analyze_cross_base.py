"""Exploratory matched-V3 interaction; use complete sealed bootstrap outputs from both bases."""
import argparse
import csv
from pathlib import Path
import numpy as np
import ablation_stats as S
from protocol import save,sha
from study_guard import require_bootstrap,PLAN


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h7',required=True);parser.add_argument('--v6',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();plan_sha=sha(Path(__file__).resolve().parent/PLAN)
    path7,meta7=require_bootstrap(args.h7,'h7',plan_sha);path6,meta6=require_bootstrap(args.v6,'v6',plan_sha)
    for key in ('solution_sha256','labels_groups_sha256','indices_sha256','resamples','seeds','analysis_code_sha256'):
        if meta7[key]!=meta6[key]:raise ValueError('Cross-base analysis identity differs: '+key)
    if meta7['dry_run'] or meta6['dry_run']:raise ValueError('Finish planned5000 draws before cross-base reporting.')
    sha7,ids7,p7,d7=S.load_draws(path7);sha6,ids6,p6,d6=S.load_draws(path6)
    if sha7!=sha6 or ids7!=ids6:raise ValueError('Need exactly the same video resamples.')
    rows=[]
    for arm in S.ARMS6:
        for seed in S.SEEDS:
            if any((role,seed) not in d for role in (arm,'b0') for d in (d7,d6)):
                raise ValueError('Complete both bases/all six arms; do not silently omit a missing arm.')
        for metric in S.METRICS:
            point=[(p7[(arm,s)][metric]-p7[('b0',s)][metric])-(p6[(arm,s)][metric]-p6[('b0',s)][metric]) for s in S.SEEDS]
            draw=np.stack([(d7[(arm,s)][metric]-d7[('b0',s)][metric])-(d6[(arm,s)][metric]-d6[('b0',s)][metric]) for s in S.SEEDS])
            result=S.summarize(point,draw)
            rows.append(dict(arm=arm,metric=metric,kind='exploratory_interaction_uncorrected',
                             **{k:v for k,v in result.items() if k!='per_seed_pp'},
                             **{'seed%d_pp'%s:v for s,v in zip(S.SEEDS,result['per_seed_pp'])},verdict=S.verdict(result)))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'cross_base_interaction.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'cross_base_interaction.json',dict(definition='I=(X-B0)_H7-(X-B0)_V6',analysis_plan_sha256=plan_sha,
         h7_bootstrap_sha256=sha(path7),v6_bootstrap_sha256=sha(path6),indices_sha256=sha7,analysis_code_sha256=meta7['analysis_code_sha256'],
         scope='exploratory uncorrected interaction; fixed three seeds; no causal ceiling claim',rows=rows))
    print('CROSS BASE COMPLETE: '+str(out),flush=True)


if __name__=='__main__':main()
