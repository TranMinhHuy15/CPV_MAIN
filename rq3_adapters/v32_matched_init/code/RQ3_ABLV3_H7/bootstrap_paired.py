"""Matched V3 bootstrap: b0 and all six adapters are scored within this study, no prior predictions."""
import argparse
import csv
from pathlib import Path
import numpy as np
import ablation_stats as S
from protocol import ARMS,load,save,sha
from study_guard import require_evaluation,finish_bootstrap,shared_analysis_hashes,VERSION

BASE='h7'
NEW=('b0',)+ARMS
PRIOR={}
SIDE=()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True);parser.add_argument('--solution',required=True)
    parser.add_argument('--resamples',type=int,default=S.RESAMPLES)
    args=parser.parse_args()
    if args.resamples<100:raise ValueError('Need at least100 resamples; default5000 is the analysis plan.')
    root=Path(args.output_root);contract=require_evaluation(root,Path(__file__).resolve().parent)
    if sha(args.solution)!=contract['solution_sha256']:raise ValueError('Solution differs from pinned checked source.')
    ids,labels,groups=S.load_solution(args.solution)
    indices=S.shared_indices(labels,groups,args.resamples)
    counts=S.bootstrap_counts(indices,len(ids))
    points,draws={},{}
    for arm in NEW:
        for seed in S.SEEDS:
            scores=S.load_scores(root/'test/runs'/arm/('seed_%d'%seed)/'run/test_predictions.json',ids)
            points[(arm,seed)],draws[(arm,seed)]=S.role_draws(labels,groups,scores,indices,counts=counts)
            print('BOOTSTRAP arm=%s seed=%d mAP=%.4f%%'%(arm,seed,points[(arm,seed)]['mAP']*100),flush=True)
    metadata=dict(study_version=VERSION,base=BASE,analysis_plan_sha256=contract['analysis_plan']['sha256'],
                  analysis_code_sha256=shared_analysis_hashes(Path(__file__).resolve().parent),
                  study_contract_sha256=sha(root/'study_contract.json'),solution_sha256=sha(args.solution),
                  labels_groups_sha256=S.sha_bytes(np.stack([labels,groups]).astype(np.int32).tobytes()),
                  indices_sha256=S.sha_bytes(indices.tobytes()),resamples=args.resamples,
                  planned_resamples=S.RESAMPLES,dry_run=args.resamples!=S.RESAMPLES,seeds=list(S.SEEDS),
                  uncertainty='shared stratified video bootstrap with three fixed seeds, not seed-population uncertainty')
    out=root/'comparison'
    S.save_draws(out/'bootstrap_draws.npz',ids,indices,points,draws,metadata=metadata)
    rows=[];decisions={}
    plan=[(l,r,q,'main') for l,r,q in S.MAIN]+[(a,'b0','adapter vs same-seed frozen base','secondary') for a in ARMS]
    for left,right,question,kind in plan:
        summaries={}
        for metric in S.METRICS:
            summary=S.summarize([points[(left,s)][metric]-points[(right,s)][metric] for s in S.SEEDS],
                                np.stack([draws[(left,s)][metric]-draws[(right,s)][metric] for s in S.SEEDS]))
            summaries[metric]=summary
            rows.append(dict(base=BASE,left=left,right=right,kind=kind,metric=metric,
                             **{k:v for k,v in summary.items() if k!='per_seed_pp'},
                             **{'seed%d_pp'%s:v for s,v in zip(S.SEEDS,summary['per_seed_pp'])},p_holm=''))
        decisions[left+'_vs_'+right]=dict(left=left,right=right,kind=kind,question=question,metrics=summaries,cross_study=False)
    family=[r for r in rows if r['kind']=='main' and r['metric'] in S.FAMILY_METRICS]
    for row,p in zip(family,S.holm([r['p_two_sided'] for r in family])):
        row['p_holm']=p;decisions[row['left']+'_vs_'+row['right']]['metrics'][row['metric']]['p_holm']=p
    for decision in decisions.values():
        decision['verdict']={m:S.verdict(v,v.get('p_holm')) for m,v in decision['metrics'].items()}
    with (out/'paired_bootstrap.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(out/'paired_bootstrap_decisions.json',dict(metadata=metadata,decisions=decisions,
         rule='PREREGISTERED_ANALYSIS.md',holm_family=[dict(left=r['left'],right=r['right'],metric=r['metric']) for r in family],
         p_value_note='Approximate percentile-tail bootstrap p; zero is a Monte Carlo zero, not an exact probability.'))
    finish_bootstrap(root,metadata)
    print('BOOTSTRAP COMPLETE resamples=%d dry_run=%s'%(args.resamples,metadata['dry_run']),flush=True)


if __name__=='__main__':main()
