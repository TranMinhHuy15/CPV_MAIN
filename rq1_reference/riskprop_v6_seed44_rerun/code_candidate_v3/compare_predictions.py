"""Per-video comparison of two test_predictions.json files (old seed44 vs rerun/evaluate). Read-only."""
import argparse,json,math
def load(p):return {str(r['id']):float(r['score']) for r in json.load(open(p,encoding='utf-8'))}
def main():
    a=argparse.ArgumentParser();a.add_argument('old');a.add_argument('new');a.add_argument('--top',type=int,default=15);a=a.parse_args()
    o,n=load(a.old),load(a.new);ids=sorted(set(o)&set(n))
    if set(o)!=set(n):print('WARNING id sets differ: only-old',len(set(o)-set(n)),'only-new',len(set(n)-set(o)))
    d=[(abs(o[i]-n[i]),i) for i in ids];mx=max(d)[0] if d else float('nan')
    mean=sum(x for x,_ in d)/len(d)
    xs=[o[i] for i in ids];ys=[n[i] for i in ids];mx_,my_=sum(xs)/len(xs),sum(ys)/len(ys)
    cov=sum((x-mx_)*(y-my_) for x,y in zip(xs,ys));r=cov/math.sqrt(sum((x-mx_)**2 for x in xs)*sum((y-my_)**2 for y in ys))
    print(f'videos={len(ids)} max|d|={mx:.3e} mean|d|={mean:.3e} pearson={r:.6f}')
    print('IDENTICAL (<=1e-6)' if mx<=1e-6 else 'DIFFERENT: largest gaps')
    for x,i in sorted(d,reverse=True)[:a.top]:print(i,f'old={o[i]:.4f} new={n[i]:.4f} d={x:.4f}')
if __name__=='__main__':main()
