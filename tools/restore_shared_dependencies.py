from pathlib import Path
import hashlib,shutil
R=Path(__file__).resolve().parents[1]
for h in (7,13,26):
    for vendor in ('riskprop','top'):
        src=R/f'evaluator/rq2-28-v1/engine/references/{vendor}/original_source.zip'
        dst=R/f'rq2_context_length/h{h}/code/rq1_evaluation/engine/references/{vendor}/original_source.zip'
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise RuntimeError(f'Refuse to overwrite differing dependency: {dst}')
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        print(dst.relative_to(R))
