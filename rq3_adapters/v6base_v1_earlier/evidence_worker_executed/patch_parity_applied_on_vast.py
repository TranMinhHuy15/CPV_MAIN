from pathlib import Path
p=Path('worker.py');s=p.read_text()
a="VALIDATION_PARITY_MAX_ABS=2e-3  # probability units; V6 validation was FP32, cuDNN may differ slightly"
b="passed=bool(diff.max()<=VALIDATION_PARITY_MAX_ABS),"
assert a in s and b in s, 'worker.py khong dung ban mong doi'
s=s.replace(a,"VALIDATION_PARITY_MAX_ABS=1e-2   # V6 validation used PyTorch-default cuDNN TF32 convs; this path is strict FP32\nVALIDATION_PARITY_MEAN_ABS=1e-3  # a wrong checkpoint or pipeline gives mean |dp| >> 1e-2")
s=s.replace(b,"passed=bool(diff.max()<=VALIDATION_PARITY_MAX_ABS and diff.mean()<=VALIDATION_PARITY_MEAN_ABS),mean_tolerance=VALIDATION_PARITY_MEAN_ABS,")
p.write_text(s);print('PATCH OK')
