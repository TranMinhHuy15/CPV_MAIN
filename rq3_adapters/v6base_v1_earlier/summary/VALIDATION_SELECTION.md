# Validation (một cấu hình duy nhất c2_lr_double, định trước)

Mỗi run chọn epoch bằng earliest max validation mAUC@. Không dùng test.

| Module | Rank | Config | Seed42 mAUC% | Seed43 mAUC% | Seed44 mAUC% | Mean% | SD pp | Chọn |
|---|---:|---|---:|---:|---:|---:|---:|---|
| m1 | 1 | c2_lr_double | 87.7185 | 87.2548 | 86.7719 | 87.2484 | 0.4734 | yes |
| m1_m2 | 1 | c2_lr_double | 87.7052 | 87.3644 | 86.7719 | 87.2805 | 0.4723 | yes |
| diff_h14 | 1 | c2_lr_double | 87.6889 | 87.3348 | 86.8681 | 87.2973 | 0.4117 | yes |

Hyperparameters và lý do từng cấu hình: chosen_hyperparameters.json; rationale định trước trong protocol.py.
