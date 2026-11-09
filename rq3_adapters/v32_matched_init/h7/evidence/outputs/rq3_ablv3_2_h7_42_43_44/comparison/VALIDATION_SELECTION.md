# Validation (một cấu hình duy nhất c2_lr_double, định trước)

Mỗi run chọn epoch bằng earliest max validation mAUC@. Không dùng test.

| Module | Rank | Config | Seed42 mAUC% | Seed43 mAUC% | Seed44 mAUC% | Mean% | SD pp | Chọn |
|---|---:|---|---:|---:|---:|---:|---:|---|
| c7_current_repeat | 1 | c2_lr_double | 86.1274 | 86.0504 | 86.5763 | 86.2514 | 0.2840 | yes |
| r7_last_state | 1 | c2_lr_double | 86.1970 | 85.9363 | 86.6370 | 86.2568 | 0.3542 | yes |
| a7_m1 | 1 | c2_lr_double | 86.2044 | 85.9081 | 86.7111 | 86.2746 | 0.4060 | yes |
| a26_m1 | 1 | c2_lr_double | 86.4385 | 86.6726 | 86.8370 | 86.6494 | 0.2003 | yes |
| u3_uniform | 1 | c2_lr_double | 86.2904 | 86.5496 | 86.7170 | 86.5190 | 0.2150 | yes |
| g3_learned | 1 | c2_lr_double | 86.3081 | 86.6889 | 86.9215 | 86.6395 | 0.3096 | yes |

Hyperparameters và lý do từng cấu hình: chosen_hyperparameters.json; rationale định trước trong protocol.py.
