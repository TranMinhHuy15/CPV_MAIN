# Validation (một cấu hình duy nhất c2_lr_double, định trước)

Mỗi run chọn epoch bằng earliest max validation mAUC@. Không dùng test.

| Module | Rank | Config | Seed42 mAUC% | Seed43 mAUC% | Seed44 mAUC% | Mean% | SD pp | Chọn |
|---|---:|---|---:|---:|---:|---:|---:|---|
| c7_current_repeat | 1 | c2_lr_double | 87.6281 | 87.4148 | 86.8563 | 87.2998 | 0.3986 | yes |
| r7_last_state | 1 | c2_lr_double | 87.6874 | 87.2844 | 86.7304 | 87.2341 | 0.4805 | yes |
| a7_m1 | 1 | c2_lr_double | 87.7393 | 87.3067 | 86.7422 | 87.2627 | 0.5000 | yes |
| a26_m1 | 1 | c2_lr_double | 87.8504 | 87.3704 | 86.8074 | 87.3427 | 0.5220 | yes |
| u3_uniform | 1 | c2_lr_double | 87.6874 | 87.3807 | 86.7896 | 87.2859 | 0.4563 | yes |
| g3_learned | 1 | c2_lr_double | 87.7793 | 87.3111 | 86.7600 | 87.2835 | 0.5102 | yes |

Hyperparameters và lý do từng cấu hình: chosen_hyperparameters.json; rationale định trước trong protocol.py.
