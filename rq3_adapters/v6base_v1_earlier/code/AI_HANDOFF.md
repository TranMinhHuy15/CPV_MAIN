# RQ3-V6BASE trên Vast RTX4080 — đặc tả dữ liệu, mô hình và quy trình

Tài liệu mô tả **code thực tế trong ZIP** để người chạy hoặc AI hỗ trợ nối đúng dữ liệu. ZIP không chứa video,
checkpoint, cache hay kết quả. Gói được dựng từ v1.1 (module trên H7), chỉ đổi: **base đóng băng là RiskProp V6**,
module thứ ba là **M1+DX**, chỉ **một cấu hình**, so sánh với **V6 cùng seed** và một **đối chứng làm mượt 0 tham số**.

## 1. Nghiên cứu đã chốt

| Mục | Giá trị |
|---|---|
| GPU | Một RTX4080 16GB, GPU0 |
| Base đóng băng | RiskProp V6 (code tác giả, `with_rnn=False`): seed42 `best_mAUC@_epoch_25`, seed43 `epoch_11` (run gốc 3 seed), seed44 **run chạy lại 03/11** (run 44 gốc bị loại vì lỗi resume) |
| Module train | M1; M1+M2; M1+DX — tất cả gắn trên V6 đóng băng |
| Cấu hình | Một: `c2_lr_double` (AdamW lr 2e-4, wd 1e-4, dropout 0.1, width 512), là cấu hình thắng của cả 3 module ở v1.1; 4 cấu hình ở v1.1 chênh ≤0.2pp validation mAUC nên bỏ dò |
| Ngân sách | 3 module × 1 cấu hình × 3 seed = **9 run** × 50 epoch; effective batch 16 (2×8), clip 40, MultiStep 20/40 |
| Chọn epoch | Earliest maximum validation `mAUC@` của từng run (300 video validation) |
| Test | b0 (V6 chấm lại) + 3 module, dùng chung một lượt CNN V6/video, 1.344 video, 28 chỉ số |
| Thống kê | `bootstrap_paired.py`, luật trong `PREREGISTERED_ANALYSIS.md` |

## 2. Cấu trúc và công thức đang thực thi

V6 chấm mỗi snippet: `z_V6 = fc_cls(mean_{T,H,W} F)` (AnticipationHead: avg-pool H,W → mean T → dropout(eval) → fc).
Gói giữ nguyên CNN + fc của V6, bỏ hẳn LSTM của head RQ2 (`del self.rnn`), và cộng phần hiệu chỉnh có chặn vào logit.

```text
u_t = LayerNorm(W_down mean_time(F_t) + b_down)                           # 2048 -> 512
M1+DX:  d_t = [u_t - u_(t-1), u_t - u_(t-3), u_t - u_(t-6)]   (chỉ quá khứ, chặn ở token đầu)
        u_t += eta * (phi(d_t) - phi(0)),  phi = Linear(1536,512) -> ReLU -> Linear(512,512) -> LayerNorm, eta0 = 0.1
h_window = LSTM512(u[t-H+1:t+1]), reset mỗi cửa sổ
e_i = <Wq u_t, Wk h_i>/sqrt(512) - softplus(d)*age_i/25 ;  a = softmax(e) ;  c_H = sum_i a_i h_i
M1, M1+DX: c = c_7          M1+M2: c = sum_H softmax(Linear([c_7,c_13,c_26]))_H c_H
g = sigmoid(Linear(Dropout(c))) (bias0 = -2) ;  raw = Linear(Dropout(c)) (khởi tạo 0)
z = z_V6 + g * tanh(raw)        # bound 1; raw=0 lúc đầu => mô hình bắt đầu ĐÚNG bằng V6
```

| Module | Tham số train | Lịch sử tối đa (token) |
|---|---:|---:|
| M1 | 3,676,675 | 7 |
| M1+M2 | 3,681,286 | 26 |
| M1+DX | 4,727,300 | 13 (cửa sổ 7 + lag 6) |

Tổng mô hình = V6 31.64M (đóng băng) + module 3.7–4.7M ≈ 35.3–36.4M, nhỏ hơn H7+module v1.1 (~69M).
Không future token, nhãn hay thời điểm va chạm đi vào forward. Đối chứng **smooth7**: trung bình nhân quả 7 điểm
cuối của đường rủi ro V6 (cố định trước, không chọn trên test).

## 3. Dữ liệu và tensor (giữ nguyên v1.1)

| Thành phần | Cấu trúc |
|---|---|
| Dataset | Nexar revision `7535d0656dac31d7da2846913bb69c6331d2a70a`; 1.200 train + 300 val (split tác giả); 1.344 test |
| Clip | 55 token = 25 warm-up + 30 target; snippet 5 frame quá khứ, stride gốc 3 |
| Cache/video | FP32 `[55,2048,T+1,1,1]`, cột cuối là mean đúng thứ tự reduction gốc; **trích bằng CNN của V6 từng seed** |
| Teacher logits | FP32 `[30]`/video = `fc_cls` V6 trên 30 token target; cache `[1500,30]` |
| Train view | Một view đã augment cố định/video/seed, dùng chung cho cả 3 module cùng seed |
| Precision | FP32, không autocast/TF32 |

Tiền xử lý dữ liệu V6 và RQ2 dùng cùng `kaggle_adapter.py` (sha `8c8804e9…`), cùng source commit `579376fa…`,
cùng preprocessing `author-native-frame-opencv-jpeg-v2`; preflight kiểm cả ba.

## 4. Cổng an toàn

1. **Preflight** (`protocol.validate_v6_baseline`): đủ 5 file metadata + đúng 1 file `.pth` = checkpoint earliest-max
   validation mAUC; seed/source/data/preprocessing/adapter khớp; không phải run có `rq2` (LSTM); 3 checkpoint khác nhau;
   recipe khoa học giống nhau giữa 3 seed.
2. **V6 validation parity** (`worker.validation_parity`, chạy ngay sau khi trích cache từng seed, TRƯỚC khi train):
   sigmoid(teacher logits) trên 300 video validation phải khớp `validation_predictions.json` của chính run V6 đó
   (300×30 snippet) với max |Δp| ≤ 2e-3. Sai ⇒ dừng. Kết quả ghi `teachers/seed_X/v6_validation_parity.json`.
3. Smoke 24 microbatch/3 update: correction ban đầu = 0, gradient tới LSTM module, tham số đóng băng không đổi.
4. Sau train: trọng số đóng băng trong checkpoint module phải bằng từng bit checkpoint V6.
5. Test: 8 video đầu mỗi path so full pipeline với đường cache (≤1e-4).
6. Report: so V6 chấm lại với `test_predictions.json` gốc của V6 (`v6_rescore_parity.csv`, chẩn đoán).

## 5. Input và cây thư mục trên máy chạy

```text
/workspace/RQ3_V6BASE_v1/           # giải nén ZIP này
/workspace/rq3v6_inputs/
  rq3_selected_source_v1.zip         # sha256 464a7635…
  V6_BASES/seed_42/run/  seed_43/run/  seed_44_rerun/run/
      run_manifest.json training_status.json validation_history.jsonl
      validation_predictions.json test_predictions.json official_test_metrics.json best_mAUC@_epoch_N.pth
/workspace/Nexar_HF/{train,test-public,test-private}/
/workspace/runtime/rq3_checked/      # venv riêng (setup)
/workspace/outputs/rq3_v6base_42_43_44r/
```

`V6_BASES.tar` dựng trên Windows bằng `prep_v6_bases.ps1` (đọc `riskprop_v6_FULL_backup.tar` và
`seed44_rerun2_PACK_20261103_114108.tar`, chỉ lấy đúng file cần, không sửa bản gốc). `inputs.json` dùng khóa
`v6_runs` (42/43/44 → 3 thư mục trên). Trong code, tên trường nội bộ cũ `h7_runs`/`h7_checkpoint` vẫn được dùng
làm đường ống; giá trị là V6.

## 6. Pha chạy (`run.py --config inputs.json all`)

| Pha | Việc | Ước tính RTX4080 |
|---|---|---|
| setup | venv Torch 2.1.0+cu118, mmcv 2.2.0 | ~10 phút |
| extract + teacher | 3 seed × 1.500 video, ngay sau mỗi seed kiểm **V6 parity** | ~2.4 h |
| tune | 9 run × 50 epoch (log mỗi 20 batch) | ~3.5 h |
| freeze | Khóa chọn (1 cấu hình) trước test | giây |
| test | 3 seed × 1.344 video, 4 path chung 1 CNN | ~3–3.5 h |
| report | Bảng per-seed, 28 chỉ số, parity V6 | phút |
| bootstrap | 7 so sánh × 3 seed × 5.000 lần | ~15 phút |

## 7. Output (`comparison/`)

`RESULTS.md`; `per_seed_summary.csv` (9 dòng: module vs V6 cùng seed); `per_seed_all28_metrics_vs_V6.csv` (252 dòng);
`all_models_all28_metrics.csv`; `v6_rescore_parity.csv`; `VALIDATION_SELECTION.md`; `chosen_hyperparameters.json`;
`paired_bootstrap.csv`; `paired_bootstrap_decisions.json` (module vs V6 = quyết định; smooth7 vs V6 = đối chứng;
module vs smooth7 = mô tả). Hoàn tất khi có `TEST_COMPARISON_COMPLETE.json`.

## 8. Bản sửa nhỏ so với v1.1
- `run.py`: log sự kiện cuối đổi khóa `phase` → `completed_phase` (lỗi `TypeError ... multiple values for argument 'phase'` ở v1.1, xảy ra sau khi report đã ghi xong).
- Teacher chạy ngay sau extract từng seed để cổng parity dừng sớm.
