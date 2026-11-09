# H7 + ba module — cấu hình thử nghiệm RQ3 trên T4

Đọc file này trước. Đây là **nhánh mới**, chưa có kết quả CUDA/Nexar thật.
Không cam kết tăng điểm hoặc không bao giờ gặp lỗi runtime. Gói giữ nguyên
gói H7 trong `base_h7/`; chỉ dùng launcher mới ở thư mục ngoài.

## 1. Gói này làm gì?

- Nền: **H7 đã train đủ 50 epoch, ba seed 42/43/44**.
- Đóng băng encoder, LSTM2048 H7 và classifier; toàn bộ giữ eval mode.
- M1: projection 2048→512 + LSTM512 + attention theo hiện tại/độ trễ;
  hệ số phạt độ cũ không âm qua softplus, không dùng khoảng cách tới event.
- M2: gate hợp nhất **H7/H13/H26**. Mỗi cửa sổ chạy riêng từ trạng thái zero,
  dùng chung trọng số; không lấy đoạn cuối hidden H26 rồi gọi là H7 độc lập.
  Đây là ba cửa sổ của nhánh mới dùng encoder H7, không phải ensemble ba
  checkpoint H7/H13/H26 đã train riêng ở RQ2. Gói chỉ cần checkpoint nền H7.
- M3: giữ trục thời gian thật của feature map, biến đổi phi tuyến từng
  sai phân rồi mới gộp. Không dùng trực tiếp công thức mean sai phân cũ.
  Trừ đáp ứng phi(0) để sai phân bằng zero không tự sinh hiệu chỉnh chuyển động.
- Logit cuối = logit H7 + sigmoid(gate) × tanh(residual); gate phụ thuộc context,
  bias −2 nên đầu ra gate khác zero; chỉ zero-init lớp ra residual.
  Không có LayerNorm trên đường nền, không có gate zero thứ hai.
- Không có cấu hình H1, H3 hoặc H5 trong nhánh đề xuất.

**Đây là adapter trên nền H7, không thay trực tiếp LSTM2048 đã train.**
Các cửa sổ có chồng quan sát với nền H7 có chủ đích. Không mô tả chúng là
lịch sử H21 không chồng frame của RQ3 cũ.

## 2. Chuẩn bị dữ liệu và checkpoint

Cần đúng bộ **2.844 MP4** đã dùng: 1.200 train + 300 validation + 1.344 test.
Không đổi split. Các clip test vẫn được index ở prepare nhưng không dùng để
cập nhật trọng số/chọn checkpoint.

Cần ba thư mục `run/` H7 hoàn chỉnh, mỗi thư mục gồm:

```
run_manifest.json
training_status.json
validation_history.jsonl
<best_checkpoint>.pth
resume_latest.pth
rng_epoch50_rank0.pth
safe_resume_snapshot.zip
```

Launcher kiểm seed, history=7, sampling, data revision và commit snapshot.
Checkpoint H1 có shape giống H7 vẫn bị từ chối bằng contract. Không sửa
các thư mục nguồn. Nếu H7 chưa đủ ba seed/50 epoch, hoàn thành nền trước.
Không thể dùng riêng file `.pth` thiếu bằng chứng cấu hình để nhận là nền đã kiểm.

## 3. Cách dùng trên Kaggle/T4 hoặc Linux

Giải nén gói vào thư mục **ghi được**, ví dụ:
`/kaggle/working/RQ3_H7_ThreeModules_T4_v1`.
Trên Vast có thể dùng `/workspace/…`. Đặt Internet ON khi cài runtime.
Máy cần `ffmpeg/ffprobe`, libGL và libglib; launcher báo lỗi rõ nếu thiếu.

Chỉ sửa đường dẫn trong bản sao `configs/inputs.example.json`, lưu thành
`inputs.json`. Không chỉnh `configs/protocol.json` để chạy tiếp một OUT cũ.
Các đường dẫn trong ví dụ là chỗ cần thay, không phải dữ liệu được tải kèm.

**Kaggle:** notebook mẫu `KAGGLE_H7_3MODULES.ipynb` chứa đúng các bước dưới.
Chỉ dùng một GPU; nếu T4×2 thì đặt `CUDA_VISIBLE_DEVICES=0` trước khi khởi tạo
CUDA. Hai T4 không tự cộng thành một bộ nhớ 32 GB.

Tại thư mục gói:

```bash
python bootstrap.py --runtime-root /kaggle/working/rq3_runtime
```

Runtime riêng dùng Python 3.8.5, Torch 2.1.0+cu118, MMEngine 0.10.7,
MMCV 2.2.0 và mã MMAction gốc kèm trong ZIP. Không thay packages của kernel.
Các lệnh tiếp theo dùng Python mà bootstrap in ra, mặc định:

```bash
PY=/kaggle/working/rq3_runtime/venv/bin/python
export CUDA_VISIBLE_DEVICES=0
$PY run_modules.py --inputs inputs.json --mode prepare --hours 2
$PY run_modules.py --inputs inputs.json --mode smoke --hours 2
$PY run_modules.py --inputs inputs.json --mode train --hours 10
```

`--hours` là **ngân sách thời gian cho lần gọi**, không phải ước tính thời gian
hoàn tất. Chọn phù hợp session/quota thực tế. Đến ngân sách, lưu epoch hoàn chỉnh
và dừng; chạy lại cùng lệnh để resume. Không tự chấm test sau khi train.

Smoke thực chạy trên train seed 42, từng arm, **24 microbatch = 3 optimizer
updates**, đúng AdamW/batch/accumulation/clip/FP32 của train thật. Kiểm tra
logit ban đầu bằng nền, LSTM nhận gradient, nhánh học được, nền/BN không đổi,
ghi peak VRAM và thời gian. Receipt steps=0 không mở được train.
Smoke là cổng kỹ thuật, không phải bằng chứng mô hình tăng điểm.

Sau khi tất cả arm × ba seed đủ 50 epoch:

```bash
$PY run_modules.py --inputs inputs.json --mode evaluate --hours 6
$PY analyze.py --inputs inputs.json --draws 2000
```

Evaluate tạo cùng một risk curve cho mỗi clip/model: điểm cuối cho
AP/AUC/pAUC, toàn curve cho index-based mTTA theo evaluator cũ. Không gọi
mTTA này là độ trễ vật lý đã kiểm bằng timestamp. Test không được dùng để chọn H/arm.

## 4. Lịch chạy và đối chứng

Mặc định:

| Arm | M1 | M2 | M3 | Run mới |
|---|---|---|---|---:|
| m1 | Có | Không | Không | 3 |
| m1_m2 | Có | Có | Không | 3 |
| m1_m3 | Có | Không | Có | 3 |
| full | Có | Có | Có | 3 |
| b0 | H7 gốc đóng băng | — | — | 0 train, 3 chấm lại |

Tổng **12 run train mới**, mỗi run 50 epoch. Cấu hình `arms: ["full"]`
chỉ train 3 run, nhưng thiếu ablation cho đóng góp riêng M2/M3. Khóa arms
trước test; thay arms/code/checkpoint phải dùng OUT mới.

M1 là một gói bottleneck+attention+bounded residual; các đối chứng trên không
tách riêng từng thành phần bên trong M1, không cân bằng hoàn toàn số tham số.
Không gọi đây là thí nghiệm đã chứng minh attention tốt hơn last-state.

## 5. Bộ nhớ và thời gian

**Không dùng cache vector 2048 chiều cũ** vì M3 cần chiều T.
Encoder đóng băng nhưng vẫn decode/augment/forward ảnh trong từng epoch.
Forward chia chunk=2, không lưu graph encoder; chỉ nhánh module cần gradient.
Batch=2, accumulation=8; FP32 train/val/test, TF32 tắt, dropout tường minh .1.
Nếu đổi chunk để xử lý OOM, dùng OUT mới và chạy lại smoke; không tự giảm H,
batch hoặc độ phân giải để lách lỗi. Chưa đo VRAM/tốc độ thật trên T4.
Giữ tối thiểu 4 GiB trống trước mỗi run cho việc lưu checkpoint/snapshot;
theo dõi thêm dung lượng của toàn bộ lịch nhiều run.

Đặt ngân sách sau smoke/epoch đầu. Không lấy thời gian adapter .npy cũ để ước
tính nhánh này. Nền đóng băng làm thay đổi phạm vi thích nghi so với end-to-end.

## 6. Kết quả, resume và sao lưu

- `resolved_*_config.py`: cấu hình thật đưa cho Runner.
- `smoke_receipt.json`: kiểm kỹ thuật/peak VRAM của arm.
- `run/validation_history.jsonl`: mỗi epoch, chọn best bằng validation full mAUC@.
- `run/safe_resume_snapshot.zip`: checkpoint+optimizer+scheduler+RNG+contract.
- `run/rq3_h7_test_report.json`: 28 metric theo evaluator đã giữ.
- `run/test_capture/curves/`: prediction gốc từng video.
- `summary.json`: mean±sample SD, từng seed và chênh lệch ghép cặp.
- `paired_intervals.json`: bootstrap video 2.000 lần, phân tầng group×label,
  cùng mẫu cho các model và ba seed cố định.

Khoảng tin cậy điều kiện trên ba seed này, không bao phủ toàn bộ biến động train.
Primary là full−B0 trên AP@1.5s; guardrail mAP −0.01. Mọi kết quả vẫn là khám phá
do test đã được xem ở các vòng trước. Không tự tuyên bố vượt paper hoặc có ý nghĩa
thống kê từ chênh lệch trung bình.

Trước khi session Kaggle hết, lưu **toàn bộ out_root**, không chỉ best `.pth`.
Giữ nguyên package/runtime và các đường dẫn tham chiếu khi resume. Sử dụng cơ chế
Save Version/output của nền tảng hoặc tải snapshot về. Launcher không tự dừng tính
tiền máy thuê, không tự gửi/upload dữ liệu ra dịch vụ ngoài.

## 7. Giới hạn kiểm chứng bản phát hành

Xem `VERIFICATION.json` và log kiểm thử kèm. Test module/integration ở máy soạn
dùng CPU và registry fixture; không coi fixture là GPU/MMCV/Nexar thật.
Preflight CUDA trong `--mode smoke` là bước bắt buộc trước train dài.
