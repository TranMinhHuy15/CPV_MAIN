# Test bổ sung RiskProp V6 + TOP, không train lại

Gói này dùng đúng 6 checkpoint đã chọn (seed 42/43/44), giữ nguyên inference
của gói Eval_Once đã kiểm. Chỉ thay phần báo cáo mTTA bằng công thức code tác giả.
Không chỉnh loss, batch train, checkpoint, score scalar gốc, source hoặc precision.
Đây KHÔNG phải cam kết tái tạo chính xác evaluator Bảng 1 chưa công bố.

## Chạy một lệnh trên Vast

Giải nén ZIP code vào `/workspace`. Chuẩn bị hai backup gốc và thư mục chứa video
test đã giải nén. Không cần giải nén toàn bộ backup checkpoint. Giữ mọi thứ Private.

```bash
python3 /workspace/RiskProp_TOP_Test_AuthorConvention/run_author_eval.py \
  --riskprop-backup '/workspace/riskprop_v6_FULL_backup-001.tar' \
  --top-backup '/workspace/TOP-20261012T154637Z-1-001.zip' \
  --data-root '/workspace/Nexar_HF' \
  --output '/workspace/riskprop_top_author_test' \
  --hours 10
```

Điều kiện: Linux x86_64, NVIDIA CUDA GPU, ffprobe (gói ffmpeg), Internet khi cần
cài runtime riêng. Có thể thêm `--runtime-python /duong/dan/venv/bin/python` nếu
vẫn còn đúng môi trường V6. Không tự nâng cấp hoặc cài thư viện vào Python hệ thống.
Gói kiểm Python 3.8.5, Torch 2.1.0+cu118 và các thư viện đã pin trong engine.

Chạy tuần tự: kiểm hash backup/video → dựng runtime → preflight cả 6 checkpoint
→ thu risk curve toàn clip cho 6 × 1.344 video → tính tất cả metric → mean ± SD3.
Nếu một preflight lỗi, gói dừng trước full capture. Không có training/optimizer.

Thêm `--prepare-only` để chỉ kiểm CPU; `--preflight-only` để chỉ kiểm 8 video cho
mỗi checkpoint. Bỏ cờ đó để chạy đầy đủ. Không có cam kết hoàn thành trong 10 giờ.
`--hours` giới hạn tổng một lần chạy; không phải giới hạn mỗi seed.

Nếu hết giờ: giữ toàn bộ output và chạy lại cùng lệnh/cùng thư mục. Curve đã xong
được kiểm rồi bỏ qua, không tính lại GPU. Không trộn GPU/runtime/checkpoint khác.
Ctrl+C/SIGTERM/deadline dừng nhóm tiến trình con. **Không tự tắt instance Vast;
phí máy vẫn tiếp tục cho đến khi bạn chủ động dừng máy.**

## Nếu đã có đủ curves từ gói Eval_Once

Dùng CHÍNH output cũ, không cần dataset và không inference lại:

```bash
python3 /workspace/RiskProp_TOP_Test_AuthorConvention/run_author_eval.py \
  --output '/workspace/riskprop_top_temporal_eval' \
  --runtime-python '/duong/dan/venv/bin/python' \
  --report-only --hours 1
```

Report-only vẫn kiểm hash checkpoint, dữ liệu định danh, provenance và toàn bộ
curve. Nếu thiếu curve, dừng chứ không xuất kết quả hoàn chỉnh giả. Không cần GPU
để tính lại báo cáo. Riêng report-only bắt buộc truyền runtime sẵn có với
NumPy 1.24.4 / scikit-learn 1.3.2; không gọi cài đặt hoặc kiểm CUDA.

## Kết quả

`OUTPUT/author_code_test_v1/comparison.csv`: mean ± sample SD của 3 seed cho mỗi model.
`summary.json`: đủ scalar metric, mTTA, threshold, FAR thực đo, provenance, cảnh báo.
`inferred_index_anchors.json`: mốc onset/event suy luận dùng chung cho hai model.
Không ghi đè báo cáo native-PTS cũ; không đổi các backup/trọng số gốc.
Giá trị mTTA mới nằm ở `temporal_metrics` và cột `mTTA_test_author_index10_inferred_s`
trong bảng tổng hợp. Mục mTTA null của scalar evaluator gốc vẫn được giữ làm provenance.

- mAP, mAUC, partial AUC: đọc score test gốc; Public/Private và Combined đều có.
  Combined tính lại trên toàn bộ 1.344 mẫu, không trung bình hai bảng Public/Private.
- mTTA test: giữ nguyên các phép toán mTTA trong `AnticipationMetric.calculate`:
  threshold theo rank từng negative observation, giữ ties, miss = 0, chia index cho 10.
- Tích hợp mTTA vào test là phần bổ sung của nhóm: source test gốc không xuất mTTA.
  Event = cuối stream + lead; onset trừ khoảng event-alert trong metadata. Đổi sang
  native frame bằng floor và tìm endpoint stride3 đầu tiên >= frame, giống quy tắc
  source. Index event có thể ở ngoài clip; KHÔNG thêm score tương lai hoặc lặp score cuối.
- Positive không có điểm quan sát sau onset được tính zero, không xóa khỏi mẫu số.
  Việc xác định cửa sổ ở đây theo index đã lượng tử hóa kiểu source; có thể lệch
  so với so sánh PTS chính xác. Báo cáo xuất riêng cả hai số đếm để không giấu sai khác.
- Giữ `/10` của source nên số có hậu tố `_s` là giây theo đồng hồ source; với video
  không phải 30 FPS, nó không nhất thiết bằng giây vật lý. Không dùng làm latency.
- Threshold ties có thể khiến FAR thực đo > 0.1; báo rõ, không âm thầm lọc threshold.
  Threshold dùng test negatives để mô tả như code tác giả, không phải calibration
  cho triển khai. FAR này là tỷ lệ negative frame, không phải số báo động/giờ.

**Có thể so sánh RiskProp V6 với TOP bằng cùng quy ước mới; chưa thể khẳng định
mTTA này đồng chuẩn với 0.958 của paper.** Partial AUC cũng xuất cả hai chuẩn hóa,
không chọn chuẩn theo số nào gần paper hơn. `paper_test_equivalence_confirmed=false`
luôn được giữ trong output. Xem PROTOCOL.json cho phần reported/inferred/missing.

Chỉ chạy CLI bên ngoài `run_author_eval.py`, không chạy `engine/run_all.py` hoặc
stage report của engine: chúng được giữ byte-nguyên để tương thích curves cũ.
Đọc LOCAL_VERIFICATION.json để biết phạm vi kiểm tra thực tế, không coi unit test
là bằng chứng đã chạy toàn bộ trên Vast của bạn.
