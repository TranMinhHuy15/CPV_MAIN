# RiskProp V6 + TOP: đánh giá bổ sung một lượt, không train lại

Ngày 12/10/2026. Gói này dùng chính sáu checkpoint từ hai backup đã kiểm tra.
Không có optimizer step. Không chỉnh loss, backbone, checkpoint selection hoặc
trọng số. Không trộn metric validation với test. Giữ mọi input/output Private.

## Một lệnh trên Vast

Giải nén gói code vào /workspace. Dataset đã giải nén phải có đủ video test.
Đặt các backup tại các đường dẫn bên dưới, rồi chạy:

```bash
python3 /workspace/RiskProp_TOP_Eval_Once/run_all.py \
  --riskprop-backup '/workspace/riskprop_v6_FULL_backup-001.tar' \
  --top-backup '/workspace/TOP-20261012T154637Z-1-001.zip' \
  --data-root '/workspace/Nexar_HF' \
  --output '/workspace/riskprop_top_temporal_eval' \
  --hours 10
```

Không cần ZIP bảng Excel RiskProp để chạy: số gốc đọc trực tiếp từ backup TAR.
TOP ZIP đã chứa TAR checkpoint. Không cần tự giải nén toàn bộ hai backup.
Nếu video thực tế nằm thư mục khác, chỉ sửa --data-root. Không chọn thư mục có
nhiều bản sao cùng video. Không trỏ output vào code/dataset/backup cũ.

Điều kiện: Linux x86_64, NVIDIA GPU, ffprobe (cài ffmpeg nếu thiếu), Internet nếu
cần thiết lập runtime. Gói tự tạo môi trường Python3.8.5/torch2.1.0/cu118 riêng,
không thay Python hệ thống. Nếu môi trường V6 đã còn, tránh cài lại bằng cách thêm
`--runtime-python /đường/dẫn/venv/bin/python`. Runtime phải đúng các phiên bản pin.
Không tự chuyển AMP/giảm ảnh khi OOM.

Để kiểm toàn bộ dữ liệu/clock/backup trước, không chạy GPU:

```bash
python3 /workspace/RiskProp_TOP_Eval_Once/run_all.py \
  --riskprop-backup '/workspace/riskprop_v6_FULL_backup-001.tar' \
  --top-backup '/workspace/TOP-20261012T154637Z-1-001.zip' \
  --data-root '/workspace/Nexar_HF' \
  --output '/workspace/riskprop_top_temporal_eval' --hours 2 --prepare-only
```

Sau đó dùng lệnh đầy đủ với cùng output. --hours là ngân sách tổng mỗi lần gọi,
gồm cài môi trường và các giai đoạn, không bảo đảm hoàn thành trong 10h. Có thể
dùng tmux trên Vast để tránh mất SSH, nhưng đừng chạy hai job chung output.
Ctrl+C/SIGTERM/deadline dừng nhóm tiến trình con. **Dừng job KHÔNG tắt máy Vast,
phí instance vẫn tiếp tục.** Khi DONE phải tải output và chủ động dừng instance.

## Kaggle

Import notebook RiskProp_TOP_EvaluateOnly_Kaggle.ipynb, chọn GPU, bật Internet
nếu cần cài runtime. Attach Private: gói code ZIP, dataset video, backup RiskProp,
TOP ZIP/TAR. Run All. Notebook có các ô đường dẫn để sửa nếu input bị trùng.
Dùng một GPU để evaluation (GPU còn lại không bắt buộc). Không chạy DDP/trainer.
Khi bị ngắt, Save output, attach toàn bộ output đó ở phiên kế tiếp rồi đặt
RESUME_ROOT. Nếu mount path/mtime thay đổi, code tự re-hash video để xác minh byte
không đổi; clock và curve có thể dùng tiếp. Không nhận GPU/runtime/checkpoint khác.

## Pipeline và rào chắn

1. Kiểm SHA256 gói code và đúng 30 file kết quả/checkpoint được chốt từ backup.
   Chỉ unpack các artifact cho phép, không thực thi source/config từ backup.
2. Kiểm SHA256 1.344 video test. Đọc PTS từng frame bằng ffprobe, kiểm frame count.
   blind_index đưa vào model KHÔNG có target/event/onset/lead-time.
3. Dựng model bằng source commit đã pin, cơ học như gói V6/TOP đã train; load
   state dict strict, FP32 eval/no gradients. Source/backbone/head không đổi.
4. Preflight TẤT CẢ 6 checkpoint trên 8 video chọn theo ID, kiểm:
   score cuối so với test cũ <=1e-4; cached input pixels phải giống tuyệt đối;
   cached/source risk <=1e-5; snippet độc lập, cùng hình dạng batch <=1e-5.
   Batch1/batch10 chỉ báo diagnostic vì CUDA có thể chọn kernel số học khác.
   Chunks căn từ cuối để giữ hình dạng batch của
   cửa sổ source. Bất kỳ seed nào lỗi đều dừng trước lượt dài.
5. Capture toàn clip quan sát với snippet5frame causal, source stride3 end-anchored.
   TOP risk = max20horizons; RiskProp risk = scalar, đúng source test rule.
   Decode/JPEG/resize từng frame unique một lần mỗi video để tránh decode lại
   theo chunk. Cache chỉ ở RAM của video hiện tại, không tạo kho frames lớn.
6. Lưu curve atomic từng video. Chạy lại cùng lệnh/output sẽ kiểm và bỏ qua video
   đã xong. Không nhận curve khác checkpoint/GPU/runtime. Không có fallback âm thầm.
7. Test scalar vẫn lấy predictions gốc bất biến; report thêm replay delta. Không
   thay điểm cũ bằng điểm mới. Drift metric >.001 sẽ dừng để điều tra.
8. Chỉ xuất summary hoàn chỉnh sau đủ 6 x1.344 curves; không Mean3seed từ run thiếu.

## mTTA: phép tính đầy đủ nhưng KHÔNG tuyên bố tương đương Bảng1

Dataset card Nexar ghi clip test kết thúc trước event0.5/1.0/1.5s. Thời gian gốc
event/alert cùng một clock; hiệu event-alert giữ nguyên khi cắt video. Quy ước
chính được CHỌN TRƯỚC khi xem mTTA:

```
cut = video stream start_time + stream duration
event_clip = cut + time_to_accident
onset_clip = event_clip - (time_of_event - time_of_alert)
```

Ranh giới stream trên là suy luận, chưa tác giả xác minh. Báo thêm sensitivity
dùng last_frame_PTS thay cut. Không gán verified=True cho mapping suy ra.
Theo metadata,23/672 positive có onset tại/sau cut; có thể thêm video không có
endpoint sau onset vì lưới stride3. Báo chính xác số không quan sát được ở mỗi
policy. Không extrapolate curve đến collision/đoạn bị cắt, không kéo onset về trước
để làm đẹp TTA.

Lấy cảnh báo đầu tiên SAU onset, TTA=event_clip-t_alarm. Ngưỡng theo negative
FRAME ranks như evaluator công khai, giữ trọng số multiplicity, loại threshold
có frame FAR thực>.1 do ties. Đây là FRAME FAR, không phải video FAR/false alarms
mỗi giờ. Báo thêm negative-video alarm rate để tránh đánh đồng.

Hai policy riêng:
- mTTA_observed_tp_only_s: meanTTA true positives từng admissible threshold, rồi
  mean theo negative-rank measure; threshold không có TP không có TTA, không gán0.
- mTTA_observed_misses_zero_s: misses=0, mean qua toàn672positive như source proxy.

Báo mean recall, boundary sensitivity và flags paper_equivalent=False. Nếu không
có admissible threshold, fail rõ ràng, không tạo metric giả. Partial AUC báo raw
area, raw/0.1 và sklearn; không chọn chuẩn dựa trên số gần paper. Để khẳng định cùng
chuẩn Table1 vẫn cần tác giả xác minh normalization, threshold expectation,
miss policy và clipped-tail treatment. Chạy code này không khắc phục khác biệt
training batch/AMP đã tồn tại của V6.

## Kết quả

- comparison.csv: Mean ± sample SD đủ3seed, AP/AUC theo lead time, các chuẩn partial
  AUC và mTTA quan sát được. Không thay bảng cũ.
- summary.json: trạng thái toàn job và thống kê.
- paper_context.csv: đặt số Bảng1 cạnh số của nhóm để tham khảo; đánh dấu chưa
  xác minh cùng evaluator, KHÔNG tự kết luận thắng/thua paper.
- model/seed_x/evaluation.json: scalar original/replay + temporal/sensitivity.
- model/seed_x/curves/*.json: đầy đủrisk/timestamp/checkpoint/video hashes.
- preflight.json, capture_contract.json, data_summary.json, alignment_inferred.json
  và logs: evidence, timing và các giả định để ghi trong báo cáo.

Nguồn phương pháp: RiskProp CVPR2026 trên CVF; pinned RiskProp commit579376fa;
pinned TOP commit772c6e57; Nexar dataset card/metadata/LICENSE đi kèm references.
Không có chỉ số TOP Nexar được công bố trong Bảng1 RiskProp để gọi là tái lập.
