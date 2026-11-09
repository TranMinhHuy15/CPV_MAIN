# Sửa AdaLEA V3 Fixed — 09/10/2026

## Đã sửa

- Cache NPZ: materialize 4 members mỗi member đúng một lần; warm lookup vector hóa,
  giữ thứ tự/duplicate frames. Không giải nén toàn mảng cho mỗi cached frame nữa.
- Cache validation: index integer/sorted/unique/native-bounded; shape, mask dtype,
  finite feature. Cache hỏng sẽ dừng, không âm thầm tái tính/thay protocol.
- Timing: `perf_counter`; cold probe tạm riêng trên mỗi preflight, không dựa vào cache
  production đã nóng. Đo cold decode/extract/write và warm read riêng. Probe tự dọn.
- VRAM: reset trước nạp pretrained extractor, báo feature peak và total peak.
  Test maximum full-image+5 crops kể cả hai video probe không đủ detections;
  chạy thêm max-crop branch khi predictor/optimizer đã resident.
- Preflight: đếm optimizer.step thật, bắt buộc 16 microbatches /2 updates;
  causal-prefix và finite FP32 evaluation. Không tự đổi AMP/batch/input khi lỗi.
- Dừng job: từ chối NaN/Infinity ở CLI và inherited deadline, đóng log cả khi Popen lỗi;
  môi trường root sạch để luôn sở hữu process group. Test deadline/Ctrl+C/SIGTERM/
  POSIX orphan bắt buộc trên máy thuê; chạy CPU regression trước production worker.
- Fail-fast Linux x86_64, libGL/libglib, đúng một RTX4080 trước cài runtime.
- Disk gate trước extraction: conservative còn thiếu + reserve cho atomic rewrite;
  credit cache đã có lúc resume, không yêu cầu lại toàn bộ 64GiB free mỗi lần.
- Fixed protocol/output và checksum mới, không nhận checkpoint V3 cũ.

## Giữ nguyên

`model.py`, `runner.py`, `data.py`, shared sampler/metrics, annotation, solution,
pretrained feature method và giá trị cấu hình khoa học không đổi byte/value
so với V3; chỉ protocol ID trong config đổi. Predictor/loss không đổi.
Batch2acc8 effective16, seeds42/43/44,50epochs, optimizer/LR và selection giữ nguyên.

## Đã kiểm và giới hạn

Xem `verification.json`: CPU fixtures, real MP4 decode, cache lạnh/nóng, checkpoint
resume,3seed fixture/test1344 IDs, pooled evaluator, process cleanup và CUDA predictor
synthetic. Không coi fixtures là điểm benchmark hoặc proof RTX4080 readiness.
Pretrained extractor/max-crops/full224 Nexar trên RTX4080 và POSIX cleanup chưa
chạy local; launcher phải kiểm ở máy thuê trước train dài.

Không biến implementation này thành code Suzuki gốc. VGG ImageNet/COCO detector
và các giả định vẫn khai báo trong PROTOCOL_VI. Test mTTA vẫn null vì chưa có
evaluator/time alignment Table1 được xác minh; không lấy validation mTTA thay test.
Runtime estimate không bảo đảm đủ 3seed trong ngân sách. Dừng code không dừng billing.
