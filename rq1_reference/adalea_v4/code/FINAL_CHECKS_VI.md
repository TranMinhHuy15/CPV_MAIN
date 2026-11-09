# Kiểm tra trước khi giao V4 - 24/10/2026

## Đã chạy thực tế ở máy local

- test_adalea.py:16 tests PASS, gồm train/val/scalar-test1344ID x3seed bằng fixture và resume RNG/optimizer/ATTC.
- test_full_eval.py:11 tests PASS, gồm4032 full-curve artifacts giả lập, scalar parity, temporal metric/source arithmetic parity, subset observations, missing onset, causal windows, read-only cache, hash tamper và resume không inference lại.
- test_process_safety.py:4 tests,3 PASS;1 POSIX orphan test SKIP vì local Windows. Launcher chạy lại suite này trên Linux Vast; không ghi skip là passed.
- CLI --help chạy được. Kiểm bổ sung: legacy contract fixture được chấp nhận; partial49epoch bị từ chối trước GPU; checkpoint input không bị sửa.
- GPU local RTX3050 Laptop4GB, torch2.1.0+cu118: synthetic predictor có2optimizer updates/16microbatches, all-empty object mask hữu hạn dưới AMP, validationFP32.
- Full-curve CUDA FP32,54endpoints,feature4096/max5objects: batch8 vs batch1 sai khác tối đa5.960464477539063e-08. Peakallocated khoảng.056GiB cho predictor/feature giả lập, KHÔNG gồm VGG/detector.
- GPU full-curve allocation/prefix check mới cũng qua trên local; extractor không resident ở lượt local này. Trên worker Vast nó chạy sau khi extractor đã resident.
- model.py, runner.py, shared_reference.py, shared_test_metrics.py khớp SHA256 của V3 nguyên bản. Các trường training config khớp, chỉ thêm nhãn protocol/eval_batch_windows.

## Chưa được xác minh thực tế

- Chưa chạy trên RTX4080/Vast của người dùng.
- Chưa chạy full2844video/50epochs x3seed bằng bản V4.
- Chưa load checkpoint V3 thật trong lượt này; chỉ test cấu trúc/contract bằng checkpoint fixture do test tạo.
- ffprobe parser đã test JSON fixture; local không có ffprobe trong PATH để xác nhận end-to-end video clock thật ở lượt này. Launcher bắt buộc ffprobe trước job.
- Không xác nhận tương đương evaluator Table1 hoặc code gốc Suzuki. Onset/event là inferred metadata adapter, nhãn output ghi rõ.
- Không dùng runtime/VRAM predictor giả lập làm dự báo tổng thời gian/chi phí extractor+train+test.

Logs CPU thật trong verification_logs/. LOCAL_VERIFICATION.json ghi phiên bản và thời gian các suite; LOCAL_CUDA_PREDICTOR.json ghi GPU local. Các số fixture không được đưa vào bảng nghiên cứu.

## Kiểm tra chất lượng bàn giao

Accuracy4/5: tests/source hashes có bằng chứng; chưa có real-data/targetGPU proof nên không gọi100%.
Completeness4/5: train3seed và fulltest/mTTA có code, eval-only V3 có guard; chưa chạy thí nghiệm trả phí là ngoài phạm vi bàn giao code.
Clarity4/5: tách metric index10/physical và giữ tên inferred; cần đọc README_V4 thay các ghi chú V3 lịch sử.
Actionability4/5: có lệnh setup/preflight/train/all/evaluate, cap và resume; người dùng phải thay path runtime/data/output cho máy thật.
Conciseness4/5: chi tiết nằm trong tệp, câu trả lời chỉ dẫn ZIP/hướng dẫn.
Trung bình4.0/5. Ưu tiên tiếp: preflight máy thật, kiểm clock và dự toán từ đo thực trước khi duyệt chạy dài. Không thay tests bằng lời hứa tăng điểm.

Phán quyết: giao gói đã kiểm local với giới hạn rõ; chưa chứng nhận targetGPU/realdata. Không tự khởi động trả phí.
