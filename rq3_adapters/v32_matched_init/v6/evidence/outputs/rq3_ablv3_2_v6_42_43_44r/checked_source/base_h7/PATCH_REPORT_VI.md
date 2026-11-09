# Bản vá H7 — 29/10/2026

1. Phục hồi epoch0: archive giữ bằng chứng, không ghi đè partial; resume có epoch commit vẫn strict.
2. Test gate: restore toàn bộ selected seeds trước khi kiểm complete. `--evaluate-only` độc lập với `--train`; direct train queue cũng tách train/validation tất cả seed rồi mới test.
3. Complete validator dùng chung: epoch/status, history liên tục đúng50epoch, mAUC hữu hạn, checkpoint/RNG/resume/snapshot đủ, CRC khớp.
4. Settings atomic, so full contract trước overwrite; checkpoint không có contract bị từ chối.
5. Cây tiến trình: giữ group/deadline của outer supervisor; Job Objects thay taskkill ở Windows. Kiểm tra Linux POSIX vẫn phải chạy trên host đích.
6. Ngân sách: áp cho hash dữ liệu/backup; tên backup có microsecond tránh ghi đè bản cùng giây.
7. Giữ nguyên toàn bộ model/loss/preprocess/evaluator cùng gói v1. Các fixture là kiểm phần mềm, không phải điểm Nexar.

Chạy: `HOURS=72 bash /workspace/RQ2_Medium_H7_Checked_v1_FIXED/run_auto.sh`.
Output mặc định: `/workspace/rq2_medium_h7_fixed/`.
Chạy tiếp với cùng ZIP FIXED và output để strict resume. Không hot-swap từ v1.

Chưa chạy CNN224/CUDA/MMCV/Nexar thật, live HF download, Linux signals hoặc Bash runtime trong môi trường đóng gói. GPU smoke trên host thuê là bắt buộc. Không tự hạ batch/đổi mô hình khi thiếu VRAM.
