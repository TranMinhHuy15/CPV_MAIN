# Sửa TOP V2 Fixed — 09/10/2026

## Đã sửa

- Supervisor kiểm finite explicit/inherited deadline trước khi tạo process;
  NaN/Infinity không vô hiệu hóa time cap được nữa.
- Đóng stdout pipe khi reader hoàn tất; đóng log nếu Popen thất bại.
- SHA256 code/data tĩnh toàn gói được kiểm trước queue/single-run setup/train;
  báo lỗi thay vì chạy file đã bị sửa. Runtime reports nằm ngoài static manifest.
- Thêm regression tests nonfinite deadlines, pipe closure; giữ deadline/Ctrl+C/
  SIGTERM/orphan/lock/snapshot tests sẵn có.
- Fixed protocol, output/runtime mới để tránh trộn checkpoint TOP V2 cũ.

## Giữ nguyên

Source TOP commit772c6e57, model/head/loss/20horizons/BCEpos_weight10;
V6 annotation/sampler/augmentation/evaluator; SGD/AMP/batch2acc8 effective16;
50epochs ×seeds42/43/44; selection validation fullmAUC. Các file model adapter,
worker, metrics và profile giữ nguyên byte so với TOP V2.

## Kiểm tra và giới hạn

Xem verification.json: original-model forward/backward và loss parity, AMP adapter,
31 unit tests (1 POSIX skip trên Windows), real Runner train/val/atomic snapshot/
resume/test fixture với32px,random init,2epochs. Không phải benchmark scores.
Production gate trên Linux RTX4080 vẫn bắt buộc: original-pretraining smoke224,
16microsteps/2 optimizer updates thật,FP32 mini-validation. Nếu thất bại sẽ không train.

Không thêm test mTTA giả; scalar-test V6 chưa đủ Table1 evaluator equivalence.
TOP không có published Nexar reference score trong RiskProp Table1.
Không bảo đảm 150epochs xong trong24h. Dừng code không dừng billing Vast.
