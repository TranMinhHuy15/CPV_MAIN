# RiskProp V6 Seed44 Rerun — FIXED v2

## Dùng gói này để làm gì?

Chạy lại seed 44 từ pretrain theo recipe V6 cũ, với các cổng kiểm tra và backup an toàn hơn. Không thay model/loss/sampling/batch/LR/epoch hoặc cách chọn checkpoint. Không bảo đảm tăng mAP.

Gói không chứa video, checkpoint đã train hoặc runtime CUDA. Mã V6 cũ nằm trong legacy_v6/ và giữ nguyên byte; không chỉnh file trong đó. Mọi lệnh sử dụng dưới đây chạy từ thư mục gốc gói mới. verification.json trong legacy_v6 chỉ là lịch sử gói cũ; xem DELIVERY_VERIFICATION.json/TEST_RESULTS.txt được giao cùng ZIP cho kiểm tra bản mới.

## 1. Upload và giải nén

Yêu cầu Linux x86_64, đúng một RTX 4080 và đủ 2.844 MP4 Nexar bản cũ. Dữ liệu, workspace, thư mục gói và runtime phải tách riêng. Đây KHÔNG phải gói Kaggle/T4. Windows chỉ hỗ trợ audit và test CPU.

```bash
cd /workspace
python3 -m zipfile -e RiskProp_V6_Seed44_Rerun_FIXED_v2.zip /workspace
cd /workspace/RiskProp_V6_Seed44_Rerun_FIXED_v2
python3 verify_package.py
python3 run_selftests.py --report /workspace/seed44_cpu_tests.json
```

Nếu dùng một workspace mới:

```bash
export DATA=/workspace/Nexar_HF
export WS=/workspace/riskprop_v6_seed44_rerun_FIXED_v2
export RUNTIME=/workspace/riskprop_v6_vast_runtime
export HOURS=40
bash run_seed44_rerun.sh preflight
bash run_seed44_rerun.sh train
```

Có thể chạy Python trực tiếp nếu không dùng shell wrapper:

```bash
python3 run_seed44.py preflight --data-root /workspace/Nexar_HF --workspace /workspace/riskprop_v6_seed44_rerun_FIXED_v2
python3 run_seed44.py train --data-root /workspace/Nexar_HF --workspace /workspace/riskprop_v6_seed44_rerun_FIXED_v2
```

Mỗi lần gọi có khóa workspace, runtime và package. Chỉ một job được dùng các tài nguyên đó; backup cũng dùng khóa workspace và sẽ từ chối sao lưu trong khi job còn chạy. Khóa tự nhả khi tiến trình thoát; không cần xóa file lock.

Trước setup/GPU, chương trình tự chạy kiểm tra thật trên Linux về deadline, Ctrl+C, SIGTERM, orphan process, lock và snapshot. Test lỗi hoặc bị bỏ qua sẽ chặn GPU. Sau đó vẫn phải qua smoke/mini-validation V6. Những cổng này chưa được xác nhận trên GPU ở máy tác giả bản vá; không bỏ qua nếu chúng thất bại.

HOURS=40 là giới hạn công việc, không phải bảo đảm chạy liền và không dừng tính phí máy. Khoảng 19 giờ chỉ là ước lượng từ lượt cũ; tốc độ thực phụ thuộc máy/runtime/dữ liệu.

## 2. Resume phải được gọi rõ

```bash
bash run_seed44_rerun.sh resume
```

Lệnh dùng WS/run làm resume-root, kiểm snapshot/contract/seed/epoch rồi mới tiếp tục. Không tự chạy mới đè lên attempt đang có. Nếu đã complete 50 epoch thì không train thêm; dùng evaluate. Nếu lần rerun phải resume, ghi rõ điều đó khi báo cáo.

Nếu chết trước epoch đầu và chưa có commit, backup chẩn đoán trước. Sau khi xem hồ sơ có thể dùng:

```bash
python3 recover_uncommitted.py --run "$WS/run"
```

Lệnh chuyển attempt sang run_uncommitted_* cùng cấp, không xóa. Nó từ chối nếu thấy status/snapshot hoặc dấu hiệu epoch muộn, và không dùng cho seed 44 cũ đã hoàn thành.

## 3. Backup đầy đủ trước khi hủy máy

Chạy sau khi job dừng hoặc hoàn tất. OUT phải nằm ngoài WS, cần đủ dung lượng cho bản ZIP toàn workspace (có thể trên 1 GB).

```bash
export OUT=/workspace/seed44_backups
bash backup_seed44_rerun.sh
```

Backup gồm toàn bộ file thông thường trong workspace: best/resume/epoch checkpoints hiện có, RNG, snapshot, log, metadata, source và cấu hình. Không chỉ sao chép hash checkpoint. Không bỏ qua lỗi sao chép. ZIP chỉ được công bố sau khi đọc kiểm lại CRC và SHA256 từng file; có tệp .zip.verified.json bên cạnh.

Không nhúng lại 2.844 raw MP4 hoặc môi trường CUDA đã cài. Link MP4 tại prepared_nexar/train và test được ghi vào manifest thay vì nhúng byte video; cần giữ bộ Nexar gốc riêng. Gói runtime pin sẽ được cài lại trên máy mới. Khi restore, stage prepare xác minh video và tạo link lại.

Nếu chưa có commit hoặc bộ run đã hỏng, mặc định backup chặn để không nhận nhầm là có thể resume. Muốn giữ bằng chứng để chẩn đoán:

```bash
python3 backup_seed44.py --workspace "$WS" --output-dir "$OUT" --diagnostic
```

Manifest ghi rõ diagnostic và committed_artifacts_verified=false nếu không xác minh được commit. Bản chẩn đoán không được coi là backup resume hợp lệ.

Kiểm ZIP backup sau khi tải:

```bash
python3 verify_backup.py --archive /path/to/seed44_full_....zip
```

Khôi phục sang thư mục MỚI, không giải nén đè workspace cũ:

```bash
python3 -m zipfile -e /path/to/seed44_full_....zip /workspace/restored_seed44
export WS=/workspace/restored_seed44/seed_44
bash run_seed44_rerun.sh resume
```

Nếu backup complete 50 epoch, dùng evaluate thay resume. Giữ đường dẫn data/runtime hợp lệ hoặc truyền tham số mới; prepare kiểm lại nội dung video và contract trước khi chạy.

## 4. Đánh giá lại / kiểm prediction cũ

```bash
bash run_seed44_rerun.sh evaluate
python3 audit_seed.py --run "$WS/run" --report /workspace/seed44_audit.json --seal
```

evaluate-only giữ bản prediction/metric trước trong evaluation_before_rerun_* cùng cấp run. File seal chỉ tạo khi đủ artifact, ID và metric tính lại khớp. audit_seed không chạy GPU; seal của nó có inference_rerun=false. Kết quả cũ bị thay file sau seal sẽ mất xác nhận hoàn tất.

safe_queue.py là tùy chọn nếu quản lý ba seed V6. Gói chính chỉ chạy seed44; không cần train lại 42/43. Không gọi trực tiếp script điều phối trong legacy_v6 để tránh bỏ qua các bản vá.

## 5. Quy tắc báo cáo

- Giữ ba seed đầu tiên ở bảng chính; lần rerun seed44 ghi riêng như kiểm tra độ lặp lại.
- Không thay seed44 cũ bằng rerun vì điểm đẹp hơn; không chọn seed theo test.
- Nondeterministic nghĩa là kết quả có thể khác, không bắt buộc phải khác.
- Chạy lại không resume không tự cô lập nguyên nhân resume; muốn kết luận nhân quả cần thí nghiệm cặp có kiểm soát trạng thái và điểm ngắt.
- V6 này là RiskProp with_rnn=False, không phải H7 LSTM/module.

## 6. Giới hạn còn lại

40 test CPU đã đạt trước đóng gói; kiểm artifact seed44 thật được ghi riêng trong evidence/. Chưa chạy lại CUDA/Nexar hoặc test POSIX trên máy này. Ba test dọn cây tiến trình Windows của gói cũ không được dùng để chứng nhận Linux; gói mới chỉ cho train/evaluate trên Linux và bắt buộc kiểm nhánh POSIX tại runtime đích. Mã V6/contract cũ giữ nguyên, không tuyên bố đã sửa hành vi training Windows vốn không được hỗ trợ.

Không khẳng định bitwise resume của persistent DataLoader workers; RNG ở tiến trình chính không chứa RNG từng worker. Snapshot staging giảm nguy cơ bộ file trộn nhưng không thể làm cả bộ nhiều file trở thành một phép ghi nguyên tử duy nhất khi mất điện. Nếu bị ngắt trong restore, dùng lại snapshot gốc. Không có cam kết mọi lỗi tương lai hoặc rủi ro GPU/credit đã được loại bỏ.

## Bổ sung v3 (sau review)
- `safe_entry` verify hash lần 2 sau khi giữ lock (chống race khi khởi động).
- Preflight báo rõ test nào bị skip/fail; skip thường do container Vast hạn chế -> thuê máy khác, không bỏ qua gate.
- `python compare_predictions.py <test_predictions_cu.json> <test_predictions_moi.json>`: so từng video giữa run cũ và run mới/evaluate.
- Backup chỉ chạy khi train đã dừng (workspace đổi trong lúc backup sẽ bị từ chối, đúng thiết kế).
- Run mới nên chạy liên tục (không resume) để loại giả thuyết resume; HOURS=40 chỉ là giới hạn mềm, tự đặt cảnh báo tiền trên Vast.
- Báo cáo cả hai run seed 44; giữ run gốc trong bảng chính, không thay thế.
- Thứ tự gợi ý: `preflight` -> `evaluate` (checkpoint epoch 17, ~30 phút) -> so sánh; chỉ train lại 19h nếu cần.
