# Các lỗi đã xử lý

| Vấn đề đã kiểm | Cách xử lý |
|---|---|
| Backup thiếu RNG/snapshot/checkpoint | Backup ZIP toàn workspace, kiểm từng SHA256 và CRC sau ghi |
| cp giấu lỗi / ls tên không tồn tại | Bỏ logic cp/ls; Python liệt kê file thật, lỗi thì không công bố ZIP |
| Mất checkpoint vẫn nhận complete | Gate kiểm best/resume/RNG/history/snapshot và hash contract |
| Thiếu prediction vẫn nhận evaluated | Gate đủ 1.344 test ID, 300 validation ID, score hữu hạn, submission, metric và seal |
| Settings bị ghi quá sớm | Kiểm contract trước khi ghi; JSON file tạm riêng + flush/fsync + replace |
| Restore file bị trộn khi đang giải nén | Stage đủ thành viên sau CRC, công bố status cuối; giữ snapshot nguồn |
| Chết trước commit đầu bị kẹt metadata | Công cụ lưu nguyên attempt chưa commit, có guard từ chối run muộn |
| Hai job ghi cùng workspace | Khóa từ trước setup đến hết evaluate; backup dùng cùng khóa |
| Evaluate ghi mất prediction cũ | Copy và hash bản trước khi đánh giá lại |
| Test phụ thuộc thư mục ngoài ZIP | Suite CPU mới tự chứa fixture/reference và test backup roundtrip |
| Queue chạy test cũ làm thay báo cáo pin hash | Chuyển preflight queue sang entry mới, báo cáo nằm ngoài legacy |
| Cờ resume không rõ ràng | Các mode preflight/train/resume/evaluate tách rõ; không tự thay run cũ |

Giữ nguyên 57 file V6 gốc, bao gồm chín file được run_manifest pin SHA256. Bản vá là lớp vận hành ngoài legacy; receipt ghi hash wrapper riêng. Không chỉnh trọng số, thuật toán, augmentation, split, seed, recipe hay evaluator để tăng điểm.

Ba ca Windows process-tree và ca test tham chiếu ngoài của gói cũ là phát hiện audit, không được đổi nhãn thành pass. Production Linux có gate POSIX bắt buộc; kiểm Windows production không được hỗ trợ. Bộ CPU mới 40 ca độc lập không thay cho gate Linux/GPU.

Đọc START_HERE_VI.md; kiểm báo cáo giao hàng để biết phạm vi đã chạy thật.
