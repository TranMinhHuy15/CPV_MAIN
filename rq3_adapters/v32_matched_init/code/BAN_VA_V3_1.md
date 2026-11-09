# RQ3 Ablation V3.1

Dùng thư mục kết quả mới cho V3.1. Tạo config bằng `make_inputs_from_v2.py` hoặc điền `inputs.example.json`. Dùng `GUI_BAN_CHAY_TIEP.md` để xem lệnh cho H7 và V6.

- Preflight có thể thử lại sau khi sửa đường dẫn baseline nếu output chỉ chứa source đang chuẩn bị. Source vẫn phải qua xác minh hash.
- Khi mất điện hoặc tiến trình bị dừng giữa lúc lưu epoch 1, run dở được chuyển sang `run_uncommitted_epoch1_*` và nhánh chạy lại từ cùng seed. Nếu có snapshot hợp lệ, worker phục hồi snapshot trước khi xác định epoch cần chạy tiếp. Snapshot hỏng hoặc mất ở các epoch sau được báo lỗi để khôi phục từ bản sao.
- Pilot xác minh checkpoint ở mốc 2 và 3. Mốc đã lưu hợp lệ có thể dùng lại khi chạy tiếp.
- Diagnostics và log bước train được ghi lại bằng file tạm và đổi tên nguyên tử. Resume kiểm đủ diagnostics của các epoch đã lưu.
- `extract` kiểm toàn bộ cache mượn từ V2. Cache lỗi ở lần xác minh đầu sẽ được trích lại dưới output V3.1, và đường dẫn cùng hash cache được khóa trước khi tạo teacher và train. Nếu cache đã khóa hỏng, khôi phục đúng bản cache từ bản sao trước khi chạy tiếp.
- Có kiểm dung lượng ước tính cho lượt train và kiểm phần trống trước từng epoch. Để tiếp tục sau cảnh báo, mở rộng ổ đĩa đang chứa output. Chạy từng phase `extract` và `pilot` nếu cần đo tài nguyên trước lượt dài.

Gói cần dataset Nexar và checkpoint nền đã kiểm. Lịch chạy là 18 run mỗi nền, 50 epoch mỗi run. Kết quả kiểm bản phát hành ở thư mục `validation`. Pilot CUDA/Nexar trên máy đích là bước xác minh tiếp theo.
