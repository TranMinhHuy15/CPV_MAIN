# RQ3 Ablation V3.2

Đọc `GUI_BAN_CHAY_TIEP.md` để lấy lệnh cho H7 và V6. Tạo output mới cho V3.2 để hash cấu hình và code khớp với lượt chạy.

- Validation history khi chuẩn bị test được công bố bằng file tạm và đổi tên nguyên tử. Bản cắt dở khớp tiền tố của nguồn được lưu riêng trước khi tạo lại.
- File hỗ trợ cache được tạo trong thư mục staging trước khi công bố. Điều kiện chuẩn bị test kiểm cả config nguồn, manifest dữ liệu và hash annotation.
- Trên Linux, worker của launcher tham gia cùng nhóm tiến trình. Launcher xử lý Ctrl+C hoặc SIGTERM và gửi tín hiệu dừng đến cả nhóm.
- Bốn bản vá phục hồi, đường trích cache thay thế và kiểm dung lượng của V3.1 tiếp tục áp dụng. Xem `BAN_VA_V3_1.md` để biết chi tiết.

H7 và V6 đều đạt 58 test CPU. Kiểm xử lý tín hiệu Linux bằng mô phỏng. Đã tái hiện và kiểm hồi phục với hàm xuất cache gốc. Pilot và train CUDA/Nexar cần xác minh trên máy chạy.
