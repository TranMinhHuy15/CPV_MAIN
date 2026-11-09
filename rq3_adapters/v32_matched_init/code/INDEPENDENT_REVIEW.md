> Ghi nhận lịch sử của V3. Đối với bản đang giao, xem BAN_VA_V3_1.md và validation/V31_VERIFICATION.json.

# Review độc lập V3

Reviewer `code_reviewer` chỉ đọc đã kiểm diff thực tế so snapshotV2, các caller của init/backend/runner/diagnostics/statistics, helper/hướng dẫn và builder. Không sửa file, cài runtime hay chạy GPU.

Phát hiện đã xử lý và kiểm lại:

- Epoch của sample validation lệch một. Sửa về `runner.epoch`, thêm test mẫu epoch1.
- Diagnostics ghi sau commit có thể thiếu row nếu ngắt đúng cửa sổ đó. Tách hook priority80, ghi/fsync trước sourcecommit90; namedlog/plannedstop ở100; thêm test ngắt sau commit, trim row chưa commit.
- Cross-base thiếu ràng buộc code phân tích. Bind SHA các file phân tích chung vào bootstrap và kiểm lại khi gộp hai nền.
- Worker cần kiểm cfg/prepared đúng contract; thêm guard và kiểm lại source đã pin. Fixture kiểm việc đổi retention sau preflight bị từ chối.
- Builder còn metadata V2 và lệnh cross-base dùng hostPython. Cập nhật 6arm/18run/no prior và dùng runtimePython.

Re-review cuối: không còn finding cần sửa trong phạm vi đã đọc. Đối chiếu 46/46 testCPU mỗi nền, init chung/isolatedRNG, source pin và legacy/equations giữ nguyên byte.

Approval giới hạn ở code/fixture/đóng gói. Chưa chạy GPU end-to-end, pilot thật hoặc chứng minh resume CUDA bit-exact.
