# RQ2 Checked — lịch sử ngắn và dài, terminal, một RTX4080

Bản kiểm tra bổ sung ngày15/10/2026: chỉ đánh dấu evaluate hoàn tất SAU khi đủ JSON, CSV,1344 risk curves và commit-marker có checksum; bắt buộc đủ28metrics. Không thay đổi mô hình/loss/lịch học. Protocol mới và output mặc định riêng `/workspace/rq2_lstm_checked`; không trộn/resume output bản trước. ZIP cũ được giữ nguyên để bảo toàn khả năng tiếp tục nếu đã dùng.

Gói mở rộng có đối chứng, KHÔNG thay thế/sửa kết quả RQ1. Chạy `run_rq2.py`, không chạy các script nội bộ riêng lẻ. Giữ toàn bộ code/data/outputs PRIVATE vì metadata có ground truth test Nexar. Giữ LICENSE dataset và code tác giả.

## Hai nhánh mới

| Nhánh | Một dự đoán được dùng | Mục đích |
|---|---|---|
| short | Một snippet5frame qua LSTM một bước | Đối chứng kiến trúc, cùng số tham số với long |
| long |26snippet chồng lấp, mỗi snippet5frame qua cùng LSTM | Kiểm tra lịch sử dài |

V6 cũ là baseline tham khảo không train lại. So sánh **long với short** để cô lập tác dụng lịch sử; so long với V6 còn bao gồm thay đổi kiến trúc và checkpoint activation. Không gọi LSTM là kiến trúc mới của nhóm.

**Tên0.5s/3s là danh nghĩa theo đồng hồ lấy mẫu V6.** Short có5slot10Hz, khoảng cách frame đầu-cuối12nativeframe (0.4s nếu30FPS). Long có30slot, khoảng cách đầu-cuối87nativeframe (2.9s nếu30FPS). Video khác30FPS có khoảng thời gian vật lý khác. Gói cố ý giữ stride3/nativeJPEG của V6 để tránh đổi đồng thời cách lấy mẫu. Không tuyên bố cửa sổ3s vật lý hay tái lập đúng100% paper.

Mỗi video train vẫn có30 mốc dự đoán/loss giống V6. Cả hai nhánh nhận thêm25snippet warm-up phía trước, tổng55snippet; chỉ30mốc cuối đi vào loss/validation. Nhánh short không dùng feature warm-up trong head. Cả hai có cùng BatchNorm train trên55snippet nên đây không phải giữ nguyên tuyệt đối batch statistics của V6. BN chạy cùng cấu hình giữa hai nhánh mới. Cửa sổ LSTM được reset độc lập ở từng mốc; không dùng future, không giữ hidden-state xuyên video/chunk. Đầu video lặp frame0, không bịa tương lai.

## Giữ nguyên và thay đổi

- Giữ1200train/300val/1344test, source commit579376fa, SlowOnly-R50, khởi tạo Kinetics710 gốc cho TỪNG seed,224px, truebatch2, accumulation8/effective16, AMP forward/lossFP32, SGD0.002, milestones20/40,50epochs, seed42/43/44.
- Loss BCE+1.5FFR+1.1AMC kế thừa đúng code tác giả/AMP adapter V6 trên30mốc. Không đổi trọng số, không đổi quy tắc chọn checkpoint: validation full mAUC@ cao nhất. Không resume weight V6 cũ vào RQ2.
- Thêm LSTM một chiều2048, một layer, và activation checkpoint ở residual blocks trong CẢ HAI nhánh. Khôi phục BN buffers khi recompute để không đếm forward hai lần. Không tự hạ batch/ảnh/precision nếu OOM.
- Chỉ chạy FP32 khi inference. Full test risk curves, mAP/mAUC/pAUC từ score CUỐI video; mTTA từ toàn đường risk curve. Pooled1344 là chính, Public/Private phụ; không lấy trung bình hai bảng để thay pooled.
- Reuse nguyên scalar evaluator và phép tính mTTA AuthorConvention RQ1. Mốc onset/event suy luận; clock chỉ số/10 không luôn là giây vật lý. Partial AUC raw/0.1 và sklearn đều xuất, KHÔNG khẳng định tương đương Bảng1. Nguồn evaluator được đóng băng trong rq1_evaluation/.

## Trên Vast

Linux x86_64, một RTX4080 được expose, Internet; CPU/RAM/SSD đủ. Dataset phải giải nén đủ2844MP4, mỗiID đúng một file. Dùng thư mục riêng code/data/runtime/output; không lồng vào nhau. Cần ít nhất15GiB TRỐNG lúc bắt đầu mỗi run ngoài dataset/runtime; đây chỉ là guard, phải dự phòng thêm cho6run/checkpoint/snapshot. Không xóa checkpoint đang chạy để dọn đĩa.

Upload ZIP và Nexar_HF đã giải nén. Giải nén ZIP vào/workspace. Nếu container chưa có:

```bash
apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 tmux
cd /workspace
python3 -m zipfile -e RQ2_LSTM_RTX4080_Terminal_Checked.zip /workspace
tmux new -s rq2
```

Kiểm tra kỹ thuật hai nhánh seed42 trước (KHÔNG train epoch nào):

```bash
python3 /workspace/RQ2_LSTM_RTX4080_Terminal_Checked/run_rq2.py --data-root /workspace/Nexar_HF --seeds 42 --total-hours 2
```

Controller kiểm tra process-tree cleanup thật trên Linux, hash source/dataset, runtime, CPU tests, rồi smoke16microsteps đúng batch2/55snippet/224px và mini-validation. Smoke dùng weight tạm rồi bỏ, không phải kết quả nghiên cứu. Chỉ khi qua gate mới được train. Đọc output_root/short hoặc long/seed_42/smoke_report.json để xem peakVRAM và thời gian thực; chưa bảo đảmRTX4080 đủ16GB hoặc2h đủ setup.

Pilot seed42 cho CẢ HAI nhánh (2run x50epoch), chỉ xem validation, CHƯA mở test; giới hạn ví dụ12h TỔNG:

```bash
python3 /workspace/RQ2_LSTM_RTX4080_Terminal_Checked/run_rq2.py --data-root /workspace/Nexar_HF --seeds 42 --train --validation-only --total-hours 12
```

Hoặc toàn bộ3seed cho hai nhánh: **6run x50epoch =300epoch mới**, tuần tự, không train6job cùng lúc.24h dưới đây chỉ là cap ví dụ, KHÔNG cam kết chạy xong:

```bash
python3 /workspace/RQ2_LSTM_RTX4080_Terminal_Checked/run_rq2.py --data-root /workspace/Nexar_HF --train --total-hours 24
```

Gói mặc định chọn seed42/43/44 và hai nhánh; CHỈ `--train` mới train dài. `--total-hours` bắt buộc, tính cả setup/hash/smoke/validation/fulltest. Deadline không dừng billing, không tự stop/destroy Vast instance. Mỗi lần chạy lại lệnh bắt đầu budget mới. Backup trước khi tự stop máy thuê.

Ctrl+B,D detach tmux; `tmux attach -t rq2` để xem lại. Ctrl+C trong job yêu cầu dừng cây tiến trình. ĐóngSSH khi dùngtmux không dừng train. Chạy lại cùng lệnh và cùng output để rollback về snapshot epoch hoàn tất, resume optimizer/scaler/scheduler/RNG. Worker RNG khi workers>0 không hứa bitwise-equivalent sau restart. Seed/nhánh hoàn tất bỏ qua; partial chặn việc chạy tiếp nhánh/seed sau. Không sửa code giữa run rồi resume vào cùng output.

## Output và quyết định nghiên cứu

Default `/workspace/rq2_lstm_checked/{short,long}/seed_{42,43,44}/`:

- `smoke_report.json`, `settings.json`, `resolved_train_config.py`, môi trường/runtime freeze.
- `run/run_manifest.json`, `training_status.json`, `validation_history.jsonl`, checkpoint và safe_resume_snapshot.zip.
- `run/test_capture/curves/*.json`:1344 risk curves + timestamp/frame/chứng cứ causal/checkpoint fingerprint.
- `run/rq2_test_report.json`, `rq2_test_metrics.csv`: đủ pooled/split/group scalar metrics và mTTA.
- `run/evaluation_complete.json`: commit cuối cùng, hash của bộ báo cáo; chỉ đủ marker/artifacts/1344curvehash/28metrics mới được tính run hoàn tất và tổng hợp3seed.
- Output root có `summary.json`, `mean_sd.csv`: chỉ tính mean±sampleSD cho một nhánh khi đủ3seed; không lấy1seed làm kết luận chính thức3seed.

Checkpoint luôn chọn trên validation full mAUC@ như V6. RQ2 ưu tiên phân tích low-FAR tại1.5s (raw pAUC/0.1); đối chiếu1.0s, mAP/mAUC/mTTA và chi phí. Không chỉnh hyperparameter theo test. Dùng `--validation-only` cho pilot quyết định có chạy thêm seed hay không. Khi bỏ cờ này, checkpoint đã hoàn tất chỉ được evaluate, không train lại. Train bình thường hoàn tất sẽ tự test để tránh thiếu metric. Không có cơ chế dừng sớm do điểm thấp.

Tăng điểm không được đảm bảo. Kết quả âm vẫn phải giữ và báo cáo. Nghiên cứu3s vật lý cần một thí nghiệm clock/PTS riêng;5s và RQ3/keyframe chưa có trong gói này.

## Kiểm chứng

Đọc LOCAL_VERIFICATION.json để phân biệt unit tests, fixture32px/GPU cục bộ, và các gateRTX4080/full224/Linux CHƯA chạy tại đây. Không coi fixture/synthetic report là điểm nghiên cứu. Trước thuê dài, chạy preflight trên đúng host và kiểm tra đủ dung lượng.
