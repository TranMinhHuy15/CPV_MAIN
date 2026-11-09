# TOP V2 Fixed — cấu hình chung với RiskProp V6

Đọc `FIXES_VI.md`: sửa finite-deadline, đóng pipe sau child exit, kiểm SHA256
toàn bộ code/data tĩnh trước setup/train và tăng test hồi quy.
Dùng output Fixed mới; không resume output TOP V2 cũ. Model/loss/protocol khoa học không đổi.

Đây là TOP từ repo chính thức, áp dụng protocol V6 cho benchmark Nexar. Không phải tái lập cấu hình huấn luyện gốc TOP hay bảo đảm khớp điểm RiskProp PDF.

## Cấu hình

- Một RTX 4080, Linux x86_64; runtime riêng Python 3.8.5, Torch 2.1+cu118, MMEngine 0.10.7, MMCV 2.2 full.
- SlowOnly-R50, pretraining Kinetics710; 30 snippet × 5 frame × 224px.
- Cùng annotation, sampler và augmentation V6; 1.200 train / 300 validation / 1.344 test; nominal FPS=30, stride=3 như V6, không tuyên bố resample vật lý 10Hz.
- Batch thật 2 × accumulation 8 = effective 16; AMP train, loss FP32, eval FP32/batch1, workers2.
- SGD 0.002, momentum0.9, weight_decay0.0001, gradclip40; milestones20/40, gamma0.1; 50 epoch/seed, seeds42/43/44 nối tiếp, mỗi seed từ pretraining.
- Chọn best theo full validation `mAUC@`; `mAUC@0.1` là standardized sklearn partial AUC.
- Giữ kiến trúc/loss TOP: decoder20 horizons, annotation labels, BCE pos_weight10. Không thêm FFR/AMC.

## Terminal

Upload ZIP và Nexar_HF đã giải nén đủ 2.844 MP4, không nhân đôi ID. Không chạy đồng thời với V6/AdaLEA trên cùng GPU.

```bash
cd /workspace
python3 -m zipfile -e TOP_V2_V6Matched_RTX4080_3Seeds_TerminalSafe_Fixed.zip /workspace
tmux new -s top_v2
python3 /workspace/TOP_V2_V6Matched_RTX4080_3Seeds_TerminalSafe_Fixed/run_three_seeds.py --data-root /workspace/Nexar_HF --total-hours 1
```

Lệnh trên chỉ preflight. Sau khi qua kiểm tra CPU, CUDA runtime, original pretraining, smoke224px và mini-validation:

```bash
python3 /workspace/TOP_V2_V6Matched_RTX4080_3Seeds_TerminalSafe_Fixed/run_three_seeds.py --data-root /workspace/Nexar_HF --train --hours-per-seed 24 --total-hours 24
```

`--total-hours` là budget TỔNG cho cả hàng đợi, gồm setup/train/validation/test, không phải mỗi seed; thay theo ngân sách. Không bảo đảm đủ 150 epoch trong 24h. Smoke16 microsteps phải có hai optimizer updates thật qua hai chu kỳ accumulation8; AMP skip sẽ chặn train. Không tự giảm batch/input/precision khi lỗi.

Ctrl+B,D để detach tmux; `tmux attach -t top_v2` để xem lại; Ctrl+C yêu cầu dừng cây tiến trình. Backup TOÀN BỘ output khi dừng rồi chạy lại cùng lệnh để resume. Không nhận checkpoint V1/V6/V7. **Code dừng không ngừng billing Vast**: bạn phải backup và stop/destroy instance.

Default output: `/workspace/top_v2_v6matched_fixed_3seeds/`. Mean ± sample SD chỉ có khi cả ba seed hoàn tất50epoch và final test; không chọn best seed.

## Metrics và giới hạn

Validation dùng 22 metrics V6 và chọn best full AUC. Risk mỗi snippet của TOP lấy max qua20 horizons. Test inference đủ1344 video; Combined/Public/Private được tính từ prediction records. Combined không lấy trung bình Public/Private. Hai chuẩn hóa partial AUC có tên riêng.

`report_existing_v6.py` tính Combined tương tự từ prediction V6 cũ, không thay model/checkpoint/retrain. Ví dụ:
```bash
python3 /workspace/TOP_V2_V6Matched_RTX4080_3Seeds_TerminalSafe_Fixed/report_existing_v6.py --v6-run /workspace/riskprop_v6_vast_3seeds/seed_42/run --output /workspace/v6_seed42_combined.json
```
Script cần runtime nghiên cứu có numpy/sklearn; có thể dùng Python venv đã cài nếu Python hệ thống thiếu package.

**Test mTTA vẫn null** như scalar-test V6; chưa có evaluator full risk curve tương đương PDF. Không đặt validation mTTA vào cột test paper. Đây là benchmark chung giữa các triển khai của nhóm, không chứng minh tái lập Table1.

Đọc verification.json: CPU/fixture có phạm vi rõ, chưa kiểm full224 Nexar trên RTX4080. Giữ dataset/metadata/testlabels/output PRIVATE và LICENSE Nexar.

Nguồn: https://github.com/happytianhao/TOP/tree/772c6e57e664ecde83f9d92d0641e1cba99abfb7 ; https://papers.nips.cc/paper_files/paper/2025/hash/0d29fdaf83ee24f9fa22d7a255bf75e7-Abstract-Conference.html ; https://github.com/xingyueye5/RiskProp/tree/579376fa1d879f28a9d68c2565d1e449de24b666
