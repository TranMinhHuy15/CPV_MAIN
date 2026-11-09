# RiskProp v6 — TERMINAL SAFE — một RTX4080, ba seed

Dùng ZIP `RiskProp_V6_VastAI_RTX4080_3Seeds_TerminalSafe.zip`, không dùng gói v2 cũ để chạy unattended. Linux x86_64, GPU RTX4080 được expose trong container, có Python3/pip/Internet. Gói giữ runtime tách biệt Python3.8.5/Torch2.1+cu118/MMEngine0.10.7/MMCV2.2 full, không thay driver hoặc Python hệ thống.

## Lệnh terminal

Upload cả ZIP và thư mục dataset Nexar_HF đã giải nén, rồi:

```bash
cd /workspace
python3 -m zipfile -e RiskProp_V6_VastAI_RTX4080_3Seeds_TerminalSafe.zip /workspace
tmux new -s riskprop
```

Kiểm tra trước, KHÔNG train epoch nào:

```bash
python3 /workspace/RiskProp_V6_VastAI_RTX4080_3Seeds_TerminalSafe/run_three_seeds.py --data-root /workspace/Nexar_HF --total-hours 1
```

Controller chạy kiểm tra dừng cây tiến trình thật trên Linux trước, rồi setup/hash/source/loss/GPU smoke full224/mini-validation cho từng seed. Chỉ khi qua hết mới train:

```bash
python3 /workspace/RiskProp_V6_VastAI_RTX4080_3Seeds_TerminalSafe/run_three_seeds.py --data-root /workspace/Nexar_HF --train --hours-per-seed 24 --total-hours 24
```

**Mặc định24h TỔNG** cho một lần gọi, tính cả setup/smoke/train/validation/test. Không còn mặc định cho phép72h cả ba seed. Có thể thay `--total-hours` theo ngân sách; giá trị mới tính lại từ lúc chạy lại lệnh. `--hours-per-seed` là soft training budget của mỗi seed, không vượt total deadline. Chưa bảo đảm ba seed hoàn thành trong24h,72h hoặc bất kỳ budget nào; cần đo full smoke trên RTX4080 thật. Cleanup sau deadline có thể thêm vài giây và OS scheduling có thể trễ; không coi đây là đồng hồ billing tuyệt đối.

Tmux: Ctrl+B rồi D để detach, job vẫn chạy; `tmux attach -t riskprop` để xem lại. Trong phiên đang chạy, Ctrl+C yêu cầu dừng cả nhóm tiến trình. Đóng kết nối SSH KHÔNG phải lệnh dừng khi đã dùng tmux. Chạy lại đúng lệnh train để tiếp tục: seed hoàn tất bỏ qua, seed partial resume đúng checkpoint. Có lock ngăn chạy hai job cùng output root.

Không có cơ chế tự stop/destroy Vast instance. **Dù train đã dừng, máy thuê vẫn có thể tính phí**. Budget bắt đầu từ lệnh chạy code, không từ lúc thuê/upload dữ liệu. Bạn phải kiểm tra, sao lưu và chủ động dừng máy thuê.

## Output và phục hồi

Default `/workspace/riskprop_v6_vast_3seeds/`: seed_42, seed_43, seed_44, summary.json, per_seed_results.csv, mean_sd.csv; job_stop.json nếu bị ngắt/hết giờ/lỗi child.

Sau mỗi epoch+validation hoàn tất, ngoài resume_latest/best/RNG/history/manifest/status, gói tạo atomic `run/safe_resume_snapshot.zip` (không nén để tránh tốn CPU). Snapshot mới chỉ thay snapshot cũ sau khi đã ghi xong. Trước resume, phục hồi đồng bộ checkpoint/optimizer/scaler/scheduler/RNG/best/status/history từ snapshot hoàn tất, tránh dùng nửa file hoặc metadata không cùng epoch sau hard-stop.

Nếu stop khi đang ghi snapshot mới, epoch cuối đã commit trước đó được dùng lại; có thể phải chạy lại phần chưa commit. Nếu chưa có epoch nào hoàn tất/snapshot và metadata dở dang, script dừng để bạn kiểm tra; dùng output root mới thay vì tự ghi đè run lỗi. Không hứa phục hồi khi disk hỏng/mất volume, snapshot bị corrupt hoặc máy bị destroy. Back up toàn bộ output, bao gồm safe_resume_snapshot.zip. Không chỉ tải notebook hoặc best.pth.

Snapshot dùng thêm dung lượng cỡ một bộ resume+best; giữ dung lượng dự phòng cho cả ba seed. Guard15GiB trống ở đầu mỗi seed không bảo đảm đủ mọi tình huống. Dataset2.844MP4 (~31.4GB), runtime/caches và checkpoint cần SSD đủ; không nên giữ thêm RAR nếu đã giải nén. CPU/RAM/SSD của máy thuê chưa được xác nhận chỉ từ ảnh GPU.

## Protocol không đổi về học máy

Seed42/43/44 chạy nối tiếp, mỗi seed bắt đầu từ pretraining Kinetics710 gốc, không lấy weights đã train của seed trước. Cùng fixed author split1200train/300val/1344test. SlowOnly-R50,30×5×224, truebatch2, accumulation8/effective16, AMPtrain và lossFP32, evalFP32batch1; BCE+1.5FFR+1.1AMC, SGD0.002, milestones20/40,50epochs/seed (150total).

Best theo validation full mAUC@, test sau50epochs mỗi seed. Không chọn best seed. Mean ± sampleSD (n−1) chỉ khi cả ba seed train/evaluate xong. Validation proxy và official test Public/Private tách riêng, không tạo testmTTA từ scalar score. Partial AUC normalization/scope PDF vs source chưa được giải quyết; không đổi metric để gần điểm paper. AMP/1GPU/effective16 khác paper8A800/batch64; không gọi là tái lập100% Bảng1.

Protocol mới `riskprop-v6-vast-batch2-eff16-amp-single-gpu-3seed-terminal-v3`. Từ chối checkpoint Kaggle, gói một seed hoặc v2; đây là thí nghiệm mới. Cả train-resume và evaluation kiểm tra checksum/runtime contract. Sửa source/runtime mà không đồng nhất contract sẽ dừng, không tự bỏ check. Hai bản sửa evaluation đã tích hợp, không sửa file giữa seed.

Giữ dataset/code/outputs PRIVATE; gói có metadata/ground-truth test cho final evaluator. Không dùng test để tune, không public video/labels. Giữ LICENSE Nexar. Scripts không thuê/upload/destroy thay bạn.

## Bằng chứng kiểm tra

- Sáu CPU tests config/seed/split-ID thật/source/adapter/queue/resume/thống kê.
- Chín tests tham chiếu loss/gradients/model/sampler/native-JPEG/metrics/reports/resume.
- Tám safety tests: deadline, Ctrl+C và SIGTERM với child+grandchild thật, lock, không launch khi hết giờ, rollback snapshot giả lập ghi dở, global-budget/evaluation-contract, cleanup khi parent thoát. Windows local bỏ qua test orphan cleanup POSIX; test đó BẮT BUỘC chạy trên Linux thuê trước train.
- Cả ba seed đã chạy production smoke worker với ORIGINAL PRETRAINING trên GPU cục bộ RTX3050 Laptop, fixture32px, workers0,16microsteps/seed: backbone SGD cập nhật, loss hữu hạn, accumulation8, FP32 mini-validation qua. Không phải full224/RAM/VRAM/throughput RTX4080, không phải150epochs đã hoàn thành.
- Đã chạy actual Runner train→validation→checkpoint, giả lập file lưu bị ghi dở, khôi phục snapshot đúng hash, rồi actual Runner resume sang epoch tiếp theo (không lặp history) trên RTX3050 với4video/32px fixture. Có partial accumulation flush do fixture ngắn; không dùng loss/metric fixture làm kết quả paper.

Đọc verification.json và các báo cáo để phân biệt kiểm tra thực, fixture/mock và phần chưa kiểm chứng. Chưa cam kết fullRTX4080/Linux không có lỗi; preflight tại máy thuê là gate bắt buộc. Nếu lỗi, gói dừng thay vì tự giảm input/batch/precision để làm sai thí nghiệm.

Nguồn: https://github.com/xingyueye5/RiskProp/tree/579376fa1d879f28a9d68c2565d1e449de24b666 ; https://openaccess.thecvf.com/content/CVPR2026/html/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.html ; https://docs.vast.ai/guides/instances/connect/ssh ; https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
