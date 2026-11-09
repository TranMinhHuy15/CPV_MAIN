# AdaLEA V4 Full Metrics - terminal Vast RTX4080

**Đọc README_V4_VI.md trước. Phần V3 dưới đây chỉ giữ làm lịch sử training.**

V4 giữ nguyên model/loss/train V3, thêm full-curve test và ba quy ước mTTA.
Không dùng các đoạn dưới nói "mTTA chưa có" để mô tả evaluator V4.

## Sửa kỹ thuật ngày 09/10/2026

Đọc `FIXES_VI.md`. Cache NPZ đọc mỗi mảng một lần, giữ nguyên feature;
preflight luôn đo cache lạnh/nóng riêng, VRAM gồm extractor và max6 VGG crops,
hai SGD updates thật, FP32 evaluation. Giới hạn giờ từ chối NaN/Infinity;
test dừng cây tiến trình chạy bắt buộc trên máy thuê. Disk gate tính phần cache
còn thiếu khi resume, không yêu cầu lại toàn bộ dung lượng đã dùng.

Đây là protocol Fixed mới: dùng output mới, không resume checkpoint V3 cũ.
ZIP/V6/TOP cũ không bị ghi đè; model, loss, dataset, sampler và metric không đổi.

Tái triển khai Suzuki CVPR2018 **DSA + QRNN + adaptive loss** trên Nexar,
không phải historical head của RiskProp, không phải chỉ đổi loss SlowOnly.
TOP dùng gói `TOP_V2_V6Matched_RTX4080_3Seeds_TerminalSafe_Fixed`; giữ nguyên RiskProp V6.

Đọc `PROTOCOL_VI.md` trước khi chạy. Code gốc Suzuki/AdaLEA dùng trong Table1
RiskProp chưa xác minh được; bản này không thể được gọi là “100% code tác giả”.

## Giữ chung và thay đổi

Giữ chung với V6/TOP: 1.200 train / 300 validation / 1.344 test đúng ID;
temporal endpoint sampler, selection validation full mAUC, evaluator 22 validation
metrics, test Combined/Public/Private, seeds42/43/44, 50epoch,
SGD.002/momentum.9/WD.0001/milestones20/40, batch2acc8 effective16.

Đổi để bám AdaLEA: VGG global/local, DSA hidden-conditioned, QRNN causal window2,
AdaLEA positive weight Eq.5, gamma5 **frame**, cập nhật ATTC sau validation,
negative BCE; **không FFR/AMC**. Checkpoint lưu ATTC/RNG/optimizer/scheduler/scaler.

Khác biệt khai báo: VGG ImageNet frozen (không NIDB); COCO FasterRCNN frozen
(không VOC/Nexar bbox fine-tune), top5 objects/threshold.5 inferred; cacheFP16;
không spatial RandomResizedCrop/Flip của V6. Không matched-backbone/FLOPs với V6.
Đây là so sánh nội bộ cùng dữ liệu/evaluator, không khẳng định Table1 equivalence.

## Chuẩn bị

1. Giải nén ZIP vào `/workspace`; dataset toàn bộ Nexar MP4 nằm dưới một data root.
   Không để hai bản sao cùng ID dưới root đó.
2. Dùng Python runtime V6 **torch2.1.0+cu118 / vision0.16.0+cu118** nếu có. Không dùng
   thẳng Python/PyTorch mới của template Vast. Nếu chưa có runtime, chạy setup bên dưới.
3. Output mới, không trỏ vào thư mục V6/TOP hoặc AdaLEA historical. Chỉ một GPU visible.
4. Giữ dataset/gói metadata chứa test labels **Private**, giữ LICENSE Nexar.
5. Cache upper bound chưa nén từ inventory hiện tại khoảng64.28GiB, cộng8GiB
   reserve và runtime/checkpoint. Preflight kiểm disk; không tự giảm object để lách OOM/disk.

Ví dụ các path là minh họa; sửa đúng path máy bạn:

```bash
cd /workspace/AdaLEA_V3_PaperMethod_Nexar_RTX4080_3Seeds_Fixed

# Chỉ khi chưa có runtime V6 tương thích. Giới hạn setup 1 giờ.
python3 run_vast.py setup --output /workspace/adalea_v3_fixed_results --hours 1

# Không train dài: hash toàn dataset + pretrained extraction + gradient/AMP/causal/timing.
python3 run_vast.py preflight \
  --data-root /workspace/Nexar_HF \
  --output /workspace/adalea_v3_fixed_results --hours 1
```

Nếu dùng runtime V6 có sẵn, **bỏ setup** và thêm vào preflight/train:

```bash
--runtime-python /path/to/EXISTING_V6_RUNTIME/venv/bin/python
```

Preflight tải VGG (~528MB) và FasterRCNN (~160MB) nếu chưa có. Nó chỉ dùng hai
video **TRAIN** (1 dương/1 âm), không đọc solution test để chọn hyperparameters.
Report `preflight.json` có thời gian thực, VRAM, disk upper bound và thời gian
feature extraction upper estimate. Không khẳng định “chắc xong trong 3 ngày”.

**Chỉ chạy tiếp khi preflight success và bạn chấp nhận ước lượng chi phí.**

```bash
# 70 giờ là giới hạn TOÀN JOB, không phải mỗi seed; không bảo đảm hoàn thành.
python3 run_vast.py train \
  --data-root /workspace/Nexar_HF \
  --output /workspace/adalea_v3_fixed_results \
  --hours 70 --ack-approximation
```

Một lệnh chạy tuần tự seed42 ->43 ->44; mỗi seed init riêng, train50epoch,
chọn best validation rồi test đủ1.344. Frozen feature cache dùng chung, **không**
chuyển predictor/optimizer/adaptive state từ seed42 sangseed43. Không chọn bestseed.

Nếu muốn giữ terminal sau disconnect, dùng tmux trên máy thuê:

```bash
tmux new -s adalea3
# Chạy lệnh train ở trên; detach Ctrl+B rồi D; trở lại: tmux attach -t adalea3
```

## Dừng và chạy tiếp

Ctrl+C hoặc deadline dừng toàn process group trên Linux. Checkpoint chỉ commit sau
epoch đã train+validation xong. Epoch dở dang bị chạy lại khi resume; không giả làm
đã hoàn tất. Chạy lại **cùng lệnh, cùng output/config/runtime** sẽ resume checkpoint
của seed hiện tại; seed hoàn tất có hash đúng sẽ được bỏ qua.

Giữ **toàn bộ output**, gồm feature cache, manifest, feature weight hashes,
`seed_*/resume_latest.pth`, logs và completion. Không chỉ copy bảngCSV.
Không resume V6/V7/historical AdaLEA. Nếu đổi config/code phải dùng protocol/output mới.

Quan trọng: script không thuê/tắt/xóa instance. **Dừng script không dừng phí Vast.**
Bạn cần stop/destroy instance sau khi backup đủ outputs theo lựa chọn của bạn.

## Kết quả và metric

- `seed_42/43/44/validation_history.json`: từng epoch, LR, train loss, ATTC_prev/next,
  22 metric V6. ATTC curriculum không phải `mtta@0.1`.
- `validation_predictions.json`, `validation_metrics.json`: tại best checkpoint.
- `test_predictions.json` / `submission.csv`: đúng 1.344 ID, score cuối window.
- `test_tail_curves.json`: 30 endpoint, **không phải full clip curve**.
- `official_test_metrics.json`: Combined tính pooled1.344 trướcAP/AUC, rồi
  Public/Private; chạy official Nexar evaluator để kiểm parity AP Public/Private.
- `three_seed_summary.json/csv`: mean và sample SD(ddof1), chỉ khi **cả ba** seed
  hoàn tất đủ50epoch + đủtest. Không lấy trung bình checkpoint hay ensemble scores.
- Partial AUC báo cả sklearn-standardized và raw-area/0.1, tên riêng; không chọn
  công thức theo số nào gầnpaper. **Test mTTA vẫn null**, không suy ra từ scalar.
  Chưa đủ test time alignment/evaluator Table1 để gọi cùng chuẩn vớipaper.

## Đã kiểm / chưa kiểm

Chạy test local bằng runtime có torch/numpy/opencv/sklearn:

```bash
python test_adalea.py
python test_process_safety.py  # bắt buộc bởi launcher; POSIX trên máy thuê
python test_gpu_predictor.py  # optional CUDA, synthetic features ONLY
```

`verification.json` ghi rõ CPU fixtures, predictor CUDA local và những giới hạn.
CPU fixture3seed chạy2epoch với feature giả và1.344 ID thật để kiểm plumbing/
official evaluator, **không phải điểm benchmark**. Chưa train dataset thật,
chưa chạy pretrained VGG+detector trên RTX4080: production preflight là bắt buộc.

## Nguồn

- [Suzuki CVPR2018 chính thống](https://openaccess.thecvf.com/content_cvpr_2018/html/Suzuki_Anticipating_Traffic_Accidents_CVPR_2018_paper.html): p4Eq4-6, p6global/local/DSA/QRNN/gamma5.
- [Chan DSA author code](https://github.com/smallcorgi/Anticipating-Accidents/blob/master/accident.py): kiến trúc tham khảo, **không phải code Suzuki**.
- [QRNN author repository](https://github.com/salesforce/pytorch-qrnn): tham khảo recurrence, bản này không dùng CUDA extension cũ.
- RiskProp sampler/metric commit579376fa và evaluator Nexar được vendored cơ học,
  source hashes trong `provenance.json`; checksum toàn gói `PACKAGE_SHA256.json`.
