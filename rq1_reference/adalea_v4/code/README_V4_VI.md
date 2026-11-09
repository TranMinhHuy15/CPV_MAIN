# AdaLEA V4: train/test đầy đủ và mTTA test

Máy mục tiêu: Linux x86_64, 1 RTX4080 16GB. Terminal, không cần notebook.
Gói không thuê/dừng instance Vast và không tự tải video.

## Điều cần hiểu trước khi tốn phí

- Đây là **V3 training unchanged + full-curve evaluation**, không phải sửa toàn bộ protocol thành đúng Suzuki hoặc RiskProp Table1.
- Giữ nguyên model.py, runner.py, loss, DSA/QRNN, feature extractor VGG/COCO, batch2acc8 effective16, AMP, SGD.002,50epochs và seed42/43/44.
- Giữ các thay thế đã khai báo: không NIDB pretraining, detectorCOCO thay VOC+fine-tuning, clock/nativeFPS và reduction V3, nhiều hyperparameter suy ra. Không hứa điểm tăng hay ngang paper.
- KHÔNG thay training/checkpoint selection để tối ưu mTTA test.
- Nếu đã có V3 hoàn tất50epoch, dùng chế độ evaluate, **không cần train lại**. V3 partial không được gọi là full; seed44 mới2epoch thì không thể báo mean3seed.
- Không resume training V3 vào V4 vì contract khác; chỉ đọc checkpoint V3 để evaluation. Dữ liệu và output cũ không bị sửa.
- CPU tests không thay thế kiểm tra GPU thật. Launcher chạy GPU predictor check; preflight đo extractor thực trên máy mục tiêu trước train. Không có chứng nhận RTX4080 từ máy local.

## Đầu vào

1. Giải nén gói vào /workspace/AdaLEA_V4_FullMetrics_RTX4080_Terminal.
2. Giải nén Nexar vào /workspace/Nexar_HF: đủ2844 MP4 duy nhất, nội dung khớp inventory.
3. ffprobe phải có trong PATH (gói hệ thống ffmpeg), libGL.so.1 và libglib-2.0.so.0 có sẵn.
4. Torch/vision weight download cần Internet lần đầu. Cần dung lượng trống; preflight tính upper bound cache +8GiB reserve, không tự xóa dữ liệu.
5. Dataset và thư mục output không để lồng vào output cũ chứa bản sao video, tránh duplicateID.

Các path trong lệnh là VÍ DỤ; đổi thành path thực trên máy bạn.

## Môi trường

Có thể dùng runtime V6/V3 đang có nếu đúng Python3.8.5, torch2.1.0+cu118, torchvision0.16.0+cu118 và các pin requirements-kaggle.txt.

Nếu cần tạo runtime riêng, chạy từ thư mục gói:

```bash
python3 run_vast.py setup --output /workspace/adalea_v4_run --hours 1
```

Setup kế thừa installer cô lập của V3, không đổi Python hệ thống. Thời gian cap là giới hạn, KHÔNG phải dự báo thời gian setup.
Nếu thiếu ffmpeg/thư viện hệ thống, cài chúng theo quyền quản trị container trước. Không tắt kiểm tra dependency để tiếp tục.

## A. Train mới đủ3seed rồi test có mTTA

Khuyến nghị đo preflight trước:

```bash
python3 run_vast.py preflight --data-root /workspace/Nexar_HF --output /workspace/adalea_v4_run --hours 1
```

Đọc preflight.json: VRAM, thời gian cold feature extraction, dung lượng cache. Nó KHÔNG dự báo đầy đủ tổng thời gian3seed. Sau khi chấp nhận chi phí:

```bash
python3 run_vast.py train --data-root /workspace/Nexar_HF --output /workspace/adalea_v4_run --hours 72 --ack-approximation
```

Một lệnh tự preflight -> train -> test cả3seed cũng có:

```bash
python3 run_vast.py all --data-root /workspace/Nexar_HF --output /workspace/adalea_v4_run --hours 72 --ack-approximation
```

72h là CAP tổng wall-clock cho một invocation, gồm mọi seed/feature/test/checks; **không phải cam kết sẽ xong trong72h**. Có thể đặt cap khác, tối đa168h. Hết cap exit75, chạy lại cùng lệnh/path/config để tiếp tục từ checkpoint epoch hoàn tất và từng curve đã lưu. Không cộng riêng72h cho mỗi seed.

Muốn dùng runtime đã có, thêm vào mỗi lệnh, ví dụ:

```bash
--runtime-python /workspace/adalea_v3_fixed_runtime/venv/bin/python
```

Đường dẫn phải đúng thực tế, không đoán nếu runtime của bạn tên khác.

## B. Chỉ bổ sung full test/mTTA cho checkpoint V3 đã train

Giữ nguyên output V3 đã giải nén gồm feature_weights.json và các seed_42/seed_43/seed_44 với resume_latest.pth, completion.json, test_predictions.json.
Không đưa TAR chưa giải nén vào --legacy-output. Không cần copy toàn bộ cache; nếu thư mục features còn, chương trình đọc nó làm fallback chỉ đọc.

```bash
python3 run_vast.py evaluate --data-root /workspace/Nexar_HF --output /workspace/adalea_v3_full_metrics --legacy-output /workspace/adalea_v3_fixed_output --seeds 42 43 --runtime-python /workspace/adalea_v3_fixed_runtime/venv/bin/python --hours 12 --ack-approximation --trust-v3-checkpoints
```

Chỉ dùng --trust-v3-checkpoints với checkpoint do chính bạn tạo, không tải pickle không rõ nguồn. Kiểm contract/hash/source/best-validation trước model inference.
Nếu đủ cả3seed đã50epoch, đổi --seeds thành42 43 44. Bảng luôn ghi n thực tế; không giả lập seed còn thiếu.
12h chỉ là cap, không phải ETA. Có thể tăng cap sau khi đo máy thực.

## Output chính

- full_metrics_summary.csv / .json: Combined1344, Public, Private; từng seed + mean +sampleSD(ddof1); n1 thì SD để trống.
- seed_*/full_test_metrics.json: scalar và temporal metrics cùng diagnostics.
- seed_*/full_test_curves/*.json: mọi endpoint trên toàn clip quan sát được, scores, native frame indices, timestamps, checkpoint/video/evaluation provenance.
- seed_*/capture_complete.json: hash đủ1344 curves.
- seed_*/inferred_test_anchors.json: onset/event suy ra, KHÔNG nhãn được tác giả xác nhận trực tiếp.
- seed_*/original_test_predictions.json: điểm scalar cuối clip gốc được giữ lại để bảo toàn mAP/AUC cũ.
- checkpoint/history/ATTC/resume giữ như V3; complete training và complete evaluation là hai trạng thái khác nhau.
- Những tệp official_test_metrics.json hoặc three_seed_summary cũ nói mTTA=null là scalar-only V3; dùng full_metrics_summary cho V4.

## Định nghĩa đầy đủ metric

Scalar: mAP, mAUC_grouped, mPAUC_sklearn_standardized, mPAUC_raw_divided_by_0.1, và AP/AUC/pAUC từng nhóm lead0.5/1/1.5s. Combined được tính trên tất cả1344 prediction, không lấy trung bình Public và Private.

Temporal:

1. **mTTA_test_author_index10_inferred_s**: phép tính index/10 theo quy ước V6/TOP. Threshold từ rank negative-frame, giữ số lần xuất hiện và điều kiện >=, nên ties có thể làm actual FAR lớn hơn. Dùng đối chiếu nội bộ cột mTTA đã tính cho V6/TOP; KHÔNG gọi chắc chắn là giây vật lý.
2. **mTTA_test_physical_strictFAR_inferred_s**: dùng ffprobe PTS thật, cảnh báo chỉ sau onset và trước event; chọn threshold ngay trên các rank score âm bằng nextafter để FAR không vượt0.1. Miss/no observed onset window đóng góp0 trong mẫu số tất cả positive.
3. **mTTA_test_physical_TPonly_strictFAR_inferred_s**: thêm biến thể TP-only tại từng threshold; threshold không có TP đóng góp0. Không lấy nó thay cột1 khi so V6/TOP.

Onset/event suy ra: event = stream_end + time_to_accident; onset = event - (time_of_event - time_of_alert). Đây là metadata adapter như audit V6/TOP trước, chưa được xác minh trực tiếp với tác giả. Clip đã cắt trước event: không dựng risk tương lai, không kéo dài đường dự đoán. Onset ngoài clip là trường hợp không quan sát được, không bị xóa khỏi mẫu số.

Hai chuẩn pAUC và ba mTTA không được gộp hay đổi tên thành metric paper đã xác nhận. Để so công bằng vật lý với V6/TOP phải áp dụng cùng physical evaluator lên curves V6/TOP, không đặt cột vật lý mới cạnh cột index10 cũ rồi kết luận tốt hơn.

## Cách quét nhân quả và bảo toàn scalar

Grid cuối-clip stride3 native frames (giống sampling V3/V6), không tuyên bố luôn đúng10FPS vật lý. Mỗi endpoint chạy QRNN lại trên tối đa30 endpoint quá khứ, padding frame0 ở đầu clip; không dùng frames tương lai và không mang hidden state vượt cửa sổ đã khai báo.
Frame CNN features được cache một lần và dùng lại giữa các seed/cửa sổ. Điểm cuối full curve phải khớp tail inference cũ trong1e-4; sai thì dừng. FP32 evaluation, không autocast. Không đổi scalar score chỉ để làm bảng đẹp.

## An toàn vận hành

- Dùng tmux nếu muốn giữ terminal job khi SSH ngắt. Đóng tmux session không tương đương dừng instance.
- Ctrl+C/deadline dừng process tree; không tự giảm batch, không tự đổi AMP, không tiếp tục sau nonfinite.
- File khóa chặn hai job cùng output. Manifest/hash ngăn trộn các phiên bản.
- Read-only legacy cache chỉ được dùng khi feature weight provenance khớp.
- **Dừng script không dừng tính phí Vast. Bạn phải Stop/Destroy instance trong Vast sau khi tải output.**
- Không xóa cache/checkpoint/log khi chương trình còn chạy. Tải cả output hoặc giữ ít nhất checkpoint, curves, manifest, histories và summaries.

## Kiểm thử

```bash
/path/to/pinned/python test_adalea.py
/path/to/pinned/python test_full_eval.py
/path/to/pinned/python test_process_safety.py
```

Các fixture dùng dự đoán/feature giả lập, không phải số nghiên cứu. LOCAL_VERIFICATION.json ghi chính xác những kiểm thử local đã làm và phần GPU chưa xác minh.
