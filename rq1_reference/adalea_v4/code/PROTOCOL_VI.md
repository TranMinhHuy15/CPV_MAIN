# Hợp đồng training AdaLEA V3 được V4 giữ nguyên

**Evaluation V4 được định nghĩa trong README_V4_VI.md. Phần dưới mô tả training gốc V3; các câu thiếu mTTA chỉ áp dụng gói V3.**

Sửa vận hành trong `FIXES_VI.md`, không đổi scientific settings bên dưới.
Protocol `adalea-v3-paper-method-nexar-v6eval-frozen-vgg-coco5-b2acc8-fixed-v2`.
Dùng output mới; strict contract từ chối checkpoint trước bản Fixed.

Đây là **tái triển khai phương pháp**, không phải code gốc Suzuki và không phải
bản AdaLEA chưa được công bố của RiskProp Table 1. Không thay V6/TOP cũ.

## Cố định trước khi xem test

- Nexar: đúng inventory 2.844 video, annotation native-frame của tác giả RiskProp.
- Đúng ID split V6: 1.200 train / 300 validation / 1.344 test.
- Seed 42, 43, 44; 50 epoch; SGD 0.002, momentum .9, weight decay .0001;
  milestones 20/40, gamma .1, clip grad 40; batch 2, accumulation 8 = effective 16.
- Chọn checkpoint bằng validation full `mAUC@`; không chọn seed tốt nhất.
- Sampler temporal V6 30 endpoint, stride 3 native frames, end-anchored/jitter như
  source; clock đánh giá V6 index/10 giữ nguyên, **không phải physical 10 FPS**.
- Test không nhận target/event/onset; chỉ đọc solution sau khi đủ 50 epoch và
  checkpoint đã chọn. Tính Combined 1.344 từ predictions, không trung bình Public/Private.
- Báo validation 22 metrics và test AP/AUC hai chuẩn partial AUC tách rõ tên.
- Validation dùng đồng thời chọn checkpoint và cập nhật curriculum ATTC theo
  Fig.2 Suzuki; đây là khác biệt training-policy cần công bố với V6/TOP.

## Phần bám paper Suzuki CVPR 2018

Nguồn: https://openaccess.thecvf.com/content_cvpr_2018/papers/Suzuki_Anticipating_Traffic_Accidents_CVPR_2018_paper.pdf

| Thành phần | Trạng thái | Triển khai |
|---|---|---|
| Global/local VGG features + DSA | reported, p.6 | full image + crop object, attention có hidden quá khứ |
| QRNN + sigmoid | reported, p.4 | QRNN causal window 2, fo pooling, 1 layer |
| Adaptive positive loss Eq.4-5 | reported, p.4 | exp(-max(0, d - F*ATTC_prev - gamma)); d tính native frame |
| Negative loss Eq.6 | reported, p.4 | BCE negative; không thêm FFR/AMC |
| gamma=5 | reported, p.6 | **5 frame**, không phải 5 giây |
| ATTC lấy từ epoch trước | reported, Fig.2 | cập nhật sau validation, trước epoch tiếp theo |
| Hidden 512 / embedding 256 | inferred | dựa kiểu DSA Chan 2016, Suzuki không công bố đủ |
| Predictor initialization/dropout | inferred | mặc định Linear PyTorch; dropout .5, fresh riêng mỗi seed |
| ATTC threshold/duplicate recall policy | inferred | unique thresholds, TTC chỉ TP, max TTC tại cùng recall, mean qua distinct positive recall |
| Epoch đầu | inferred/ambiguous | ATTC_prev=0, gamma=5 theo Eq.5; không ép đồng nhất EL với Fig.2 |
| Loss reduction | inferred | mean qua time hợp lệ, rồi mean batch; Eq.4/6 viết sum |

## Những khác biệt bắt buộc phải công bố

- Không có NIDB-pretrain/checkpoint của tác giả. Dùng VGG16 ImageNet FC7 4096d,
  frozen, cả global/local. Không tuyên bố tái lập cấu hình best NIDB.
- Không có annotation bbox Nexar để fine-tune detector VOC như paper. Dùng
  torchvision Faster R-CNN ResNet50-FPN COCO frozen; chỉ các lớp giao thông.
- Object cap 5 và threshold .5 là lựa chọn tài nguyên **inferred**, không phải
  hyperparameter Suzuki. Đổi chúng tạo protocol mới, không resume checkpoint cũ.
- DSA/QRNN viết lại PyTorch; không nhập historical `model_AdaLEA.py` của RiskProp.
- Frozen features được cache FP16, predictor train AMP, validation/test FP32.
  Không dùng RandomResizedCrop/Flip của V6 vì cached feature phải ổn định.
- AdaLEA dùng feature tại endpoint, không 5-frame 3D CNN. Do đó đây là so sánh
  phương pháp cùng dữ liệu/evaluator/training schedule, **không matched-backbone,
  không matched-FLOPs và không cùng mọi augmentation** với V6/TOP.
- ATTC adaptive không phải `mtta@0.1`. Test clip bị cắt trước event, clock/onset
  chưa xác minh: không tạo mTTA test giả. Lưu 30 endpoint tail cùng scalar test
  score giống cửa sổ V6; **không gọi đây là full observed risk curve**. Đánh giá
  full-clip temporal cần thêm một lượt inference/mapping, không nằm trong job này.
- Table 1 RiskProp không công bố đủ AdaLEA setup/normalization. So sánh nội bộ
  V6/TOP được; so số paper chỉ là tham khảo có caveat, không khẳng định cùng chuẩn.

## Rào chắn chi phí

Preflight kiểm CUDA, weights, decode, gradient, causal prefix, accumulation và
đo thời gian trích features thực. Feature extraction có thể là phần tốn thời gian
nhất. Không bảo đảm cả 3 seed xong trong 3 ngày; phải xem preflight trên máy thuê.
Một giới hạn giờ chung bao gồm setup/feature/train/test; Ctrl+C dừng process group.
Dừng script **không tự tắt instance Vast**, phí máy vẫn tiếp tục.

Các điểm thiếu trên được đặt thành lựa chọn cố định, không tối ưu theo điểm test.
