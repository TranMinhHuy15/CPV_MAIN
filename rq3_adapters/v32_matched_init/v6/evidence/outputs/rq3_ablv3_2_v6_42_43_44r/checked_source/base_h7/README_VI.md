# RQ2 Medium H7 — v1 FIXED, 29/10/2026

Gói này CHỈ train H7, seed42/43/44,50epoch/seed. Nền: ZIP RQ2 Checked
SHA256 a7d7e3505667ec6e689a90bcb7e2ece06c1378451826f9f2e995afb83b8525e7. Không warm-start từ H1/H26; mỗi seed khởi tạo lại từ Kinetics710.
Không thêm attention, projection trước LSTM, loss mới hoặc đổi batch.
H1/H26 cũ và gói của H còn lại không bị ghi đè/resume vào đây.


## Bản vá vận hành 1.1 — đọc trước khi chạy

- Mô hình, loss, input/split, optimizer, checkpoint selection và evaluator giữ nguyên byte so với gói v1 cùng H.
- Mặc định ghi vào `/workspace/rq2_medium_h7_fixed/` để không trộn output gói cũ.
- Không thay code khi job cũ đang chạy. Partial run của v1 không được tự trộn với contract FIXED; giữ ZIP cũ để tiếp tục job cũ hoặc dùng output mới cho FIXED.
- Ngắt trước epoch đầu: chỉ khi contract đúng và chưa có epoch commit, thư mục partial được giữ nguyên dưới `run_uncommitted_*` rồi khởi tạo lại cùng seed/pretrained. Artifact không rõ nguồn hoặc có dấu hiệu epoch muộn vẫn bị chặn để bảo toàn dữ liệu.
- Restore mọi seed trước cổng test; evaluate-only tuyệt đối không train. Thiếu seed/epoch/artifact hoặc parity contract thì dừng trước inference.
- Kiểm đủ checkpoint/resume/RNG/history/snapshot; CRC và đối chiếu snapshot phát hiện artifact bị sửa. Cùng process chỉ cache kiểm tra khi toàn bộ signature file không đổi.
- Settings ghi nguyên tử và kiểm full resume contract trước khi thay file settings cũ.
- Deadline được giữ qua tiến trình lồng nhau, kiểm cả hash dữ liệu và backup. Linux chỉ outer supervisor sở hữu process group; Windows dùng Job Object và chờ toàn bộ cây tiến trình dừng, không phụ thuộc taskkill.
- Khi hết cap, output còn nguyên; có thể chạy riêng stage backup. Cap không dừng billing hoặc destroy máy thuê.
- Xem `PATCH_REPORT_VI.md` và `VERIFICATION.json`. Không có cam kết tăng điểm hoặc hết mọi lỗi GPU/mạng/đĩa chưa được chạy thật.

## Cấu hình khoa học

- 1200train(600+/600-),300val(150+/150-),1344test(672+/672-).
- HF revision7535d0656dac31d7da2846913bb69c6331d2a70a; đủ2844MP4.
- 25prefix +30target =55snippet;5frame/snippet; stride3 nativeframe;224x224.
- LSTM một chiều2048->2048,một lớp; mỗi cửa sổ reset h/c=0.
- H7 vector/cửa sổ; khoảng quan sát danh nghĩa 1.0s ở30FPS.
- SGD0.002,momentum0.9,wd1e-4; milestones20/40,gamma0.1;clip40.
- Microbatch2,accumulation8,effective16;world_size1.
- BCE+1.5FFR+1.1AMC;pos_weight1. Loss chỉ trên30target.
- Encoder/LSTM/head đều train; AMP forward,lossFP32; inferenceFP32.
- Checkpoint: validation FULL mAUC@ cao nhất, không phải mAP/low-FPR mAUC.
- Giữ nguyên decoder/JPEG/native-frame pipeline và28metric của RQ2 Checked.
- mTTA là index-based/inferred-anchor; rawpAUC/0.1 khác sklearn standardizedpAUC.
- Các kết quả mở rộng là nghiên cứu tiếp nối sau prior test inspection.

## Upload và chạy trên Vast

Chỉ cần upload ZIP code này. Không cần upload checkpoint H1/H26 để khởi tạo.
Dataset tự tải khi thiếu; cần HF account đã được Nexar cấp quyền.
Token chỉ nhập tại prompt ẩn hoặc HF_TOKEN của terminal, không đưa vào lệnh/ZIP/log.

Cài system tools trong terminal:

    apt-get update && apt-get install -y tmux unzip ffmpeg libgl1 libglib2.0-0

Giải nén ZIP vào /workspace, rồi chạy TRONG tmux:

    tmux new -s rq2_h7
    HOURS=72 bash /workspace/RQ2_Medium_H7_Checked_v1_FIXED/run_auto.sh

Chuỗi tự động:
verify package -> tải/resume+hash Nexar -> setup/runtime checks ->
CPU tests + full224 GPU smoke -> train3seed (validation-only) ->
evaluate3seed -> summary150epoch -> verified TAR+SHA256.

Ba seed chỉ được đưa sang final test sau khi toàn bộ seed được chọn đã train đủ50epoch.
Pipeline mặc định không train lại H1/H26, không đổi recipe khi OOM/điểm thấp.
72h là cap cho toàn thao tác, KHÔNG bảo đảm đủ. Hết cap không dừng billing.
Nếu chỉ smoke: dùng stage preflight; nếu chỉ train/val: stage train.

    python3 /workspace/RQ2_Medium_H7_Checked_v1_FIXED/medium_ops.py preflight --hours 6
    python3 /workspace/RQ2_Medium_H7_Checked_v1_FIXED/medium_ops.py train --hours 72

Chạy lại cùng lệnh all/train trên cùng output để strict resume.
Nếu train đã hoàn tất, all bỏ qua train và tiếp tục phần test/backup còn thiếu.
Không dùng --resume-root trỏ sang gói cũ hoặc một seed/H khác.

## GPU, đĩa và mạng

Ưu tiên4080/4080SUPER16GB; không khóa theo tên model.
Yêu cầu một GPU expose,VRAM>=12GiB,runtime CUDA tương thích.
12GiB chưa được chứng minh full224; GPU smoke thực tế quyết định có chạy được không.
Không tự hạ batch/ảnh. GPU mới không tương thích torch2.1/cu118 cũng phải dừng.
Log cũ4080SUPER: khoảng17h/50epoch có validation; dự phòng17-20h/seed.
Một máy chạy3seed: khoảng51-60h + setup/test, chưa phải đo gói H7.
MP4 khoảng29.2GiB; cần thêm runtime/checkpoint/backup. Reserve output24GiB là guard,
không phải cam kết mọi ổ đĩa24GiB đều đủ. Ưu tiên SSD đủ dư, không xóa checkpoint đang chạy.

## Xem tiến độ và toàn bộ epoch (terminal thứ hai)

    python3 /workspace/RQ2_Medium_H7_Checked_v1_FIXED/medium_ops.py status
    python3 /workspace/RQ2_Medium_H7_Checked_v1_FIXED/medium_ops.py epochs

Mỗi gói có150epoch validation khi đủ3seed. CSV đầy đủ được tự lưu:
/workspace/rq2_medium_h7_fixed/all_validation_epochs.csv.
status/epochs chỉ đọc; không tác động job. Kết quả per-seed ở:
/workspace/rq2_medium_h7_fixed/medium_h7/seed_42/run/ (tương tự43/44).

## Backup và rời máy

Ctrl+B rồi D để detach; tmux attach -t rq2_h7 để xem lại.
Đóng web sau detach không dừng server. Không sửa code đang chạy.
Sau complete, TAR ở /workspace/rq2_medium_backups/, kèm .tar.sha256.
TAR gồm code/checkpoint/resume/RNG/curve/report/log; không có dataset/runtime.
Nếu job dừng, backup bằng:

    python3 /workspace/RQ2_Medium_H7_Checked_v1_FIXED/medium_ops.py backup

Download TAR+SHA256, verify ở máy bạn rồi mới tự Destroy instance.
Code không tự stop/destroy Vast. Không có tiền/metric nào được bảo đảm.

## Tái sử dụng nghiên cứu cũ và mở rộng sau này

Report H1/H26 cũ chỉ là reference riêng; không chép vào output để giả làm run mới.
Giữ bản ZIP cũ để đọc/resume đúng contract cũ. Gói này chỉ tổng hợp H7 ba seed.
Checkpoint mới và metadata đủ cho bridge RQ3 về sau; hiện chưa có RQ3/attention.
Không dùng cache encoder khác seed/H thay cho train end-to-end.

## Phạm vi kiểm chứng

Đọc VERIFICATION.json và LOCAL_TEST_REPORT.json.
Kiểm CPU chạy ở runtime audit riêng, không được trình bày thành fullCUDA production.
Torch production vẫn2.1.0+cu118/Python3.8.5; GPU smoke trên host là gate bắt buộc.
Legacy fixtures ghi32px/constant scores trong LEGACY_LOCAL_VERIFICATION.json chỉ là
lịch sử kiểm phần mềm, không phải metric nghiên cứu của H7.
