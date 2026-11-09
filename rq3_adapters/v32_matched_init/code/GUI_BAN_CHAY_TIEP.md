# Gửi bạn chạy tiếp — RQ3 ablation V3.2

Gói này chứa **hai bộ code H7/V6 và source gốc đã kiểm hash**, chưa có kết quả V3. Mỗi nền train mới sáu adapter C7/R7/A7/A26/U3/G3 × seed42/43/44 ×50epoch = **18run/nền**, tổng36run nếu chạy cảhai.

Giữ nguyên loss, c2, width512, dropout0,1, bound1. Mục đích là sửa phép so sánh init/RNG và theo dõi bão hòa, không thay thuật toán để chạy theo testV2. Các outputV2 phải được giữ riêng.

Xem [bản vá V3.2](BAN_VA_V3_2.md) và [chi tiết V3.1](BAN_VA_V3_1.md) để biết cách phục hồi, xử lý cache và kiểm dung lượng.

## 1. Máy cần có gì?

- Linux với GPU NVIDIA, `ffmpeg`, `ffprobe`, `nvidia-smi`; runtimeV2 tương thích có thể dùng lại.
- DatasetNexar với `train`, `test-public`, `test-private`.
- Ba baselineH7 hoặcV6 đúng checkpoint đã dùng trước, đủ metadata/validation/testseal. V6seed44 là rerunepoch35.
- Nếu còn cacheV2, giữ nguyên để mượn theohash; không có sẽ extract lại.
- Dung lượng để giữ full checkpoint/snapshot của18run và cache. Mặc định không compact/xóa; reserveguard chỉ là mức tốithiểu, không đảm bảo đủ cho cảround.

**Video và baselineweights không nằm trong ZIP code này.** Chạy trên máyV2 cũ sẽ thuận tiện nhất vì các đầu vào đã có.

## 2. Chạy H7 trên máy H7

Giải nén bundle, vào `RQ3_ABLV3_H7`, rồi chạy (đổi đườngconfigV2 cho đúngmáy):

```bash
python3 make_inputs_from_v2.py \
  --v2-config /workspace/outputs/rq3_ablv2_h7_42_43_44/inputs.resolved.json \
  --output-root /workspace/outputs/rq3_ablv3_2_h7_42_43_44 \
  --write inputs.json
python3 launch.py --config inputs.json preflight
python3 launch.py --config inputs.json all
```

## 3. Chạy V6 trên máy V6

Vào `RQ3_ABLV3_V6`:

```bash
python3 make_inputs_from_v2.py \
  --v2-config /workspace/outputs/rq3_ablv2_v6_42_43_44r/inputs.resolved.json \
  --output-root /workspace/outputs/rq3_ablv3_2_v6_42_43_44r \
  --write inputs.json
python3 launch.py --config inputs.json preflight
python3 launch.py --config inputs.json all
```

Nếu không có config cũ, điền `inputs.example.json` theo README từnggói. `prior_study_root` đểnull; mainV3 không cần prediction A7/G3 cũ. MỗiGPU chạy mộtgói, không chạyhai job lêncùngGPU0.

`all` tự dùngruntime hiệncó hoặc setup nếu thiếu, verify/extractcache, tạo teacherlogits, pilot,train18run,freeze,test vàbootstrap5000. Helperchạy thống kê bằngPythonvenv đãpin, không phụthuộc host cósklearn.

## 4. Pilot và resume hoạt động thế nào?

Pilot dùng C7seed42, dừngsauepoch2 đãcommit, mởprocessmới tiếpđếnepoch3, rồi mới chạyroundđầyđủ. Exit75 của workerpilot là dừngchủđích. Baepochđó được dùngtiếp, khôngtrainlại từđầu.

`PILOT_RESUME_RECEIPT.json` ghi nhận history1,2,3 khôngtrùng và snapshotcommit đượckiểm. Đây chưaphảichứngminh bit-exactCUDA so vớichạyliền. NếuCUDA/dataset/runtime gặp vấnđề, dừng ởpilot trước khi tốnfullround.

Nếu đangchạy bịngắt, giữnguyêncode/config/path/output và chạylạicùnglệnh. Khôngxóa manifest/contract đểépresume. Đổi codehoặccấuhình thìđặtoutputrootmới.

## 5. Bản này có gì mới?

- ETA xử lý đủ0..6arm.
- Finaladapter khởi tạo canonicalchung, táchCPURNG; sharedtensors vàRNGtraining giữa6arm được kiểm ởheadtíchhợp. Parentnguồn vẫn tạo compatibilityM1rồiđượcthay, khôngtuyênbốtrùnginitintegratedM1cũ.
- A7/G3 train mới cùnground. Khôngreuse priorprediction đểtạo kếtluậnmain.
- Bindplan/code/config/source/retention; worker vàanalysis từchối dữliệu cũhoặc đãđổi.
- Diagnostics passivegate/raw/correction/attentionentropy/omega trênvalidation; ghi/fsync trướccommitepoch vàtrimuncommittedkhiresume. Khôngđổiloss/chọnepoch/loạiseedtheodiagnostics.
- Giữfullcheckpoint/resume mặcđịnh. Cacheformat vàsourcegốc khôngsửabyte.
- Bootstrapweighted nhanh, đãkiểmscoreties/videolặp so sklearn; cross-base kiểmprovenance/codehash.

## 6. Sau khi xong gửi gì về?

Gửi folderoutput hoặc ZIP đủ code/config/log50epoch, gồm:

1. `study_contract.json`, `analysis_plan.md`, `inputs.resolved.json`, `runtime_identity.json`, pilotreceipt, selectionlock.
2. Mỗi18run: validationhistory đủ50, trainingstatus, namedsteps/namedepochs, diagnosticsvalidation+samples, resolvedconfig, smoke/timingreceipt, rawlog, selectedadapterNPZ+manifest. Giữfullcheckpoint ởmáy/bảnsao antoàn.
3. `test/`: prediction1344video/role/seed, rawcurves/parity/commit/evaluationseal.
4. `comparison/`: bảng28metric, bootstrapdraws, CSV,decisions,`BOOTSTRAP_COMPLETE.json`.

Không chỉ gửi ảnh bảng. Đừng loại seed44, đổi bestepoch theo test hoặcchỉ gửi nhánhthắng. Nếu thiếu dunglượng thì nén metadata/logriêng, giữadapter/hash vàbảnsao trước khidọn.

## 7. Gộp hai nền

Đưa `comparison/` hai nền về cùngmáy, dùng Pythonruntime cóNumPy/sklearn:

```bash
/workspace/runtime/rq3_checked/venv/bin/python analyze_cross_base.py \
  --h7 /path/H7_OUT/comparison \
  --v6 /path/V6_OUT/comparison \
  --out /path/CROSS_V3
```

Thay prefixruntime theo `runtime_root` thật. Cần đủdraws/CSV/decisions/completemarker. Không trộnV2 vớiV3, khôngsửa codephântích saubootstrap.

## 8. Đã kiểm gì, chưa kiểm gì?

Xem `verification.json`, log CPU V3.2 trong `validation/` và `BAN_VA_V3_2.md`. `INDEPENDENT_REVIEW.md` là ghi nhận lịch sử của V3. Có unit/fixture choheadinit/RNG,ETA,guard,diagnostics/resume,sharedCNN7head vàbootstrap. Codeđượcreviewđộclậpchỉđọc và sửa finding.

**Chưa chạy CUDAtraining hoặcpilotGPU tại nơi đónggói.** Pilotthật nằm tronglệnhchạytrênmáyGPU. Khôngcó sốkếtquảV3 được tựđiền; testV2 đãđượcxem nênroundmới vẫn exploratory.
