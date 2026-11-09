# RQ3 ablation V3 — V6, sáu adapter train đồng bộ

**Code để chạy tiếp, chưa có kết quả V3.** Đích chạy là Linux + GPU NVIDIA; đã test CPU trên Windows.
Giữ nguyên loss/c2/bound1, train mới C7/R7/A7/A26/U3/G3, seed42/43/44: **18run×50epoch/nền**.
Không lấy prediction A7/G3 cũ vào main. Đây là follow-up exploratory sau khi đã xem testV2.

## Chạy trên máy đã chạy V2

Giải nén bundle, vào folder `RQ3_ABLV3_V6`. Giữ dataset, basecheckpoint và cacheV2 ở chỗ cũ.
Tạo config mới từ config cũ (đường dẫn dưới là ví dụ, thay bằng file thật trên máy):

```bash
python3 make_inputs_from_v2.py \
  --v2-config /workspace/outputs/rq3_ablv2_v6_42_43_44r/inputs.resolved.json \
  --output-root /workspace/outputs/rq3_ablv3_v6_42_43_44r \
  --write inputs.json
python3 launch.py --config inputs.json preflight
python3 launch.py --config inputs.json all
```

Helper không sửa config/outputV2; tự dùng sourcegốc trong `../inputs/` nếu có, nếu không dùng đườngsource cũ.
Cache chỉ được mượn nếu có manifest; worker kiểm lại checkpoint/source/toolhash trước dùng. Không có cache thì extract lại.
`all` chạy setup nếu chưa córuntime, extract/teacher, pilot,18run,freeze,test vàbootstrap5000.
Mỗi GPU chạy một gói. Hai nền có thể chạy trên hai máy/GPU riêng; không chạy đồng thời hai gói lên cùng GPU0.

## Chạy từng bước / tiếp tục khi bị ngắt

```bash
python3 launch.py --config inputs.json setup       # nếu chưa có runtime
python3 launch.py --config inputs.json extract     # verify cache hoặc extract, rồi teacher
python3 launch.py --config inputs.json pilot       # stop sau commit2, resume tới commit3
python3 launch.py --config inputs.json tune        # tự pilot nếu chưa có receipt; train đủ18run
python3 launch.py --config inputs.json test        # chỉ qua sau khi đủ18run/50epoch
python3 launch.py --config inputs.json analyze     # bootstrap trong runtime đã pin
```

Pilot dùng luôn C7seed42, không bỏ3epoch đó; `tune` tiếp từepoch3 tới50. Exit75 của worker pilot là dừng có chủđích.
`PILOT_RESUME_RECEIPT.json` xác nhận đã qua2→3 epoch duy nhất và snapshotcommit hợp lệ. Nó không chứng minh
CUDA bit-exact so với một lượt chạy liền. Nếu crash, chạy lại cùng lệnh/config/code/path; giữ nguyên output.
Không xóa studycontract/manifest để ép nhận code mới. Đổi code/config/plan thì dùng outputrootmới.

## Nếu không còn config/máy V2

Copy `inputs.example.json` thành `inputs.json`, điền:
- `source_zip`: sourcegốc đã pin, bundle kèm `inputs/rq3_selected_source_v1.zip`.
- `data_root`: Nexar có `train/`, `test-public/`, `test-private/`.
- `v6_runs`: đúng3run baseline với checkpointđãchọn và metadata/validation/test đã seal.
- `output_root`: folder mới; `runtime_root`: runtime riêng vớioutput, có thể dùngruntimeV2 tương thích.
- `cache_roots`: đườngcacheđúngseed hoặcnull. `prior_study_root`: đểnull, không cần prediction cũ.

Riêng V6seed44 là rerun epoch35, SHA `5c0e817995491932240f7c2533423b61d47f4d9b5af96e34b65978594b901a17`.
Không thay baseline bằng mộtseed/checkpoint khác chỉvì có điểmtestcao hơn.

## Bản này sửa gì?

1. ETA dùng đủ sáuarm, không crash sauarm3.
2. Finaladapter canonicalG3 + CPURNGfork: sharedtensors trùng giữa6arm ởcùngseed; không tiêu thụ thêm RNGtraining theoarm.
   Parent nguồn vẫn tạo compatibilityM1 rồi được thay; không tuyênbố V3 trùng init của integratedM1cũ.
3. PlanMD/codehash bindtrước train; kiểm ở worker/analysis, config/source/retention cũng bịbind.
4. Sáuarm đều train mới, B0cùngseed; bootstrap/cross-base không lén reuse prior.
5. Diagnostics passive ghi trước commit(priority80 trước90), fsync và trim uncommitted khi resume.
6. Mặc định `compact_completed_trials=false`: giữ fullcheckpoint/resume. Không tựxóa log/dataset/checkpointgốc.

Giữ c2 AdamW2e-4,wd1e-4,dropout.1,width512,batch2×8,50epoch,MultiStep20/40,clip40,FP32,bound1.
Không sửa loss/gatebound theo failureseed44. Bãohòa chỉđược log, không tựđổi epoch/loạiseed/dừngtheometric.

## Cần giữ / gửi về sau chạy

- Folderoutputđầyđủ: studycontract/analysisplan, inputsresolved, runtimeidentity, pilotreceipt, selectionlock.
- Mỗi18run: full50validationhistory, trainingstatus, namedsteps/namedepochs, diagnostics summary+samples,
  resolvedconfig, smoke/timingreceipt, rawworkerlog, compactmanifest/trialcomplete và selectedadapterNPZ.
- `test/`:1344prediction/role/seed,rawcurves,parity/commit/evaluationseal; `comparison/`:28metric,bootstrapdraws/CSV/decision/complete.
- Không chỉ gửi ảnh bảng. Nếu thiếu dunglượng, nén code/log/metadata riêng; giữ ít nhất selectedadapter+hash và không xóa nguồn khi chưa có bản sao.

Logdiagnostics tại `trials/c2_lr_double/ARM/seed_SEED/run/`:
`adapter_diagnostics_validation.jsonl` đủ50epoch (gate/raw/correction/entropy/omega),
`adapter_samples_validation.jsonl` hai validationbatch đầu mỗiepoch.
Moments là mô tả token tươngquan, không phải independentn. `named_epochs` loss là giá trị log tổng hợp, không tựcoi là exactrawepochloss.

## Thống kê và chi phí

MainHolm10test/nền:5contrast×mAP/AP1.5; secondary/cross-base exploratory. CIvideo với3seedfixed.
Cùng5000indices cho mọiarm/seed/nền. Fastweightedbootstrap có test ties/duplicate so sklearn.
Không gọi pMonteCarlo0 là exactp0, không gọi CI chứa0 là equivalent.
Parameter và MAC ghi trong timingprofile; cachefitVRAM/thời gian không phải onlineCNNlatency.
H7base65207361parameter;V6base31636545. C7/A7/A26/U3adapter3676675,R73152386,G33681286.

Sau cảhainền:
```bash
/workspace/runtime/rq3_checked/venv/bin/python analyze_cross_base.py --h7 /path/H7_OUT --v6 /path/V6_OUT --out /path/CROSS_V3
```
Thay prefixruntime tronglệnh bằng `runtime_root` thật nếu khác vídụ. Hostpython có thể chưa cóNumPy/sklearn.
Đưa folder `comparison/` của hai nền về cùngmáy; helper nhậnroot hoặccomparison hoặcfileNPZ.
Cần `BOOTSTRAP_COMPLETE.json`,draws,CSV,decisions và code/plan đúngversion; không trộnV2/V3.

## Kiểm chứng và giới hạn

Xem `verification.json` cho sốtestCPU thựcchạy. `python -m unittest discover -s tests -v` là bộtestnhỏ,
không tảidata/install/GPU. Chưa chạy trainingCUDA/end-to-end hoặc pilotGPU tại nơi đónggói; phải qua preflight/pilot trênmáychạy.
Sourcegốc/cacheformat giữ nguyênbyte và được pinSHA. Không có video/baseweights/resultsV3 trong codeZIP.
Mặc định giữ fullcheckpoint nên cần nhiềuGBtrống; thêm dunglượng cho cache nếu extractlại. Guardreserve không phải ước lượngđủ cho cảround.
