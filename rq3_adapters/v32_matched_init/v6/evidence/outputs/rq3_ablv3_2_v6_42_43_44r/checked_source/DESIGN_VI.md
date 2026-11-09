# RQ3 H7: ba module tren T4

Day la nhanh moi, khong phai ket qua RQ3 tren H1 va khong thay doi goi H7 goc.
Giua nguyen split 1200/300/1344, seeds 42/43/44 va loss tac gia.

Nen: checkpoint H7 da train du 50 epoch, dung seed, du ho so commit.
Encoder, LSTM2048 H7 va classifier deu dong bang va o eval mode.
Nhanh moi duoc train: logit = logit_H7 + sigmoid(gate)*tanh(residual).
Gate phu thuoc context, bias=-2 nen dau ra khac zero.
Chi lop ra residual khoi tao zero; khong co gate nhan zero thu hai.

M1: projection 2048->512, LayerNorm chi trong nhanh moi, LSTM512
unidirectional reset moi cua so, current-query attention voi age bias.
M2: cac cua so H7/H13/H26 dung chung trong so nhung chay RIENG tu zero;
softmax gate tu ba context. Khong co H1. Day la observation-window fusion,
khong dung nhan lead time hay khoang cach toi event de tao trong so.
M3: spatial pooling giu truc T that cua encoder; chieu tung sai phan qua
projection khong bias va MLP phi tuyen TRUOC khi mean, tranh telescoping.
M3 khong phai optical flow hoac uoc luong van toc vat ly.

Tat ca cua so ket thuc o thoi diem dang du doan. Chung co overlap co chu dich;
KHONG su dung lai dinh nghia lag 5..25/H21 cua adapter H1 cu.
55 snippet = 25 prefix + 30 target; clamp frame dau giong RQ2.

Default: m1, m1_m2, m1_m3, full x 3 seed = 12 run train moi, 50 epoch/run.
B0-H7 duoc cham lai cung evaluator, khong train. Full-only = 3 run, nhung
khong du de quy loi ich rieng cho tung module. Khoa tap arm truoc khi xem test.
Primary: full - B0, AP@1.5s; guardrail delta mAP >= -0.01; checkpoint selection
highest validation full mAUC@. Test da xem truoc => exploratory, khong claim confirmatory.

T4: mot GPU, FP32; encoder forward no_grad theo chunk nho, head train FP32.
Batch 2 x accumulation 8, AdamW lr 1e-4, wd 1e-4, clip 40, epochs 50,
LR milestones 20/40 factor .1, explicit dropout .1. TF32 va autocast tat.
Giu image augmentation RQ2 luc train; encoder dong bang nhung anh train van duoc
decode/augment lai. KHONG dung .npy vector H1 cu, KHONG goi day la cache-training.
Chi phi encoder lap lai moi epoch; khong hua train trong vai chuc phut.

Pipeline duoc tai su dung: base_h7/repro.py prepare+config, taa author loss,
base_h7/kaggle_adapter.py commit/snapshot/RNG, rq2_capture.py 1344 full curves,
RQ1 scalar evaluator + index-based mTTA. Temporal-feature capture duoc thay the
ro rang trong integration.py, doi chieu direct inference truoc khi cham.

Kiem thu: shape, windows independent/order/causality, baseline init parity,
bounded correction, gradient >=2 optimizer steps, non-telescoping motion,
config provenance/resume/test gate, serialized roundtrip. Local CPU fixture
khong thay the smoke CUDA/MMCV/Nexar that.
