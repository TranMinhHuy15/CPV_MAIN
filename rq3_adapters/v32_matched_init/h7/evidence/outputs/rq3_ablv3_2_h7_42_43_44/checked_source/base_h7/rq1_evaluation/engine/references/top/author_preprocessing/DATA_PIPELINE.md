# Nexar 数据处理流程说明

本文档记录当前 `others/nexar/` 目录相对 HuggingFace 官方数据集
[`nexar-ai/nexar_collision_prediction`](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)
所做的全部额外数据处理流程，便于复现与维护。

---

## 1. 数据来源对比

### 1.1 HuggingFace 官方版本

- **媒体形式**：`mp4` 视频，1280×720 @ 30 fps，约 40 秒。
- **规模**：2844 条（train 1500 + test 1344）。
- **标注字段**（每条一行）：
  - `video`：原始视频
  - `time_of_event`：碰撞 / 险情发生秒数（float）
  - `time_of_alert`：应当报警时刻（float，秒）
  - `light_conditions`：`Normal` / `Bright` / `Dark` / `Twilight`
  - `weather`：`Clear` / `Cloudy` / `Rain` / ...
  - `scene`：`Urban` / `Sub-urban` / `Highway` / `Other` ...
  - `time_to_accident`：仅 test，表示 test 视频在事件前多少秒被截断；train 为 `null`
- **正负样本平衡**：50% 正例 / 50% 负例。
- **测试集特点**：每条 ≈ 10 秒，并在事件前 500 / 1000 / 1500 ms 处截断；划分为 public / private 两子集。
- **提交模板**：`sample_submission.csv`，仅含 `id,score`。

### 1.2 当前目录（处理后）

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `train.csv` | 1500 | 原始训练标注（id, time_of_event, time_of_alert, target） |
| `test.csv` | 1344 | 原始测试 ID 列表（id） |
| `annotations.csv` | 2844 | 抽帧后合并标注（已转帧索引） |
| `nexar_train_annotations.csv` | 1200 | 重新切分的训练集 |
| `nexar_val_annotations.csv` | 300 | 从 train 中抽出的验证集（150 正 + 150 负） |
| `nexar_test_annotations.csv` | 1344 | 测试集（结构对齐 train/val） |
| `nexar_val_references.csv` | 6000 | 验证集逐帧参考标签（300 × 20） |
| `index.csv` | — | 上游训练框架统一索引（unified schema） |
| `sample_submission.csv` | — | Kaggle 提交模板（id, target） |

---

## 2. 整体处理流程

```
            HF 原版 (mp4 + 秒级标注)
                     │
                     ▼
        ┌───────────────────────────┐
        │ extract_raw_frames.py     │  视频 → JPG 帧目录 + 秒→帧
        └───────────────────────────┘
                     │   生成 annotations.csv
                     ▼
        ┌───────────────────────────┐
        │ split.py + splits.py      │  从 train 中随机选出 300 条 val
        │ generate_nexar_splits.py  │  → nexar_train/val/test_annotations.csv
        └───────────────────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │ generate_nexar_val_       │  每条 val 视频 → 末尾 20 帧（步长 3）
        │ references.py             │  → nexar_val_references.csv
        └───────────────────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │ duplicate.py / match.py   │  SSIM 去重（可选清洗）
        │ count.py / copy.py        │  完整性校验 / 末 15 帧轻量子集
        └───────────────────────────┘
                     │
                     ▼
                index.csv（与其他事故预测数据集合并）
```

---

## 3. 各步骤详解

### 3.1 视频抽帧（`extract_raw_frames.py`）

将 `train/`、`test/` 目录下的 `mp4` 抽成连续 JPG，输出到：

```
train_raw_frames/<id>/000000.jpg, 000001.jpg, ...
test_raw_frames/<id>/...
```

同时把 HF 的**秒级标注换算成帧级**：

```python
abnormal_start_frame = int(time_of_alert * fps)
accident_frame       = int(time_of_event * fps)
# 负样本一律置 -1
```

输出 `annotations.csv`：

| 列 | 含义 |
| --- | --- |
| `id` | 视频 ID（5 位补零） |
| `is_test` | 0=train，1=test |
| `total_frames` | 视频总帧数 |
| `fps` | 帧率 |
| `abnormal_start_frame` | 报警起始帧（负样本 -1） |
| `accident_frame` | 事故发生帧（负样本 -1） |
| `target` | 0/1（test 为 -1） |

### 3.2 自定义 train / val / test 切分

HF 原版**没有 val 集合**，本地从 1500 条 train 中**手动抽 300 条**（150 正 + 150 负）作为验证集。

- **`split.py`**：辅助脚本，使用 `random.sample` 随机抽出 150 正 + 150 负的 ID 字典，
  人工固化到 `splits.py` 的 `nexar_val` 中，保证可复现。
- **`splits.py`**：固化的 `nexar_val` 字典（300 条 video_id）。
- **`generate_nexar_splits.py`**：读取 `annotations.csv` + `nexar_val`，
  生成三份对齐 schema 的 csv：

  ```text
  video_id, path, fps, total_frames,
  abnormal_start_frame, accident_occur_frame, have_accident
  ```

  字段重命名：`target → have_accident`，`accident_frame → accident_occur_frame`；
  新增 `path`（如 `train_raw_frames/00822` / `test_raw_frames/00204`），
  方便上游框架直接定位帧目录。

### 3.3 验证集帧级稠密 GT（`generate_nexar_val_references.py`）

为 mAP / 帧级评估提供参考帧集合：每条 val 视频取**末尾 20 帧（间隔 3）**：

```python
if have_accident == 1:
    end = accident_frame
else:
    end = total_frames - 1
start  = end - 57          # 57 = (20-1) * 3
frames = list(range(start, end + 1, 3))   # 恰好 20 帧
```

输出 `nexar_val_references.csv`（共 `300 × 20 = 6000` 行）：

| 列 | 含义 |
| --- | --- |
| `video_id` | 视频 ID |
| `frame_id` | 帧索引 |
| `have_accident` | 该帧的二分类标签（与视频级一致） |

### 3.4 数据清洗 / 辅助脚本

- **`duplicate.py`**：对 `test_raw_frames/<id>/000000.jpg` 缩放至 90×60 灰度图，
  两两计算 SSIM，阈值 > 0.6 判定为重复，结果写入 `score_90.csv`。
- **`match.py`**：把 `score.csv` 中两两相似对**合并成连通簇**，便于批量去重。
- **`count.py`**：统计 `train_raw_frames/<id>` 下 JPG 数量，做完整性校验，
  与 `total_frames` 字段比对。
- **`copy.py`**：取每个 `test_raw_frames/<id>` 的**最后 15 帧**复制到 `test_raw_frames_15/`，
  构造轻量推理子集（注释写 30，实际取 15）。

### 3.5 上游训练统一索引（`index.csv`）

把 nexar 与其他事故预测数据集合并到统一 schema：

```
uid, video_id, frame_dir, total_frames, fps,
have_accident, accident_frame, frame_start_idx, dataset, subset
```

示例：

```
nexar_train_01924, 01924, data/nexar-collision-prediction/train_raw_frames/01924,
1200, 30.0, 0, -1, 0, nexar, train
nexar_train_00822, 00822, data/nexar-collision-prediction/train_raw_frames/00822,
1214, 30.0, 1, 585, 0, nexar, train
```

便于 `accident_anticipation` 训练框架以同一 dataloader 加载多个数据集。

---

## 5. 复现步骤（推荐顺序）

```bash
# 0. 准备：从 HF / Kaggle 下载原始视频到 train/ 与 test/
# 同时把官方 train.csv / test.csv 放到当前目录

# 1. 抽帧 + 生成 annotations.csv
python extract_raw_frames.py

# 2. （可选）随机抽 val 候选；结果已固化到 splits.py
# python split.py

# 3. 切分 train / val / test
python generate_nexar_splits.py

# 4. 生成 val 帧级参考标签
python generate_nexar_val_references.py

# 5. （可选）数据完整性 & 去重
python count.py
python duplicate.py
python match.py

# 6. （可选）末 15 帧轻量子集
python copy.py
```

---

## 6. 字段对应表（HF ↔ 本地）

| HF 字段 | 本地字段 | 转换关系 |
| --- | --- | --- |
| `target` | `have_accident` | 直接重命名 |
| `time_of_event`（秒） | `accident_occur_frame`（帧） | `int(time_of_event * fps)`，负样本 -1 |
| `time_of_alert`（秒） | `abnormal_start_frame`（帧） | `int(time_of_alert * fps)`，负样本 -1 |
| `id` | `video_id` | `str(id).zfill(5)` |
| —（视频本体） | `path` | `train_raw_frames/<id>` 或 `test_raw_frames/<id>` |
| `light_conditions` / `weather` / `scene` / `time_to_accident` | — | **未保留** |

---

## 7. 备注

- 本流程默认处理 30 fps 视频；若部分视频 fps 不为 30（如 30.2、29.9），脚本仍按真实 `cv2.CAP_PROP_FPS` 折算。
- `nexar_val_references.py` 中的 `video_id.zfill(6)` 防御逻辑实际不会触发，仍统一为 5 位。
- `submission2.csv` / `submission_rnn_decoder.csv` 是历史模型推理结果，与数据处理无关。
- `accident_anticipation/splits.py` 是上游训练框架的 splits 文件，本目录通过 `importlib` 动态加载它来共享 `nexar_val` 划分。
