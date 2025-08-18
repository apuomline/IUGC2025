## 项目简介

本仓库为 MICCAI 相关挑战赛（IUGC2025）提交的深度学习项目代码与材料。项目采用 `DenseNet121-UNet` 热力图回归方案进行关键点/目标的定位学习，支持 MixUp 增强与学习率调度。训练流程通过单一配置文件驱动，使用者只需正确设置配置文件即可启动训练与复现实验。

### 主要特性
- **配置驱动**：通过 `codes/config/*.yaml` 完整配置模型、数据、训练与保存策略。
- **可复现实验**：固定随机种子，提供最终使用的模型权重与教师模型权重。
- **热力图监督**：内置高斯热力图生成参数（尺寸、sigma、关键点数）。
- **增强策略**：支持 MixUp 概率与系数动态控制。

---

## 目录结构

```text
.
├─ codes/                        # 源代码
│  ├─ config/                    # 训练/推理配置文件（YAML）
│  │  └─ densenet_121_unet_prob.yaml
│  ├─ models/                    # 模型定义
│  └─ heatmap_train_only_3_mixup_prob_moda.py  # 最终训练脚本
│
├─ datasets/                     # 数据集（训练/验证）
│
├─ pse_se_csvs/                  # 伪标签筛选产物
│  └─ unlabeled_ex_imgs_threshold_60_fliter2.csv  # 最终使用的未标注图像标注文件
│
├─ trained_model_pths/           # 模型权重（训练得到/教师模型）
│
└─ final_test_submmit_files/
   └─ F56/                       # 最终测试提交文件夹
```

---

## 环境与安装
```bash
# 创建环境
conda create -n uni python=3.10
conda activate uni
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
pip install -r requirements.txt
---

## 数据准备

- 将训练/验证数据放置于 `datasets/` 或自定义目录。
- 将对应的 `train_csv`、`val_csv` 及图像目录路径配置到 YAML 配置中。
- 伪标签相关文件位于 `pse_se_csvs/`，最终使用的未标注图像标注文件为：
  - `pse_se_csvs/unlabeled_ex_imgs_threshold_60_fliter2.csv`

CSV 的字段格式请与实际代码实现保持一致（通常包含图像路径/文件名与标注信息）。若你的 CSV 在其他目录，请在配置文件中相应修改路径。

---

## 配置文件说明（示例）

以 `codes/config/densenet_121_unet_prob.yaml` 为例，核心字段说明：

```yaml
model:
  name: 'densenet_unet'
  arch: 'densenet121'

data:
  train_csv: 'pse_training_inputs\pse_825_se_100_le_220_unimatched_37\train_320.csv'
  train_dir: 'pse_training_inputs\pse_825_se_100_le_220_unimatched_37\train_imgs'
  val_csv:   'pse_training_inputs\pse_825_se_100_le_220_unimatched_37\val_80.csv'
  val_dir:   'pse_training_inputs\pse_825_se_100_le_220_unimatched_37\val_imgs'

heatmap:
  size: 64          # 热力图尺寸
  sigma: 6.0        # 高斯核 sigma
  num_keypoints: 3  # 关键点数量

training:
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 1e-4
  epochs: 150
  seed: 42

scheduler:
  type: 'StepLR'
  step_size: 10
  gamma: 0.90
  patience: 3
  min_lr: 1e-6

save:
  dir: 'results_heatmap_train_only'
  interval: 50
  model_suffix: 'mixup_prob_pse825_100_le_220_10_01'
  timestamp: false

mixup_alpha: 0.20
mixup_prob: 1.0
mixup_final_prob: 0.1
```

- **model**: 模型名称与主干网络架构。
- **data**: 训练/验证 CSV 与图像目录路径（相对或绝对路径均可）。
- **heatmap**: 热力图监督参数（尺寸、sigma、关键点个数）。
- **training**: 批大小、初始学习率、权重衰减、训练轮数、随机种子等。
- **scheduler**: 学习率调度策略与其超参（StepLR/ReduceLROnPlateau/MultiStepLR/CosineAnnealingLR）。
- **save**: 模型保存目录、保存间隔、文件后缀与是否追加时间戳。
- **mixup_***: MixUp 增强的初始概率与训练后期概率、alpha 系数等。

> 只需根据你的数据与实验需求修改该 YAML，即可直接启动训练。

---

## 训练

默认训练入口脚本：`codes/heatmap_train_only_3_mixup_prob_moda.py`

从仓库根目录启动（推荐）：

```bash
# 方式一：显式指定配置文件路径
python codes/heatmap_train_only_3_mixup_prob_moda.py --config codes/config/densenet_121_unet_prob.yaml

# 方式二：脚本内部有默认配置时，可直接运行（如实现支持）
python codes/heatmap_train_only_3_mixup_prob_moda.py
```

或进入 `codes/` 目录后运行：

```bash
cd codes
python heatmap_train_only_3_mixup_prob_moda.py --config config\densenet_121_unet_prob.yaml
```

训练过程中的日志与权重会保存至配置中的 `save.dir`，并按 `save.interval` 周期性写入；最终模型文件名将包含 `save.model_suffix`。

---

## 模型权重

- `trained_model_pths/` 内包含：
  - **最终推理所用权重**：用于生成提交结果；
  - **初始教师模型权重**：用于伪标签筛选阶段。

如需从头训练，可忽略该目录；如需复现实验或直接推理，请在推理脚本/配置中指向相应权重路径。

---

## 推理与最终提交

- 最终测试提交文件夹：`final_test_submmit_files/F56/`
  - 按竞赛平台/评测环境要求，将该文件夹及所需权重打包/上传；
  - 如需本地推理，请参照该目录内的脚本与说明（将权重路径指向 `trained_model_pths/` 中的最终模型）。

若仓库包含独立的推理脚本/入口（例如 `inference.py` 或提交打包脚本），请在运行前确认配置中数据与权重路径的正确性。

---

## 复现实验与随机性

- 默认随机种子：`training.seed = 42`。
- 为了可复现，建议固定：随机数种子、CUDA 相关环境、数据划分与依赖版本。

---

---

## 致谢与许可证

- 本项目基于 PyTorch 等开源组件，感谢其社区贡献。
- https://github.com/google-research/noisystudent
- 许可证（License）：请根据仓库实际 `LICENSE` 文件为准；如未提供，默认保留所有权利（All rights reserved）。

---

## 联系方式

如有问题或合作意向，欢迎在 Issues 中反馈或直接联系项目维护者。
