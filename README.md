## Project Overview

This repository contains the deep learning project and materials submitted to a MICCAI-related challenge (IUGC2025). The project adopts a `DenseNet121-UNet` heatmap regression scheme for keypoint/target localization, with MixUp augmentation and learning-rate scheduling. The entire training pipeline is configuration-driven: set the YAML config properly and you can start training and reproduce results.

### Key Features
- **Config-driven**: Everything is controlled by `codes/config/*.yaml` (model, data, training, saving, etc.).
- **Reproducible experiments**: Fixed random seed; final model weights and teacher model weights are provided.
- **Heatmap supervision**: Built-in Gaussian heatmap generation (size, sigma, number of keypoints).
- **Augmentation**: MixUp with configurable probability and alpha scheduling.

---

## Directory Structure

```text
.
├─ codes/                        # Source code
│  ├─ config/                    # Training/inference configs (YAML)
│  │  └─ densenet_121_unet_prob.yaml
│  ├─ models/                    # Model definitions
│  └─ heatmap_train_only_3_mixup_prob_moda.py  # Final training script
│
├─ datasets/                     # Datasets (train/val)
│
├─ pse_se_csvs/                  # Pseudo-label filtering outputs
│  └─ unlabeled_ex_imgs_threshold_60_fliter2.csv  # Final annotations for unlabeled images
│
├─ trained_model_pths/           # Model weights (final and teacher)
│
└─ final_test_submmit_files/
   └─ F56/                       # Final test submission folder
```

---

## Environment & Installation

```bash
conda create -n uni python=3.10 -y
conda activate uni
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Data Preparation

- Place training/validation data under `datasets/` or a custom directory.
- Set `train_csv`, `val_csv`, and image directories in the YAML config to match your data.
- Pseudo-label related files are under `pse_se_csvs/`. The final annotation file for unlabeled images is:
  - `pse_se_csvs/unlabeled_ex_imgs_threshold_60_fliter2.csv`

Ensure your CSV schema matches the code expectations (typically includes image path/filename and annotations). If your CSVs live elsewhere, update the config paths accordingly.

---

## Config File Overview (Example)

Take `codes/config/densenet_121_unet_prob.yaml` as an example:

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
  size: 64          # Heatmap size
  sigma: 6.0        # Gaussian sigma
  num_keypoints: 3  # Number of keypoints

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

- **model**: Model name and backbone architecture.
- **data**: Train/val CSV files and image directories (relative or absolute paths).
- **heatmap**: Supervision parameters (size, sigma, number of keypoints).
- **training**: Batch size, initial learning rate, weight decay, epochs, random seed, etc.
- **scheduler**: LR scheduler type and hyperparameters (StepLR/ReduceLROnPlateau/MultiStepLR/CosineAnnealingLR).
- **save**: Output directory, checkpoint interval, filename suffix, and whether to append timestamps.
- **mixup_***: MixUp probabilities and alpha scheduling.

> Update the YAML to match your data and experimental needs, then start training directly.

---

## Training

Main training entry script: `codes/heatmap_train_only_3_mixup_prob_moda.py`

From repository root (recommended):

```bash
# Option 1: Specify config path explicitly
python codes/heatmap_train_only_3_mixup_prob_moda.py --config codes/config/densenet_121_unet_prob.yaml

# Option 2: If the script has a default config, run directly (if supported)
python codes/heatmap_train_only_3_mixup_prob_moda.py
```

Or from within the `codes/` directory:

```bash
cd codes
python heatmap_train_only_3_mixup_prob_moda.py --config config\densenet_121_unet_prob.yaml
```

Logs and checkpoints will be saved under `save.dir` as specified in the config, at the frequency of `save.interval`. Final model filenames will include `save.model_suffix`.

---

## Model Weights

- Under `trained_model_pths/` you will find:
  - **Final weights for inference**: used to produce submission results.
  - **Initial teacher model weights**: used during pseudo-label filtering.

If training from scratch, you may ignore this directory. For reproduction or direct inference, point your scripts/config to the appropriate paths here.

---

## Inference & Final Submission

- Final submission folder: `final_test_submmit_files/F56/`
  - Package/upload this folder along with required weights according to the competition platform/evaluator.
  - For local inference, follow the scripts/instructions in this folder (point the weight path to the final model under `trained_model_pths/`).

If there is a dedicated inference or packaging script (e.g., `inference.py`), verify all data and weight paths before running.

---

## Reproducibility

- Default random seed: `training.seed = 42`.
- For better reproducibility, fix seeds, CUDA/cudnn settings, dataset splits, and dependency versions.

---


## License & Acknowledgements

- Built upon PyTorch and other open-source components—thanks to their communities.
- Reference and inspiration: [Noisy Student (Google Research)](https://github.com/google-research/noisystudent)
- License: please refer to the repository `LICENSE`. If absent, all rights reserved by default.

---

## Contact

For questions or collaboration, please open an issue or contact the maintainers directly. 
