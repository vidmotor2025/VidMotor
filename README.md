# VidMotor

This repository provides the official implementation of the paper:

**A video-based motor function foundation model for universal and widely-available medical assessment**

All materials released in this repository can **ONLY** be used for **RESEARCH** purposes and not for commercial applications.

The authors' institution preserves the copyright and all legal rights of these codes.

# Table of Contents

- [System Requirements and Installation](#system-requirements-and-installation)
- [Dataset Preparation (MedMOT-40K)](#dataset-preparation-medmot-40k)
- [VidMotor Pretraining (Default Backbone Architecture)](#vidmotor-pretraining-default-backbone-architecture)
- [VidMotor Fine-tuning (Default Backbone Architecture)](#vidmotor-fine-tuning-default-backbone-architecture)
  - [1) Direct fine-tuning](#1-direct-fine-tuning)
  - [2) Fine-tuning in data-scarce scenarios](#2-fine-tuning-in-data-scarce-scenarios)
  - [3) Measuring inference time](#3-measuring-inference-time)
- [VidMotor's Generality across Other Alternative Backbone Architectures (Model Zoo)](#vidmotors-generality-across-other-alternative-backbone-architectures-model-zoo)
  - [1) CTR-GCN](#1-ctr-gcn)
  - [2) PoseFormerV2](#2-poseformerv2)
  - [3) SkateFormer](#3-skateformer)
  - [4) DSTformer](#4-dstformer)
- [Comparative Approaches](#comparative-approaches)
  - [1) SkateFormer](#1-skateformer-1)
  - [2) HD-GCN](#2-hd-gcn)
  - [3) FR-Head](#3-fr-head)
  - [4) MotionBERT](#4-motionbert)
  - [5) SkeleT-GCN](#5-skelet-gcn)
  - [6) MMN](#6-mmn)
  - [7) ProtoGCN](#7-protogcn)
- [Reproducibility of Our Results](#reproducibility-of-our-results)
- [Figures in Our Paper](#figures-in-our-paper)


# System Requirements and Installation

- **Hardware requirements:** Our method was implemented on NVIDIA GeForce RTX 3090 Ti GPUs (24 GB memory) and has also been tested on NVIDIA GeForce RTX 4090 and RTX 2080 Ti GPUs.

- **Operating system requirements:** The code has been tested on the system of Linux (Ubuntu 20.04).

- **Dependencies:** Our method was developed using **Python 3.10.13** and **PyTorch 2.0.1 with CUDA 11.8**. Other key Python packages include NumPy and PyYAML.

- **Environment setup:** You can easily reproduce the environment using Conda:
  ```shell
  conda env create -f environment.yaml
  conda activate pytorch
  ```

# Dataset Preparation (MedMOT-40K)

VidMotor was pretrained and fine-tuned on 39,319 structured cases aggregated from 36 publicly available datasets, standardized into skeleton-based representations.

To download our preprocessed structured dataset, access original data sources, or learn about our data preprocessing guideline, please refer to [README_data.md](data/README_data.md) in `./data`.

- Directory structure of `./data`:
  
  ```textile
  ./data
  ├── Pretrain          # Datasets for VidMotor pretraining
  │   ├── data_and_label_<dataset_name>.pkl
  │   └── ...
  ├── Finetune          # Datasets for VidMotor fine-tuning and validation
  │   ├── data_and_label_<dataset_name>.pkl
  │   └── ...
  ├── Preprocessing     # Preprocessing scripts for datasets
  │   ├── <dataset_name>.py
  │   └── ...
  └── README_data.md    # Dataset access and processing guide
  ```

# VidMotor Pretraining (Default Backbone Architecture)

Main scripts are located in `./VidMotor`. Run:

```bash
python VidMotor/main_ST-GCN_pretrain.py --config VidMotor/config/Pretrain_ST-GCN/Pretrain.yaml \
                                        --work-dir ./results/VidMotor_pretrain \
                                        --device 0 1
```

> 💡Notes:
> 
> - Please carefully read [README_data.md](data/README_data.md) before running. Ensure that all datasets are properly stored in `./data/Pretrain`, except the 3D-Yoga, KIMORE, and UI-PRMD datasets, which require additional permission from their corresponding authors.
> - The above pretraining may require two GPUs to ensure efficient training.
> - **We provide the pretrained model weights, `pretrain-ST-GCN.pt`, in `./VidMotor`, which can be directly used for fine-tuning.**

- Example output:
  
  ```textile
  ./results/VidMotor_pretrain
  ├── config.yaml               # Configuration file
  ├── log.txt                   # Pretraining log
  └── epoch-<epoch_number>.pt   # Pretrained model weights
  ```

# VidMotor Fine-tuning (Default Backbone Architecture)

Pretrained weights and main scripts are located in `./VidMotor`. 

## 1) Direct fine-tuning

To fine-tune VidMotor on a specific dataset, run (example: SSBD-Line-Gait dataset):

```bash
python VidMotor/main_ST-GCN_finetune.py --config VidMotor/config/Finetune_ST-GCN/SSBD-Line-Gait.yaml \
                                        --work-dir ./results/VidMotor_finetune \
                                        --model-pretrain VidMotor/pretrain-ST-GCN.pt \
                                        --device 0
```

> 💡Notes:
> 
> - Please carefully read [README_data.md](data/README_data.md) before running. Ensure that the dataset is properly stored in `./data/Finetune`.
> - The fine-tuning only requires one GPU to ensure efficient training.
> - This direct fine-tuning supports tasks in abnormality recognition, clinical diagnosis and severity grading, and physical rehabilitation assessment.
> - The above is only for the SSBD-Line-Gait dataset example. To fine-tune on other datasets, replace the YAML configuration file specified in the command-line arguments with the corresponding file in `./VidMotor/config/Finetune`.

- Example output:
  
  ```textile
  <work_dir>/<dataset_name>
  ├── config.yaml                     # Configuration file
  ├── log.txt                         # Fine-tuning log
  ├── epoch-<epoch_number>.pt         # Fine-tuned model checkpoint
  └── epoch<epoch_number>_score.pkl   # Model prediction outputs
  ```

To fine-tune VidMotor on a single-center dataset and test it on multi-center datasets, run (example: PDMC-Arising-from-Chair dataset):

```bash
python VidMotor/main_ST-GCN_finetune_multicenter.py --config VidMotor/config/Finetune_ST-GCN/PDMC-Arising-from-Chair.yaml \
                                                    --work-dir ./results/VidMotor_finetune_multicenter \
                                                    --model-pretrain VidMotor/pretrain-ST-GCN.pt \
                                                    --device 0
```

> 💡Notes:
> 
> - You can use your own multi-center datasets for this process.
> - Ensure that your multi-center datasets are properly stored in `./data/Finetune_multicenter` before running, and that the dataset paths in `./config/Finetune/PDMC-Arising-from-Chair.yaml` are correctly specified.

- Example output:
  
  ```textile
  <work_dir>/<dataset_name>
  ├── config.yaml                            # Configuration file
  ├── log.txt                                # Fine-tuning log
  ├── epoch-<epoch_number>.pt                # Fine-tuned model checkpoint
  ├── <dataset_name of center-1>             # Model prediction outputs of center-1
  │                 └── epoch<epoch_number>_score.pkl
  ├── <dataset_name of center-2>             # Model prediction outputs of center-2
  │                 └── epoch<epoch_number>_score.pkl
  ├── <dataset_name of center-3>             # Model prediction outputs of center-3
  │                 └── epoch<epoch_number>_score.pkl
  └── <dataset_name of center-4>             # Model prediction outputs of center-4
                    └── epoch<epoch_number>_score.pkl
  ```

## 2) Fine-tuning in data-scarce scenarios

To fine-tune VidMotor on a specific dataset in data-scarce scenarios, run (example: SSBD-Line-Gait dataset with 10% training data for fine-tuning):

```bash
python VidMotor/main_ST-GCN_finetune_ratio.py --config VidMotor/config/Finetune_ST-GCN/SSBD-Line-Gait.yaml \
                                              --work-dir ./results/VidMotor_finetune \
                                              --model-pretrain VidMotor/pretrain-ST-GCN.pt \
                                              --train-ratio 0.1 \
                                              --device 0
```

To fine-tune VidMotor on a single-center dataset in data-scarce scenarios and test it on multi-center datasets, run (example: PDMC-Arising-from-Chair dataset with 10% training data for fine-tuning):

```bash
python VidMotor/main_ST-GCN_finetune_multicenter_ratio.py --config VidMotor/config/Finetune_ST-GCN/PDMC-Arising-from-Chair.yaml \
                                                          --work-dir ./results/VidMotor_finetune_multicenter \
                                                          --model-pretrain VidMotor/pretrain-ST-GCN.pt \
                                                          --train-ratio 0.1 \
                                                          --device 0
```

- Example output: `<work_dir>/<dataset_name>_training-data0.1`, with a structure similar to that of direct fine-tuning.

## 3) Measuring inference time

To measure the per-sample inference time of VidMotor on a specific dataset, run (example: SSBD-Line-Gait dataset):

```bash
python VidMotor/main_ST-GCN_finetune_time.py --config VidMotor/config/Finetune_ST-GCN/SSBD-Line-Gait.yaml \
                                             --work-dir ./results/VidMotor_finetune \
                                             --model-pretrain VidMotor/pretrain-ST-GCN.pt \
                                             --device 0
```


To measure the per-sample inference time of VidMotor on multi-center datasets, run (example: PDMC-Arising-from-Chair dataset):

```bash
python VidMotor/main_ST-GCN_finetune_multicenter_time.py --config VidMotor/config/Finetune_ST-GCN/PDMC-Arising-from-Chair.yaml \
                                                         --work-dir ./results/VidMotor_finetune_multicenter \
                                                         --model-pretrain VidMotor/pretrain-ST-GCN.pt \
                                                         --device 0
```

- Example output:
  
  ```textile
  <work_dir>/<dataset_name>_inference-time
  ├── config.yaml                            # Configuration file
  ├── log.txt                                # Fine-tuning log for recording per-sample inference time
  └── epoch-<epoch_number>.pt                # Fine-tuned model checkpoint
  ```

# VidMotor's Generality across Other Alternative Backbone Architectures (Model Zoo)

VidMotor supports other alternative backbone architecture options (ST-GCN architecture serving as the default backbone): [CTR-GCN, PoseFormerV2, SkateFormer, DSTformer].

Main scripts are located in `./VidMotor`. The pretrained weights can be downloaded at the following link, except for those corresponding to the ST-GCN architecture, which are already included in `./VidMotor`:

https://figshare.com/s/985cfbfe791a60a0f04a

The downloaded weights should also be placed in `./VidMotor`.

|                   Backbone architecture                    |                           Pretrained weights                            |         Pretraining script         | Pretraining config directory  |                              Fine-tuning scripts                               | Fine-tuning config directory  |
|:----------------------------------------------------------:|:-----------------------------------------------------------------------:|:----------------------------------:|:-----------------------------:|:------------------------------------------------------------------------------:|:-----------------------------:|
|     [ST-GCN](https://github.com/open-mmlab/mmskeleton)     |         [pretrain-ST-GCN.pt](VidMotor/pretrain-ST-GCN.pt)               |      main_ST-GCN_pretrain.py       |    config/Pretrain_ST-GCN/    |        main_ST-GCN_finetune.py,<br/>main_ST-GCN_finetune_multicenter.py        |    config/Finetune_ST-GCN/    |
|    [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)        |   [pretrain-CTR-GCN.pt](https://figshare.com/s/985cfbfe791a60a0f04a)    |      main_CTR-GCN_pretrain.py      |       config/Pretrain_CTR-GCN/       |       main_CTR-GCN_finetune.py,<br/>main_CTR-GCN_finetune_multicenter.py       |   config/Finetune_CTR-GCN/    |
| [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2)  | [pretrain-PoseFormerV2.pt](https://figshare.com/s/985cfbfe791a60a0f04a) |   main_PoseFormerV2_pretrain.py    | config/Pretrain_PoseFormerV2/ |  main_PoseFormerV2_finetune.py,<br/>main_PoseFormerV2_finetune_multicenter.py  | config/Finetune_PoseFormerV2/ |
| [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer) | [pretrain-SkateFormer.pt](https://figshare.com/s/985cfbfe791a60a0f04a)  |    main_SkateFormer_pretrain.py    |   config/Pretrain_SkateFormer/    |   main_SkateFormer_finetune.py,<br/>main_SkateFormer_finetune_multicenter.py   | config/Finetune_SkateFormer/  |
|   [DSTformer](https://github.com/Walter0807/MotionBERT)    |  [pretrain-DSTformer.pt](https://figshare.com/s/985cfbfe791a60a0f04a)   |         main_DSTformer_pretrain.py |     config/Pretrain_DSTformer/      |     main_DSTformer_finetune.py,<br/>main_DSTformer_finetune_multicenter.py     |  config/Finetune_DSTformer/   |


## 1) CTR-GCN

```bash
# Pretraining
python VidMotor/main_CTR-GCN_pretrain.py --config VidMotor/config/Pretrain_CTR-GCN/Pretrain.yaml --work-dir ./results/Backbone/CTR-GCN_pretrain --device 0 1 2
# Ensure the pretrained weights have been placed in `./VidMotor`.
# Fine-tuning on SSBD-Line-Gait dataset
python VidMotor/main_CTR-GCN_finetune.py --config VidMotor/config/Finetune_CTR-GCN/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/CTR-GCN_finetune --model-pretrain VidMotor/pretrain-CTR-GCN.pt --device 0
python VidMotor/main_CTR-GCN_finetune_time.py --config VidMotor/config/Finetune_CTR-GCN/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/CTR-GCN_finetune --model-pretrain VidMotor/pretrain-CTR-GCN.pt --device 0
# Fine-tuning on PDMC-Arising-from-Chair dataset
python VidMotor/main_CTR-GCN_finetune_multicenter.py --config VidMotor/config/Finetune_CTR-GCN/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/CTR-GCN_finetune_multicenter --model-pretrain VidMotor/pretrain-CTR-GCN.pt --device 0
python VidMotor/main_CTR-GCN_finetune_multicenter_time.py --config VidMotor/config/Finetune_CTR-GCN/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/CTR-GCN_finetune_multicenter --model-pretrain VidMotor/pretrain-CTR-GCN.pt --device 0
```

## 2) PoseFormerV2

```bash
# Pretraining
python VidMotor/main_PoseFormerV2_pretrain.py --config VidMotor/config/Pretrain_PoseFormerV2/Pretrain.yaml --work-dir ./results/Backbone/PoseFormerV2_pretrain --device 0
# Ensure the pretrained weights have been placed in `./VidMotor`.
# Fine-tuning on SSBD-Line-Gait dataset
python VidMotor/main_PoseFormerV2_finetune.py --config VidMotor/config/Finetune_PoseFormerV2/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/PoseFormerV2_finetune --model-pretrain VidMotor/pretrain-PoseFormerV2.pt --device 0
python VidMotor/main_PoseFormerV2_finetune_time.py --config VidMotor/config/Finetune_PoseFormerV2/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/PoseFormerV2_finetune --model-pretrain VidMotor/pretrain-PoseFormerV2.pt --device 0
# Fine-tuning on PDMC-Arising-from-Chair dataset
python VidMotor/main_PoseFormerV2_finetune_multicenter.py --config VidMotor/config/Finetune_PoseFormerV2/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/PoseFormerV2_finetune_multicenter --model-pretrain VidMotor/pretrain-PoseFormerV2.pt --device 0
python VidMotor/main_PoseFormerV2_finetune_multicenter_time.py --config VidMotor/config/Finetune_PoseFormerV2/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/PoseFormerV2_finetune_multicenter --model-pretrain VidMotor/pretrain-PoseFormerV2.pt --device 0
```

## 3) SkateFormer

```bash
# Pretraining
python VidMotor/main_SkateFormer_pretrain.py --config VidMotor/config/Pretrain_SkateFormer/Pretrain.yaml --work-dir ./results/Backbone/SkateFormer_pretrain --device 0 1 2 3
# Ensure the pretrained weights have been placed in `./VidMotor`.
# Fine-tuning on SSBD-Line-Gait dataset
python VidMotor/main_SkateFormer_finetune.py --config VidMotor/config/Finetune_SkateFormer/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/SkateFormer_finetune --model-pretrain VidMotor/pretrain-SkateFormer.pt --device 0
python VidMotor/main_SkateFormer_finetune_time.py --config VidMotor/config/Finetune_SkateFormer/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/SkateFormer_finetune --model-pretrain VidMotor/pretrain-SkateFormer.pt --device 0
# Fine-tuning on PDMC-Arising-from-Chair dataset
python VidMotor/main_SkateFormer_finetune_multicenter.py --config VidMotor/config/Finetune_SkateFormer/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/SkateFormer_finetune_multicenter --model-pretrain VidMotor/pretrain-SkateFormer.pt --device 0
python VidMotor/main_SkateFormer_finetune_multicenter_time.py --config VidMotor/config/Finetune_SkateFormer/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/SkateFormer_finetune_multicenter --model-pretrain VidMotor/pretrain-SkateFormer.pt --device 0
```

## 4) DSTformer

```bash
# Pretraining
python VidMotor/main_DSTformer_pretrain.py --config VidMotor/config/Pretrain_DSTformer/Pretrain.yaml --work-dir ./results/Backbone/DSTformer_pretrain --device 0 1 2 3 4 5 6 7
# Ensure the pretrained weights have been placed in `./VidMotor`.
# Fine-tuning on SSBD-Line-Gait dataset
python VidMotor/main_DSTformer_finetune.py --config VidMotor/config/Finetune_DSTformer/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/DSTformer_finetune --model-pretrain VidMotor/pretrain-DSTformer.pt --device 0
python VidMotor/main_DSTformer_finetune_time.py --config VidMotor/config/Finetune_DSTformer/SSBD-Line-Gait.yaml --work-dir ./results/Backbone/DSTformer_finetune --model-pretrain VidMotor/pretrain-DSTformer.pt --device 0
# Fine-tuning on PDMC-Arising-from-Chair dataset
python VidMotor/main_DSTformer_finetune_multicenter.py --config VidMotor/config/Finetune_DSTformer/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/DSTformer_finetune_multicenter --model-pretrain VidMotor/pretrain-DSTformer.pt --device 0
python VidMotor/main_DSTformer_finetune_multicenter_time.py --config VidMotor/config/Finetune_DSTformer/PDMC-Arising-from-Chair.yaml --work-dir ./results/Backbone/DSTformer_finetune_multicenter --model-pretrain VidMotor/pretrain-DSTformer.pt --device 0
```
All output structures are similar to those of VidMotor, as shown below:

- Pretraining output: `<work_dir>`.

- Fine-tuning example output: `<work_dir>/<dataset_name>`.



# Comparative Approaches

Comparative approach options in our paper: [SkateFormer, HD-GCN, FR-Head, MotionBERT, SkeleT-GCN, MMN, ProtoGCN].
Main scripts are located in `./SOTA`.


|                   Backbone architecture                    | Script directory |                             Run scripts                              |
|:----------------------------------------------------------:|:----------------:|:--------------------------------------------------------------------:|
| [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer) |  SkateFormer/    |         main_SkateFormer.py, main_SkateFormer_multicenter.py         |
|       [HD-GCN](https://github.com/Jho-Yonsei/HD-GCN)       |     HD-GCN/      |              main_HD-GCN.py, main_HD-GCN_multicenter.py              |
|      [FR-Head](https://github.com/zhysora/FR-Head)         |     FR-Head/     |             main_FR-Head.py, main_FR-Head_multicenter.py             |
|   [MotionBERT](https://github.com/Walter0807/MotionBERT)   |   MotionBERT/    | main_MotionBERT_finetune.py, main_MotionBERT_finetune_multicenter.py |
|    [SkeleT-GCN](https://github.com/YijieYang23/PSE-GCN)    |   SkeleT-GCN/    |          main_SkeleT-GCN.py, main_SkeleT-GCN_multicenter.py          |
|          [MMN](https://github.com/momiji-bit/MMN)          |       MMN/       |                 main_MMN.py, main_MMN_multicenter.py                 |
|     [ProtoGCN](https://github.com/firework8/ProtoGCN)      |    ProtoGCN/     |            main_ProtoGCN.py, main_ProtoGCN_multicenter.py            |


## 1) SkateFormer

```bash
# Running on SSBD-Line-Gait dataset
python SOTA/SkateFormer/main_SkateFormer.py --config SOTA/SkateFormer/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/SkateFormer --device 0
python SOTA/SkateFormer/main_SkateFormer_time.py --config SOTA/SkateFormer/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/SkateFormer --device 0
# Running on PDMC-Arising-from-Chair dataset
python SOTA/SkateFormer/main_SkateFormer_multicenter.py --config SOTA/SkateFormer/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/SkateFormer_multicenter --device 0
python SOTA/SkateFormer/main_SkateFormer_multicenter_time.py --config SOTA/SkateFormer/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/SkateFormer_multicenter --device 0
```


## 2) HD-GCN

```bash
# Running on SSBD-Line-Gait dataset
python SOTA/HD-GCN/main_HD-GCN.py --config SOTA/HD-GCN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/HD-GCN --device 0
python SOTA/HD-GCN/main_HD-GCN_time.py --config SOTA/HD-GCN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/HD-GCN --device 0
# Running on PDMC-Arising-from-Chair dataset
python SOTA/HD-GCN/main_HD-GCN_multicenter.py --config SOTA/HD-GCN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/HD-GCN_multicenter --device 0
python SOTA/HD-GCN/main_HD-GCN_multicenter_time.py --config SOTA/HD-GCN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/HD-GCN_multicenter --device 0
```

## 3) FR-Head

```bash
# Running on SSBD-Line-Gait dataset
python SOTA/FR-Head/main_FR-Head.py --config SOTA/FR-Head/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/FR-Head --device 0
python SOTA/FR-Head/main_FR-Head_time.py --config SOTA/FR-Head/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/FR-Head --device 0
# Running on PDMC-Arising-from-Chair dataset
python SOTA/FR-Head/main_FR-Head_multicenter.py --config SOTA/FR-Head/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/FR-Head_multicenter --device 0
python SOTA/FR-Head/main_FR-Head_multicenter_time.py --config SOTA/FR-Head/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/FR-Head_multicenter --device 0
```

## 4) MotionBERT

The pretrained weights of MotionBERT ('latest_epoch.bin') can be obtained from the following link shared by its contributors, and should be placed in `./SOTA/MotionBERT` before running: 

https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2

```bash
# Ensure the pretrained weights have been placed in `./SOTA/MotionBERT`.
# Fine-tuning on SSBD-Line-Gait dataset
python SOTA/MotionBERT/main_MotionBERT_finetune.py --config SOTA/MotionBERT/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/MotionBERT --model-pretrain SOTA/MotionBERT/latest_epoch.bin --device 0
python SOTA/MotionBERT/main_MotionBERT_finetune_time.py --config SOTA/MotionBERT/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/MotionBERT --model-pretrain SOTA/MotionBERT/latest_epoch.bin --device 0
# Fine-tuning on PDMC-Arising-from-Chair dataset
python SOTA/MotionBERT/main_MotionBERT_finetune_multicenter.py --config SOTA/MotionBERT/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/MotionBERT_multicenter --model-pretrain SOTA/MotionBERT/latest_epoch.bin --device 0
python SOTA/MotionBERT/main_MotionBERT_finetune_multicenter_time.py --config SOTA/MotionBERT/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/MotionBERT_multicenter --model-pretrain SOTA/MotionBERT/latest_epoch.bin --device 0
```

## 5) SkeleT-GCN

```bash
# Running on SSBD-Line-Gait dataset
python SOTA/SkeleT-GCN/main_SkeleT-GCN.py --config SOTA/SkeleT-GCN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/SkeleT-GCN --device 0
python SOTA/SkeleT-GCN/main_SkeleT-GCN_time.py --config SOTA/SkeleT-GCN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/SkeleT-GCN --device 0
# Running on PDMC-Arising-from-Chair dataset
python SOTA/SkeleT-GCN/main_SkeleT-GCN_multicenter.py --config SOTA/SkeleT-GCN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/SkeleT-GCN_multicenter --device 0
python SOTA/SkeleT-GCN/main_SkeleT-GCN_multicenter_time.py --config SOTA/SkeleT-GCN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/SkeleT-GCN_multicenter --device 0
```

## 6) MMN

```bash
# Running on SSBD-Line-Gait dataset
python SOTA/MMN/main_MMN.py --config SOTA/MMN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/MMN --device 0
python SOTA/MMN/main_MMN_time.py --config SOTA/MMN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/MMN --device 0
# Running on PDMC-Arising-from-Chair dataset
python SOTA/MMN/main_MMN_multicenter.py --config SOTA/MMN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/MMN_multicenter --device 0
python SOTA/MMN/main_MMN_multicenter_time.py --config SOTA/MMN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/MMN_multicenter --device 0
```

## 7) ProtoGCN

```bash
# Running on SSBD-Line-Gait dataset
python SOTA/ProtoGCN/main_ProtoGCN.py --config SOTA/ProtoGCN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/ProtoGCN --device 0
python SOTA/ProtoGCN/main_ProtoGCN_time.py --config SOTA/ProtoGCN/config/SSBD-Line-Gait.yaml --work-dir ./results/SOTA/ProtoGCN --device 0
# Running on PDMC-Arising-from-Chair dataset
python SOTA/ProtoGCN/main_ProtoGCN_multicenter.py --config SOTA/ProtoGCN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/ProtoGCN_multicenter --device 0
python SOTA/ProtoGCN/main_ProtoGCN_multicenter_time.py --config SOTA/ProtoGCN/config/PDMC-Arising-from-Chair.yaml --work-dir ./results/SOTA/ProtoGCN_multicenter --device 0
```
All output structures are similar to those of VidMotor, as shown below:

- Running example output: `<work_dir>/<dataset_name>`.


# Reproducibility of Our Results

To reproduce the reported performance results of **VidMotor and all comparative approaches** across the available datasets, run:

```bash
python reproducibility/run_VidMotor_and_SOTA_reproducibility.py
```

- Expected output: classification metrics saved to `./results/reproduce_VidMotor_and_SOTA_results.xlsx`.

- To visualize the comparison results through figures, run (ensure that the reproduction process has been completed and `./results/reproduce_VidMotor_and_SOTA_results.xlsx` has been generated):
  
  ```bash
  cd plot_figures
  python Fig2.py --file-path ../results/reproduce_VidMotor_and_SOTA_results.xlsx --save-dir ./figures/reproduce_figures/Fig2
  python Fig3_4.py --file-path ../results/reproduce_VidMotor_and_SOTA_results.xlsx --save-dir ./figures/reproduce_figures
  ```
  - Expected output: generated figures saved to `./plot_figures/figures/reproduce_figures/Fig2`, `./plot_figures/figures/reproduce_figures/Fig3`, and `./plot_figures/figures/reproduce_figures/Fig4`.

To reproduce the reported performance results of **VidMotor in data-scarce scenarios** across the available datasets, as well as the reported **inference-time results** of VidMotor and all comparative approaches, run:

```bash
# By default, VidMotor is evaluated using 10% training data for fine-tuning
python reproducibility/run_ratio_reproducibility.py
python reproducibility/run_time_VidMotor_and_SOTA_reproducibility.py
```

- Expected output: classification metrics saved to `./results/reproduce_results_10%training.xlsx`, per-sample inference time saved to `./results/reproduce_results_runtime.xlsx`.
- To visualize the performance of VidMotor fine-tuned with only 10% of the data versus other approaches trained on 100%, as well as to compare per-sample inference time through figures, run (ensure that the reproduction process has been completed and `./results/reproduce_results_10%training.xlsx` and `./results/reproduce_results_runtime.xlsx` have been generated):
  
  ```bash
  cd plot_figures
  python Fig6.py --file-path-all ../results/reproduce_VidMotor_and_SOTA_results.xlsx --file-path-ours ../results/reproduce_results_10%training.xlsx --file-path-multicenter None --file-path-runtime ../results/reproduce_results_runtime.xlsx --save-dir ./figures/reproduce_figures/Fig6
  ```
  - If you have not conducted experiments on your own private multi-center datasets, the figures corresponding to the multi-center application scenario will be empty.
  - Expected output: generated figures saved to `./plot_figures/figures/reproduce_figures/Fig6`.

To reproduce the reported performance results and inference-time results of **VidMotor with different backbone architectures** across the available datasets, run:

```bash
python reproducibility/run_VidMotor_with_backbones_reproducibility.py
python reproducibility/run_time_VidMotor_with_backbones_reproducibility.py
```

- Expected output: classification metrics saved to `./results/reproduce_VidMotor_with_backbones_results.xlsx`, per-sample inference time saved to `./results/reproduce_VidMotor_with_backbones_results_runtime.xlsx`.

# Figures in Our Paper

To reproduce Figures 2–6 in our paper, run:

```bash
cd plot_figures
python Fig2.py 
python Fig3_4.py
python Fig5.py
python Fig6.py
```

- Original result files used by the above scripts are stored in `./plot_figures/results_examples`.
- Expected output: generated figures saved to `./plot_figures/figures/examples`.
