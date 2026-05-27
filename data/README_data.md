# MedMOT-40K Dataset Access Guide

This document provides detailed guidance on downloading the preprocessed structured dataset, accessing the original data sources, and following preprocessing guidelines for the VidMotor project.

# Download Instructions for the Structured Dataset

You can download our preprocessed structured dataset (skeleton-level data) from the following link:

https://figshare.com/s/7c4039a91a0fb4db549b

The dataset contains two main folders:

- `Pretrain/` – Development datasets used for VidMotor pretraining. Each dataset is stored in `.pkl` format organized in a consistent structured format as follows:
  
  ```textile
  {
      "subject_name": <subject identifier>,  # Identifier of the subject (e.g., participant ID)
      "sample_name": <sample name>,  # Unique name or index of the sample
      "data": <array of shape (C, T, V, M)>,  # C: number of channels (2 for 2D coordinates, 3 for 3D coordinates); T: temporal length (number of frames); V: number of skeletal joints (may varies across datasets); M: number of persons (typically 1)
      "score_label": <movement quality score>,  # Quantitative score representing movement quality
      "class_label": <movement category>  # Categorical label representing the type of movement
  }
  ```

- `Finetune/` – Validation datasets used for fine-tuning and validation across downstream tasks. Each dataset is stored in `.pkl` format organized in a consistent structured format as follows:
  
  ```textile
  {
      <sample name>: 
      {
          "data": <array of shape (C, T, V, M)>,  # C: number of channels (2 for 2D coordinates, 3 for 3D coordinates); T: temporal length (number of frames); V: number of skeletal joints (may varies across datasets); M: number of persons (typically 1)
          "class_label": <category>  # For example, different types of abnormalities, whether a disease is present, the severity grade of the disease
  }
  ```

If you wish to directly test our code or reproduce our results, please place both folders under the current directory after downloading.

This ensures that the provided scripts can locate and load the datasets correctly.

# Original Data Sources

We integrates 26 publicly available datasets. Below is a categorized summary of the original data sources.

## Development datasets (10)

| Dataset name        | Download link                                                                                                                                       |
|:-------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
| Rhythmic Gymnastics | https://github.com/qinghuannn/ACTION-NET                                                                                                            |
| AGF-Olympics        | https://github.com/saniazahan/AGF-Olympics                                                                                                          |
| MMFS                | https://github.com/dingyn-Reno/MMFS                                                                                                                 |
| FineFS              | https://github.com/yanliji/FineFS-dataset                                                                                                           |
| Push-Up             | https://github.com/Kelly510/RehabExerAssess                                                                                                         |
| 3D-Yoga             | https://3DYogabsu.github.io                                                                                                                         |
| UMONS-TAICHI        | https://github.com/numediart/UMONS-TAICHI                                                                                                           |
| FMS                 | https://plus.figshare.com/collections/Datasets_supporting_Functional_movement_screen_dataset_collected_with_two_Azure_Kinect_depth_sensors_/5774969 |
| KIMORE              | https://vrai.dii.univpm.it/content/kimore-dataset                                                                                                   |
| UI-PRMD             | http://webpages.uidaho.edu/ui-prmd                                                                                                                  |

## Validation datasets (26)

### 1) Validation datasets for abnormality recognition (5)

| Dataset name           | Download link                                                                      |
|:----------------------:|:----------------------------------------------------------------------------------:|
| SPHERE-Stair-Gait      | https://data.bris.ac.uk/data/dataset/bgresiy3olk41nilo7k6xpkqf                     |
| SPHERE-Surface-Gait    | http://cs.swansea.ac.uk/~csadeline/datasets/SPHERE-Walking2015_skeletons_only.zip  |
| SPHERE-Sit             | http://cs.swansea.ac.uk/~csadeline/datasets/SPHERE-SitStand2015_skeletons_only.zip |
| SPHERE-Stand           | http://cs.swansea.ac.uk/~csadeline/datasets/SPHERE-SitStand2015_skeletons_only.zip |
| Walking-Treadmill-Gait | http://www-labs.iro.umontreal.ca/~labimage/GaitDataset                             |

### 2) Validation datasets for clinical diagnosis and severity grading (11)

| Dataset name      | Download link                                                                                                                                            |
|:-----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|
| SSBD-Line-Gait    | https://datadryad.org/dataset/doi:10.5061/dryad.s7h44j150                                                                                                |
| EHE-Hands-Wave    | https://github.com/bruceyo/egcnplusplus/tree/main/EHE_dataset                                                                                            |
| EHE-Hands-UpDown  | https://github.com/bruceyo/egcnplusplus/tree/main/EHE_dataset                                                                                            |
| EHE-Waist-BendL   | https://github.com/bruceyo/egcnplusplus/tree/main/EHE_dataset                                                                                            |
| EHE-Waist-BendR   | https://github.com/bruceyo/egcnplusplus/tree/main/EHE_dataset                                                                                            |
| EHE-Forward-Gait  | https://github.com/bruceyo/egcnplusplus/tree/main/EHE_dataset                                                                                            |
| EHE-Backward-Gait | https://github.com/bruceyo/egcnplusplus/tree/main/EHE_dataset                                                                                            |
| PD-Round-Gait     | https://figshare.com/articles/dataset/PDWalk_rar/19196138                                                                                                |
| PD-Walkway-Gait   | https://figshare.com/articles/dataset/A_dataset_of_overground_walking_full-body_kinematics_and_kinetics_in_individuals_with_Parkinson_s_disease/14896881 |
| PD4T-Round-Gait   | https://github.com/Plrbear/PECoP                                                                                                                         |
| PD4T-Leg-Agility  | https://github.com/Plrbear/PECoP                                                                                                                         |

### 3) Validation datasets for physical rehabilitation assessment (10)

| Dataset name             | Download link                                                                |
|:------------------------:|:----------------------------------------------------------------------------:|
| IRDS-Elbow-FlexionL      | https://zenodo.org/records/4610859                                           |
| IRDS-Elbow-FlexionR      | https://zenodo.org/records/4610859                                           |
| IRDS-Shoulder-FlexionL   | https://zenodo.org/records/4610859                                           |
| IRDS-Shoulder-FlexionR   | https://zenodo.org/records/4610859                                           |
| IRDS-Shoulder-AbductionL | https://zenodo.org/records/4610859                                           |
| IRDS-Shoulder-AbductionR | https://zenodo.org/records/4610859                                           |
| IRDS-Shoulder-Forward    | https://zenodo.org/records/4610859                                           |
| IRDS-Side-TapL           | https://zenodo.org/records/4610859                                           |
| IRDS-Side-TapR           | https://zenodo.org/records/4610859                                           |
| TRSP-Seated-Motion       | https://www.kaggle.com/datasets/derekdb/toronto-robot-stroke-posture-dataset |

# 💡Notes

All structured data provided in this repository are ready for direct use. However, several datasets require prior approval or restricted access:

- 3D-Yoga, PD4T-Round-Gait, and PD4T-Leg-Agility datasets require explicit permission from their corresponding authors via the links above.

- KIMORE, UI-PRMD, SPHERE-Surface-Gait, SPHERE-Sit, and SPHERE-Stand datasets are no longer publicly downloadable but may be obtained by contacting the original authors directly.

If you have successfully obtained approval or raw data, you may:

- Process them using our standardized preprocessing scripts in `./Preprocessing/`, or forward the approval email to us to receive the preprocessed structured data.

The released datasets are sufficient to reproduce primary VidMotor validation results and to further extend the research findings.
