import subprocess
import os


# List of available datasets for reproducibility experiments
# Please refer to './data/README_data.md' for download
available_datasets = [
    'SPHERE-Stair-Gait', 'Walking-Treadmill-Gait', 'SSBD-Line-Gait', 'EHE-Hands-Wave', 'EHE-Hands-UpDown', 'EHE-Waist-BendL',
    'EHE-Waist-BendR', 'EHE-Forward-Gait', 'EHE-Backward-Gait', 'PD-Round-Gait', 'PD-Walkway-Gait', 'IRDS-Elbow-FlexionL',
    'IRDS-Elbow-FlexionR', 'IRDS-Shoulder-FlexionL', 'IRDS-Shoulder-FlexionR', 'IRDS-Shoulder-AbductionL',
    'IRDS-Shoulder-AbductionR', 'IRDS-Shoulder-Forward', 'IRDS-Side-TapL', 'IRDS-Side-TapR', 'TRSP-Seated-Motion'
]

# The following datasets require prior approval or have restricted access
# Please refer to './data/README_data.md' for access instructions
# request_based_datasets = ['SPHERE-Surface-Gait', 'SPHERE-Sit', 'SPHERE-Stand', 'PD4T-Round-Gait', 'PD4T-Leg-Agility']

# List of models to run for reproducibility
model_list = ['VidMotor', 'SkateFormer', 'HD-GCN', 'FR-Head', 'MotionBERT', 'SkeleT-GCN', 'MMN', 'ProtoGCN']

# Set the GPU ID for running experiments
device = 0

# Remove existing results file to start fresh
# If the results Excel file exists, delete it to avoid overwriting issues
save_path = './results/reproduce_VidMotor_and_SOTA_results.xlsx'
if os.path.exists(save_path):
    os.remove(save_path)

# Loop over each dataset
for dataset in available_datasets:
    print(f'\nDataset: {dataset}')
    # Loop over each model
    for model in model_list:
        print(f'\n\tModel: {model}')
        # Run VidMotor experiments with specific pretrained model and config
        if model == 'VidMotor':
            cmd = ('python reproducibility/main_VidMotor_ST-GCN_reproducibility.py '
                    '--config VidMotor/config/Finetune_ST-GCN/%s.yaml --model-pretrain VidMotor/pretrain-ST-GCN.pt --save-path %s --device %s'
                    % (dataset, save_path, str(device)))
            result = subprocess.run(cmd, shell=True)
        # Run MotionBERT experiments with specific pretrained model and config
        elif model == 'MotionBERT':
            cmd = ('python reproducibility/main_%s_reproducibility.py '
                   '--config SOTA/%s/config/%s.yaml --model-pretrain SOTA/MotionBERT/latest_epoch.bin --save-path %s --device %s'
                   % (model, model, dataset, save_path, str(device)))
            result = subprocess.run(cmd, shell=True)
        # Run other models (SkateFormer, HD-GCN, FR-Head, SkeleT-GCN, MMN, and ProtoGCN) with their config
        else:
            cmd = ('python reproducibility/main_%s_reproducibility.py '
                    '--config SOTA/%s/config/%s.yaml --save-path %s --device %s'
                    % (model, model, dataset, save_path, str(device)))
            result = subprocess.run(cmd, shell=True)
