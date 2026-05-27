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
model_list = ['VidMotor_ST-GCN', 'VidMotor_CTR-GCN', 'VidMotor_PoseFormerV2', 'VidMotor_SkateFormer', 'VidMotor_DSTformer']

# Set the GPU ID for running experiments
device = 0

# Remove existing results file to start fresh
# If the results Excel file exists, delete it to avoid overwriting issues
save_path = './results/reproduce_VidMotor_with_backbones_results.xlsx'
if os.path.exists(save_path):
    os.remove(save_path)

# Loop over each dataset
for dataset in available_datasets:
    print(f'\nDataset: {dataset}')
    # Loop over each model
    for model in model_list:
        print(f'\n\tModel: {model}')
        # Run experiments with specific pretrained model and config
        if model == 'VidMotor_ST-GCN':
            cmd = ('python reproducibility/main_%s_reproducibility.py '
                    '--config VidMotor/config/Finetune_ST-GCN/%s.yaml --model-pretrain VidMotor/pretrain-ST-GCN.pt --save-path %s --device %s'
                    % (model, dataset, save_path, str(device)))
            result = subprocess.run(cmd, shell=True)
        elif model == 'VidMotor_CTR-GCN':
                cmd = ('python reproducibility/main_%s_reproducibility.py '
                        '--config VidMotor/config/Finetune_CTR-GCN/%s.yaml --model-pretrain VidMotor/pretrain-CTR-GCN.pt --save-path %s --device %s'
                        % (model, dataset, save_path, str(device)))
                result = subprocess.run(cmd, shell=True)
        elif model == 'VidMotor_PoseFormerV2':
                cmd = ('python reproducibility/main_%s_reproducibility.py '
                        '--config VidMotor/config/Finetune_PoseFormerV2/%s.yaml --model-pretrain VidMotor/pretrain-PoseFormerV2.pt --save-path %s --device %s'
                        % (model, dataset, save_path, str(device)))
                result = subprocess.run(cmd, shell=True)
        elif model == 'VidMotor_SkateFormer':
            cmd = ('python reproducibility/main_%s_reproducibility.py '
                   '--config VidMotor/config/Finetune_SkateFormer/%s.yaml --model-pretrain VidMotor/pretrain-SkateFormer.pt --save-path %s --device %s'
                   % (model, dataset, save_path, str(device)))
            result = subprocess.run(cmd, shell=True)
        elif model == 'VidMotor_DSTformer':
            cmd = ('python reproducibility/main_%s_reproducibility.py '
                    '--config VidMotor/config/Finetune_DSTformer/%s.yaml --model-pretrain VidMotor/pretrain-DSTformer.pt --save-path %s --device %s'
                    % (model, dataset, save_path, str(device)))
            result = subprocess.run(cmd, shell=True)

