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

# Set the GPU ID for running experiments
device = 0

# List of training data ratios to simulate limited data conditions
ratio = 0.1

# Remove existing results file to start fresh
# If the results Excel file exists, delete it to avoid overwriting issues
save_path = './results/reproduce_results_10%training.xlsx'
if os.path.exists(save_path):
    os.remove(save_path)

# Loop over each dataset
for dataset in available_datasets:
    print(f'\nDataset: {dataset}; Training data: {ratio*100}%\n')
    # Run VidMotor experiments with specific pretrained model and config
    cmd = ('python reproducibility/main_VidMotor_ST-GCN_ratio_reproducibility.py '
                '--config VidMotor/config/Finetune_ST-GCN/%s.yaml --model-pretrain VidMotor/pretrain-ST-GCN.pt --save-path %s --device %s --train-ratio %s'
                % (dataset, save_path, str(device), str(ratio)))
    result = subprocess.run(cmd, shell=True)
