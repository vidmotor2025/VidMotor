import numpy as np
import pickle
import re
import os
import pandas as pd

'''
Dataset Description
The dataset contains two viewpoints: front and side.
Each sample has 7 channels: xyz_x, xyz_y, xyz_z, wxyz_w, wxyz_x, wxyz_y, wxyz_z.  
In total, there are 3,327 samples and 7,488 key frames.  
'''

if __name__ == "__main__":
    score_path = 'score.txt'
    score_all = []
    with open(score_path, "r", encoding="utf-8") as file:
        for line in file:
            score_all.append(line.strip().split('/')[-1])

    data_path = './3d_yoga_skeleton'

    all_subject_id = []
    all_samplename = []
    all_data = []
    all_class_label = []
    all_score_label = []
    columns_to_extract = ['xyz_x', 'xyz_y', 'xyz_z', 'wxyz_w', 'wxyz_x', 'wxyz_y', 'wxyz_z']

    score_dict = {}
    for item in score_all:
        # Split out the class name and its corresponding class values
        name, label = item.rsplit(" ", 1)
        class_name = name.split('f')[0]
        label = int(label)
        if class_name not in score_dict:
            score_dict[class_name] = label
        else:
            if score_dict[class_name] != label:
                print(f"Inconsistent class values found for class {class_name}.")
                break

    for key, value in score_dict.items():
        if 'M' in key:
            gender = 'M'
        else:
            gender = 'F'
        front_path = os.path.join(data_path, 'Front', gender + key.split(gender)[-1].split('A')[0],
                                  key.split('A')[-1].split('a')[0], key.split('a')[-1])
        side_path = os.path.join(data_path, 'Side', gender + key.split(gender)[-1].split('A')[0],
                                 key.split('A')[-1].split('a')[0], key.split('a')[-1])

        files = os.listdir(front_path)
        front_data = np.zeros((7, len(files), 32))
        side_data = np.zeros((7, len(files), 32))

        for index, file in enumerate(files):
            front_path_now = os.path.join(front_path, file)
            side_path_now = re.sub(r'f(?=\d)', 's', front_path_now.replace('Front', 'Side'))
            front_data[:, index, :] = pd.read_csv(front_path_now)[columns_to_extract].to_numpy().transpose(1, 0)
            side_data[:, index, :] = pd.read_csv(side_path_now)[columns_to_extract].to_numpy().transpose(1, 0)

        all_subject_id.append(key.split('L')[-1][1:].split('A')[0])
        all_samplename.append(key)
        all_data.append(np.expand_dims(front_data[:3, :, :], axis=-1))
        all_class_label.append(int(key.split('A')[-1].split('a')[0]))
        all_score_label.append(float(value))

    data_dict = [
        {"subject_name": subject_id, "sample_name": name, "data": data, "score_label": sco, "class_label": cls}
        for subject_id, name, data, sco, cls in
        zip(all_subject_id, all_samplename, all_data, all_score_label, all_class_label)
    ]
    with open('data_and_label_3D-Yoga.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
