import os
import json
import numpy as np
import csv
import pickle

def load_pose_sequence(json_folder_path):
    """
    Read all OpenPose JSON files from the specified folder and generate a joint sequence as a NumPy array.
    Parameters:
        folder_path (str): Path to the folder containing JSON files for all video frames.
    Returns:
        np.ndarray: Joint sequence with shape (2, time, 25).
    """
    json_files = sorted(os.listdir(json_folder_path))
    pose_sequence = []
    for json_file in json_files:
        file_path = os.path.join(json_folder_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            keypoints = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)[:, :2]
            pose_sequence.append(keypoints)

    pose_sequence = np.array(pose_sequence)  # (t, 25, 2)
    pose_sequence = pose_sequence.transpose(2, 0, 1)  # (2, t, 25)
    return np.expand_dims(pose_sequence, axis=-1)


csv_score_files = ["./Annotations/Gait/train.csv", "./Annotations/Gait/test.csv"]
skeleton_folder = "./Videos/Gait_OpenPose"
train_samples = []
train_names = []
train_labels = []
test_samples = []
test_names = []
test_labels = []
for csv_score_file in csv_score_files:
    samples = []
    scores = []
    with open(csv_score_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            samples.append(row[0])
            scores.append(int(row[2]))
    for index in range(len(samples)):
        sample = samples[index]
        score = scores[index]
        patient_id = sample.split("_")[1]
        visit_number = sample.split("_")[0]
        skeleton_folder_path = os.path.join(skeleton_folder, patient_id, visit_number)
        pose_npy = load_pose_sequence(skeleton_folder_path)
        if "train.csv" in csv_score_file:
            train_samples.append(pose_npy)
            train_names.append(sample)
            train_labels.append(score)
        elif "test.csv" in csv_score_file:
            test_samples.append(pose_npy)
            test_names.append(sample)
            test_labels.append(score)
        else:
            print("Error.")

all_names = train_names + test_names
all_data = train_samples + test_samples
all_scores = train_labels + test_labels
a_data_dict = {name: {"data": data, "class_label": cls} for name, data, cls in zip(all_names, all_data, all_scores)}
with open('data_and_label_PD4T-Round-Gait.pkl', 'wb') as f:
    pickle.dump(a_data_dict, f)
