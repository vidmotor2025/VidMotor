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
            # Check if the file contains human data
            if data["people"]:
                # Retrieve the (x, y) coordinates of 25 keypoints for the first person
                keypoints = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)[:, :2]
                pose_sequence.append(keypoints)
            else:
                # If no human is detected in the current frame, append zero padding
                pose_sequence.append(np.zeros((25, 2)))
    pose_sequence = np.array(pose_sequence)
    pose_sequence = pose_sequence.transpose(2, 0, 1)
    return np.expand_dims(pose_sequence, axis=-1)


def calculate_no_person_ratio(json_folder_path):
    total_frames = 0
    no_person_frames = 0
    for json_file in os.listdir(json_folder_path):
        if json_file.endswith(".json"):
            total_frames += 1
            json_path = os.path.join(json_folder_path, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data.get("people", []):  # "people" being empty indicates that no person was detected
                    no_person_frames += 1
    no_person_ratio = no_person_frames / total_frames
    return no_person_ratio, no_person_frames, total_frames


csv_score_files = ["./Annotations/Leg agility/train.csv", "./Annotations/Leg agility/test.csv"]
skeleton_folder = "./Videos/Leg_Agility_OpenPose"
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
        patient_id = sample.split("_")[2]
        if patient_id in ["009", "013", "015", "019", "022", "023", "063"]:  # The videos need to be preprocessed and standardized to a frame rate of 30 fps. (from moviepy.editor import VideoFileClip)
            visit_number = sample.split("_")[0] + "_" + sample.split("_")[1] + "_30fps"
        else:
            visit_number = sample.split("_")[0] + "_" + sample.split("_")[1]
        skeleton_folder_path = os.path.join(skeleton_folder, patient_id, visit_number)
        no_person_ratio, no_person_frames, total_frames = calculate_no_person_ratio(skeleton_folder_path)
        if no_person_frames >= 30:
            continue
        else:
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
# Samples with a score of 4 were excluded because they all came from the same subject and the sample size was insufficient for reliable analysis.
indices_to_remove = [i for i, score in enumerate(all_scores) if score == 4]
for idx in sorted(indices_to_remove, reverse=True):
    del all_names[idx]
    del all_data[idx]
    del all_scores[idx]

a_data_dict = {name: {"data": data, "class_label": cls} for name, data, cls in zip(all_names, all_data, all_scores)}
with open('data_and_label_PD4T-Leg-Agility.pkl', 'wb') as f:
    pickle.dump(a_data_dict, f)
