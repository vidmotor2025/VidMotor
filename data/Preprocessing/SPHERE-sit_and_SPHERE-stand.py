import os
import numpy as np
import pickle

# training set
folder_path = "./skeletons_only/Training"
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
train_samples_SittingDown = []
train_names_SittingDown = []
train_labels_SittingDown = []
train_samples_StandingUp = []
train_names_StandingUp = []
train_labels_StandingUp = []
for file in txt_files:
    file_path = os.path.join(folder_path, file)
    data = np.loadtxt(file_path, delimiter=',')
    frame_number_path = os.path.join(folder_path, "frame_number", "F_" + file)
    frame_number_data = np.loadtxt(frame_number_path, delimiter=',')
    # Identify the indices where the sequence is discontinuous
    diffs = np.diff(frame_number_data[:, 1])
    split_indices = np.where(diffs != 1)[0] + 1
    segments = np.split(frame_number_data, split_indices)
    for i, segment in enumerate(segments):
        indices = segment[:, 0]
        mask = np.isin(data[:, 1], indices)
        matched_data = data[mask, 4:]
        S = matched_data.reshape(matched_data.shape[0], 25, 4)
        final_data = np.expand_dims(S[:, :, 1:].transpose(2, 0, 1), axis=-1)
        if "SittingDown" in file:
            train_samples_SittingDown.append(final_data)
            train_names_SittingDown.append(file.split('.')[0] + '_segment' + str(i + 1))
            train_labels_SittingDown.append(0)
        elif "StandingUp" in file:
            train_samples_StandingUp.append(final_data)
            train_names_StandingUp.append(file.split('.')[0] + '_segment' + str(i + 1))
            train_labels_StandingUp.append(0)
        else:
            print("Error.")

# testing set
folder_paths = ["./skeletons_only/Testing/Abnormal",
                "./skeletons_only/Testing/Normal"]
test_samples_SittingDown = []
test_names_SittingDown = []
test_labels_SittingDown = []
test_samples_StandingUp = []
test_names_StandingUp = []
test_labels_StandingUp = []
for folder_path in folder_paths:
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        data = np.loadtxt(file_path, delimiter=',')
        matched_data = data[:, 4:]
        S = matched_data.reshape(matched_data.shape[0], 25, 4)
        final_data = np.expand_dims(S[:, :, 1:].transpose(2, 0, 1), axis=-1)
        if "SittingDown" in file:
            test_samples_SittingDown.append(final_data)
            test_names_SittingDown.append(file.split('.')[0])
            if "Testing/Abnormal" in folder_path:
                test_labels_SittingDown.append(1)
            elif "Testing/Normal" in folder_path:
                test_labels_SittingDown.append(0)
            else:
                print("Error.")
        elif "StandingUp" in file:
            test_samples_StandingUp.append(final_data)
            test_names_StandingUp.append(file.split('.')[0])
            if "Testing/Abnormal" in folder_path:
                test_labels_StandingUp.append(1)
            elif "Testing/Normal" in folder_path:
                test_labels_StandingUp.append(0)
            else:
                print("Error.")
        else:
            print("Error.")


print("Training set size (Normal) of SittingDown: ", len(train_samples_SittingDown), "; StandingUp: ", len(train_samples_StandingUp))
print("Test set size (Normal, Abnormal) of SittingDown: ", len(test_samples_SittingDown),
      {value: test_labels_SittingDown.count(value) for value in set(test_labels_SittingDown)},
      "; StandingUp: ", len(test_samples_StandingUp), {value: test_labels_StandingUp.count(value) for value in set(test_labels_StandingUp)})

all_samplename = train_names_SittingDown + test_names_SittingDown
all_data = train_samples_SittingDown + test_samples_SittingDown
all_score_label = train_labels_SittingDown + test_labels_SittingDown
data_dict = {name: {"data": data, "class_label": sco} for name, data, sco in
             zip(all_samplename, all_data, all_score_label)}
with open('data_and_label_SPHERE-Sit.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

all_samplename = train_names_StandingUp + test_names_StandingUp
all_data = train_samples_StandingUp + test_samples_StandingUp
all_score_label = train_labels_StandingUp + test_labels_StandingUp
data_dict = {name: {"data": data, "class_label": sco} for name, data, sco in
             zip(all_samplename, all_data, all_score_label)}
with open('data_and_label_SPHERE-Stand.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

