import numpy as np
import os
import pickle

# training set
train_path = "./skeletons_only/Training"
txt_files = [f for f in os.listdir(train_path) if f.endswith('.txt')]
train_samples = []
train_names = []
train_labels = []
for file in txt_files:
    file_path = os.path.join(train_path, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        row = list(map(float, line.split()))
        data.append(row)
    data = np.array(data)
    # After excluding the time column (assuming the first column is time), the data has shape time × 60,
    # where each of the 15 keypoints has 4 elements: [flag (can be ignored), x, y, z]
    S = data[:, 1:].reshape(data.shape[0], 15, 4)
    final_matrix_test = np.expand_dims(S[:, :, 1:].transpose(2, 0, 1), axis=-1)
    train_samples.append(final_matrix_test)
    train_names.append(file.split('.')[0])
    train_labels.append(0)

# testing set
test_path = "./skeletons_only/Testing"
txt_files = [f for f in os.listdir(test_path) if f.endswith('.txt')]
test_samples_Normal = []
test_names_Normal = []
test_labels_Normal = []
test_samples_Parkinson = []
test_names_Parkinson = []
test_labels_Parkinson = []
test_samples_Stroke = []
test_names_Stroke = []
test_labels_Stroke = []
for file in txt_files:
    file_path = os.path.join(test_path, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        row = list(map(float, line.split()))
        data.append(row)
    data = np.array(data)
    S = data[:, 1:].reshape(data.shape[0], 15, 4)
    final_matrix_test = np.expand_dims(S[:, :, 1:].transpose(2, 0, 1), axis=-1)
    if "Normal" in file.split(".")[0]:
        test_samples_Normal.append(final_matrix_test)
        test_names_Normal.append(file.split('.')[0])
        test_labels_Normal.append(0)
    elif "Parkinson" in file.split(".")[0]:
        test_samples_Parkinson.append(final_matrix_test)
        test_names_Parkinson.append(file.split('.')[0])
        test_labels_Parkinson.append(1)
    elif "Stroke" in file.split(".")[0]:
        test_samples_Stroke.append(final_matrix_test)
        test_names_Stroke.append(file.split('.')[0])
        test_labels_Stroke.append(2)
    else:
        print("Error.")

print("Training set size (Normal): ", len(train_samples))
print("Testing set size (Normal, Parkinson, Stroke): ", len(test_samples_Normal), len(test_samples_Parkinson), len(test_samples_Stroke))
all_samplename = train_names + test_names_Normal + test_names_Parkinson + test_names_Stroke
all_data = train_samples + test_samples_Normal + test_samples_Parkinson + test_samples_Stroke
all_class_label = train_labels + test_labels_Normal + test_labels_Parkinson + test_labels_Stroke
data_dict = {name: {"data": data, "class_label": cls} for name, data, cls in zip(all_samplename, all_data, all_class_label)}
with open('data_and_label_SPHERE-Surface-Gait.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
