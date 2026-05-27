import numpy as np
import pickle
import csv

score_path = './Reduced Data/Data and Scores csv/'
data_path = './UI-PRMD-Data/Kinetic/'
save_path = './UI-PRMD-Reduced-Data/Kinetic/'
list_e = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']  # 10 movements

preserve_dict = {
    '1': {'1': [2, 3, 4, 5, 6, 7, 8, 9, 10], '2': [2, 3, 4, 5, 6, 7, 8, 9, 10], '3': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '4': [2, 3, 4, 5, 6, 7, 8, 9, 10], '5': [2, 3, 4, 5, 6, 7, 8, 9, 10], '6': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          '7': [2, 3, 5, 6, 7, 8, 9, 10], '8': [2, 3, 4, 5, 6, 7, 8, 9, 10], '9': [2, 3, 4, 6, 7, 8, 9, 10],
          '10': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    '2': {'1': [2, 5, 6, 7, 8, 9], '2': [], '3': [2, 3, 4, 5, 6, 8, 9, 10], '4': [2, 3, 4, 5, 6, 8, 9],
          '5': [3, 5, 6, 7, 8, 9, 10], '6': [2, 3, 4, 5, 6, 7, 8, 9, 10], '7': [], '8': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '9': [1, 2, 3, 4, 5, 6, 7, 8, 9], '10': []},
    '3': {'1': [4, 5, 6, 8, 9, 10], '2': [], '3': [], '4': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '5': [2, 3, 4, 5, 6, 7, 8, 9, 10], '6': [2, 3, 4, 5, 6, 7, 8, 9, 10], '7': [],
          '8': [2, 3, 4, 5, 6, 7, 8, 9, 10], '9': [2, 3, 4, 5, 6, 7, 8, 9, 10], '10': []},
    '4': {'1': [2, 3, 4, 5, 6, 7, 8, 9, 10], '2': [2, 3, 4, 5, 6, 7, 8, 9, 10], '3': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '4': [2, 3, 4, 5, 6, 7, 8, 9, 10], '5': [2, 3, 4, 5, 6, 7, 8, 9, 10], '6': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '7': [], '8': [2, 3, 4, 5, 6, 7, 8, 9, 10], '9': [2, 3, 4, 5, 6, 7, 8], '10': []},
    '5': {'1': [2, 3, 4, 5, 6, 7, 8, 9, 10], '2': [2, 3, 4, 5, 6, 7, 8, 9, 10], '3': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '4': [2, 3, 5, 6, 7, 9, 10], '5': [2, 3, 4, 5, 6, 7, 8, 9, 10], '6': [2, 3, 4, 6, 7, 8, 9, 10],
          '7': [2, 3, 4, 5, 6, 7, 8, 9, 10], '8': [2, 3, 4, 5, 6, 7, 8, 9], '9': [2, 3, 4, 5, 6, 7, 8],
          '10': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    '6': {'1': [2, 3, 4, 5, 6, 7, 8, 9, 10], '2': [2, 3, 4, 5, 6, 7, 8, 9, 10], '3': [3, 4, 5, 6, 7, 8, 9],
          '4': [2, 3, 4, 5, 6, 7, 8, 9, 10], '5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '6': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          '7': [], '8': [2, 3, 4, 5, 6, 7, 8, 9, 10], '9': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '10': []},
    '7': {'1': [2, 3, 4, 5, 6, 7, 8, 9, 10], '2': [2, 3, 4, 5, 6, 7, 8, 9, 10], '3': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '4': [2, 3, 4, 5, 6, 7, 8, 9, 10], '5': [2, 3, 4, 5, 6, 7, 8, 9, 10], '6': [], '7': [],
          '8': [2, 3, 4, 5, 6, 7, 8, 9, 10], '9': [2, 3, 4, 5, 6, 7, 8, 9, 10], '10': []},
    '8': {'1': [2, 4, 5, 6, 9], '2': [2, 3, 4, 5, 6, 7, 8, 9], '3': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '4': [2, 3, 4, 5, 6, 7, 8, 9, 10], '5': [2, 3, 4, 5, 6, 7, 8, 9], '6': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '7': [], '8': [3, 4, 6, 7, 8, 9], '9': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '10': []},
    '9': {'1': [], '2': [2, 3, 4, 5, 6, 7, 8, 9, 10], '3': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '4': [2, 3, 4, 5, 6, 7, 8], '5': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '6': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          '7': [], '8': [2, 3, 4, 5, 6, 7, 8, 9, 10], '9': [2, 3, 4, 5, 8, 9, 10],
          '10': []},
    '10': {'1': [2, 3, 4, 5, 6, 7, 8, 9, 10], '2': [2, 3, 4, 5, 6, 7, 8, 9, 10], '3': [2, 3, 4, 5, 6, 7, 8, 9, 10],
           '4': [2, 3, 4, 5, 6, 7, 8, 9, 10], '5': [], '6': [2, 3, 4, 5, 6, 7, 8, 9, 10],
           '7': [], '8': [2, 3, 4, 5, 6, 7, 8, 9, 10], '9': [],
           '10': []},
}

all_score_label = []
all_class_label = []
all_data = []
all_samplename = []
all_subject_id = []
for e in range(10):
    correct_class_label = []
    incorrect_class_label = []
    correct_score_label = []
    incorrect_score_label = []
    with open(score_path + 'Correct_score_S' + str(e + 1) + '.csv') as f:
        data = csv.reader(f)
        for i in data:
            correct_class_label.append(e)
            correct_score_label.append(float(i[0]))
    with open(score_path + 'Incorrect_score_S' + str(e + 1) + '.csv') as f:
        data = csv.reader(f)
        for i in data:
            incorrect_class_label.append(e)
            incorrect_score_label.append(float(i[0]))

    # correct
    correct_data_reduce = []
    correct_samplename_reduce = []
    correct_subject_reduce = []
    with open(data_path + 'correct_data_m' + list_e[e] + '.pkl', 'rb') as f:
        correct_data = pickle.load(f)
    for m in range(1, 11):
        if preserve_dict[str(e + 1)][str(m)]:
            for i in preserve_dict[str(e + 1)][str(m)]:  # indices of retained segments for each subject (relative to the 10 segments of that subject)
                idx = (m - 1) * 10 + (i - 1)  # positions of the retained segments within the full sequence of length 100
                correct_data_reduce.append(np.expand_dims(correct_data[idx], axis=-1))  # Extract the retained segments from the full dataset to form the reduced dataset.
                correct_samplename_reduce.append(
                    "Exercise" + str(e + 1) + "_Subject" + str(m) + "_Repetition" + str(i) + "_Correct")
                correct_subject_reduce.append("Subject" + str(m).zfill(2))

    # incorrect
    incorrect_data_reduce = []
    incorrect_samplename_reduce = []
    incorrect_subject_reduce = []
    with open(data_path + 'incorrect_data_m' + list_e[e] + '.pkl', 'rb') as f:
        incorrect_data = pickle.load(f)
    for m in range(1, 11):
        if preserve_dict[str(e + 1)][str(m)]:
            for i in preserve_dict[str(e + 1)][str(m)]:
                idx = (m - 1) * 10 + (i - 1)
                incorrect_data_reduce.append(np.expand_dims(incorrect_data[idx], axis=-1))
                incorrect_samplename_reduce.append(
                    "Exercise" + str(e + 1) + "_Subject" + str(m) + "_Repetition" + str(i) + "_Incorrect")
                incorrect_subject_reduce.append("Subject" + str(m).zfill(2))
    all_class_label.extend(correct_class_label + incorrect_class_label)
    all_score_label.extend(correct_score_label + incorrect_score_label)
    all_data.extend(correct_data_reduce + incorrect_data_reduce)
    all_samplename.extend(correct_samplename_reduce + incorrect_samplename_reduce)
    all_subject_id.extend(correct_subject_reduce + incorrect_subject_reduce)


data_dict = [
    {"subject_name": subject_id, "sample_name": name, "data": data, "score_label": sco, "class_label": cls}
    for subject_id, name, data, sco, cls in
    zip(all_subject_id, all_samplename, all_data, all_score_label, all_class_label)
]
with open('data_and_label_UI-PRMD.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
