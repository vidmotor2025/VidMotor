import os
import argparse
import pickle
import numpy as np
import re
import pandas as pd
import warnings
warnings.simplefilter("ignore")

max_body = 1
num_joint = 25
toolbar_width = 30
files_ = os.listdir('./KIMORE/skeleton')


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'c',
                        'ang_x', 'ang_y', 'ang_z', 'ang_w'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_ang(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    if seq_info['numFrame'] > 150:
        # https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        sample_indexs = f(150, seq_info['numFrame'])
        seq_info['frameInfo'] = [f for n, f in enumerate(seq_info['frameInfo']) if n in sample_indexs] # range(start,end)]
        seq_info['numFrame'] = 150
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['ang_x'], v['ang_y'], v['ang_z']]
                else:
                    pass
    return data


def read_xyz_ori_length(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data


def read_xyzang(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    if seq_info['numFrame'] > 150:
        # https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        sample_indexs = f(150, seq_info['numFrame'])
        seq_info['frameInfo'] = [f for n, f in enumerate(seq_info['frameInfo']) if n in sample_indexs] # range(start,end)]
        seq_info['numFrame'] = 150
    data = np.zeros((6, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z'], v['ang_x'], v['ang_y'], v['ang_z']]
                else:
                    pass
    return data


def find_all_xlsx_files(directory):
    xlsx_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
                xlsx_files.append(os.path.join(root, file))

    return xlsx_files


def gendata(data_path, ori_data_path, action, feature='both'):
    r = re.compile(".*" + "E00" + str(action) + ".*.skeleton")
    files = list(filter(r.match, files_))
    score_path_dict = {'G001': os.path.join(ori_data_path, 'CG/Expert'),
                       'G002': os.path.join(ori_data_path, 'CG/NotExpert'),
                       'G003': os.path.join(ori_data_path, 'GPP/BackPain'),
                       'G004': os.path.join(ori_data_path, 'GPP/Parkinson'),
                       'G005': os.path.join(ori_data_path, 'GPP/Stroke')}
    score_sub_path_dict = {2: 'Es1', 3: 'Es2', 4: 'Es2', 5: 'Es3', 6: 'Es3', 7: 'Es4', 8: 'Es4', 9: 'Es5'}
    score_sub_path_dict_int = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7}
    score_title_xlsx_dict = {2: 'Ex#1', 3: 'Ex#2', 4: 'Ex#2', 5: 'Ex#3', 6: 'Ex#3', 7: 'Ex#4', 8: 'Ex#4', 9: 'Ex#5'}
    group_name = {1: 'CGExpert', 2: 'CGNotExpert', 3: 'GPPBackPain', 4: 'GPPParkinson', 5: 'GPPStroke'}
    ex_name = {2: '1', 3: '2l', 4: '2r', 5: '3l', 6: '3r', 7: '4l', 8: '4r', 9: '5'}

    count = 0
    ori_sample_name = []
    sample_name = []
    sample_subject = []
    sample_label = []
    sample_cTS_score = []
    sample_cPO_score = []
    sample_cCF_score = []
    for group in ['G001', 'G003', 'G004', 'G005']:
        for subject in range(1, 28):
            sub_str = 'S00' + str(subject) if subject < 10 else 'S0' + str(subject)
            r = re.compile(group + sub_str + ".*.skeleton")
            files_g_s = list(filter(r.match, files))
            if len(files_g_s) == 0:
                continue
            files_g_s.sort()
            for i in range(0, len(files_g_s)):
                numbers = [int(num) for num in re.findall(r'[A-Za-z](\d+)', files_g_s[i])]
                ori_sample_name.append(files_g_s[i])
                sample_name.append("Exercise" + str(ex_name[numbers[2]]) + "_Subject" + str(numbers[1]) + "_Repetition" + str(numbers[3]) + "_" + group_name[numbers[0]])
                sample_subject.append(files_g_s[i][:8])
                label = score_sub_path_dict_int[action]
                sample_label.append(label)
                all_xlsx_files = find_all_xlsx_files(score_path_dict[group])
                filtered_files = [file for file in all_xlsx_files
                                  if 'ID' + str(subject) + '.xlsx' in file and score_sub_path_dict[action] in file and 'ClinicalAssessment' in file and '~' not in file]
                assert len(filtered_files) == 1
                df = pd.read_excel(os.path.join(filtered_files[0]))
                sample_cTS_score.append(df['clinical TS ' + score_title_xlsx_dict[action]].tolist()[0])
                sc = df['clinical TS ' + score_title_xlsx_dict[action]].tolist()[0]
                if np.isnan(sc):
                    count += 1
                    # print(filtered_files, sample_name[-1])
                sample_cPO_score.append(df['clinical PO ' + score_title_xlsx_dict[action]].tolist()[0])
                sample_cCF_score.append(df['clinical CF ' + score_title_xlsx_dict[action]].tolist()[0])
        fp = []
        for i, s in enumerate(ori_sample_name):
            if feature == 'position':
                data = read_xyz_ori_length(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            elif feature == 'angle':
                data = read_ang(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            else:
                data = read_xyzang(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            fp.append(data)
    ori_sample_name_test = []
    sample_name_test = []
    sample_subject_test = []
    sample_label_test = []
    sample_cTS_score_test = []
    sample_cPO_score_test = []
    sample_cCF_score_test = []
    for group in ['G002']:
        for subject in range(1, 28):
            sub_str = 'S00' + str(subject) if subject < 10 else 'S0' + str(subject)
            r = re.compile(group + sub_str + ".*.skeleton")
            files_g_s = list(filter(r.match, files))
            if len(files_g_s) == 0:
                continue
            files_g_s.sort()
            for i in range(0, len(files_g_s)):
                numbers = [int(num) for num in re.findall(r'[A-Za-z](\d+)', files_g_s[i])]
                ori_sample_name_test.append(files_g_s[i])
                sample_name_test.append("Exercise" + str(ex_name[numbers[2]]) + "_Subject" + str(numbers[1]) + "_Repetition" + str(numbers[3]) + "_" + group_name[numbers[0]])
                sample_subject_test.append(files_g_s[i][:8])
                label = score_sub_path_dict_int[action]
                sample_label_test.append(label)
                all_xlsx_files = find_all_xlsx_files(score_path_dict[group])
                filtered_files = [file for file in all_xlsx_files if
                                  'ID' + str(subject) + '.xlsx' in file and score_sub_path_dict[
                                      action] in file and 'ClinicalAssessment' in file and '~' not in file]
                assert len(filtered_files) == 1
                df = pd.read_excel(os.path.join(filtered_files[0]))
                sample_cTS_score_test.append(df['clinical TS ' + score_title_xlsx_dict[action]].tolist()[0])
                sample_cPO_score_test.append(df['clinical PO ' + score_title_xlsx_dict[action]].tolist()[0])
                sample_cCF_score_test.append(df['clinical CF ' + score_title_xlsx_dict[action]].tolist()[0])
        fp_test = []
        for i, s in enumerate(ori_sample_name_test):
            if feature == 'position':
                data = read_xyz_ori_length(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            elif feature == 'angle':
                data = read_ang(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            else:
                data = read_xyzang(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            fp_test.append(data)

    return fp, sample_label, sample_cTS_score, sample_cPO_score, sample_cCF_score, sample_name, sample_subject, \
           fp_test, sample_label_test, sample_cTS_score_test, sample_cPO_score_test, sample_cCF_score_test, sample_name_test, sample_subject_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Converter.')
    parser.add_argument('--data_path', default='./KIMORE/skeleton', help='the path of the skeleton data')
    parser.add_argument('--ori_data_path', default='./KIMORE/origin', help='the path of the original data')
    parser.add_argument('--joint_feature', default='position', choices=['angle', 'position', 'both'], help='the feature of the skeleton data')
    arg = parser.parse_args()
    kimore_exercises = {
        2: 'Es1,   ',
        3: 'Es2(L),',
        4: 'Es2(R),',
        5: 'Es3(L),',
        6: 'Es3(R),',
        7: 'Es4(L),',
        8: 'Es4(R),',
        9: 'Es5,   ',
    }
    all_samplename = []
    all_data = []
    all_class_label = []
    all_score_label = []
    all_subject_id = []

    for act in [2, 3, 4, 5, 6, 7, 8, 9]:
        print('Generate action: ', kimore_exercises[act])
        data, yn_class, cTS_score, cPO_score, cCF_score, name, subject, data_test, yn_class_test, cTS_score_test, cPO_score_test, cCF_score_test, name_test, subject_test = \
            gendata(arg.data_path, arg.ori_data_path, act, arg.joint_feature)
        data, name, cTS_score, yn_class, subject = zip(*[(w, x, y, z, p) for w, x, y, z, p in zip(data, name, cTS_score, yn_class, subject) if not np.isnan(y)])

        all_data.extend(data)
        all_class_label.extend(yn_class)
        all_score_label.extend(cTS_score)
        all_samplename.extend(name)
        all_subject_id.extend(subject)

        all_data.extend(data_test)
        all_class_label.extend(yn_class_test)
        all_score_label.extend(cTS_score_test)
        all_samplename.extend(name_test)
        all_subject_id.extend(subject_test)

    data_dict = [
        {"subject_name": subject_id, "sample_name": name, "data": data, "score_label": sco, "class_label": cls}
        for subject_id, name, data, sco, cls in
        zip(all_subject_id, all_samplename, all_data, all_score_label, all_class_label)
    ]
    with open('data_and_label_KIMORE.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
