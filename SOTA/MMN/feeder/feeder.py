import pickle
import random

import numpy as np

from .pose_utils import *
from torch.utils.data import Dataset
import math


class Feeder(Dataset):
    """Feeder for loading, preprocessing, and serving datasets."""

    def __init__(self, dataset_info, data_centered, data_dim, data_norm, score_norm, label_type, data_length=64,
                 data_indices=None, stage=None):
        self.dataset_name = dataset_info['name']
        self.data_path = dataset_info['data_path']
        self.data_centered = data_centered
        self.data_dim = data_dim
        assert self.data_dim == '3D'
        self.data_norm = data_norm
        self.label_type = label_type
        if 'data_length' in dataset_info:
            data_length = dataset_info['data_length']
        self.data_length = data_length
        self.indices = data_indices
        self.stage = stage
        self.partition = dataset_info['partition']
        if self.partition:
            self.right_arm = np.array([14,15,16])
            self.left_arm = np.array([11,12,13])
            self.right_leg = np.array([1,2,3])
            self.left_leg = np.array([4,5,6])
            self.torso = np.array([7,8, 9, 0, 10])
            self.new_idx = np.concatenate(
                (self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.torso), axis=-1)
        # Load raw data
        self.load_data()
        # Preprocess labels if normalization is required
        if score_norm:
            self.min_score = float(dataset_info['score_range']['min'])
            self.max_score = float(dataset_info['score_range']['max'])
            self.score_trend = dataset_info['score_trend']
            self.normalize_scores()
            if not self.score_trend:
                self.unify_scores_trend()
        # Preprocess pose data
        self.unify_keypoints_and_channel()
        if self.stage is not None:
            self.unify_time_length_and_prepocess()
        if 'Leg-Agility' in self.dataset_name:
            self.unify_left_right()
        if self.data_centered == 7:
            self.center_poses()
        if self.data_norm == 'zscore':
            self.normalize_poses()

    def load_data(self):
        """Load raw skeleton data and labels from a pickle file."""
        try:
            with open(self.data_path, 'rb') as f:
                data_dict = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'Data file not found: {self.data_path}. Please refer to README_data.md in ./data')
        if self.indices is not None:
            data_dict_indices = {k: data_dict[k] for k in self.indices}
            data_dict = data_dict_indices
        self.sample_name = list(data_dict.keys())
        self.data = [sample["data"] for sample in data_dict.values()]
        self.label = [float(sample[self.label_type]) for sample in data_dict.values()]
        self.ori_label = self.label.copy()
        self.index_list = [None] * len(self.data)


    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.label)

    def __getitem__(self, index):
        """Return a single sample."""
        label = self.label[index]
        # if self.partition:
        #     data = np.array(self.data[index][:, :, self.new_idx])
        # else:
        data = np.array(self.data[index])
        index_t = self.index_list[index]
        return data, label, index_t, index

    def normalize_scores(self):
        """Min-Max normalize labels to the [0,1] range."""
        float_list = [float(i) for i in self.label]
        self.label = [(i - self.min_score) / (self.max_score - self.min_score) for i in float_list]

    def unify_scores_trend(self):
        """Invert labels so that higher values indicate better performance."""
        self.label = [1 - item for item in self.label]

    def center_poses(self):
        """
        Center each skeleton sequence around a reference joint.
        For example, the spine center is subtracted from all joints in the first frame.
        """
        for i in range(len(self.data)):
            self.data[i] -= self.data[i][:, 0, self.data_centered:self.data_centered + 1][:, None, :]
            assert not np.isnan(self.data[i]).any()

    def normalize_poses(self):
        """Normalize joint coordinates per sample."""
        for sample_index in range(len(self.data)):
            poses = self.data[sample_index].copy()
            channel_is_zero = np.all(poses == 0, axis=(1, 2))  # shape: (C,)
            non_zero_channels = np.where(~channel_is_zero)[0]
            if len(non_zero_channels) == 0:
                continue
            for channel_idx in non_zero_channels:
                channel_data = poses[channel_idx]  # shape: (T, V)
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                poses[channel_idx] = (channel_data - mean) / std
            self.data[sample_index] = poses

    def unify_left_right(self):
        """Flip left-side samples to match right-side coordinates for symmetry."""
        # for sample_index in range(len(self.data)):
        #     if self.sample_name[sample_index].split("_")[1] == 'l':
        #         self.data[sample_index][0, :, :] = self.data[sample_index][0, :, :] * (-1)

        if self.dataset_name == 'PD4T-Leg-Agility':
            for sample_index in range(len(self.data)):
                if self.sample_name[sample_index].split("_")[1] == 'l':
                    self.data[sample_index][0, :, :] = self.data[sample_index][0, :, :] * (-1)
        else:
            for sample_index in range(len(self.data)):
                if self.sample_name[sample_index].split("_")[-1] == 'left':
                    self.data[sample_index][0, :, :] = self.data[sample_index][0, :, :] * (-1)

    def unify_keypoints_and_channel(self):
        """Map keypoints to a unified skeleton structure based on dataset-specific rules."""
        for sample_index in range(len(self.data)):
            poses = self.data[sample_index].copy()
            if self.dataset_name == 'UI-PRMD':
                self.data[sample_index] = microsoft_kinect_v2_22keypoints(poses, self.data_dim)
            elif self.dataset_name == 'SSBD-Line-Gait':
                self.data[sample_index] = microsoft_kinect_v2_25keypoints_ssbd_3d(poses, self.data_dim)
            elif self.dataset_name in ['KIMORE', 'Walking-Treadmill-Gait'] or 'EHE' in self.dataset_name:
                self.data[sample_index] = microsoft_kinect_v2_25keypoints_xyz1(poses, self.data_dim)
            elif self.dataset_name in ['SPHERE-Sit', 'SPHERE-Stand'] or 'IRDS' in self.dataset_name:
                self.data[sample_index] = microsoft_kinect_v2_25keypoints_xyz2(poses, self.data_dim)
            elif self.dataset_name == 'TRSP-Seated-Motion':
                self.data[sample_index] = microsoft_kinect_v2_25keypoints_xyz3(poses, self.data_dim)
            elif self.dataset_name == 'UMONS-TAICHI':
                self.data[sample_index] = microsoft_kinect_v2_25keypoints_zxy(poses, self.data_dim)
            elif self.dataset_name in ['MMFS', 'PD-Round-Gait']:
                self.data[sample_index] = hrnet_17keypoints(poses, self.data_dim)
            elif self.dataset_name == '3D-Yoga' or 'FMS' in self.dataset_name:
                self.data[sample_index] = microsoft_kinect_azure_32keypoints(poses, self.data_dim)
            elif self.dataset_name == 'Push-Up':
                self.data[sample_index] = learnable_triangulation_17keypoints(poses, self.data_dim)
            elif self.dataset_name in ['AGF-Olympics',
                                       'Rhythmic-Gymnastics'] or 'PD4T' in self.dataset_name or 'center' in self.dataset_name:
                self.data[sample_index] = openpose_25keypoints(poses, self.data_dim)
            elif self.dataset_name == 'FineFS':
                self.data[sample_index] = human36M3D_17keypoints(poses, self.data_dim)
            elif self.dataset_name == 'PD-Walkway-Gait':
                self.data[sample_index] = track_3d_17keypoints(poses, self.data_dim)
            elif self.dataset_name in ['SPHERE-Stair-Gait', 'SPHERE-Surface-Gait']:
                self.data[sample_index] = asus_xmotion_15keypoints(poses, self.data_dim)

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def unify_time_length_and_prepocess(self):
        for sample_index in range(len(self.data)):
            poses = self.data[sample_index].copy()
            poses = np.array(poses)[:, :, :, 0].transpose(1, 2, 0)
            T, V, C = poses.shape
            self.p=0.5
            if self.stage == 'train':
                random.random()
                # Temporal Sampling
                data = np.zeros((self.data_length, poses.shape[1], 3))
                length = poses.shape[0]
                random_idx = random.sample(list(np.arange(length)) * self.data_length, self.data_length)
                random_idx.sort()
                data[:, :, :] = poses[random_idx, :, :]
                index_t = 2 * np.array(random_idx).astype(np.float32) / length - 1

                # Affine Transformation
                def affine_transform(value):
                    T, V, C = value.shape
                    assert C == 2
                    angle = np.random.uniform(-15, 15) * np.pi / 180
                    cos_val, sin_val = np.cos(angle), np.sin(angle)
                    rotation = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
                    scale = np.random.uniform(0.9, 1.1)
                    scale_matrix = np.eye(2) * scale
                    translation = np.random.uniform(-0.1, 0.1, size=(1, 1, 2))
                    value = np.matmul(value, rotation.T)
                    value = np.matmul(value, scale_matrix)
                    value += translation
                    return value

                if random.random() < self.p:
                    poses[:,:,:2] = affine_transform(poses[:,:,:2])

                # Temporal Jitter
                def temporal_jitter(value, max_jitter=3):
                    T = value.shape[0]
                    jittered = np.zeros_like(value)
                    for t in range(T):
                        offset = np.random.randint(-max_jitter, max_jitter + 1)
                        new_t = min(max(t + offset, 0), T - 1)
                        jittered[t] = value[new_t]
                    return jittered
                # Apply slight temporal perturbation to increase “temporal ambiguity” in motion representation
                if random.random() < self.p:
                    poses = temporal_jitter(poses)
                # Temporal Reversal
                if random.random() < self.p:
                    poses = poses[::-1]

                # Axis Masking
                if random.random() < self.p:
                    axis_next = random.randint(0, 1)
                    data[:, :, axis_next] = 0
                    data[:, :, 2] = 0

                # Joint Masking
                if random.random() < self.p:
                    T, V, C = data.shape
                    # joint_count = random.randint(1, 11)
                    joint_count = 0
                    joints_to_drop = random.sample(range(V), joint_count)
                    frame_count = random.randint(1, 16)
                    frames_to_drop = random.sample(range(T), frame_count)
                    data[np.ix_(frames_to_drop, joints_to_drop)] = 0

                # Temporal Block Dropout
                if random.random() < self.p:
                    block_size = random.randint(4, 16)
                    start = random.randint(0, self.data_length - block_size)
                    data[start:start + block_size, :, :] = 0

            elif self.stage=='test':
                random.random()
                data = np.zeros((self.data_length, poses.shape[1], 3))
                length = poses.shape[0]
                idx = np.linspace(0, length - 1, self.data_length).astype(int)
                data[:, :, :] = poses[idx, :, :]
                index_t = 2 * idx.astype(np.float32) / length - 1
            # Transpose data dimensions to (C, T, V)
            data = np.transpose(data, (2, 0, 1))
            C, T, V = data.shape
            # Add an extra dimension, shape becomes (C, T, V, 1)
            data = np.reshape(data, (C, T, V, 1))
            if self.partition:
                data = data[:, :, self.new_idx]
            self.data[sample_index] = data
            self.index_list[sample_index] = index_t

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
