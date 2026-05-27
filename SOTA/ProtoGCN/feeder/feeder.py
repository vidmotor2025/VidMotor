import pickle
from .pose_utils import *
from torch.utils.data import Dataset
import math


class Feeder(Dataset):
    """Feeder for loading, preprocessing, and serving datasets."""

    def __init__(self, dataset_info, data_centered, data_dim, data_norm, score_norm, label_type, data_length=100,
                 data_indices=None, stage=None, num_clips=1):
        self.dataset_name = dataset_info['name']
        self.data_path = dataset_info['data_path']
        self.data_centered = data_centered
        self.data_dim = data_dim
        assert self.data_dim == '3D'
        self.data_norm = data_norm
        self.label_type = label_type
        self.data_length = data_length
        self.indices = data_indices
        self.stage = stage
        self.num_clips = num_clips
        # Load raw data
        self.load_data()
        # Preprocess labels if normalization is required
        if self.stage != None:
            if score_norm:
                self.min_score = float(dataset_info['score_range']['min'])
                self.max_score = float(dataset_info['score_range']['max'])
                self.score_trend = dataset_info['score_trend']
                self.normalize_scores()  # Min-Max normalization
                if not self.score_trend:
                    self.unify_scores_trend()
            # Preprocess pose data
            self.unify_keypoints_and_channel()  # Map keypoints to a unified skeleton structure
            self.pre_normalize_3d()
            self.random_rot()
            self.unify_time_length()  # Pad or downsample sequences to fixed length
            if 'Leg-Agility' in self.dataset_name:
                self.unify_left_right()  # Flip left-side samples for consistency
            if self.data_centered == 7:
                self.center_poses()  # Center poses around a reference joint
            if self.data_norm == 'zscore':
                self.normalize_poses()  # Normalize joint coordinates per sample

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

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.label)

    def __getitem__(self, index):
        """Return a single sample."""
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        return data_numpy, label, index

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
            self.data[i] -= self.data[i][:, :, 0, self.data_centered:self.data_centered + 1][:, :, None, :]

    def normalize_poses(self):
        """Normalize joint coordinates per sample."""
        for sample_index in range(len(self.data)):
            sample = self.data[sample_index]
            for n in range(sample.shape[0]):
                poses = sample[n].copy()
                mask_last_channel = np.all(poses[-1] == 0)
                poses_to_normalize = poses[:-1] if mask_last_channel else poses
                means = np.mean(poses_to_normalize, axis=(1, 2), keepdims=True)
                stds = np.std(poses_to_normalize, axis=(1, 2), keepdims=True)
                poses_to_normalize = (poses_to_normalize - means) / stds
                # If the last channel contains all zeros, it is left unchanged.
                if mask_last_channel:
                    self.data[sample_index][n, :-1] = poses_to_normalize
                else:
                    self.data[sample_index][n] = poses_to_normalize

    def pre_normalize_3d(self):
        for sample_index in range(len(self.data)):
            poses = self.data[sample_index].copy()  # C,T,V,M
            poses = poses.transpose(3, 1, 2, 0)  # M,T,V,C
            self.align_center = False
            self.align_spine = False
            if self.align_center:
                main_body_center = poses[0, 0, 8].copy()
                mask = ((poses != 0).sum(-1) > 0)[..., None]
                poses = (poses - main_body_center) * mask

            def unit_vector(vector):
                """Returns the unit vector of the vector. """
                return vector / np.linalg.norm(vector)

            def angle_between(v1, v2):
                """Returns the angle in radians between vectors 'v1' and 'v2'. """
                if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
                    return 0
                v1_u = unit_vector(v1)
                v2_u = unit_vector(v2)
                return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

            def rotation_matrix(axis, theta):
                """Return the rotation matrix associated with counterclockwise rotation
                about the given axis by theta radians."""
                if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
                    return np.eye(3)
                axis = np.asarray(axis)
                axis = axis / np.sqrt(np.dot(axis, axis))
                a = np.cos(theta / 2.0)
                b, c, d = -axis * np.sin(theta / 2.0)
                aa, bb, cc, dd = a * a, b * b, c * c, d * d
                bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
                return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

            if self.align_spine:
                self.zaxis = [0, 7]
                self.xaxis = [14, 11]
                joint_bottom = poses[0, 0, self.zaxis[0]]
                joint_top = poses[0, 0, self.zaxis[1]]
                axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
                angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
                matrix_z = rotation_matrix(axis, angle)
                poses = np.einsum('abcd,kd->abck', poses, matrix_z)

                joint_rshoulder = poses[0, 0, self.xaxis[0]]
                joint_lshoulder = poses[0, 0, self.xaxis[1]]
                axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
                angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
                matrix_x = rotation_matrix(axis, angle)
                poses = np.einsum('abcd,kd->abck', poses, matrix_x)

            poses = poses.transpose(3, 1, 2, 0)  # C,T,V,M
            self.data[sample_index] = poses

    def random_rot(self):
        self.theta = 0.3

        def _rot3d(theta):
            cos, sin = np.cos(theta), np.sin(theta)
            rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
            ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
            rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

            rot = np.matmul(rz, np.matmul(ry, rx))
            return rot

        def _rot2d(theta):
            cos, sin = np.cos(theta), np.sin(theta)
            return np.array([[cos, -sin], [sin, cos]])

        for sample_index in range(len(self.data)):
            poses = self.data[sample_index].copy()  # C,T,V,M
            poses = poses.transpose(3, 1, 2, 0)  # M,T,V,C
            M, T, V, C = poses.shape
            assert C == 3
            if np.allclose(poses[..., 2], 0):
                theta = np.random.uniform(-self.theta)
                rot_mat = _rot2d(theta)
                poses[..., :-1]=np.einsum('ab,mtvb->mtva', rot_mat, poses[..., :-1])
            else:
                theta = np.random.uniform(-self.theta, self.theta, size=3)
                rot_mat = _rot3d(theta)
                poses=np.einsum('ab,mtvb->mtva', rot_mat, poses)
            poses = poses.transpose(3, 1, 2, 0)  # C,T,V,M
            self.data[sample_index] = poses

    def unify_left_right(self):
        """Flip left-side samples to match right-side coordinates for symmetry."""
        # for sample_index in range(len(self.data)):
        #     if self.sample_name[sample_index].split("_")[1] == 'l':
        #         self.data[sample_index][0, :, :] = self.data[sample_index][0, :, :] * (-1)

        if self.dataset_name == 'PD4T-Leg-Agility':
            for sample_index in range(len(self.data)):
                if self.sample_name[sample_index].split("_")[1] == 'l':
                    self.data[sample_index][:, 0, :, :] = self.data[sample_index][:, 0, :, :] * (-1)
        else:
            for sample_index in range(len(self.data)):
                if self.sample_name[sample_index].split("_")[-1] == 'left':
                    self.data[sample_index][:, 0, :, :] = self.data[sample_index][:, 0, :, :] * (-1)

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

    def unify_time_length(self, p_interval=1, seed=255):
        """Pad or downsample sequences to a fixed temporal length."""
        if self.stage is None:
            return

        if not isinstance(p_interval, tuple):
            p_interval = (p_interval, p_interval)

        def _get_clips(full_kp, clip_len):
            # M, T, V, C = full_kp.shape
            C, T, V, M = full_kp.shape
            clips = []

            for clip_idx in range(self.num_clips):
                pi = p_interval
                ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
                num_frames = int(ratio * T)
                off = np.random.randint(T - num_frames + 1)

                if num_frames < clip_len:
                    start = np.random.randint(0, num_frames)
                    inds = (np.arange(start, start + clip_len) % num_frames) + off
                    clip = full_kp[:, inds].copy()
                elif clip_len <= num_frames < 2 * clip_len:
                    basic = np.arange(clip_len)
                    inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                    offset = np.zeros(clip_len + 1, dtype=np.int64)
                    offset[inds] = 1
                    inds = basic + np.cumsum(offset)[:-1] + off
                    clip = full_kp[:, inds].copy()
                else:
                    bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                    bsize = np.diff(bids)
                    bst = bids[:clip_len]
                    offset = np.random.randint(bsize)
                    inds = bst + offset + off
                    clip = full_kp[:, inds].copy()
                clips.append(clip)
            return np.concatenate(clips, 1)

        for sample_index in range(len(self.data)):
            poses = self.data[sample_index].copy()
            if self.stage=='test':
                np.random.seed(seed)

            poses=_get_clips(poses,self.data_length)
            assert poses.ndim == 4
            C, T, V, M = poses.shape
            assert T % self.num_clips==0
            # nc,C,T,V,M
            self.data[sample_index] = poses.reshape(C, self.num_clips, T // self.num_clips, V, M).transpose(1, 0, 2, 3, 4)

    def top_k(self, score, top_k):
        """Evaluate top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
