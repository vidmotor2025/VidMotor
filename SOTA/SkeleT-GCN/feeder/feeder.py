import pickle
from .pose_utils import *
from torch.utils.data import Dataset


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

    def unify_left_right(self):
        """Flip left-side samples to match right-side coordinates for symmetry."""
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

    def unify_time_length(self, p_interval=1, seed=289800):
        """Uniformly sample frames from the video."""
        if self.stage is None:
            return

        if not isinstance(p_interval, tuple):
            p_interval = (p_interval, p_interval)
        for sample_index in range(len(self.data)):
            poses = self.data[sample_index].copy()  # C,T,V,M
            num_frames = poses.shape[1]

            if self.stage == 'train':
                allinds = []
                for clip_idx in range(self.num_clips):
                    old_num_frames = num_frames
                    pi = p_interval
                    ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
                    num_frames = int(ratio * num_frames)
                    off = np.random.randint(old_num_frames - num_frames + 1)
                    if num_frames < self.data_length:
                        start = np.random.randint(0, num_frames)
                        inds = np.arange(start, start + self.data_length)
                    elif self.data_length <= num_frames < 2 * self.data_length:
                        basic = np.arange(self.data_length)
                        inds = np.random.choice(
                            self.data_length + 1, num_frames - self.data_length, replace=False)
                        offset = np.zeros(self.data_length + 1, dtype=np.int64)
                        offset[inds] = 1
                        offset = np.cumsum(offset)
                        inds = basic + offset[:-1]
                    else:
                        bids = np.array(
                            [i * num_frames // self.data_length for i in range(self.data_length + 1)])
                        bsize = np.diff(bids)
                        bst = bids[:self.data_length]
                        offset = np.random.randint(bsize)
                        inds = bst + offset
                    inds = inds + off
                    num_frames = old_num_frames
                    allinds.append(inds)
                allinds = np.concatenate(allinds)
            elif self.stage == 'test':
                np.random.seed(seed)
                allinds = []
                for i in range(self.num_clips):
                    old_num_frames = num_frames
                    pi = p_interval
                    ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
                    num_frames = int(ratio * num_frames)
                    off = np.random.randint(old_num_frames - num_frames + 1)
                    if num_frames < self.data_length:
                        start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                        inds = np.arange(start_ind, start_ind + self.data_length)
                    elif self.data_length <= num_frames < self.data_length * 2:
                        basic = np.arange(self.data_length)
                        inds = np.random.choice(self.data_length + 1, num_frames - self.data_length, replace=False)
                        offset = np.zeros(self.data_length + 1, dtype=np.int64)
                        offset[inds] = 1
                        offset = np.cumsum(offset)
                        inds = basic + offset[:-1]
                    else:
                        bids = np.array([i * num_frames // self.data_length for i in range(self.data_length + 1)])
                        bsize = np.diff(bids)
                        bst = bids[:self.data_length]
                        offset = np.random.randint(bsize)
                        inds = bst + offset

                    allinds.append(inds + off)
                    num_frames = old_num_frames
                allinds = np.concatenate(allinds)
            allinds = np.mod(allinds, num_frames)
            poses = poses[:, allinds]
            assert poses.ndim == 4
            C, T, V, M = poses.shape
            assert T % self.num_clips == 0
            # nc,C,T,V,M
            self.data[sample_index] = poses.reshape(C, self.num_clips, T // self.num_clips, V, M).transpose(1, 0, 2, 3, 4)

    def top_k(self, score, top_k):
        """Evaluate top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
