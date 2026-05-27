from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import CubicSpline


def list_statistics(lst):
    mean_value = np.mean(lst)
    median_value = np.median(lst)
    percentile_75 = np.percentile(lst, 75)
    return mean_value, median_value, percentile_75


def hrnet_17keypoints(data, data_dim):
    keypoint_mapping = [0, 12, 14, 16, 11, 13, 15, 0, 0, 0, 0, 5, 7, 9, 6, 8, 10]
    # Select the key points according to the mapping
    processed_data = data[:, :, keypoint_mapping, :]
    processed_data[:, :, 0, :] = (processed_data[:, :, 1, :] + processed_data[:, :, 4, :]) / 2
    processed_data[:, :, 8, :] = (processed_data[:, :, 11, :] + processed_data[:, :, 14, :]) / 2
    processed_data[:, :, 9, :] = (processed_data[:, :, 8, :] + processed_data[:, :, 10, :]) / 2
    processed_data[:, :, 7, :] = (processed_data[:, :, 0, :] + processed_data[:, :, 8, :]) / 2
    if data_dim == '2D':
        # Keep only the first two dimensions (X, Y)
        processed_data = processed_data[:2, :, :, :]
        return processed_data
    elif data_dim == '3D':
        processed_data = processed_data[:2, :, :, :]
        zeros = np.zeros((1, processed_data.shape[1], processed_data.shape[2], processed_data.shape[3]))
        return np.concatenate((processed_data, zeros), axis=0)


def learnable_triangulation_17keypoints(data, data_dim):
    keypoint_mapping = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]
    processed_data = data[:, :, keypoint_mapping, :]
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        return processed_data


def track_3d_17keypoints(data, data_dim):
    if data_dim == '2D':
        processed_data = data[1:, :, :, :]
        processed_data = processed_data[::-1, :, :, :]
        return processed_data
    elif data_dim == '3D':
        assert data.shape[0] == 3
        processed_data = data[1:, :, :, :]
        processed_data = processed_data[::-1, :, :, :]
        processed_data = np.concatenate([processed_data, np.expand_dims(data[0], axis=0)], axis=0)
        return processed_data


def microsoft_kinect_v2_22keypoints(data, data_dim):
    keypoint_mapping = [0, 18, 19, 20, 14, 15, 16, 1, 2, 3, 4, 7, 8, 9, 11, 12, 13]
    processed_data = data[:, :, keypoint_mapping, :]
    processed_data[:, :, 8, :] = (processed_data[:, :, 11, :] + processed_data[:, :, 14, :]) / 2
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        return processed_data


def microsoft_kinect_v2_25keypoints_ssbd_2d(data, data_dim):
    keypoint_mapping = [19, 12, 14, 1, 13, 15, 2, 0, 20, 16, 11, 18, 4, 24, 17, 3, 23]
    processed_data = data[:, :, keypoint_mapping, :]
    if data_dim == '2D':
        return processed_data
    elif data_dim == '3D':
        zeros = np.zeros((1, processed_data.shape[1], processed_data.shape[2], processed_data.shape[3]))
        return np.concatenate((processed_data, zeros), axis=0)


def microsoft_kinect_v2_25keypoints_ssbd_3d(data, data_dim):
    keypoint_mapping = [19, 12, 14, 1, 13, 15, 2, 0, 20, 16, 11, 18, 4, 24, 17, 3, 23]
    processed_data = data[:, :, keypoint_mapping, :]
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        processed_data[1, :, :, :] *= -1
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        processed_data[1, :, :, :] *= -1
        return processed_data


def microsoft_kinect_v2_25keypoints_xyz1(data, data_dim):
    keypoint_mapping = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 4, 5, 6, 8, 9, 10]
    processed_data = data[:, :, keypoint_mapping, :]
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        processed_data[0, :, :, :] *= -1
        processed_data[1, :, :, :] *= -1
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        processed_data[0, :, :, :] *= -1
        processed_data[1, :, :, :] *= -1
        return processed_data


def microsoft_kinect_v2_25keypoints_xyz2(data, data_dim):
    keypoint_mapping = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 4, 5, 6, 8, 9, 10]
    processed_data = data[:, :, keypoint_mapping, :]
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        processed_data[0, :, :, :] *= -1
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        processed_data[0, :, :, :] *= -1
        return processed_data


def microsoft_kinect_v2_25keypoints_xyz3(data, data_dim):
    keypoint_mapping = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 4, 5, 6, 8, 9, 10]
    processed_data = data[:, :, keypoint_mapping, :]
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        processed_data[1, :, :, :] *= -1
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        processed_data[1, :, :, :] *= -1
        return processed_data


def microsoft_kinect_v2_25keypoints_zxy(data, data_dim):
    keypoint_mapping = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 4, 5, 6, 8, 9, 10]
    processed_data = data[:, :, keypoint_mapping, :]
    if data_dim == '2D':
        processed_data = processed_data[1:3, :, :, :]
        processed_data[1, :, :, :] *= -1
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        processed_data = np.concatenate([processed_data[1:3], processed_data[0:1]], axis=0)
        processed_data[1, :, :, :] *= -1
        return processed_data


def microsoft_kinect_azure_32keypoints(data, data_dim):
    keypoint_mapping = [0, 22, 23, 24, 18, 19, 20, 1, 2, 3, 26, 5, 6, 7, 12, 13, 14]
    processed_data = data[:, :, keypoint_mapping, :]
    processed_data[:, :, 8, :] = (processed_data[:, :, 11, :] + processed_data[:, :, 14, :]) / 2
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        return processed_data


def asus_xmotion_15keypoints(data, data_dim):
    keypoint_mapping = [8, 10, 12, 14, 9, 11, 13, 8, 1, 0, 0, 2, 4, 6, 3, 5, 7]
    processed_data = data[:, :, keypoint_mapping, :]
    processed_data[:, :, 0, :] = (processed_data[:, :, 1, :] + processed_data[:, :, 4, :]) / 2
    processed_data[:, :, 9, :] = (processed_data[:, :, 8, :] + processed_data[:, :, 10, :]) / 2
    if data_dim == '2D':
        processed_data = processed_data[:2, :, :, :]
        processed_data[0, :, :, :] *= -1
        processed_data[1, :, :, :] *= -1
        return processed_data
    elif data_dim == '3D':
        assert processed_data.shape[0] == 3
        processed_data[0, :, :, :] *= -1
        processed_data[1, :, :, :] *= -1
        return processed_data


def human36M3D_17keypoints(data, data_dim):
    if data_dim == '2D':
        return data[:2, :, :, :]
    elif data_dim == '3D':
        assert data.shape[0] == 3
        return data


def openpose_25keypoints(data, data_dim):
    keypoint_mapping = [8, 9, 10, 11, 12, 13, 14, 8, 1, 1, 0, 5, 6, 7, 2, 3, 4]
    processed_data = data[:, :, keypoint_mapping, :]
    processed_data[:, :, 7, :] = (processed_data[:, :, 0, :] + processed_data[:, :, 8, :]) / 2
    processed_data[:, :, 9, :] = (processed_data[:, :, 8, :] + processed_data[:, :, 10, :]) / 2
    if data_dim == '2D':
        return processed_data
    elif data_dim == '3D':
        zeros = np.zeros((1, processed_data.shape[1], processed_data.shape[2], processed_data.shape[3]))
        return np.concatenate((processed_data, zeros), axis=0)


def uniform_sample_wo_offset(data, target_length):
    # Divide the original n frames into m segments and compute the base position for each sample point.
    # i is the current sample index (from 0 to m-1), and n // m gives the length of each segment (integer division)
    f = lambda m, n: [i * n // m for i in range(m)]
    time_resampling_indexes = f(target_length, data.shape[1])
    return data[:, time_resampling_indexes, :, :]


def uniform_sample_w_offset(data, target_length):
    # https://github.com/bruceyo/EGCN/blob/master/tools/gen/kimore_read.py
    f = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
    # Slightly adjust each sample point to make the selection more uniform by adding n // (2 * m) to the base position of each point
    time_resampling_indexes = f(target_length, data.shape[1])
    return data[:, time_resampling_indexes, :, :]


def cut_fixed_length(data, target_length):
    return data[:, :target_length, :, :]


def pad_zero(data, target_length):
    pad_length = target_length - data.shape[1]
    return np.pad(data, ((0, 0), (0, pad_length), (0, 0), (0, 0)), mode='constant', constant_values=0)
    # Pad with zeros at the end to reach pad_length


def repeat_last(data, target_length):
    pad_length = target_length - data.shape[1]
    last_frame = data[:, -1:, :, :]
    return np.concatenate([data, np.tile(last_frame, (1, pad_length, 1, 1))], axis=1)
    # Concatenate duplicated frames


def resample_interp1d(data, target_length):
    # When exceeding the length, extrapolate using values beyond the original range
    interp_func = interp1d(np.arange(0, data.shape[1]), data, kind='linear', fill_value='extrapolate')
    return interp_func(np.linspace(0, data.shape[1] - 1, target_length))


def resample_cubic(data, target_length):
    spline = CubicSpline(np.arange(0, data.shape[1]), data, bc_type='natural')
    return spline(np.linspace(0, data.shape[1] - 1, target_length))
