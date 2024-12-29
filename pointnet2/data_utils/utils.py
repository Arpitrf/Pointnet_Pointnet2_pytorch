import torch
import collections
import h5py

import numpy as np
import open3d as o3d

def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of 
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be 
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert(list not in type_func_dict)
    assert(tuple not in type_func_dict)
    assert(dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(
                'Cannot handle data type %s' % str(type(x)))

def pad_sequence_single(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad input tensor or array @seq in the time dimension (dimension 1).

    Args:
        seq (np.ndarray or torch.Tensor): sequence to be padded
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (np.ndarray or torch.Tensor)
    """
    assert isinstance(seq, (np.ndarray, torch.Tensor))
    assert pad_same or pad_values is not None
    if pad_values is not None:
        assert isinstance(pad_values, float)
    repeat_func = np.repeat if isinstance(seq, np.ndarray) else torch.repeat_interleave
    concat_func = np.concatenate if isinstance(seq, np.ndarray) else torch.cat
    ones_like_func = np.ones_like if isinstance(seq, np.ndarray) else torch.ones_like
    seq_dim = 1 if batched else 0

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        pad = seq[[0]] if pad_same else ones_like_func(seq[[0]]) * pad_values
        begin_pad.append(repeat_func(pad, padding[0], seq_dim))
    if padding[1] > 0:
        pad = seq[[-1]] if pad_same else ones_like_func(seq[[-1]]) * pad_values
        end_pad.append(repeat_func(pad, padding[1], seq_dim))

    return concat_func(begin_pad + [seq] + end_pad, seq_dim)        


def pad_sequence(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad a nested dictionary or list or tuple of sequence tensors in the time dimension (dimension 1).

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (dict or list or tuple)
    """
    return recursive_dict_list_tuple_apply(
        seq,
        {
            torch.Tensor: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                pad_sequence_single(x, p, b, ps, pv),
            np.ndarray: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                pad_sequence_single(x, p, b, ps, pv),
            type(None): lambda x: x,
        }
    )

def generate_point_cloud_from_depth(depth_image, intrinsic_matrix, mask, extrinsic_matrix):
    """
    Generate a point cloud from a depth image and intrinsic matrix.
    
    Parameters:
    - depth_image: np.array, HxW depth image (in meters).
    - intrinsic_matrix: np.array, 3x3 intrinsic matrix of the camera.
    
    Returns:
    - point_cloud: Open3D point cloud.
    """
    
    # Get image dimensions
    height, width = depth_image.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the pixel coordinates and depth values
    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth_image.flatten()
    mask_flat = mask.flatten()

    # Filter points where the mask is 1
    valid_indices = np.where(mask_flat == 1)

    # Apply the mask to the pixel coordinates and depth
    u_valid = u_flat[valid_indices]
    v_valid = v_flat[valid_indices]
    depth_valid = depth_flat[valid_indices]

    # fix to handle inf depth values (which leads to nan)
    depth_valid[depth_valid == np.inf] = 3.0
    # print("depth_valid: ", depth_valid)

    # Generate normalized pixel coordinates in homogeneous form
    pixel_coords = np.vstack((u_valid, v_valid, np.ones_like(u_valid)))

    # Compute inverse intrinsic matrix
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)

    # Apply the inverse intrinsic matrix to get normalized camera coordinates
    cam_coords = intrinsic_inv @ pixel_coords

    # Multiply by depth to get 3D points in camera space
    cam_coords *= depth_valid

    # # Reshape the 3D coordinates
    # x = cam_coords[0].reshape(height, width)
    # y = cam_coords[1].reshape(height, width)
    # z = depth_image

    # # Stack the coordinates into a single 3D point array
    # points = np.dstack((x, y, z)).reshape(-1, 3)

    points = np.vstack((cam_coords[0], cam_coords[1], depth_valid)).T

    # pad points so that the total number of points are 128*128
    target_size=(height*width, 3)
    N_i = points.shape[0]
    pad_rows = target_size[0] - N_i
    padding = ((0, pad_rows), (0, 0))
    points = np.pad(points, padding, mode='edge')

    # print("points shape: ", points.shape)

    # remove later
    # points = points[points[:, 2] > 0.5]
    # print("points: ", points[:, 2])

    # transform points to world frame
    # make points homogeneous
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = extrinsic_matrix @ points.T
    points = points.T
    # remove homogeneous coordinate
    points = points[:, :3]

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud