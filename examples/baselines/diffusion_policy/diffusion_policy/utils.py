import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from h5py import Dataset, File, Group
from torch.utils.data.sampler import Sampler
from typing import Union


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(raw_data["traj_0"]["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor, depth = True):
    img_dict = obs["sensor_data"]
    ls = ["rgb"]
    if depth:
        ls = ["rgb", "depth"]

    new_img_dict = {
        key: transpose_fn(
            concat_fn([v[key] for v in img_dict.values()])
        )  # (C, H, W) or (B, C, H, W)
        for key in ls
    }
    if "depth" in new_img_dict and isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
    }

    if "depth" in new_img_dict:
        out_dict["depth"] = new_img_dict["depth"]


    return out_dict


def build_obs_space(env, depth_dtype, state_obs_extractor):
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(
                -float("inf"), float("inf"), shape=(n_images, h, w), dtype=depth_dtype
            ),
        }
    )


def build_state_obs_extractor(env_id):
    # NOTE: You can tune/modify state observations specific to each environment here as you wish. By default we include all data
    # but in some use cases you might want to exclude e.g. obs["agent"]["qvel"] as qvel is not always something you query in the real world.
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())


# visualize object pose
def visualize_pose(
        rgb: Union[torch.Tensor, np.ndarray],
        pose: torch.Tensor,
        intrinsic: torch.Tensor,
        axes_len: float = 0.1
) -> np.ndarray:
    """
    Visualizes an object's pose in the camera coordinate system on an RGB image.

    Args:
        rgb (torch.Tensor): The original image, shape (3, H, W), RGB order.
                            Assumed to be in [0, 1] float or [0, 255] uint8.
        pose (torch.Tensor): Object's pose in camera coordinates, shape (7,).
                             Represents (tx, ty, tz, qw, qx, qy, qz) where
                             (tx, ty, tz) is position and (qw, qx, qy, qz) is WXYZ quaternion.
        intrinsic (torch.Tensor): Camera intrinsic matrix, shape (3, 3).
                                  [[fx, 0,  cx],
                                   [0,  fy, cy],
                                   [0,  0,  1 ]]
        axes_len (float): Length of the coordinate axes to be drawn for the object.

    Returns:
        np.ndarray: Image with the axes drawn, shape (H, W, 3), BGR order (for cv2.imshow).
    """

    if isinstance(rgb, torch.Tensor):
        img_tensor_chw = rgb.cpu()
        img_np_hwc_rgb = img_tensor_chw.permute(1, 2, 0).numpy()
    else:
        img_np_hwc_rgb = rgb

    # 1. Convert input tensors to NumPy arrays on CPU
    pose_np = pose.cpu().numpy()
    intrinsic_np = intrinsic.cpu().numpy()

    # 2. Prepare the image:
    #    - Permute to HWC
    #    - Convert to NumPy array
    #    - Normalize to 0-255 uint8 if it's float
    #    - Convert RGB to BGR for OpenCV drawing

    # if img_np_hwc_rgb.dtype == np.float32 or img_np_hwc_rgb.dtype == np.float64:
    #     if img_np_hwc_rgb.max() <= 1.0:  # Assuming 0-1 range for float
    #         img_np_hwc_rgb = (img_np_hwc_rgb * 255).astype(np.uint8)
    #     else:  # Assuming 0-255 range for float
    #         img_np_hwc_rgb = img_np_hwc_rgb.astype(np.uint8)
    # elif img_np_hwc_rgb.dtype != np.uint8:  # Other types, try to convert
    #     img_np_hwc_rgb = img_np_hwc_rgb.astype(np.uint8)


    # Make a writable copy to draw on
    output_img_rgb = img_np_hwc_rgb.copy()

    # 3. Parse pose: extract translation and rotation
    position = pose_np[:3]  # tx, ty, tz
    quaternion_wxyz = pose_np[3:]  # qw, qx, qy, qz

    # Scipy's Rotation expects quaternion in (x, y, z, w) order
    quaternion_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
    try:
        rotation_matrix = R_scipy.from_quat(quaternion_xyzw).as_matrix()
    except Exception as e:
        print(f"Error converting quaternion: {e}. Quaternion was: {quaternion_xyzw}")
        # Return original image if pose is invalid
        return img_np_hwc_rgb

    # 4. Define 3D axes points in the object's local coordinate system
    #    Origin, X-end, Y-end, Z-end
    axes_points_object = np.array([
        [0, 0, 0],  # Origin
        [axes_len, 0, 0],  # X-axis endpoint
        [0, axes_len, 0],  # Y-axis endpoint
        [0, 0, axes_len]  # Z-axis endpoint
    ], dtype=np.float32)

    # 5. Transform these points to the camera coordinate system
    #    P_camera = R * P_object + t
    axes_points_camera = (rotation_matrix @ axes_points_object.T).T + position

    # 6. Project 3D points from camera coordinates to 2D image plane
    #    p_image_homogeneous = K @ P_camera
    #    (u, v) = (p_image_homogeneous[0]/p_image_homogeneous[2], p_image_homogeneous[1]/p_image_homogeneous[2])

    projected_points_2d_list = []
    valid_projection_mask = []

    for point_3d_cam in axes_points_camera:
        # Check if point is in front of the camera (Z > 0)
        if point_3d_cam[2] <= 1e-5:  # Add a small epsilon for stability
            valid_projection_mask.append(False)
            projected_points_2d_list.append(np.array([-1, -1], dtype=int))  # Placeholder for invalid points
            continue

        valid_projection_mask.append(True)
        # P_camera is (X, Y, Z)^T
        # K @ P_camera results in (u*Z, v*Z, Z)^T
        uvw = intrinsic_np @ point_3d_cam.reshape(3, 1)
        u = uvw[0, 0] / uvw[2, 0]
        v = uvw[1, 0] / uvw[2, 0]
        projected_points_2d_list.append(np.array([int(round(u)), int(round(v))], dtype=int))

    projected_points_2d = np.array(projected_points_2d_list)

    # 7. Draw the axes on the image
    #    Colors: X=Red, Y=Green, Z=Blue (BGR format for OpenCV)
    colors = {
        "x": (255, 0, 0),  # Red
        "y": (0, 255, 0),  # Green
        "z": (0, 0, 255)  # Blue
    }
    thickness = 1  # You can make this a parameter

    origin_2d = tuple(projected_points_2d[0])

    # Only draw if origin is validly projected
    if valid_projection_mask[0]:
        # Draw X-axis (Origin to X-endpoint)
        if valid_projection_mask[1]:
            x_axis_2d = tuple(projected_points_2d[1])
            cv2.line(output_img_rgb, origin_2d, x_axis_2d, colors["x"], thickness)

        # Draw Y-axis (Origin to Y-endpoint)
        if valid_projection_mask[2]:
            y_axis_2d = tuple(projected_points_2d[2])
            cv2.line(output_img_rgb, origin_2d, y_axis_2d, colors["y"], thickness)

        # Draw Z-axis (Origin to Z-endpoint)
        if valid_projection_mask[3]:
            z_axis_2d = tuple(projected_points_2d[3])
            cv2.line(output_img_rgb, origin_2d, z_axis_2d, colors["z"], thickness)
    else:
        print("Warning: Object origin is behind or too close to the camera. Axes not drawn.")

    return output_img_rgb
