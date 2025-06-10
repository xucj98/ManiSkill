from typing import Tuple

import numpy as np
import cv2
import torch

from odpc.utils.math import pose_to_matrix, project_points_to_pixels
from odpc.data.obs_processors.base_processor import BaseObsProcessor

def crop_image_around_pixel(
        image: np.ndarray, 
        center_pixel: np.ndarray, # (x, y) or (u, v)
        crop_size: tuple,       # (crop_height, crop_width)
        padding_value: int = 0,
) -> np.ndarray:
    """
    Crops an image around a center pixel with padding if out of bounds.
    Args:
        image: Input image (H, W, C) or (H, W).
        center_pixel: (x, y) coordinates for the center of the crop.
        crop_size: Desired (height, width) of the cropped image.
        padding_value: Value to use for padding.
    Returns:
        Cropped image.
    """
    img_h, img_w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    center_x, center_y = int(round(center_pixel[0])), int(round(center_pixel[1]))

    x_start = center_x - crop_w // 2
    x_end = x_start + crop_w
    y_start = center_y - crop_h // 2
    y_end = y_start + crop_h

    # Calculate padding needed
    pad_left = max(0, -x_start)
    pad_right = max(0, x_end - img_w)
    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_end - img_h)

    # Pad the image
    if image.ndim == 3:
        padding_dims = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else: # Grayscale image
        padding_dims = ((pad_top, pad_bottom), (pad_left, pad_right))
    
    padded_image = np.pad(
        image,
        padding_dims,
        mode='constant',
        constant_values=padding_value
    )

    # Calculate crop coordinates in the padded image
    crop_x_start_padded = x_start + pad_left
    crop_y_start_padded = y_start + pad_top
    
    cropped_image = padded_image[
        crop_y_start_padded : crop_y_start_padded + crop_h,
        crop_x_start_padded : crop_x_start_padded + crop_w,
    ]
    
    # Ensure the output has the exact crop_size, even if padding was extensive
    # This should be guaranteed by the np.pad and slicing logic if crop_size is positive
    if cropped_image.shape[0] != crop_h or cropped_image.shape[1] != crop_w:
        # This might happen if crop_size is larger than image + padding, 
        # or due to rounding issues with very small images.
        # Fallback: create an empty image of correct size and try to place current crop
        # For simplicity, we assume crop_size is reasonable.
        # A robust solution might involve resizing or further padding the `cropped_image`
        # to meet `crop_size` exactly.
        # However, the current padding logic should make this rare for typical scenarios.
        # Let's add a resize as a fallback if shapes don't match exactly,
        # but ideally, the padding and slicing are precise.
        if image.ndim == 3:
            target_shape = (crop_h, crop_w, image.shape[2])
        else:
            target_shape = (crop_h, crop_w)
        
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0: # Crop was entirely outside
            return np.full(target_shape, padding_value, dtype=image.dtype)

        # If still not matching, resize (this is a fallback, ideally padding is perfect)
        # print(f"Warning: Cropped image shape {cropped_image.shape} differs from target {target_shape}. Resizing.")
        # This resize should ideally not be needed if padding math is perfect.
        # It's more likely an issue if the center_pixel is way off or crop_size is huge.
        final_crop = cv2.resize(cropped_image, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST if image.dtype==np.uint8 or image.dtype==np.uint16 or image.dtype==np.uint32 else cv2.INTER_LINEAR)
        if final_crop.ndim == 2 and image.ndim == 3: # Handle grayscale resize to color
            final_crop = np.expand_dims(final_crop, axis=-1)
        return final_crop


    return cropped_image


class TcpCropProcessor(BaseObsProcessor):
    def __init__(
            self, 
            crop_size: tuple = (128, 128),    # (H, W) for RGB crop
            padding_value: int = 0,
            camera_name: str = "base_camera",
            output_postfix: str = "_crop",
    ):
        """
        Args:
            crop_size: The size of the cropped image.
            padding_value: The value to use for padding.
            camera_name: The name of the camera to crop.
            output_postfix: The postfix to add to the cropped image keys.
        """
        super().__init__()
        self.crop_size = crop_size
        self.padding_value = padding_value
        self.camera_name = camera_name
        self.output_postfix = output_postfix

    def _get_tcp_in_cam_coords(
            self, 
            world_tcp_pose_t: np.ndarray, 
            cam_world_pose_t: np.ndarray
    ) -> np.ndarray:
        """Calculates TCP 3D position in camera coordinates for a single timestep t."""
        T_world_tcp = pose_to_matrix(world_tcp_pose_t)
        T_cam_world = pose_to_matrix(cam_world_pose_t)
        T_cam_tcp = T_cam_world @ T_world_tcp
        tcp_pos_cam = T_cam_tcp[:3, 3]
        return tcp_pos_cam

    def _process_frame(
            self, 
            obs_frame: dict,  
            extra_frame: dict,
            cam_intrinsics: np.ndarray,
    ) -> dict:
        """
        Processes a single frame of observation data.
        """
        processed = obs_frame.copy() # Start with a copy

        tcp_pose_world = extra_frame["tcp_pose"] # (7,)
        cam_pose_world = extra_frame["cam0_world_pose"] # (7,)
        tcp_pos_in_cam_frame = self._get_tcp_in_cam_coords(tcp_pose_world, cam_pose_world) # (3,)
        tcp_pixel_coords_arr = project_points_to_pixels(tcp_pos_in_cam_frame.reshape(1,3), cam_intrinsics) # (1,2)
        
        if np.isnan(tcp_pixel_coords_arr).any():
            print(f"Warning: TCP projected to NaN pixel coordinates (possibly behind camera or Z_cam too small). Using image center.")
            # Fallback: use image center if projection fails
            img_h, img_w = obs_frame["rgb"].shape[:2]
            tcp_pixel_coords = np.array([img_w / 2, img_h / 2], dtype=np.float32)
        else:
            tcp_pixel_coords = tcp_pixel_coords_arr[0] # (2,) [u,v] or [x,y]

        for modality in obs_frame.keys():
            cropped = crop_image_around_pixel(
                obs_frame[modality], 
                tcp_pixel_coords, 
                self.crop_size, 
                self.padding_value
            )
            processed[modality + self.output_postfix] = cropped
    
        return processed

    def _process_traj(self, obs: dict) -> dict:
        """
        Processes a full trajectory (multiple timesteps).
        """
        res = obs["sensor_data"][self.camera_name]
        num_timesteps = res["rgb"].shape[0]
        
        all_cropped = []

        for t in range(num_timesteps):
            # Construct per-frame dictionaries to pass to process_frame
            obs_frame_t = {
                modality: data[t] 
                for modality, data in res.items()
            }
            
            extra_frame_t = {
                key: value[t] for key, value in obs["extra"].items()
            }

            # Process this single frame
            processed_frame_data = self._process_frame(
                obs_frame_t, extra_frame_t, obs["sensor_param"][self.camera_name]["intrinsic_cv"][t])
            
            # Collect the results
            all_cropped.append(processed_frame_data)

        # Stack the list of cropped images into a single NumPy array (t, ch, cw, C)
        for modality, data in all_cropped[0].items():
            stacked_cropped = np.stack([cropped[modality] for cropped in all_cropped], axis=0)
            res[modality] = stacked_cropped
            
        return res

    def process(self, obs: dict) -> dict:
        """
        Processes a full trajectory (multiple timesteps).
        Args:
            obs: The input dictionary structure as you provided.
                "sensor_data": {
                    "base_camera": {
                        "rgb": {"shape": ("t", 256, 256, 3), "dtype": "uint8"},
                        "depth": {"shape": ("t", 256, 256, 1), "dtype": "int16"},
                        "segmentation": {"shape": ("t", 256, 256, 1), "dtype": "int32"},
                    }
                },
                "sensor_param": {
                    "base_camera": {
                        "intrinsic_cv": {"shape": ("t", 3, 3), "dtype": "float32"},
                    }
                },
                "extra": {
                    "tcp_pose": {"shape": ("t", 7), "dtype": "float32"},
                    "cam0_world_pose": {"shape": ("t", 7), "dtype": "float32"},
                }
        Returns:
            A new obs dictionary with added cropped images for each frame.
                "sensor_data": {
                    "base_camera": {
                        "rgb": {"shape": ("t", 256, 256, 3), "dtype": "uint8"},
                        "rgb_crop": {"shape": ("t", 128, 128, 3), "dtype": "uint8"},
                        "depth": {"shape": ("t", 256, 256, 1), "dtype": "int16"},
                        "depth_crop": {"shape": ("t", 128, 128, 1), "dtype": "int16"},
                        "segmentation": {"shape": ("t", 256, 256, 1), "dtype": "int32"},
                        "segmentation_crop": {"shape": ("t", 128, 128, 1), "dtype": "int32"},
                    }
                },
                "sensor_param": {
                    "base_camera": {
                        "intrinsic_cv": {"shape": ("t", 3, 3), "dtype": "float32"},
                    }
                },
                "extra": {
                    "tcp_pose": {"shape": ("t", 7), "dtype": "float32"},
                    "cam0_world_pose": {"shape": ("t", 7), "dtype": "float32"},
                }
        """
        rgb = obs["sensor_data"][self.camera_name]["rgb"]
        if isinstance(rgb, torch.Tensor):
            obs = self.to_numpy(obs)

        if rgb.ndim == 4:
            obs["sensor_data"][self.camera_name] = self._process_traj(obs)
            if isinstance(rgb, torch.Tensor):
                obs = self.to_tensor(obs, rgb.device)
            return obs
        elif rgb.ndim == 3:
            raise NotImplementedError("Single frame processing not implemented yet")
        elif rgb.ndim == 5:
            num_batchs = rgb.shape[0]
            all_cropped = []
            for i in range(num_batchs):
                obs_traj = {
                    "sensor_data": {
                        self.camera_name: {
                            modality: data[i]
                            for modality, data in obs["sensor_data"][self.camera_name].items()
                        }
                    },
                    "sensor_param": {
                        self.camera_name: {
                            "intrinsic_cv": obs["sensor_param"][self.camera_name]["intrinsic_cv"][i]
                        }
                    },
                    "extra": {
                        "tcp_pose": obs["extra"]["tcp_pose"][i],
                        "cam0_world_pose": obs["extra"]["cam0_world_pose"][i],
                    }
                }
                all_cropped.append(self._process_traj(obs_traj))
            for modality, data in all_cropped[0].items():
                stacked_cropped = np.stack([cropped[modality] for cropped in all_cropped], axis=0)
                obs["sensor_data"][self.camera_name][modality] = stacked_cropped
            if isinstance(rgb, torch.Tensor):
                obs = self.to_tensor(obs, rgb.device)
            return obs
        raise NotImplementedError("Unsupported number of dimensions for rgb")
