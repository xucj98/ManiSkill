from typing import Tuple

import numpy as np
import cv2

from odpc.utils.math import pose_to_matrix, project_points_to_pixels


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


class ObsPreprocessor:
    def __init__(
            self, 
            crop_size: tuple = (128, 128),    # (H, W) for RGB crop
            padding_value: int = 0,
            camera_name: str = "base_camera",
    ):
        self.crop_size = crop_size
        self.padding_value = padding_value
        self.camera_name = camera_name
       
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

    def process_frame(
            self, 
            obs_frame: dict, 
            sensor_params_frame: dict, 
            extra_frame: dict
    ) -> dict:
        """
        Processes a single frame of observation data.
        obs_frame: Contains sensor_data for one timestep.
                   e.g., {"base_camera": {"rgb": (h,w,3), "depth": (h,w,1), ...}}
        sensor_params_frame: Contains sensor_param for one timestep.
                             e.g., {"base_camera": {"intrinsic_cv": (3,3)}}
        extra_frame: Contains extra data for one timestep.
                     e.g., {"tcp_pose": (7,), "cam0_world_pose": (7,)}
        Returns:
            A new dictionary with the added cropped image.
        """
        processed = obs_frame[self.camera_name].copy() # Start with a copy

        cam_intrinsics = sensor_params_frame[self.camera_name]["intrinsic_cv"] # (3,3)
        tcp_pose_world = extra_frame["tcp_pose"] # (7,)
        cam_pose_world = extra_frame["cam0_world_pose"] # (7,)
        tcp_pos_in_cam_frame = self._get_tcp_in_cam_coords(tcp_pose_world, cam_pose_world) # (3,)
        tcp_pixel_coords_arr = project_points_to_pixels(tcp_pos_in_cam_frame.reshape(1,3), cam_intrinsics) # (1,2)
        
        if np.isnan(tcp_pixel_coords_arr).any():
            print(f"Warning: TCP projected to NaN pixel coordinates (possibly behind camera or Z_cam too small). Using image center.")
            # Fallback: use image center if projection fails
            img_h, img_w = obs_frame[self.camera_name]["rgb"].shape[:2]
            tcp_pixel_coords = np.array([img_w / 2, img_h / 2], dtype=np.float32)
        else:
            tcp_pixel_coords = tcp_pixel_coords_arr[0] # (2,) [u,v] or [x,y]

        for modality in obs_frame[self.camera_name].keys():
            cropped = crop_image_around_pixel(
                obs_frame[self.camera_name][modality], 
                tcp_pixel_coords, 
                self.crop_size, 
                self.padding_value
            )
            processed[modality + "_crop"] = cropped
    
        obs_frame[self.camera_name] = processed
        return obs_frame

    def process(self, trajectory_data: dict) -> dict:
        """
        Processes a full trajectory (multiple timesteps).
         trajectory_data: The input dictionary structure as you provided.
                         e.g., {"sensor_data": {"base_camera": {"rgb": (t,h,w,3)...}}, ...}
        Returns:
            A new trajectory_data dictionary with added cropped images for each frame.
            The new keys (e.g., "ee_rgb_crop") will also have a time dimension (t, ch, cw, 3).
        """

        num_timesteps = trajectory_data["sensor_data"][self.camera_name]["rgb"].shape[0]
        
        all_cropped = []

        for t in range(num_timesteps):
            # Construct per-frame dictionaries to pass to process_frame
            obs_frame_t = {
                cam_name: {
                    modality: data[t] 
                    for modality, data in modalities.items()
                }
                for cam_name, modalities in trajectory_data["sensor_data"].items()
            }
            
            sensor_params_frame_t = {
                cam_name: {
                    param_name: data[t]
                    for param_name, data in params.items()
                }
                for cam_name, params in trajectory_data["sensor_param"].items()
            }

            extra_frame_t = {
                key: value[t] for key, value in trajectory_data["extra"].items()
            }

            # Process this single frame
            processed_frame_data = self.process_frame(obs_frame_t, sensor_params_frame_t, extra_frame_t)
            
            # Collect the results
            all_cropped.append(processed_frame_data[self.camera_name])

        # Stack the list of cropped images into a single NumPy array (t, ch, cw, C)
        for modality, data in all_cropped[0].items():
            stacked_cropped = np.stack([cropped[modality] for cropped in all_cropped], axis=0)
            trajectory_data["sensor_data"][self.camera_name][modality] = stacked_cropped
            
        return trajectory_data
