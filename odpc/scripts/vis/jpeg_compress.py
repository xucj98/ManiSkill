import argparse
import numpy as np
import h5py
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-path', type=str, required=True, help='Path to the compressed HDF5 file.')
    parser.add_argument('--delay-ms', type=int, default=10, help='Delay between frames in ms.')
    return parser.parse_args()

def main():
    args = get_args()
    exit_key_pressed = False

    with h5py.File(args.traj_path, 'r') as file:
        for traj_key in file.keys():
            if not traj_key.startswith('traj_'):
                continue
            
            sensor_data_group = file[f'{traj_key}/obs/sensor_data']
            for cam_name in sensor_data_group.keys():
                rgb_dset = sensor_data_group[f'{cam_name}/rgb']
                window_name = cam_name

                for i in range(len(rgb_dset)):
                    compressed_frame_uint8 = rgb_dset[i]
                    decoded_image_bgr = cv2.imdecode(compressed_frame_uint8, cv2.IMREAD_COLOR)
                    
                    cv2.imshow(window_name, decoded_image_bgr)
                    
                    key = cv2.waitKey(args.delay_ms)
                    if key == 27 or key == ord('q'):
                        exit_key_pressed = True
                        break
                
                if exit_key_pressed:
                    break
            if exit_key_pressed:
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()