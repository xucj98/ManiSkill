
import h5py
import numpy as np
import open3d as o3d
import argparse

def find_traj_and_step_from_global_index(h5_file, camera_name):
    """Calculates total steps and creates a mapping from global index to traj/local index."""
    total_steps = 0
    index_map = []
    with h5py.File(h5_file, 'r') as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith('traj_')])
        for traj_key in traj_keys:
            # Check if the camera group exists for this trajectory
            if camera_name in f[traj_key]['obs']['sensor_data']:
                num_steps_in_traj = f[traj_key]['obs']['sensor_data'][camera_name]['point_cloud'].shape[0]
                for i in range(num_steps_in_traj):
                    index_map.append({"traj_key": traj_key, "local_step": i})
                total_steps += num_steps_in_traj
            else:
                print(f"Warning: Camera '{camera_name}' not found in trajectory '{traj_key}'. Skipping.")
    return total_steps, index_map

class InteractiveVisualizer:
    def __init__(self, h5_file, camera_name, start_step=0):
        self.h5_file = h5_file
        self.camera_name = camera_name
        self.total_steps, self.index_map = find_traj_and_step_from_global_index(h5_file, camera_name)
        self.current_step = start_step
        
        if self.total_steps == 0:
            print(f"Error: No point clouds found for camera '{self.camera_name}'. Exiting.")
            return

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.is_initialized = False

    def load_point_cloud(self, step_index):
        if not (0 <= step_index < self.total_steps):
            print(f"Warning: Step {step_index} is out of bounds.")
            return None

        map_entry = self.index_map[step_index]
        traj_key, local_step_index = map_entry['traj_key'], map_entry['local_step']

        with h5py.File(self.h5_file, 'r') as f:
            pc_data = f[traj_key]['obs']['sensor_data'][self.camera_name]['point_cloud'][local_step_index]
            points = pc_data[:, :3]
            colors = pc_data[:, 3:] / 255.0
            
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(points)
            temp_pcd.colors = o3d.utility.Vector3dVector(colors)
            print(f"Showing Camera: {self.camera_name} | Traj: {traj_key}, Local Step: {local_step_index}, Global Step: {step_index}/{self.total_steps-1}")
            return temp_pcd

    def update_geometry(self, vis):
        new_pcd = self.load_point_cloud(self.current_step)
        if new_pcd:
            self.pcd.points = new_pcd.points
            self.pcd.colors = new_pcd.colors
            if not self.is_initialized:
                vis.add_geometry(self.pcd)
                vis.add_geometry(self.world_frame)
                self.is_initialized = True
            else:
                vis.update_geometry(self.pcd)

    def next_frame(self, vis):
        self.current_step = (self.current_step + 1) % self.total_steps
        self.update_geometry(vis)
        return False

    def prev_frame(self, vis):
        self.current_step = (self.current_step - 1 + self.total_steps) % self.total_steps
        self.update_geometry(vis)
        return False

    def quit_vis(self, vis):
        print("Exiting visualizer.")
        vis.destroy_window()
        return False

    def run(self):
        if self.total_steps == 0: return
        self.vis.create_window(window_name='Interactive Point Cloud Visualizer')
        self.vis.register_key_callback(ord("n"), self.next_frame); self.vis.register_key_callback(ord("N"), self.next_frame)
        self.vis.register_key_callback(ord("b"), self.prev_frame); self.vis.register_key_callback(ord("B"), self.prev_frame)
        self.vis.register_key_callback(ord("q"), self.quit_vis); self.vis.register_key_callback(ord("Q"), self.quit_vis)
        self.vis.register_key_callback(256, self.quit_vis) # ESC

        print("--- Controls ---"); print("  n: Next Frame"); print("  b: Previous Frame"); print("  q / ESC: Quit"); print("----------------")
        self.update_geometry(self.vis)
        self.vis.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactively visualize point clouds from a structured H5 file.")
    parser.add_argument('--file', type=str, default='peg_insertion_pointcloud_dp3_structured_all_cams.h5')
    parser.add_argument('--camera_name', type=str, default='base_camera', help='Camera to visualize (e.g., base_camera, hand_camera).')
    parser.add_argument('--step', type=int, default=0, help='The global step index to start visualization from.')
    args = parser.parse_args()
    
    viewer = InteractiveVisualizer(args.file, args.camera_name, args.step)
    viewer.run()
