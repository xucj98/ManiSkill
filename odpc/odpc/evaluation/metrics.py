import torch
import torch.nn.functional as F

def calculate_trajectory_errors(
    actual_traj: torch.Tensor, 
    desired_traj: torch.Tensor
) -> dict:
    """
    计算并评估两条轨迹之间的位置和姿态（角度）误差。

    Args:
        actual_traj (torch.Tensor): 机器人实际执行的轨迹，形状为 (T, 7)。
                                    格式为 [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]。
        desired_traj (torch.Tensor): 控制器期望跟踪的目标轨迹，形状为 (T, 7)。
                                     格式与 actual_traj 相同。

    Returns:
        dict: 一个包含详细误差统计信息的字典。
              - 'position': 位置误差 (单位与输入位置单位相同, e.g., meters)
                - 'mean': 平均绝对误差 (MAE)
                - 'rmse': 均方根误差 (RMSE)
                - 'max':  最大误差
              - 'orientation': 姿态误差 (单位: degrees)
                - 'mean': 平均绝对角度误差
                - 'rmse': 均方根角度误差
                - 'max':  最大角度误差
    """
    # --- 1. 输入验证 ---
    if actual_traj.shape != desired_traj.shape:
        raise ValueError("两条轨迹的形状必须相同。")
    if actual_traj.shape[1] != 7:
        raise ValueError("轨迹的第二个维度必须是7 ([pos, quat])。")

    # --- 2. 分离位置和姿态数据 ---
    pos_actual = actual_traj[:, :3]
    pos_desired = desired_traj[:, :3]

    quat_actual = actual_traj[:, 3:]
    quat_desired = desired_traj[:, 3:]

    # --- 3. 计算位置误差 ---
    # 计算每个时间步的欧几里得距离
    pos_error_per_step = torch.linalg.norm(pos_actual - pos_desired, dim=1)

    # 计算位置误差的统计量
    pos_mae = torch.mean(pos_error_per_step)
    pos_rmse = torch.sqrt(torch.mean(pos_error_per_step**2))
    pos_max = torch.max(pos_error_per_step)

    # --- 4. 计算姿态误差 ---
    # 为保证鲁棒性，先将四元数归一化，以防它们不是单位四元数
    quat_actual = F.normalize(quat_actual, p=2, dim=1)
    quat_desired = F.normalize(quat_desired, p=2, dim=1)

    # 计算点积
    dot_product = torch.sum(quat_actual * quat_desired, dim=1)

    # 使用绝对值来处理 q 和 -q 等价的问题
    # 使用 clamp 来防止因浮点数精度问题导致 acos 输入超出 [-1, 1] 范围
    dot_product_abs_clamped = torch.clamp(torch.abs(dot_product), -1.0, 1.0)
    
    # 计算每个时间步的角距离（弧度）
    # 公式: 2 * acos(|q1 · q2|)
    rot_error_rad_per_step = 2 * torch.acos(dot_product_abs_clamped)

    # 转换为度，更直观
    rot_error_deg_per_step = torch.rad2deg(rot_error_rad_per_step)

    # 计算姿态误差的统计量
    rot_mae = torch.mean(rot_error_deg_per_step)
    rot_rmse = torch.sqrt(torch.mean(rot_error_deg_per_step**2))
    rot_max = torch.max(rot_error_deg_per_step)
    
    # --- 5. 格式化输出 ---
    errors = {
        'position': {
            'mean': pos_mae.item(),
            'rmse': pos_rmse.item(),
            'max': pos_max.item(),
        },
        'orientation': {
            'mean': rot_mae.item(),
            'rmse': rot_rmse.item(),
            'max': rot_max.item(),
        }
    }
    return errors