使用 Diffusion Policy 估计
$$
\Delta T_t =  T^{\text{cam}, t}_{\text{obj}, t+1} \cdot \left( T^{\text{cam}, t}_{\text{obj}, t}\right)^{-1} = T^{\text{cam}, t}_{\text{cam}, t+1} \cdot T^{\text{cam}, t+1}_{\text{obj}, t+1} \cdot \left( T^{\text{cam}, t}_{\text{obj}, t}\right)^{-1}
$$
表示物体在当前时刻相机坐标系下的位置变化，其中 $T^{\text{cam}, t}_{\text{cam}, t+1}$可以通过 SFM / SLAM 获得，$T^{\text{cam}, t}_{\text{obj}, t}$ 可以通过 Object 6DoF Pose Estimation算法获得。

注意到这是一个和物体坐标系无关的定义。

则机械臂控制可以通过如下公式计算
$$
T^{\text{base}, t+1}_{\text{ee}, t+1} = T^{\text{base}, t+1}_{\text{cam}, t} \cdot T^{\text{cam}, t}_{\text{obj}, t+1} \cdot \left[ \left( T^{\text{cam}, t}_{\text{obj}, t} \right)^{-1} \cdot T^{\text{cam}, t}_{\text{obj}, t} \right] \cdot T^{\text{obj}, t+1}_{\text{ee}, t+1}
$$
我们假设物体和末端执行器之间没有相对滑动，即
$$
T^{\text{obj}, t+1}_{\text{ee}, t+1} = T^{\text{obj}, t}_{\text{ee}, t}
$$
那么可以得到
$$
T^{\text{base}, t+1}_{\text{ee}, t+1} = T^{\text{base}, t+1}_{\text{cam}, t} \cdot \Delta T_t \cdot T^{\text{cam}, t}_{\text{obj}, t} \cdot T^{\text{obj}, t}_{\text{ee}, t}
$$
最终
$$
T^{\text{base}, t+1}_{\text{ee}, t+1} = T^{\text{base}, t+1}_{\text{cam}, t} \cdot \Delta T_t \cdot T^{\text{cam}, t}_{\text{base}, t} \cdot T^{\text{base}, t}_{\text{ee}, t}
$$
其中 $T^{\text{base}, t+1}_{\text{cam}, t}$, $T^{\text{cam}, t}_{\text{base}, t}$ 通过相机标定得到（如果base是动的可以考虑结合机器人的运动学模型），$\Delta T_t$ 由Diffusion Policy进行估计，$T^{\text{base}, t}_{\text{ee}, t}$ 由机械臂FK给出。

注意到这个公式中避免了物体和末端之间的标定 $T^{\text{obj}}_{\text{ee}}$，而且假设某一时刻物体相对末端发生了滑动（后续不再滑动），不影响后续的控制。

## Receding Horizon Control

$$
T^{\text{base}, t+k}_{\text{ee}, t+k} = T^{\text{base}, t+k}_{\text{cam}, t} \cdot T^{\text{cam}, t}_{\text{obj}, t+k} \cdot \left[ \left( T^{\text{cam}, t}_{\text{obj}, t} \right)^{-1} \cdot T^{\text{cam}, t}_{\text{obj}, t} \right] \cdot T^{\text{obj}, t+k}_{\text{ee}, t+k} \quad \left\{k = 1,2,...,\tau \right\}
$$
令
$$
\Delta T_t(k) =  T^{\text{cam}, t}_{\text{obj}, t+k} \cdot \left( T^{\text{cam}, t}_{\text{obj}, t}\right)^{-1} = T^{\text{cam}, t}_{\text{cam}, t+k} \cdot T^{\text{cam}, t+k}_{\text{obj}, t+k} \cdot \left( T^{\text{cam}, t}_{\text{obj}, t}\right)^{-1}
$$
则
$$T^{\text{base}, t+k}_{\text{ee}, t+k} = T^{\text{base}, t+k}_{\text{cam}, t} \cdot \Delta T_t(k) \cdot T^{\text{cam}, t}_{\text{base}, t} \cdot T^{\text{base}, t}_{\text{ee}, t}$$
由于 $\Delta T_t(k)$ 的数值范围较大，不利于 DP 表示，我们让 DP 预测
$$
\Delta T'_t(k) =  T^{\text{cam}, t}_{\text{obj}, t+k} \cdot \left( T^{\text{cam}, t}_{\text{obj}, t+k-1}\right)^{-1}
$$
此时
$$
\Delta T_t(k) = \Delta T'_t(k) \cdot \Delta T'_t(k-1) \cdot ... \cdot \Delta T'_t(1)
$$