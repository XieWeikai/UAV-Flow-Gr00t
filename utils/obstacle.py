import numpy as np
from scipy.special import expit
from utils.coordinate import PointCloudESDF

def compute_yaw_rate(yaws_deg: np.ndarray, dt: float, smoothing_window: int = 10) -> np.ndarray:
    """
    Compute absolute yaw rate in degrees/second using unwrap and gradient.
    Args:
        yaws_deg: [N,] array of yaws in degrees.
        dt: time step in seconds.
        smoothing_window: size of the smoothing window (odd integer). 0 or 1 means no smoothing.
    Returns:
        yaw_rate_deg_s: [N,] array of absolute yaw rates in deg/s.
    """
    yaws_rad = np.deg2rad(yaws_deg)
    yaws_unwrapped = np.unwrap(yaws_rad)
    # Gradient handles boundaries better than diff
    yaw_rates_rad_s = np.gradient(yaws_unwrapped, dt)
    yaw_rate = np.rad2deg(np.abs(yaw_rates_rad_s))

    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        # Use mode='same' to keep the original size
        yaw_rate = np.convolve(yaw_rate, kernel, mode='same')
    
    return yaw_rate


def compute_avoidance_scores_copilot(
    poses: np.ndarray,      # [N, 4] (x, y, z, yaw_deg)
    pcd: PointCloudESDF,
    dt: float = 0.1
) -> np.ndarray:
    """
    Computes avoidance intent scores (0-1) for a trajectory.
    Score = (Risk_Ghost - Risk_Real) * Action_Intensity
    """
    N = len(poses)
    lookahead_time = 4.0
    lookahead_steps = int(lookahead_time / dt)
    
    # Parameters that can be tuned
    safe_dist = 0.5         # Meters
    sigma_d = 0.2           # Sigmoid slope for distance
    turn_thresh = 15.0      # Degrees/sec
    sigma_turn = 5.0        # Sigmoid slope for turn
    
    # Pre-compute distances for the actual trajectory
    # Query all trajectory points [N, 3]
    real_coords, real_dists = pcd.query(poses[:, :3], k=1)
    # [N, 1] -> [N,]
    real_dists = real_dists.flatten() 
    
    # Velocities
    diffs = np.diff(poses[:, :3], axis=0)
    dists_step = np.linalg.norm(diffs, axis=1)
    speeds = dists_step / dt
    # speeds = np.append(speeds, speeds[-1])
    # should add first speed not the last
    speeds = np.insert(speeds, 0, speeds[0])
    
    yaws = poses[:, 3]
    window_interaction = 5
    yaw_energy = compute_yaw_rate(yaws, dt, smoothing_window=window_interaction)

    scores = np.zeros(N)

    def get_risk(dist):
        # Risk approaches 1 as dist < safe_dist
        return expit((safe_dist - dist) / sigma_d)

    # Main Loop
    for i in range(N):
        # 1. ACTUAL RISK (Future)
        end_idx = min(i + lookahead_steps, N)
        if end_idx > i:
            min_dist_real = np.min(real_dists[i:end_idx])
            risk_real = get_risk(min_dist_real)
        else:
            risk_real = 0

        # 2. GHOST RISK (Counterfactual Straight)
        if speeds[i] > 0.1 and end_idx > i:
            current_yaw_rad = np.deg2rad(yaws[i])
            num_ghost_steps = end_idx - i
            t_steps = np.arange(1, num_ghost_steps + 1) * dt
            
            dx = speeds[i] * t_steps * np.cos(current_yaw_rad)
            dy = speeds[i] * t_steps * np.sin(current_yaw_rad)
            dz = np.zeros_like(dx)
            
            ghost_points = poses[i, :3] + np.stack([dx, dy, dz], axis=1)
            
            _, ghost_dists = pcd.query(ghost_points, k=1)
            if len(ghost_dists) > 0:
                min_dist_ghost = np.min(ghost_dists)
                risk_ghost = get_risk(min_dist_ghost)
            else:
                 risk_ghost = 0
        else:
            risk_ghost = risk_real
            
        # 3. SCORE
        delta_risk = max(0, risk_ghost - risk_real)
        turn_factor = expit((yaw_energy[i] - turn_thresh) / sigma_turn)
        
        scores[i] = delta_risk * turn_factor
        
    return scores

def compute_avoidance_scores_gemini(
    poses: np.ndarray,      
    pcd: PointCloudESDF,
    dt: float = 0.1
) -> np.ndarray:
    """
    计算轨迹中每个点的“主动避障概率”。
    
    核心逻辑：
    1. 计算真实轨迹未来的碰撞风险 (Risk_Real)。
    2. 计算假设保持当前速度直走未来的碰撞风险 (Risk_Ghost)。
    3. 如果 (Risk_Ghost - Risk_Real > 0) 且 (当前有显著转向动作)，则判定为主动避障。

    Args:
        poses: [N, 4] numpy array. 列分别代表 (x, y, z, yaw_deg)。xyz单位m, yaw单位度。
        pcd: PointCloudESDF 实例，用于查询最近障碍物距离。
        dt: 轨迹采样时间间隔，单位秒。

    Returns:
        scores: [N,] numpy array. 范围 0.0 ~ 1.0，值越高代表避障意图越强。
    """

    # ==========================================
    # 1. 超参数配置 (Hyperparameters)
    # ==========================================
    PRED_HORIZON_S = 4.0   # 向前预测的时间长度 (秒)
    RISK_AT_SAFE_DIST = 0.75  # 锚点：当距离等于 SAFE_DIST_M 时的风险值 (0~1)。
    SAFE_DIST_M    = 0.5   # 安全距离阈值 (米)。
    RISK_BETA      = 8.0   # 风险函数的敏感度。越大，距离变化对风险的影响越剧烈。
    TURN_THRES     = 0.2   # 角速度阈值 (rad/s)。低于此值被视为直线行驶噪声。
    WINDOW_SIZE    = 5     # 动作平滑窗口大小 (帧数)

    # 预测步数
    H_STEPS = int(PRED_HORIZON_S / dt)
    N = poses.shape[0]

    # ==========================================
    # 2. 数据预处理与运动学解算
    # ==========================================
    # 提取位置 [N, 3]
    positions = poses[:, :3] 
    
    # 提取 Yaw 并转为弧度 [N,]
    yaw_rad = np.deg2rad(poses[:, 3])

    # --- 计算线速度向量 v_vec [N, 3] ---
    # 使用后向差分或者前后差分，这里使用简单的后向差分填充
    # shape: [N, 3]
    vel_vec = np.zeros_like(positions)
    vel_vec[:-1] = (positions[1:] - positions[:-1]) / dt
    vel_vec[-1] = vel_vec[-2] # 补全最后一个点

    # --- 计算角速度 yaw_rate [N,] ---
    yaw_rates_deg = compute_yaw_rate(poses[:, 3], dt, smoothing_window=WINDOW_SIZE)
    # 之前代码里这里用了 smooth 后的 absolute rate
    avg_yaw_energy = np.deg2rad(yaw_rates_deg) 

    # Sigmoid 激活函数：将角速度映射到 0~1
    # 逻辑：当平均角速度超过 TURN_THRES 时，action_score 趋近于 1
    # 公式：1 / (1 + exp(-k * (omega - threshold)))
    action_scores = 1.0 / (1.0 + np.exp(-10.0 * (avg_yaw_energy - TURN_THRES)))

    # ==========================================
    # 4. 定义辅助函数 (Helper Functions)
    # ==========================================
    # ==========================================
    #    预计算常量 (Pre-calculation)
    # ==========================================
    # 为了避免在循环中重复计算 Logit 偏移量，这里只计算一次
    # 公式推导: Sigmoid(0 + bias) = p  =>  1/(1+e^-bias) = p  =>  bias = ln(p/(1-p))
    # Clip一下防止 p=1.0 导致 log(inf)
    _p = np.clip(RISK_AT_SAFE_DIST, 1e-6, 1.0 - 1e-6)
    risk_bias = np.log(_p / (1.0 - _p))
    def calc_risk_from_dists(dists_array):
        """
        将物理距离映射为风险值 [0, 1]。
        dists_array: [M,] 距离数组
        Return: [M,] 风险数组
        """
        # 逻辑：
        # dist = safe_dist -> logits = 0 + bias -> sigmoid(bias) = 0.95
        # dist < safe_dist -> logits 变大 -> risk -> 1.0
        # dist > safe_dist -> logits 变小 -> risk -> 0.0
        logits = -RISK_BETA * (dists_array - SAFE_DIST_M) + risk_bias
        return 1.0 / (1.0 + np.exp(-logits))

    # 结果容器
    final_scores = np.zeros(N, dtype=np.float32)

    # ==========================================
    # 5. 主循环：双重未来推演
    # ==========================================
    # 考虑到 pcd.query 可能较耗时，我们在循环中分批查询
    # 对于每个点 t，我们需要看 [t, t+H] 的数据
    
    for t in range(N):
        # 边界检查：如果剩余时间不足预测长度，风险设为0或保持最后状态
        if t + H_STEPS >= N:
            final_scores[t] = 0.0
            continue

        # ----------------------------------
        # A. 构建 真实未来 (Real Future)
        # ----------------------------------
        # 取出未来 H 步的真实坐标
        # shape: [H_STEPS, 3]
        future_real_pts = positions[t : t + H_STEPS]

        # 查询距离
        # returns: coords [H, 1, 3], dists [H, 1]
        _, dists_real_raw = pcd.query(future_real_pts, k=1)
        
        # Flatten distance array -> [H,]
        dists_real = dists_real_raw.flatten()

        # 计算真实轨迹路径上的最大风险 (木桶效应：最危险的那一刻决定了风险)
        # scalar
        risk_real = np.max(calc_risk_from_dists(dists_real))

        # ----------------------------------
        # B. 构建 幽灵未来 (Ghost Future)
        # ----------------------------------
        # 假设：机器人像个“幽灵”一样，保持 t 时刻的线速度向量 v_vec[t] 直走
        # 这种假设比单纯用 Yaw 角更准，因为涵盖了侧滑或非完整约束运动
        curr_v = vel_vec[t] # [3,]
        
        # 生成时间步长数组: [1, 2, ..., H]
        steps = np.arange(1, H_STEPS + 1, dtype=np.float32).reshape(-1, 1) # [H, 1]
        
        # 幽灵轨迹: Pos_ghost = Pos_curr + V_curr * (dt * steps)
        # shape: [H, 3]
        future_ghost_pts = positions[t] + curr_v * (dt * steps)

        # 查询幽灵路径的距离
        _, dists_ghost_raw = pcd.query(future_ghost_pts, k=1)
        dists_ghost = dists_ghost_raw.flatten() # [H,]

        # 计算直走路径的最大风险
        risk_ghost = np.max(calc_risk_from_dists(dists_ghost))

        # ----------------------------------
        # C. 融合计算 (Score Fusion)
        # ----------------------------------
        
        # 1. 风险收益 (Risk Gain): 直走有多危险？实际走有多安全？
        # 如果 risk_ghost (0.9) > risk_real (0.1)，说明实际操作极大降低了风险 -> gain = 0.8
        # 如果 risk_ghost (0.1) < risk_real (0.9)，说明实际操作反而更危险 -> gain = 0
        risk_gain = max(0.0, risk_ghost - risk_real)

        # 2. 最终得分 = 风险收益 * 动作激活度
        # 含义：既要确实避开了风险，又必须伴随着显著的转向动作
        # 避免了“前方变宽阔了，机器人直走，风险降低”这种伪正例
        score = risk_gain * action_scores[t]

        final_scores[t] = score

    return final_scores

def compute_collision_prob(
    poses: np.ndarray,      # [N, 4] (x, y, z, yaw_deg)
    pcd: PointCloudESDF,
    dt: float = 0.1
) -> np.ndarray:
    """
    Simple Ghost Risk Score combined with Turning Intensity.
    Avoidance Score = (Risk of going straight) * (Turning Intensity)
    
    Interpretation:
    - High Risk, Low Turn -> Score 0 (Crashing, not avoiding)
    - High Risk, High Turn -> Score 1 (Active avoidance)
    - Low Risk, High Turn -> Score 0 (Turning for other reasons)
    - Low Risk, Low Turn -> Score 0 (Safe flight)
    """
    # --- Parameters ---
    LOOKAHEAD_TIME = 3.5    # Seconds to project future
    COLLISION_DIST = 0.01   # Meters. Distances below this score 1.0 (collision)
    SAFE_DIST = 1.2         # Meters. Distances above this score 0.0 (safe)
    # ------------------
    
    N = len(poses)
    lookahead_steps = int(LOOKAHEAD_TIME / dt)
    if lookahead_steps <= 0:
        return np.zeros(N)

    # 1. Calculate speeds
    # Distance between consecutive points in 3D
    diffs = np.diff(poses[:, :3], axis=0)
    dists_step = np.linalg.norm(diffs, axis=1)
    speeds = dists_step / dt
    # Align speeds with poses (prepend first speed)
    speeds = np.insert(speeds, 0, speeds[0]) 

    # 2. Yaws
    yaws = poses[:, 3]
    
    # Compute Risk Scores (Ghost Trajectory)
    risk_scores = np.zeros(N)

    for i in range(N):
        current_speed = speeds[i]
        
        # If speed is negligible, ghost trajectory is basically staying in place
        # But we still run the logic to check static proximity
        
        # Ghost Trajectory Projection (Constant Velocity Model in 2D Plane, Fixed Z)
        # Note: If you want 3D velocity projection, would need pitch/vz.
        current_yaw_rad = np.deg2rad(yaws[i])
        
        t_steps = np.arange(1, lookahead_steps + 1) * dt
        
        dx = current_speed * t_steps * np.cos(current_yaw_rad)
        dy = current_speed * t_steps * np.sin(current_yaw_rad)
        dz = np.zeros_like(dx) 
        
        # [K, 3] points
        ghost_points = poses[i, :3] + np.stack([dx, dy, dz], axis=1)
        
        # Query PCD
        # This returns distances to the nearest obstacle for each ghost point
        _, ghost_dists = pcd.query(ghost_points, k=1)
        
        if len(ghost_dists) > 0:
            min_dist = np.min(ghost_dists)
        else:
            # Should not happen if pcd is valid, but fallback to safe
            min_dist = float('inf')
            
        # Linear Score Calculation
        if min_dist <= COLLISION_DIST:
            risk_scores[i] = 1.0
        elif min_dist >= SAFE_DIST:
            risk_scores[i] = 0.0
        else:
            # Linear interpolation 
            # 0.01 -> 1.0
            # 1.0  -> 0.0
            score = (SAFE_DIST - min_dist) / (SAFE_DIST - COLLISION_DIST)
            risk_scores[i] = score


    return risk_scores

# compute_avoidance_scores = compute_avoidance_scores_copilot
# compute_avoidance_scores = compute_avoidance_scores_gemini
compute_avoidance_scores = compute_collision_prob
