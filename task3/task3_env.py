import torch
import numpy as np
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab_assets import UNITREE_GO2_CFG

# 导入我们已经测试无误的 Task3 世界模型
from task3_world import Task3WorldCfg, Task3World

# ===================================================================
# 1. RL 与环境参数配置类 (Config)
# ===================================================================
class Task3Config:
    """
    Task 3: 视觉导航与动态避障 环境参数配置
    集中保存所有强化学习超参数、奖励系数和物理同步频率。
    """
    num_envs = 1024
    device = "cuda:0"
    
    # --- 控制与同步频率 ---
    sim_dt = 0.005           # 底层物理步长 (200Hz)
    decimation = 4           # 控制抽取率 (RL 策略 50Hz)
    max_episode_length = 2500 # 最大 RL 步数 (20s)
    
    # --- 动作平滑与缩放 ---
    action_scale = 0.25      # 动作缩放系数
    action_ema_alpha = 0.5   # EMA 平滑滤波系数
    
    # --- 状态与死区 ---
    target_height = 0.32
    height_deadband = 0.02       
    clearance_threshold = 0.12

   # ================= 1. 导航主任务 (拆分速度激励) =================
    w_progress = 0.5      
    w_speed_bonus = 0.4    
    w_yaw = 0.15           
    w_stall = -0.08        
    
    stall_epsilon = 0.1    
    gate_dist = 1.0        

    # ================= 2. 稳定与避障 =================
    w_obs = -0.2           
    w_hit = -0.2           
    w_h = -0.02            
    w_ori = -0.02          
    w_omega = -0.02        

    # ================= 3. 经济效率层 (引入死区阈值) =================
    w_da = -0.05           
    w_pow = -0.05          
    w_lim = -0.015         

    action_deadband = 0.05 # 允许的动作变化平方和阈值
    power_deadband = 30.0  # 允许的标称奔跑功率阈值 (W)

    # ================= 4. 终局奖励与物理标量 =================
    rew_success = 15.0      
    rew_fall = -10.0        
    target_vel = 1.0       
    power_nom = 150.0

# --- 机器狗安全配置 ---
robot_cfg_safe = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
if robot_cfg_safe.spawn.rigid_props is None:
    robot_cfg_safe.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
robot_cfg_safe.spawn.rigid_props.max_depenetration_velocity = 5.0 
if robot_cfg_safe.spawn.articulation_props is None:
    robot_cfg_safe.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg()
robot_cfg_safe.spawn.articulation_props.enabled_self_collisions = False


# ===================================================================
# 2. 场景定义类 (Scene)
# ===================================================================
@configclass
class Task3SceneCfg(InteractiveSceneCfg):
    """
    配置环境的物理实体、地形与传感器。
    """
    num_envs: int = Task3Config.num_envs
    env_spacing: float = 0.0 # 环境重叠，依赖世界模型的绝对坐标管理
    
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", 
        collision_group=-1
    )
    
    robot: ArticulationCfg = robot_cfg_safe
    
    
    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", 
        filter_prim_paths_expr=[], # 留空，捕获躯干与地面的接触
        update_period=0.0, 
    )


# ===================================================================
# 3. 视觉导航强化学习环境主体 (Env)
# ===================================================================
class Task3VisualNavEnv(gym.Env):
    """
    任务 3：基于 1D-CNN 特征拼接的视觉导航环境
    """
    def __init__(self, cfg: Task3Config):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        
        # --- 初始化底层世界模型 ---
        self.world_cfg = Task3WorldCfg()
        self.world = Task3World(self.world_cfg, self.num_envs, self.device)
        
        # --- 初始化物理仿真引擎 ---
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim_dt, 
            device=self.device,
            physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True)
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        # --- 初始化场景 ---
        scene_cfg = Task3SceneCfg(num_envs=self.num_envs)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)
        
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene.articulations["robot"]
        self.contact = self.scene.sensors["contact"]
        
        # --- 观测与动作空间定义 ---
        self.base_obs_dim = 48   # 基座状态 + 本体感觉 + 动作记忆
        self.lidar_obs_dim = 90 # 16线 * 90点 = 1440
        self.total_obs_dim = self.base_obs_dim + self.lidar_obs_dim
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))
        
        # --- 张量缓冲区 ---
        self.last_action = torch.zeros((self.num_envs, 12), device=self.device)
        self.last_lin_vel = torch.zeros((self.num_envs, 3), device=self.device) 
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.sim.reset()

        # 安全站立初始姿态
        safe_standing_pose = torch.tensor([
             0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
             0.1, 1.0, -1.5,
            -0.1, 1.0, -1.5 
        ], device=self.device)
        self.default_joint_pos = safe_standing_pose.unsqueeze(0).repeat(self.num_envs, 1)

        self.default_root_state = torch.zeros((self.num_envs, 13), device=self.device)
        self.default_root_state[:, 3] = 1.0  # 四元数 w=1
        
        # 提取真实的足端索引 (Unitree 官方模型一般带有 'foot' 命名)
        foot_idx, _ = self.robot.find_bodies(".*foot.*")
        if len(foot_idx) == 0:
            num_bodies = self.robot.data.body_pos_w.shape[1]
            foot_idx = [max(0, num_bodies - 4 + i) for i in range(4)]
        self.foot_indices = torch.tensor(foot_idx, dtype=torch.long, device=self.device)
        
        # 提取关节中位和范围用于限幅屏障 (假设 Go2 的配置)
        self.q_mid = self.default_joint_pos.clone()
        self.q_range = torch.full((self.num_envs, 12), 0.8, device=self.device) # 允许偏离中位 0.8 rad
        
        # 全局状态跟踪器
        self.last_distance_to_target = torch.zeros(self.num_envs, device=self.device)
        self.prog_ema = torch.zeros(self.num_envs, device=self.device) # 进展 EMA 滑窗
        # 动态回合寿命追踪器
        self.env_max_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def set_goal_dist_range(self, min_dist: float, max_dist: float):
        """动态课程控制器：由外部训练脚本调用，修改起终点的生成距离"""
        self.world.cfg.goal_dist_range = [min_dist, max_dist]
        
    def reset(self, env_ids=None, options=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long)
            
        # 1. 重置世界环境拓扑（生成静态/动态障碍物与起终点）
        self.world.reset_envs(env_ids)
        
        # 2. 将机器狗放置在随机生成的起点
        root_state = self.default_root_state[env_ids].clone()
        root_state[:, :2] = self.world.start_pos[env_ids]
        root_state[:, 2] = 0.6  # 留出下落缓冲高度
        
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        
        joint_p = self.default_joint_pos[env_ids]
        joint_v = torch.zeros_like(joint_p)
        self.robot.write_joint_state_to_sim(joint_p, joint_v, env_ids=env_ids)
        
        zero_velocities = torch.zeros((len(env_ids), 6), device=self.device)
        self.robot.write_root_velocity_to_sim(zero_velocities, env_ids=env_ids)
        
        # 清理缓冲区
        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.last_lin_vel[env_ids] = 0.0
        
        self.scene.update(0.0)
        
        obs_single = self._compute_obs()

        # 在 reset 方法的末尾补充：
        dist_to_target = torch.norm(self.world.start_pos[env_ids] - self.world.target_pos[env_ids], dim=-1)
        self.last_distance_to_target[env_ids] = dist_to_target
        self.prog_ema[env_ids] = 0.0
        dt = self.cfg.sim_dt * self.cfg.decimation
        time_allowance = dist_to_target * 1.5 + 5.0 
        self.env_max_steps[env_ids] = (time_allowance / dt).to(torch.long)
        return obs_single, {}

    def step(self, action: torch.Tensor):
        # 1. 动作平滑与下发
        current_action = self.cfg.action_ema_alpha * action + (1.0 - self.cfg.action_ema_alpha) * self.last_action
        old_action = self.last_action.clone()
        self.last_action = current_action.clone()
        
        target_pos = self.default_joint_pos + current_action * self.cfg.action_scale
        self.robot.set_joint_position_target(target_pos)
            
        # 2. 物理步进与世界模型刷新
        dt = self.cfg.sim_dt * self.cfg.decimation
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
        
        # 推进动态障碍物
        self.world.step_kinematics(dt)
        self.scene.update(dt)
        self.episode_length_buf += 1

        num_bodies = self.robot.data.body_pos_w.shape[1]
        if num_bodies > 0:
            self.last_lin_vel = self.robot.data.root_lin_vel_b.clone()
        
        # 3. 计算观测与奖励
        obs_single = self._compute_obs()
        rewards, terminated, truncated, info = self._compute_rewards_and_dones(current_action, old_action)

        # 4. 环境截断处理
        resets = terminated | truncated
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            info["terminal_observation"] = obs_single[reset_env_ids].clone()
            self.reset(reset_env_ids)

        return obs_single, rewards, terminated, truncated, info

    def _compute_obs(self):
        if self.robot.data.body_pos_w.shape[1] == 0:
            return torch.zeros((self.num_envs, self.total_obs_dim), device=self.device)

        base_pos = self.robot.data.root_pos_w
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b
        
        relative_z = base_pos[:, 2]
        z_err = (relative_z - self.cfg.target_height).unsqueeze(1)
        
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        target_polar = self.world.get_target_polar_coords(base_pos, yaw)
        
        joint_pos_residual = self.robot.data.joint_pos - self.default_joint_pos
        joint_vel = self.robot.data.joint_vel
        
        base_obs = torch.cat([
            base_lin_vel, base_ang_vel, proj_gravity, z_err, 
            target_polar, 
            joint_pos_residual, joint_vel,                            
            self.last_action                           
        ], dim=-1) 
        
        # 直接调用我们刚刚手写的张量化物理算法
        lidar_distances = self.world.compute_lidar_tensors(base_pos, yaw)
        
        # 归一化输入给神经网络 (范围 [0, 1])
        lidar_obs = lidar_distances / 5.0 
        
        total_obs = torch.cat([base_obs, lidar_obs], dim=-1)
        return total_obs

    def _compute_rewards_and_dones(self, action, old_action):
        if self.robot.data.body_pos_w.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.device), torch.ones(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), {}

        base_pos = self.robot.data.root_pos_w
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        torques = self.robot.data.applied_torque
        
        dt = self.cfg.sim_dt * self.cfg.decimation
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        # ================= 1. 导航主任务层 =================
        current_distance = torch.norm(base_pos[:, :2] - self.world.target_pos, dim=-1)
        target_polar = self.world.get_target_polar_coords(base_pos, yaw)
        err_yaw = target_polar[:, 1]
        v_x = base_lin_vel[:, 0]

        # A. 纯净势能差进展 
        raw_progress = (self.last_distance_to_target - current_distance) / dt
        r_prog = torch.clamp(raw_progress / self.cfg.target_vel, min=-1.0, max=1.0)
        
        # B. EMA 停滞惩罚
        self.prog_ema = 0.95 * self.prog_ema + 0.05 * raw_progress
        stall_risk = torch.clamp(self.cfg.stall_epsilon - self.prog_ema, min=0.0)
        p_stall = torch.tanh(stall_risk / 0.1) 

        # C. 前向扇区 Softmin 与门控朝向
        lidar_distances = self.world.compute_lidar_tensors(base_pos, yaw)
        front_indices = torch.cat([torch.arange(0, 12, device=self.device), torch.arange(78, 90, device=self.device)])
        front_rays = lidar_distances[:, front_indices]
        m_front = -0.2 * torch.logsumexp(-5.0 * front_rays, dim=-1)
        
        g_clear = torch.sigmoid((m_front - self.cfg.gate_dist) / 0.2)
        r_yaw = g_clear * torch.exp(-torch.square(err_yaw) / 0.25)
        
        # 只有在道路相对空旷时，跑到 1.0m/s 附近才会给出指数级的大量加分
        r_speed_bonus = g_clear * torch.exp(-torch.square(raw_progress - self.cfg.target_vel) / 0.1)

        # ================= 2. 安全稳定层 =================
        d_safe = 0.4 + 0.2 * torch.abs(v_x) 
        obs_risk = torch.nn.functional.softplus(d_safe - m_front)
        p_obs = torch.tanh(obs_risk / 0.3) 

        contact_forces = self.contact.data.net_forces_w
        has_contact = (torch.norm(contact_forces, dim=-1).max(dim=-1)[0] > 1.0).float()
        p_hit = has_contact

        h_err = torch.clamp(torch.abs(base_pos[:, 2] - self.cfg.target_height) - self.cfg.height_deadband, min=0.0)
        p_h = torch.tanh(h_err / 0.05)
        
        ori_err = torch.square(proj_gravity[:, 0]) + torch.square(proj_gravity[:, 1])
        p_ori = torch.tanh(ori_err / 0.1)
        
        omega_err = torch.clamp(torch.norm(base_ang_vel[:, :2], dim=-1) - 1.5, min=0.0) 
        p_omega = torch.tanh(omega_err / 1.0)

        # ================= 3. 经济效率层 =================
        # A. 动作突变：低于 action_deadband 视为正常发力，不扣分
        da_sq = torch.sum(torch.square(action - old_action), dim=1)
        da_err = torch.clamp(da_sq - self.cfg.action_deadband, min=0.0)
        p_da = torch.sqrt(da_err + 0.01) - 0.1 
        
        # B. 归一化功率：低于 power_deadband 视为正常步态消耗，不扣分
        power = torch.mean(torch.abs(torques * joint_vel), dim=-1)
        power_err = torch.clamp(power - self.cfg.power_deadband, min=0.0)
        p_pow = torch.tanh(power_err / self.cfg.power_nom)
        
        # C. 关节限幅屏障
        lim_err = torch.mean(torch.clamp(torch.abs(joint_pos - self.q_mid) - self.q_range, min=0.0), dim=-1)
        p_lim = torch.tanh(lim_err / 0.05)

        # ================= 4. 汇总与终局判定 =================
        val_prog = self.cfg.w_progress * r_prog
        val_speed = self.cfg.w_speed_bonus * r_speed_bonus 
        val_yaw = self.cfg.w_yaw * r_yaw
        val_stall = self.cfg.w_stall * p_stall
        
        val_obs = self.cfg.w_obs * p_obs
        val_hit = self.cfg.w_hit * p_hit
        val_h = self.cfg.w_h * p_h
        val_ori = self.cfg.w_ori * p_ori
        val_omega = self.cfg.w_omega * p_omega
        
        val_da = self.cfg.w_da * p_da
        val_pow = self.cfg.w_pow * p_pow
        val_lim = self.cfg.w_lim * p_lim

        # 总 Dense Reward
        step_reward = (val_prog + val_speed + val_yaw + val_stall + 
                       val_obs + val_hit + val_h + val_ori + val_omega + 
                       val_da + val_pow + val_lim)

        is_success = current_distance < 0.5
        body_crash = base_pos[:, 2] < 0.15
        posture_crash = proj_gravity[:, 2] > -0.4
        is_fallen = (body_crash | posture_crash) & (~is_success)

        rew_total = step_reward.clone()
        rew_total = torch.where(is_fallen, rew_total + self.cfg.rew_fall, rew_total)
        rew_total = torch.where(is_success, rew_total + self.cfg.rew_success, rew_total)

        terminated = is_fallen | is_success
        truncated = self.episode_length_buf >= self.env_max_steps
        
        self.last_distance_to_target = current_distance.clone()
        
        # ================= 5. Info 字典更新 =================
        info = {
            "is_fallen": is_fallen, 
            "is_success": is_success, 
            "reward_components": {
                "Task_R_Prog": val_prog.mean().item(),
                "Task_R_Speed": val_speed.mean().item(), 
                "Task_R_Yaw": val_yaw.mean().item(),
                "Task_P_Stall": val_stall.mean().item(),
                "Safe_P_Obs": val_obs.mean().item(),
                "Safe_P_Hit": val_hit.mean().item(),
                "Safe_P_Height": val_h.mean().item(),
                "Safe_P_Ori": val_ori.mean().item(),
                "Safe_P_Omega": val_omega.mean().item(),
                "Eff_P_Action": val_da.mean().item(),
                "Eff_P_Power": val_pow.mean().item(),
                "Eff_P_Limit": val_lim.mean().item(),
            },
            "telemetry": {
                "mean_distance": current_distance.mean().item(),
                "R_Progress": raw_progress.mean().item() 
            }
        }
        return rew_total, terminated, truncated, info
    
    