import torch
import numpy as np
import gymnasium as gym
import math
from typing import Dict, Tuple, Any
import warnings
import logging

warnings.filterwarnings("ignore", message=".*set_external_force_and_torque.*")
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

# 导入原生宇树 Go2 配置
from isaaclab_assets import UNITREE_GO2_CFG

# ===================================================================
# 1. RL 与环境参数配置类 (Task4 专属 Sim2Real 极限抗扰配置)
# ===================================================================
class Task4Config:
    # --- 1. 基础并行与仿真参数 ---
    num_envs = 2048            
    device = "cuda:0"
    sim_dt = 0.005             # 物理仿真频率 200Hz
    decimation = 4             # 策略控制频率 50Hz (dt = 0.02s)
    max_episode_length = 1000  # 最大步数约 20s
    frame_stack = 5            # RMA 要求：5 帧时序观测堆叠 (模拟 0.1s 历史)
    
    # --- 2. 动作与控制参数 ---
    action_scale = 0.25        
    action_ema_alpha = 0.5     # 动作低通滤波系数
    target_height = 0.30       
    
    # --- 3. 域随机化参数 (Domain Randomization) ---
    # 负载与重心突变
    payload_mass_range = [0.0, 5.0]     # 随机背负 0~5kg 重物
    com_shift_range = [-0.1, 0.1]       # 重心在 X/Y/Z 轴随机偏移 ±10cm
    
    # 执行器衰减 (模拟电机老化或损坏)
    motor_strength_range = [0.7, 1.0]   # 随机 1~2 个关节扭矩输出衰减至 70%~90%
    
    # 外部脉冲飞踢 (Impulse)
    push_interval_s = [3.0, 8.0]        # 每 3~8 秒踢一次
    push_duration_s = 0.1               # 每次持续 0.1 秒
    push_magnitude_range = [100.0, 300.0] # 飞踢力度 100~300N
    
    # 感知噪声 (Gaussian Noise)
    noise_scales = {
        "base_ang_vel": 0.05,           # IMU 角速度噪声
        "proj_gravity": 0.02,           # IMU 倾角漂移
        "joint_pos": 0.01,              # 编码器位置噪声
        "joint_vel": 0.05               # 编码器速度噪声
    }

    # --- 4. 随机指令范围 ---
    cmd_vx_range = [0.5, 0.5]  
    cmd_vy_range = [0, 0] 
    cmd_wz_range = [0, 0] 

    # --- 5. 混合驱动奖励系数 (四层结构) ---
    # 主任务 (正向)
    w_vx = 2.0
    w_wz = 0.4
    w_ori = 0.6                 # 水平姿态强约束
    w_h = 0.15
    w_rec = 0.0                 # 扰动恢复专属奖励
    
    # 经济效率与物理约束 (惩罚)
    w_tau = -0.0005               # 扭矩惩罚
    w_act = -0.1               # 动作抖动惩罚
    w_slip = -0.1               # 足端打滑惩罚
    w_imp = -0.01                # 落地冲击惩罚
    w_j = -0.05                 # 关节极限惩罚
    
    # 离散极刑
    rew_fall = -10.0            # 跌倒重置绝对惩罚
    
    # 高斯/二次函数的敏感度
    sigma_v = 4.0
    sigma_w = 4.0
    sigma_u = 2.0               # 对倾角非常敏感
    sigma_h = 10.0
    impact_threshold = 200.0    # 足端冲击力宽容阈值 (N)

# ===================================================================
# 2. 交互场景配置 (包含机器人与接触力传感器)
# ===================================================================
@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    num_envs: int = Task4Config.num_envs
    env_spacing: float = 2.5
    
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 添加接触力传感器以计算冲击力与打滑
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

# ===================================================================
# 3. 机器狗 Sim2Real RMA 抗扰环境类
# ===================================================================
class QuadrupedSim2RealEnv(gym.Env):
    def __init__(self, cfg: Task4Config):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        self.dt = self.cfg.sim_dt * self.cfg.decimation
        
        # 1. 物理引擎初始化
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.sim_dt, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
        light_cfg.func("/World/Light", light_cfg)
        
        # 2. 生成交互场景
        scene_cfg = QuadrupedSceneCfg()
        scene_cfg.num_envs = self.num_envs
        self.scene = InteractiveScene(scene_cfg)
        
        self.robot = self.scene.articulations["robot"]
        self.contact = self.scene.sensors["contact_forces"]
        
        # 3. RMA 状态空间定义
        self.obs_dim_per_frame = 48  
        self.history_dim = self.obs_dim_per_frame * self.cfg.frame_stack
        
        self.priv_dim = 1 + 1 + 3 + 2 + 12 
        self.total_obs_dim = self.history_dim + self.priv_dim
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))
        
        # 4. 核心张量缓存
        self.obs_stack = torch.zeros((self.num_envs, self.cfg.frame_stack, self.obs_dim_per_frame), device=self.device)
        self.priv_info = torch.zeros((self.num_envs, self.priv_dim), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 12), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_tracking_error = torch.zeros(self.num_envs, device=self.device)
        
        self.dr_payload_mass = torch.zeros(self.num_envs, device=self.device)
        self.dr_com_shift = torch.zeros((self.num_envs, 3), device=self.device)
        self.dr_friction = torch.ones(self.num_envs, device=self.device)
        self.dr_motor_strength = torch.ones((self.num_envs, 12), device=self.device)
        
        self.push_forces = torch.zeros((self.num_envs, 3), device=self.device)
        self.push_timers = torch.zeros(self.num_envs, device=self.device)

        self.sim.reset()
        
        # 引擎重置后，才可以安全地获取关节索引和默认张量
        self.foot_indices, _ = self.robot.find_bodies(".*_foot")
        
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.q_limits_low = self.robot.data.soft_joint_pos_limits[0, :, 0]
        self.q_limits_high = self.robot.data.soft_joint_pos_limits[0, :, 1]

    def reset(self, env_ids=None, options=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # 1. 物理状态重置
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, 2] += torch.rand_like(state[:, 2]) * 0.05 
        self.robot.write_root_state_to_sim(state, env_ids)
        self.robot.write_joint_state_to_sim(self.default_joint_pos[env_ids], torch.zeros_like(self.default_joint_pos[env_ids]), env_ids=env_ids)
        
        # 2. 随机指令下发
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_vx_range[1] - self.cfg.cmd_vx_range[0]) + self.cfg.cmd_vx_range[0]
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_vy_range[1] - self.cfg.cmd_vy_range[0]) + self.cfg.cmd_vy_range[0]
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_wz_range[1] - self.cfg.cmd_wz_range[0]) + self.cfg.cmd_wz_range[0]
        
        # 3. 执行域随机化 (Static Randomization)
        self._randomize_domains(env_ids)
        
        # 4. 缓存清理
        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.last_tracking_error[env_ids] = 0.0
        self.push_timers[env_ids] = torch.rand(len(env_ids), device=self.device) * (self.cfg.push_interval_s[1] - self.cfg.push_interval_s[0]) + self.cfg.push_interval_s[0]
        
        self.scene.update(0.0)
        obs_single, priv_single = self._compute_obs()
        
        for i in range(self.cfg.frame_stack):
            self.obs_stack[env_ids, i, :] = obs_single[env_ids]
        self.priv_info[env_ids] = priv_single[env_ids]

        # 返回压平的 Box 向量：[H_t, E_t]
        full_obs = torch.cat([self.obs_stack[env_ids].view(len(env_ids), -1), self.priv_info[env_ids]], dim=-1)
        return full_obs, {}

    def step(self, action: torch.Tensor):
        # 1. 动作平滑与失效退化 (Motor Degradation)
        current_action = self.cfg.action_ema_alpha * action + (1.0 - self.cfg.action_ema_alpha) * self.last_action
        old_action = self.last_action.clone()
        self.last_action = current_action.clone()
        
        # 应用电机强度衰减，模拟现实中电机过热或无力
        degraded_action = current_action * self.dr_motor_strength
        target_pos = self.default_joint_pos + degraded_action * self.cfg.action_scale
        
        # 2. 施加外部物理抗扰 (Payload Wrench & Impulse Push)
        self._apply_external_disturbances()
        
        # 3. 物理闭环解算
        self.robot.set_joint_position_target(target_pos)
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            
        self.scene.update(self.dt)
        self.episode_length_buf += 1

        # 4. 观测更新与奖励计算
        obs_single, priv_single = self._compute_obs()
        self.obs_stack = torch.roll(self.obs_stack, shifts=-1, dims=1)
        self.obs_stack[:, -1, :] = obs_single
        self.priv_info = priv_single
        
        rewards, terminated, truncated, info = self._compute_rewards_and_dones(current_action, old_action)

        # 5. 重置处理
        resets = terminated | truncated
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            full_term_obs = torch.cat([self.obs_stack[reset_env_ids].view(len(reset_env_ids), -1), self.priv_info[reset_env_ids]], dim=-1)
            info["terminal_observation"] = full_term_obs
            self.reset(reset_env_ids)

        full_obs = torch.cat([self.obs_stack.view(self.num_envs, -1), self.priv_info], dim=-1)
        return full_obs, rewards, terminated, truncated, info

    def _randomize_domains(self, env_ids):
        num_resets = len(env_ids)
        
        # 负载 0~5kg
        self.dr_payload_mass[env_ids] = torch.rand(num_resets, device=self.device) * (self.cfg.payload_mass_range[1] - self.cfg.payload_mass_range[0]) + self.cfg.payload_mass_range[0]
        # 重心偏移 ±10cm
        self.dr_com_shift[env_ids] = torch.rand((num_resets, 3), device=self.device) * (self.cfg.com_shift_range[1] - self.cfg.com_shift_range[0]) + self.cfg.com_shift_range[0]
        # 模拟摩擦力 0.2~1.5 (隐式传入 PrivInfo 供 Teacher 学习)
        self.dr_friction[env_ids] = torch.rand(num_resets, device=self.device) * 1.3 + 0.2
        
        # 电机衰减 (随机取 2 个关节，输出削弱到 70%-100%)
        self.dr_motor_strength[env_ids] = 1.0
        num_degraded = 2
        for i in range(num_resets):
            bad_joints = torch.randperm(12)[:num_degraded]
            degrade_factors = torch.rand(num_degraded, device=self.device) * (self.cfg.motor_strength_range[1] - self.cfg.motor_strength_range[0]) + self.cfg.motor_strength_range[0]
            self.dr_motor_strength[env_ids[i], bad_joints] = degrade_factors

    def _apply_external_disturbances(self):
        # 1. 计算负载恒定扭矩 (Payload Wrench = CoM_Shift x Force_Gravity)
        payload_force = torch.zeros((self.num_envs, 3), device=self.device)
        payload_force[:, 2] = -self.dr_payload_mass * 9.81
        payload_torque = torch.cross(self.dr_com_shift, payload_force, dim=-1)
        
        # 2. 飞踢逻辑 (Impulse Push)
        self.push_timers -= self.dt
        push_active = self.push_timers <= 0.0
        
        # 生成随机推力
        push_angles = torch.rand(self.num_envs, device=self.device) * 2 * math.pi
        push_mags = torch.rand(self.num_envs, device=self.device) * (self.cfg.push_magnitude_range[1] - self.cfg.push_magnitude_range[0]) + self.cfg.push_magnitude_range[0]
        
        self.push_forces[:, 0] = torch.cos(push_angles) * push_mags
        self.push_forces[:, 1] = torch.sin(push_angles) * push_mags
        self.push_forces[:, 2] = (torch.rand(self.num_envs, device=self.device) - 0.5) * 50.0 # 轻微上下砸力
        
        # 非触发态力清零
        self.push_forces = torch.where(push_active.unsqueeze(-1), self.push_forces, torch.zeros_like(self.push_forces))
        
        # 重置触发完成的 Timer
        reset_mask = self.push_timers <= -self.cfg.push_duration_s
        self.push_timers[reset_mask] = torch.rand(reset_mask.sum(), device=self.device) * (self.cfg.push_interval_s[1] - self.cfg.push_interval_s[0]) + self.cfg.push_interval_s[0]

        # 3. 施加到仿真引擎 Base Link
        total_forces = payload_force + self.push_forces
        
        # 将标量 0 改为序列 [0]，同时对力进行维度扩展 (num_envs, 1, 3) 适配底层接口
        self.robot.set_external_force_and_torque(
            total_forces.unsqueeze(1), 
            payload_torque.unsqueeze(1), 
            body_ids=[0]
        )

    def _compute_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_lin_vel = self.robot.data.root_lin_vel_b       
        base_ang_vel = self.robot.data.root_ang_vel_b       
        proj_gravity = self.robot.data.projected_gravity_b  
        joint_pos_residual = self.robot.data.joint_pos - self.default_joint_pos 
        joint_vel = self.robot.data.joint_vel               
        
        # 注入感知高斯噪声 (Sensory Noise)
        base_ang_vel_noisy = base_ang_vel + torch.randn_like(base_ang_vel) * self.cfg.noise_scales["base_ang_vel"]
        proj_gravity_noisy = proj_gravity + torch.randn_like(proj_gravity) * self.cfg.noise_scales["proj_gravity"]
        joint_pos_noisy = joint_pos_residual + torch.randn_like(joint_pos_residual) * self.cfg.noise_scales["joint_pos"]
        joint_vel_noisy = joint_vel + torch.randn_like(joint_vel) * self.cfg.noise_scales["joint_vel"]

        obs_single = torch.cat([
            base_lin_vel,          # 3
            base_ang_vel_noisy,    # 3
            proj_gravity_noisy,    # 3
            joint_pos_noisy,       # 12
            joint_vel_noisy,       # 12
            self.commands,         # 3
            self.last_action       # 12
        ], dim=-1)
        
        # 组装特权信息 E_t (Teacher 开卷答案)
        priv_info = torch.cat([
            self.dr_friction.unsqueeze(-1),        # 1
            self.dr_payload_mass.unsqueeze(-1),    # 1
            self.dr_com_shift,                     # 3
            self.push_forces[:, :2],               # 2 (仅 XY 推力)
            self.dr_motor_strength                 # 12
        ], dim=-1)
        
        return obs_single, priv_info

    def _compute_rewards_and_dones(self, action, old_action):
        base_pos = self.robot.data.root_pos_w - self.scene.env_origins
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b
        torques = self.robot.data.applied_torque
        
        # --- 1. 主任务追踪奖励 (Risk-Aware Tracking) ---
        
        # 注意：这里我们单独抽出 X 轴、Y 轴和 Yaw 轴，进行解耦计算
        vx_err = torch.abs(base_lin_vel[:, 0] - 0.5)
        vy_err = torch.abs(base_lin_vel[:, 1] - 0.0)
        w_z_err = torch.abs(base_ang_vel[:, 2] - 0.0)
        
        # 综合平面速度误差 (Y轴偏离也会算作误差)
        v_xy_err = torch.sqrt(torch.square(vx_err) + torch.square(vy_err))
        
        r_vx = torch.clamp(1.0 - (v_xy_err / 0.5), min=0.0)
        r_wz = torch.clamp(1.0 - (w_z_err / 0.3), min=0.0) # 转向容忍度给 0.3rad/s
        
        # --- 2. 姿态与抗扰恢复 (Posture & Recovery) ---
        # 隐式计算 Roll 和 Pitch 平方和 (proj_gravity x/y 分量)
        rp_sq = torch.square(proj_gravity[:, 0]) + torch.square(proj_gravity[:, 1])
        r_ori = torch.exp(-self.cfg.sigma_u * rp_sq)
        
        h_err = torch.square(base_pos[:, 2] - self.cfg.target_height)
        r_h = torch.exp(-self.cfg.sigma_h * h_err)
        
        # 核心：扰动恢复机制 (Recovery Reward)
        # 误差定义：速度偏差 + 姿态倾斜
        current_error = v_xy_err + w_z_err + torch.sqrt(rp_sq)
        # 若当前误差小于上一帧，且幅度明显，给予高额恢复奖励
        r_rec = torch.clamp((self.last_tracking_error - current_error) / (self.last_tracking_error + 1e-4), min=0.0, max=1.0)
        self.last_tracking_error = current_error.detach()
        
        # --- 3. 经济效率代价惩罚 (Penalty Costs) ---
        p_tau = torch.sum(torch.square(torques), dim=-1)
        p_act = torch.sum(torch.square(action - old_action), dim=-1)
        
        # 打滑惩罚 (Slip)：仅在足端触地时惩罚足端水平速度
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_indices, :2]
        foot_forces = self.contact.data.net_forces_w[:, :, 2]
        in_contact = foot_forces > 5.0
        foot_slip_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)
        p_slip = torch.sum(foot_slip_sq * in_contact.float(), dim=-1)
        
        # 冲击惩罚 (Impact)：抑制落地跺脚损坏硬件
        p_imp = torch.sum(torch.clamp(foot_forces - self.cfg.impact_threshold, min=0.0), dim=-1)
        
        # 关节极限惩罚
        joint_pos = self.robot.data.joint_pos
        p_j = torch.sum(torch.square(torch.clamp(joint_pos - self.q_limits_high, min=0.0)) + 
                        torch.square(torch.clamp(self.q_limits_low - joint_pos, min=0.0)), dim=-1)
        r_ori = r_ori * r_vx
        r_h = r_h * r_vx
        # --- 4. 汇总截断与极刑 ---
        continuous_rew = (
            self.cfg.w_vx * r_vx + self.cfg.w_wz * r_wz + 
            self.cfg.w_ori * r_ori + self.cfg.w_h * r_h + 
            self.cfg.w_rec * r_rec +
            self.cfg.w_tau * p_tau + self.cfg.w_act * p_act +
            self.cfg.w_slip * p_slip + self.cfg.w_imp * p_imp + self.cfg.w_j * p_j
        )
        
        # 确保 Dense 连续奖励被死死收束在 [-1.0, 1.0] 安全阈值内，预留极刑空间
        step_reward = torch.clamp(continuous_rew, min=-1.0, max=1.0)
        
        # 跌倒判定：底盘高度过低 或 躯干严重翻滚(Z轴重力投影反转)
        is_fallen = (base_pos[:, 2] < 0.22) | (proj_gravity[:, 2] > -0.4)
        
        # 触发极刑：脱离截断覆盖，保证死亡产生尖锐梯度
        rew_total = torch.where(is_fallen, step_reward + self.cfg.rew_fall, step_reward)
        
        terminated = is_fallen
        truncated = self.episode_length_buf >= self.cfg.max_episode_length
        
        # --- 5. 构建遥测分析字典 ---
        info = {
            "reward_components": {
                "R_Vel_XY": (self.cfg.w_vx * r_vx).mean().item(),
                "R_Vel_Wz": (self.cfg.w_wz * r_wz).mean().item(),
                "R_Posture": (self.cfg.w_ori * r_ori).mean().item(),
                "R_Height": (self.cfg.w_h * r_h).mean().item(),
                "R_Recovery": (self.cfg.w_rec * r_rec).mean().item(),
                "P_Torque": (self.cfg.w_tau * p_tau).mean().item(),
                "P_Action": (self.cfg.w_act * p_act).mean().item(),
                "P_Slip": (self.cfg.w_slip * p_slip).mean().item(),
                "P_Impact": (self.cfg.w_imp * p_imp).mean().item(),
                "P_JointLim": (self.cfg.w_j * p_j).mean().item(),
            },
            "telemetry": {
                "mean_vel_err": v_xy_err.mean().item(),
                "fall_rate": is_fallen.float().mean().item(),
                "active_push_ratio": (self.push_timers <= 0.0).float().mean().item()
            }
        }
        info["is_success"] = truncated # 活到最后即成功
        
        return rew_total, terminated, truncated, info