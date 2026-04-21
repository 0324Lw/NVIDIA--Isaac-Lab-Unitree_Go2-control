import torch
import numpy as np
import gymnasium as gym
import math
from typing import Dict, Tuple, Any

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils import configclass

# 导入原生宇树 Go2 配置
from isaaclab_assets import UNITREE_GO2_CFG

# ===================================================================
# 1. RL 与环境参数配置类
# ===================================================================
class Task1Config:
    # --- 1. 基础并行与仿真参数 ---
    num_envs = 2048            # RTX 5060 8G 显存足以支撑 4000+ 并行
    device = "cuda:0"
    sim_dt = 0.005             # 物理仿真频率 200Hz (解算 PD)
    decimation = 4             # 策略控制频率 50Hz (dt = 0.005 * 4 = 0.02s)
    max_episode_length = 1000  # 最大步数约 20s
    frame_stack = 3            # 3 帧观测堆叠
    
    # --- 2. 动作与控制参数 ---
    action_scale = 0.25        # 神经网络输出的 [-1, 1] 映射到真实的弧度偏移
    action_ema_alpha = 0.5     # 动作平滑系数 (指数移动平均)
    target_height = 0.30       # 期望奔跑的基座高度 (米)

    # --- 3. 随机指令范围 ---
    cmd_vx_range = [0.5, 1.0]  # 前进线速度 (m/s)
    cmd_vy_range = [-0.3, 0.3] # 侧向线速度 (m/s)
    cmd_wz_range = [-0.3, 0.3] # 转向角速度 (rad/s)

    # --- 4. 高斯奖励系数与权重 ---
    w_surv = 0.05              # 活着只给一点点安慰奖
    w_vx   = 0.40              # 前向速度追踪
    w_vy   = 0.10
    w_wz   = 0.10              
    w_h    = 0.10              
    w_ori  = 0.10              
    w_act  = 0.05              
    w_tau  = 0.10              # 稍微加大对电机发热的惩罚
    
    # 高斯函数的敏感度控制
    sigma_vel = 0.25           # 速度误差敏感度
    sigma_wz  = 0.25
    sigma_h   = 0.02           # 高度极其敏感
    sigma_ori = 0.05           # 姿态倾斜敏感度
    sigma_act = 2.0
    sigma_tau = 2000.0         # 扭矩平方和通常较大

    # --- 5. 离散极刑 ---
    rew_fall = -20.0           # 跌倒重置惩罚

# ===================================================================
# 2. 交互场景配置 (仅包含可交互的机器人)
# ===================================================================
@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    num_envs: int = Task1Config.num_envs
    env_spacing: float = 2.5
    # 直接利用 Isaac Lab 的原生 Go2 资产
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# ===================================================================
# 3. 机器狗平地奔跑环境类
# ===================================================================
class QuadrupedFlatEnv(gym.Env):
    def __init__(self, cfg: Task1Config):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        
        # 1. 物理引擎初始化
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.sim_dt, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        # 手动生成静态资产，避免 InteractiveScene 解析报错
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
        light_cfg.func("/World/Light", light_cfg)
        
        # 2. 生成交互场景
        scene_cfg = QuadrupedSceneCfg()
        scene_cfg.num_envs = self.num_envs
        self.scene = InteractiveScene(scene_cfg)
        
        self.robot = self.scene.articulations["robot"]
        
        # 3. 空间与缓冲区定义
        self.obs_dim_per_frame = 48  # 基座9 + 关节24 + 指令3 + 历史动作12 = 48
        self.total_obs_dim = self.obs_dim_per_frame * self.cfg.frame_stack
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))
        
        # 张量缓存
        self.obs_stack = torch.zeros((self.num_envs, self.cfg.frame_stack, self.obs_dim_per_frame), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 12), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device) # [Vx, Vy, Wz]
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 提取默认站立姿态
        self.sim.reset()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

    def reset(self, env_ids=None, options=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # 1. 物理状态重置
        state = self.robot.data.default_root_state[env_ids].clone()
        # 加入微小的出生高度扰动，防止同一时间坠落
        state[:, 2] += torch.rand_like(state[:, 2]) * 0.05 
        self.robot.write_root_state_to_sim(state, env_ids)
        # 必须显式指定 env_ids=env_ids，否则会被错误识别为 joint_ids
        self.robot.write_joint_state_to_sim(self.default_joint_pos[env_ids], torch.zeros_like(self.default_joint_pos[env_ids]), env_ids=env_ids)
        
        # 2. 随机生成本回合的目标速度指令
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_vx_range[1] - self.cfg.cmd_vx_range[0]) + self.cfg.cmd_vx_range[0]
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_vy_range[1] - self.cfg.cmd_vy_range[0]) + self.cfg.cmd_vy_range[0]
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_wz_range[1] - self.cfg.cmd_wz_range[0]) + self.cfg.cmd_wz_range[0]
        
        # 3. 缓存重置
        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        
        self.scene.update(0.0)
        obs_single = self._compute_obs()
        for i in range(self.cfg.frame_stack):
            self.obs_stack[env_ids, i, :] = obs_single[env_ids]

        return self.obs_stack[env_ids].view(len(env_ids), -1), {}

    def step(self, action: torch.Tensor):
        # 1. 动作平滑与解算 (EMA Filter)
        current_action = self.cfg.action_ema_alpha * action + (1.0 - self.cfg.action_ema_alpha) * self.last_action
        old_action = self.last_action.clone()
        self.last_action = current_action.clone()
        
        # 物理映射：目标角度 = 默认角度 + 网络输出 * 缩放
        target_pos = self.default_joint_pos + current_action * self.cfg.action_scale
        
        # 2. 物理仿真进帧 (运行 decimation 次 PD 闭环)
        self.robot.set_joint_position_target(target_pos)
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            
        self.scene.update(self.cfg.sim_dt * self.cfg.decimation)
        self.episode_length_buf += 1

        # 3. 计算观测与奖励
        obs_single = self._compute_obs()
        self.obs_stack = torch.roll(self.obs_stack, shifts=-1, dims=1)
        self.obs_stack[:, -1, :] = obs_single
        
        rewards, terminated, truncated, info = self._compute_rewards_and_dones(current_action, old_action)

        # 4. 环境重置处理
        resets = terminated | truncated
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # PPO Bootstrap 所需的真实终端状态
            info["terminal_observation"] = self.obs_stack[reset_env_ids].clone().view(len(reset_env_ids), -1)
            self.reset(reset_env_ids)

        return self.obs_stack.view(self.num_envs, -1), rewards, terminated, truncated, info

    def _compute_obs(self):
        # 从底层读取物理张量，全部在基座局部坐标系 (Local Frame)
        base_lin_vel = self.robot.data.root_lin_vel_b       # [num_envs, 3]
        base_ang_vel = self.robot.data.root_ang_vel_b       # [num_envs, 3]
        proj_gravity = self.robot.data.projected_gravity_b  # [num_envs, 3]
        
        joint_pos_residual = self.robot.data.joint_pos - self.default_joint_pos # [num_envs, 12]
        joint_vel = self.robot.data.joint_vel               # [num_envs, 12]
        
        # 拼接 48 维单帧特征
        obs = torch.cat([
            base_lin_vel,          # 3
            base_ang_vel,          # 3
            proj_gravity,          # 3
            joint_pos_residual,    # 12
            joint_vel,             # 12
            self.commands,         # 3
            self.last_action       # 12
        ], dim=-1)
        
        return obs

    def _compute_rewards_and_dones(self, action, old_action):
        # --- 数据读取 ---
        base_pos = self.robot.data.root_pos_w - self.scene.env_origins
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b
        torques = self.robot.data.applied_torque
        
        # --- 1. 高斯驱动连续奖励计算 (均保证结果在 0~1 之间) ---
        # A. 速度追踪误差
        err_vx = torch.square(base_lin_vel[:, 0] - self.commands[:, 0])
        err_vy = torch.square(base_lin_vel[:, 1] - self.commands[:, 1])
        err_wz = torch.square(base_ang_vel[:, 2] - self.commands[:, 2])
        r_vx = torch.exp(-err_vx / self.cfg.sigma_vel)
        r_vy = torch.exp(-err_vy / self.cfg.sigma_vel)
        r_wz = torch.exp(-err_wz / self.cfg.sigma_wz)
        
        # B. 姿态与高度
        err_h = torch.square(base_pos[:, 2] - self.cfg.target_height)
        r_h = torch.exp(-err_h / self.cfg.sigma_h)
        
        err_ori = torch.square(proj_gravity[:, 0]) + torch.square(proj_gravity[:, 1])
        r_ori = torch.exp(-err_ori / self.cfg.sigma_ori)
        
        # C. 动作平滑与电机保护
        err_act = torch.sum(torch.square(action - old_action), dim=1)
        r_act = torch.exp(-err_act / self.cfg.sigma_act)
        
        err_tau = torch.sum(torch.square(torques), dim=1)
        r_tau = torch.exp(-err_tau / self.cfg.sigma_tau)
        
        # --- 2. 连续分数组合与截断 ---
        # 依据数学权重，sum(weights) = 1.0，因此 continuous_rew 必然落在 (0, 1.0] 内
        continuous_rew = (
            self.cfg.w_surv * 1.0 + 
            self.cfg.w_vx * r_vx + self.cfg.w_vy * r_vy + self.cfg.w_wz * r_wz +
            self.cfg.w_h * r_h + self.cfg.w_ori * r_ori + 
            self.cfg.w_act * r_act + self.cfg.w_tau * r_tau
        )
        # 数值稳定性：截断在 [-1, 1] 之间 (预留梯度)
        rew_total = torch.clamp(continuous_rew, -1.0, 1.0)
        
        # --- 3. 终局条件与极刑惩罚 ---
        # 1. 高度低于 0.2m 视为底盘托底。
        # 2. 正常站立时 proj_gravity[:, 2] 约为 -1.0。如果 > -0.2，说明狗已经倾斜超过 78度（即将四脚朝天）。
        is_fallen = (base_pos[:, 2] < 0.20) | (proj_gravity[:, 2] > -0.2)
        
        # 极刑惩罚脱离截断，保证尖锐信号
        rew_total = torch.where(is_fallen, rew_total + self.cfg.rew_fall, rew_total)
        
        terminated = is_fallen
        truncated = self.episode_length_buf >= self.cfg.max_episode_length
        
        # --- 4. Info 字典构建，用于遥测监控 ---
        info = {
            "reward_components": {
                "r_vx": r_vx.mean().item() * self.cfg.w_vx,
                "r_vy": r_vy.mean().item() * self.cfg.w_vy,
                "r_wz": r_wz.mean().item() * self.cfg.w_wz,
                "r_height": r_h.mean().item() * self.cfg.w_h,
                "r_ori": r_ori.mean().item() * self.cfg.w_ori,
                "r_action": r_act.mean().item() * self.cfg.w_act,
                "r_torque": r_tau.mean().item() * self.cfg.w_tau
            },
            "telemetry": {
                "mean_vel_x": base_lin_vel[:, 0].mean().item(),
                "mean_height": base_pos[:, 2].mean().item(),
                "fall_rate": is_fallen.float().mean().item()
            }
        }
        info["is_success"] = truncated
        return rew_total, terminated, truncated, info