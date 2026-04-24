import torch
import numpy as np
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

from isaaclab_assets import UNITREE_GO2_CFG
from task2_world import Task2TerrainCfg, Task2World, TerrainCurriculum

# ===================================================================
# 1. RL 與環境參數配置類
# ===================================================================
class Task2Config:
    num_envs = 1024
    device = "cuda:0"
    sim_dt = 0.005
    decimation = 4
    max_episode_length = 1000
    frame_stack = 5
    
    action_scale = 0.5
    action_ema_alpha = 0.5
    
    
    # 死区与高度配置
    target_height = 0.32
    height_deadband = 0.02       
    clearance_threshold = 0.12

    # 二维指令课程
    cmd_vx_range = [0.3, 1.2]    
    cmd_vy_range = [0.0, 0.0]    
    cmd_wz_range = [0.0, 0.0]    

    # ================= 1. 任务追踪 (正向激励, 占比 ~65%) =================
    w_vx   = 2.0       
    w_vy   = 0.0
    w_wz   = 0.0
    
    # ================= 2. 步态约束 (条件正向激励, 占比 ~15%) =================
    w_clearance = 0.15   # [上调] 略微强化抬腿，以应对 Level 9 的乱石与台阶
    w_air_time = 0.20    # 维持 0.2，巩固对角小跑 (Trot) 的肌肉记忆
    
    # ================= 3. 底盘稳定 (负向惩罚，防跌倒/防侧翻) =================
    w_lin_vel_z = 0.02 
    w_ang_vel_xy = 0.005 # 维持 0.005，允许合理的身体律动，但不允许剧烈抽搐
    w_ori = 0.15         # 略微增强水平面约束，防爬台阶时仰翻
    w_height = 0.5       # 配合死区机制，严打“丧尸匍匐”

    # ================= 4. 能耗、平顺与硬件保护 (负向惩罚，防烧电机) =================
    w_dof_pos = 0.02     # 软皮筋约束，不影响跑步，但限制了畸形扭曲
    w_tau = 1e-5         # 限制扭矩
    w_dof_vel = 5e-5     # 限制关节超速
    w_action_rate = 0.005 # 保证动作平滑，防抖动

    # ================= 5. 安全边界与高斯核 =================
    rew_fall = -10.0   

    sigma_vel = 0.05     # [核心过滤器] 极其敏锐的速度追踪门槛
    sigma_h   = 0.05
    sigma_ori = 0.1
    sigma_act = 2.0
    sigma_tau = 2000.0

robot_cfg_safe = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# 限制最大反穿透速度
if robot_cfg_safe.spawn.rigid_props is None:
    robot_cfg_safe.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
robot_cfg_safe.spawn.rigid_props.max_depenetration_velocity = 5.0 

# 开启自我碰撞
if robot_cfg_safe.spawn.articulation_props is None:
    robot_cfg_safe.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg()
robot_cfg_safe.spawn.articulation_props.enabled_self_collisions = False

@configclass
class QuadrupedRoughSceneCfg(InteractiveSceneCfg):
    num_envs: int = Task2Config.num_envs
    env_spacing: float = 0.0
    robot: ArticulationCfg = robot_cfg_safe

# ===================================================================
# 3. 盲爬多地形环境
# ===================================================================
class QuadrupedRoughEnv(gym.Env):
    def __init__(self, cfg: Task2Config):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        
        self.terrain_cfg = Task2TerrainCfg()
        self.world = Task2World(self.terrain_cfg, self.device)
        self.curriculum = TerrainCurriculum(self.num_envs, self.terrain_cfg, self.device)
        
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim_dt, 
            device=self.device,
            physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True)
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        scene_cfg = QuadrupedRoughSceneCfg(num_envs=self.num_envs)
        scene_cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=self.world.generator_cfg,
            max_init_terrain_level=self.terrain_cfg.num_cols - 1,
            collision_group=-1
        )
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)
        
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene.articulations["robot"]
        
        self.obs_dim_per_frame = 56
        self.actor_obs_dim = self.obs_dim_per_frame * self.cfg.frame_stack 
        self.critic_priv_dim = 81 + 4 
        self.critic_total_dim = self.actor_obs_dim + self.critic_priv_dim  
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.actor_obs_dim,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))
        
        self.obs_stack = torch.zeros((self.num_envs, self.cfg.frame_stack, self.obs_dim_per_frame), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 12), device=self.device)
        self.last_lin_vel = torch.zeros((self.num_envs, 3), device=self.device) 
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)     
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_frictions = torch.ones(self.num_envs, device=self.device)
        self.env_terrain_z = torch.zeros(self.num_envs, device=self.device)
        
        self.sim.reset()

        safe_standing_pose = torch.tensor([
             0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
             0.1, 1.0, -1.5,
            -0.1, 1.0, -1.5 
        ], device=self.device)
        self.default_joint_pos = safe_standing_pose.unsqueeze(0).repeat(self.num_envs, 1)

        self.default_root_state = torch.zeros((self.num_envs, 13), device=self.device)
        self.default_root_state[:, 3] = 1.0  

        calf_idx, _ = self.robot.find_bodies(".*calf.*")
        if len(calf_idx) == 0:
            num_bodies = self.robot.data.body_pos_w.shape[1]
            calf_idx = [max(0, num_bodies - 4 + i) for i in range(4)]
        self.calf_indices = torch.tensor(calf_idx, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None, options=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long)
            
        env_types, env_levels = self.curriculum.get_current_grid_indices(env_ids)
        
        flat_indices = env_types * self.terrain_cfg.num_cols + env_levels
        safe_indices = torch.clamp(flat_indices, min=0, max=self.scene.terrain.env_origins.shape[0] - 1)
        origins = self.scene.terrain.env_origins[safe_indices].clone()
        self.env_terrain_z[env_ids] = origins[:, 2].clone()
        
        scatter_x = (torch.rand(len(env_ids), device=self.device) - 0.5) * 4.0
        scatter_y = (torch.rand(len(env_ids), device=self.device) - 0.5) * 4.0
        origins[:, 0] += scatter_x
        origins[:, 1] += scatter_y
        origins[:, 2] += 0.6  
        
        root_state = self.default_root_state[env_ids].clone()
        root_state[:, :3] = origins
        
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        
        joint_p = self.default_joint_pos[env_ids]
        joint_v = torch.zeros_like(joint_p)
        self.robot.write_joint_state_to_sim(joint_p, joint_v, env_ids=env_ids)
        
        zero_velocities = torch.zeros((len(env_ids), 6), device=self.device)
        self.robot.write_root_velocity_to_sim(zero_velocities, env_ids=env_ids)
        
        self.env_frictions[env_ids] = torch.rand(len(env_ids), device=self.device) * (self.terrain_cfg.friction_range[1] - self.terrain_cfg.friction_range[0]) + self.terrain_cfg.friction_range[0]
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_vx_range[1] - self.cfg.cmd_vx_range[0]) + self.cfg.cmd_vx_range[0]
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_vy_range[1] - self.cfg.cmd_vy_range[0]) + self.cfg.cmd_vy_range[0]
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (self.cfg.cmd_wz_range[1] - self.cfg.cmd_wz_range[0]) + self.cfg.cmd_wz_range[0]
        
        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.last_lin_vel[env_ids] = 0.0
        
        self.curriculum.register_start_positions(env_ids, origins[:, 0])
        self.scene.update(0.0)
        
        obs_single, priv_single = self._compute_obs()
        
        for i in range(self.cfg.frame_stack):
            self.obs_stack[env_ids, i, :] = obs_single[env_ids]

        actor_obs = self.obs_stack[env_ids].view(len(env_ids), -1)
        critic_obs = torch.cat([actor_obs, priv_single[env_ids]], dim=-1)
        info = {"privileged_obs": critic_obs}
        
        return actor_obs, info

    def step(self, action: torch.Tensor):
        current_action = self.cfg.action_ema_alpha * action + (1.0 - self.cfg.action_ema_alpha) * self.last_action
        old_action = self.last_action.clone()
        self.last_action = current_action.clone()
        
        target_pos = self.default_joint_pos + current_action * self.cfg.action_scale
        self.robot.set_joint_position_target(target_pos)
            
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            
        self.scene.update(self.cfg.sim_dt * self.cfg.decimation)
        self.episode_length_buf += 1

        obs_single, priv_single = self._compute_obs()
        
        num_bodies = self.robot.data.body_pos_w.shape[1]
        if num_bodies > 0:
            self.last_lin_vel = self.robot.data.root_lin_vel_b.clone()
        
        self.obs_stack = torch.roll(self.obs_stack, shifts=-1, dims=1)
        self.obs_stack[:, -1, :] = obs_single
        
        actor_obs = self.obs_stack.view(self.num_envs, -1)
        critic_obs = torch.cat([actor_obs, priv_single], dim=-1)
        
        rewards, terminated, truncated, info = self._compute_rewards_and_dones(current_action, old_action)
        info["privileged_obs"] = critic_obs

        resets = terminated | truncated
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            if num_bodies > 0:
                current_x = self.robot.data.root_pos_w[:, 0]
                real_fallen = info["is_fallen"][reset_env_ids]
                self.curriculum.update_curriculum(reset_env_ids, current_x, info["is_fallen"])
            
            info["terminal_observation"] = actor_obs[reset_env_ids].clone()
            info["terminal_privileged_obs"] = critic_obs[reset_env_ids].clone()
            
            self.reset(reset_env_ids)

        return self.obs_stack.view(self.num_envs, -1), rewards, terminated, truncated, info

    def _compute_obs(self):
        num_bodies = self.robot.data.body_pos_w.shape[1]
        if num_bodies == 0:
            return torch.zeros((self.num_envs, 56), device=self.device), torch.zeros((self.num_envs, 85), device=self.device)

        max_idx = max(0, num_bodies - 1)
        safe_calf_indices = torch.clamp(self.calf_indices, min=0, max=max_idx)

        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b
        
        dt = self.cfg.sim_dt * self.cfg.decimation
        lin_acc = (base_lin_vel - self.last_lin_vel) / dt
        
        base_pos = self.robot.data.root_pos_w
        relative_z = base_pos[:, 2] - self.env_terrain_z
        z_err = (relative_z - self.cfg.target_height).unsqueeze(1)
        
        joint_pos_residual = self.robot.data.joint_pos - self.default_joint_pos
        joint_vel = self.robot.data.joint_vel
        
        calf_pos_z = self.robot.data.body_pos_w[:, safe_calf_indices, 2]
        contacts = (calf_pos_z < (self.env_terrain_z.unsqueeze(1) + 0.12)).float()
        
        obs_single = torch.cat([
            base_lin_vel, base_ang_vel, proj_gravity, lin_acc, z_err, 
            joint_pos_residual, joint_vel,                            
            contacts,                                                 
            self.commands, self.last_action                           
        ], dim=-1) 
        
        terrain_scan = torch.zeros((self.num_envs, 81), device=self.device) 
        dummy_pushes = torch.zeros((self.num_envs, 3), device=self.device) 
        priv_single = torch.cat([
            terrain_scan,                               
            self.env_frictions.unsqueeze(1),            
            dummy_pushes                                
        ], dim=-1) 
        
        return obs_single, priv_single

    def _compute_rewards_and_dones(self, action, old_action):
        if self.robot.data.body_pos_w.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.device), torch.ones(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), {}

        base_pos = self.robot.data.root_pos_w
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b
        torques = self.robot.data.applied_torque
        joint_vel = self.robot.data.joint_vel
        
        max_idx = max(0, self.robot.data.body_pos_w.shape[1] - 1)
        safe_calf_indices = torch.clamp(self.calf_indices, min=0, max=max_idx)
        calf_pos_z = self.robot.data.body_pos_w[:, safe_calf_indices, 2]
        
        relative_z = base_pos[:, 2] - self.env_terrain_z
        
        # ---------------- 1. 任务追踪层 ----------------
        err_vx = torch.square(base_lin_vel[:, 0] - self.commands[:, 0])
        err_vy = torch.square(base_lin_vel[:, 1] - self.commands[:, 1])
        err_wz = torch.square(base_ang_vel[:, 2] - self.commands[:, 2])
        r_vx = torch.exp(-err_vx / self.cfg.sigma_vel)
        r_vy = torch.exp(-err_vy / self.cfg.sigma_vel)
        r_wz = torch.exp(-err_wz / self.cfg.sigma_vel)
        
        # ---------------- 2. 步态与足端层  ----------------
        contacts = calf_pos_z < (self.env_terrain_z.unsqueeze(1) + 0.12)
        swing_phase = ~contacts
        
        # 计算基础步态得分
        clearance_err = torch.square(torch.clamp_max(self.cfg.clearance_threshold - (calf_pos_z - self.env_terrain_z.unsqueeze(1)), 0.0))
        base_clearance_reward = torch.exp(-torch.sum(clearance_err * swing_phase, dim=1) / 0.05)
        
        num_air_legs = swing_phase.sum(dim=1)
        base_air_time_reward = (num_air_legs >= 2).float()
        
        # 只有当实际前向追踪得分 (r_vx) 很高时，才发放步态奖励
        # 如果狗在原地罚站 (r_vx 很低)，步态奖励将直接归零
        r_clearance = base_clearance_reward * r_vx
        r_air_time = base_air_time_reward * r_vx

        # ---------------- 3. 底盘稳定层  ----------------
        r_lin_vel_z = torch.square(base_lin_vel[:, 2])
        r_ang_vel_xy = torch.square(base_ang_vel[:, 0]) + torch.square(base_ang_vel[:, 1])
        cmd_norm = torch.sqrt(torch.square(self.commands[:, 0]) + torch.square(self.commands[:, 1]))
        
        # 高速奔跑时，自然放宽对姿态倾斜的容忍度
        ori_multiplier = torch.clamp(1.0 - 0.6 * cmd_norm, min=0.2, max=1.0)
        r_ori = (torch.square(proj_gravity[:, 0]) + torch.square(proj_gravity[:, 1])) * ori_multiplier
        
        # 高度误差在死区内(±0.04m)免于扣分
        height_error_abs = torch.abs(relative_z - self.cfg.target_height)
        height_error_deadband = torch.clamp(height_error_abs - self.cfg.height_deadband, min=0.0)
        r_height = torch.square(height_error_deadband)

        # ---------------- 4. 能耗、平顺与姿态约束层 ----------------
        joint_pos_residual = self.robot.data.joint_pos - self.default_joint_pos
        r_dof_pos = torch.sum(torch.square(joint_pos_residual), dim=1)
        
        r_tau = torch.sum(torch.square(torques), dim=1)
        r_dof_vel = torch.sum(torch.square(joint_vel), dim=1)
        r_action_rate = torch.sum(torch.square(action - old_action), dim=1)

        # ---------------- 5. 权重汇总 ----------------
        rew_total = (
            self.cfg.w_vx * r_vx + self.cfg.w_vy * r_vy + self.cfg.w_wz * r_wz +
            self.cfg.w_clearance * r_clearance + self.cfg.w_air_time * r_air_time -
            self.cfg.w_lin_vel_z * r_lin_vel_z - self.cfg.w_ang_vel_xy * r_ang_vel_xy -
            self.cfg.w_ori * r_ori - self.cfg.w_height * r_height -
            self.cfg.w_tau * r_tau - self.cfg.w_dof_vel * r_dof_vel - self.cfg.w_action_rate * r_action_rate -
            self.cfg.w_dof_pos * r_dof_pos  
        )
        rew_total = torch.clamp(rew_total, min=-5.0, max=5.0)

        body_crash = relative_z < 0.15
        posture_crash = proj_gravity[:, 2] > -0.4
        z_out_of_bounds = base_pos[:, 2] < -1.0
        distance_walked = base_pos[:, 0] - self.curriculum.env_start_pos_x
        is_success = distance_walked > 3.0
        
        is_fallen = (posture_crash | body_crash | z_out_of_bounds) & (~is_success)
        
        rew_total = torch.where(is_fallen, rew_total + self.cfg.rew_fall, rew_total)
        rew_total = torch.where(is_success, rew_total + 5.0, rew_total) 
        
        terminated = is_fallen | is_success
        truncated = self.episode_length_buf >= self.cfg.max_episode_length
        
        # ---------------- 7. 详尽输出分析数据 ----------------
        info = {
            "is_fallen": is_fallen, 
            "reward_components": {
                # 追踪项 (+)
                "R_Vx_Track": (self.cfg.w_vx * r_vx).mean().item(),
                "R_Vy_Track": (self.cfg.w_vy * r_vy).mean().item(),
                "R_Wz_Track": (self.cfg.w_wz * r_wz).mean().item(),
                # 步态项 (+)
                "R_Clearance": (self.cfg.w_clearance * r_clearance).mean().item(),
                "R_Air_Time": (self.cfg.w_air_time * r_air_time).mean().item(),
                # 稳定惩罚项 (-)
                "P_Lin_Vel_Z": (-self.cfg.w_lin_vel_z * r_lin_vel_z).mean().item(),
                "P_Ang_Vel_XY": (-self.cfg.w_ang_vel_xy * r_ang_vel_xy).mean().item(),
                "P_Ori": (-self.cfg.w_ori * r_ori).mean().item(),
                "P_Height": (-self.cfg.w_height * r_height).mean().item(),
                # 平顺与能耗惩罚项 (-)
                "P_Tau": (-self.cfg.w_tau * r_tau).mean().item(),
                "P_Dof_Vel": (-self.cfg.w_dof_vel * r_dof_vel).mean().item(),
                "P_Action_Rate": (-self.cfg.w_action_rate * r_action_rate).mean().item(),
                "P_Dof_Pos": (-self.cfg.w_dof_pos * r_dof_pos).mean().item() 
            },
            "telemetry": {
                "mean_vel_x": base_lin_vel[:, 0].mean().item(),
                "fall_rate": is_fallen.float().mean().item(),
                "mean_relative_z": relative_z.mean().item(),
                "probe_height_deadband_rate": (height_error_abs <= self.cfg.height_deadband).float().mean().item(),
                "probe_ori_multiplier_mean": ori_multiplier.mean().item()
            }
        }
        info["telemetry"].update(self.curriculum.log_curriculum_stats())
        
        return rew_total, terminated, truncated, info