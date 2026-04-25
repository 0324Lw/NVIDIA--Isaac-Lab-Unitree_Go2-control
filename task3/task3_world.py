import torch
import math
from typing import Tuple

from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

# ===================================================================
# 1. 任务 3：视觉导航与动态避障 世界参数配置类 (Config)
# ===================================================================
@configclass
class Task3WorldCfg:
    pd_control_freq = 200.0
    rl_policy_freq = 50.0
    decimation = int(pd_control_freq / rl_policy_freq)

    env_size = 30.0  
    max_episode_length_s = 20.0 
    max_episode_steps = int(max_episode_length_s * rl_policy_freq) 

    goal_dist_range = [5.0, 10.0]  
    safe_zone_radius = 2.0          

    obstacle_height = 2.0
    
    num_static_obs = 15
    static_radius_range = [1.0, 1.5]
    min_static_spacing = 2
    
    num_dynamic_obs = 5
    dynamic_radius = 0.5      
    dynamic_speed_range = [0.5, 1.0] 

    robot_radius = 0.35       
    collision_threshold = 0.2 

    lidar_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", 
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        ray_alignment="yaw",
        mesh_prim_paths=["{ENV_REGEX_NS}/Obstacle_.*"], 
        pattern_cfg=patterns.BpearlPatternCfg(), 
        max_distance=5.0,
        debug_vis=False,
    )
    
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle_.*"], 
        update_period=0.0, 
    )

# ===================================================================
# 2. 任务 3 世界模型与逻辑管理类 (World)
# ===================================================================
class Task3World:
    def __init__(self, cfg: Task3WorldCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        self.start_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 2), device=self.device)
        
        self.static_obs = torch.zeros((self.num_envs, self.cfg.num_static_obs, 3), device=self.device)
        
        self.dynamic_obs_pos = torch.zeros((self.num_envs, self.cfg.num_dynamic_obs, 2), device=self.device)
        self.dynamic_obs_vel = torch.zeros((self.num_envs, self.cfg.num_dynamic_obs, 2), device=self.device)
        
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset_envs(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        num_resets = len(env_ids)
        half_env = self.cfg.env_size / 2.0
        
        # 提取最大边界，避免安全区贴墙
        spawn_bound = half_env - self.cfg.safe_zone_radius

        # ---------------------------------------------------------
        # 阶段 1: 随机生成起终点
        # 使用拒绝采样：不停在合法边界内成对随机生成起终点，直到距离满足 [20, 30]m
        # ---------------------------------------------------------
        valid_targets = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        while not valid_targets.all():
            pending = ~valid_targets
            num_pending = pending.sum()
            
            # 在合法场地内随机生成起点和终点
            test_starts = (torch.rand((num_pending, 2), device=self.device) * 2 - 1) * spawn_bound
            test_targets = (torch.rand((num_pending, 2), device=self.device) * 2 - 1) * spawn_bound
            
            # 计算距离
            dists = torch.norm(test_starts - test_targets, dim=-1)
            
            # 判断距离是否满足 20~30m
            is_valid = (dists >= self.cfg.goal_dist_range[0]) & (dists <= self.cfg.goal_dist_range[1])
            
            valid_indices = pending.nonzero(as_tuple=True)[0][is_valid]
            self.start_pos[env_ids[valid_indices]] = test_starts[is_valid]
            self.target_pos[env_ids[valid_indices]] = test_targets[is_valid]
            valid_targets[valid_indices] = True

        # ---------------------------------------------------------
        # 阶段 2: 静态障碍物
        # ---------------------------------------------------------
        for i in range(self.cfg.num_static_obs):
            valid_mask = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
            self.static_obs[env_ids, i, 2] = torch.empty(num_resets, device=self.device).uniform_(
                self.cfg.static_radius_range[0], self.cfg.static_radius_range[1])
            
            while not valid_mask.all():
                pending = ~valid_mask
                num_pending = pending.sum()
                r_test = self.static_obs[env_ids[pending], i, 2]
                
                max_bound = half_env - r_test.unsqueeze(1)
                test_pos = (torch.rand((num_pending, 2), device=self.device) * 2 - 1) * max_bound
                
                dist_to_start = torch.norm(test_pos - self.start_pos[env_ids[pending]], dim=-1)
                dist_to_target = torch.norm(test_pos - self.target_pos[env_ids[pending]], dim=-1)
                
                safe_start = dist_to_start > (self.cfg.safe_zone_radius + r_test)
                safe_target = dist_to_target > (self.cfg.safe_zone_radius + r_test)
                is_valid = safe_start & safe_target
                
                if i > 0:
                    dist_to_existing = torch.norm(test_pos.unsqueeze(1) - self.static_obs[env_ids[pending], :i, :2], dim=-1)
                    r_existing = self.static_obs[env_ids[pending], :i, 2] 
                    threshold = r_test.unsqueeze(1) + r_existing + self.cfg.min_static_spacing
                    conflict = (dist_to_existing < threshold).any(dim=-1)
                    is_valid &= ~conflict
                
                valid_indices = pending.nonzero(as_tuple=True)[0][is_valid]
                self.static_obs[env_ids[valid_indices], i, :2] = test_pos[is_valid]
                valid_mask[valid_indices] = True

        # ---------------------------------------------------------
        # 阶段 3: 动态障碍物 (修复穿模卡死 Bug)
        # ---------------------------------------------------------
        dyn_r = self.cfg.dynamic_radius
        for i in range(self.cfg.num_dynamic_obs):
            valid_mask = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
            
            while not valid_mask.all():
                pending = ~valid_mask
                num_pending = pending.sum()
                
                # 生成坐标时同样考虑环境边界约束
                max_bound = half_env - dyn_r
                test_pos = (torch.rand((num_pending, 2), device=self.device) * 2 - 1) * max_bound
                
                # 【修复核心】：检查是否与已生成的静态障碍物重叠（增加 0.1m 缓冲防止开局瞬间碰撞）
                dist_to_static = torch.norm(test_pos.unsqueeze(1) - self.static_obs[env_ids[pending], :, :2], dim=-1)
                stat_r = self.static_obs[env_ids[pending], :, 2]
                conflict_static = (dist_to_static < (dyn_r + stat_r + 0.1)).any(dim=-1)
                
                is_valid = ~conflict_static
                
                valid_indices = pending.nonzero(as_tuple=True)[0][is_valid]
                self.dynamic_obs_pos[env_ids[valid_indices], i] = test_pos[is_valid]
                valid_mask[valid_indices] = True
            
            # 生成合法速度
            vel_angles = torch.rand(num_resets, device=self.device) * 2 * math.pi
            vel_mags = torch.empty(num_resets, device=self.device).uniform_(self.cfg.dynamic_speed_range[0], self.cfg.dynamic_speed_range[1])
            self.dynamic_obs_vel[env_ids, i, 0] = vel_mags * torch.cos(vel_angles)
            self.dynamic_obs_vel[env_ids, i, 1] = vel_mags * torch.sin(vel_angles)

        self.episode_steps[env_ids] = 0

    def step_kinematics(self, dt: float):
        if self.num_envs == 0:
            return

        self.dynamic_obs_pos += self.dynamic_obs_vel * dt
        
        half_env = self.cfg.env_size / 2.0
        out_of_bounds_x = torch.abs(self.dynamic_obs_pos[:, :, 0]) > (half_env - self.cfg.dynamic_radius)
        out_of_bounds_y = torch.abs(self.dynamic_obs_pos[:, :, 1]) > (half_env - self.cfg.dynamic_radius)
        
        self.dynamic_obs_vel[:, :, 0] = torch.where(out_of_bounds_x, -self.dynamic_obs_vel[:, :, 0], self.dynamic_obs_vel[:, :, 0])
        self.dynamic_obs_vel[:, :, 1] = torch.where(out_of_bounds_y, -self.dynamic_obs_vel[:, :, 1], self.dynamic_obs_vel[:, :, 1])

        dyn_pos_exp = self.dynamic_obs_pos.unsqueeze(2)           
        stat_pos_exp = self.static_obs[..., :2].unsqueeze(1)      
        dist_matrix = torch.norm(dyn_pos_exp - stat_pos_exp, dim=-1) 
        
        stat_radius_exp = self.static_obs[..., 2].unsqueeze(1)
        collision_threshold = self.cfg.dynamic_radius + stat_radius_exp
        
        collision_mask = dist_matrix < collision_threshold 
        dyn_collision_flags = collision_mask.any(dim=-1)
        
        self.dynamic_obs_vel[dyn_collision_flags] *= -1.0
        
        self.episode_steps += 1

    def check_terminations(self, robot_pos: torch.Tensor, contact_forces: torch.Tensor, is_fallen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        timeout = self.episode_steps >= self.cfg.max_episode_steps
        dones |= timeout
        
        dones |= is_fallen
        rewards = torch.where(is_fallen, rewards - 50.0, rewards) 

        has_contact = torch.norm(contact_forces, dim=-1).max(dim=-1)[0] > 0.1 
        dones |= has_contact
        rewards = torch.where(has_contact, rewards - 100.0, rewards) 

        dist_to_target = torch.norm(robot_pos[:, :2] - self.target_pos, dim=-1)
        reached_target = dist_to_target < 0.5
        dones |= reached_target
        rewards = torch.where(reached_target, rewards + 200.0, rewards) 

        return dones, rewards

    def get_target_polar_coords(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor) -> torch.Tensor:
        delta_x = self.target_pos[:, 0] - robot_pos[:, 0]
        delta_y = self.target_pos[:, 1] - robot_pos[:, 1]
        
        distance = torch.norm(torch.stack([delta_x, delta_y], dim=1), dim=-1)
        target_angle = torch.atan2(delta_y, delta_x)
        relative_angle = torch.atan2(torch.sin(target_angle - robot_yaw), torch.cos(target_angle - robot_yaw))
        
        return torch.stack([distance, relative_angle], dim=-1)
    
    def compute_lidar_tensors(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor) -> torch.Tensor:
        """
        极速张量化 2D 激光雷达模拟 (返回形状: [num_envs, 90])
        基于数学解析几何的射线-圆相交算法，完全在 GPU 显存内并行运算。
        """
        num_rays = 90
        max_dist = 5.0

        # 1. 生成 90 条射线方向向量 D: [num_envs, 90, 2]
        # 将 360 度均匀切割，并叠加上机器狗当前的偏航角 (Yaw)
        ray_angles = torch.linspace(0, 2 * math.pi * (num_rays - 1) / num_rays, num_rays, device=self.device)
        global_angles = robot_yaw.unsqueeze(1) + ray_angles.unsqueeze(0)
        D = torch.stack([torch.cos(global_angles), torch.sin(global_angles)], dim=-1)

        # 2. 聚合所有障碍物信息 (静态 + 动态)
        C_s = self.static_obs[..., :2] # 静态坐标 [E, 15, 2]
        R_s = self.static_obs[..., 2]  # 静态半径 [E, 15]
        
        C_d = self.dynamic_obs_pos     # 动态坐标 [E, 5, 2]
        R_d = torch.full((self.num_envs, self.cfg.num_dynamic_obs), self.cfg.dynamic_radius, device=self.device)
        
        C = torch.cat([C_s, C_d], dim=1) # 合并中心坐标 [E, 20, 2]
        R = torch.cat([R_s, R_d], dim=1) # 合并半径 [E, 20]

        # 3. 核心计算：射线方程 (P = O + t*D) 与 圆方程 (||P - C||^2 = R^2) 的联立
        O = robot_pos[:, :2]           # 射线原点 [E, 2]
        V = O.unsqueeze(1) - C         # 原点到圆心的向量 [E, 20, 2]

        # 二次方程系数 a, b, c (因为 D 是单位向量，a = 1)
        # b = V · D (利用批量矩阵乘法 bmm 极速求点积) -> [E, 90, 20]
        b = torch.bmm(D, V.transpose(1, 2))
        
        # c = ||V||^2 - R^2 -> [E, 1, 20]
        c = (torch.sum(V**2, dim=-1) - R**2).unsqueeze(1)

        # 判别式 Δ = b^2 - c
        delta = b**2 - c

        # 求解最小正根 t = -b - sqrt(Δ)
        t = -b - torch.sqrt(torch.clamp(delta, min=0.0))

        # 掩码过滤：必须有交点 (Δ >= 0) 且 交点在前方 (t > 0)
        valid_mask = (delta >= 0) & (t > 0)

        # 将无效交点（未击中障碍物）距离强制设为最大探测距离 5.0m
        t = torch.where(valid_mask, t, max_dist)

        # 获取每条射线在所有 20 个障碍物中的最短命中距离 -> [E, 90]
        min_dist, _ = t.min(dim=-1)

        return torch.clamp(min_dist, min=0.0, max=max_dist)