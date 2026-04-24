import torch
import numpy as np
from typing import Tuple, Dict

# 导入 Isaac Lab 原生地形生成库
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

# ===================================================================
# 1. 地形与世界参数配置类 (Config)
# ===================================================================
@configclass
class Task2TerrainCfg:
    """
    Task 2 复杂地形世界配置类。
    网格结构：X 轴（行）代表不同的地形类型，Y 轴（列）代表难度等级（从左到右变难）。
    """
    # 基础网格设定
    num_rows = 4           
    num_cols = 10          
    terrain_length = 8.0   
    terrain_width = 8.0    
    
    flat_retention_ratio = 0.15  
    
    friction_range = [0.4, 1.2]  

    rough_amplitude_range = [0.0, 0.06]   # 粗糙地面上限：6cm 
    slope_pitch_range = [0.05, 0.35]      # 倾斜斜坡上限：约20度 
    stepping_stones_height = [0.02, 0.10] # 乱石高差上限：10cm 
    stairs_height_range = [0.05, 0.12]    # 台阶高度上限：12cm 
# ===================================================================
# 2. 地形生成与世界管理类 (World Environment Model)
# ===================================================================
class Task2World:
    """
    世界地形生成与管理器。
    负责调用 Isaac Lab 底层 API 渲染地形网格，并提供各区块中心坐标的张量映射。
    """
    def __init__(self, cfg: Task2TerrainCfg, device: str):
        self.cfg = cfg
        self.device = device
        
        self.terrain_sub_cfgs = {
            "rough_flat": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25, noise_range=self.cfg.rough_amplitude_range, noise_step=0.02
            ),
            "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.25, slope_range=self.cfg.slope_pitch_range, platform_width=2.0
            ),
            "stepping_stones": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.25, grid_width=0.45, grid_height_range=self.cfg.stepping_stones_height, platform_width=2.0
            ),
            "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.25, step_height_range=self.cfg.stairs_height_range, step_width=0.35, platform_width=2.0
            )
        }

        self.generator_cfg = TerrainGeneratorCfg(
            size=(self.cfg.terrain_length, self.cfg.terrain_width),
            num_rows=self.cfg.num_rows,
            num_cols=self.cfg.num_cols,
            sub_terrains=self.terrain_sub_cfgs, 
            use_cache=False,  
            color_scheme="height" 
        )

        self._build_origin_mapping()

    def _build_origin_mapping(self):
        """预计算并缓存所有地形区块的中心坐标张量"""
        origins = torch.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), device=self.device)
        start_x = - (self.cfg.num_rows * self.cfg.terrain_length) / 2.0 + (self.cfg.terrain_length / 2.0)
        start_y = - (self.cfg.num_cols * self.cfg.terrain_width) / 2.0 + (self.cfg.terrain_width / 2.0)

        for r in range(self.cfg.num_rows):
            for c in range(self.cfg.num_cols):
                origins[r, c, 0] = start_x + r * self.cfg.terrain_length
                origins[r, c, 1] = start_y + c * self.cfg.terrain_width
                origins[r, c, 2] = 0.0 
                
        self.terrain_origins = origins

    def get_origins_from_indices(self, env_rows: torch.Tensor, env_cols: torch.Tensor) -> torch.Tensor:
        return self.terrain_origins[env_rows, env_cols].clone()

# ===================================================================
# 3. 动态课程学习管理器 (带有防遗忘驻留机制与探针)
# ===================================================================
class TerrainCurriculum:
    """
    纯张量化动态难度控制器。
    """
    def __init__(self, num_envs: int, world_cfg: Task2TerrainCfg, device: str):
        self.num_envs = num_envs
        self.cfg = world_cfg
        self.device = device
        
        self.env_types = torch.randint(0, self.cfg.num_rows, (self.num_envs,), device=self.device)
        self.env_levels = torch.randint(0, 3, (self.num_envs,), device=self.device)
        
        # 分配平地锚点
        num_anchors = int(self.num_envs * self.cfg.flat_retention_ratio)
        self.anchor_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.anchor_mask[:num_anchors] = True
        
        # 强制将锚点组的初始等级锁定在 0 (平地)
        self.env_levels[self.anchor_mask] = 0
        
        self.env_start_pos_x = torch.zeros(self.num_envs, device=self.device)

        # 🔍【植入测试探针】：用于记录课程变动，供外部测试脚本调用
        self.probe_upgrades_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.probe_downgrades_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.probe_max_level_reached = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def register_start_positions(self, env_ids, pos_x):
        self.env_start_pos_x[env_ids] = pos_x

    def update_curriculum(self, env_ids: torch.Tensor, current_pos_x: torch.Tensor, fall_flags: torch.Tensor):
        if len(env_ids) == 0:
            return
            
        start_x = self.env_start_pos_x[env_ids]
        curr_x = current_pos_x[env_ids]
        felled = fall_flags[env_ids]
        levels = self.env_levels[env_ids]
        anchors = self.anchor_mask[env_ids]
        
        distance_walked = curr_x - start_x
        
        # 晋升条件：单回合行走距离过半，且没有跌倒
        move_up_mask = (distance_walked > 3.0) & (~felled)
        
        # 降级条件：没走两步就摔倒了
        # 只有当极其无能（走不到 0.5 米就暴毙）时，才触发降级判定
        is_terrible_fail = (distance_walked < 0.5) & fall_flags
        
        # 引入 50% 的“宽恕概率” (Forgiveness Rate)
        # 即使它摔得很惨，也有 50% 的几率保住当前的等级，让它下回合原地复活继续尝试
        chance_to_forgive = torch.rand(len(env_ids), device=self.device) > 0.5
        
        # 只有摔得很惨，且没有被宽恕的狗，才会降级
        move_down_mask = is_terrible_fail & chance_to_forgive
        
        # 处于锚点组的机器狗，绝对不允许晋升
        move_up_mask &= (~anchors)
        
        # 应用掩码计算新等级
        levels = torch.where(move_up_mask, levels + 1, levels)
        levels = torch.where(move_down_mask, levels - 1, levels)
        
        # 边界截断防越界
        levels = torch.clamp(levels, min=0, max=self.cfg.num_cols - 1)
        
        # 将通关(满级)的非锚点狗随机打乱到其他地形的 5 级，保持持续训练压力
        max_level_mask = (levels == (self.cfg.num_cols - 1)) & move_up_mask
        if max_level_mask.any():
            self.env_types[env_ids[max_level_mask]] = torch.randint(0, self.cfg.num_rows, (max_level_mask.sum().item(),), device=self.device)
            levels[max_level_mask] = 5 
        
        # 更新探针数据
        self.probe_upgrades_count[env_ids] += move_up_mask.long()
        self.probe_downgrades_count[env_ids] += move_down_mask.long()
        self.probe_max_level_reached[env_ids] = torch.maximum(self.probe_max_level_reached[env_ids], levels)
        
        # 写回
        self.env_levels[env_ids] = levels

    def get_current_grid_indices(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.env_types[env_ids], self.env_levels[env_ids]

    def log_curriculum_stats(self) -> dict:
        # 获取非锚点组 (正在攀登困难地形的狗) 的平均水平
        active_dogs_levels = self.env_levels[~self.anchor_mask].float()
        mean_active_level = active_dogs_levels.mean().item() if len(active_dogs_levels) > 0 else 0.0
        
        return {
            "Curriculum/Mean_Level_Active": mean_active_level,
            "Curriculum/Max_Level_Reached": self.env_levels.max().item(),
            "Curriculum/Anchor_Dogs_Count": self.anchor_mask.sum().item()
        }