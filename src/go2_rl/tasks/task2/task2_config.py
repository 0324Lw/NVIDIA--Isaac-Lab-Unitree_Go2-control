# Copyright (c) 2026
# Unitree Go2 Task2: multi-terrain / multi-material locomotion config.
#
# Strict refactor notes:
# 1. This file contains Python dataclass configs only.
# 2. task2_world.py imports Task2TerrainCfg from here.
# 3. task2_env.py will import Task2Config from here in the next stage.
# 4. AppLauncher must not be started here.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Task2TerrainCfg:
    """Go2 Task2 terrain world configuration.

    Logical terrain index:
        terrain_type:
            0 rough_flat
            1 slopes
            2 stepping_stones
            3 stairs

        terrain_level:
            0 easiest -> 9 hardest

    Isaac TerrainGenerator layout:
        row = terrain_level
        col = terrain_type
        flat_index = terrain_level * num_terrain_types + terrain_type
    """

    # ----------------------------- Grid -----------------------------
    num_terrain_types: int = 4
    num_levels: int = 10

    terrain_length: float = 8.0
    terrain_width: float = 8.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_width: float = 1.0

    platform_width: float = 2.0
    spawn_radius: float = 0.65
    spawn_height_offset: float = 0.36

    # ----------------------------- Anti-forgetting anchor -----------------------------
    flat_retention_ratio: float = 0.15

    # ----------------------------- Terrain difficulty ranges -----------------------------
    rough_amplitude_min: float = 0.000
    rough_amplitude_max: float = 0.070
    rough_noise_step: float = 0.035

    slope_min: float = 0.02
    slope_max: float = 0.38

    stone_grid_width_min: float = 0.55
    stone_grid_width_max: float = 0.32
    stone_height_min: float = 0.015
    stone_height_max: float = 0.120

    stair_step_width_min: float = 0.42
    stair_step_width_max: float = 0.28
    stair_height_min: float = 0.025
    stair_height_max: float = 0.135

    # ----------------------------- Materials -----------------------------
    friction_range: Tuple[float, float] = (0.35, 1.45)
    restitution_range: Tuple[float, float] = (0.00, 0.12)

    low_friction_range: Tuple[float, float] = (0.35, 0.65)
    normal_friction_range: Tuple[float, float] = (0.70, 1.05)
    high_friction_range: Tuple[float, float] = (1.05, 1.45)

    material_count: int = 3

    # ----------------------------- Height scan -----------------------------
    scan_x_min: float = -0.80
    scan_x_max: float = 1.20
    scan_y_min: float = -0.60
    scan_y_max: float = 0.60
    scan_num_x: int = 9
    scan_num_y: int = 9
    height_scan_clip: float = 2.0

    # ----------------------------- Curriculum thresholds -----------------------------
    success_distance: float = 1.4
    failure_distance: float = 0.45
    downgrade_forgiveness_prob: float = 0.70

    max_level_reset_to_min: int = 4
    max_level_reset_to_max: int = 6

    @property
    def height_scan_dim(self) -> int:
        return int(self.scan_num_x * self.scan_num_y)

    @property
    def terrain_priv_dim(self) -> int:
        # height scan 81 + friction 1 + terrain onehot 4 + difficulty 1 + terrain params 4
        return int(self.height_scan_dim + 1 + self.num_terrain_types + 1 + 4)


@dataclass
class Task2Config:
    """Go2 Task2 environment configuration.

    The environment file will be generated in the next stage. This config is
    already placed here to keep strict project structure stable.
    """

    # ----------------------------- Basic -----------------------------
    num_envs: int = 1024
    device: str = "cuda:0"

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length: int = 1200

    num_observations: int = 87
    num_privileged_obs: int = 178
    num_actions: int = 12

    # ----------------------------- Terrain -----------------------------
    terrain_cfg: Task2TerrainCfg = field(default_factory=Task2TerrainCfg)
    terrain_curriculum_total_steps: int = 600_000_000

    # ----------------------------- Names -----------------------------
    action_joint_names: Tuple[str, ...] = (
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    )

    foot_body_names: Tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    # ----------------------------- Control -----------------------------
    action_ema_alpha: float = 0.55
    hip_action_scale: float = 0.20
    thigh_action_scale: float = 0.35
    calf_action_scale: float = 0.35
    max_joint_vel_abs: float = 75.0

    # ----------------------------- Command curriculum -----------------------------
    resample_command_steps: int = 200
    zero_command_prob: float = 0.06
    command_smoothing: float = 0.12

    cmd_vx_stage0: Tuple[float, float] = (0.0, 0.0)
    cmd_vy_stage0: Tuple[float, float] = (0.0, 0.0)
    cmd_wz_stage0: Tuple[float, float] = (0.0, 0.0)

    cmd_vx_stage1: Tuple[float, float] = (0.05, 0.30)
    cmd_vy_stage1: Tuple[float, float] = (0.0, 0.0)
    cmd_wz_stage1: Tuple[float, float] = (0.0, 0.0)

    cmd_vx_stage2: Tuple[float, float] = (0.12, 0.55)
    cmd_vy_stage2: Tuple[float, float] = (0.0, 0.0)
    cmd_wz_stage2: Tuple[float, float] = (-0.15, 0.15)

    cmd_vx_stage3: Tuple[float, float] = (0.18, 0.80)
    cmd_vy_stage3: Tuple[float, float] = (-0.10, 0.10)
    cmd_wz_stage3: Tuple[float, float] = (-0.25, 0.25)

    cmd_vx_stage4: Tuple[float, float] = (0.15, 1.00)
    cmd_vy_stage4: Tuple[float, float] = (-0.20, 0.20)
    cmd_wz_stage4: Tuple[float, float] = (-0.40, 0.40)

    cmd_vx_stage5: Tuple[float, float] = (-0.15, 1.20)
    cmd_vy_stage5: Tuple[float, float] = (-0.30, 0.30)
    cmd_wz_stage5: Tuple[float, float] = (-0.55, 0.55)

    # ----------------------------- Robot state -----------------------------
    target_height: float = 0.30
    fall_height: float = 0.18
    jump_height: float = 0.60
    bad_orientation_xy: float = 0.80

    # ----------------------------- Gait -----------------------------
    gait_freq_hz: float = 2.0
    contact_force_threshold: float = 1.0
    duty_factor: float = 0.55
    foot_clearance_target: float = 0.075
    air_time_target: float = 0.15

    # ----------------------------- Reward weights -----------------------------
    w_cmd_lin: float = 0.16
    w_cmd_speed: float = 0.38
    w_under_speed: float = 0.28
    w_over_speed: float = 0.10
    w_cmd_yaw: float = 0.08
    w_stand_still: float = 0.020

    w_phase_contact: float = 0.14
    w_air_time: float = 0.12
    w_clearance: float = 0.10
    w_terrain_progress: float = 0.06
    w_double_contact: float = 0.12
    w_foot_slip: float = 0.075

    w_upright: float = 0.09
    w_height: float = 0.09
    w_base_ang_vel: float = 0.035
    w_base_acc: float = 0.001
    w_z_vel: float = 0.018
    w_default_pose: float = 0.018
    w_alive: float = 0.002
    w_joint_limit: float = 0.05
    w_action_rate: float = 0.008
    w_action_mag: float = 0.002
    w_torque: float = 0.00025
    w_energy: float = 0.0008

    penalty_fall: float = -5.0

    sigma_cmd_lin: float = 3.0
    sigma_cmd_yaw: float = 3.5
    sigma_stand: float = 8.0
    sigma_height: float = 30.0

    continuous_reward_clip: float = 1.0
    episode_return_abs_limit: float = 1000.0

    print_debug_info: bool = False

    @property
    def control_dt(self) -> float:
        return float(self.sim_dt * self.decimation)
