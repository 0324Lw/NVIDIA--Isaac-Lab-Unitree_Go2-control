# Copyright (c) 2026
# Unitree Go2 Task2: multi-terrain / multi-material locomotion config.
#
# Strict refactor notes:
# 1. This file contains Python dataclass configs only.
# 2. task2_world.py imports Task2TerrainCfg from here.
# 3. task2_env.py imports Task2Config from here.
# 4. AppLauncher must not be started here.
# 5. Reward-V2.1 no longer uses legacy w_* reward weights.

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
    """Go2 Task2 environment configuration."""

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
    zero_command_prob: float = 0.02
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

    # ----------------------------- Reward-V2.1 group weights -----------------------------
    reward_group_cmd: float = 0.66
    reward_group_locomotion: float = 0.12
    reward_group_contact: float = 0.08
    reward_group_stability: float = 0.10
    reward_group_control: float = 0.04

    # ----------------------------- Reward-V2.1 command group -----------------------------
    reward_cmd_w_lin_gated: float = 0.08
    reward_cmd_w_speed: float = 0.34
    reward_cmd_w_yaw_gated: float = 0.08
    reward_cmd_w_under: float = 0.45
    reward_cmd_w_over: float = 0.03
    reward_cmd_w_reverse: float = 0.02

    reward_cmd_lin_sigma: float = 6.0
    reward_cmd_yaw_sigma: float = 3.5
    reward_cmd_speed_sigma: float = 4.0
    reward_cmd_speed_error_clip: float = 2.0

    reward_cmd_lin_move_threshold: float = 0.05
    reward_cmd_yaw_move_threshold: float = 0.08
    reward_target_speed_min: float = 0.08

    reward_speed_ratio_min: float = -2.0
    reward_speed_ratio_max: float = 2.0
    reward_under_ratio_max: float = 1.5
    reward_over_speed_ratio: float = 1.35
    reward_over_ratio_max: float = 1.5
    reward_reverse_ratio_max: float = 1.5
    reward_yaw_gate_min: float = 0.20

    # ----------------------------- Reward-V2.1 locomotion quality group -----------------------------
    reward_loco_w_air_time: float = 0.55
    reward_loco_w_clearance: float = 0.35
    reward_loco_w_phase: float = 0.10

    reward_air_time_clip: float = 0.5
    reward_clearance_sigma: float = 20.0

    # ----------------------------- Reward-V2.1 contact quality group -----------------------------
    reward_contact_w_foot_slip: float = 0.35
    reward_contact_w_double_contact: float = 0.65

    reward_contact_scale_k1: float = 0.25
    reward_contact_scale_k2: float = 0.60
    reward_contact_scale_early: float = 0.35
    reward_contact_scale_middle: float = 0.65
    reward_contact_scale_late: float = 1.00

    reward_double_contact_threshold: float = 2.4
    reward_foot_slip_clip: float = 8.0

    reward_low_speed_static_threshold: float = 0.50
    reward_low_speed_static_gain: float = 1.50

    # ----------------------------- Reward-V2.1 stability quality group -----------------------------
    reward_stability_w_upright: float = 0.28
    reward_stability_w_height: float = 0.28
    reward_stability_w_base_ang: float = 0.18
    reward_stability_w_z_vel: float = 0.12
    reward_stability_w_base_acc: float = 0.07
    reward_stability_w_stand: float = 0.07

    reward_height_sigma: float = 30.0
    reward_stand_sigma: float = 8.0
    reward_base_ang_clip: float = 6.0
    reward_base_acc_clip: float = 30.0

    # ----------------------------- Reward-V2.1 control regularization group -----------------------------
    reward_control_w_action_rate: float = 0.25
    reward_control_w_action_mag: float = 0.10
    reward_control_w_torque: float = 0.20
    reward_control_w_energy: float = 0.15
    reward_control_w_joint_limit: float = 0.20
    reward_control_w_default_pose: float = 0.10

    reward_joint_limit_margin: float = 0.04
    reward_torque_clip: float = 40.0
    reward_energy_clip: float = 20.0

    # ----------------------------- Reward event / clipping -----------------------------
    penalty_fall: float = -5.0
    continuous_reward_clip: float = 1.2
    episode_return_abs_limit: float = 1000.0

    print_debug_info: bool = False

    @property
    def control_dt(self) -> float:
        return float(self.sim_dt * self.decimation)