# Copyright (c) 2026
# Unitree Go2 Task3: autonomous navigation / obstacle avoidance config.
#
# Strict refactor notes:
# 1. Task3WorldCfg is pure torch analytical world config.
# 2. Task3Config is IsaacLab Go2 environment config.
# 3. task3_world.py must not import IsaacLab.
# 4. task3_env.py imports IsaacLab but must not start AppLauncher.
# 5. Actor single obs dim = 257.
# 6. Critic single obs dim = 325 = actor obs 257 + world privileged 68.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Task3WorldCfg:
    """Analytical navigation world config for Go2 Task3."""

    # ----------------------------- Frequency -----------------------------
    pd_control_freq: float = 200.0
    rl_policy_freq: float = 50.0
    decimation: int = 4
    policy_dt: float = 0.02

    # ----------------------------- Map -----------------------------
    env_size: float = 30.0
    wall_margin: float = 1.5
    safe_zone_radius: float = 2.0

    # ----------------------------- Episode -----------------------------
    max_episode_length_s: float = 30.0

    # Stage0 is intentionally easier to bootstrap.
    success_radius: float = 0.90
    success_radius_by_stage: Tuple[float, ...] = (
        0.90,
        0.82,
        0.76,
        0.70,
        0.65,
        0.60,
    )

    # ----------------------------- Robot / collision -----------------------------
    robot_radius: float = 0.35
    collision_margin: float = 0.03
    warning_margin: float = 0.55

    # ----------------------------- Lidar -----------------------------
    num_lidar_rays: int = 90
    lidar_max_distance: float = 6.0
    lidar_noise_std: float = 0.0

    # ----------------------------- Obstacles -----------------------------
    max_static_obs: int = 25
    max_dynamic_obs: int = 8

    static_radius_range: Tuple[float, float] = (0.30, 1.00)
    dynamic_radius_range: Tuple[float, float] = (0.28, 0.52)

    min_static_spacing: float = 0.40
    min_dynamic_spacing: float = 0.25
    obstacle_spawn_buffer: float = 0.30
    max_rejection_iters: int = 96

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 800_000_000

    stage_thresholds: Tuple[float, ...] = (
        0.0,
        0.08,
        0.20,
        0.36,
        0.56,
        0.78,
    )

    goal_dist_ranges: Tuple[Tuple[float, float], ...] = (
        (1.6, 3.2),
        (3.0, 5.5),
        (4.5, 8.0),
        (6.5, 11.0),
        (8.5, 16.0),
        (10.0, 22.0),
    )

    static_count_ranges: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (1, 3),
        (3, 6),
        (6, 10),
        (10, 16),
        (14, 22),
    )

    dynamic_count_ranges: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (0, 0),
        (1, 2),
        (2, 4),
        (4, 6),
        (5, 8),
    )

    dynamic_speed_ranges: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.15, 0.40),
        (0.30, 0.70),
        (0.45, 1.00),
        (0.60, 1.30),
    )

    target_speed_ranges: Tuple[Tuple[float, float], ...] = (
        (0.30, 0.55),
        (0.45, 0.75),
        (0.65, 1.00),
        (0.85, 1.30),
        (1.00, 1.60),
        (1.20, 1.80),
    )

    # ----------------------------- Event rewards -----------------------------
    rew_success: float = 30.0
    rew_collision: float = -16.0
    rew_fall: float = -10.0
    rew_timeout: float = -6.0

    # ----------------------------- Privileged features -----------------------------
    privileged_static_k: int = 6
    privileged_dynamic_k: int = 4

    @property
    def max_episode_steps(self) -> int:
        return int(self.max_episode_length_s * self.rl_policy_freq)


@dataclass
class Task3Config:
    """Go2 Task3 IsaacLab environment config."""

    # ----------------------------- Basic -----------------------------
    num_envs: int = 1024
    device: str = "cuda:0"

    sim_dt: float = 0.005
    decimation: int = 4

    world_cfg: Task3WorldCfg = field(default_factory=Task3WorldCfg)

    # Actor obs single frame:
    # base_lin_vel_b 3
    # base_ang_vel_b 3
    # projected_gravity_b 3
    # target_obs 3
    # target_speed 1
    # progress_ema 1
    # joint_pos_error 12
    # joint_vel 12
    # last_action 12
    # action_delta 12
    # foot_contact 4
    # lidar 90
    # lidar_delta 90
    # risk_features 8
    # base_height 1
    # phase sin/cos 2
    num_observations: int = 257

    # Critic obs = actor_obs 257 + world privileged 68
    num_privileged_obs: int = 325
    num_actions: int = 12

    # ----------------------------- Scene -----------------------------
    env_spacing: float = 35.0

    # ----------------------------- Names -----------------------------
    action_joint_names: Tuple[str, ...] = (
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    )

    foot_body_names: Tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    # ----------------------------- Control -----------------------------
    action_ema_alpha: float = 0.50
    hip_action_scale: float = 0.20
    thigh_action_scale: float = 0.35
    calf_action_scale: float = 0.35
    max_joint_vel_abs: float = 85.0

    # ----------------------------- Reset / safety -----------------------------
    target_height: float = 0.30
    fall_height: float = 0.17
    jump_height: float = 0.65
    bad_orientation_xy: float = 0.85
    reset_joint_noise: float = 0.025

    init_yaw_noise_stage0: float = 0.12
    init_yaw_noise_stage5: float = math.pi

    # ----------------------------- Gait -----------------------------
    gait_freq_hz: float = 2.0
    gait_freq_speed_gain: float = 0.35
    duty_factor: float = 0.55
    contact_force_threshold: float = 1.0

    foot_clearance_target: float = 0.080
    air_time_target: float = 0.14

    # ----------------------------- Reward weights: navigation -----------------------------
    w_progress: float = 0.62
    w_goal_speed: float = 0.22
    w_goal_heading: float = 0.045
    w_goal_distance: float = 0.012

    w_finish_pull: float = 0.68
    w_finish_hesitation: float = 0.34
    w_under_speed: float = 0.24
    w_backtrack: float = 0.32
    w_deadline: float = 0.06

    # ----------------------------- Reward weights: obstacle / safety -----------------------------
    w_obstacle_risk: float = 0.26
    w_front_clearance: float = 0.075
    w_ttc_proxy: float = 0.14
    w_active_avoid_heading: float = 0.08
    w_boundary: float = 0.045

    # ----------------------------- Reward weights: running / gait -----------------------------
    w_phase_contact: float = 0.045
    w_air_time: float = 0.040
    w_clearance: float = 0.040
    w_contact_count: float = 0.035
    w_foot_slip: float = 0.060

    # ----------------------------- Reward weights: stability / regularization -----------------------------
    w_upright: float = 0.045
    w_height: float = 0.038
    w_base_ang_vel: float = 0.035
    w_base_acc: float = 0.0008
    w_z_vel: float = 0.014
    w_default_pose: float = 0.010
    w_alive: float = 0.0005
    w_joint_limit: float = 0.035
    w_action_rate: float = 0.0055
    w_action_mag: float = 0.0010
    w_torque: float = 0.00010
    w_energy: float = 0.00016
    w_specific_energy: float = 0.0035

    # ----------------------------- Reward kernels -----------------------------
    sigma_heading: float = 1.8
    sigma_speed: float = 2.0
    sigma_goal_distance: float = 0.18
    sigma_height: float = 32.0
    sigma_clearance: float = 18.0

    # Progress shaping
    progress_scale: float = 1.65
    progress_clip_neg: float = 1.60
    progress_clip_pos: float = 2.20

    min_forward_speed: float = 0.20
    min_progress_speed: float = 0.14

    # Success zone
    finish_outer_radius_scale: float = 1.75
    finish_inner_radius_scale: float = 1.05
    hesitation_speed: float = 0.24

    # Deadline pressure
    deadline_start_frac: float = 0.45

    # Obstacle safety
    safe_obstacle_distance: float = 0.85
    critical_obstacle_distance: float = 0.35

    # Continuous reward safety
    continuous_reward_clip: float = 1.50
    episode_return_abs_limit: float = 1200.0

    print_debug_info: bool = False

    @property
    def control_dt(self) -> float:
        return float(self.sim_dt * self.decimation)

    @property
    def max_episode_length(self) -> int:
        return int(self.world_cfg.max_episode_steps)

    @property
    def curriculum_total_steps(self) -> int:
        return int(self.world_cfg.curriculum_total_steps)
