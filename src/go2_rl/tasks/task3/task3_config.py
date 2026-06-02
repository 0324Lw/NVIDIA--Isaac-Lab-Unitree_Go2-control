# Copyright (c) 2026
# Unitree Go2 Task3: 导航避障任务配置。
#
# 本文件只定义 Task3 配置参数，不启动 IsaacLab AppLauncher，也不创建环境实例。
# 配置覆盖仿真参数、机器人控制参数、导航世界参数、观测维度、动作维度、课程参数和奖励权重。
#
# Gymnasium API:
#   环境入口位于 task3_env.py
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# 观测维度:
#   actor obs = 208
#   privileged obs = 276
#   world privileged tail = 68
#   lidar rays = 60
#   action dim = 12
#
# 训练入口位于 task3_train.py，模型评估入口位于 task3_model_test.py。
#
# 工程说明:
#   Task3 在多地形运动基础上加入目标导航、静态/动态障碍物、解析 lidar 和分阶段课程。
#   privileged obs 由 actor obs 和 world privileged tail 拼接得到。
#   world privileged tail 只供 asymmetric critic 使用，不进入 policy observation。
#
# Unitree Go2 Task3: navigation and obstacle-avoidance task configuration.
#
# This file only defines Task3 configuration parameters. It does not launch
# IsaacLab AppLauncher or create environment instances. The configuration covers
# simulation parameters, robot-control parameters, navigation-world parameters,
# observation dimensions, action dimensions, curriculum settings, and reward weights.
#
# Gymnasium API:
#   Environment entry is task3_env.py
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# Observation dimensions:
#   actor obs = 208
#   privileged obs = 276
#   world privileged tail = 68
#   lidar rays = 60
#   action dim = 12
#
# Training entry is task3_train.py, and model evaluation entry is task3_model_test.py.
#
# Engineering notes:
#   Task3 adds goal navigation, static/dynamic obstacles, analytical lidar, and
#   staged curriculum on top of locomotion. privileged obs is formed by
#   concatenating actor obs and the world privileged tail. The world privileged
#   tail is used only by the asymmetric critic and is not part of the policy observation.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Task3WorldCfg:
    """Go2 Task3-V3.2 解析导航世界配置。"""

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

    # Stage 0 focuses on short-range target entry. Stage 5 remains a rare
    # stress probe / future evaluation stage to avoid changing task3_world.py.
    success_radius: float = 0.90
    success_radius_by_stage: Tuple[float, ...] = (
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.68,
    )

    # ----------------------------- Robot / collision -----------------------------
    robot_radius: float = 0.35
    collision_margin: float = 0.03
    warning_margin: float = 0.55

    # ----------------------------- Lidar -----------------------------
    num_lidar_rays: int = 60
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
    curriculum_total_steps: int = 900_000_000

    stage_thresholds: Tuple[float, ...] = (
        0.0,
        0.18,
        0.38,
        0.62,
        0.84,
        0.96,
    )

    goal_dist_ranges: Tuple[Tuple[float, float], ...] = (
        (1.5, 3.0),
        (2.5, 5.0),
        (4.0, 7.0),
        (5.0, 9.0),
        (7.0, 12.0),
        (8.0, 14.0),
    )

    target_speed_ranges: Tuple[Tuple[float, float], ...] = (
        (0.25, 0.45),
        (0.30, 0.50),
        (0.35, 0.65),
        (0.45, 0.75),
        (0.55, 0.90),
        (0.60, 1.00),
    )

    static_count_ranges: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (1, 2),
        (2, 5),
        (4, 8),
        (6, 10),
        (8, 12),
    )

    dynamic_count_ranges: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (0, 0),
        (0, 0),
        (1, 2),
        (2, 3),
        (2, 4),
    )

    dynamic_speed_ranges: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.15, 0.45),
        (0.25, 0.65),
        (0.30, 0.80),
    )

    # ----------------------------- Performance-gated curriculum -----------------------------
    use_performance_gated_curriculum: bool = True
    curriculum_resume_k_floor: float = 0.0

    curriculum_current_stage_ratio: float = 0.70
    curriculum_prev_stage_ratio: float = 0.20
    curriculum_next_stage_ratio: float = 0.10

    curriculum_stage0_current_ratio: float = 0.90
    curriculum_stage0_next_ratio: float = 0.10
    curriculum_max_stage_current_ratio: float = 0.80
    curriculum_max_stage_prev_ratio: float = 0.20

    curriculum_min_stage_steps: int = 25_000_000
    curriculum_check_interval_steps: int = 2_000_000
    curriculum_min_window_done: int = 256

    curriculum_success_gate: Tuple[float, ...] = (
        0.20,
        0.15,
        0.10,
        0.06,
        0.03,
    )
    curriculum_reduction_gate: Tuple[float, ...] = (
        0.60,
        0.55,
        0.45,
        0.35,
        0.30,
    )
    curriculum_heading_gate: Tuple[float, ...] = (
        0.80,
        0.75,
        0.65,
        0.55,
        0.50,
    )
    curriculum_fall_gate: Tuple[float, ...] = (
        0.03,
        0.035,
        0.04,
        0.05,
        0.06,
    )
    curriculum_collision_gate: Tuple[float, ...] = (
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
    )
    curriculum_timeout_final_dist_gate: Tuple[float, ...] = (
        1.20,
        1.60,
        2.30,
        3.20,
        4.50,
    )

    # ----------------------------- Event rewards -----------------------------
    rew_success: float = 30.0
    rew_collision: float = -12.0
    rew_fall: float = -7.5
    rew_timeout: float = -2.0
    rew_out_of_bounds: float = -8.0

    # ----------------------------- Privileged features -----------------------------
    privileged_static_k: int = 6
    privileged_dynamic_k: int = 4

    @property
    def max_episode_steps(self) -> int:
        return int(self.max_episode_length_s * self.rl_policy_freq)


@dataclass
class Task3Config:
    """Go2 Task3-Navigation-V3.2 IsaacLab environment config."""

    # ----------------------------- Basic -----------------------------
    num_envs: int = 1024
    device: str = "cuda:0"

    sim_dt: float = 0.005
    decimation: int = 4

    world_cfg: Task3WorldCfg = field(default_factory=Task3WorldCfg)

    # Actor obs: proprioception 62 + navigation 19 + obstacle summary 7 + lidar 60 + lidar_delta 60 = 208
    num_observations: int = 208

    # Critic obs = actor obs 208 + existing Task3World privileged features 68
    num_privileged_obs: int = 276
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

    init_yaw_noise_stage0: float = 0.10
    init_yaw_noise_stage5: float = 1.20

    # ----------------------------- Gait -----------------------------
    gait_freq_hz: float = 2.0
    gait_freq_speed_gain: float = 0.35
    duty_factor: float = 0.55
    contact_force_threshold: float = 1.0

    foot_clearance_target: float = 0.080
    air_time_target: float = 0.14

    # ----------------------------- Reward weights: navigation task -----------------------------
    w_progress_step: float = 1.35
    w_distance_reduction: float = 0.45
    w_goal_heading: float = 0.12
    w_goal_speed: float = 0.30
    w_backtrack: float = 0.60
    w_stuck: float = 0.225
    w_stuck_recovery: float = 0.12

    # ----------------------------- Reward weights: finish / terminal shaping -----------------------------
    w_near_goal: float = 0.176
    w_finish_pull: float = 1.43
    w_finish_hesitation: float = 0.715
    w_timeout_final_dist: float = 4.0

    # ----------------------------- Reward weights: obstacle / safety -----------------------------
    w_clearance_moving: float = 0.045
    w_obstacle_risk: float = 0.24
    w_ttc_proxy: float = 0.12
    w_boundary: float = 0.05

    # ----------------------------- Reward weights: locomotion stability -----------------------------
    # Final stability tune: slightly increase roll/pitch damping and foot-slip penalty.
    w_upright: float = 0.02975
    w_height: float = 0.0255
    w_base_ang_vel: float = 0.0345
    w_base_acc: float = 0.0007
    w_z_vel: float = 0.012
    w_phase_contact: float = 0.030
    w_air_time: float = 0.025
    w_clearance: float = 0.025
    w_contact_count: float = 0.025
    w_foot_slip: float = 0.06325

    # ----------------------------- Reward weights: control regularization -----------------------------
    w_default_pose: float = 0.008
    w_alive: float = 0.0005
    w_joint_limit: float = 0.030
    w_action_rate: float = 0.0050
    w_action_mag: float = 0.0010
    w_torque: float = 0.00010
    w_energy: float = 0.00014
    w_specific_energy: float = 0.0025

    # ----------------------------- Reward kernels -----------------------------
    sigma_heading: float = 2.2
    sigma_speed: float = 2.0
    sigma_height: float = 32.0
    sigma_clearance: float = 18.0

    # ----------------------------- Reward-V3.2 moving / progress gates -----------------------------
    # Final stability tune: progress step is softened to reduce clip saturation.
    reward_heading_speed_gate_ref: float = 0.20
    reward_progress_gate_ref: float = 0.002
    reward_progress_step_ref: float = 0.004
    reward_forward_gate_ref: float = 0.25
    reward_heading_floor: float = 0.15

    # ----------------------------- Reward-V3.2 speed shaping -----------------------------
    reward_forward_ratio_weight_in_speed: float = 0.75
    reward_speed_gaussian_weight_in_speed: float = 0.25

    # ----------------------------- Reward-V3.2 stuck shaping -----------------------------
    stuck_free_margin: float = 0.20

    progress_clip_neg: float = 1.40
    progress_clip_pos: float = 1.80
    progress_ema_alpha: float = 0.10
    progress_obs_scale: float = 2.0

    near_goal_radius_scale: float = 2.0
    near_goal_min_speed_ratio: float = 0.30
    min_finish_progress: float = 0.02

    moving_speed_threshold: float = 0.05
    stuck_progress_threshold: float = 0.015
    stuck_counter_limit: int = 120

    safe_obstacle_distance: float = 0.85
    critical_obstacle_distance: float = 0.35

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