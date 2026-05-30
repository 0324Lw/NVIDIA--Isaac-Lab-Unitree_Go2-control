# Copyright (c) 2026
# Unitree Go2 Task4: Sim2Real / RMA robust locomotion config.
#
# Strict refactor notes:
# 1. This file contains dataclass configs only.
# 2. task4_env.py imports IsaacLab but must not start AppLauncher.
# 3. Default mode is teacher_mode=True:
#       return_obs = actor_history + privileged_obs
# 4. Actor single obs dim = 48.
# 5. Actor history obs dim = 48 * 5 = 240.
# 6. Privileged obs dim = 25.
# 7. Teacher obs dim = 240 + 25 = 265.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task4Config:
    """Go2 Task4 Sim2Real / RMA robust locomotion config.

    Training route:
        Phase 1:
            teacher_mode=True
            obs = actor_history + privileged_obs
            Train privileged teacher policy.

        Phase 2:
            teacher_mode=False
            obs = actor_history
            Train student / adaptation module later.

    This environment only builds the teacher/student-ready environment.
    The skrl training wrapper will be generated in the training stage.
    """

    # ----------------------------- Basic -----------------------------
    num_envs: int = 2048
    device: str = "cuda:0"

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length_s: float = 20.0

    frame_stack: int = 5
    teacher_mode: bool = True

    # ----------------------------- Dimensions -----------------------------
    single_actor_obs_dim: int = 48
    actor_obs_dim: int = 240
    privileged_obs_dim: int = 25
    teacher_obs_dim: int = 265

    num_actions: int = 12

    # ----------------------------- Scene -----------------------------
    env_spacing: float = 3.0

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 600_000_000

    # stage 0: clean locomotion
    # stage 1: light noise + friction + weak push
    # stage 2: light payload / COM shift
    # stage 3: medium disturbance
    # stage 4: motor degradation
    # stage 5: full robust Sim2Real
    stage_thresholds: Tuple[float, ...] = (0.0, 0.10, 0.24, 0.42, 0.62, 0.80)
    # -1 表示使用正常 curriculum；0~5 可用于 Stage 固定训练/诊断。
    force_stage: int = -1

    cmd_vx_ranges: Tuple[Tuple[float, float], ...] = (
        (0.00, 0.25),
        (0.10, 0.35),
        (0.18, 0.50),
        (0.25, 0.70),
        (0.35, 0.90),
        (0.40, 1.05),
    )

    cmd_vy_ranges: Tuple[Tuple[float, float], ...] = (
        (0.00, 0.00),
        (-0.05, 0.05),
        (-0.08, 0.08),
        (-0.10, 0.10),
        (-0.15, 0.15),
        (-0.20, 0.20),
    )

    cmd_wz_ranges: Tuple[Tuple[float, float], ...] = (
        (-0.15, 0.15),
        (-0.25, 0.25),
        (-0.35, 0.35),
        (-0.45, 0.45),
        (-0.55, 0.55),
        (-0.65, 0.65),
    )

    command_resampling_time_s: float = 6.0

    # ----------------------------- Control -----------------------------
    action_ema_alpha: float = 0.55
    action_scale: float = 0.25
    target_height: float = 0.30

    action_joint_names: Tuple[str, ...] = (
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    )

    foot_body_names: Tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    # ----------------------------- Domain Randomization -----------------------------
    # friction is currently exported as privileged random variable.
    # Real physical material randomization can be added later at scene/material level.
    friction_ranges: Tuple[Tuple[float, float], ...] = (
        (0.80, 1.20),
        (0.70, 1.30),
        (0.60, 1.40),
        (0.45, 1.50),
        (0.35, 1.60),
        (0.25, 1.70),
    )

    payload_mass_ranges: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.5),
        (0.0, 2.5),
        (0.0, 4.0),
        (0.0, 5.0),
    )

    com_shift_ranges: Tuple[Tuple[float, float], ...] = (
        (0.000, 0.000),
        (-0.015, 0.015),
        (-0.030, 0.030),
        (-0.050, 0.050),
        (-0.075, 0.075),
        (-0.100, 0.100),
    )

    motor_strength_ranges: Tuple[Tuple[float, float], ...] = (
        (1.00, 1.00),
        (0.95, 1.00),
        (0.90, 1.00),
        (0.85, 1.00),
        (0.78, 1.00),
        (0.70, 1.00),
    )

    max_degraded_joints_by_stage: Tuple[int, ...] = (0, 0, 1, 1, 2, 2)

    push_interval_ranges_s: Tuple[Tuple[float, float], ...] = (
        (1e6, 1e6),
        (8.0, 12.0),
        (6.0, 10.0),
        (5.0, 9.0),
        (4.0, 8.0),
        (3.0, 7.0),
    )

    push_magnitude_ranges: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (30.0, 80.0),
        (60.0, 120.0),
        (80.0, 180.0),
        (100.0, 240.0),
        (120.0, 300.0),
    )

    push_duration_s: float = 0.10
    post_push_recovery_window_s: float = 1.00

    # Observation noise by stage.
    noise_level_by_stage: Tuple[float, ...] = (0.20, 0.35, 0.50, 0.70, 0.85, 1.00)
    noise_base_ang_vel: float = 0.050
    noise_proj_gravity: float = 0.020
    noise_joint_pos: float = 0.010
    noise_joint_vel: float = 0.050
    noise_base_height: float = 0.010

    # ----------------------------- Safety / termination -----------------------------
    fall_height: float = 0.18
    jump_height: float = 0.65
    bad_orientation_xy: float = 0.85
    max_joint_vel_abs: float = 90.0

    contact_force_threshold: float = 1.0
    foot_slip_force_threshold: float = 5.0
    impact_threshold: float = 220.0

    reset_joint_noise: float = 0.025
    reset_yaw_noise: float = 0.05

    # ----------------------------- Gait -----------------------------
    gait_freq_hz: float = 2.0
    gait_freq_speed_gain: float = 0.25
    duty_factor: float = 0.55
    air_time_target: float = 0.14
    foot_clearance_target: float = 0.075

    # ----------------------------- Reward weights -----------------------------
    # A. Command tracking
    w_cmd_lin: float = 0.85
    w_cmd_yaw: float = 0.20
    w_lateral_vel: float = 0.20
    # Task4-V1.1: 移动命令下的显式前进/欠速约束。
    # 目标是防止“站稳活到 timeout”被误当成速度跟踪成功。
    w_forward_ratio: float = 0.30
    w_under_speed: float = 0.30
    required_speed_ratio: float = 0.50
    move_cmd_threshold: float = 0.08
    cmd_speed_min: float = 0.05
    cmd_lin_speed_gate_ref: float = 0.50
    cmd_lin_speed_gate_floor: float = 0.25
    stability_speed_gate_ref: float = 0.50
    stability_speed_gate_floor: float = 0.40

    # B. Recovery
    w_tracking_recovery: float = 0.30
    w_post_push_stability: float = 0.20
    w_push_survival: float = 0.08

    # C. Stability
    w_upright: float = 0.55
    w_height: float = 0.24
    w_low_height: float = 0.45
    w_base_ang_vel: float = 0.070
    w_z_vel: float = 0.045
    w_alive: float = 0.010

    # D. Gait/contact
    w_phase_contact: float = 0.030
    w_air_time: float = 0.025
    w_clearance: float = 0.025
    w_contact_count: float = 0.030
    w_foot_slip: float = 0.080
    w_impact: float = 0.0035

    # E. Regularization
    w_torque: float = 0.00008
    w_energy: float = 0.00018
    w_action_rate: float = 0.008
    w_action_mag: float = 0.0010
    w_joint_vel: float = 0.0015
    w_joint_limit: float = 0.050

    # Event rewards
    rew_fall: float = -35.0
    rew_timeout_alive: float = 2.0

    # Reward kernels
    sigma_cmd_lin: float = 5.0
    sigma_cmd_yaw: float = 3.0
    sigma_upright: float = 3.0
    sigma_height: float = 35.0
    sigma_clearance: float = 18.0

    continuous_reward_clip: float = 4.0
    episode_return_abs_limit: float = 1500.0

    # ----------------------------- Tracking success metrics -----------------------------
    # 这些阈值只用于日志/事件成功定义，不改变 observation 维度。
    tracking_success_speed_ratio: float = 0.50
    tracking_success_yaw_error: float = 0.25
    tracking_success_min_height: float = 0.24
    tracking_success_max_height: float = 0.36
    tracking_success_max_roll_pitch: float = 0.35

    # ----------------------------- Debug -----------------------------
    print_debug_info: bool = False

    @property
    def policy_dt(self) -> float:
        return float(self.sim_dt * self.decimation)

    @property
    def max_episode_length(self) -> int:
        return int(self.max_episode_length_s / max(self.policy_dt, 1e-6))

    @property
    def num_observations(self) -> int:
        return int(self.teacher_obs_dim if self.teacher_mode else self.actor_obs_dim)

    @property
    def num_privileged_obs(self) -> int:
        return int(self.privileged_obs_dim)