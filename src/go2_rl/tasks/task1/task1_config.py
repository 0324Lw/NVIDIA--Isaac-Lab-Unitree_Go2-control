# Copyright (c) 2026
# Unitree Go2 Task1: flat-ground locomotion config.
#
# Strict refactor notes:
# 1. This file contains Python dataclass config only.
# 2. YAML configs under configs/ are public-facing references.
# 3. Env / test / train should import Task1Config from this file.
# 4. AppLauncher must not be started here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task1Config:
    # ----------------------------- Basic -----------------------------
    num_envs: int = 1024
    device: str = "cuda:0"

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length: int = 1000
    env_spacing: float = 2.5

    # Single-frame actor observation. Training wrapper performs frame stack.
    num_observations: int = 87
    num_privileged_obs: int = 0
    num_actions: int = 12

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
    max_joint_vel_abs: float = 65.0

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 350_000_000
    resample_command_steps: int = 200

    # Stage 0: stand
    cmd_vx_stage0: Tuple[float, float] = (0.0, 0.0)
    cmd_vy_stage0: Tuple[float, float] = (0.0, 0.0)
    cmd_wz_stage0: Tuple[float, float] = (0.0, 0.0)

    # Stage 1: tiny forward
    cmd_vx_stage1: Tuple[float, float] = (0.00, 0.18)
    cmd_vy_stage1: Tuple[float, float] = (0.0, 0.0)
    cmd_wz_stage1: Tuple[float, float] = (0.0, 0.0)

    # Stage 2: slow forward
    cmd_vx_stage2: Tuple[float, float] = (0.10, 0.45)
    cmd_vy_stage2: Tuple[float, float] = (0.0, 0.0)
    cmd_wz_stage2: Tuple[float, float] = (0.0, 0.0)

    # Stage 3: forward + yaw
    cmd_vx_stage3: Tuple[float, float] = (0.15, 0.75)
    cmd_vy_stage3: Tuple[float, float] = (0.0, 0.0)
    cmd_wz_stage3: Tuple[float, float] = (-0.25, 0.25)

    # Stage 4: omni low-mid speed
    cmd_vx_stage4: Tuple[float, float] = (-0.15, 1.00)
    cmd_vy_stage4: Tuple[float, float] = (-0.25, 0.25)
    cmd_wz_stage4: Tuple[float, float] = (-0.45, 0.45)

    # Stage 5: full command range
    cmd_vx_stage5: Tuple[float, float] = (-0.25, 1.20)
    cmd_vy_stage5: Tuple[float, float] = (-0.35, 0.35)
    cmd_wz_stage5: Tuple[float, float] = (-0.60, 0.60)

    zero_command_prob: float = 0.08
    command_smoothing: float = 0.12

    # ----------------------------- Robot state -----------------------------
    target_height: float = 0.32
    fall_height: float = 0.18
    jump_height: float = 0.55
    bad_orientation_xy: float = 0.75

    # ----------------------------- Gait -----------------------------
    gait_freq_hz: float = 2.0
    contact_force_threshold: float = 1.0
    duty_factor: float = 0.55
    foot_clearance_target: float = 0.075
    air_time_target: float = 0.16

    # ----------------------------- Reward weights -----------------------------
    # 60% command / locomotion / gait
    w_cmd_lin: float = 0.18
    w_cmd_yaw: float = 0.08
    w_cmd_speed: float = 0.35
    w_under_speed: float = 0.22
    w_double_contact: float = 0.10
    w_stand_still: float = 0.03
    w_phase_contact: float = 0.13
    w_air_time: float = 0.10
    w_clearance: float = 0.09

    # 25% stability
    w_upright: float = 0.09
    w_height: float = 0.09
    w_base_ang_vel: float = 0.035
    w_base_acc: float = 0.001
    w_z_vel: float = 0.015

    # 15% safety / efficiency
    w_default_pose: float = 0.025
    w_alive: float = 0.002
    w_joint_limit: float = 0.05
    w_action_rate: float = 0.008
    w_action_mag: float = 0.002
    w_foot_slip: float = 0.06
    w_torque: float = 0.00025
    w_energy: float = 0.0008

    penalty_fall: float = -5.0

    # Reward kernels
    sigma_cmd_lin: float = 4.0
    sigma_cmd_yaw: float = 4.0
    sigma_stand: float = 8.0
    sigma_height: float = 30.0

    continuous_reward_clip: float = 1.0
    episode_return_abs_limit: float = 1000.0

    # Debug
    debug_print_names: bool = False

    @property
    def control_dt(self) -> float:
        return float(self.sim_dt * self.decimation)
