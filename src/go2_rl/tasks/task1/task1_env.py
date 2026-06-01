# Copyright (c) 2026
# Unitree Go2 Task1: 平地运动 IsaacLab 环境。
#
# 本文件只定义 Task1 IsaacLab 环境，不启动 AppLauncher。
# 环境采用 Gymnasium step API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# 观测维度:
#   actor obs = 87
#   privileged obs = 0
#   action dim = 12
#
# 训练入口位于 task1_train.py，模型评估入口位于 task1_model_test.py。
#
# 工程说明:
#   Task1 使用平地场景和 Unitree Go2 articulation。
#   env_origins 用于并行环境的局部坐标基准，使 base height、foot height 和 reset root pose
#   都相对于各自环境原点计算，避免不同 env 网格位置影响观测和奖励。
#   info 中保留 GPU tensor，低频日志阶段再转换为标量，以减少 step 内 CPU 同步。
#
# Unitree Go2 Task1: flat locomotion IsaacLab environment.
#
# This file only defines the Task1 IsaacLab environment and does not launch AppLauncher.
# The environment follows the Gymnasium step API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# Observation dimensions:
#   actor obs = 87
#   privileged obs = 0
#   action dim = 12
#
# Training entry is task1_train.py, and model evaluation entry is task1_model_test.py.
#
# Engineering notes:
#   Task1 uses a flat-ground scene and the Unitree Go2 articulation.
#   env_origins provides local coordinate references for parallel environments,
#   so base height, foot height, and reset root pose are computed relative to
#   each environment origin rather than affected by grid placement.
#   info keeps GPU tensors and converts them to scalars only during low-frequency
#   logging to reduce CPU synchronization inside step.

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

try:
    from isaaclab_assets import UNITREE_GO2_CFG
except Exception:
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from go2_rl.tasks.task1.task1_config import Task1Config


@configclass
class Go2Task1SceneCfg(InteractiveSceneCfg):
    """IsaacLab scene config for Unitree Go2 flat-ground Task1."""

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        debug_vis=False,
    )

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0),
    )


class Go2Task1Env(gym.Env):
    """Unitree Go2 flat-ground locomotion environment.

    Gymnasium step API:
        reset() -> obs, info
        step(action) -> obs, reward, terminated, truncated, info

    Observation layout, dim = 87:
        base_lin_vel       3
        base_ang_vel       3
        projected_gravity  3
        smoothed_cmd       3
        q_err             12
        qd                12
        last_action       12
        action_delta      12
        foot_contact       4
        foot_rel_pos      12
        foot_vel_xy        8
        base_height        1
        sin_phase          1
        cos_phase          1
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task1Config):
        super().__init__()

        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.dt = float(cfg.control_dt)

        sim_cfg = sim_utils.SimulationCfg(
            dt=float(cfg.sim_dt),
            device=str(cfg.device),
            physx=sim_utils.PhysxCfg(
                enable_external_forces_every_iteration=True,
                min_position_iteration_count=4,
                max_position_iteration_count=8,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=2,
            ),
        )

        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene = InteractiveScene(
            Go2Task1SceneCfg(
                num_envs=int(cfg.num_envs),
                env_spacing=float(cfg.env_spacing),
            )
        )
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact: ContactSensor = self.scene["contact_forces"]

        self.robot_joint_names = list(self.robot.joint_names)
        self.robot_body_names = list(self.robot.body_names)

        self.action_joint_ids = self._joint_ids(cfg.action_joint_names)
        self.foot_body_ids = self._body_ids(cfg.foot_body_names)
        self.contact_foot_ids = self._contact_ids(cfg.foot_body_names)

        self.action_joint_ids_t = torch.as_tensor(self.action_joint_ids, dtype=torch.long, device=self.device)
        self.foot_body_ids_t = torch.as_tensor(self.foot_body_ids, dtype=torch.long, device=self.device)
        self.contact_foot_ids_t = torch.as_tensor(self.contact_foot_ids, dtype=torch.long, device=self.device)

        self.default_joint_pos_all = self.robot.data.default_joint_pos.detach().clone()
        self.default_joint_vel_all = torch.zeros_like(self.default_joint_pos_all)
        self.default_joint_pos = self.default_joint_pos_all[:, self.action_joint_ids_t]

        self.joint_limits = self.robot.data.joint_pos_limits.detach().clone()
        self.joint_lower = self.joint_limits[:, self.action_joint_ids_t, 0]
        self.joint_upper = self.joint_limits[:, self.action_joint_ids_t, 1]

        self.action_scale = self._make_action_scale()
        self.joint_position_targets = self.default_joint_pos_all.detach().clone()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_observations),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(cfg.num_actions),),
            dtype=np.float32,
        )

        n = self.num_envs
        a = int(cfg.num_actions)

        self.global_steps = 0
        self.episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.target_cmd = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.smoothed_cmd = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.last_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.last_base_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_acc_obs = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.phase = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.prev_foot_contact = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        self.feet_air_time = torch.zeros((n, 4), dtype=torch.float32, device=self.device)

        if bool(cfg.debug_print_names):
            self._print_debug_info()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _joint_ids(self, names: Iterable[str]) -> List[int]:
        names = list(names)
        missing = [name for name in names if name not in self.robot_joint_names]
        if missing:
            raise RuntimeError(
                f"[Go2Task1Env] Missing joints: {missing}\n"
                f"Available joints: {self.robot_joint_names}"
            )
        return [self.robot_joint_names.index(name) for name in names]

    def _body_ids(self, names: Iterable[str]) -> List[int]:
        ids: List[int] = []
        missing: List[str] = []

        for target in names:
            if target in self.robot_body_names:
                ids.append(self.robot_body_names.index(target))
            else:
                matches = [i for i, name in enumerate(self.robot_body_names) if target in name]
                if matches:
                    ids.append(matches[0])
                else:
                    missing.append(target)

        if missing:
            raise RuntimeError(
                f"[Go2Task1Env] Missing body links: {missing}\n"
                f"Available bodies: {self.robot_body_names}"
            )
        return ids

    def _contact_ids(self, names: Iterable[str]) -> List[int]:
        contact_names = list(self.contact.body_names)
        ids: List[int] = []
        missing: List[str] = []

        for target in names:
            if target in contact_names:
                ids.append(contact_names.index(target))
            else:
                matches = [i for i, name in enumerate(contact_names) if target in name]
                if matches:
                    ids.append(matches[0])
                else:
                    missing.append(target)

        if missing:
            raise RuntimeError(
                f"[Go2Task1Env] Missing contact links: {missing}\n"
                f"Available contact bodies: {contact_names}"
            )
        return ids

    def _make_action_scale(self) -> torch.Tensor:
        scale = torch.zeros(int(self.cfg.num_actions), dtype=torch.float32, device=self.device)
        for i, joint_id in enumerate(self.action_joint_ids):
            name = self.robot_joint_names[joint_id]
            if "hip" in name:
                scale[i] = float(self.cfg.hip_action_scale)
            elif "thigh" in name:
                scale[i] = float(self.cfg.thigh_action_scale)
            elif "calf" in name:
                scale[i] = float(self.cfg.calf_action_scale)
            else:
                scale[i] = 0.25
        return scale

    def _print_debug_info(self) -> None:
        print("\n" + "=" * 110)
        print(" [Go2Task1Env] Unitree Go2 Flat Locomotion Environment Initialized")
        print("=" * 110)
        print(f" num_envs              : {self.cfg.num_envs}")
        print(f" device                : {self.device}")
        print(f" sim_dt / decimation   : {self.cfg.sim_dt} / {self.cfg.decimation}")
        print(f" control_dt            : {self.dt}")
        print(f" num_joints            : {self.robot.num_joints}")
        print(f" num_actions           : {self.cfg.num_actions}")
        print(f" num_observations      : {self.cfg.num_observations}")
        print(f" action_joint_names    : {list(self.cfg.action_joint_names)}")
        print(f" action_joint_ids      : {self.action_joint_ids}")
        print(f" foot_body_names       : {list(self.cfg.foot_body_names)}")
        print(f" foot_body_ids         : {self.foot_body_ids}")
        print(f" contact_foot_ids      : {self.contact_foot_ids}")
        print(f" contact body names    : {list(self.contact.body_names)}")
        print("=" * 110 + "\n")

    @staticmethod
    def _mean_detached(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().mean()

    @staticmethod
    def _float_tensor(value: float, device: str) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=device)

    # -------------------------------------------------------------------------
    # Curriculum / command
    # -------------------------------------------------------------------------
    def _curriculum_k(self) -> float:
        return min(1.0, float(self.global_steps) / max(int(self.cfg.curriculum_total_steps), 1))

    def _command_stage(self) -> int:
        k = self._curriculum_k()
        if k < 0.06:
            return 0
        if k < 0.18:
            return 1
        if k < 0.36:
            return 2
        if k < 0.58:
            return 3
        if k < 0.82:
            return 4
        return 5

    def _command_ranges(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        stage = self._command_stage()
        if stage == 0:
            return self.cfg.cmd_vx_stage0, self.cfg.cmd_vy_stage0, self.cfg.cmd_wz_stage0
        if stage == 1:
            return self.cfg.cmd_vx_stage1, self.cfg.cmd_vy_stage1, self.cfg.cmd_wz_stage1
        if stage == 2:
            return self.cfg.cmd_vx_stage2, self.cfg.cmd_vy_stage2, self.cfg.cmd_wz_stage2
        if stage == 3:
            return self.cfg.cmd_vx_stage3, self.cfg.cmd_vy_stage3, self.cfg.cmd_wz_stage3
        if stage == 4:
            return self.cfg.cmd_vx_stage4, self.cfg.cmd_vy_stage4, self.cfg.cmd_wz_stage4
        return self.cfg.cmd_vx_stage5, self.cfg.cmd_vy_stage5, self.cfg.cmd_wz_stage5

    def _sample_commands(self, n: int) -> torch.Tensor:
        vx_range, vy_range, wz_range = self._command_ranges()
        cmd = torch.zeros((int(n), 3), dtype=torch.float32, device=self.device)

        cmd[:, 0] = torch.empty(int(n), device=self.device).uniform_(float(vx_range[0]), float(vx_range[1]))
        cmd[:, 1] = torch.empty(int(n), device=self.device).uniform_(float(vy_range[0]), float(vy_range[1]))
        cmd[:, 2] = torch.empty(int(n), device=self.device).uniform_(float(wz_range[0]), float(wz_range[1]))

        zero = torch.rand(int(n), device=self.device) < float(self.cfg.zero_command_prob)
        cmd[zero] = 0.0
        return cmd

    def _build_trot_contact_ref(self) -> torch.Tensor:
        phase_a = self.phase
        phase_b = torch.remainder(self.phase + 0.5, 1.0)

        stance_a = (phase_a < float(self.cfg.duty_factor)).float()
        stance_b = (phase_b < float(self.cfg.duty_factor)).float()

        # Foot order: FL, FR, RL, RR
        # Trot diagonal pairs: FL + RR, FR + RL
        return torch.stack([stance_a, stance_b, stance_b, stance_a], dim=-1)

    # -------------------------------------------------------------------------
    # Gym API
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_idx(env_ids)
        return self._compute_obs(), {}

    @torch.no_grad()
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        n = int(env_ids.numel())

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = self.scene.env_origins[env_ids, 0:2]
        root_state[:, 2] = self.scene.env_origins[env_ids, 2] + float(self.cfg.target_height) + 0.03
        root_state[:, 2] += torch.empty(n, device=self.device).uniform_(-0.015, 0.015)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        root_state[:, 7:13] = 0.0

        q = self.default_joint_pos_all[env_ids].clone()
        qd = torch.zeros_like(q)

        q[:, self.action_joint_ids_t] += torch.empty(
            (n, int(self.cfg.num_actions)),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-0.02, 0.02)

        lower = self.joint_limits[env_ids, :, 0]
        upper = self.joint_limits[env_ids, :, 1]
        q = torch.clamp(q, lower, upper)

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)

        self.robot.reset(env_ids)
        try:
            self.contact.reset(env_ids)
        except Exception:
            pass

        self.scene.update(dt=0.0)

        new_cmd = self._sample_commands(n)
        self.target_cmd[env_ids] = new_cmd
        self.smoothed_cmd[env_ids] = new_cmd

        self.last_action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0

        self.last_base_vel[env_ids] = self.robot.data.root_lin_vel_b[env_ids]
        self.base_acc_obs[env_ids] = 0.0

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0

        self.phase[env_ids] = torch.rand(n, dtype=torch.float32, device=self.device)
        self.prev_foot_contact[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0

        self.joint_position_targets[env_ids] = self.default_joint_pos_all[env_ids]

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        actions = actions.to(device=self.device, dtype=torch.float32)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)

        if actions.shape != (self.num_envs, int(self.cfg.num_actions)):
            raise RuntimeError(
                f"[Go2Task1Env] action shape mismatch: got {tuple(actions.shape)}, "
                f"expected {(self.num_envs, int(self.cfg.num_actions))}"
            )

        actions = torch.clamp(actions, -1.0, 1.0)

        resample = (self.episode_steps % int(self.cfg.resample_command_steps) == 0) & (self.episode_steps > 0)
        resample_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self.target_cmd[resample_ids] = self._sample_commands(int(resample_ids.numel()))

        self.smoothed_cmd.mul_(1.0 - float(self.cfg.command_smoothing))
        self.smoothed_cmd.add_(self.target_cmd, alpha=float(self.cfg.command_smoothing))

        self.prev_action.copy_(self.last_action)
        self.last_action.mul_(1.0 - float(self.cfg.action_ema_alpha))
        self.last_action.add_(actions, alpha=float(self.cfg.action_ema_alpha))
        self.last_action.clamp_(-1.0, 1.0)

        target_joint_pos = self.default_joint_pos + self.last_action * self.action_scale.unsqueeze(0)
        target_joint_pos = torch.clamp(target_joint_pos, self.joint_lower, self.joint_upper)

        self.joint_position_targets[:, self.action_joint_ids_t] = target_joint_pos
        self.robot.set_joint_position_target(self.joint_position_targets)

        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(float(self.cfg.sim_dt))

        self.global_steps += self.num_envs
        self.episode_steps += 1

        cmd_speed = torch.clamp(
            torch.norm(self.smoothed_cmd[:, :2], dim=-1) + 0.35 * torch.abs(self.smoothed_cmd[:, 2]),
            0.3,
            2.0,
        )
        self.phase = torch.remainder(
            self.phase + self.dt * float(self.cfg.gait_freq_hz) * cmd_speed,
            1.0,
        )

        obs_before_reset = self._compute_obs()
        rewards, terminated, truncated, info = self._compute_rewards_and_dones()

        self.episode_return += rewards
        info["telemetry"]["Episode_Return"] = self.episode_return.detach().float().mean()

        done = terminated | truncated
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)

        obs = obs_before_reset
        if reset_ids.numel() > 0:
            info["terminal_observation"] = obs_before_reset[reset_ids].clone()
            self._reset_idx(reset_ids)
            obs_after_reset = self._compute_obs()
            obs[reset_ids] = obs_after_reset[reset_ids]

        return obs, rewards, terminated, truncated, info

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Observation / contact
    # -------------------------------------------------------------------------
    def _get_foot_contact(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.contact.data

        if hasattr(data, "net_forces_w_history") and data.net_forces_w_history is not None:
            forces = data.net_forces_w_history[:, :, self.contact_foot_ids_t, :]
            normal_force = torch.max(forces[..., 2], dim=1)[0]
        else:
            forces = data.net_forces_w[:, self.contact_foot_ids_t, :]
            normal_force = forces[..., 2]

        contact = (normal_force > float(self.cfg.contact_force_threshold)).float()
        return contact, normal_force

    def _compute_obs(self) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        q = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        q_err = q - self.default_joint_pos

        contact, _ = self._get_foot_contact()

        root_pos = self.robot.data.root_pos_w
        base_height = (root_pos[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1)

        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        foot_rel_pos = (foot_pos - root_pos.unsqueeze(1)).reshape(self.num_envs, -1)

        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]
        foot_vel_xy_flat = foot_vel_xy.reshape(self.num_envs, -1)

        action_delta = self.last_action - self.prev_action

        sin_phase = torch.sin(2.0 * math.pi * self.phase).unsqueeze(-1)
        cos_phase = torch.cos(2.0 * math.pi * self.phase).unsqueeze(-1)

        obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                self.smoothed_cmd,
                q_err,
                qd,
                self.last_action,
                action_delta,
                contact,
                foot_rel_pos,
                foot_vel_xy_flat,
                base_height,
                sin_phase,
                cos_phase,
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"[Go2Task1Env] Observation dim mismatch: "
                f"got {obs.shape[-1]}, expected {self.cfg.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    # -------------------------------------------------------------------------
    # Reward / termination
    # -------------------------------------------------------------------------
    def _compute_rewards_and_dones(self):
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        vx, vy, vz = base_lin_vel[:, 0], base_lin_vel[:, 1], base_lin_vel[:, 2]
        wx, wy, wz = base_ang_vel[:, 0], base_ang_vel[:, 1], base_ang_vel[:, 2]

        root_pos = self.robot.data.root_pos_w
        base_height = root_pos[:, 2] - self.scene.env_origins[:, 2]

        base_acc = (base_lin_vel - self.last_base_vel) / max(self.dt, 1e-6)
        self.base_acc_obs.copy_(base_acc)
        self.last_base_vel.copy_(base_lin_vel)

        q = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        q_err = q - self.default_joint_pos

        contact, normal_force = self._get_foot_contact()
        contact_count = contact.sum(dim=-1)
        ref_contact = self._build_trot_contact_ref()

        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        foot_height = foot_pos[:, :, 2] - self.scene.env_origins[:, 2].unsqueeze(-1)
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]

        cmd = self.smoothed_cmd
        cmd_xy_norm = torch.norm(cmd[:, :2], dim=-1)
        cmd_xy_norm_safe = torch.clamp(cmd_xy_norm, min=1e-6)

        actual_along_cmd = torch.sum(base_lin_vel[:, :2] * cmd[:, :2], dim=-1) / cmd_xy_norm_safe
        target_speed = torch.clamp(cmd_xy_norm, min=0.08)
        cmd_yaw_abs = torch.abs(cmd[:, 2])

        move_gate = ((cmd_xy_norm > 0.05) | (cmd_yaw_abs > 0.08)).float()
        stand_gate = 1.0 - move_gate

        r_cmd_speed = move_gate * torch.exp(
            -4.0 * torch.square(torch.clamp(actual_along_cmd / target_speed - 1.0, min=-2.0, max=2.0))
        )
        p_under_speed = -move_gate * torch.clamp(0.65 * target_speed - actual_along_cmd, min=0.0)
        p_double_contact = -move_gate * torch.clamp(contact_count - 2.4, min=0.0)

        lin_err = torch.square(vx - cmd[:, 0]) + torch.square(vy - cmd[:, 1])
        yaw_err = torch.square(wz - cmd[:, 2])

        r_cmd_lin = torch.exp(-float(self.cfg.sigma_cmd_lin) * lin_err)
        r_cmd_yaw = torch.exp(-float(self.cfg.sigma_cmd_yaw) * yaw_err)

        r_stand = torch.exp(
            -float(self.cfg.sigma_stand)
            * (
                torch.square(vx)
                + torch.square(vy)
                + 0.5 * torch.square(wz)
                + 0.5 * torch.square(vz)
            )
        )

        r_cmd_lin_final = move_gate * r_cmd_lin + stand_gate * r_stand
        r_cmd_yaw_final = move_gate * r_cmd_yaw + stand_gate * torch.exp(
            -float(self.cfg.sigma_stand) * torch.square(wz)
        )

        r_phase_contact = (1.0 - torch.mean(torch.abs(contact - ref_contact), dim=-1)) * move_gate

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)
        self.feet_air_time += self.dt
        r_air_time = (
            torch.sum(
                torch.clamp(self.feet_air_time - float(self.cfg.air_time_target), min=0.0, max=0.5)
                * first_contact.float(),
                dim=-1,
            )
            * move_gate
        )
        self.feet_air_time = torch.where(contact > 0.5, torch.zeros_like(self.feet_air_time), self.feet_air_time)
        self.prev_foot_contact.copy_(contact)

        r_clearance = (
            torch.mean(
                (1.0 - contact)
                * torch.exp(-20.0 * torch.abs(foot_height - float(self.cfg.foot_clearance_target))),
                dim=-1,
            )
            * move_gate
        )

        r_upright = (1.0 - projected_gravity[:, 2]) * 0.5

        h_err = torch.square(base_height - float(self.cfg.target_height))
        r_height = torch.exp(-float(self.cfg.sigma_height) * h_err)

        p_base_ang = -torch.clamp(torch.square(wx) + torch.square(wy), max=6.0)
        p_base_acc = -torch.clamp(torch.sum(torch.square(base_acc), dim=-1), max=30.0)
        p_z_vel = -torch.abs(vz)

        p_default_pose = -torch.mean(torch.square(q_err), dim=-1)

        lower_margin = q - self.joint_lower
        upper_margin = self.joint_upper - q
        p_joint_limit = -torch.mean(
            torch.square(torch.clamp(0.04 - lower_margin, min=0.0))
            + torch.square(torch.clamp(0.04 - upper_margin, min=0.0)),
            dim=-1,
        )

        p_action_rate = -torch.mean(torch.square(self.last_action - self.prev_action), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.last_action), dim=-1)

        raw_foot_slip = torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * contact, dim=-1)
        p_foot_slip = -torch.clamp(raw_foot_slip, max=8.0)

        torques = getattr(self.robot.data, "applied_torque", torch.zeros_like(self.robot.data.joint_vel))
        tau = torques[:, self.action_joint_ids_t]
        p_torque = -torch.clamp(torch.mean(torch.square(tau), dim=-1), max=40.0)
        p_energy = -torch.clamp(torch.mean(torch.abs(tau * qd), dim=-1), max=20.0)

        r_alive = torch.ones_like(vx)

        continuous_raw = (
            float(self.cfg.w_cmd_lin) * r_cmd_lin_final
            + float(self.cfg.w_cmd_yaw) * r_cmd_yaw_final
            + float(self.cfg.w_cmd_speed) * r_cmd_speed
            + float(self.cfg.w_under_speed) * p_under_speed
            + float(self.cfg.w_double_contact) * p_double_contact
            + float(self.cfg.w_stand_still) * r_stand * stand_gate
            + float(self.cfg.w_phase_contact) * r_phase_contact
            + float(self.cfg.w_air_time) * r_air_time
            + float(self.cfg.w_clearance) * r_clearance
            + float(self.cfg.w_upright) * r_upright
            + float(self.cfg.w_height) * r_height
            + float(self.cfg.w_base_ang_vel) * p_base_ang
            + float(self.cfg.w_base_acc) * p_base_acc
            + float(self.cfg.w_z_vel) * p_z_vel
            + float(self.cfg.w_default_pose) * p_default_pose
            + float(self.cfg.w_alive) * r_alive
            + float(self.cfg.w_joint_limit) * p_joint_limit
            + float(self.cfg.w_action_rate) * p_action_rate
            + float(self.cfg.w_action_mag) * p_action_mag
            + float(self.cfg.w_foot_slip) * p_foot_slip
            + float(self.cfg.w_torque) * p_torque
            + float(self.cfg.w_energy) * p_energy
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        joint_vel_abs_max = torch.abs(self.robot.data.joint_vel).max(dim=-1)[0]

        is_fallen = (
            (base_height < float(self.cfg.fall_height))
            | (base_height > float(self.cfg.jump_height))
            | (torch.norm(projected_gravity[:, :2], dim=-1) > float(self.cfg.bad_orientation_xy))
            | (~torch.isfinite(base_height))
            | (~torch.isfinite(self.robot.data.joint_pos).all(dim=-1))
            | (joint_vel_abs_max > float(self.cfg.max_joint_vel_abs))
        )

        event_fall = torch.where(
            is_fallen,
            torch.full_like(continuous, float(self.cfg.penalty_fall)),
            torch.zeros_like(continuous),
        )

        reward_raw = continuous + event_fall

        projected_return = self.episode_return + reward_raw
        no_event = event_fall.abs() < 1e-6

        reward = torch.where(
            (projected_return > float(self.cfg.episode_return_abs_limit)) & no_event,
            float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward_raw,
        )
        reward = torch.where(
            (projected_return < -float(self.cfg.episode_return_abs_limit)) & no_event,
            -float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward,
        )

        terminated = is_fallen
        truncated = self.episode_steps >= int(self.cfg.max_episode_length)

        info = {
            "reward_components": {
                "R_Cmd_Lin": self._mean_detached(float(self.cfg.w_cmd_lin) * r_cmd_lin_final),
                "R_Cmd_Yaw": self._mean_detached(float(self.cfg.w_cmd_yaw) * r_cmd_yaw_final),
                "R_Cmd_Speed": self._mean_detached(float(self.cfg.w_cmd_speed) * r_cmd_speed),
                "P_Under_Speed": self._mean_detached(float(self.cfg.w_under_speed) * p_under_speed),
                "P_Double_Contact": self._mean_detached(float(self.cfg.w_double_contact) * p_double_contact),
                "R_Stand_Still": self._mean_detached(float(self.cfg.w_stand_still) * r_stand * stand_gate),
                "R_Phase_Contact": self._mean_detached(float(self.cfg.w_phase_contact) * r_phase_contact),
                "R_Air_Time": self._mean_detached(float(self.cfg.w_air_time) * r_air_time),
                "R_Clearance": self._mean_detached(float(self.cfg.w_clearance) * r_clearance),
                "R_Upright": self._mean_detached(float(self.cfg.w_upright) * r_upright),
                "R_Height": self._mean_detached(float(self.cfg.w_height) * r_height),
                "P_Base_Ang": self._mean_detached(float(self.cfg.w_base_ang_vel) * p_base_ang),
                "P_Base_Acc": self._mean_detached(float(self.cfg.w_base_acc) * p_base_acc),
                "P_Z_Vel": self._mean_detached(float(self.cfg.w_z_vel) * p_z_vel),
                "P_Default_Pose": self._mean_detached(float(self.cfg.w_default_pose) * p_default_pose),
                "R_Alive": self._mean_detached(float(self.cfg.w_alive) * r_alive),
                "P_Joint_Limit": self._mean_detached(float(self.cfg.w_joint_limit) * p_joint_limit),
                "P_Action_Rate": self._mean_detached(float(self.cfg.w_action_rate) * p_action_rate),
                "P_Action_Mag": self._mean_detached(float(self.cfg.w_action_mag) * p_action_mag),
                "P_Foot_Slip": self._mean_detached(float(self.cfg.w_foot_slip) * p_foot_slip),
                "P_Torque": self._mean_detached(float(self.cfg.w_torque) * p_torque),
                "P_Energy": self._mean_detached(float(self.cfg.w_energy) * p_energy),
                "Continuous": self._mean_detached(continuous),
                "Event_Fall": self._mean_detached(event_fall),
                "Total": self._mean_detached(reward),
            },
            "events": {
                "Fall_Rate": self._mean_detached(is_fallen.float()),
                "Timeout_Rate": self._mean_detached(truncated.float()),
            },
            "telemetry": {
                "Curriculum_K": self._float_tensor(self._curriculum_k(), self.device),
                "Command_Stage": self._float_tensor(float(self._command_stage()), self.device),
                "Cmd_Vx": self._mean_detached(cmd[:, 0]),
                "Cmd_Vy": self._mean_detached(cmd[:, 1]),
                "Cmd_Wz": self._mean_detached(cmd[:, 2]),
                "Actual_Vx": self._mean_detached(vx),
                "Actual_Vy": self._mean_detached(vy),
                "Actual_Wz": self._mean_detached(wz),
                "Actual_Along_Cmd": self._mean_detached(actual_along_cmd * move_gate),
                "Lin_Error": self._mean_detached(lin_err),
                "Yaw_Error": self._mean_detached(yaw_err),
                "Base_Height": self._mean_detached(base_height),
                "Contact_Count": self._mean_detached(contact_count),
                "FL_Contact": self._mean_detached(contact[:, 0]),
                "FR_Contact": self._mean_detached(contact[:, 1]),
                "RL_Contact": self._mean_detached(contact[:, 2]),
                "RR_Contact": self._mean_detached(contact[:, 3]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Global_Steps": self._float_tensor(float(self.global_steps), self.device),
            },
            "debug": {
                "Obs_Dim": self._float_tensor(float(self.cfg.num_observations), self.device),
                "Reward_Min": reward.detach().min(),
                "Reward_Max": reward.detach().max(),
                "Continuous_Min": continuous.detach().min(),
                "Continuous_Max": continuous.detach().max(),
                "Base_Height_Min": base_height.detach().min(),
                "Base_Height_Max": base_height.detach().max(),
                "JointVel_Max": joint_vel_abs_max.detach().max(),
            },
        }

        return reward, terminated, truncated, info


# Backward-compatible aliases for older local scripts.
QuadrupedFlatEnv = Go2Task1Env
UnitreeGo2FlatEnv = Go2Task1Env
