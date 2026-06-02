# Copyright (c) 2026
# Unitree Go2 Task4: Sim2Real / RMA teacher IsaacLab 环境。
#
# 本文件只定义 Task4 IsaacLab 环境，不启动 AppLauncher。
# 环境采用 Gymnasium step API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# 观测维度:
#   single actor obs = 48
#   actor history obs = 240
#   privileged obs = 25
#   teacher obs = 265
#   action dim = 12
#
# 训练入口位于 task4_train.py，模型评估入口位于 task4_model_test.py。
#
# 工程说明:
#   teacher_mode=True 时，reset()/step() 返回 actor history 和 privileged obs 拼接后的 teacher obs。
#   teacher_mode=False 时，reset()/step() 返回 actor history，用于后续 student / adaptation 阶段。
#   domain randomization 包括 friction、payload mass、COM shift、motor strength、latency 和 external push。
#   info 中保留 GPU tensor，低频日志阶段再转换为标量，以减少 step 内 CPU 同步。
#
# Unitree Go2 Task4: Sim2Real / RMA teacher IsaacLab environment.
#
# This file only defines the Task4 IsaacLab environment and does not launch AppLauncher.
# The environment follows the Gymnasium step API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# Observation dimensions:
#   single actor obs = 48
#   actor history obs = 240
#   privileged obs = 25
#   teacher obs = 265
#   action dim = 12
#
# Training entry is task4_train.py, and model evaluation entry is task4_model_test.py.
#
# Engineering notes:
#   When teacher_mode=True, reset()/step() returns teacher obs formed by concatenating
#   actor history and privileged obs. When teacher_mode=False, reset()/step() returns
#   actor history for a later student / adaptation stage.
#   Domain randomization includes friction, payload mass, COM shift, motor strength,
#   latency, and external push. info keeps GPU tensors and converts them to scalars
#   only during low-frequency logging to reduce CPU synchronization inside step.

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*set_external_force_and_torque.*")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

try:
    from isaaclab_assets import UNITREE_GO2_CFG
except Exception:
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from go2_rl.tasks.task4.task4_config import Task4Config


def make_go2_task4_scene_cfg(cfg: Task4Config):
    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    try:
        robot_cfg.spawn.activate_contact_sensors = True
    except Exception:
        pass

    @configclass
    class Go2Task4SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = float(cfg.env_spacing)

        ground: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        robot: ArticulationCfg = robot_cfg

        contact_forces: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
            update_period=0.0,
            history_length=3,
            track_air_time=False,
            debug_vis=False,
        )

        light: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

    return Go2Task4SceneCfg(num_envs=int(cfg.num_envs), env_spacing=float(cfg.env_spacing))


class Go2Task4Env(gym.Env):
    """Go2 Task4 Sim2Real / RMA robust locomotion environment.

    Actor single-frame observation, dim = 48:
        base_ang_vel_b       3
        projected_gravity_b  3
        joint_pos_error     12
        joint_vel           12
        commands             3
        last_action         12
        phase sin/cos        2
        base_height          1

    Actor history observation, dim = 48 * frame_stack = 240.

    Privileged observation, dim = 25:
        base_lin_vel_b       3
        friction             1
        payload_mass_norm    1
        com_shift            3
        motor_strength      12
        push_force_body      3
        push_active          1
        post_push_timer      1

    Teacher observation, dim = 240 + 25 = 265.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task4Config):
        super().__init__()

        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.dt = float(cfg.policy_dt)

        self.single_actor_obs_dim = int(cfg.single_actor_obs_dim)
        self.actor_obs_dim = int(cfg.actor_obs_dim)
        self.privileged_obs_dim = int(cfg.privileged_obs_dim)
        self.teacher_obs_dim = int(cfg.teacher_obs_dim)

        if self.single_actor_obs_dim != 48:
            raise ValueError("Task4 single_actor_obs_dim must be 48.")
        if self.actor_obs_dim != self.single_actor_obs_dim * int(cfg.frame_stack):
            raise ValueError("Task4 actor_obs_dim must equal single_actor_obs_dim * frame_stack.")
        if self.privileged_obs_dim != 25:
            raise ValueError("Task4 privileged_obs_dim must be 25.")
        if self.teacher_obs_dim != self.actor_obs_dim + self.privileged_obs_dim:
            raise ValueError("Task4 teacher_obs_dim must equal actor_obs_dim + privileged_obs_dim.")

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
        self.scene = InteractiveScene(make_go2_task4_scene_cfg(cfg))
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact: ContactSensor = self.scene["contact_forces"]

        self.env_origins = self._get_env_origins()

        self.robot_joint_names = list(self.robot.joint_names)
        self.robot_body_names = list(self.robot.body_names)

        self.action_joint_ids = self._joint_ids(cfg.action_joint_names)
        self.foot_body_ids = self._body_ids(cfg.foot_body_names)
        self.contact_foot_ids = self._contact_ids(cfg.foot_body_names)

        self.action_joint_ids_t = torch.as_tensor(self.action_joint_ids, dtype=torch.long, device=self.device)
        self.foot_body_ids_t = torch.as_tensor(self.foot_body_ids, dtype=torch.long, device=self.device)
        self.contact_foot_ids_t = torch.as_tensor(self.contact_foot_ids, dtype=torch.long, device=self.device)

        self.num_actions = int(cfg.num_actions)

        self.default_joint_pos_all = self.robot.data.default_joint_pos.detach().clone()
        self.default_joint_vel_all = torch.zeros_like(self.default_joint_pos_all)
        self.default_joint_pos = self.default_joint_pos_all[:, self.action_joint_ids_t].detach().clone()

        lower, upper = self._get_joint_limits()
        self.joint_lower_all = lower
        self.joint_upper_all = upper
        self.joint_lower = lower[:, self.action_joint_ids_t]
        self.joint_upper = upper[:, self.action_joint_ids_t]

        obs_dim = int(cfg.num_observations)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.privileged_obs_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        # Public dims for training / testing.
        self.num_observations = obs_dim
        self.num_privileged_obs = self.privileged_obs_dim

        # ----------------------------- Buffers -----------------------------
        self.actor_obs_stack = torch.zeros(
            (self.num_envs, int(cfg.frame_stack), self.single_actor_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.privileged_obs_buf = torch.zeros(
            (self.num_envs, self.privileged_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.last_action = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)

        self.joint_position_targets = self.default_joint_pos_all.detach().clone()

        self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.command_time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.global_steps = 0

        # Domain randomization buffers.
        self.env_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dr_friction = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.dr_payload_mass = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dr_com_shift = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.dr_motor_strength = torch.ones((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)

        # External disturbance buffers.
        self.push_force_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.push_force_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.push_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.push_time_left = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.next_push_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.post_push_timer = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Reward helper buffers.
        self.last_tracking_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_base_vel = self.robot.data.root_lin_vel_b.detach().clone()
        self.base_acc_obs = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self.phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_foot_contact = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.feet_air_time = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        # Episode counters.
        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        # success 现在表示 tracking success，不再等同于 timeout alive。
        self.total_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_fall_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_alive_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_tracking_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_tracking_fail_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        stage_count = len(self.cfg.stage_thresholds)
        self.stage_done_counter = torch.zeros(stage_count, dtype=torch.float32, device=self.device)
        # success 现在统计 tracking success；timeout alive 单独统计。
        self.stage_success_counter = torch.zeros(stage_count, dtype=torch.float32, device=self.device)
        self.stage_fall_counter = torch.zeros(stage_count, dtype=torch.float32, device=self.device)
        self.stage_timeout_counter = torch.zeros(stage_count, dtype=torch.float32, device=self.device)
        self.stage_alive_timeout_counter = torch.zeros(stage_count, dtype=torch.float32, device=self.device)
        self.stage_tracking_fail_timeout_counter = torch.zeros(stage_count, dtype=torch.float32, device=self.device)

        self.reset()

        if bool(self.cfg.print_debug_info):
            self._print_debug_info()

    # -------------------------------------------------------------------------
    # Helpers / robot mapping
    # -------------------------------------------------------------------------
    def _get_env_origins(self) -> torch.Tensor:
        if hasattr(self.scene, "env_origins"):
            return self.scene.env_origins.to(self.device)

        try:
            return self.scene._default_env_origins.to(self.device)
        except Exception:
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _get_joint_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.robot.data, "soft_joint_pos_limits"):
            limits = self.robot.data.soft_joint_pos_limits
        elif hasattr(self.robot.data, "joint_pos_limits"):
            limits = self.robot.data.joint_pos_limits
        else:
            raise RuntimeError("[Go2Task4Env] Cannot find joint position limits.")

        limits = limits.detach().clone()

        if limits.shape[0] == 1:
            limits = limits.repeat(self.num_envs, 1, 1)

        if limits.shape[0] != self.num_envs:
            limits = limits[:1].repeat(self.num_envs, 1, 1)

        lower = limits[:, :, 0]
        upper = limits[:, :, 1]

        return lower.to(self.device), upper.to(self.device)

    def _joint_ids(self, names: Iterable[str]) -> List[int]:
        names = list(names)
        ids: List[int] = []
        missing: List[str] = []

        for target in names:
            if target in self.robot_joint_names:
                ids.append(self.robot_joint_names.index(target))
            else:
                matches = [i for i, name in enumerate(self.robot_joint_names) if target in name]
                if matches:
                    ids.append(matches[0])
                else:
                    missing.append(target)

        if len(ids) != 12:
            raise RuntimeError(
                f"[Go2Task4Env] Missing or invalid action joints: {missing}\n"
                f"Resolved ids: {ids}\n"
                f"Available joints: {self.robot_joint_names}"
            )

        return ids

    def _body_ids(self, names: Iterable[str]) -> List[int]:
        names = list(names)
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

        if len(ids) != 4:
            raise RuntimeError(
                f"[Go2Task4Env] Missing or invalid foot bodies: {missing}\n"
                f"Resolved ids: {ids}\n"
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

        if len(ids) != 4:
            raise RuntimeError(
                f"[Go2Task4Env] Missing or invalid contact bodies: {missing}\n"
                f"Resolved ids: {ids}\n"
                f"Available contact bodies: {contact_names}"
            )

        return ids

    @staticmethod
    def _quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _yaw_to_quat_wxyz(self, yaw: torch.Tensor) -> torch.Tensor:
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device).view(-1)
        quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=self.device)
        quat[:, 0] = torch.cos(yaw * 0.5)
        quat[:, 3] = torch.sin(yaw * 0.5)
        return quat

    def _rotate_world_to_body(self, vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device).view(-1)
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)

        x = vec_w[:, 0]
        y = vec_w[:, 1]

        bx = c * x - s * y
        by = s * x + c * y

        if vec_w.shape[-1] == 2:
            return torch.stack([bx, by], dim=-1)

        return torch.stack([bx, by, vec_w[:, 2]], dim=-1)

    @staticmethod
    def _mean_detached(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().mean()

    def _float_tensor(self, value: float) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print(" [Go2Task4Env] Sim2Real / RMA Robust Locomotion Environment Initialized")
        print("=" * 120)
        print(f" num_envs              : {self.num_envs}")
        print(f" device                : {self.device}")
        print(f" num_joints            : {self.robot.num_joints}")
        print(f" num_actions           : {self.num_actions}")
        print(f" single_actor_obs_dim  : {self.single_actor_obs_dim}")
        print(f" actor_obs_dim         : {self.actor_obs_dim}")
        print(f" privileged_obs_dim    : {self.privileged_obs_dim}")
        print(f" teacher_obs_dim       : {self.teacher_obs_dim}")
        print(f" returned_obs_dim      : {self.num_observations}")
        print(f" teacher_mode          : {self.cfg.teacher_mode}")
        print(f" sim_dt                : {self.cfg.sim_dt}")
        print(f" policy_dt             : {self.dt}")
        print(f" decimation            : {self.cfg.decimation}")
        print(f" max_episode_length    : {self.cfg.max_episode_length}")
        print(f" action_joint_ids      : {self.action_joint_ids}")
        print(f" foot_body_ids         : {self.foot_body_ids}")
        print(f" contact_foot_ids      : {self.contact_foot_ids}")
        print(f" contact body names    : {list(self.contact.body_names)}")
        print("=" * 120 + "\n")

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------
    def curriculum_k(self) -> float:
        return min(
            1.0,
            max(0.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0)),
        )

    def stage_from_progress(self, k: float) -> int:
        k = float(max(0.0, min(1.0, k)))
        stage = 0
        for i, th in enumerate(self.cfg.stage_thresholds):
            if k >= float(th):
                stage = i
        return int(min(stage, len(self.cfg.stage_thresholds) - 1))

    def stage_from_global_steps(self, global_steps: int) -> int:
        k = min(
            1.0,
            max(0.0, float(global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0)),
        )
        return self.stage_from_progress(k)

    def _stage_float_range(self, ranges: Tuple[Tuple[float, float], ...], stages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stages = torch.as_tensor(stages, dtype=torch.long, device=self.device)
        stages = torch.clamp(stages, 0, len(ranges) - 1)

        mins = torch.tensor([r[0] for r in ranges], dtype=torch.float32, device=self.device)[stages]
        maxs = torch.tensor([r[1] for r in ranges], dtype=torch.float32, device=self.device)[stages]
        return mins, maxs

    def _sample_stage_float(self, ranges: Tuple[Tuple[float, float], ...], stages: torch.Tensor) -> torch.Tensor:
        mins, maxs = self._stage_float_range(ranges, stages)
        return mins + torch.rand_like(mins) * (maxs - mins)

    # -------------------------------------------------------------------------
    # Gym API
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return self._get_return_obs(), {}

        self._reset_idx(env_ids)

        return self._get_return_obs(env_ids=None), {}

    @torch.no_grad()
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        n = int(env_ids.numel())

        if int(getattr(self.cfg, "force_stage", -1)) >= 0:
            current_stage = int(max(0, min(int(getattr(self.cfg, "force_stage", -1)), len(self.cfg.stage_thresholds) - 1)))
        else:
            current_stage = self.stage_from_global_steps(int(self.global_steps))
        stages = torch.full((n,), current_stage, dtype=torch.long, device=self.device)
        self.env_stage[env_ids] = stages

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.env_origins[env_ids]

        root_state[:, 2] = float(self.cfg.target_height) + torch.empty(
            n,
            dtype=torch.float32,
            device=self.device,
        ).uniform_(0.0, 0.03)

        yaw_noise = torch.empty(n, dtype=torch.float32, device=self.device).uniform_(
            -float(self.cfg.reset_yaw_noise),
            float(self.cfg.reset_yaw_noise),
        )
        root_state[:, 3:7] = self._yaw_to_quat_wxyz(yaw_noise)
        root_state[:, 7:13] = 0.0

        q = self.default_joint_pos_all[env_ids].clone()
        qd = torch.zeros_like(q)

        q[:, self.action_joint_ids_t] += torch.empty(
            (n, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-float(self.cfg.reset_joint_noise), float(self.cfg.reset_joint_noise))

        lower = self.joint_lower_all[env_ids]
        upper = self.joint_upper_all[env_ids]
        q = torch.clamp(q, lower, upper)

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)

        self.robot.reset(env_ids)
        try:
            self.contact.reset(env_ids)
        except Exception:
            pass

        self.scene.update(dt=0.0)

        self._resample_commands(env_ids)
        self._randomize_domains(env_ids)
        self._reset_push_schedule(env_ids)

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0

        self.last_action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.joint_position_targets[env_ids] = self.default_joint_pos_all[env_ids]

        self.last_tracking_error[env_ids] = 0.0
        self.last_base_vel[env_ids] = self.robot.data.root_lin_vel_b[env_ids]
        self.base_acc_obs[env_ids] = 0.0

        self.phase[env_ids] = torch.rand(n, dtype=torch.float32, device=self.device)
        self.prev_foot_contact[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0

        obs_single, priv = self._compute_obs()

        for i in range(int(self.cfg.frame_stack)):
            self.actor_obs_stack[env_ids, i, :] = obs_single[env_ids]

        self.privileged_obs_buf[env_ids] = priv[env_ids]

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        actions = actions.to(device=self.device, dtype=torch.float32)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)

        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise RuntimeError(
                f"[Go2Task4Env] action shape mismatch: got {tuple(actions.shape)}, "
                f"expected {(self.num_envs, self.num_actions)}"
            )

        actions = torch.clamp(actions, -1.0, 1.0)

        self.prev_action.copy_(self.last_action)

        filtered_action = (
            float(self.cfg.action_ema_alpha) * actions
            + (1.0 - float(self.cfg.action_ema_alpha)) * self.last_action
        )
        self.last_action.copy_(torch.clamp(filtered_action, -1.0, 1.0))

        self.command_time_left -= self.dt
        resample_ids = (self.command_time_left <= 0.0).nonzero(as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._resample_commands(resample_ids)

        degraded_action = self.last_action * self.dr_motor_strength

        target_joint_pos = self.default_joint_pos + degraded_action * float(self.cfg.action_scale)
        target_joint_pos = torch.clamp(target_joint_pos, self.joint_lower, self.joint_upper)

        self.joint_position_targets[:, self.action_joint_ids_t] = target_joint_pos
        self.robot.set_joint_position_target(self.joint_position_targets)

        self._update_external_disturbances()

        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(float(self.cfg.sim_dt))

        self.global_steps += self.num_envs
        self.episode_steps += 1

        speed_for_phase = torch.clamp(torch.abs(self.commands[:, 0]), 0.2, 1.5)
        self.phase = torch.remainder(
            self.phase
            + self.dt * (float(self.cfg.gait_freq_hz) + float(self.cfg.gait_freq_speed_gain) * speed_for_phase),
            1.0,
        )

        obs_single, priv = self._compute_obs()

        self.actor_obs_stack = torch.roll(self.actor_obs_stack, shifts=-1, dims=1)
        self.actor_obs_stack[:, -1, :] = obs_single
        self.privileged_obs_buf = priv

        reward, terminated, truncated, info = self._compute_rewards_and_dones()

        obs_before_reset = self._get_return_obs()
        self.episode_return += reward

        info["telemetry"]["Episode_Return"] = self._mean_detached(self.episode_return)

        done = terminated | truncated
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)

        obs = obs_before_reset
        if reset_ids.numel() > 0:
            info["terminal_observation"] = obs_before_reset[reset_ids].clone()
            self._reset_idx(reset_ids)
            reset_obs = self._get_return_obs()
            obs[reset_ids] = reset_obs[reset_ids]

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            zero_force = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
            self.robot.set_external_force_and_torque(zero_force, zero_force, body_ids=[0])
        except Exception:
            pass

        try:
            self.sim.stop()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Commands / Domain Randomization / Disturbances
    # -------------------------------------------------------------------------
    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        stages = self.env_stage[env_ids]

        vx_min, vx_max = self._stage_float_range(self.cfg.cmd_vx_ranges, stages)
        vy_min, vy_max = self._stage_float_range(self.cfg.cmd_vy_ranges, stages)
        wz_min, wz_max = self._stage_float_range(self.cfg.cmd_wz_ranges, stages)

        n = int(env_ids.numel())

        self.commands[env_ids, 0] = vx_min + torch.rand(n, dtype=torch.float32, device=self.device) * (vx_max - vx_min)
        self.commands[env_ids, 1] = vy_min + torch.rand(n, dtype=torch.float32, device=self.device) * (vy_max - vy_min)
        self.commands[env_ids, 2] = wz_min + torch.rand(n, dtype=torch.float32, device=self.device) * (wz_max - wz_min)

        self.command_time_left[env_ids] = float(self.cfg.command_resampling_time_s)

    def _randomize_domains(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        stages = self.env_stage[env_ids]
        n = int(env_ids.numel())

        self.dr_friction[env_ids] = self._sample_stage_float(self.cfg.friction_ranges, stages)
        self.dr_payload_mass[env_ids] = self._sample_stage_float(self.cfg.payload_mass_ranges, stages)

        com_min, com_max = self._stage_float_range(self.cfg.com_shift_ranges, stages)
        com_rand = torch.rand((n, 3), dtype=torch.float32, device=self.device)
        self.dr_com_shift[env_ids] = com_min.unsqueeze(-1) + com_rand * (com_max - com_min).unsqueeze(-1)

        self.dr_motor_strength[env_ids] = 1.0

        motor_min, motor_max = self._stage_float_range(self.cfg.motor_strength_ranges, stages)

        max_deg = torch.tensor(
            [self.cfg.max_degraded_joints_by_stage[int(s.item())] for s in stages],
            dtype=torch.long,
            device=self.device,
        )

        if max_deg.max().item() > 0:
            rand_rank = torch.rand((n, self.num_actions), dtype=torch.float32, device=self.device)
            order = torch.argsort(rand_rank, dim=-1)

            rank = torch.empty_like(order)
            rank.scatter_(
                1,
                order,
                torch.arange(self.num_actions, dtype=torch.long, device=self.device).unsqueeze(0).repeat(n, 1),
            )

            degraded_mask = rank < max_deg.unsqueeze(-1)

            strengths = motor_min.unsqueeze(-1) + torch.rand(
                (n, self.num_actions),
                dtype=torch.float32,
                device=self.device,
            ) * (motor_max - motor_min).unsqueeze(-1)

            self.dr_motor_strength[env_ids] = torch.where(
                degraded_mask,
                strengths,
                torch.ones_like(strengths),
            )

    def _reset_push_schedule(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        stages = self.env_stage[env_ids]
        interval_min, interval_max = self._stage_float_range(self.cfg.push_interval_ranges_s, stages)

        self.next_push_time[env_ids] = interval_min + torch.rand(
            int(env_ids.numel()),
            dtype=torch.float32,
            device=self.device,
        ) * (interval_max - interval_min)

        self.push_time_left[env_ids] = 0.0
        self.post_push_timer[env_ids] = 0.0
        self.push_active[env_ids] = False
        self.push_force_w[env_ids] = 0.0
        self.push_force_b[env_ids] = 0.0

    def _update_external_disturbances(self) -> None:
        self.post_push_timer = torch.clamp(self.post_push_timer - self.dt, min=0.0)

        active_ids = self.push_active.nonzero(as_tuple=False).squeeze(-1)
        if active_ids.numel() > 0:
            self.push_time_left[active_ids] -= self.dt

            finished = active_ids[self.push_time_left[active_ids] <= 0.0]
            if finished.numel() > 0:
                self.push_active[finished] = False
                self.push_time_left[finished] = 0.0
                self.post_push_timer[finished] = float(self.cfg.post_push_recovery_window_s)
                self.push_force_w[finished] = 0.0

                stages = self.env_stage[finished]
                interval_min, interval_max = self._stage_float_range(self.cfg.push_interval_ranges_s, stages)
                self.next_push_time[finished] = interval_min + torch.rand(
                    int(finished.numel()),
                    dtype=torch.float32,
                    device=self.device,
                ) * (interval_max - interval_min)

        inactive = ~self.push_active
        self.next_push_time[inactive] -= self.dt

        start_push = inactive & (self.next_push_time <= 0.0)
        start_ids = start_push.nonzero(as_tuple=False).squeeze(-1)

        if start_ids.numel() > 0:
            stages = self.env_stage[start_ids]
            mag_min, mag_max = self._stage_float_range(self.cfg.push_magnitude_ranges, stages)

            mags = mag_min + torch.rand(
                int(start_ids.numel()),
                dtype=torch.float32,
                device=self.device,
            ) * (mag_max - mag_min)

            angles = torch.rand(int(start_ids.numel()), dtype=torch.float32, device=self.device) * 2.0 * math.pi

            self.push_force_w[start_ids, 0] = torch.cos(angles) * mags
            self.push_force_w[start_ids, 1] = torch.sin(angles) * mags
            self.push_force_w[start_ids, 2] = (
                torch.rand(int(start_ids.numel()), dtype=torch.float32, device=self.device) - 0.5
            ) * 40.0

            self.push_active[start_ids] = mags > 1e-6
            self.push_time_left[start_ids] = float(self.cfg.push_duration_s)

        payload_force_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        payload_force_w[:, 2] = -self.dr_payload_mass * 9.81

        payload_torque_w = torch.cross(self.dr_com_shift, payload_force_w, dim=-1)

        total_force_w = payload_force_w + self.push_force_w
        total_torque_w = payload_torque_w

        yaw = self._quat_yaw(self.robot.data.root_quat_w)
        self.push_force_b = self._rotate_world_to_body(self.push_force_w, yaw)

        try:
            self.robot.set_external_force_and_torque(
                total_force_w.unsqueeze(1),
                total_torque_w.unsqueeze(1),
                body_ids=[0],
            )
        except Exception:
            # Some IsaacLab versions expose this API but may not enable it in a given
            # articulation backend. Keep privileged buffers valid even if the force
            # application backend is unavailable.
            pass

    # -------------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------------
    def _compute_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b
        base_lin_vel = self.robot.data.root_lin_vel_b

        joint_pos = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        joint_vel = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        joint_pos_err = joint_pos - self.default_joint_pos

        base_height = self._base_height()

        noise_table = torch.tensor(
            list(self.cfg.noise_level_by_stage),
            dtype=torch.float32,
            device=self.device,
        )
        stages = torch.clamp(self.env_stage, 0, len(self.cfg.noise_level_by_stage) - 1)
        noise_scale = noise_table[stages].unsqueeze(-1)

        base_ang_vel_obs = base_ang_vel + torch.randn_like(base_ang_vel) * float(self.cfg.noise_base_ang_vel) * noise_scale
        proj_gravity_obs = proj_gravity + torch.randn_like(proj_gravity) * float(self.cfg.noise_proj_gravity) * noise_scale
        joint_pos_obs = joint_pos_err + torch.randn_like(joint_pos_err) * float(self.cfg.noise_joint_pos) * noise_scale
        joint_vel_obs = joint_vel + torch.randn_like(joint_vel) * float(self.cfg.noise_joint_vel) * noise_scale

        base_height_obs = base_height.unsqueeze(-1) + torch.randn(
            (self.num_envs, 1),
            dtype=torch.float32,
            device=self.device,
        ) * (float(self.cfg.noise_base_height) * noise_scale)

        phase_sin = torch.sin(2.0 * math.pi * self.phase).unsqueeze(-1)
        phase_cos = torch.cos(2.0 * math.pi * self.phase).unsqueeze(-1)

        actor_obs = torch.cat(
            [
                base_ang_vel_obs * 0.25,
                proj_gravity_obs,
                joint_pos_obs,
                joint_vel_obs * 0.05,
                self.commands,
                self.last_action,
                phase_sin,
                phase_cos,
                base_height_obs,
            ],
            dim=-1,
        )

        if actor_obs.shape[-1] != self.single_actor_obs_dim:
            raise RuntimeError(
                f"[Go2Task4Env] actor_obs dim mismatch: got {actor_obs.shape[-1]}, "
                f"expected {self.single_actor_obs_dim}"
            )

        payload_norm = self.dr_payload_mass.unsqueeze(-1) / 5.0
        push_active_f = self.push_active.float().unsqueeze(-1)
        post_push_norm = (
            self.post_push_timer / max(float(self.cfg.post_push_recovery_window_s), 1e-6)
        ).unsqueeze(-1)

        priv = torch.cat(
            [
                base_lin_vel,
                self.dr_friction.unsqueeze(-1),
                payload_norm,
                self.dr_com_shift / 0.10,
                self.dr_motor_strength,
                self.push_force_b / 300.0,
                push_active_f,
                post_push_norm,
            ],
            dim=-1,
        )

        if priv.shape[-1] != self.privileged_obs_dim:
            raise RuntimeError(
                f"[Go2Task4Env] privileged obs dim mismatch: got {priv.shape[-1]}, "
                f"expected {self.privileged_obs_dim}"
            )

        actor_obs = torch.nan_to_num(
            torch.clamp(actor_obs, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        priv = torch.nan_to_num(
            torch.clamp(priv, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        return actor_obs, priv

    def compute_actor_obs(self) -> torch.Tensor:
        return torch.nan_to_num(
            torch.clamp(self.actor_obs_stack.reshape(self.num_envs, -1), -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def compute_privileged_obs(self) -> torch.Tensor:
        return torch.nan_to_num(
            torch.clamp(self.privileged_obs_buf, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def compute_teacher_obs(self) -> torch.Tensor:
        return torch.cat([self.compute_actor_obs(), self.compute_privileged_obs()], dim=-1)

    def _get_return_obs(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.teacher_mode:
            obs = self.compute_teacher_obs()
        else:
            obs = self.compute_actor_obs()

        if env_ids is not None:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            return obs[env_ids]

        return obs

    # -------------------------------------------------------------------------
    # Sensors / kinematics
    # -------------------------------------------------------------------------
    def _base_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2]

    def _foot_heights(self) -> torch.Tensor:
        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        return foot_pos[:, :, 2]

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

    def _build_trot_contact_ref(self) -> torch.Tensor:
        phase_a = self.phase
        phase_b = torch.remainder(self.phase + 0.5, 1.0)

        stance_a = (phase_a < float(self.cfg.duty_factor)).float()
        stance_b = (phase_b < float(self.cfg.duty_factor)).float()

        return torch.stack([stance_a, stance_b, stance_b, stance_a], dim=-1)

    # -------------------------------------------------------------------------
    # Reward / done
    # -------------------------------------------------------------------------
    def _compute_rewards_and_dones(self):
        base_height = self._base_height()

        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        proj_gravity = self.robot.data.projected_gravity_b

        roll_pitch_mag = torch.norm(proj_gravity[:, :2], dim=-1)

        vx = base_lin_vel[:, 0]
        vy = base_lin_vel[:, 1]
        vz = base_lin_vel[:, 2]

        wx = base_ang_vel[:, 0]
        wy = base_ang_vel[:, 1]
        wz = base_ang_vel[:, 2]

        q = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        q_err = q - self.default_joint_pos

        torques = getattr(self.robot.data, "applied_torque", torch.zeros_like(self.robot.data.joint_vel))
        tau = torques[:, self.action_joint_ids_t]

        base_acc = (base_lin_vel - self.last_base_vel) / max(self.dt, 1e-6)
        self.base_acc_obs.copy_(base_acc)
        self.last_base_vel.copy_(base_lin_vel)

        contact, normal_force = self._get_foot_contact()
        contact = contact.float()
        contact_count = contact.sum(dim=-1)

        foot_height = self._foot_heights()
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]

        # ----------------------------- A. Command tracking -----------------------------
        cmd_vx = self.commands[:, 0]
        cmd_vy = self.commands[:, 1]
        cmd_wz = self.commands[:, 2]

        cmd_speed_abs = torch.clamp(torch.abs(cmd_vx), min=float(self.cfg.cmd_speed_min))
        cmd_sign = torch.where(cmd_vx >= 0.0, torch.ones_like(cmd_vx), -torch.ones_like(cmd_vx))
        actual_forward = vx * cmd_sign
        speed_ratio = actual_forward / cmd_speed_abs
        forward_ratio = torch.clamp(speed_ratio, 0.0, 1.0)
        cmd_move_gate = (torch.abs(cmd_vx) > float(self.cfg.move_cmd_threshold)).float()

        lin_err_sq = torch.square(vx - cmd_vx) + 0.5 * torch.square(vy - cmd_vy)
        yaw_err_sq = torch.square(wz - cmd_wz)

        height_gate = torch.clamp((base_height - 0.22) / 0.08, 0.0, 1.0)
        upright_gate = torch.clamp(1.0 - roll_pitch_mag / 0.45, 0.0, 1.0)
        stability_gate = height_gate * upright_gate

        speed_gate = torch.clamp(speed_ratio / max(float(self.cfg.cmd_lin_speed_gate_ref), 1e-6), 0.0, 1.0)
        cmd_lin_gate = float(self.cfg.cmd_lin_speed_gate_floor) + (1.0 - float(self.cfg.cmd_lin_speed_gate_floor)) * speed_gate
        stability_speed_gate = float(self.cfg.stability_speed_gate_floor) + (1.0 - float(self.cfg.stability_speed_gate_floor)) * speed_gate

        r_cmd_lin_base = torch.exp(-float(self.cfg.sigma_cmd_lin) * lin_err_sq) * stability_gate
        r_cmd_lin = r_cmd_lin_base * ((1.0 - cmd_move_gate) + cmd_move_gate * cmd_lin_gate)
        r_cmd_yaw = torch.exp(-float(self.cfg.sigma_cmd_yaw) * yaw_err_sq) * upright_gate
        r_forward_ratio = forward_ratio * cmd_move_gate * stability_gate
        under_ratio = torch.relu(float(self.cfg.required_speed_ratio) - speed_ratio)
        p_under_speed = -torch.clamp(torch.square(under_ratio), 0.0, 2.0) * cmd_move_gate * stability_gate
        p_lateral_vel = -torch.clamp(torch.square(vy), max=2.0)
        p_low_height = -torch.square(torch.clamp((0.285 - base_height) / 0.10, 0.0, 2.0))

        # ----------------------------- B. Recovery -----------------------------
        tracking_error = torch.sqrt(lin_err_sq + 1e-6) + 0.5 * torch.sqrt(yaw_err_sq + 1e-6) + roll_pitch_mag

        error_improvement = torch.clamp(
            (self.last_tracking_error - tracking_error) / torch.clamp(self.last_tracking_error.abs() + 1e-4, min=1e-4),
            min=0.0,
            max=1.0,
        )

        recovery_gate = torch.clamp(
            self.post_push_timer / max(float(self.cfg.post_push_recovery_window_s), 1e-6),
            0.0,
            1.0,
        )

        r_tracking_recovery = recovery_gate * error_improvement
        self.last_tracking_error.copy_(tracking_error.detach())

        r_post_push_stability = recovery_gate * torch.exp(-2.0 * torch.square(roll_pitch_mag))
        r_push_survival = self.push_active.float() * torch.exp(-2.5 * torch.square(roll_pitch_mag))

        # ----------------------------- C. Stability -----------------------------
        r_upright_base = torch.exp(-float(self.cfg.sigma_upright) * torch.square(roll_pitch_mag))
        r_height_base = torch.exp(-float(self.cfg.sigma_height) * torch.square(base_height - float(self.cfg.target_height)))
        stability_reward_gate = (1.0 - cmd_move_gate) + cmd_move_gate * stability_speed_gate
        r_upright = r_upright_base * stability_reward_gate
        r_height = r_height_base * stability_reward_gate

        p_base_ang = -torch.clamp(torch.square(wx) + torch.square(wy), max=6.0)
        p_z_vel = -torch.abs(vz)
        r_alive = torch.ones_like(vx)

        # ----------------------------- D. Gait / contact -----------------------------
        ref_contact = self._build_trot_contact_ref()
        move_gate = torch.clamp(torch.abs(cmd_vx) / 0.6, 0.2, 1.0) * stability_gate

        r_phase_contact = (1.0 - torch.mean(torch.abs(contact - ref_contact), dim=-1)) * move_gate

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)
        self.feet_air_time += self.dt

        r_air_time = (
            torch.sum(
                torch.clamp(self.feet_air_time - float(self.cfg.air_time_target), min=0.0, max=0.45)
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
                * torch.exp(
                    -float(self.cfg.sigma_clearance)
                    * torch.abs(foot_height - float(self.cfg.foot_clearance_target))
                ),
                dim=-1,
            )
            * move_gate
        )

        r_contact_count = torch.exp(-torch.square(contact_count - 2.6)) * (0.25 + 0.75 * move_gate)

        slip_force_mask = (normal_force > float(self.cfg.foot_slip_force_threshold)).float()
        raw_foot_slip = torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * slip_force_mask, dim=-1)
        p_foot_slip = -torch.clamp(raw_foot_slip, max=8.0)

        impact = torch.clamp(normal_force.max(dim=-1)[0] - float(self.cfg.impact_threshold), min=0.0)
        p_impact = -torch.clamp(impact / max(float(self.cfg.impact_threshold), 1e-6), max=5.0)

        # ----------------------------- E. Regularization -----------------------------
        p_torque = -torch.clamp(torch.mean(torch.square(tau), dim=-1), max=40.0)
        p_energy = -torch.clamp(torch.mean(torch.abs(tau * qd), dim=-1), max=20.0)
        p_action_rate = -torch.mean(torch.square(self.last_action - self.prev_action), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.last_action), dim=-1)
        p_joint_vel = -torch.mean(torch.square(qd), dim=-1)

        lower_margin = q - self.joint_lower
        upper_margin = self.joint_upper - q
        p_joint_limit = -torch.mean(
            torch.square(torch.clamp(0.04 - lower_margin, min=0.0))
            + torch.square(torch.clamp(0.04 - upper_margin, min=0.0)),
            dim=-1,
        )

        continuous_raw = (
            float(self.cfg.w_cmd_lin) * r_cmd_lin
            + float(self.cfg.w_forward_ratio) * r_forward_ratio
            + float(self.cfg.w_under_speed) * p_under_speed
            + float(self.cfg.w_cmd_yaw) * r_cmd_yaw
            + float(self.cfg.w_lateral_vel) * p_lateral_vel
            + float(self.cfg.w_tracking_recovery) * r_tracking_recovery
            + float(self.cfg.w_post_push_stability) * r_post_push_stability
            + float(self.cfg.w_push_survival) * r_push_survival
            + float(self.cfg.w_upright) * r_upright
            + float(self.cfg.w_height) * r_height
            + float(self.cfg.w_low_height) * p_low_height
            + float(self.cfg.w_base_ang_vel) * p_base_ang
            + float(self.cfg.w_z_vel) * p_z_vel
            + float(self.cfg.w_alive) * r_alive
            + float(self.cfg.w_phase_contact) * r_phase_contact
            + float(self.cfg.w_air_time) * r_air_time
            + float(self.cfg.w_clearance) * r_clearance
            + float(self.cfg.w_contact_count) * r_contact_count
            + float(self.cfg.w_foot_slip) * p_foot_slip
            + float(self.cfg.w_impact) * p_impact
            + float(self.cfg.w_torque) * p_torque
            + float(self.cfg.w_energy) * p_energy
            + float(self.cfg.w_action_rate) * p_action_rate
            + float(self.cfg.w_action_mag) * p_action_mag
            + float(self.cfg.w_joint_vel) * p_joint_vel
            + float(self.cfg.w_joint_limit) * p_joint_limit
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
            | (roll_pitch_mag > float(self.cfg.bad_orientation_xy))
            | (~torch.isfinite(base_height))
            | (~torch.isfinite(self.robot.data.joint_pos).all(dim=-1))
            | (joint_vel_abs_max > float(self.cfg.max_joint_vel_abs))
        )

        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        terminated = is_fallen
        truncated = timeout & (~terminated)

        alive_timeout = truncated
        lin_track_ok = speed_ratio > float(self.cfg.tracking_success_speed_ratio)
        yaw_track_ok = torch.abs(wz - cmd_wz) < float(self.cfg.tracking_success_yaw_error)
        height_ok = (base_height > float(self.cfg.tracking_success_min_height)) & (base_height < float(self.cfg.tracking_success_max_height))
        stable_ok = roll_pitch_mag < float(self.cfg.tracking_success_max_roll_pitch)
        tracking_success = alive_timeout & lin_track_ok & yaw_track_ok & height_ok & stable_ok
        tracking_fail_timeout = alive_timeout & (~tracking_success)
        success = tracking_success

        event_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        event_reward = torch.where(is_fallen, event_reward + float(self.cfg.rew_fall), event_reward)
        event_reward = torch.where(tracking_success, event_reward + float(self.cfg.rew_timeout_alive), event_reward)

        reward_raw = continuous + event_reward

        projected_return = self.episode_return + reward_raw
        no_event = event_reward.abs() < 1e-6

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

        done = terminated | truncated

        if done.any():
            done_f = done.float()
            self.total_done_episodes += done_f.sum()
            self.total_success_episodes += success.float().sum()
            self.total_fall_episodes += is_fallen.float().sum()
            self.total_timeout_episodes += truncated.float().sum()
            self.total_alive_timeout_episodes += alive_timeout.float().sum()
            self.total_tracking_success_episodes += tracking_success.float().sum()
            self.total_tracking_fail_timeout_episodes += tracking_fail_timeout.float().sum()

            for stage in range(len(self.cfg.stage_thresholds)):
                mask = done & (self.env_stage == stage)
                if mask.any():
                    self.stage_done_counter[stage] += mask.float().sum()
                    self.stage_success_counter[stage] += (mask & success).float().sum()
                    self.stage_fall_counter[stage] += (mask & is_fallen).float().sum()
                    self.stage_timeout_counter[stage] += (mask & truncated).float().sum()
                    self.stage_alive_timeout_counter[stage] += (mask & alive_timeout).float().sum()
                    self.stage_tracking_fail_timeout_counter[stage] += (mask & tracking_fail_timeout).float().sum()

        total_done_safe = torch.clamp(self.total_done_episodes, min=1.0)

        stage_success_rate = self.stage_success_counter / torch.clamp(self.stage_done_counter, min=1.0)
        stage_fall_rate = self.stage_fall_counter / torch.clamp(self.stage_done_counter, min=1.0)
        stage_alive_timeout_rate = self.stage_alive_timeout_counter / torch.clamp(self.stage_done_counter, min=1.0)
        stage_tracking_fail_timeout_rate = self.stage_tracking_fail_timeout_counter / torch.clamp(self.stage_done_counter, min=1.0)
        active_stage_idx = int(max(0, min(self.stage_from_global_steps(int(self.global_steps)), len(self.cfg.stage_thresholds) - 1)))

        info = {
            "reward_components": {
                "R_Cmd_Lin": self._mean_detached(float(self.cfg.w_cmd_lin) * r_cmd_lin),
                "R_Forward_Ratio": self._mean_detached(float(self.cfg.w_forward_ratio) * r_forward_ratio),
                "P_Under_Speed": self._mean_detached(float(self.cfg.w_under_speed) * p_under_speed),
                "R_Cmd_Yaw": self._mean_detached(float(self.cfg.w_cmd_yaw) * r_cmd_yaw),
                "P_Lateral_Vel": self._mean_detached(float(self.cfg.w_lateral_vel) * p_lateral_vel),
                "R_Tracking_Recovery": self._mean_detached(float(self.cfg.w_tracking_recovery) * r_tracking_recovery),
                "R_Post_Push_Stability": self._mean_detached(float(self.cfg.w_post_push_stability) * r_post_push_stability),
                "R_Push_Survival": self._mean_detached(float(self.cfg.w_push_survival) * r_push_survival),
                "R_Upright": self._mean_detached(float(self.cfg.w_upright) * r_upright),
                "R_Height": self._mean_detached(float(self.cfg.w_height) * r_height),
                "P_Low_Height": self._mean_detached(float(self.cfg.w_low_height) * p_low_height),
                "P_Base_Ang": self._mean_detached(float(self.cfg.w_base_ang_vel) * p_base_ang),
                "P_Z_Vel": self._mean_detached(float(self.cfg.w_z_vel) * p_z_vel),
                "R_Alive": self._mean_detached(float(self.cfg.w_alive) * r_alive),
                "R_Phase_Contact": self._mean_detached(float(self.cfg.w_phase_contact) * r_phase_contact),
                "R_Air_Time": self._mean_detached(float(self.cfg.w_air_time) * r_air_time),
                "R_Clearance": self._mean_detached(float(self.cfg.w_clearance) * r_clearance),
                "R_Contact_Count": self._mean_detached(float(self.cfg.w_contact_count) * r_contact_count),
                "P_Foot_Slip": self._mean_detached(float(self.cfg.w_foot_slip) * p_foot_slip),
                "P_Impact": self._mean_detached(float(self.cfg.w_impact) * p_impact),
                "P_Torque": self._mean_detached(float(self.cfg.w_torque) * p_torque),
                "P_Energy": self._mean_detached(float(self.cfg.w_energy) * p_energy),
                "P_Action_Rate": self._mean_detached(float(self.cfg.w_action_rate) * p_action_rate),
                "P_Action_Mag": self._mean_detached(float(self.cfg.w_action_mag) * p_action_mag),
                "P_Joint_Vel": self._mean_detached(float(self.cfg.w_joint_vel) * p_joint_vel),
                "P_Joint_Limit": self._mean_detached(float(self.cfg.w_joint_limit) * p_joint_limit),
                "Continuous": self._mean_detached(continuous),
                "Event": self._mean_detached(event_reward),
                "Total": self._mean_detached(reward),
            },
            "events": {
                "Success_Rate": self._mean_detached(success.float()),
                "Tracking_Success_Rate": self._mean_detached(tracking_success.float()),
                "Alive_Timeout_Rate": self._mean_detached(alive_timeout.float()),
                "Tracking_Fail_Timeout_Rate": self._mean_detached(tracking_fail_timeout.float()),
                "Fall_Rate": self._mean_detached(is_fallen.float()),
                "Timeout_Rate": self._mean_detached(truncated.float()),
                "Done_Rate": self._mean_detached(done.float()),
                "Episode_Success_Total_Rate": self.total_success_episodes / total_done_safe,
                "Episode_Tracking_Success_Total_Rate": self.total_tracking_success_episodes / total_done_safe,
                "Episode_Alive_Timeout_Total_Rate": self.total_alive_timeout_episodes / total_done_safe,
                "Episode_Tracking_Fail_Timeout_Total_Rate": self.total_tracking_fail_timeout_episodes / total_done_safe,
                "Episode_Fall_Total_Rate": self.total_fall_episodes / total_done_safe,
                "Episode_Timeout_Total_Rate": self.total_timeout_episodes / total_done_safe,
            },
            "telemetry": {
                "Curriculum_K": self._float_tensor(self.curriculum_k()),
                "Command_Stage": self._mean_detached(self.env_stage.float()),
                "Cmd_Vx": self._mean_detached(cmd_vx),
                "Cmd_Vy": self._mean_detached(cmd_vy),
                "Cmd_Wz": self._mean_detached(cmd_wz),
                "Actual_Vx": self._mean_detached(vx),
                "Actual_Vy": self._mean_detached(vy),
                "Actual_Wz": self._mean_detached(wz),
                "Actual_Forward": self._mean_detached(actual_forward),
                "Cmd_Speed_Abs": self._mean_detached(cmd_speed_abs),
                "Speed_Ratio": self._mean_detached(speed_ratio),
                "Forward_Ratio": self._mean_detached(forward_ratio),
                "Under_Ratio": self._mean_detached(under_ratio),
                "Cmd_Lin_Speed_Gate": self._mean_detached(cmd_lin_gate),
                "Stability_Speed_Gate": self._mean_detached(stability_reward_gate),
                "Tracking_Error": self._mean_detached(tracking_error),
                "Base_Height": self._mean_detached(base_height),
                "RollPitch_Mag": self._mean_detached(roll_pitch_mag),
                "Contact_Count": self._mean_detached(contact_count),
                "FL_Contact": self._mean_detached(contact[:, 0]),
                "FR_Contact": self._mean_detached(contact[:, 1]),
                "RL_Contact": self._mean_detached(contact[:, 2]),
                "RR_Contact": self._mean_detached(contact[:, 3]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Impact_Max": self._mean_detached(normal_force.max(dim=-1)[0]),
                "Foot_Slip_Raw": self._mean_detached(raw_foot_slip),
                "Friction": self._mean_detached(self.dr_friction),
                "Payload_Mass": self._mean_detached(self.dr_payload_mass),
                "COM_Shift_Norm": self._mean_detached(torch.norm(self.dr_com_shift, dim=-1)),
                "Motor_Strength_Mean": self._mean_detached(self.dr_motor_strength),
                "Motor_Strength_Min": self.dr_motor_strength.detach().min(),
                "Push_Active_Rate": self._mean_detached(self.push_active.float()),
                "Push_Force_Body_Norm": self._mean_detached(torch.norm(self.push_force_b, dim=-1)),
                "Post_Push_Timer": self._mean_detached(self.post_push_timer),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Global_Steps": self._float_tensor(float(self.global_steps)),
            },
            "curriculum": {
                "Stage0_Ratio": self._mean_detached((self.env_stage == 0).float()),
                "Stage1_Ratio": self._mean_detached((self.env_stage == 1).float()),
                "Stage2_Ratio": self._mean_detached((self.env_stage == 2).float()),
                "Stage3_Ratio": self._mean_detached((self.env_stage == 3).float()),
                "Stage4_Ratio": self._mean_detached((self.env_stage == 4).float()),
                "Stage5_Ratio": self._mean_detached((self.env_stage == 5).float()),
                "Stage_Success_Rate_Mean": self._mean_detached(stage_success_rate),
                "Stage_Tracking_Success_Rate_Mean": self._mean_detached(stage_success_rate),
                "Stage_Alive_Timeout_Rate_Mean": self._mean_detached(stage_alive_timeout_rate),
                "Stage_Tracking_Fail_Timeout_Rate_Mean": self._mean_detached(stage_tracking_fail_timeout_rate),
                "Stage_Fall_Rate_Mean": self._mean_detached(stage_fall_rate),
                "Active_Stage": self._float_tensor(float(active_stage_idx)),
                "Active_Stage_Tracking_Success_Rate": stage_success_rate[active_stage_idx],
                "Active_Stage_Alive_Timeout_Rate": stage_alive_timeout_rate[active_stage_idx],
                "Active_Stage_Tracking_Fail_Timeout_Rate": stage_tracking_fail_timeout_rate[active_stage_idx],
                "Active_Stage_Fall_Rate": stage_fall_rate[active_stage_idx],
            },
            "debug": {
                "Returned_Obs_Dim": self._float_tensor(float(self.num_observations)),
                "Actor_Single_Obs_Dim": self._float_tensor(float(self.single_actor_obs_dim)),
                "Actor_Obs_Dim": self._float_tensor(float(self.actor_obs_dim)),
                "Privileged_Obs_Dim": self._float_tensor(float(self.privileged_obs_dim)),
                "Teacher_Obs_Dim": self._float_tensor(float(self.teacher_obs_dim)),
                "Teacher_Mode": self._float_tensor(1.0 if self.cfg.teacher_mode else 0.0),
                "Reward_Min": reward.detach().min(),
                "Reward_Max": reward.detach().max(),
                "Continuous_Min": continuous.detach().min(),
                "Continuous_Max": continuous.detach().max(),
                "Event_Min": event_reward.detach().min(),
                "Event_Max": event_reward.detach().max(),
                "Base_Height_Min": base_height.detach().min(),
                "Base_Height_Max": base_height.detach().max(),
                "JointVel_Max": joint_vel_abs_max.detach().max(),
            },
        }

        return reward, terminated, truncated, info


# Backward-compatible aliases.
QuadrupedSim2RealEnv = Go2Task4Env
Go2Task4Sim2RealEnv = Go2Task4Env
Go2RMAEnv = Go2Task4Env
UnitreeGo2Task4Env = Go2Task4Env