# Copyright (c) 2026
# Unitree Go2 Task3: autonomous navigation + obstacle avoidance + running environment.
#
# Strict refactor notes:
# 1. This file defines IsaacLab Go2 environment only.
# 2. It does not start AppLauncher.
# 3. Analytical navigation world is Task3World.
# 4. Obstacles are not real prims. Lidar, collision, target, risk, and privileged
#    features are computed by Task3World GPU tensors.
# 5. Actor obs = 257.
# 6. Privileged obs = 325 = actor obs 257 + world privileged 68.
# 7. Training frame-stack is handled later in task3_train.py.

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

from go2_rl.tasks.task3.task3_config import Task3Config
from go2_rl.tasks.task3.task3_world import Task3World


def make_go2_task3_scene_cfg(cfg: Task3Config):
    """Build IsaacLab scene config for Task3.

    The analytical obstacles are not spawned as prims. The scene contains:
        1. default ground plane,
        2. Unitree Go2,
        3. foot contact sensor,
        4. dome light.
    """

    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    try:
        robot_cfg.spawn.activate_contact_sensors = True
    except Exception:
        pass

    @configclass
    class Go2Task3SceneCfg(InteractiveSceneCfg):
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
            spawn=sim_utils.DomeLightCfg(intensity=2800.0),
        )

    return Go2Task3SceneCfg(num_envs=int(cfg.num_envs), env_spacing=float(cfg.env_spacing))


class Go2Task3Env(gym.Env):
    """Go2 Task3: autonomous navigation + obstacle avoidance + running.

    Actor observation layout, dim = 257:
        base_lin_vel_b       3
        base_ang_vel_b       3
        projected_gravity_b  3
        target_obs           3
        target_speed         1
        progress_ema         1
        q_err               12
        qd                  12
        last_action         12
        action_delta        12
        foot_contact         4
        lidar               90
        lidar_delta         90
        risk_features        8
        base_height          1
        sin_phase            1
        cos_phase            1

    Privileged obs layout, dim = 325:
        actor obs           257
        world privileged     68
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task3Config):
        super().__init__()

        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.dt = float(cfg.control_dt)

        self.world = Task3World(
            cfg=cfg.world_cfg,
            num_envs=cfg.num_envs,
            device=cfg.device,
        )

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
        self.scene = InteractiveScene(make_go2_task3_scene_cfg(cfg))
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

        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_privileged_obs),),
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
        self.stuck_counter = torch.zeros(n, dtype=torch.long, device=self.device)

        self.last_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.last_base_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_acc_obs = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.prev_lidar = torch.ones(
            (n, int(cfg.world_cfg.num_lidar_rays)),
            dtype=torch.float32,
            device=self.device,
        )

        self.progress_ema = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.phase = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.prev_foot_contact = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        self.feet_air_time = torch.zeros((n, 4), dtype=torch.float32, device=self.device)

        # Episode-level counters for stable logging.
        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_collision_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_fall_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_out_of_bounds_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        if bool(cfg.print_debug_info):
            self._print_debug_info()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _get_env_origins(self) -> torch.Tensor:
        if hasattr(self.scene, "env_origins"):
            return self.scene.env_origins.to(self.device)

        try:
            return self.scene._default_env_origins.to(self.device)
        except Exception:
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _joint_ids(self, names: Iterable[str]) -> List[int]:
        names = list(names)
        missing = [name for name in names if name not in self.robot_joint_names]
        if missing:
            raise RuntimeError(
                f"[Go2Task3Env] Missing joints: {missing}\n"
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
                f"[Go2Task3Env] Missing body links: {missing}\n"
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
                f"[Go2Task3Env] Missing contact links: {missing}\n"
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

    @staticmethod
    def _quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device).view(-1)
        quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=self.device)
        quat[:, 0] = torch.cos(yaw * 0.5)
        quat[:, 3] = torch.sin(yaw * 0.5)
        return quat

    def _root_pos_local(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        pos = root_pos_w.clone()
        pos[:, :2] = root_pos_w[:, :2] - self.env_origins[:, :2]
        return pos

    def _root_lin_vel_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "root_lin_vel_w"):
            return self.robot.data.root_lin_vel_w

        v_b = self.robot.data.root_lin_vel_b
        yaw = self._quat_yaw(self.robot.data.root_quat_w)
        c = torch.cos(yaw)
        s = torch.sin(yaw)

        vx = c * v_b[:, 0] - s * v_b[:, 1]
        vy = s * v_b[:, 0] + c * v_b[:, 1]
        return torch.stack([vx, vy, v_b[:, 2]], dim=-1)

    @staticmethod
    def _mean_detached(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().mean()

    @staticmethod
    def _float_tensor(value: float, device: str) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=device)

    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print(" [Go2Task3Env] Navigation / Obstacle Avoidance / Running Environment Initialized")
        print("=" * 120)
        print(f" num_envs           : {self.cfg.num_envs}")
        print(f" device             : {self.device}")
        print(f" sim_dt / decimation: {self.cfg.sim_dt} / {self.cfg.decimation}")
        print(f" control_dt         : {self.dt}")
        print(f" num_joints         : {self.robot.num_joints}")
        print(f" num_actions        : {self.cfg.num_actions}")
        print(f" num_observations   : {self.cfg.num_observations}")
        print(f" num_privileged_obs : {self.cfg.num_privileged_obs}")
        print(f" world priv dim     : {self.world.privileged_feature_dim()}")
        print(f" lidar rays         : {self.cfg.world_cfg.num_lidar_rays}")
        print(f" max_static_obs     : {self.cfg.world_cfg.max_static_obs}")
        print(f" max_dynamic_obs    : {self.cfg.world_cfg.max_dynamic_obs}")
        print(f" env_spacing        : {self.cfg.env_spacing}")
        print(f" action_joint_ids   : {self.action_joint_ids}")
        print(f" foot_body_ids      : {self.foot_body_ids}")
        print(f" contact_foot_ids   : {self.contact_foot_ids}")
        print(f" contact body names : {list(self.contact.body_names)}")
        print("=" * 120 + "\n")

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
    ) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return self._compute_obs(update_lidar_history=False), {}

        self._reset_idx(env_ids)
        obs = self._compute_obs(update_lidar_history=False)
        return obs, {}

    @torch.no_grad()
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        n = int(env_ids.numel())

        self.world.reset_envs(env_ids, global_steps=int(self.global_steps))

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = self.env_origins[env_ids, :2] + self.world.start_pos[env_ids]
        root_state[:, 2] = float(self.cfg.target_height)

        delta = self.world.target_pos[env_ids] - self.world.start_pos[env_ids]
        yaw_to_goal = torch.atan2(delta[:, 1], delta[:, 0])

        k = self.world.curriculum_k(int(self.global_steps))
        yaw_noise = float(self.cfg.init_yaw_noise_stage0) + k * (
            float(self.cfg.init_yaw_noise_stage5) - float(self.cfg.init_yaw_noise_stage0)
        )
        yaw = yaw_to_goal + torch.empty(n, dtype=torch.float32, device=self.device).uniform_(-yaw_noise, yaw_noise)

        root_state[:, 3:7] = self._yaw_to_quat(yaw)
        root_state[:, 7:13] = 0.0

        q = self.default_joint_pos_all[env_ids].clone()
        qd = torch.zeros_like(q)

        q[:, self.action_joint_ids_t] += torch.empty(
            (n, int(self.cfg.num_actions)),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-float(self.cfg.reset_joint_noise), float(self.cfg.reset_joint_noise))

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

        self.last_action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0

        self.last_base_vel[env_ids] = self.robot.data.root_lin_vel_b[env_ids]
        self.base_acc_obs[env_ids] = 0.0

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.stuck_counter[env_ids] = 0
        self.progress_ema[env_ids] = 0.0

        self.phase[env_ids] = torch.rand(n, dtype=torch.float32, device=self.device)
        self.prev_foot_contact[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0

        self.joint_position_targets[env_ids] = self.default_joint_pos_all[env_ids]

        root_pos_local = self._root_pos_local()
        yaw_all = self._quat_yaw(self.robot.data.root_quat_w)

        # Task3World lidar is full-batch because world obstacle tensors are [num_envs, ...].
        lidar = self.world.compute_lidar_tensors(root_pos_local, yaw_all, normalize=True)
        self.prev_lidar[env_ids] = lidar[env_ids]

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        actions = actions.to(device=self.device, dtype=torch.float32)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)

        if actions.shape != (self.num_envs, int(self.cfg.num_actions)):
            raise RuntimeError(
                f"[Go2Task3Env] action shape mismatch: got {tuple(actions.shape)}, "
                f"expected {(self.num_envs, int(self.cfg.num_actions))}"
            )

        actions = torch.clamp(actions, -1.0, 1.0)

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

        self.world.step_kinematics(float(self.dt))

        self.global_steps += self.num_envs
        self.episode_steps += 1

        speed_for_phase = torch.clamp(self.world.env_target_speed, 0.2, 2.0)
        self.phase = torch.remainder(
            self.phase + self.dt * (float(self.cfg.gait_freq_hz) + float(self.cfg.gait_freq_speed_gain) * speed_for_phase),
            1.0,
        )

        reward, terminated, truncated, info = self._compute_rewards_and_dones()

        obs_before_reset = self._compute_obs(update_lidar_history=True)

        self.episode_return += reward
        info["telemetry"]["Episode_Return"] = self.episode_return.detach().float().mean()

        done = terminated | truncated
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)

        obs = obs_before_reset
        if reset_ids.numel() > 0:
            info["terminal_observation"] = obs_before_reset[reset_ids].clone()
            self._reset_idx(reset_ids)

            reset_obs = self._compute_obs(update_lidar_history=False)
            obs[reset_ids] = reset_obs[reset_ids]

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Sensors / observation
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

    def _compute_base_height(self) -> torch.Tensor:
        # Task3 uses flat ground plane. Height is world z.
        return self.robot.data.root_pos_w[:, 2]

    def _compute_foot_heights(self) -> torch.Tensor:
        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        return foot_pos[:, :, 2]

    def _build_trot_contact_ref(self) -> torch.Tensor:
        phase_a = self.phase
        phase_b = torch.remainder(self.phase + 0.5, 1.0)

        stance_a = (phase_a < float(self.cfg.duty_factor)).float()
        stance_b = (phase_b < float(self.cfg.duty_factor)).float()

        # Foot order: FL, FR, RL, RR
        return torch.stack([stance_a, stance_b, stance_b, stance_a], dim=-1)

    def _compute_obs(self, update_lidar_history: bool = False) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        q = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        q_err = q - self.default_joint_pos

        contact, _ = self._get_foot_contact()

        root_pos_local = self._root_pos_local()
        yaw = self._quat_yaw(self.robot.data.root_quat_w)

        target_obs = self.world.get_target_obs(root_pos_local, yaw)
        target_speed = torch.clamp(self.world.env_target_speed / 2.0, 0.0, 2.0).unsqueeze(-1)

        lidar = self.world.compute_lidar_tensors(root_pos_local, yaw, normalize=True)
        lidar_delta = self.world.compute_lidar_delta(lidar, self.prev_lidar)

        if update_lidar_history:
            self.prev_lidar.copy_(lidar)

        risk_features = self.world.compute_risk_features(root_pos_local, yaw)

        action_delta = self.last_action - self.prev_action

        base_height = self._compute_base_height().unsqueeze(-1)
        sin_phase = torch.sin(2.0 * math.pi * self.phase).unsqueeze(-1)
        cos_phase = torch.cos(2.0 * math.pi * self.phase).unsqueeze(-1)

        obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                target_obs,
                target_speed,
                torch.clamp(self.progress_ema / 2.0, -2.0, 2.0).unsqueeze(-1),
                q_err,
                qd,
                self.last_action,
                action_delta,
                contact,
                lidar,
                lidar_delta,
                risk_features,
                base_height,
                sin_phase,
                cos_phase,
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"[Go2Task3Env] Observation dim mismatch: got {obs.shape[-1]}, "
                f"expected {self.cfg.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def compute_privileged_obs(self) -> torch.Tensor:
        actor_obs = self._compute_obs(update_lidar_history=False)

        root_pos_local = self._root_pos_local()
        yaw = self._quat_yaw(self.robot.data.root_quat_w)

        world_priv = self.world.make_privileged_features(root_pos_local, yaw)
        priv = torch.cat([actor_obs, world_priv], dim=-1)

        if priv.shape[-1] != int(self.cfg.num_privileged_obs):
            raise RuntimeError(
                f"[Go2Task3Env] Privileged obs dim mismatch: got {priv.shape[-1]}, "
                f"expected {self.cfg.num_privileged_obs}"
            )

        return torch.nan_to_num(
            torch.clamp(priv, -20.0, 20.0),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

    # -------------------------------------------------------------------------
    # Reward / termination
    # -------------------------------------------------------------------------
    def _compute_rewards_and_dones(self):
        root_pos_local = self._root_pos_local()
        yaw = self._quat_yaw(self.robot.data.root_quat_w)

        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_lin_vel_w = self._root_lin_vel_w()
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        vx_b = base_lin_vel_b[:, 0]
        vy_b = base_lin_vel_b[:, 1]
        vz_b = base_lin_vel_b[:, 2]

        wx = base_ang_vel[:, 0]
        wy = base_ang_vel[:, 1]
        wz = base_ang_vel[:, 2]

        base_height = self._compute_base_height()

        base_acc = (base_lin_vel_b - self.last_base_vel) / max(self.dt, 1e-6)
        self.base_acc_obs.copy_(base_acc)
        self.last_base_vel.copy_(base_lin_vel_b)

        q = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        q_err = q - self.default_joint_pos

        contact, normal_force = self._get_foot_contact()
        contact = contact.float()
        contact_count = contact.sum(dim=-1)

        ref_contact = self._build_trot_contact_ref().float()

        foot_height = self._compute_foot_heights()
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]

        # ----------------------------- Goal geometry -----------------------------
        target_vec = self.world.target_pos - root_pos_local[:, :2]
        dist_to_goal = torch.norm(target_vec, dim=-1)
        dir_to_goal = target_vec / torch.clamp(dist_to_goal.unsqueeze(-1), min=1e-6)

        target_angle = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        rel_angle = torch.atan2(
            torch.sin(target_angle - yaw),
            torch.cos(target_angle - yaw),
        )

        heading_cos = torch.cos(rel_angle)
        heading_gate = torch.clamp(heading_cos, 0.0, 1.0)

        actual_along_goal = torch.sum(base_lin_vel_w[:, :2] * dir_to_goal, dim=-1)
        lateral_speed = torch.abs(vy_b)

        target_speed = self.world.env_target_speed
        success_radius = self.world.success_radius_tensor()

        # ----------------------------- Progress -----------------------------
        progress = self.world.compute_progress(root_pos_local, dt=self.dt)

        progress_clamped = torch.clamp(
            progress,
            -float(self.cfg.progress_clip_neg),
            float(self.cfg.progress_clip_pos),
        )

        self.progress_ema.mul_(0.90)
        self.progress_ema.add_(progress_clamped, alpha=0.10)

        r_progress = torch.tanh(float(self.cfg.progress_scale) * progress_clamped) * heading_gate

        p_backtrack = -torch.clamp(
            -progress_clamped,
            min=0.0,
            max=float(self.cfg.progress_clip_neg),
        )

        r_goal_distance = torch.exp(-float(self.cfg.sigma_goal_distance) * dist_to_goal)
        r_goal_heading = torch.exp(-float(self.cfg.sigma_heading) * torch.square(rel_angle))

        # ----------------------------- Finish zone -----------------------------
        finish_outer = float(self.cfg.finish_outer_radius_scale) * success_radius
        finish_inner = float(self.cfg.finish_inner_radius_scale) * success_radius

        finish_gate = torch.clamp(
            (finish_outer - dist_to_goal) / torch.clamp(finish_outer - finish_inner, min=1e-6),
            0.0,
            1.0,
        )

        forward_finish_ratio = torch.clamp(
            actual_along_goal / max(float(self.cfg.hesitation_speed), 1e-6),
            0.0,
            1.0,
        )

        r_finish_pull = finish_gate * heading_gate * forward_finish_ratio

        outside_success = (dist_to_goal > success_radius).float()
        p_finish_hesitation = -finish_gate * outside_success * heading_gate * torch.clamp(
            (float(self.cfg.hesitation_speed) - actual_along_goal) / max(float(self.cfg.hesitation_speed), 1e-6),
            0.0,
            1.0,
        )

        required_speed = torch.clamp(
            0.55 * target_speed,
            min=float(self.cfg.min_forward_speed),
            max=0.42,
        )

        far_from_goal = (dist_to_goal > float(self.cfg.finish_outer_radius_scale) * success_radius).float()
        p_under_speed = -far_from_goal * heading_gate * torch.clamp(
            (required_speed - actual_along_goal) / torch.clamp(required_speed, min=1e-6),
            0.0,
            1.0,
        )

        episode_frac = self.episode_steps.float() / max(float(self.cfg.max_episode_length), 1.0)
        deadline_gate = torch.clamp(
            (episode_frac - float(self.cfg.deadline_start_frac))
            / max(1.0 - float(self.cfg.deadline_start_frac), 1e-6),
            0.0,
            1.0,
        )

        normalized_remaining = torch.clamp(
            (dist_to_goal - success_radius) / torch.clamp(self.world.env_target_speed * 3.0, min=0.5),
            min=0.0,
            max=3.0,
        )
        p_deadline = -deadline_gate * normalized_remaining

        # ----------------------------- Lidar / obstacle / safety -----------------------------
        lidar = self.world.compute_lidar_tensors(root_pos_local, yaw, normalize=True)
        lidar_delta = self.world.compute_lidar_delta(lidar, self.prev_lidar)
        risk_features = self.world.compute_risk_features(root_pos_local, yaw)

        front_clearance_norm = risk_features[:, 0]
        left_clearance_norm = risk_features[:, 1]
        right_clearance_norm = risk_features[:, 2]
        collision_risk = risk_features[:, 7]

        obstacle_gate = torch.clamp(self.world.env_stage.float() / 2.0, 0.0, 1.0)

        min_signed, min_static_signed, min_dynamic_signed = self.world.obstacle_signed_distance(root_pos_local)

        obstacle_margin_risk = torch.clamp(
            (float(self.cfg.safe_obstacle_distance) - min_signed)
            / max(float(self.cfg.safe_obstacle_distance) - float(self.cfg.critical_obstacle_distance), 1e-6),
            0.0,
            2.0,
        )

        p_obstacle_risk = -obstacle_gate * torch.square(torch.maximum(collision_risk, obstacle_margin_risk))

        r_front_clearance = obstacle_gate * torch.clamp(front_clearance_norm, 0.0, 1.0)

        approaching = torch.clamp(-lidar_delta.min(dim=-1)[0], 0.0, 1.0)
        p_ttc_proxy = -obstacle_gate * approaching * torch.clamp(collision_risk + 0.25, 0.0, 1.0)

        side_preference = torch.tanh(3.0 * (right_clearance_norm - left_clearance_norm))
        avoid_turn_hint = side_preference * torch.clamp(1.0 - front_clearance_norm, 0.0, 1.0)

        r_active_avoid_heading = obstacle_gate * torch.clamp(
            wz * avoid_turn_hint,
            min=-0.5,
            max=0.5,
        )

        boundary_dist = self.world.boundary_signed_distance(root_pos_local)
        p_boundary = -torch.square(
            torch.clamp(
                (float(self.cfg.world_cfg.warning_margin) - boundary_dist)
                / max(float(self.cfg.world_cfg.warning_margin), 1e-6),
                0.0,
                2.0,
            )
        )

        # ----------------------------- Goal speed tracking under safety gate -----------------------------
        safe_speed_gate = torch.clamp(1.0 - 0.70 * collision_risk * obstacle_gate, 0.25, 1.0)
        desired_along_speed = target_speed * safe_speed_gate
        speed_error = actual_along_goal - desired_along_speed

        r_goal_speed = (
            torch.exp(-float(self.cfg.sigma_speed) * torch.square(speed_error))
            * heading_gate
            * torch.clamp(1.0 - 1.5 * lateral_speed, 0.0, 1.0)
        )

        # ----------------------------- Running / gait -----------------------------
        move_gate = torch.clamp(
            actual_along_goal / torch.clamp(target_speed, min=0.25),
            0.0,
            1.0,
        )
        move_gate = move_gate * heading_gate * torch.clamp(1.0 - collision_risk, 0.20, 1.0)
        gait_gate = 0.20 + 0.80 * move_gate

        r_phase_contact = (1.0 - torch.mean(torch.abs(contact - ref_contact), dim=-1)) * gait_gate

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)
        self.feet_air_time += self.dt

        r_air_time = (
            torch.sum(
                torch.clamp(
                    self.feet_air_time - float(self.cfg.air_time_target),
                    min=0.0,
                    max=0.45,
                )
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
                * torch.exp(-float(self.cfg.sigma_clearance) * torch.abs(foot_height - float(self.cfg.foot_clearance_target))),
                dim=-1,
            )
            * move_gate
        )

        r_contact_count = torch.exp(-torch.square(contact_count - 2.6)) * gait_gate

        raw_foot_slip = torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * contact, dim=-1)
        p_foot_slip = -torch.clamp(raw_foot_slip, max=8.0)

        # ----------------------------- Stability / regularization -----------------------------
        r_upright = (1.0 - projected_gravity[:, 2]) * 0.5

        r_height = torch.exp(-float(self.cfg.sigma_height) * torch.square(base_height - float(self.cfg.target_height)))

        p_base_ang = -torch.clamp(torch.square(wx) + torch.square(wy), max=6.0)
        p_base_acc = -torch.clamp(torch.sum(torch.square(base_acc), dim=-1), max=30.0)
        p_z_vel = -torch.abs(vz_b)

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

        torques = getattr(self.robot.data, "applied_torque", torch.zeros_like(self.robot.data.joint_vel))
        tau = torques[:, self.action_joint_ids_t]

        p_torque = -torch.clamp(torch.mean(torch.square(tau), dim=-1), max=40.0)
        p_energy = -torch.clamp(torch.mean(torch.abs(tau * qd), dim=-1), max=20.0)

        distance_2d = torch.clamp(torch.norm(base_lin_vel_w[:, :2], dim=-1), min=0.05)
        p_specific_energy = -torch.clamp(
            torch.mean(torch.abs(tau * qd), dim=-1) / distance_2d,
            max=40.0,
        )

        r_alive = torch.ones_like(vx_b)

        # ----------------------------- Terminations and event rewards -----------------------------
        joint_vel_abs_max = torch.abs(self.robot.data.joint_vel).max(dim=-1)[0]

        is_fallen = (
            (base_height < float(self.cfg.fall_height))
            | (base_height > float(self.cfg.jump_height))
            | (torch.norm(projected_gravity[:, :2], dim=-1) > float(self.cfg.bad_orientation_xy))
            | (~torch.isfinite(base_height))
            | (~torch.isfinite(self.robot.data.joint_pos).all(dim=-1))
            | (joint_vel_abs_max > float(self.cfg.max_joint_vel_abs))
        )

        terminated_world, truncated_world, event_reward, event_info = self.world.check_terminations(
            root_pos_local,
            is_fallen=is_fallen,
        )

        success = event_info["success"]
        collision = event_info["collision"]
        out_of_bounds = event_info["out_of_bounds"]
        timeout = event_info["timeout"]

        terminated = terminated_world
        truncated = truncated_world

        # ----------------------------- Reward total -----------------------------
        continuous_raw = (
            float(self.cfg.w_progress) * r_progress
            + float(self.cfg.w_goal_speed) * r_goal_speed
            + float(self.cfg.w_goal_heading) * r_goal_heading
            + float(self.cfg.w_goal_distance) * r_goal_distance
            + float(self.cfg.w_finish_pull) * r_finish_pull
            + float(self.cfg.w_finish_hesitation) * p_finish_hesitation
            + float(self.cfg.w_under_speed) * p_under_speed
            + float(self.cfg.w_backtrack) * p_backtrack
            + float(self.cfg.w_deadline) * p_deadline
            + float(self.cfg.w_obstacle_risk) * p_obstacle_risk
            + float(self.cfg.w_front_clearance) * r_front_clearance
            + float(self.cfg.w_ttc_proxy) * p_ttc_proxy
            + float(self.cfg.w_active_avoid_heading) * r_active_avoid_heading
            + float(self.cfg.w_boundary) * p_boundary
            + float(self.cfg.w_phase_contact) * r_phase_contact
            + float(self.cfg.w_air_time) * r_air_time
            + float(self.cfg.w_clearance) * r_clearance
            + float(self.cfg.w_contact_count) * r_contact_count
            + float(self.cfg.w_foot_slip) * p_foot_slip
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
            + float(self.cfg.w_torque) * p_torque
            + float(self.cfg.w_energy) * p_energy
            + float(self.cfg.w_specific_energy) * p_specific_energy
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

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
            self.total_collision_episodes += collision.float().sum()
            self.total_fall_episodes += is_fallen.float().sum()
            self.total_timeout_episodes += timeout.float().sum()
            self.total_out_of_bounds_episodes += out_of_bounds.float().sum()

        success_total_safe = torch.clamp(self.total_done_episodes, min=1.0)

        world_stats = self.world.world_stats(root_pos_local)

        info = {
            "reward_components": {
                "R_Progress": self._mean_detached(float(self.cfg.w_progress) * r_progress),
                "R_Goal_Speed": self._mean_detached(float(self.cfg.w_goal_speed) * r_goal_speed),
                "R_Goal_Heading": self._mean_detached(float(self.cfg.w_goal_heading) * r_goal_heading),
                "R_Goal_Distance": self._mean_detached(float(self.cfg.w_goal_distance) * r_goal_distance),
                "R_Finish_Pull": self._mean_detached(float(self.cfg.w_finish_pull) * r_finish_pull),
                "P_Finish_Hesitation": self._mean_detached(float(self.cfg.w_finish_hesitation) * p_finish_hesitation),
                "P_Under_Speed": self._mean_detached(float(self.cfg.w_under_speed) * p_under_speed),
                "P_Backtrack": self._mean_detached(float(self.cfg.w_backtrack) * p_backtrack),
                "P_Deadline": self._mean_detached(float(self.cfg.w_deadline) * p_deadline),
                "P_Obstacle_Risk": self._mean_detached(float(self.cfg.w_obstacle_risk) * p_obstacle_risk),
                "R_Front_Clearance": self._mean_detached(float(self.cfg.w_front_clearance) * r_front_clearance),
                "P_TTC_Proxy": self._mean_detached(float(self.cfg.w_ttc_proxy) * p_ttc_proxy),
                "R_Active_Avoid_Heading": self._mean_detached(float(self.cfg.w_active_avoid_heading) * r_active_avoid_heading),
                "P_Boundary": self._mean_detached(float(self.cfg.w_boundary) * p_boundary),
                "R_Phase_Contact": self._mean_detached(float(self.cfg.w_phase_contact) * r_phase_contact),
                "R_Air_Time": self._mean_detached(float(self.cfg.w_air_time) * r_air_time),
                "R_Clearance": self._mean_detached(float(self.cfg.w_clearance) * r_clearance),
                "R_Contact_Count": self._mean_detached(float(self.cfg.w_contact_count) * r_contact_count),
                "P_Foot_Slip": self._mean_detached(float(self.cfg.w_foot_slip) * p_foot_slip),
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
                "P_Torque": self._mean_detached(float(self.cfg.w_torque) * p_torque),
                "P_Energy": self._mean_detached(float(self.cfg.w_energy) * p_energy),
                "P_Specific_Energy": self._mean_detached(float(self.cfg.w_specific_energy) * p_specific_energy),
                "Continuous": self._mean_detached(continuous),
                "Event": self._mean_detached(event_reward),
                "Total": self._mean_detached(reward),
            },
            "events": {
                "Success_Rate": self._mean_detached(success.float()),
                "Collision_Rate": self._mean_detached(collision.float()),
                "Static_Collision_Rate": self._mean_detached(event_info["static_collision"].float()),
                "Dynamic_Collision_Rate": self._mean_detached(event_info["dynamic_collision"].float()),
                "Fall_Rate": self._mean_detached(is_fallen.float()),
                "Out_Of_Bounds_Rate": self._mean_detached(out_of_bounds.float()),
                "Timeout_Rate": self._mean_detached(truncated.float()),
                "Done_Rate": self._mean_detached(done.float()),
                "Episode_Success_Total_Rate": self.total_success_episodes / success_total_safe,
                "Episode_Collision_Total_Rate": self.total_collision_episodes / success_total_safe,
                "Episode_Fall_Total_Rate": self.total_fall_episodes / success_total_safe,
                "Episode_Timeout_Total_Rate": self.total_timeout_episodes / success_total_safe,
            },
            "telemetry": {
                "Curriculum_K": self._float_tensor(self.world.curriculum_k(int(self.global_steps)), self.device),
                "Command_Stage": self._mean_detached(self.world.env_stage.float()),
                "Target_Speed": self._mean_detached(target_speed),
                "Actual_Along_Goal": self._mean_detached(actual_along_goal),
                "Progress": self._mean_detached(progress),
                "Progress_EMA": self._mean_detached(self.progress_ema),
                "Distance_To_Goal": self._mean_detached(dist_to_goal),
                "Success_Radius": self._mean_detached(success_radius),
                "Heading_Error": self._mean_detached(torch.abs(rel_angle)),
                "Heading_Cos": self._mean_detached(heading_cos),
                "Collision_Risk": self._mean_detached(collision_risk),
                "Front_Clearance_Norm": self._mean_detached(front_clearance_norm),
                "Min_Signed_Distance": self._mean_detached(min_signed),
                "Min_Static_Signed_Distance": self._mean_detached(min_static_signed),
                "Min_Dynamic_Signed_Distance": self._mean_detached(min_dynamic_signed),
                "Boundary_Distance": self._mean_detached(boundary_dist),
                "Base_Height": self._mean_detached(base_height),
                "Actual_Vx_Body": self._mean_detached(vx_b),
                "Actual_Vy_Body": self._mean_detached(vy_b),
                "Actual_Wz_Body": self._mean_detached(wz),
                "Contact_Count": self._mean_detached(contact_count),
                "FL_Contact": self._mean_detached(contact[:, 0]),
                "FR_Contact": self._mean_detached(contact[:, 1]),
                "RL_Contact": self._mean_detached(contact[:, 2]),
                "RR_Contact": self._mean_detached(contact[:, 3]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Static_Count": self._float_tensor(world_stats.get("Mean_Static_Count", 0.0), self.device),
                "Dynamic_Count": self._float_tensor(world_stats.get("Mean_Dynamic_Count", 0.0), self.device),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Global_Steps": self._float_tensor(float(self.global_steps), self.device),
            },
            "debug": {
                "Obs_Dim": self._float_tensor(float(self.cfg.num_observations), self.device),
                "Privileged_Obs_Dim": self._float_tensor(float(self.cfg.num_privileged_obs), self.device),
                "World_Priv_Dim": self._float_tensor(float(self.world.privileged_feature_dim()), self.device),
                "Reward_Min": reward.detach().min(),
                "Reward_Max": reward.detach().max(),
                "Continuous_Min": continuous.detach().min(),
                "Continuous_Max": continuous.detach().max(),
                "Event_Min": event_reward.detach().min(),
                "Event_Max": event_reward.detach().max(),
                "Base_Height_Min": base_height.detach().min(),
                "Base_Height_Max": base_height.detach().max(),
                "JointVel_Max": joint_vel_abs_max.detach().max(),
                "Lidar_Min": lidar.detach().min(),
                "Lidar_Max": lidar.detach().max(),
            },
        }

        return reward, terminated, truncated, info


# Backward-compatible aliases.
Go2NavigationEnv = Go2Task3Env
Go2Task3NavigationEnv = Go2Task3Env
UnitreeGo2Task3Env = Go2Task3Env
