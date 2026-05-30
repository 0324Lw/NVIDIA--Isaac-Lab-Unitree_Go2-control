# Copyright (c) 2026
# Unitree Go2 Task2: multi-terrain / multi-material locomotion environment.
#
# Strict refactor notes:
# 1. This file only defines the IsaacLab environment logic.
# 2. It does not start AppLauncher.
# 3. It imports Task2Config from task2_config.py.
# 4. It imports Task2World / TerrainCurriculum from task2_world.py.
# 5. Single-frame actor obs = 87.
# 6. Single-frame privileged obs = 178 = actor obs 87 + terrain privileged 91.
# 7. Training frame stack is handled by common/go2_skrl_wrappers.py.
# 8. Info values are mostly GPU tensors to reduce CPU synchronization during training.

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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

try:
    from isaaclab_assets import UNITREE_GO2_CFG
except Exception:
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from go2_rl.tasks.task2.task2_config import Task2Config
from go2_rl.tasks.task2.task2_world import Task2World, TerrainCurriculum


def make_go2_task2_scene_cfg(cfg: Task2Config, world: Task2World):
    """Build InteractiveSceneCfg for Task2.

    This is a function rather than a global class because the terrain generator
    is created from the runtime Task2World instance.
    """

    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    try:
        robot_cfg.spawn.activate_contact_sensors = True
    except Exception:
        pass

    @configclass
    class Go2Task2SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = 0.0

        terrain: TerrainImporterCfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=world.generator_cfg,
            max_init_terrain_level=int(cfg.terrain_cfg.num_levels) - 1,
            collision_group=-1,
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

    return Go2Task2SceneCfg(num_envs=int(cfg.num_envs), env_spacing=0.0)


class Go2Task2Env(gym.Env):
    """Unitree Go2 multi-terrain locomotion environment.

    Gymnasium step API:
        reset() -> obs, info
        step(action) -> obs, reward, terminated, truncated, info

    Actor observation layout, dim = 87:
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

    Privileged observation layout, dim = 178:
        actor_obs          87
        terrain_priv       91
            height_scan       81
            friction           1
            terrain onehot     4
            difficulty         1
            terrain params     4
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task2Config):
        super().__init__()

        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.dt = float(cfg.control_dt)

        self.world = Task2World(cfg.terrain_cfg, device=cfg.device)
        self.terrain_curriculum = TerrainCurriculum(
            num_envs=cfg.num_envs,
            world_cfg=cfg.terrain_cfg,
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
        self.scene = InteractiveScene(make_go2_task2_scene_cfg(cfg, self.world))
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact: ContactSensor = self.scene["contact_forces"]

        self._try_set_scene_terrain_origins()

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

        self.target_cmd = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.smoothed_cmd = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.last_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.prev_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.last_base_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_acc_obs = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.phase = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.prev_foot_contact = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        self.feet_air_time = torch.zeros((n, 4), dtype=torch.float32, device=self.device)

        self.env_terrain_types = torch.zeros(n, dtype=torch.long, device=self.device)
        self.env_terrain_levels = torch.zeros(n, dtype=torch.long, device=self.device)

        self.env_friction = torch.ones(n, dtype=torch.float32, device=self.device)
        self.env_restitution = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.env_material_id = torch.ones(n, dtype=torch.long, device=self.device)
        self.env_material_onehot = torch.zeros(
            (n, int(cfg.terrain_cfg.material_count)),
            dtype=torch.float32,
            device=self.device,
        )
        if int(cfg.terrain_cfg.material_count) > 1:
            self.env_material_onehot[:, 1] = 1.0
        else:
            self.env_material_onehot[:, 0] = 1.0

        self.terrain_height_under_base = torch.zeros(n, dtype=torch.float32, device=self.device)

        if bool(cfg.print_debug_info):
            self._print_debug_info()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _try_set_scene_terrain_origins(self) -> None:
        """Attach Isaac terrain env_origins to Task2World when available."""

        candidates = []

        try:
            candidates.append(getattr(self.scene, "terrain", None))
        except Exception:
            pass

        try:
            candidates.append(self.scene["terrain"])
        except Exception:
            pass

        for terrain_obj in candidates:
            if terrain_obj is None:
                continue
            if hasattr(terrain_obj, "env_origins"):
                try:
                    origins = terrain_obj.env_origins.reshape(-1, 3).to(self.device)
                    if origins.shape[0] >= self.cfg.terrain_cfg.num_levels * self.cfg.terrain_cfg.num_terrain_types:
                        self.world.set_scene_env_origins(origins)
                        return
                except Exception:
                    pass

    def _joint_ids(self, names: Iterable[str]) -> List[int]:
        names = list(names)
        missing = [name for name in names if name not in self.robot_joint_names]
        if missing:
            raise RuntimeError(
                f"[Go2Task2Env] Missing joints: {missing}\n"
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
                f"[Go2Task2Env] Missing body links: {missing}\n"
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
                f"[Go2Task2Env] Missing contact links: {missing}\n"
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
        print(" [Go2Task2Env] Multi-Terrain / Multi-Material Locomotion Environment Initialized")
        print("=" * 110)
        print(f" num_envs              : {self.cfg.num_envs}")
        print(f" device                : {self.device}")
        print(f" sim_dt / decimation   : {self.cfg.sim_dt} / {self.cfg.decimation}")
        print(f" control_dt            : {self.dt}")
        print(f" num_joints            : {self.robot.num_joints}")
        print(f" num_actions           : {self.cfg.num_actions}")
        print(f" num_observations      : {self.cfg.num_observations}")
        print(f" num_privileged_obs    : {self.cfg.num_privileged_obs}")
        print(f" terrain types         : {self.world.terrain_type_names}")
        print(f" terrain levels        : {self.cfg.terrain_cfg.num_levels}")
        print(f" terrain priv dim      : {self.cfg.terrain_cfg.terrain_priv_dim}")
        print(f" action_joint_ids      : {self.action_joint_ids}")
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
    # Curriculum / commands
    # -------------------------------------------------------------------------
    def _curriculum_k(self) -> float:
        return min(1.0, float(self.global_steps) / max(int(self.cfg.terrain_curriculum_total_steps), 1))

    def _command_stage(self) -> int:
        k = self._curriculum_k()
        if k < 0.05:
            return 0
        if k < 0.15:
            return 1
        if k < 0.30:
            return 2
        if k < 0.50:
            return 3
        if k < 0.75:
            return 4
        return 5

    def _max_allowed_terrain_level(self) -> int:
        k = self._curriculum_k()
        if k < 0.12:
            return 0
        if k < 0.25:
            return 2
        if k < 0.45:
            return 4
        if k < 0.70:
            return 7
        return int(self.cfg.terrain_cfg.num_levels) - 1

    def _restrict_terrain_for_global_stage(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        stage = self._command_stage()
        max_level = self._max_allowed_terrain_level()

        self.terrain_curriculum.env_levels[env_ids] = torch.clamp(
            self.terrain_curriculum.env_levels[env_ids],
            0,
            max_level,
        )

        anchors = self.terrain_curriculum.anchor_mask[env_ids]
        n = int(env_ids.numel())

        if stage == 0:
            self.terrain_curriculum.env_types[env_ids] = 0
            self.terrain_curriculum.env_levels[env_ids] = 0

        elif stage == 1:
            rand_type = torch.randint(0, 2, (n,), device=self.device)
            self.terrain_curriculum.env_types[env_ids] = torch.where(
                anchors,
                torch.zeros_like(rand_type),
                rand_type,
            )
            self.terrain_curriculum.env_levels[env_ids] = torch.clamp(
                self.terrain_curriculum.env_levels[env_ids],
                0,
                1,
            )

        elif stage == 2:
            rand_type = torch.randint(0, 2, (n,), device=self.device)
            sampled_level = torch.randint(0, max_level + 1, (n,), device=self.device)
            current_level = self.terrain_curriculum.env_levels[env_ids]
            explore = (~anchors) & (torch.rand(n, device=self.device) < 0.25)

            self.terrain_curriculum.env_types[env_ids] = torch.where(
                anchors,
                torch.zeros_like(rand_type),
                rand_type,
            )
            self.terrain_curriculum.env_levels[env_ids] = torch.where(
                explore,
                sampled_level,
                current_level,
            )

        elif stage == 3:
            rand_type = torch.randint(0, 3, (n,), device=self.device)
            sampled_level = torch.randint(0, max_level + 1, (n,), device=self.device)
            current_level = self.terrain_curriculum.env_levels[env_ids]
            explore = (~anchors) & (torch.rand(n, device=self.device) < 0.35)

            self.terrain_curriculum.env_types[env_ids] = torch.where(
                anchors,
                torch.zeros_like(rand_type),
                rand_type,
            )
            self.terrain_curriculum.env_levels[env_ids] = torch.where(
                explore,
                sampled_level,
                current_level,
            )

        else:
            rand_type = torch.randint(0, 4, (n,), device=self.device)
            sampled_level = torch.randint(0, max_level + 1, (n,), device=self.device)
            current_level = self.terrain_curriculum.env_levels[env_ids]
            explore = (~anchors) & (torch.rand(n, device=self.device) < 0.45)

            self.terrain_curriculum.env_types[env_ids] = torch.where(
                anchors,
                torch.zeros_like(rand_type),
                rand_type,
            )
            self.terrain_curriculum.env_levels[env_ids] = torch.where(
                explore,
                sampled_level,
                current_level,
            )

        self.terrain_curriculum.env_types[self.terrain_curriculum.anchor_mask] = 0
        self.terrain_curriculum.env_levels[self.terrain_curriculum.anchor_mask] = 0

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

        self._restrict_terrain_for_global_stage(env_ids)

        terrain_types, terrain_levels = self.terrain_curriculum.get_current_grid_indices(env_ids)
        self.env_terrain_types[env_ids] = terrain_types
        self.env_terrain_levels[env_ids] = terrain_levels

        mats = self.world.sample_material_params(terrain_types, terrain_levels)
        self.env_friction[env_ids] = mats["friction"]
        self.env_restitution[env_ids] = mats["restitution"]
        self.env_material_id[env_ids] = mats["material_id"]
        self.env_material_onehot[env_ids] = mats["material_onehot"]

        spawn_origins = self.world.sample_spawn_origins(
            terrain_types=terrain_types,
            terrain_levels=terrain_levels,
            randomize_xy=True,
            prefer_scene_origins=True,
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = spawn_origins
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        root_state[:, 7:13] = 0.0

        q = self.default_joint_pos_all[env_ids].clone()
        qd = torch.zeros_like(q)

        q[:, self.action_joint_ids_t] += torch.empty(
            (n, int(self.cfg.num_actions)),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-0.025, 0.025)

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

        self.terrain_curriculum.register_start_positions(env_ids, spawn_origins[:, 0])

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
                f"[Go2Task2Env] action shape mismatch: got {tuple(actions.shape)}, "
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
            current_x = self.robot.data.root_pos_w[reset_ids, 0].detach().clone()
            fall_flags = terminated[reset_ids].detach().clone()

            self.terrain_curriculum.update_curriculum(
                env_ids=reset_ids,
                current_x=current_x,
                fall_flags=fall_flags,
            )

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
    # Terrain / contact / observation
    # -------------------------------------------------------------------------
    def _get_terrain_height_under_points(
        self,
        xy_w: torch.Tensor,
        terrain_types: Optional[torch.Tensor] = None,
        terrain_levels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xy_w = torch.as_tensor(xy_w, dtype=torch.float32, device=self.device)

        squeeze_points = False
        if xy_w.ndim == 2:
            xy_w = xy_w.unsqueeze(1)
            squeeze_points = True

        if terrain_types is None:
            terrain_types = self.env_terrain_types
        if terrain_levels is None:
            terrain_levels = self.env_terrain_levels

        terrain_types = torch.as_tensor(terrain_types, dtype=torch.long, device=self.device)
        terrain_levels = torch.as_tensor(terrain_levels, dtype=torch.long, device=self.device)

        origins = self.world.get_origins_from_indices(
            terrain_types,
            terrain_levels,
            prefer_scene_origins=True,
        )

        local_xy = xy_w - origins[:, None, 0:2]

        local_height = self.world._analytical_terrain_height(
            local_xy,
            terrain_types,
            terrain_levels,
        )

        # 关键修复：
        # _analytical_terrain_height() 返回的是局部地形高度；
        # root_pos_w / foot_pos_w 是世界坐标；
        # 因此必须加上 terrain origin.z，保证 base_height / foot_height 使用同一世界坐标系。
        world_height = local_height + origins[:, None, 2]

        if squeeze_points:
            return world_height.squeeze(1)

        return world_height

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
        root_pos = self.robot.data.root_pos_w
        terrain_h = self._get_terrain_height_under_points(root_pos[:, :2]).squeeze(-1)
        self.terrain_height_under_base.copy_(terrain_h)
        return root_pos[:, 2] - terrain_h

    def _compute_foot_heights(self) -> torch.Tensor:
        foot_pos = self.robot.data.body_pos_w[:, self.foot_body_ids_t, :]
        foot_h = self._get_terrain_height_under_points(foot_pos[:, :, :2])
        return foot_pos[:, :, 2] - foot_h

    def _compute_obs(self) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        q = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        q_err = q - self.default_joint_pos

        contact, _ = self._get_foot_contact()

        root_pos = self.robot.data.root_pos_w
        base_height = self._compute_base_height().unsqueeze(-1)

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
                f"[Go2Task2Env] Observation dim mismatch: "
                f"got {obs.shape[-1]}, expected {self.cfg.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    def compute_privileged_obs(self) -> torch.Tensor:
        actor_obs = self._compute_obs()

        terrain_priv = self.world.make_privileged_terrain_features(
            base_pos_w=self.robot.data.root_pos_w,
            terrain_types=self.env_terrain_types,
            terrain_levels=self.env_terrain_levels,
            friction=self.env_friction,
            base_quat_wxyz=self.robot.data.root_quat_w,
            prefer_scene_origins=True,
        )

        priv = torch.cat([actor_obs, terrain_priv], dim=-1)

        if priv.shape[-1] != int(self.cfg.num_privileged_obs):
            raise RuntimeError(
                f"[Go2Task2Env] Privileged obs dim mismatch: "
                f"got {priv.shape[-1]}, expected {self.cfg.num_privileged_obs}"
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
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b

        vx, vy, vz = base_lin_vel[:, 0], base_lin_vel[:, 1], base_lin_vel[:, 2]
        wx, wy, wz = base_ang_vel[:, 0], base_ang_vel[:, 1], base_ang_vel[:, 2]

        root_pos = self.robot.data.root_pos_w
        base_height = self._compute_base_height()

        base_acc = (base_lin_vel - self.last_base_vel) / max(self.dt, 1e-6)
        self.base_acc_obs.copy_(base_acc)
        self.last_base_vel.copy_(base_lin_vel)

        q = self.robot.data.joint_pos[:, self.action_joint_ids_t]
        qd = self.robot.data.joint_vel[:, self.action_joint_ids_t]
        q_err = q - self.default_joint_pos

        contact, normal_force = self._get_foot_contact()
        contact_count = contact.sum(dim=-1)
        ref_contact = self._build_trot_contact_ref()

        foot_height = self._compute_foot_heights()
        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_ids_t, :2]

        cmd = self.smoothed_cmd
        cmd_xy = cmd[:, :2]
        cmd_xy_norm = torch.norm(cmd_xy, dim=-1)
        cmd_xy_norm_safe = torch.clamp(cmd_xy_norm, min=1.0e-6)
        cmd_yaw_abs = torch.abs(cmd[:, 2])

        lin_move_gate = (cmd_xy_norm > float(self.cfg.reward_cmd_lin_move_threshold)).float()
        yaw_move_gate = (cmd_yaw_abs > float(self.cfg.reward_cmd_yaw_move_threshold)).float()
        move_gate = torch.clamp(lin_move_gate + yaw_move_gate, 0.0, 1.0)
        stand_gate = 1.0 - move_gate

        cmd_dir = cmd_xy / cmd_xy_norm_safe.unsqueeze(-1)
        actual_along_cmd = torch.sum(base_lin_vel[:, :2] * cmd_dir, dim=-1)
        target_speed = torch.clamp(cmd_xy_norm, min=float(self.cfg.reward_target_speed_min))

        raw_speed_ratio = actual_along_cmd / target_speed
        speed_ratio_clamped = torch.clamp(
            raw_speed_ratio,
            min=float(self.cfg.reward_speed_ratio_min),
            max=float(self.cfg.reward_speed_ratio_max),
        )
        speed_gate = torch.clamp(speed_ratio_clamped, min=0.0, max=1.0) * lin_move_gate

        under_ratio = torch.clamp(
            (target_speed - actual_along_cmd) / target_speed,
            min=0.0,
            max=float(self.cfg.reward_under_ratio_max),
        ) * lin_move_gate

        over_ratio = torch.clamp(
            (actual_along_cmd - float(self.cfg.reward_over_speed_ratio) * target_speed) / target_speed,
            min=0.0,
            max=float(self.cfg.reward_over_ratio_max),
        ) * lin_move_gate

        reverse_ratio = torch.clamp(
            -actual_along_cmd / target_speed,
            min=0.0,
            max=float(self.cfg.reward_reverse_ratio_max),
        ) * lin_move_gate

        lin_err = torch.square(vx - cmd[:, 0]) + torch.square(vy - cmd[:, 1])
        yaw_err = torch.square(wz - cmd[:, 2])

        r_cmd_lin = torch.exp(-float(self.cfg.reward_cmd_lin_sigma) * lin_err)
        r_cmd_yaw = torch.exp(-float(self.cfg.reward_cmd_yaw_sigma) * yaw_err)

        r_cmd_speed = lin_move_gate * torch.exp(
            -float(self.cfg.reward_cmd_speed_sigma)
            * torch.square(
                torch.clamp(
                    raw_speed_ratio - 1.0,
                    min=-float(self.cfg.reward_cmd_speed_error_clip),
                    max=float(self.cfg.reward_cmd_speed_error_clip),
                )
            )
        )

        p_under_speed = -torch.square(under_ratio)
        p_over_speed = -torch.square(over_ratio)
        p_reverse = -torch.square(reverse_ratio)

        r_stand = torch.exp(
            -float(self.cfg.reward_stand_sigma)
            * (
                torch.square(vx)
                + torch.square(vy)
                + 0.5 * torch.square(wz)
                + 0.5 * torch.square(vz)
            )
        )
        r_stand_yaw = torch.exp(-float(self.cfg.reward_stand_sigma) * torch.square(wz))

        # 移动命令下，线速度奖励必须受到 speed_gate 约束；
        # zero command 下仍保留站立奖励，避免 stage0 没有学习信号。
        r_cmd_lin_gated = lin_move_gate * r_cmd_lin * speed_gate + stand_gate * r_stand

        yaw_speed_gate = float(self.cfg.reward_yaw_gate_min) + (
            1.0 - float(self.cfg.reward_yaw_gate_min)
        ) * speed_gate
        r_cmd_yaw_gated = move_gate * r_cmd_yaw * yaw_speed_gate + stand_gate * r_stand_yaw

        phase_error = torch.mean(torch.abs(contact - ref_contact), dim=-1)
        r_phase_contact = (1.0 - phase_error) * move_gate

        first_contact = (contact > 0.5) & (self.prev_foot_contact < 0.5)
        self.feet_air_time += self.dt

        r_air_time = (
            torch.sum(
                torch.clamp(
                    self.feet_air_time - float(self.cfg.air_time_target),
                    min=0.0,
                    max=float(self.cfg.reward_air_time_clip),
                )
                * first_contact.float(),
                dim=-1,
            )
            * move_gate
        )

        self.feet_air_time = torch.where(
            contact > 0.5,
            torch.zeros_like(self.feet_air_time),
            self.feet_air_time,
        )
        self.prev_foot_contact.copy_(contact)

        r_clearance = (
            torch.mean(
                (1.0 - contact)
                * torch.exp(
                    -float(self.cfg.reward_clearance_sigma)
                    * torch.abs(foot_height - float(self.cfg.foot_clearance_target))
                ),
                dim=-1,
            )
            * move_gate
        )

        progress_x = root_pos[:, 0] - self.terrain_curriculum.start_x
        r_terrain_progress = move_gate * torch.clamp(
            progress_x / max(float(self.cfg.terrain_cfg.success_distance), 1.0e-6),
            min=-1.0,
            max=1.0,
        )

        p_double_contact = -move_gate * torch.clamp(
            contact_count - float(self.cfg.reward_double_contact_threshold),
            min=0.0,
        )

        raw_foot_slip = torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * contact, dim=-1)
        p_foot_slip = -torch.clamp(
            raw_foot_slip,
            max=float(self.cfg.reward_foot_slip_clip),
        )

        r_upright = (1.0 - projected_gravity[:, 2]) * 0.5

        h_err = torch.square(base_height - float(self.cfg.target_height))
        r_height = torch.exp(-float(self.cfg.reward_height_sigma) * h_err)

        p_base_ang = -torch.clamp(
            torch.square(wx) + torch.square(wy),
            max=float(self.cfg.reward_base_ang_clip),
        )
        p_base_acc = -torch.clamp(
            torch.sum(torch.square(base_acc), dim=-1),
            max=float(self.cfg.reward_base_acc_clip),
        )
        p_z_vel = -torch.abs(vz)

        p_default_pose = -torch.mean(torch.square(q_err), dim=-1)

        lower_margin = q - self.joint_lower
        upper_margin = self.joint_upper - q
        p_joint_limit = -torch.mean(
            torch.square(torch.clamp(float(self.cfg.reward_joint_limit_margin) - lower_margin, min=0.0))
            + torch.square(torch.clamp(float(self.cfg.reward_joint_limit_margin) - upper_margin, min=0.0)),
            dim=-1,
        )

        p_action_rate = -torch.mean(torch.square(self.last_action - self.prev_action), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.last_action), dim=-1)

        torques = getattr(self.robot.data, "applied_torque", torch.zeros_like(self.robot.data.joint_vel))
        tau = torques[:, self.action_joint_ids_t]
        p_torque = -torch.clamp(
            torch.mean(torch.square(tau), dim=-1),
            max=float(self.cfg.reward_torque_clip),
        )
        p_energy = -torch.clamp(
            torch.mean(torch.abs(tau * qd), dim=-1),
            max=float(self.cfg.reward_energy_clip),
        )

        r_alive = torch.ones_like(vx)

        # ------------------------------------------------------------------
        # Reward-V2.1：
        # 所有组间权重、组内权重、比例惩罚、低速静态接触抑制都来自 config。
        # ------------------------------------------------------------------
        r_command_task = (
            float(self.cfg.reward_cmd_w_lin_gated) * r_cmd_lin_gated
            + float(self.cfg.reward_cmd_w_speed) * r_cmd_speed
            + float(self.cfg.reward_cmd_w_yaw_gated) * r_cmd_yaw_gated
            + float(self.cfg.reward_cmd_w_under) * p_under_speed
            + float(self.cfg.reward_cmd_w_over) * p_over_speed
            + float(self.cfg.reward_cmd_w_reverse) * p_reverse
        )

        r_locomotion_quality = (
            float(self.cfg.reward_loco_w_air_time) * r_air_time
            + float(self.cfg.reward_loco_w_clearance) * r_clearance
            + float(self.cfg.reward_loco_w_phase) * r_phase_contact
        )

        k_curriculum = self._curriculum_k()
        if k_curriculum < float(self.cfg.reward_contact_scale_k1):
            contact_weight_scale = float(self.cfg.reward_contact_scale_early)
        elif k_curriculum < float(self.cfg.reward_contact_scale_k2):
            contact_weight_scale = float(self.cfg.reward_contact_scale_middle)
        else:
            contact_weight_scale = float(self.cfg.reward_contact_scale_late)

        low_speed_gap = torch.clamp(
            float(self.cfg.reward_low_speed_static_threshold) - speed_gate,
            min=0.0,
            max=float(self.cfg.reward_low_speed_static_threshold),
        )
        low_speed_static_scale = 1.0 + float(self.cfg.reward_low_speed_static_gain) * (
            low_speed_gap / max(float(self.cfg.reward_low_speed_static_threshold), 1.0e-6)
        )

        p_double_contact_scaled = p_double_contact * low_speed_static_scale

        p_contact_quality = (
            float(self.cfg.reward_contact_w_foot_slip) * p_foot_slip
            + float(self.cfg.reward_contact_w_double_contact) * p_double_contact_scaled
        )

        r_stability_quality = (
            float(self.cfg.reward_stability_w_upright) * r_upright
            + float(self.cfg.reward_stability_w_height) * r_height
            + float(self.cfg.reward_stability_w_base_ang) * p_base_ang
            + float(self.cfg.reward_stability_w_z_vel) * p_z_vel
            + float(self.cfg.reward_stability_w_base_acc) * p_base_acc
            + float(self.cfg.reward_stability_w_stand) * r_stand * stand_gate
        )

        p_control_regularization = (
            float(self.cfg.reward_control_w_action_rate) * p_action_rate
            + float(self.cfg.reward_control_w_action_mag) * p_action_mag
            + float(self.cfg.reward_control_w_torque) * p_torque
            + float(self.cfg.reward_control_w_energy) * p_energy
            + float(self.cfg.reward_control_w_joint_limit) * p_joint_limit
            + float(self.cfg.reward_control_w_default_pose) * p_default_pose
        )

        continuous_raw = (
            float(self.cfg.reward_group_cmd) * r_command_task
            + float(self.cfg.reward_group_locomotion) * r_locomotion_quality
            + float(self.cfg.reward_group_contact) * contact_weight_scale * p_contact_quality
            + float(self.cfg.reward_group_stability) * r_stability_quality
            + float(self.cfg.reward_group_control) * p_control_regularization
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        joint_vel_abs_max = torch.abs(self.robot.data.joint_vel).max(dim=-1)[0]

        fall_low = base_height < float(self.cfg.fall_height)
        fall_high = base_height > float(self.cfg.jump_height)
        fall_bad_orientation = torch.norm(projected_gravity[:, :2], dim=-1) > float(self.cfg.bad_orientation_xy)
        fall_nonfinite_height = ~torch.isfinite(base_height)
        fall_nonfinite_joint = ~torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
        fall_joint_vel = joint_vel_abs_max > float(self.cfg.max_joint_vel_abs)

        is_fallen = (
            fall_low
            | fall_high
            | fall_bad_orientation
            | fall_nonfinite_height
            | fall_nonfinite_joint
            | fall_joint_vel
        )

        event_fall = torch.where(
            is_fallen,
            torch.full_like(continuous, float(self.cfg.penalty_fall)),
            torch.zeros_like(continuous),
        )

        reward_raw = continuous + event_fall

        projected_return = self.episode_return + reward_raw
        no_event = event_fall.abs() < 1.0e-6

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

        curr_stats = self.terrain_curriculum.log_curriculum_stats()

        info = {
            "reward_components": {
                "R_Cmd_Lin": self._mean_detached(
                    float(self.cfg.reward_group_cmd)
                    * float(self.cfg.reward_cmd_w_lin_gated)
                    * r_cmd_lin_gated
                ),
                "R_Cmd_Yaw": self._mean_detached(
                    float(self.cfg.reward_group_cmd)
                    * float(self.cfg.reward_cmd_w_yaw_gated)
                    * r_cmd_yaw_gated
                ),
                "R_Cmd_Speed": self._mean_detached(
                    float(self.cfg.reward_group_cmd)
                    * float(self.cfg.reward_cmd_w_speed)
                    * r_cmd_speed
                ),
                "P_Under_Speed": self._mean_detached(
                    float(self.cfg.reward_group_cmd)
                    * float(self.cfg.reward_cmd_w_under)
                    * p_under_speed
                ),
                "P_Over_Speed": self._mean_detached(
                    float(self.cfg.reward_group_cmd)
                    * float(self.cfg.reward_cmd_w_over)
                    * p_over_speed
                ),
                "P_Reverse": self._mean_detached(
                    float(self.cfg.reward_group_cmd)
                    * float(self.cfg.reward_cmd_w_reverse)
                    * p_reverse
                ),
                "R_Stand_Still": self._mean_detached(
                    float(self.cfg.reward_group_stability)
                    * float(self.cfg.reward_stability_w_stand)
                    * r_stand
                    * stand_gate
                ),
                "R_Phase_Contact": self._mean_detached(
                    float(self.cfg.reward_group_locomotion)
                    * float(self.cfg.reward_loco_w_phase)
                    * r_phase_contact
                ),
                "R_Air_Time": self._mean_detached(
                    float(self.cfg.reward_group_locomotion)
                    * float(self.cfg.reward_loco_w_air_time)
                    * r_air_time
                ),
                "R_Clearance": self._mean_detached(
                    float(self.cfg.reward_group_locomotion)
                    * float(self.cfg.reward_loco_w_clearance)
                    * r_clearance
                ),
                "R_Terrain_Progress": self._mean_detached(r_terrain_progress),
                "P_Double_Contact": self._mean_detached(
                    float(self.cfg.reward_group_contact)
                    * contact_weight_scale
                    * float(self.cfg.reward_contact_w_double_contact)
                    * p_double_contact_scaled
                ),
                "P_Foot_Slip": self._mean_detached(
                    float(self.cfg.reward_group_contact)
                    * contact_weight_scale
                    * float(self.cfg.reward_contact_w_foot_slip)
                    * p_foot_slip
                ),
                "R_Upright": self._mean_detached(
                    float(self.cfg.reward_group_stability)
                    * float(self.cfg.reward_stability_w_upright)
                    * r_upright
                ),
                "R_Height": self._mean_detached(
                    float(self.cfg.reward_group_stability)
                    * float(self.cfg.reward_stability_w_height)
                    * r_height
                ),
                "P_Base_Ang": self._mean_detached(
                    float(self.cfg.reward_group_stability)
                    * float(self.cfg.reward_stability_w_base_ang)
                    * p_base_ang
                ),
                "P_Base_Acc": self._mean_detached(
                    float(self.cfg.reward_group_stability)
                    * float(self.cfg.reward_stability_w_base_acc)
                    * p_base_acc
                ),
                "P_Z_Vel": self._mean_detached(
                    float(self.cfg.reward_group_stability)
                    * float(self.cfg.reward_stability_w_z_vel)
                    * p_z_vel
                ),
                "P_Default_Pose": self._mean_detached(
                    float(self.cfg.reward_group_control)
                    * float(self.cfg.reward_control_w_default_pose)
                    * p_default_pose
                ),
                "R_Alive": self._mean_detached(r_alive),
                "P_Joint_Limit": self._mean_detached(
                    float(self.cfg.reward_group_control)
                    * float(self.cfg.reward_control_w_joint_limit)
                    * p_joint_limit
                ),
                "P_Action_Rate": self._mean_detached(
                    float(self.cfg.reward_group_control)
                    * float(self.cfg.reward_control_w_action_rate)
                    * p_action_rate
                ),
                "P_Action_Mag": self._mean_detached(
                    float(self.cfg.reward_group_control)
                    * float(self.cfg.reward_control_w_action_mag)
                    * p_action_mag
                ),
                "P_Torque": self._mean_detached(
                    float(self.cfg.reward_group_control)
                    * float(self.cfg.reward_control_w_torque)
                    * p_torque
                ),
                "P_Energy": self._mean_detached(
                    float(self.cfg.reward_group_control)
                    * float(self.cfg.reward_control_w_energy)
                    * p_energy
                ),
                "Continuous": self._mean_detached(continuous),
                "Event_Fall": self._mean_detached(event_fall),
                "Total": self._mean_detached(reward),
            },
            "reward_groups": {
                "R_Command_Task": self._mean_detached(float(self.cfg.reward_group_cmd) * r_command_task),
                "R_Locomotion_Quality": self._mean_detached(
                    float(self.cfg.reward_group_locomotion) * r_locomotion_quality
                ),
                "P_Contact_Quality": self._mean_detached(
                    float(self.cfg.reward_group_contact) * contact_weight_scale * p_contact_quality
                ),
                "R_Stability_Quality": self._mean_detached(
                    float(self.cfg.reward_group_stability) * r_stability_quality
                ),
                "P_Control_Regularization": self._mean_detached(
                    float(self.cfg.reward_group_control) * p_control_regularization
                ),
                "Contact_Weight_Scale": self._float_tensor(contact_weight_scale, self.device),
                "Reward_Clip": self._float_tensor(float(self.cfg.continuous_reward_clip), self.device),
                "R_Terrain_Progress_Aux_Disabled": self._float_tensor(0.0, self.device),
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
                "Actual_Along_Cmd": self._mean_detached(actual_along_cmd * lin_move_gate),
                "Target_Speed": self._mean_detached(target_speed * lin_move_gate),
                "Speed_Ratio": self._mean_detached(speed_ratio_clamped * lin_move_gate),
                "Speed_Gate": self._mean_detached(speed_gate),
                "Under_Ratio": self._mean_detached(under_ratio),
                "Over_Ratio": self._mean_detached(over_ratio),
                "Reverse_Ratio": self._mean_detached(reverse_ratio),
                "Low_Speed_Static_Scale": self._mean_detached(low_speed_static_scale * lin_move_gate),
                "Lin_Error": self._mean_detached(lin_err),
                "Yaw_Error": self._mean_detached(yaw_err),
                "Base_Height": self._mean_detached(base_height),
                "Terrain_Height": self._mean_detached(self.terrain_height_under_base),
                "Contact_Count": self._mean_detached(contact_count),
                "FL_Contact": self._mean_detached(contact[:, 0]),
                "FR_Contact": self._mean_detached(contact[:, 1]),
                "RL_Contact": self._mean_detached(contact[:, 2]),
                "RR_Contact": self._mean_detached(contact[:, 3]),
                "Normal_Force_Mean": self._mean_detached(normal_force),
                "Mean_Terrain_Type": self._mean_detached(self.env_terrain_types.float()),
                "Mean_Terrain_Level": self._mean_detached(self.env_terrain_levels.float()),
                "Mean_Friction": self._mean_detached(self.env_friction),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Global_Steps": self._float_tensor(float(self.global_steps), self.device),
            },
            "curriculum": {
                k: self._float_tensor(v, self.device) for k, v in curr_stats.items()
            },
            "debug": {
                "Obs_Dim": self._float_tensor(float(self.cfg.num_observations), self.device),
                "Privileged_Obs_Dim": self._float_tensor(float(self.cfg.num_privileged_obs), self.device),
                "Terrain_Priv_Dim": self._float_tensor(float(self.cfg.terrain_cfg.terrain_priv_dim), self.device),
                "Reward_Min": reward.detach().min(),
                "Reward_Max": reward.detach().max(),
                "Continuous_Min": continuous.detach().min(),
                "Continuous_Max": continuous.detach().max(),
                "Continuous_Raw_Min": continuous_raw.detach().min(),
                "Continuous_Raw_Max": continuous_raw.detach().max(),
                "Base_Height_Min": base_height.detach().min(),
                "Base_Height_Max": base_height.detach().max(),
                "JointVel_Max": joint_vel_abs_max.detach().max(),
                "Fall_Reason_Low": self._mean_detached(fall_low.float()),
                "Fall_Reason_High": self._mean_detached(fall_high.float()),
                "Fall_Reason_Bad_Orientation": self._mean_detached(fall_bad_orientation.float()),
                "Fall_Reason_Nonfinite_Height": self._mean_detached(fall_nonfinite_height.float()),
                "Fall_Reason_Nonfinite_Joint": self._mean_detached(fall_nonfinite_joint.float()),
                "Fall_Reason_JointVel": self._mean_detached(fall_joint_vel.float()),
            },
        }

        return reward, terminated, truncated, info

# Backward-compatible aliases for older local scripts.
QuadrupedTask2Env = Go2Task2Env
UnitreeGo2Task2Env = Go2Task2Env
