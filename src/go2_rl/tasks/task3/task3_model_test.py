# Copyright (c) 2026
# Unitree Go2 Task3: 导航避障模型评估入口。
#
# 本文件用于评估 Task3 导航避障任务的 skrl PPO checkpoint。
# 本文件会创建 IsaacLab AppLauncher，并在导入 IsaacLab 环境后构建评估环境。
#
# Gymnasium API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# 观测维度:
#   actor single obs = 208
#   actor stacked obs = 1040
#   world privileged tail = 68
#   raw privileged obs = 276
#   critic obs = 1108
#   lidar rays = 60
#   action dim = 12
#
# 模型评估入口:
#   python src/go2_rl/tasks/task3/task3_model_test.py --checkpoint <checkpoint>
#
# 工程说明:
#   默认评估模式打开 Isaac Sim GUI，便于观察导航轨迹、目标点和障碍物关系。
#   传入 --headless-eval 时切换为无头评估，用于服务器或批量测试。
#   评估 wrapper 与训练 wrapper 使用相同的 actor / critic 维度布局。
#
# Unitree Go2 Task3: navigation and obstacle-avoidance model evaluation entry.
#
# This file evaluates skrl PPO checkpoints for Task3 navigation and obstacle avoidance.
# It creates IsaacLab AppLauncher and builds the evaluation environment after
# IsaacLab environment modules are imported.
#
# Gymnasium API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# Observation dimensions:
#   actor single obs = 208
#   actor stacked obs = 1040
#   world privileged tail = 68
#   raw privileged obs = 276
#   critic obs = 1108
#   lidar rays = 60
#   action dim = 12
#
# Model evaluation entry:
#   python src/go2_rl/tasks/task3/task3_model_test.py --checkpoint <checkpoint>
#
# Engineering notes:
#   Evaluation opens Isaac Sim GUI by default for inspecting navigation paths,
#   target points, and obstacle relationships. Passing --headless-eval switches
#   to headless evaluation for servers or batch testing. The evaluation wrapper
#   uses the same actor / critic dimension layout as the training wrapper.

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Unitree Go2 Task3 skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--force-stage", type=int, default=-1)
parser.add_argument("--print-interval", type=int, default=100)
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--visualize", action="store_true", help="Compatibility flag; GUI is enabled by default")
parser.add_argument("--headless-eval", action="store_true", help="Run model evaluation without Isaac Sim GUI")
parser.add_argument("--show-world-markers", action="store_true", default=True)
parser.add_argument("--no-world-markers", action="store_true")
parser.add_argument(
    "--no-close-on-exit",
    action="store_true",
    help="Debug only: keep Isaac Sim open after evaluation",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Windows / GIF 评估默认打开 Isaac Sim GUI。
# 只有显式传入 --headless-eval 时，才使用无头评估。
args_cli.headless = bool(getattr(args_cli, "headless_eval", False))
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

simulation_app = AppLauncher(args_cli).app

try:
    import omni.usd
    from pxr import Gf, UsdGeom
except Exception:
    omni = None
    Gf = None
    UsdGeom = None

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from go2_rl.common.eval_curriculum_utils import force_eval_curriculum
from go2_rl.common.go2_skrl_models import Go2Actor, Go2Critic
from go2_rl.common.info_utils import flat_dict, load_normalizers
from go2_rl.common.model_eval_utils import direct_policy_action, init_agent_compat
from go2_rl.tasks.task3.task3_config import Task3Config
from go2_rl.tasks.task3.task3_env import Go2Task3Env


class Task3EvalWorldMarkers:
    """在模型测试脚本内显示 Task3 analytical world。

    这个类只读取 base_env.world 中的 tensor，用 USD marker 可视化目标点和障碍物。
    它不修改 task3_env.py / task3_world.py，不影响训练逻辑、奖励、观测、碰撞或 lidar。
    """

    def __init__(self, base_env: Go2Task3Env, enabled: bool = True):
        self.base_env = base_env
        self.enabled = bool(enabled) and omni is not None and UsdGeom is not None and Gf is not None
        self.initialized = False

        self.root_path = "/World/Task3EvalDebug"
        self.goal_path = f"{self.root_path}/Goal"
        self.static_paths: List[str] = []
        self.dynamic_paths: List[str] = []

    def _stage(self):
        if not self.enabled:
            return None
        try:
            return omni.usd.get_context().get_stage()
        except Exception:
            return None

    def _set_translation(self, prim, xyz: Tuple[float, float, float]) -> None:
        if UsdGeom is None or Gf is None:
            return

        xformable = UsdGeom.Xformable(prim)
        translate_op = None

        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()

        translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def _set_color(self, prim, rgb: Tuple[float, float, float]) -> None:
        if UsdGeom is None or Gf is None:
            return

        try:
            UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(
                [Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))]
            )
        except Exception:
            pass

    def _ensure(self) -> None:
        if self.initialized or not self.enabled:
            return

        stage = self._stage()
        if stage is None:
            return

        root = UsdGeom.Xform.Define(stage, self.root_path)
        self._set_translation(root.GetPrim(), (0.0, 0.0, 0.0))

        goal = UsdGeom.Sphere.Define(stage, self.goal_path)
        goal.CreateRadiusAttr(0.35)
        self._set_color(goal.GetPrim(), (0.1, 0.9, 0.1))

        world_cfg = self.base_env.cfg.world_cfg
        max_static = int(getattr(world_cfg, "max_static_obs", 0))
        max_dynamic = int(getattr(world_cfg, "max_dynamic_obs", 0))

        self.static_paths = []
        for i in range(max_static):
            path = f"{self.root_path}/StaticObstacle_{i:02d}"
            cyl = UsdGeom.Cylinder.Define(stage, path)
            cyl.CreateRadiusAttr(0.25)
            cyl.CreateHeightAttr(0.60)
            self._set_color(cyl.GetPrim(), (0.9, 0.15, 0.1))
            self.static_paths.append(path)

        self.dynamic_paths = []
        for i in range(max_dynamic):
            path = f"{self.root_path}/DynamicObstacle_{i:02d}"
            cyl = UsdGeom.Cylinder.Define(stage, path)
            cyl.CreateRadiusAttr(0.25)
            cyl.CreateHeightAttr(0.60)
            self._set_color(cyl.GetPrim(), (0.1, 0.35, 1.0))
            self.dynamic_paths.append(path)

        self.initialized = True

    def update(self) -> None:
        if not self.enabled:
            return

        stage = self._stage()
        if stage is None:
            return

        self._ensure()
        if not self.initialized:
            return

        env_i = 0
        base_env = self.base_env
        world = base_env.world

        try:
            origin = base_env.env_origins[env_i].detach()
        except Exception:
            origin = torch.zeros(3, device=base_env.device)

        ox = float(origin[0].detach().cpu().item())
        oy = float(origin[1].detach().cpu().item())

        # 绿色目标球。
        try:
            goal_xy = world.target_pos[env_i].detach()
            goal_prim = stage.GetPrimAtPath(self.goal_path)

            if goal_prim.IsValid():
                self._set_translation(
                    goal_prim,
                    (
                        ox + float(goal_xy[0].cpu().item()),
                        oy + float(goal_xy[1].cpu().item()),
                        0.35,
                    ),
                )
        except Exception:
            pass

        # 红色静态障碍物柱体。
        for i, path in enumerate(self.static_paths):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            try:
                active = bool(world.static_mask[env_i, i].detach().cpu().item())
                cyl = UsdGeom.Cylinder(prim)

                if active:
                    obs = world.static_obs[env_i, i].detach()
                    radius = float(obs[2].cpu().item())

                    cyl.GetRadiusAttr().Set(max(radius, 0.05))
                    cyl.GetHeightAttr().Set(0.60)

                    self._set_translation(
                        prim,
                        (
                            ox + float(obs[0].cpu().item()),
                            oy + float(obs[1].cpu().item()),
                            0.30,
                        ),
                    )
                else:
                    cyl.GetRadiusAttr().Set(0.01)
                    self._set_translation(prim, (0.0, 0.0, -10.0))
            except Exception:
                self._set_translation(prim, (0.0, 0.0, -10.0))

        # 蓝色动态障碍物柱体。
        for i, path in enumerate(self.dynamic_paths):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            try:
                active = bool(world.dynamic_mask[env_i, i].detach().cpu().item())
                cyl = UsdGeom.Cylinder(prim)

                if active:
                    pos = world.dynamic_obs_pos[env_i, i].detach()
                    radius = float(world.dynamic_obs_radius[env_i, i].detach().cpu().item())

                    cyl.GetRadiusAttr().Set(max(radius, 0.05))
                    cyl.GetHeightAttr().Set(0.60)

                    self._set_translation(
                        prim,
                        (
                            ox + float(pos[0].cpu().item()),
                            oy + float(pos[1].cpu().item()),
                            0.30,
                        ),
                    )
                else:
                    cyl.GetRadiusAttr().Set(0.01)
                    self._set_translation(prim, (0.0, 0.0, -10.0))
            except Exception:
                self._set_translation(prim, (0.0, 0.0, -10.0))


class Go2Task3EvalFrameStackWrapper(gym.Env):
    """Task3 专用评估 wrapper，严格对齐最终训练 checkpoint。

    训练 / 评估维度：
        actor single obs = 208
        actor stacked obs = 208 * 5 = 1040
        world privileged tail = 68
        critic obs = 1040 + 68 = 1108

    工程说明:
        通用 Go2FrameStackWrapper(use_privileged_obs=True) 会堆叠完整 276 维 privileged obs，
        得到 critic = 276 * 5 = 1380。Task3 最终 checkpoint 使用的是
        actor_stack 1040 + world privileged tail 68 = 1108，因此评估阶段保留任务专用 wrapper。
    """

    def __init__(self, env: Go2Task3Env, n_stack: int = 5):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device

        self.single_obs_dim = int(env.cfg.num_observations)
        self.single_priv_dim = int(env.cfg.num_privileged_obs)
        self.world_priv_dim = int(self.single_priv_dim - self.single_obs_dim)

        if self.single_obs_dim != 208:
            raise RuntimeError(f"Task3 actor single obs dim should be 208, got {self.single_obs_dim}")

        if self.world_priv_dim != 68:
            raise RuntimeError(f"Task3 world priv dim should be 68, got {self.world_priv_dim}")

        self.stacked_obs_dim = int(self.single_obs_dim * self.n_stack)
        self.critic_obs_dim = int(self.stacked_obs_dim + self.world_priv_dim)

        if self.stacked_obs_dim != 1040:
            raise RuntimeError(f"Task3 policy obs dim should be 1040, got {self.stacked_obs_dim}")

        if self.critic_obs_dim != 1108:
            raise RuntimeError(f"Task3 critic obs dim should be 1108, got {self.critic_obs_dim}")

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacked_obs_dim,),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.critic_obs_dim,),
            dtype=np.float32,
        )
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.observation_space,
                "critic": self.state_space,
            }
        )

        self.action_space = env.action_space
        self.single_action_space = env.action_space

        self.obs_stack = torch.zeros(
            (self.num_envs, self.stacked_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    @property
    def unwrapped(self):
        return self

    def _build_critic_obs(self) -> torch.Tensor:
        raw_priv = self.env.compute_privileged_obs()
        world_priv = raw_priv[:, self.single_obs_dim:]
        critic = torch.cat([self.obs_stack, world_priv], dim=-1)

        return torch.nan_to_num(
            torch.clamp(critic, -20.0, 20.0),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )

    def _pack(self):
        return {
            "policy": self.obs_stack.clone(),
            "critic": self._build_critic_obs().clone(),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)

        for i in range(self.n_stack):
            self.obs_stack[:, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs

        self.last_info = info or {}
        return self._pack(), self.last_info

    @torch.no_grad()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.obs_stack[:, :-self.single_obs_dim] = self.obs_stack[:, self.single_obs_dim :].clone()
        self.obs_stack[:, -self.single_obs_dim :] = obs

        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)

            for i in range(self.n_stack):
                self.obs_stack[
                    ids,
                    i * self.single_obs_dim : (i + 1) * self.single_obs_dim,
                ] = obs[ids]

        self.last_info = info or {}
        self.last_reward_mean = float(reward.detach().float().mean().cpu().item())
        self.last_done_count = int(done.sum().detach().cpu().item())

        return self._pack(), reward, terminated, truncated, self.last_info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def summarize(records: List[Dict[str, float]]):
    if not records:
        return {}

    keys = sorted({k for row in records for k in row.keys()})
    out = {}

    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue

        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return out


def print_table(summary):
    print("\n" + "=" * 170)
    print("Go2 Task3 Model Test Summary")
    print("=" * 170)
    print(f"{'metric':<78} | {'mean':>12} | {'std':>12} | {'min':>12} | {'max':>12}")
    print("-" * 170)

    for key in sorted(summary):
        row = summary[key]
        print(
            f"{key:<78} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )

    print("=" * 170 + "\n")


def _base_ppo_cfg_dict():
    cfg = PPO_CFG()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return cfg.copy()


def build_agent(env):
    models = {
        "policy": Go2Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=-1.35,
            min_log_std=-5.0,
            max_log_std=0.20,
        ),
        "value": Go2Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    cfg = _base_ppo_cfg_dict()

    requested = {
        "rollouts": 1,
        "learning_epochs": 1,
        "mini_batches": 1,
        "observation_preprocessor": RunningStandardScaler,
        "observation_preprocessor_kwargs": {
            "size": env.observation_space,
            "device": env.device,
        },
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {
            "size": env.state_space,
            "device": env.device,
        },
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {
            "size": 1,
            "device": env.device,
        },
    }

    for k, v in requested.items():
        if k in cfg:
            cfg[k] = v

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": str(PROJECT_ROOT / "logs" / "task3_eval_tmp"),
            "experiment_name": "eval",
            "write_interval": 0,
            "checkpoint_interval": 0,
            "store_separately": True,
            "wandb": False,
        }
    )

    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=env.device)

    return PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )


def resolve_checkpoint(path: str) -> str:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    candidates = [
        p / "go2_task3_model.pt",
        p / "agent.pt",
        p / "checkpoint.pt",
        p / "best_agent.pt",
        p / "final_checkpoint" / "go2_task3_model.pt",
    ]

    for cand in candidates:
        if cand.exists():
            return str(cand)

    return str(p)


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def step_env(env, actions):
    out = env.step(actions)
    if len(out) == 5:
        return out

    states, rewards, dones, infos = out
    return states, rewards, dones, dones, infos


def force_task3_eval_stage(base_env: Go2Task3Env, start_k: float, force_stage: int = -1) -> int:
    """只在 model_test.py 运行期强制 Task3 评估阶段，不修改 env/world 文件。"""

    k = float(max(0.0, min(1.0, float(start_k))))

    try:
        base_env.cfg.world_cfg.curriculum_resume_k_floor = k
    except Exception:
        pass

    total_steps = int(getattr(base_env.cfg.world_cfg, "curriculum_total_steps", 0))
    if total_steps <= 0:
        total_steps = int(getattr(base_env.cfg, "curriculum_total_steps", 0))

    global_steps = int(k * float(max(total_steps, 1)))
    base_env.global_steps = global_steps

    if int(force_stage) >= 0:
        stage = int(force_stage)
    else:
        try:
            stage = int(base_env.world.stage_from_progress(k))
        except Exception:
            stage = int(getattr(base_env.world, "stage_count", 6)) - 1

    stage_count = int(getattr(base_env.world, "stage_count", stage + 1))
    stage = int(max(0, min(stage_count - 1, stage)))

    for name, value in (
        ("curriculum_active_stage", stage),
        ("curriculum_stage_start_steps", global_steps),
        ("curriculum_last_check_steps", global_steps),
    ):
        if hasattr(base_env, name):
            try:
                setattr(base_env, name, value)
            except Exception:
                pass

    try:
        env_ids = torch.arange(base_env.num_envs, dtype=torch.long, device=base_env.device)
        base_env.world.set_stage(env_ids, stage)
    except Exception:
        pass

    # 只在评估进程内 monkey patch reset stage。
    # 这样 reset_env(env) 不会被 performance-gated curriculum 混合采样回 Stage0。
    try:
        import types

        def _forced_sample_reset_stages(self, env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            return torch.full(
                (int(env_ids.numel()),),
                int(stage),
                dtype=torch.long,
                device=self.device,
            )

        base_env._sample_reset_stages = types.MethodType(_forced_sample_reset_stages, base_env)
        base_env._rt_eval_forced_stage = int(stage)
    except Exception as exc:
        print(f"[WARN] failed to monkey-patch Task3 reset stage: {exc}")

    print(
        f"[TASK3_EVAL] requested start_k={k:.4f}, force_stage={stage}, "
        f"global_steps={global_steps:,}"
    )

    return stage


def main():
    set_seed(int(args_cli.seed))

    cfg = Task3Config()

    # GUI/GIF 模式只显示一个 Go2；headless-eval 才允许多环境评估。
    requested_num_envs = int(args_cli.num_envs)
    is_headless_eval = bool(getattr(args_cli, "headless_eval", False))
    cfg.num_envs = requested_num_envs if is_headless_eval else 1

    cfg.device = str(args_cli.device)
    cfg.print_debug_info = False

    # 在环境创建前设置 curriculum floor，降低 reset 采到低阶段的概率。
    try:
        cfg.world_cfg.curriculum_resume_k_floor = float(args_cli.start_k)
    except Exception:
        pass

    base_env = Go2Task3Env(cfg)

    force_eval_curriculum(base_env, args_cli.start_k, label="after_env_creation")
    forced_stage = force_task3_eval_stage(base_env, args_cli.start_k, int(args_cli.force_stage))

    stacked_env = Go2Task3EvalFrameStackWrapper(base_env, n_stack=5)
    env = wrap_env(stacked_env, wrapper="isaaclab")

    print("\n[INFO] Go2 Task3 Eval Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    if int(env.observation_space.shape[0]) != 1040:
        raise RuntimeError(f"Task3 policy input dim should be 1040, got {env.observation_space.shape[0]}")
    if int(env.state_space.shape[0]) != 1108:
        raise RuntimeError(f"Task3 critic input dim should be 1108, got {env.state_space.shape[0]}")

    agent = build_agent(env)
    init_agent_compat(agent)

    checkpoint = Path(resolve_checkpoint(args_cli.checkpoint)).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    print(f"[INFO] loading checkpoint: {checkpoint}")
    agent.load(str(checkpoint))

    normalizer_dir = checkpoint.parent
    loaded = load_normalizers(agent, str(normalizer_dir))
    print(f"[INFO] loaded normalizers: {loaded if loaded else '<none>'}")

    try:
        agent.set_running_mode("eval")
    except Exception:
        pass

    force_eval_curriculum(base_env, args_cli.start_k, label="before_rollout_reset")
    forced_stage = force_task3_eval_stage(base_env, args_cli.start_k, int(args_cli.force_stage))

    states, _ = reset_env(env)

    # reset 后再次固定 stage，防止 reset 时重新采样到低阶段。
    forced_stage = force_task3_eval_stage(base_env, args_cli.start_k, int(args_cli.force_stage))

    show_markers = (
        (not bool(getattr(args_cli, "headless_eval", False)))
        and bool(getattr(args_cli, "show-world-markers", True))
        and (not bool(getattr(args_cli, "no-world-markers", False)))
    )
    markers = Task3EvalWorldMarkers(base_env, enabled=show_markers)
    markers.update()

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0
    total_success = 0
    total_collision = 0
    total_fall = 0
    total_timeout = 0

    start = time.time()

    print("\n" + "=" * 150)
    print("Unitree Go2 Task3 skrl model test started")
    print("=" * 150)
    print(f"checkpoint   : {checkpoint}")
    print(f"num_envs     : {env.num_envs}")
    print(f"steps        : {args_cli.steps}")
    print(f"start_k      : {args_cli.start_k}")
    print(f"forced_stage : {forced_stage}")
    print(f"device       : {env.device}")
    print(f"markers      : {show_markers}")
    print("=" * 150 + "\n")

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Go2 Task3 Model Test",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for step in range(int(args_cli.steps)):
                with torch.no_grad():
                    actions = direct_policy_action(
                        agent,
                        states,
                        debug=False,
                        step=int(step),
                    )

                    action_abs_mean = float(actions.detach().abs().mean().cpu().item())
                    action_abs_max = float(actions.detach().abs().max().cpu().item())

                    states, rewards, terminated, truncated, _ = step_env(env, actions)

                markers.update()

                flat = flat_dict(stacked_env.last_info)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                num_envs = int(env.num_envs)
                total_success += int(round(flat.get("events/Success_Rate", 0.0) * num_envs))
                total_collision += int(round(flat.get("events/Collision_Rate", 0.0) * num_envs))
                total_fall += int(round(flat.get("events/Fall_Rate", 0.0) * num_envs))
                total_timeout += int(round(flat.get("events/Timeout_Rate", 0.0) * num_envs))

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                    row = {
                        "reward_mean": float(rewards.detach().float().mean().cpu().item()),
                        "terminated_rate": float(terminated.float().mean().cpu().item()),
                        "truncated_rate": float(truncated.float().mean().cpu().item()),
                        "action_abs_mean": action_abs_mean,
                        "action_abs_max": action_abs_max,
                    }
                    row.update(flat)
                    records.append(row)

                    stage_val = flat.get("telemetry/Command_Stage", flat.get("world/Stage", 0.0))
                    static_count = flat.get("telemetry/Static_Count", flat.get("world/Static_Count", 0.0))
                    dynamic_count = flat.get("telemetry/Dynamic_Count", flat.get("world/Dynamic_Count", 0.0))
                    vx_val = flat.get("telemetry/Actual_Vx", flat.get("telemetry/Actual_Vx_Body", 0.0))
                    risk_val = flat.get("telemetry/Collision_Risk", flat.get("world/Risk_All", 0.0))
                    progress_val = flat.get("telemetry/Progress_Step", flat.get("telemetry/Progress", 0.0))
                    dist_val = flat.get("telemetry/Distance_To_Goal", flat.get("world/Goal_Dist", 0.0))

                    pbar.set_postfix(
                        {
                            "rew": f"{row['reward_mean']:+.3f}",
                            "stage": f"{stage_val:.1f}",
                            "static": f"{static_count:.0f}",
                            "dyn": f"{dynamic_count:.0f}",
                            "dist": f"{dist_val:.2f}",
                            "prog": f"{progress_val:+.3f}",
                            "succ": f"{flat.get('events/Success_Rate', 0.0):.3f}",
                            "coll": f"{flat.get('events/Collision_Rate', 0.0):.3f}",
                            "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
                            "risk": f"{risk_val:.2f}",
                            "vx": f"{vx_val:+.2f}",
                            "act": f"{action_abs_mean:.2f}/{action_abs_max:.2f}",
                        }
                    )

                pbar.update(1)

        elapsed = time.time() - start
        env_steps = int(args_cli.steps) * int(env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n[OK] Go2 Task3 model test rollout finished")
        print(f"  env steps          : {env_steps:,}")
        print(f"  fps                : {fps:,.2f}")
        print(f"  total terminated   : {total_terminated:,}")
        print(f"  total truncated    : {total_truncated:,}")
        print(f"  approx success     : {total_success:,}")
        print(f"  approx collision   : {total_collision:,}")
        print(f"  approx fall        : {total_fall:,}")
        print(f"  approx timeout     : {total_timeout:,}")

        print_table(summarize(records))

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            if not bool(getattr(args_cli, "no-close-on-exit", False)):
                simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
