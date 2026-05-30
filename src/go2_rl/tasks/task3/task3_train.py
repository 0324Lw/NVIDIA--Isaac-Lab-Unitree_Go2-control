from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from isaaclab.app import AppLauncher
from go2_rl.common.paths import default_log_root


parser = argparse.ArgumentParser(description="Train Unitree Go2 Task3 Navigation / Obstacle Avoidance PPO with skrl")

# Runtime
parser.add_argument("--total-env-steps", type=int, default=800_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=20_000_000)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)
parser.add_argument(
    "--force-stage",
    type=int,
    default=-1,
    help="Force Task3 curriculum active stage after env creation/resume. Use only for curriculum debugging or staged continuation.",
)

# Resume / warm-start
parser.add_argument("--resume", type=str, default="", help="Optional Task3 skrl checkpoint file or checkpoint directory")
parser.add_argument("--pretrained-task1", type=str, default="", help="Optional Task1 skrl checkpoint file or directory for actor warm-start")
parser.add_argument("--pretrained-task2", type=str, default="", help="Optional Task2 skrl checkpoint file or directory for actor warm-start")
parser.add_argument("--pretrained-log-std", type=float, default=-1.75)

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", "--epochs", dest="learning_epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--min-lr", type=float, default=2e-5)
parser.add_argument("--max-lr", type=float, default=1.2e-4)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--kl-threshold", "--target-kl", dest="kl_threshold", type=float, default=0.015)
parser.add_argument("--entropy-coef", type=float, default=0.004)
parser.add_argument("--value-coef", type=float, default=2.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--ratio-clip", "--clip-range", dest="ratio_clip", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0.2)
parser.add_argument("--init-log-std", type=float, default=-1.35)
parser.add_argument("--min-log-std", type=float, default=-5.0)
parser.add_argument("--max-log-std", type=float, default=0.20)

# Logging / checkpoint
parser.add_argument("--log-root", type=str, default=os.environ.get("RT_GO2_TASK3_LOG_ROOT", default_log_root("task3")))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=10)
parser.add_argument("--tb-log-interval-steps", type=int, default=50)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)

# AppLauncher owns --headless / --device / --experience.
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

try:
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except ImportError:
    from skrl.resources.schedulers.torch import KLAdaptiveRL as KLAdaptiveLR

from go2_rl.common.go2_skrl_models import Go2Actor, Go2Critic
from go2_rl.common.info_utils import (
    current_lr,
    flat_dict,
    make_table,
    save_normalizers,
    to_float,
    tracking_mean,
    write_scalars,
)
from go2_rl.tasks.task3.task3_config import Task3Config
from go2_rl.tasks.task3.task3_env import Go2Task3Env


class Go2Task3AsymFrameStackWrapper(gym.Env):
    """Task3 asymmetric frame-stack wrapper for skrl.

    Task3-Navigation-V3 uses navigation-specific observations, so dimensions
    are read from Task3Config instead of being hard-coded.

    Actor:
        actor_obs_stack = single_actor_obs_dim * n_stack

    Critic:
        critic_obs = actor_obs_stack + world_privileged_dim

    Raw env.compute_privileged_obs():
        single_actor_obs + world_privileged_dim

    The wrapper returns a dict compatible with skrl IsaacLab wrapper:
        {"policy": actor_stack, "critic": critic_obs}
    """

    def __init__(
        self,
        env: Go2Task3Env,
        log_dir: str,
        n_stack: int = 5,
        tb_log_interval_steps: int = 50,
    ):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device
        self.tb_log_interval_steps = int(tb_log_interval_steps)

        self.single_obs_dim = int(env.cfg.num_observations)
        self.single_priv_dim = int(env.cfg.num_privileged_obs)
        self.world_priv_dim = self.single_priv_dim - self.single_obs_dim

        expected_single_obs_dim = int(getattr(env.cfg, "num_observations", self.single_obs_dim))
        expected_single_priv_dim = int(getattr(env.cfg, "num_privileged_obs", self.single_priv_dim))

        if self.single_obs_dim != expected_single_obs_dim:
            raise RuntimeError(
                f"Task3 actor single obs dim mismatch: expected {expected_single_obs_dim}, "
                f"got {self.single_obs_dim}"
            )

        if self.single_priv_dim != expected_single_priv_dim:
            raise RuntimeError(
                f"Task3 privileged single obs dim mismatch: expected {expected_single_priv_dim}, "
                f"got {self.single_priv_dim}"
            )

        if self.world_priv_dim <= 0:
            raise RuntimeError(
                f"Task3 world privileged dim must be positive, got {self.world_priv_dim}. "
                f"single_priv_dim={self.single_priv_dim}, single_obs_dim={self.single_obs_dim}"
            )

        self.stacked_obs_dim = self.single_obs_dim * self.n_stack
        self.critic_obs_dim = self.stacked_obs_dim + self.world_priv_dim

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

        self.writer = SummaryWriter(log_dir) if self.tb_log_interval_steps != 0 else None

        self.global_env_steps = 0
        self.local_step_count = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    @property
    def unwrapped(self):
        return self

    def _build_critic_obs(self) -> torch.Tensor:
        raw_priv = self.env.compute_privileged_obs()
        world_priv = raw_priv[:, self.single_obs_dim:]

        if world_priv.shape[-1] != self.world_priv_dim:
            raise RuntimeError(
                f"world_priv dim mismatch: got {world_priv.shape[-1]}, expected {self.world_priv_dim}"
            )

        critic = torch.cat([self.obs_stack, world_priv], dim=-1)

        if critic.shape[-1] != self.critic_obs_dim:
            raise RuntimeError(
                f"critic obs dim mismatch: got {critic.shape[-1]}, expected {self.critic_obs_dim}"
            )

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

        self.global_env_steps += self.num_envs
        self.local_step_count += 1
        self.last_info = info or {}
        self.last_reward_mean = to_float(reward) or 0.0
        self.last_done_count = int(done.sum().detach().cpu().item())

        if (
            self.writer is not None
            and self.tb_log_interval_steps > 0
            and self.local_step_count % self.tb_log_interval_steps == 0
        ):
            write_scalars(self.writer, self.last_info.get("reward_components", {}), self.global_env_steps, "rewards")
            write_scalars(self.writer, self.last_info.get("events", {}), self.global_env_steps, "events")
            write_scalars(self.writer, self.last_info.get("telemetry", {}), self.global_env_steps, "telemetry")
            write_scalars(self.writer, self.last_info.get("debug", {}), self.global_env_steps, "debug")
            self.writer.add_scalar("rollout/reward_mean_raw", self.last_reward_mean, self.global_env_steps)
            self.writer.add_scalar("rollout/done_count", self.last_done_count, self.global_env_steps)

        return self._pack(), reward, terminated, truncated, self.last_info

    def close(self):
        try:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass

        try:
            self.env.close()
        except Exception:
            pass


def make_log_dir() -> str:
    run_name = args_cli.run_name.strip()
    if not run_name:
        run_name = f"go2_task3_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = os.path.abspath(os.path.join(args_cli.log_root, run_name))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _get_curriculum_total_steps(base_env: Go2Task3Env) -> int:
    """Read Task3 curriculum total steps from config with backward-compatible fallbacks."""

    return int(
        getattr(
            base_env.cfg,
            "curriculum_total_steps",
            getattr(base_env.cfg.world_cfg, "curriculum_total_steps", 1),
        )
    )


def _get_stage_cap_from_env(base_env: Go2Task3Env) -> int:
    """Read current global curriculum cap from the env/world implementation."""

    try:
        if hasattr(base_env, "_global_stage_cap"):
            return int(base_env._global_stage_cap())
    except Exception:
        pass

    try:
        if hasattr(base_env, "world") and hasattr(base_env.world, "stage_from_global_steps"):
            return int(base_env.world.stage_from_global_steps(int(base_env.global_steps)))
    except Exception:
        pass

    return int(getattr(base_env, "curriculum_active_stage", 0))


def _apply_curriculum_start_override(
    base_env: Go2Task3Env,
    start_k: float,
    force_stage: int,
    reason: str,
    prefer_max: bool = True,
) -> None:
    """Apply --start-k / --force-stage to the real Task3 env.

    This must run after resume metadata restoration as well, because resume can
    overwrite base_env.global_steps. The wrapper and progress bar counters are
    not enough; curriculum sampling reads the underlying base_env fields.
    """

    changed = False

    if float(start_k) > 0.0:
        curriculum_total_steps = max(_get_curriculum_total_steps(base_env), 1)
        forced_steps = int(float(start_k) * curriculum_total_steps)
        old_steps = int(getattr(base_env, "global_steps", 0))
        new_steps = max(old_steps, forced_steps) if prefer_max else forced_steps
        base_env.global_steps = int(new_steps)
        changed = changed or (new_steps != old_steps)

        print(
            f"[INFO] curriculum override ({reason}): "
            f"start_k={float(start_k):.4f}, forced_steps={forced_steps:,}, "
            f"old_global_steps={old_steps:,}, new_global_steps={int(base_env.global_steps):,}, "
            f"curriculum_total_steps={curriculum_total_steps:,}"
        )

    stage_cap = _get_stage_cap_from_env(base_env)

    if int(force_stage) >= 0:
        requested_stage = int(force_stage)
        max_stage = int(getattr(getattr(base_env, "world", None), "stage_count", requested_stage + 1)) - 1
        if max_stage >= 0:
            requested_stage = max(0, min(requested_stage, max_stage))

        old_active = int(getattr(base_env, "curriculum_active_stage", 0))
        if hasattr(base_env, "curriculum_active_stage"):
            base_env.curriculum_active_stage = requested_stage
            changed = changed or (requested_stage != old_active)

        if hasattr(base_env, "curriculum_stage_start_steps"):
            base_env.curriculum_stage_start_steps = int(getattr(base_env, "global_steps", 0))
        if hasattr(base_env, "curriculum_last_check_steps"):
            base_env.curriculum_last_check_steps = int(getattr(base_env, "global_steps", 0))

        print(
            f"[INFO] curriculum override ({reason}): "
            f"force_stage={requested_stage}, old_active_stage={old_active}, "
            f"global_cap_now={stage_cap}"
        )

    elif changed and hasattr(base_env, "curriculum_active_stage"):
        # When only --start-k is used, promote active stage up to the new cap.
        # This prevents a mature Stage0 checkpoint from staying in Stage0 after
        # resume metadata was restored.
        old_active = int(getattr(base_env, "curriculum_active_stage", 0))
        new_active = max(old_active, int(stage_cap))
        base_env.curriculum_active_stage = int(new_active)

        if new_active != old_active:
            if hasattr(base_env, "curriculum_stage_start_steps"):
                base_env.curriculum_stage_start_steps = int(getattr(base_env, "global_steps", 0))
            if hasattr(base_env, "curriculum_last_check_steps"):
                base_env.curriculum_last_check_steps = int(getattr(base_env, "global_steps", 0))

        print(
            f"[INFO] curriculum override ({reason}): "
            f"active_stage {old_active} -> {new_active}, global_cap_now={stage_cap}"
        )


def task3_progress_postfix(env_steps: int, start_time: float, reward_mean: float, done_count: int, info: Dict[str, Any]):
    flat = flat_dict(info)
    fps = env_steps / max(time.time() - start_time, 1e-6)

    return {
        "steps": f"{env_steps:,}",
        "fps": f"{fps:,.0f}",
        "rew": f"{reward_mean:+.3f}",
        "done": int(done_count),
        "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.1f}",
        "dist": f"{flat.get('telemetry/Distance_To_Goal', 0.0):.2f}",
        "dr": f"{flat.get('telemetry/Distance_Reduction_Ratio', 0.0):+.2f}",
        "prog": f"{flat.get('telemetry/Progress_Step', flat.get('telemetry/Progress', 0.0)):+.3f}",
        "win_succ": f"{flat.get('telemetry/Current_Window_Success_Rate', flat.get('events/Success_Rate', 0.0)):.3f}",
        "coll": f"{flat.get('events/Collision_Rate', 0.0):.3f}",
        "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
        "h": f"{flat.get('telemetry/Base_Height', 0.0):.2f}",
    }


def print_update(pbar, update_id, env_steps, total_steps, elapsed, num_envs, rollouts, info, ppo, lr):
    stat = {
        "update": float(update_id),
        "total_env_steps": float(env_steps),
        "target_env_steps": float(total_steps),
        "progress_percent": 100.0 * env_steps / max(total_steps, 1),
        "num_envs": float(num_envs),
        "rollouts_per_update": float(rollouts),
        "fps_env_steps": env_steps / max(elapsed, 1e-6),
        "learning_rate": lr,
    }

    pbar.write(
        "\n".join(
            [
                "\n" + "=" * 124,
                f"📊 [Go2 Task3 skrl PPO 更新 {update_id}] 总步数: {env_steps:,} / {total_steps:,} | "
                f"环境 FPS: {stat['fps_env_steps']:,.0f} | LR: {lr:.3e}",
                "=" * 124,
                make_table("time / progress", stat),
                make_table("env info: reward_components + events + telemetry + debug", flat_dict(info)),
                make_table("ppo update info", ppo),
                "=" * 124 + "\n",
            ]
        )
    )


def _base_ppo_cfg_dict():
    cfg = PPO_CFG()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return cfg.copy()


def _set_if_supported(cfg: dict, requested: dict) -> None:
    skipped = []
    for key, value in requested.items():
        if key in cfg:
            cfg[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"[WARN] 当前 skrl.PPO_CFG 不支持这些字段，已跳过: {skipped}")


def build_skrl_cfg(env, log_dir):
    cfg = _base_ppo_cfg_dict()

    requested = {
        "rollouts": int(args_cli.rollouts),
        "learning_epochs": int(args_cli.learning_epochs),
        "mini_batches": int(args_cli.mini_batches),
        "discount_factor": float(args_cli.gamma),
        "gae_lambda": float(args_cli.gae_lambda),
        "learning_rate": float(args_cli.lr),
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {
            "kl_threshold": float(args_cli.kl_threshold),
            "min_lr": float(args_cli.min_lr),
            "max_lr": float(args_cli.max_lr),
        },
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
        "grad_norm_clip": float(args_cli.grad_clip),
        "ratio_clip": float(args_cli.ratio_clip),
        "value_clip": float(args_cli.value_clip),
        "entropy_loss_scale": float(args_cli.entropy_coef),
        "value_loss_scale": float(args_cli.value_coef),
    }

    _set_if_supported(cfg, requested)

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": log_dir,
            "experiment_name": "go2_task3_skrl",
            "write_interval": int(args_cli.skrl_write_interval),
            "checkpoint_interval": int(args_cli.skrl_checkpoint_interval),
            "store_separately": True,
            "wandb": False,
        }
    )

    return cfg


def _resolve_checkpoint_file(path: str, default_name: str = "") -> str:
    if not path:
        return ""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    if p.is_dir():
        names = []
        if default_name:
            names.append(default_name)

        names.extend(
            [
                "go2_task3_model.pt",
                "go2_task2_model.pt",
                "go2_task1_model.pt",
                "agent.pt",
                "checkpoint.pt",
                "best_agent.pt",
            ]
        )

        for name in names:
            cand = p / name
            if cand.exists():
                return str(cand)

        final_candidates = [
            p / "final_checkpoint" / "go2_task3_model.pt",
            p / "final_checkpoint" / "go2_task2_model.pt",
            p / "final_checkpoint" / "go2_task1_model.pt",
        ]
        for cand in final_candidates:
            if cand.exists():
                return str(cand)

    return str(p)


def _extract_policy_state(raw: Any) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(raw, dict):
        return None

    if "policy" in raw and isinstance(raw["policy"], dict):
        return raw["policy"]

    if "models" in raw and isinstance(raw["models"], dict):
        if "policy" in raw["models"] and isinstance(raw["models"]["policy"], dict):
            return raw["models"]["policy"]

    if "model" in raw and isinstance(raw["model"], dict):
        state = {}
        for k, v in raw["model"].items():
            if str(k).startswith("policy."):
                state[str(k).replace("policy.", "", 1)] = v
            elif str(k).startswith("actor."):
                state[str(k).replace("actor.", "", 1)] = v
        if state:
            return state

    expected_fragments = ["net.", "network.", "actor.", "log_std_parameter"]
    if any(any(fragment in str(k) for fragment in expected_fragments) for k in raw.keys()):
        return raw

    return None


def _find_first_linear_weight_key(state: Dict[str, torch.Tensor]) -> Optional[str]:
    candidates = []
    for k, v in state.items():
        if torch.is_tensor(v) and v.ndim == 2:
            candidates.append((str(k), v.shape[0], v.shape[1]))

    if not candidates:
        return None

    # Prefer the largest input dimension first-layer candidate.
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0]


def build_old_to_task3_column_mapping(old_single: int, new_single: int = 0, n_stack: int = 5):
    """Map Task1/Task2 stacked obs columns to Task3 stacked obs columns.

    Old Task1/Task2 single-frame obs layout assumed:
        0:3   base_lin_vel
        3:6   base_ang_vel
        6:9   projected_gravity
        9:12  command [vx, vy, wz]
        12:24 joint_pos_error
        24:36 joint_vel
        36:48 last_action
        48:60 action_delta
        60:64 foot_contact
        64:76 foot_rel_pos
        76:84 foot_vel_xy
        84:85 base_height
        85:87 phase

    Task3 single-frame obs layout:
        0:3     base_lin_vel
        3:6     base_ang_vel
        6:9     projected_gravity
        9:12    target_obs [dist_norm, sin(angle), cos(angle)]
        12:13   target_speed
        13:14   progress_ema
        14:26   joint_pos_error
        26:38   joint_vel
        38:50   last_action
        50:62   action_delta
        62:66   foot_contact
        66:156  lidar
        156:246 lidar_delta
        246:254 risk
        legacy base height / phase fields
    """

    pairs = []

    def add_range(old_start, new_start, length, frame):
        old_base = frame * old_single
        new_base = frame * new_single
        for i in range(length):
            pairs.append((old_base + old_start + i, new_base + new_start + i))

    def add_col(old_col, new_col, frame):
        old_base = frame * old_single
        new_base = frame * new_single
        pairs.append((old_base + old_col, new_base + new_col))

    for f in range(n_stack):
        # Directly aligned proprioceptive inputs.
        add_range(0, 0, 9, f)

        # Old command -> Task3 target hints.
        if old_single >= 12:
            add_col(9, 11, f)    # cmd_vx -> target cos(angle), forward prior
            add_col(9, 12, f)    # cmd_vx -> target_speed
            add_col(11, 10, f)   # cmd_wz -> target sin(angle)

        # Joint / action / contact.
        add_range(12, 14, 12, f)
        add_range(24, 26, 12, f)
        add_range(36, 38, 12, f)
        add_range(48, 50, 12, f)
        add_range(60, 62, 4, f)

        # Height / phase.
        if old_single >= 87:
            add_col(84, 254, f)
            add_range(85, 255, 2, f)

    return pairs


def smart_copy_first_layer(dst_weight: torch.Tensor, src_weight: torch.Tensor, n_stack: int = 5) -> str:
    """Legacy first-layer copy helper.

    Task3-Navigation-V3 disables Task1/Task2 warm-start because the actor
    observation layout is navigation-specific. This helper is kept only for
    backward compatibility and should not be used for V3 experiments.
    """

    with torch.no_grad():
        if tuple(src_weight.shape) == tuple(dst_weight.shape):
            dst_weight.copy_(src_weight)
            return "exact"

        if src_weight.ndim != 2 or dst_weight.ndim != 2:
            return "not_2d"

        if src_weight.shape[0] != dst_weight.shape[0]:
            return f"row_mismatch_src{tuple(src_weight.shape)}_dst{tuple(dst_weight.shape)}"

        dst_weight.zero_()

        src_input_dim = int(src_weight.shape[1])
        dst_input_dim = int(dst_weight.shape[1])

        # V3 does not support old Task1/Task2 column mapping.
        return f"legacy_smart_copy_disabled_for_dst_input_dim_{dst_input_dim}"

        if src_input_dim % n_stack != 0:
            return f"src_input_dim_not_divisible_by_stack_{src_input_dim}"

        old_single = src_input_dim // n_stack
        mapping = build_old_to_task3_column_mapping(old_single=old_single, new_single=dst_input_dim // n_stack, n_stack=n_stack)

        copied_cols = 0
        for old_col, new_col in mapping:
            if old_col < src_weight.shape[1] and new_col < dst_weight.shape[1]:
                dst_weight[:, new_col] = src_weight[:, old_col]
                copied_cols += 1

        return f"smart_old_single={old_single}_cols={copied_cols}"


def smart_copy_policy_state(dst_state: Dict[str, torch.Tensor], src_state: Dict[str, torch.Tensor], device: str, n_stack: int = 5):
    """Return copied policy state and copy report."""

    new_state = {}
    copied = 0
    exact = 0
    smart = 0
    skipped = 0
    smart_mode = "not_used"

    dst_first_key = _find_first_linear_weight_key(dst_state)
    src_first_key = _find_first_linear_weight_key(src_state)

    for k, dst_v in dst_state.items():
        src_v = src_state.get(k, None)

        if src_v is not None and torch.is_tensor(src_v) and tuple(src_v.shape) == tuple(dst_v.shape):
            new_state[k] = src_v.to(device)
            copied += 1
            exact += 1
            continue

        # If names differ, still try first-layer smart mapping by shape.
        if k == dst_first_key and src_first_key is not None:
            src_first = src_state[src_first_key]
            if torch.is_tensor(src_first):
                candidate = dst_v.clone()
                mode = smart_copy_first_layer(candidate, src_first.to(device), n_stack=n_stack)
                if mode.startswith("exact") or mode.startswith("smart_"):
                    new_state[k] = candidate
                    copied += 1
                    smart += 1
                    smart_mode = f"{mode}, src_key={src_first_key}, dst_key={dst_first_key}"
                    continue
                smart_mode = mode

        new_state[k] = dst_v
        skipped += 1

    return new_state, {
        "copied": copied,
        "exact": exact,
        "smart": smart,
        "skipped": skipped,
        "total": len(dst_state),
        "smart_mode": smart_mode,
    }


def _try_set_policy_log_std(policy, value: float) -> bool:
    try:
        if hasattr(policy, "log_std_parameter"):
            with torch.no_grad():
                policy.log_std_parameter.fill_(float(value))
            return True

        sd = policy.state_dict()
        for k in sd.keys():
            if "log_std" in str(k):
                with torch.no_grad():
                    obj = policy
                    parts = str(k).split(".")
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    param = getattr(obj, parts[-1])
                    param.fill_(float(value))
                return True
    except Exception:
        pass

    return False


def load_actor_warm_start(models: Dict[str, Any], path: str, device: str, label: str, pretrained_log_std: float) -> bool:
    # Task3-Navigation-V3 uses a new actor observation layout and must be
    # trained from scratch. Keep this function as a compatibility stub so old
    # command lines fail safely instead of silently loading incompatible priors.
    if not path:
        print(f"[INFO] 未指定 {label} actor warm-start。")
        return False

    print(
        f"[WARN] Task3-Navigation-V3 不支持 {label} actor warm-start，"
        "已跳过。请从 scratch 训练，或使用 --resume 加载 V3 checkpoint。"
    )
    return False

    default_name = "go2_task2_model.pt" if "Task2" in label else "go2_task1_model.pt"
    path = _resolve_checkpoint_file(path, default_name=default_name)

    if not path:
        print(f"[INFO] 未指定 {label} actor warm-start。")
        return False

    if not os.path.exists(path):
        print(f"[WARN] {label} warm-start checkpoint 不存在: {path}")
        return False

    print("\n" + "=" * 124)
    print(f"🔁 尝试加载 {label} actor warm-start: {path}")
    print("=" * 124)

    try:
        raw = torch.load(path, map_location=device)
        src_policy = _extract_policy_state(raw)
        if src_policy is None:
            print("[WARN] checkpoint 中没有可识别的 policy state_dict，跳过 warm-start。")
            return False

        policy = models["policy"]
        dst_state = policy.state_dict()

        new_state, report = smart_copy_policy_state(dst_state, src_policy, device=device, n_stack=5)
        policy.load_state_dict(new_state)

        log_std_ok = _try_set_policy_log_std(policy, float(pretrained_log_std))

        print(f"✅ {label} actor warm-start 完成")
        print(f"   copied_policy_tensors = {report['copied']}/{report['total']}")
        print(f"   exact tensors         = {report['exact']}")
        print(f"   smart tensors         = {report['smart']}")
        print(f"   skipped tensors       = {report['skipped']}")
        print(f"   first_layer_mode      = {report['smart_mode']}")
        print(f"   policy log_std set    = {log_std_ok}, value={pretrained_log_std}")
        print("   注意：Task3 critic 使用 world privileged obs，保持随机初始化。")
        return int(report["copied"]) > 0

    except Exception as exc:
        print(f"[WARN] {label} warm-start 失败: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def save_train_metadata(path, env_steps, num_envs, base_env, wrapped_env):
    torch.save(
        {
            "stage": "unitree_go2_task3_navigation_v3",
            "algorithm": "skrl_ppo",
            "global_env_steps": int(env_steps),
            "num_envs": int(num_envs),
            "single_actor_obs_dim": int(base_env.cfg.num_observations),
            "single_privileged_obs_dim": int(base_env.cfg.num_privileged_obs),
            "world_priv_dim": int(wrapped_env.world_priv_dim),
            "actor_obs_dim": int(wrapped_env.observation_space.shape[0]),
            "critic_obs_dim": int(wrapped_env.state_space.shape[0]),
            "num_actions": int(wrapped_env.action_space.shape[0]),
            "frame_stack": int(wrapped_env.n_stack),
            "asymmetric_critic": True,
            "critic_layout": f"actor_obs_stack_{int(wrapped_env.observation_space.shape[0])} + world_priv_{int(wrapped_env.world_priv_dim)} = {int(wrapped_env.state_space.shape[0])}",
            "action_joint_names": list(base_env.cfg.action_joint_names),
            "foot_body_names": list(base_env.cfg.foot_body_names),
            "world_cfg": {
                "num_lidar_rays": int(getattr(base_env.cfg.world_cfg, "num_lidar_rays", -1)),
                "max_static_obs": int(getattr(base_env.cfg.world_cfg, "max_static_obs", -1)),
                "max_dynamic_obs": int(getattr(base_env.cfg.world_cfg, "max_dynamic_obs", -1)),
                "env_size": float(getattr(base_env.cfg.world_cfg, "env_size", 0.0)),
                "curriculum_total_steps": int(getattr(base_env.cfg.world_cfg, "curriculum_total_steps", 0)),
            },
        },
        os.path.join(path, "train_metadata.pt"),
    )


def main():
    set_seed(int(args_cli.seed))

    log_dir = make_log_dir()

    print("\n" + "=" * 124)
    print("🚀 Unitree Go2 Task3: Navigation / Obstacle Avoidance / Running skrl PPO 训练启动")
    print("=" * 124)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_dir      = {log_dir}")
    print(f"[INFO] device       = {args_cli.device}")
    print(f"[INFO] TF32 enabled = {getattr(torch.backends.cuda.matmul, 'allow_tf32', False)}")

    env_cfg = Task3Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.print_debug_info = False

    base_env = Go2Task3Env(env_cfg)

    _apply_curriculum_start_override(
        base_env,
        start_k=float(args_cli.start_k),
        force_stage=int(args_cli.force_stage),
        reason="env_init",
        prefer_max=False,
    )

    stacked_env = Go2Task3AsymFrameStackWrapper(
        base_env,
        log_dir=log_dir,
        n_stack=5,
        tb_log_interval_steps=int(args_cli.tb_log_interval_steps),
    )

    env = wrap_env(stacked_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", stacked_env.num_envs)

    print("\n[DEBUG] Go2 Task3 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    expected_actor_dim = int(base_env.cfg.num_observations) * int(stacked_env.n_stack)
    expected_critic_dim = expected_actor_dim + int(stacked_env.world_priv_dim)

    if int(env.observation_space.shape[0]) != expected_actor_dim:
        raise RuntimeError(
            f"Task3 policy input dim mismatch: expected {expected_actor_dim}, "
            f"got {env.observation_space.shape[0]}"
        )
    if int(env.state_space.shape[0]) != expected_critic_dim:
        raise RuntimeError(
            f"Task3 critic input dim mismatch: expected {expected_critic_dim}, "
            f"got {env.state_space.shape[0]}"
        )

    models = {
        "policy": Go2Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=float(args_cli.init_log_std),
            min_log_std=float(args_cli.min_log_std),
            max_log_std=float(args_cli.max_log_std),
        ),
        "value": Go2Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    if not args_cli.resume:
        # Task3-Navigation-V3 changes the actor observation layout, so Task1/Task2
        # actor warm-start is intentionally disabled. Use scratch training for V3.
        if args_cli.pretrained_task1 or args_cli.pretrained_task2:
            print(
                "[WARN] Task3-Navigation-V3 uses a new observation layout; "
                "--pretrained-task1/--pretrained-task2 are ignored. "
                "Start from scratch, or use --resume only with a V3 checkpoint."
            )

    total_env_steps = int(args_cli.total_env_steps)
    total_vector_steps = math.ceil(total_env_steps / num_envs)
    save_freq_env_steps = int(args_cli.save_freq_env_steps)

    cfg = build_skrl_cfg(env, log_dir)
    update_env_steps = int(cfg["rollouts"]) * int(num_envs)

    print("\n[INFO] Go2 Task3 训练配置")
    print(f"  - num_envs             : {num_envs:,}")
    print(f"  - total_env_steps      : {total_env_steps:,}")
    print(f"  - total_vector_steps   : {total_vector_steps:,}")
    print(f"  - rollouts             : {cfg['rollouts']}")
    print(f"  - learning_epochs      : {cfg['learning_epochs']}")
    print(f"  - mini_batches         : {cfg['mini_batches']}")
    print(f"  - update_env_steps     : {update_env_steps:,}")
    print(f"  - save_freq_env_steps  : {save_freq_env_steps:,}")
    print(f"  - frame_stack          : 5")
    print(f"  - single_actor_obs_dim : {base_env.cfg.num_observations}")
    print(f"  - world_priv_dim       : {stacked_env.world_priv_dim}")
    print(f"  - actor obs dim        : {env.observation_space.shape[0]}")
    print(f"  - critic obs dim       : {env.state_space.shape[0]}")
    print(f"  - action dim           : {env.action_space.shape[0]}")
    print(f"  - lr/min/max           : {args_cli.lr} / {args_cli.min_lr} / {args_cli.max_lr}")
    print(f"  - gamma                : {args_cli.gamma}")
    print(f"  - entropy_coef         : {args_cli.entropy_coef}")
    print(f"  - init_log_std         : {args_cli.init_log_std}")
    print(f"  - pretrained_task1     : {args_cli.pretrained_task1 if args_cli.pretrained_task1 else '<none>'}")
    print(f"  - pretrained_task2     : {args_cli.pretrained_task2 if args_cli.pretrained_task2 else '<none>'}")
    print(f"  - resume               : {args_cli.resume if args_cli.resume else '<none>'}")
    print(f"  - start_k              : {args_cli.start_k}")
    print(f"  - force_stage          : {args_cli.force_stage if args_cli.force_stage >= 0 else '<none>'}")
    print(f"  - tensorboard          : tensorboard --logdir={args_cli.log_root}")

    memory = RandomMemory(memory_size=int(cfg["rollouts"]), num_envs=num_envs, device=env.device)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )

    resume_env_steps = 0
    if args_cli.resume:
        resume_path = _resolve_checkpoint_file(args_cli.resume, default_name="go2_task3_model.pt")
        print(f"[INFO] resume skrl checkpoint: {resume_path}")
        agent.load(resume_path)

        metadata_path = Path(resume_path).parent / "train_metadata.pt"
        if metadata_path.exists():
            try:
                meta = torch.load(str(metadata_path), map_location="cpu")
                resume_env_steps = int(meta.get("global_env_steps", 0))
                base_env.global_steps = resume_env_steps
                print(f"[INFO] restored global_env_steps from metadata: {resume_env_steps:,}")
            except Exception as exc:
                print(f"[WARN] metadata 恢复失败: {type(exc).__name__}: {exc}")

    _apply_curriculum_start_override(
        base_env,
        start_k=float(args_cli.start_k),
        force_stage=int(args_cli.force_stage),
        reason="after_resume_metadata",
        prefer_max=True,
    )

    trainer = StepTrainer(
        cfg={
            "timesteps": int(total_vector_steps),
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n🔥 [Go2 Task3 skrl PPO 已点火]")
    print("👉 训练目标：Task3-Navigation-V3 专用导航策略，从 scratch 学习进目标圈 + 避障 + 稳定运动。")
    print(
        f"👉 Actor 输入：{base_env.cfg.num_observations} 维单帧观测 × "
        f"{stacked_env.n_stack} 帧堆叠 = {env.observation_space.shape[0]}。"
    )
    print(
        f"👉 Critic 输入：actor_stack {env.observation_space.shape[0]} + "
        f"world_priv {stacked_env.world_priv_dim} = {env.state_space.shape[0]}。"
    )
    print("👉 V3 不使用 Task1/Task2 warm-start；如需继续训练，只能 resume V3 checkpoint。")
    print("👉 日志重点：Current_Window_Success_Rate / Distance_Reduction_Ratio / Near_Goal_Rate / Timeout_Final_Distance。\n")

    last_save = resume_env_steps
    update_id = 0
    start = time.time()

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=min(resume_env_steps, total_env_steps),
            desc="Go2 Task3 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                trainer.train(timestep=t, timesteps=total_vector_steps)

                env_steps = min(resume_env_steps + (t + 1) * num_envs, total_env_steps)
                previous_env_steps = min(resume_env_steps + t * num_envs, total_env_steps)
                pbar.update(env_steps - previous_env_steps)

                pbar.set_postfix(
                    task3_progress_postfix(
                        env_steps=env_steps,
                        start_time=start,
                        reward_mean=stacked_env.last_reward_mean,
                        done_count=stacked_env.last_done_count,
                        info=stacked_env.last_info,
                    )
                )

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1
                    ppo_info = tracking_mean(agent)
                    ppo_info["learning_rate"] = current_lr(agent)

                    writer = getattr(agent, "writer", None)
                    write_scalars(writer, ppo_info, env_steps, "ppo")
                    write_scalars(writer, flat_dict(stacked_env.last_info), env_steps, "env_info")

                    if update_id % max(int(args_cli.summary_interval), 1) == 0:
                        print_update(
                            pbar,
                            update_id,
                            env_steps,
                            total_env_steps,
                            time.time() - start,
                            num_envs,
                            cfg["rollouts"],
                            stacked_env.last_info,
                            ppo_info,
                            ppo_info["learning_rate"],
                        )

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                if env_steps - last_save >= save_freq_env_steps:
                    last_save = env_steps
                    save_dir = os.path.join(log_dir, f"checkpoint_{env_steps}")
                    os.makedirs(save_dir, exist_ok=True)

                    try:
                        agent.save(os.path.join(save_dir, "go2_task3_model.pt"))
                        save_normalizers(agent, save_dir)
                        save_train_metadata(save_dir, env_steps, num_envs, base_env, stacked_env)
                        pbar.write(f"\n💾 [Go2 Task3 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

                if env_steps >= total_env_steps:
                    break

    except KeyboardInterrupt:
        print("\n[WARN] 接收到手动中断信号，正在安全保存...")
    except Exception:
        print("\n[ERROR] Go2 Task3 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_dir, "final_checkpoint")
        os.makedirs(final_dir, exist_ok=True)

        final_env_steps = min(resume_env_steps + total_vector_steps * num_envs, total_env_steps)

        try:
            agent.save(os.path.join(final_dir, "go2_task3_model.pt"))
            save_normalizers(agent, final_dir)
            save_train_metadata(final_dir, final_env_steps, num_envs, base_env, stacked_env)
            print(f"✅ Go2 Task3 模型与归一化统计已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终模型失败: {type(exc).__name__}: {exc}")

        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass

        print("✅ Go2 Task3 skrl PPO 训练管线安全退出")


if __name__ == "__main__":
    main()