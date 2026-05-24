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
from typing import Any, Dict, Optional

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


parser = argparse.ArgumentParser(description="Train Unitree Go2 Task2 Multi-Terrain PPO with skrl")

# Runtime
parser.add_argument("--total-env-steps", type=int, default=600_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=20_000_000)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)

# Resume / warm-start
parser.add_argument("--resume", type=str, default="", help="Optional Task2 skrl checkpoint file or checkpoint directory")
parser.add_argument("--pretrained-task1", type=str, default="", help="Optional Task1 skrl checkpoint file or directory for actor warm-start")
parser.add_argument("--load-task1-obs-norm", action="store_true", help="Load Task1 observation normalizer if shape matches")
parser.add_argument("--pretrained-log-std", type=float, default=-1.65)

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
parser.add_argument("--entropy-coef", type=float, default=0.003)
parser.add_argument("--value-coef", type=float, default=2.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--ratio-clip", "--clip-range", dest="ratio_clip", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0.2)
parser.add_argument("--init-log-std", type=float, default=-1.25)
parser.add_argument("--min-log-std", type=float, default=-5.0)
parser.add_argument("--max-log-std", type=float, default=0.3)

# Logging / checkpoint
parser.add_argument("--log-root", type=str, default=os.environ.get("RT_GO2_TASK2_LOG_ROOT", default_log_root("task2")))
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
from go2_rl.tasks.task2.task2_config import Task2Config
from go2_rl.tasks.task2.task2_env import Go2Task2Env


class Go2Task2AsymFrameStackWrapper(gym.Env):
    """Task2 asymmetric frame-stack wrapper for skrl.

    Actor:
        actor_obs_stack = 87 * 5 = 435

    Critic:
        critic_obs = actor_obs_stack 435 + terrain_priv 91 = 526

    Raw env.compute_privileged_obs():
        single_actor_obs 87 + terrain_priv 91 = 178

    The wrapper returns a dict compatible with skrl IsaacLab wrapper:
        {"policy": actor_stack, "critic": critic_obs}
    """

    def __init__(
        self,
        env: Go2Task2Env,
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
        self.terrain_priv_dim = self.single_priv_dim - self.single_obs_dim

        if self.single_obs_dim != 87:
            raise RuntimeError(f"Task2 actor single obs dim should be 87, got {self.single_obs_dim}")
        if self.terrain_priv_dim != 91:
            raise RuntimeError(f"Task2 terrain priv dim should be 91, got {self.terrain_priv_dim}")

        self.stacked_obs_dim = self.single_obs_dim * self.n_stack
        self.critic_obs_dim = self.stacked_obs_dim + self.terrain_priv_dim

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
        terrain_priv = raw_priv[:, self.single_obs_dim:]

        if terrain_priv.shape[-1] != self.terrain_priv_dim:
            raise RuntimeError(
                f"terrain_priv dim mismatch: got {terrain_priv.shape[-1]}, expected {self.terrain_priv_dim}"
            )

        critic = torch.cat([self.obs_stack, terrain_priv], dim=-1)

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
            write_scalars(self.writer, self.last_info.get("curriculum", {}), self.global_env_steps, "curriculum")
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
        run_name = f"go2_task2_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = os.path.abspath(os.path.join(args_cli.log_root, run_name))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def task2_progress_postfix(env_steps: int, start_time: float, reward_mean: float, done_count: int, info: Dict[str, Any]):
    flat = flat_dict(info)
    fps = env_steps / max(time.time() - start_time, 1e-6)

    return {
        "steps": f"{env_steps:,}",
        "fps": f"{fps:,.0f}",
        "rew": f"{reward_mean:.3f}",
        "done": int(done_count),
        "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.0f}",
        "level": f"{flat.get('telemetry/Mean_Terrain_Level', 0.0):.2f}",
        "type": f"{flat.get('telemetry/Mean_Terrain_Type', 0.0):.2f}",
        "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):.2f}/{flat.get('telemetry/Cmd_Vx', 0.0):.2f}",
        "mu": f"{flat.get('telemetry/Mean_Friction', 0.0):.2f}",
        "h": f"{flat.get('telemetry/Base_Height', 0.0):.2f}",
        "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
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
                "\n" + "=" * 116,
                f"📊 [Go2 Task2 skrl PPO 更新 {update_id}] 总步数: {env_steps:,} / {total_steps:,} | "
                f"环境 FPS: {stat['fps_env_steps']:,.0f} | LR: {lr:.3e}",
                "=" * 116,
                make_table("time / progress", stat),
                make_table("env info: reward_components + events + telemetry + curriculum + debug", flat_dict(info)),
                make_table("ppo update info", ppo),
                "=" * 116 + "\n",
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
            "experiment_name": "go2_task2_skrl",
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
            if k.startswith("policy."):
                state[k.replace("policy.", "", 1)] = v
        if state:
            return state

    expected_fragments = ["net.", "log_std_parameter"]
    if any(any(fragment in str(k) for fragment in expected_fragments) for k in raw.keys()):
        return raw

    return None


def load_task1_actor_warm_start(models: Dict[str, Any], path: str, device: str, pretrained_log_std: float) -> bool:
    path = _resolve_checkpoint_file(path, default_name="go2_task1_model.pt")
    if not path:
        print("[INFO] 未指定 Task1 actor warm-start，从随机初始化开始。")
        return False

    if not os.path.exists(path):
        print(f"[WARN] Task1 warm-start checkpoint 不存在: {path}")
        return False

    print("\n" + "=" * 112)
    print(f"🔁 尝试加载 Task1 actor warm-start: {path}")
    print("=" * 112)

    try:
        raw = torch.load(path, map_location=device)
        src_policy = _extract_policy_state(raw)
        if src_policy is None:
            print("[WARN] checkpoint 中没有可识别的 policy state_dict，跳过 warm-start。")
            return False

        policy = models["policy"]
        dst_state = policy.state_dict()

        new_state = {}
        copied = 0
        total = len(dst_state)

        for k, dst_v in dst_state.items():
            src_v = src_policy.get(k, None)
            if src_v is not None and tuple(src_v.shape) == tuple(dst_v.shape):
                new_state[k] = src_v.to(device)
                copied += 1
            else:
                new_state[k] = dst_v

        policy.load_state_dict(new_state)

        if hasattr(policy, "log_std_parameter"):
            with torch.no_grad():
                policy.log_std_parameter.fill_(float(pretrained_log_std))

        print(f"✅ Task1 actor warm-start 完成: copied_policy_tensors={copied}/{total}")
        print(f"✅ policy log_std 已设置为 {pretrained_log_std}")
        print("注意：Task2 critic 使用 terrain privileged obs，保持随机初始化。")
        return copied > 0

    except Exception as exc:
        print(f"[WARN] Task1 warm-start 失败: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def load_task1_observation_normalizer(agent, path: str) -> None:
    if not args_cli.load_task1_obs_norm:
        return

    path = _resolve_checkpoint_file(path, default_name="go2_task1_model.pt")
    parent = Path(path).expanduser().resolve().parent

    candidates = [
        parent / "observation_preprocessor.pt",
        parent / "_observation_preprocessor.pt",
    ]

    preprocessor = getattr(agent, "observation_preprocessor", None)
    if preprocessor is None:
        preprocessor = getattr(agent, "_observation_preprocessor", None)

    if preprocessor is None:
        print("[WARN] 当前 skrl agent 没有 observation_preprocessor，无法加载 Task1 obs norm。")
        return

    for cand in candidates:
        if not cand.exists():
            continue
        try:
            state = torch.load(str(cand), map_location=getattr(preprocessor, "device", "cpu"))
            preprocessor.load_state_dict(state)
            print(f"✅ 已加载 Task1 observation normalizer: {cand}")
            return
        except Exception as exc:
            print(f"[WARN] Task1 obs norm 加载失败: {cand} | {type(exc).__name__}: {exc}")

    print("[WARN] 未找到可加载的 Task1 observation normalizer。")


def save_train_metadata(path, env_steps, num_envs, base_env, wrapped_env):
    torch.save(
        {
            "stage": "unitree_go2_task2_multi_terrain",
            "algorithm": "skrl_ppo",
            "global_env_steps": int(env_steps),
            "num_envs": int(num_envs),
            "single_actor_obs_dim": int(base_env.cfg.num_observations),
            "single_privileged_obs_dim": int(base_env.cfg.num_privileged_obs),
            "terrain_priv_dim": int(wrapped_env.terrain_priv_dim),
            "actor_obs_dim": int(wrapped_env.observation_space.shape[0]),
            "critic_obs_dim": int(wrapped_env.state_space.shape[0]),
            "num_actions": int(wrapped_env.action_space.shape[0]),
            "frame_stack": int(wrapped_env.n_stack),
            "asymmetric_critic": True,
            "critic_layout": "actor_obs_stack_435 + terrain_priv_91 = 526",
            "action_joint_names": list(base_env.cfg.action_joint_names),
            "foot_body_names": list(base_env.cfg.foot_body_names),
        },
        os.path.join(path, "train_metadata.pt"),
    )


def main():
    set_seed(int(args_cli.seed))

    log_dir = make_log_dir()

    print("\n" + "=" * 116)
    print("🚀 Unitree Go2 Task2: Multi-Terrain / Multi-Material skrl PPO 训练启动")
    print("=" * 116)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_dir      = {log_dir}")
    print(f"[INFO] device       = {args_cli.device}")
    print(f"[INFO] TF32 enabled = {getattr(torch.backends.cuda.matmul, 'allow_tf32', False)}")

    env_cfg = Task2Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.print_debug_info = False

    base_env = Go2Task2Env(env_cfg)

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * base_env.cfg.terrain_curriculum_total_steps)
        print(
            f"[INFO] 已设置初始课程进度 start_k={args_cli.start_k:.4f}, "
            f"global_steps={base_env.global_steps:,}"
        )

    stacked_env = Go2Task2AsymFrameStackWrapper(
        base_env,
        log_dir=log_dir,
        n_stack=5,
        tb_log_interval_steps=int(args_cli.tb_log_interval_steps),
    )

    env = wrap_env(stacked_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", stacked_env.num_envs)

    print("\n[DEBUG] Go2 Task2 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    if int(env.observation_space.shape[0]) != 435:
        raise RuntimeError(f"Task2 policy input dim should be 435, got {env.observation_space.shape[0]}")
    if int(env.state_space.shape[0]) != 526:
        raise RuntimeError(f"Task2 critic input dim should be 526, got {env.state_space.shape[0]}")

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

    if args_cli.pretrained_task1 and not args_cli.resume:
        load_task1_actor_warm_start(
            models=models,
            path=args_cli.pretrained_task1,
            device=env.device,
            pretrained_log_std=float(args_cli.pretrained_log_std),
        )

    total_env_steps = int(args_cli.total_env_steps)
    total_vector_steps = math.ceil(total_env_steps / num_envs)
    save_freq_env_steps = int(args_cli.save_freq_env_steps)

    cfg = build_skrl_cfg(env, log_dir)
    update_env_steps = int(cfg["rollouts"]) * int(num_envs)

    print("\n[INFO] Go2 Task2 训练配置")
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
    print(f"  - terrain_priv_dim     : {stacked_env.terrain_priv_dim}")
    print(f"  - actor obs dim        : {env.observation_space.shape[0]}")
    print(f"  - critic obs dim       : {env.state_space.shape[0]}")
    print(f"  - action dim           : {env.action_space.shape[0]}")
    print(f"  - lr/min/max           : {args_cli.lr} / {args_cli.min_lr} / {args_cli.max_lr}")
    print(f"  - entropy_coef         : {args_cli.entropy_coef}")
    print(f"  - pretrained_task1     : {args_cli.pretrained_task1 if args_cli.pretrained_task1 else '<none>'}")
    print(f"  - resume               : {args_cli.resume if args_cli.resume else '<none>'}")
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

    if args_cli.pretrained_task1 and not args_cli.resume:
        load_task1_observation_normalizer(agent, args_cli.pretrained_task1)

    resume_env_steps = 0
    if args_cli.resume:
        resume_path = _resolve_checkpoint_file(args_cli.resume, default_name="go2_task2_model.pt")
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

    trainer = StepTrainer(
        cfg={
            "timesteps": int(total_vector_steps),
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n🔥 [Go2 Task2 skrl PPO 已点火]")
    print("👉 训练目标：平地 locomotion -> 多地形 rough/slopes/stones/stairs -> 多材质鲁棒行走/奔跑")
    print("👉 Actor 输入：87 维本体感觉 × 5 帧堆叠 = 435")
    print("👉 Critic 输入：actor_stack 435 + terrain_priv 91 = 526")
    print("👉 推荐流程：先 smoke，再 laptop 小规模，再 Windows 3090 放大并发。")
    print("👉 日志重点：Fall_Rate / Actual_Vx-Cmd_Vx / Mean_Terrain_Level / Mean_Friction / P_Foot_Slip\n")

    last_save = resume_env_steps
    update_id = 0
    start = time.time()

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=min(resume_env_steps, total_env_steps),
            desc="Go2 Task2 skrl PPO",
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
                    task2_progress_postfix(
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
                        agent.save(os.path.join(save_dir, "go2_task2_model.pt"))
                        save_normalizers(agent, save_dir)
                        save_train_metadata(save_dir, env_steps, num_envs, base_env, stacked_env)
                        pbar.write(f"\n💾 [Go2 Task2 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

                if env_steps >= total_env_steps:
                    break

    except KeyboardInterrupt:
        print("\n[WARN] 接收到手动中断信号，正在安全保存...")
    except Exception:
        print("\n[ERROR] Go2 Task2 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_dir, "final_checkpoint")
        os.makedirs(final_dir, exist_ok=True)

        final_env_steps = min(resume_env_steps + total_vector_steps * num_envs, total_env_steps)

        try:
            agent.save(os.path.join(final_dir, "go2_task2_model.pt"))
            save_normalizers(agent, final_dir)
            save_train_metadata(final_dir, final_env_steps, num_envs, base_env, stacked_env)
            print(f"✅ Go2 Task2 模型与归一化统计已保存至 {final_dir}")
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

        print("✅ Go2 Task2 skrl PPO 训练管线安全退出")


if __name__ == "__main__":
    main()
