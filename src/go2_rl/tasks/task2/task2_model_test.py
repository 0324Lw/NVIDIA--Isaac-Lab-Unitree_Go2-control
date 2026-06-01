from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

parser = argparse.ArgumentParser(description="Evaluate Unitree Go2 Task2 skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--print-interval", type=int, default=100)
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--visualize", action="store_true", help="Compatibility flag; GUI is enabled by default")
parser.add_argument("--headless-eval", action="store_true", help="Run model evaluation without Isaac Sim GUI")
parser.add_argument("--no-close-on-exit", action="store_true", help="Debug only: do not explicitly call None if bool(getattr(args_cli, 'no_close_on_exit', False)) else simulation_app.close()")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Windows GUI eval policy:
# Model evaluation opens Isaac Sim GUI by default.
# Use --headless-eval only when a non-visual evaluation is explicitly needed.
if bool(getattr(args_cli, "headless_eval", False)):
    # GUI eval default: do not force headless=True here.
    args_cli.headless = False
else:
    args_cli.headless = False
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from go2_rl.common.go2_skrl_models import Go2Actor, Go2Critic
from go2_rl.common.info_utils import flat_dict, load_normalizers, to_float
from go2_rl.common.eval_curriculum_utils import force_eval_curriculum
from go2_rl.common.model_eval_utils import direct_policy_action, init_agent_compat
from go2_rl.tasks.task2.task2_config import Task2Config
from go2_rl.tasks.task2.task2_env import Go2Task2Env


class Go2Task2EvalFrameStackWrapper(gym.Env):
    """Evaluation wrapper matching Task2 training layout.

    Actor:
        obs_stack = 87 * 5 = 435

    Critic:
        obs_stack 435 + terrain_priv 91 = 526

    Returns dict:
        {"policy": obs_stack, "critic": critic_obs}
    """

    def __init__(self, env: Go2Task2Env, n_stack: int = 5):
        super().__init__()
        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device

        self.single_obs_dim = int(env.cfg.num_observations)
        self.single_priv_dim = int(env.cfg.num_privileged_obs)
        self.terrain_priv_dim = self.single_priv_dim - self.single_obs_dim

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

        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    @property
    def unwrapped(self):
        return self

    def _build_critic_obs(self) -> torch.Tensor:
        raw_priv = self.env.compute_privileged_obs()
        terrain_priv = raw_priv[:, self.single_obs_dim:]
        critic = torch.cat([self.obs_stack, terrain_priv], dim=-1)

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
        self.last_reward_mean = to_float(reward) or 0.0
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
    print("\n" + "=" * 160)
    print("Go2 Task2 Model Test Summary")
    print("=" * 160)
    print(f"{'metric':<70} | {'mean':>12} | {'std':>12} | {'min':>12} | {'max':>12}")
    print("-" * 160)
    for key in sorted(summary):
        row = summary[key]
        print(
            f"{key:<70} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )
    print("=" * 160 + "\n")


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
            init_log_std=-1.25,
            min_log_std=-5.0,
            max_log_std=0.3,
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
            "directory": str(PROJECT_ROOT / "logs" / "task2_eval_tmp"),
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
        p / "go2_task2_model.pt",
        p / "agent.pt",
        p / "checkpoint.pt",
        p / "best_agent.pt",
        p / "final_checkpoint" / "go2_task2_model.pt",
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


def main():
    set_seed(int(args_cli.seed))

    cfg = Task2Config()
    # GUI/GIF evaluation should show exactly one robot.
    # Headless evaluation can still use --num-envs for batch metrics.
    _requested_num_envs = int(args_cli.num_envs)
    _is_headless_eval = bool(getattr(args_cli, "headless_eval", False)) or bool(getattr(args_cli, "headless", False))
    cfg.num_envs = _requested_num_envs if _is_headless_eval else 1

    # Some config versions keep scene num_envs in a nested field.
    # Keep those fields synchronized when they exist.
    for _scene_attr in ("scene", "scene_cfg", "interactive_scene_cfg"):
        _scene_obj = getattr(cfg, _scene_attr, None)
        if _scene_obj is not None and hasattr(_scene_obj, "num_envs"):
            _scene_obj.num_envs = int(cfg.num_envs)
    cfg.device = str(args_cli.device)
    cfg.print_debug_info = False

    base_env = Go2Task2Env(cfg)
    force_eval_curriculum(base_env, args_cli.start_k, label="after_env_creation")

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * cfg.terrain_curriculum_total_steps)
        print(
            f"[INFO] Evaluation start_k={args_cli.start_k:.4f}, "
            f"global_steps={base_env.global_steps:,}"
        )

    stacked_env = Go2Task2EvalFrameStackWrapper(base_env, n_stack=5)
    env = wrap_env(stacked_env, wrapper="isaaclab")

    print("\n[DEBUG] Go2 Task2 Eval Spaces")
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

    force_eval_curriculum(base_env if "base_env" in locals() else env, args_cli.start_k, label="before_rollout_reset")
    states, _ = reset_env(env)
    force_eval_curriculum(base_env if "base_env" in locals() else env, args_cli.start_k, label="after_rollout_reset")

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0
    start = time.time()

    print("\n" + "=" * 140)
    print("Unitree Go2 Task2 skrl model test started")
    print("=" * 140)
    print(f"[INFO] model_test requested start_k = {args_cli.start_k}")
    print(f"checkpoint : {checkpoint}")
    print(f"num_envs   : {env.num_envs}")
    print(f"steps      : {args_cli.steps}")
    print(f"start_k    : {args_cli.start_k}")
    print(f"device     : {env.device}")
    print("=" * 140 + "\n")

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Go2 Task2 Model Test",
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
                    states, rewards, terminated, truncated, _ = step_env(env, actions)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                    flat = flat_dict(stacked_env.last_info)
                    row = {
                        "reward_mean": float(rewards.detach().float().mean().cpu().item()),
                        "terminated_rate": float(terminated.float().mean().cpu().item()),
                        "truncated_rate": float(truncated.float().mean().cpu().item()),
                    }
                    row.update(flat)
                    records.append(row)

                    pbar.set_postfix(
                        {
                            "rew": f"{row['reward_mean']:+.3f}",
                            "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
                            "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.0f}",
                            "level": f"{flat.get('telemetry/Mean_Terrain_Level', 0.0):.2f}",
                            "type": f"{flat.get('telemetry/Mean_Terrain_Type', 0.0):.2f}",
                            "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):+.2f}",
                            "cmd": f"{flat.get('telemetry/Cmd_Vx', 0.0):+.2f}",
                            "mu": f"{flat.get('telemetry/Mean_Friction', 0.0):.2f}",
                            "h": f"{flat.get('telemetry/Base_Height', 0.0):.3f}",
                            "ct": f"{flat.get('telemetry/Contact_Count', 0.0):.2f}",
                        }
                    )

                pbar.update(1)

        elapsed = time.time() - start
        env_steps = int(args_cli.steps) * int(env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n[OK] Go2 Task2 model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_table(summarize(records))

        print("Task2 model test checklist:")
        print("1. Fall_Rate should be low for a trained checkpoint.")
        print("2. Actual_Vx should roughly follow Cmd_Vx after sufficient training.")
        print("3. Mean_Terrain_Level indicates curriculum difficulty reached during evaluation.")
        print("4. P_Foot_Slip and Contact_Count help diagnose gait quality.")
        print("5. If checkpoint is from smoke training, poor locomotion is normal.")

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            None if bool(getattr(args_cli, 'no_close_on_exit', False)) else simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
