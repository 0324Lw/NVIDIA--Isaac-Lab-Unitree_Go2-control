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

parser = argparse.ArgumentParser(description="Evaluate Unitree Go2 Task3 skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.12)
parser.add_argument("--print-interval", type=int, default=100)
parser.add_argument("--deterministic", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

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
from go2_rl.tasks.task3.task3_config import Task3Config
from go2_rl.tasks.task3.task3_env import Go2Task3Env


class Go2Task3EvalFrameStackWrapper(gym.Env):
    """Evaluation wrapper matching Task3 training layout.

    Actor:
        obs_stack = 257 * 5 = 1285

    Critic:
        obs_stack 1285 + world_privileged 68 = 1353

    Returns dict:
        {"policy": obs_stack, "critic": critic_obs}
    """

    def __init__(self, env: Go2Task3Env, n_stack: int = 5):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device

        self.single_obs_dim = int(env.cfg.num_observations)
        self.single_priv_dim = int(env.cfg.num_privileged_obs)
        self.world_priv_dim = self.single_priv_dim - self.single_obs_dim

        if self.single_obs_dim != 257:
            raise RuntimeError(f"Task3 actor single obs dim should be 257, got {self.single_obs_dim}")
        if self.world_priv_dim != 68:
            raise RuntimeError(f"Task3 world privileged dim should be 68, got {self.world_priv_dim}")

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


def main():
    set_seed(int(args_cli.seed))

    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.print_debug_info = False

    base_env = Go2Task3Env(cfg)

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * cfg.curriculum_total_steps)
        print(
            f"[INFO] Evaluation start_k={args_cli.start_k:.4f}, "
            f"global_steps={base_env.global_steps:,}"
        )

    stacked_env = Go2Task3EvalFrameStackWrapper(base_env, n_stack=5)
    env = wrap_env(stacked_env, wrapper="isaaclab")

    print("\n[DEBUG] Go2 Task3 Eval Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    if int(env.observation_space.shape[0]) != 1285:
        raise RuntimeError(f"Task3 policy input dim should be 1285, got {env.observation_space.shape[0]}")
    if int(env.state_space.shape[0]) != 1353:
        raise RuntimeError(f"Task3 critic input dim should be 1353, got {env.state_space.shape[0]}")

    agent = build_agent(env)
    agent.init(trainer_cfg={"timesteps": 1, "headless": True})

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

    states, _ = reset_env(env)

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
    print(f"checkpoint : {checkpoint}")
    print(f"num_envs   : {env.num_envs}")
    print(f"steps      : {args_cli.steps}")
    print(f"start_k    : {args_cli.start_k}")
    print(f"device     : {env.device}")
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
                    actions = agent.act(states, timestep=step, timesteps=int(args_cli.steps))[0]
                    states, rewards, terminated, truncated, _ = step_env(env, actions)

                flat = flat_dict(stacked_env.last_info)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                total_success += int(round(flat.get("events/Success_Rate", 0.0) * int(env.num_envs)))
                total_collision += int(round(flat.get("events/Collision_Rate", 0.0) * int(env.num_envs)))
                total_fall += int(round(flat.get("events/Fall_Rate", 0.0) * int(env.num_envs)))
                total_timeout += int(round(flat.get("events/Timeout_Rate", 0.0) * int(env.num_envs)))

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
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
                            "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.1f}",
                            "dist": f"{flat.get('telemetry/Distance_To_Goal', 0.0):.2f}",
                            "prog": f"{flat.get('telemetry/Progress', 0.0):+.3f}",
                            "succ": f"{flat.get('events/Success_Rate', 0.0):.3f}",
                            "coll": f"{flat.get('events/Collision_Rate', 0.0):.3f}",
                            "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
                            "risk": f"{flat.get('telemetry/Collision_Risk', 0.0):.2f}",
                            "h": f"{flat.get('telemetry/Base_Height', 0.0):.3f}",
                        }
                    )

                pbar.update(1)

        elapsed = time.time() - start
        env_steps = int(args_cli.steps) * int(env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ Go2 Task3 model test rollout finished")
        print(f"  env steps          : {env_steps:,}")
        print(f"  fps                : {fps:,.2f}")
        print(f"  total terminated   : {total_terminated:,}")
        print(f"  total truncated    : {total_truncated:,}")
        print(f"  approx success     : {total_success:,}")
        print(f"  approx collision   : {total_collision:,}")
        print(f"  approx fall        : {total_fall:,}")
        print(f"  approx timeout     : {total_timeout:,}")

        print_table(summarize(records))

        print("Task3 model test checklist:")
        print("1. Smoke checkpoint 表现差是正常的，重点检查是否能稳定推理、无 NaN/Inf。")
        print("2. 正式训练 checkpoint 应逐步看到 Distance_To_Goal 下降、Progress 为正、Success_Rate 上升。")
        print("3. Stage0 重点看无障碍目标到达；Stage1+ 再看 Collision_Risk / Collision_Rate。")
        print("4. Fall_Rate 高说明底层 gait 或 action scale 有问题。")
        print("5. Collision_Rate 高但 Progress 正常，通常是避障奖励或 lidar/risk 权重需要调。")
        print("6. 如果 Distance_To_Goal 长期不下降，应优先检查 target_obs、progress reward 和 root-local 坐标对齐。")

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
