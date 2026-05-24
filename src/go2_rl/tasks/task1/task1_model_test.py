from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

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

parser = argparse.ArgumentParser(description="Evaluate Unitree Go2 Task1 skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--print-interval", type=int, default=100)
parser.add_argument("--visualize", action="store_true", help="Open Isaac Sim GUI for lightweight visualization")
parser.add_argument("--no-close-on-exit", action="store_true", help="Debug only: do not explicitly call None if bool(getattr(args_cli, 'no_close_on_exit', False)) else simulation_app.close()")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(getattr(args_cli, 'visualize', False))
# Default to stable headless evaluation. Use --visualize only for GUI.
args_cli.headless = not bool(getattr(args_cli, 'visualize', False))
# Model evaluation should open Isaac Sim GUI by default.
# Do not force headless here. Pass --headless manually if needed.

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
from go2_rl.common.go2_skrl_wrappers import Go2FrameStackWrapper
from go2_rl.common.info_utils import flat_dict, load_normalizers
from go2_rl.common.eval_curriculum_utils import force_eval_curriculum
from go2_rl.common.model_eval_utils import direct_policy_action, init_agent_compat
from go2_rl.tasks.task1.task1_config import Task1Config
from go2_rl.tasks.task1.task1_env import Go2Task1Env


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
    print("\n" + "=" * 140)
    print("Go2 Task1 Model Test Summary")
    print("=" * 140)
    print(f"{'metric':<58} | {'mean':>12} | {'std':>12} | {'min':>12} | {'max':>12}")
    print("-" * 140)
    for key in sorted(summary):
        row = summary[key]
        print(
            f"{key:<58} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )
    print("=" * 140 + "\n")


def _base_ppo_cfg_dict():
    cfg = PPO_CFG()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return cfg.copy()


def build_agent(env):
    models = {
        "policy": Go2Actor(env.observation_space, env.state_space, env.action_space, env.device),
        "value": Go2Critic(env.observation_space, env.state_space, env.action_space, env.device),
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
            "directory": str(PROJECT_ROOT / "logs" / "task1_eval_tmp"),
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

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.debug_print_names = False

    base_env = Go2Task1Env(cfg)
    force_eval_curriculum(base_env, args_cli.start_k, label="after_env_creation")

    if args_cli.start_k > 0:
        base_env.global_steps = int(float(args_cli.start_k) * cfg.curriculum_total_steps)

    stacked_env = Go2FrameStackWrapper(
        base_env,
        log_dir=str(PROJECT_ROOT / "logs" / "task1_eval_tmp"),
        n_stack=5,
        tb_log_interval_steps=0,
        use_privileged_obs=False,
    )
    env = wrap_env(stacked_env, wrapper="isaaclab")

    agent = build_agent(env)
    # skrl compatibility:
    # Some skrl versions expect trainer_cfg to be a dataclass, not a plain dict.
    # For evaluation, trainer_cfg is not required, so fallback to agent.init().
    try:
        init_agent_compat(agent)
    except TypeError as exc:
        if "asdict" not in str(exc) and "dataclass" not in str(exc):
            raise
        print("[WARN] agent.init(trainer_cfg=dict) is not supported by this skrl build; fallback to agent.init().")
        agent.init()

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
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

    records = []
    total_terminated = 0
    total_truncated = 0
    start = time.time()

    print("\n" + "=" * 120)
    print("Unitree Go2 Task1 skrl model test started")
    print("=" * 120)
    print(f"[INFO] model_test requested start_k = {args_cli.start_k}")
    print(f"checkpoint : {checkpoint}")
    print(f"num_envs   : {env.num_envs}")
    print(f"steps      : {args_cli.steps}")
    print(f"device     : {env.device}")
    print("=" * 120 + "\n")

    try:
        with tqdm(total=int(args_cli.steps), desc="Go2 Task1 Model Test", dynamic_ncols=True, mininterval=0.5) as pbar:
            for step in range(int(args_cli.steps)):
                with torch.no_grad():
                    print(f"[DEBUG][eval step {step}] before direct_policy_action", flush=True)
                    actions = direct_policy_action(
                        agent,
                        states,
                        debug=(step < 3),
                        step=int(step),
                    )
                    print(f"[DEBUG][eval step {step}] after direct_policy_action", flush=True)
                    print(f"[DEBUG][eval step {step}] before env.step", flush=True) if step < 3 else None
                    states, rewards, terminated, truncated, _ = step_env(env, actions)
                    print(f"[DEBUG][eval step {step}] after env.step", flush=True) if step < 3 else None

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
                            "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):+.2f}",
                            "cmd": f"{flat.get('telemetry/Cmd_Vx', 0.0):+.2f}",
                            "h": f"{flat.get('telemetry/Base_Height', 0.0):.3f}",
                            "ct": f"{flat.get('telemetry/Contact_Count', 0.0):.2f}",
                        }
                    )

                pbar.update(1)

        elapsed = time.time() - start
        env_steps = int(args_cli.steps) * int(env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ Go2 Task1 model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_table(summarize(records))

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
