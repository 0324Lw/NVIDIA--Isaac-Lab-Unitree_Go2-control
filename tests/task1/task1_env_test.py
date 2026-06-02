# Copyright (c) 2026
# Unitree Go2 Task1: 平地运动环境集成测试。
#
# 本文件用于检查 Task1 IsaacLab 环境的初始化、观测维度、课程采样、接触传感器、
# 奖励项和随机 rollout 数值稳定性。
#
# 测试入口:
#   bash scripts/ubuntu/test_task1_env.sh
#
# Gymnasium API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# 观测维度:
#   actor obs = 87
#   privileged obs = 0
#   action dim = 12
#
# 工程说明:
#   IsaacLab / pxr 依赖模块在 AppLauncher 启动后导入。
#   env_origins 用于检查 root pose 与并行环境局部原点的对齐关系。
#
# Unitree Go2 Task1: flat locomotion environment integration test.
#
# This file checks Task1 IsaacLab environment initialization, observation dimensions,
# curriculum sampling, contact sensors, reward terms, and random-rollout numerical stability.
#
# Test entry:
#   bash scripts/ubuntu/test_task1_env.sh
#
# Gymnasium API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# Observation dimensions:
#   actor obs = 87
#   privileged obs = 0
#   action dim = 12
#
# Engineering notes:
#   IsaacLab / pxr dependent modules are imported after AppLauncher starts.
#   env_origins is used to check root-pose alignment with each parallel environment origin.

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Unitree Go2 Task1 Env / Reward / Curriculum Test")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rollout-k", type=float, default=0.0)
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--collect-interval", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from go2_rl.tasks.task1.task1_config import Task1Config
from go2_rl.tasks.task1.task1_env import Go2Task1Env


def print_ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} must be torch.Tensor, got {type(x)}"
    assert torch.isfinite(x).all(), f"{name} has NaN or Inf"


def tensor_to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def flatten_info(info: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (info or {}).items():
        if key == "terminal_observation":
            continue
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_info(value, name))
        else:
            val = tensor_to_float(value)
            if val is not None and math.isfinite(val):
                out[name] = val
    return out


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    keys = sorted({k for row in records for k in row.keys()})
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    if not summary:
        print_warn("No valid records collected")
        return

    print("\n" + "=" * 150)
    print(" " * 45 + "Unitree Go2 Task1 Environment Statistics")
    print("=" * 150)
    print(f"{'metric':<60} | {'mean':>14} | {'std':>14} | {'min':>14} | {'max':>14}")
    print("-" * 150)
    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<60} | "
            f"{row['mean']:>14.6f} | "
            f"{row['std']:>14.6f} | "
            f"{row['min']:>14.6f} | "
            f"{row['max']:>14.6f}"
        )
    print("=" * 150 + "\n")


def quat_from_roll_pitch_yaw(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    device: str = "cuda:0",
) -> torch.Tensor:
    # Isaac/USD quaternion order: wxyz
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def force_root_pose(
    env: Go2Task1Env,
    env_ids: torch.Tensor,
    height: float | None = None,
    quat: torch.Tensor | None = None,
    zero_vel: bool = True,
) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:2] = env.scene.env_origins[env_ids, 0:2]

    if height is not None:
        root_state[:, 2] = env.scene.env_origins[env_ids, 2] + float(height)

    if quat is not None:
        root_state[:, 3:7] = quat.repeat(len(env_ids), 1)

    if zero_vel:
        root_state[:, 7:13] = 0.0

    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env.scene.update(dt=0.0)


def check_obs_shape_and_values(env: Go2Task1Env, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_observations)
    assert torch.is_tensor(obs), f"obs must be torch.Tensor, got {type(obs)}"
    assert tuple(obs.shape) == expected, f"obs shape wrong: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= 10.0001, f"obs out of clamp range: {obs.abs().max().item():.6f}"


def check_observation_slices(env: Go2Task1Env, obs: torch.Tensor) -> None:
    cursor = 0
    slices = {}

    for name, dim in [
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("smoothed_cmd", 3),
        ("q_err", 12),
        ("qd", 12),
        ("last_action", 12),
        ("action_delta", 12),
        ("contact", 4),
        ("foot_rel_pos", 12),
        ("foot_vel_xy", 8),
        ("base_height", 1),
        ("sin_phase", 1),
        ("cos_phase", 1),
    ]:
        slices[name] = obs[:, cursor:cursor + dim]
        cursor += dim

    assert cursor == env.cfg.num_observations, f"obs slice cursor={cursor}, expected={env.cfg.num_observations}"

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    contact = slices["contact"]
    assert torch.all(contact >= -1e-5) and torch.all(contact <= 1.0 + 1e-5), "contact must be in [0, 1]"
    assert torch.all(slices["sin_phase"].abs() <= 1.0001), "sin_phase out of range"
    assert torch.all(slices["cos_phase"].abs() <= 1.0001), "cos_phase out of range"

    h_mean = slices["base_height"].mean().item()
    assert 0.15 <= h_mean <= 0.55, f"base_height mean abnormal: {h_mean:.4f}"


def check_curriculum(env: Go2Task1Env) -> None:
    cfg = env.cfg
    checks = [
        (0.00, 0),
        (0.05, 0),
        (0.06, 1),
        (0.18, 2),
        (0.36, 3),
        (0.58, 4),
        (0.82, 5),
        (1.00, 5),
    ]

    old_steps = env.global_steps
    for k, expected_stage in checks:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        got = env._command_stage()
        assert got == expected_stage, f"k={k:.2f} stage wrong: got {got}, expected {expected_stage}"

    env.global_steps = old_steps
    print_ok("curriculum stage boundary check passed")


def check_command_sampling(env: Go2Task1Env) -> None:
    old_steps = env.global_steps

    for k in [0.0, 0.10, 0.25, 0.45, 0.70, 0.95]:
        env.global_steps = int(k * env.cfg.curriculum_total_steps)
        cmd = env._sample_commands(4096)
        assert_finite_tensor(f"sampled command k={k}", cmd)
        assert cmd.shape == (4096, 3)

        vx_range, vy_range, wz_range = env._command_ranges()
        nonzero = torch.norm(cmd, dim=-1) > 1e-6

        if nonzero.any():
            cmd_nz = cmd[nonzero]
            assert cmd_nz[:, 0].min().item() >= vx_range[0] - 1e-5
            assert cmd_nz[:, 0].max().item() <= vx_range[1] + 1e-5
            assert cmd_nz[:, 1].min().item() >= vy_range[0] - 1e-5
            assert cmd_nz[:, 1].max().item() <= vy_range[1] + 1e-5
            assert cmd_nz[:, 2].min().item() >= wz_range[0] - 1e-5
            assert cmd_nz[:, 2].max().item() <= wz_range[1] + 1e-5

    env.global_steps = old_steps
    print_ok("command sampling range check passed")


def check_forced_events(env: Go2Task1Env) -> None:
    cfg = env.cfg
    zero_action = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    env.reset(seed=args_cli.seed)

    low_ids = torch.arange(min(16, cfg.num_envs), device=env.device)
    force_root_pose(env, low_ids, height=cfg.fall_height * 0.5)
    _, _, terminated, _, _ = env.step(zero_action)
    low_hit = int(terminated[low_ids].sum().item())
    assert low_hit > 0, "forced low height did not trigger terminated"
    print_ok(f"low-height fall event passed: {low_hit}/{len(low_ids)}")

    env.reset(seed=args_cli.seed)

    tilt_ids = torch.arange(min(16, cfg.num_envs), device=env.device)
    bad_quat = quat_from_roll_pitch_yaw(roll=1.35, device=env.device)
    force_root_pose(env, tilt_ids, height=cfg.target_height, quat=bad_quat)
    _, _, terminated, _, _ = env.step(zero_action)
    tilt_hit = int(terminated[tilt_ids].sum().item())
    assert tilt_hit > 0, "forced tilt did not trigger terminated"
    print_ok(f"tilt fall event passed: {tilt_hit}/{len(tilt_ids)}")

    env.reset(seed=args_cli.seed)

    env.episode_steps[:] = cfg.max_episode_length - 1
    _, _, _, truncated, _ = env.step(zero_action)
    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "timeout did not trigger truncated"
    print_ok(f"timeout event passed: truncated={timeout_count}")


def run_tests() -> None:
    print("\n" + "=" * 128)
    print(" Unitree Go2 Task1 Env / Contact / Curriculum / Reward / Random Rollout Test")
    print("=" * 128)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.debug_print_names = bool(args_cli.print_names)

    env: Go2Task1Env | None = None

    try:
        print("\n[TEST 1] Environment initialization / robot mapping.")
        env = Go2Task1Env(cfg)

        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"device = {env.device}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"action_dim = {env.action_space.shape[0]}")
        print_ok(f"obs_dim = {env.observation_space.shape[0]}")

        assert env.action_space.shape[0] == cfg.num_actions
        assert env.observation_space.shape[0] == cfg.num_observations
        assert len(env.action_joint_ids) == cfg.num_actions
        assert len(env.foot_body_ids) == 4
        assert len(env.contact_foot_ids) == 4

        print("\n[TEST 2] reset / obs shape / obs finite.")
        obs, _ = env.reset(seed=args_cli.seed)
        check_obs_shape_and_values(env, obs)
        check_observation_slices(env, obs)
        print_ok("reset obs check passed")

        print("\n[TEST 3] contact sensor shape.")
        contact, normal_force = env._get_foot_contact()
        assert contact.shape == (cfg.num_envs, 4), f"contact shape wrong: {tuple(contact.shape)}"
        assert normal_force.shape == (cfg.num_envs, 4), f"normal_force shape wrong: {tuple(normal_force.shape)}"
        assert_finite_tensor("contact", contact)
        assert_finite_tensor("normal_force", normal_force)
        print_ok("contact sensor check passed")

        print("\n[TEST 4] curriculum stage / command sampling.")
        check_curriculum(env)
        check_command_sampling(env)

        print("\n[TEST 5] zero-action warmup / reward / done shape.")
        zero_action = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

        latest_info = {}
        for _ in range(50):
            obs, reward, terminated, truncated, latest_info = env.step(zero_action)
            check_obs_shape_and_values(env, obs)
            assert reward.shape == (cfg.num_envs,)
            assert terminated.shape == (cfg.num_envs,)
            assert truncated.shape == (cfg.num_envs,)
            assert_finite_tensor("reward", reward)

        flat = flatten_info(latest_info)
        for key in [
            "reward_components/Total",
            "events/Fall_Rate",
            "telemetry/Base_Height",
            "telemetry/Contact_Count",
            "debug/Obs_Dim",
        ]:
            assert key in flat, f"info missing field: {key}"

        print_ok("zero-action step / reward / info check passed")

        print("\n[TEST 6] observation slices after warmup.")
        check_observation_slices(env, obs)
        print_ok("observation slice check passed")

        print("\n[TEST 7] forced fall / tilt / timeout events.")
        check_forced_events(env)

        print("\n[TEST 8] random policy rollout.")
        env.reset(seed=args_cli.seed)
        env.global_steps = int(float(args_cli.rollout_k) * cfg.curriculum_total_steps)

        records: List[Dict[str, float]] = []
        total_terminated = 0
        total_truncated = 0
        start = time.time()

        for step in range(int(args_cli.steps)):
            action = torch.empty(
                (cfg.num_envs, cfg.num_actions),
                dtype=torch.float32,
                device=env.device,
            ).uniform_(-1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            total_terminated += int(terminated.sum().item())
            total_truncated += int(truncated.sum().item())

            if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                check_obs_shape_and_values(env, obs)
                assert_finite_tensor("reward during rollout", reward)
                assert torch.isfinite(env.target_cmd).all(), "target_cmd has NaN/Inf"
                assert torch.isfinite(env.smoothed_cmd).all(), "smoothed_cmd has NaN/Inf"
                assert torch.isfinite(env.last_action).all(), "last_action has NaN/Inf"

                row = flatten_info(info)
                row["test/step"] = float(step)
                row["test/reward_mean"] = float(reward.detach().mean().cpu().item())
                records.append(row)

                msg = (
                    f"step={step + 1:>5}/{args_cli.steps} | "
                    f"reward={row.get('test/reward_mean', 0.0):>8.4f} | "
                    f"fall={row.get('events/Fall_Rate', 0.0):>7.4f} | "
                    f"h={row.get('telemetry/Base_Height', 0.0):>6.3f} | "
                    f"contacts={row.get('telemetry/Contact_Count', 0.0):>5.2f}"
                )
                print(msg, flush=True)

        elapsed = time.time() - start
        env_steps_per_sec = int(args_cli.steps) * int(cfg.num_envs) / max(elapsed, 1e-6)

        print_ok(f"random rollout completed: {args_cli.steps} control steps")
        print_ok(f"total transitions: {int(args_cli.steps) * int(cfg.num_envs):,}")
        print_ok(f"throughput: {env_steps_per_sec:,.2f} env steps/s")
        print_ok(f"terminated count: {total_terminated:,}")
        print_ok(f"truncated count: {total_truncated:,}")

        print("\n[TEST 9] statistics report.")
        print_summary_table(summarize_records(records))

        print("Go2 Task1 pre-training checklist:")
        print("1. action_dim 期望为 12，obs_dim 期望为 87。")
        print("2. foot contact shape must be [num_envs, 4].")
        print("3. Stage 0 gait reward may be small or zero; this is normal.")
        print("4. Random policy Fall_Rate can be high, but NaN/Inf is not allowed.")
        print("5. Training metrics: Fall_Rate, Episode_Length, Actual_Vx/Cmd_Vx, Contact_Count, P_Foot_Slip.")
        print("\n[OK] Unitree Go2 Task1 environment test completed.")

    except Exception as exc:
        print("\n[FAIL] Unitree Go2 Task1 environment test failed:")
        print(type(exc).__name__, ":", exc)
        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
