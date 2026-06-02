# Copyright (c) 2026
# Unitree Go2 Task3: 导航避障环境集成测试。
#
# 本文件用于检查 Task3 IsaacLab 环境的初始化、观测切片、world privileged features、
# lidar / risk / target 接口、reset 对齐、强制事件和随机 rollout 数值稳定性。
#
# 测试入口:
#   bash scripts/ubuntu/test_task3_env.sh
#
# Gymnasium API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# 观测维度:
#   actor obs = 208
#   privileged obs = 276
#   world privileged tail = 68
#   lidar rays = 60
#   action dim = 12
#
# 工程说明:
#   IsaacLab / pxr 依赖模块在 AppLauncher 启动后导入。
#   测试重点保护 Task3 的 208 / 276 / 68 / 60 维度和解析 world tensor 接口。
#
# Unitree Go2 Task3: navigation and obstacle-avoidance environment integration test.
#
# This file checks Task3 IsaacLab environment initialization, observation slices,
# world privileged features, lidar / risk / target interfaces, reset alignment,
# forced events, and random-rollout numerical stability.
#
# Test entry:
#   bash scripts/ubuntu/test_task3_env.sh
#
# Gymnasium API:
#   reset() -> obs, info
#   step(action) -> obs, reward, terminated, truncated, info
#
# Observation dimensions:
#   actor obs = 208
#   privileged obs = 276
#   world privileged tail = 68
#   lidar rays = 60
#   action dim = 12
#
# Engineering notes:
#   IsaacLab / pxr dependent modules are imported after AppLauncher starts.
#   The test protects Task3 dimensions 208 / 276 / 68 / 60 and analytical world
#   tensor interfaces.

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


parser = argparse.ArgumentParser(description="Unitree Go2 Task3 Navigation / Obstacle Avoidance Env Test")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--rollout-k", type=float, default=0.12)
parser.add_argument("--collect-interval", type=int, default=40)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from go2_rl.tasks.task3.task3_config import Task3Config
from go2_rl.tasks.task3.task3_env import Go2Task3Env


def print_ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def heading(title: str) -> None:
    print("\n" + "=" * 128)
    print(title)
    print("=" * 128)


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} must be torch.Tensor, got {type(x)}"
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


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
        print_warn("No valid records collected.")
        return

    print("\n" + "=" * 168)
    print(" " * 48 + "Unitree Go2 Task3 Environment Statistics")
    print("=" * 168)
    print(f"{'metric':<76} | {'mean':>14} | {'std':>14} | {'min':>14} | {'max':>14}")
    print("-" * 168)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<76} | "
            f"{row['mean']:>14.6f} | "
            f"{row['std']:>14.6f} | "
            f"{row['min']:>14.6f} | "
            f"{row['max']:>14.6f}"
        )

    print("=" * 168 + "\n")


def quat_from_roll_pitch_yaw(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    device: str = "cuda:0",
) -> torch.Tensor:
    # Isaac / USD quaternion order: wxyz
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def check_project_files() -> None:
    heading("[测试 0] Task3 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task3_navigation.yaml",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task3" / "task3_config.py",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task3" / "task3_world.py",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task3" / "task3_env.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "Missing required Task3 files:\n" + "\n".join(missing)

    for p in required:
        print_ok(str(p.relative_to(PROJECT_ROOT)))

    print_ok("Task3 工程文件结构正常")


def check_obs_shape_and_values(env: Go2Task3Env, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_observations)
    assert torch.is_tensor(obs), f"obs must be torch.Tensor, got {type(obs)}"
    assert tuple(obs.shape) == expected, f"obs shape wrong: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= 10.0001, f"obs out of clamp range: {obs.abs().max().item():.6f}"


def check_priv_shape_and_values(env: Go2Task3Env, priv: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_privileged_obs)
    assert torch.is_tensor(priv), f"privileged obs must be torch.Tensor, got {type(priv)}"
    assert tuple(priv.shape) == expected, f"privileged obs shape wrong: {tuple(priv.shape)} != {expected}"
    assert_finite_tensor("privileged_obs", priv)
    assert priv.abs().max().item() <= 20.0001, f"privileged obs out of clamp range: {priv.abs().max().item():.6f}"


def check_observation_slices(env: Go2Task3Env, obs: torch.Tensor) -> None:
    cursor = 0
    slices: Dict[str, torch.Tensor] = {}

    layout = [
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("q_err", 12),
        ("qd", 12),
        ("last_action", 12),
        ("action_delta", 12),
        ("foot_contact", 4),
        ("base_height", 1),
        ("goal_dir_body", 2),
        ("goal_dist_norm", 1),
        ("goal_log_dist", 1),
        ("heading_sin_cos", 2),
        ("heading_cos", 1),
        ("actual_along_goal", 1),
        ("lateral_vel_to_goal", 1),
        ("target_speed", 1),
        ("desired_speed", 1),
        ("speed_ratio", 1),
        ("progress_step", 1),
        ("progress_ema_norm", 1),
        ("distance_reduction_ratio", 1),
        ("near_goal_flag", 1),
        ("success_radius_norm", 1),
        ("time_fraction", 1),
        ("stuck_timer_norm", 1),
        ("obstacle_summary", 7),
        ("lidar", 60),
        ("lidar_delta", 60),
    ]

    for name, dim in layout:
        slices[name] = obs[:, cursor:cursor + dim]
        cursor += dim

    assert cursor == env.cfg.num_observations, f"obs slice cursor={cursor}, expected={env.cfg.num_observations}"

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    assert torch.all(slices["foot_contact"] >= -1e-5)
    assert torch.all(slices["foot_contact"] <= 1.0 + 1e-5)

    assert torch.all(slices["goal_dir_body"].abs() <= 1.0001)
    assert torch.all(slices["goal_dist_norm"] >= -1e-5)
    assert torch.all(slices["goal_dist_norm"] <= 2.0 + 1e-5)
    assert torch.all(slices["goal_log_dist"] >= -1e-5)

    assert torch.all(slices["heading_sin_cos"].abs() <= 1.0001)
    assert torch.all(slices["heading_cos"].abs() <= 1.0001)

    assert torch.all(slices["target_speed"] >= -1e-5)
    assert torch.all(slices["target_speed"] <= 2.0 + 1e-5)
    assert torch.all(slices["desired_speed"] >= -1e-5)
    assert torch.all(slices["desired_speed"] <= 2.0 + 1e-5)

    assert torch.all(slices["progress_ema_norm"].abs() <= 2.0001)
    assert torch.all(slices["distance_reduction_ratio"].abs() <= 1.0001)
    assert torch.all(slices["near_goal_flag"] >= -1e-5)
    assert torch.all(slices["near_goal_flag"] <= 1.0 + 1e-5)
    assert torch.all(slices["success_radius_norm"] >= -1e-5)
    assert torch.all(slices["success_radius_norm"] <= 1.0 + 1e-5)
    assert torch.all(slices["time_fraction"] >= -1e-5)
    assert torch.all(slices["time_fraction"] <= 1.0 + 1e-5)
    assert torch.all(slices["stuck_timer_norm"] >= -1e-5)
    assert torch.all(slices["stuck_timer_norm"] <= 1.0 + 1e-5)

    assert torch.all(slices["obstacle_summary"] >= -1e-5)
    assert torch.all(slices["obstacle_summary"] <= 1.0 + 1e-5)

    assert torch.all(slices["lidar"] >= -1e-5)
    assert torch.all(slices["lidar"] <= 1.0 + 1e-5)

    assert torch.all(slices["lidar_delta"] >= -1.0001)
    assert torch.all(slices["lidar_delta"] <= 1.0001)

    h_mean = slices["base_height"].mean().item()
    assert 0.10 <= h_mean <= 0.75, f"base_height mean abnormal: {h_mean:.4f}"


def check_privileged_slices(env: Go2Task3Env, priv: torch.Tensor) -> None:
    actor_dim = int(env.cfg.num_observations)
    world_priv_dim = env.world.privileged_feature_dim()

    actor_part = priv[:, :actor_dim]
    world_part = priv[:, actor_dim:]

    assert actor_part.shape == (env.cfg.num_envs, actor_dim)
    assert world_part.shape == (env.cfg.num_envs, world_priv_dim)
    assert world_priv_dim == 68

    cursor = 0

    static_dim = env.cfg.world_cfg.privileged_static_k * 4
    dynamic_dim = env.cfg.world_cfg.privileged_dynamic_k * 6

    static_feat = world_part[:, cursor:cursor + static_dim]
    cursor += static_dim

    dynamic_feat = world_part[:, cursor:cursor + dynamic_dim]
    cursor += dynamic_dim

    risk = world_part[:, cursor:cursor + 8]
    cursor += 8

    target_obs = world_part[:, cursor:cursor + 3]
    cursor += 3

    stage_onehot = world_part[:, cursor:cursor + 6]
    cursor += 6

    counts = world_part[:, cursor:cursor + 3]
    cursor += 3

    assert cursor == 68

    for name, value in [
        ("static_feat", static_feat),
        ("dynamic_feat", dynamic_feat),
        ("risk", risk),
        ("target_obs", target_obs),
        ("stage_onehot", stage_onehot),
        ("counts", counts),
    ]:
        assert_finite_tensor(name, value)

    assert torch.all(risk[:, :5] >= -1e-5)
    assert torch.all(risk[:, :5] <= 1.0 + 1e-5)
    assert torch.all(risk[:, 5:7].abs() <= 1.0001)
    assert torch.all(risk[:, 7] >= -1e-5)
    assert torch.all(risk[:, 7] <= 1.0 + 1e-5)

    assert torch.all(target_obs[:, 0] >= -1e-5)
    assert torch.all(target_obs[:, 0] <= 2.0 + 1e-5)

    assert torch.allclose(
        stage_onehot.sum(dim=-1),
        torch.ones(env.cfg.num_envs, dtype=torch.float32, device=env.device),
        atol=1e-5,
    )

    assert torch.all(counts >= -1e-5)
    assert torch.all(counts <= 1.0 + 1e-5)


def force_root_pose(
    env: Go2Task3Env,
    env_ids: torch.Tensor,
    local_xy: torch.Tensor | None = None,
    height: float | None = None,
    quat: torch.Tensor | None = None,
) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    if env_ids.numel() == 0:
        return

    root_state = env.robot.data.default_root_state[env_ids].clone()

    if local_xy is None:
        root_local = env._root_pos_local()[env_ids]
        root_state[:, 0:2] = env.env_origins[env_ids, :2] + root_local[:, :2]
    else:
        local_xy = torch.as_tensor(local_xy, dtype=torch.float32, device=env.device)
        if local_xy.ndim == 1:
            local_xy = local_xy.unsqueeze(0).repeat(env_ids.numel(), 1)
        root_state[:, 0:2] = env.env_origins[env_ids, :2] + local_xy[:, :2]

    if height is None:
        root_state[:, 2] = env.cfg.target_height
    else:
        root_state[:, 2] = float(height)

    if quat is None:
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
    else:
        quat = torch.as_tensor(quat, dtype=torch.float32, device=env.device)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0).repeat(env_ids.numel(), 1)
        root_state[:, 3:7] = quat

    root_state[:, 7:13] = 0.0
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env.scene.update(dt=0.0)


def check_command_stage_reset(env: Go2Task3Env, cfg: Task3Config) -> None:
    heading("[测试 4] world curriculum stage / reset 采样检测")

    old_steps = int(env.global_steps)

    checks = [
        (0.00, 0),
        (0.179999, 0),
        (0.18, 1),
        (0.379999, 1),
        (0.38, 2),
        (0.619999, 2),
        (0.62, 3),
        (0.839999, 3),
        (0.84, 4),
        (0.959999, 4),
        (0.96, 5),
        (1.00, 5),
    ]

    for k, expected_stage in checks:
        stage = env.world.stage_from_progress(k)
        assert stage == expected_stage, f"k={k} stage wrong: got {stage}, expected {expected_stage}"

    rows = []

    for k in [0.0, 0.20, 0.45, 0.65, 0.85, 0.98]:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        obs, _ = env.reset(seed=args_cli.seed)

        check_obs_shape_and_values(env, obs)

        stage_min = int(env.world.env_stage.min().item())
        stage_max = int(env.world.env_stage.max().item())
        expected = env.world.stage_from_progress(k)

        # performance-gated curriculum 下：
        # 1. global_steps 只决定当前允许的最高课程阶段 cap；
        # 2. 实际 reset stage 由 curriculum_active_stage 及其相邻阶段混合采样决定；
        # 3. 手动设置 global_steps 不会直接把 curriculum_active_stage 推进到 expected。
        # 因此这里只检查采样结果没有超过全局允许上限。
        if bool(getattr(cfg.world_cfg, "use_performance_gated_curriculum", False)):
            cap = int(env._global_stage_cap())
            assert 0 <= stage_min <= stage_max <= cap, (
                f"stage range out of global cap: "
                f"stage_min={stage_min}, stage_max={stage_max}, cap={cap}, expected={expected}"
            )
        else:
            assert stage_min == expected and stage_max == expected

        static_count = env.world.static_mask.float().sum(dim=-1)
        dynamic_count = env.world.dynamic_mask.float().sum(dim=-1)

        if stage_max == 0:
            assert static_count.max().item() == 0
            assert dynamic_count.max().item() == 0

        rows.append(
            {
                "k": k,
                "stage": expected,
                "static_mean": float(static_count.mean().item()),
                "dynamic_mean": float(dynamic_count.mean().item()),
                "target_speed": float(env.world.env_target_speed.mean().item()),
            }
        )

    env.global_steps = old_steps

    for row in rows:
        print_ok(
            f"k={row['k']:.2f} | stage={row['stage']} | "
            f"static={row['static_mean']:.2f} | dynamic={row['dynamic_mean']:.2f} | "
            f"target_speed={row['target_speed']:.2f}"
        )

    print_ok("world curriculum stage / reset 采样正常")


def check_reset_alignment(env: Go2Task3Env, cfg: Task3Config) -> None:
    heading("[测试 5] reset root pose 与 Task3World.start_pos 对齐检测")

    env.global_steps = int(0.20 * cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

    check_obs_shape_and_values(env, obs)

    root_local = env._root_pos_local()
    xy_err = torch.norm(root_local[:, :2] - env.world.start_pos, dim=-1)

    assert xy_err.max().item() < 1e-4, f"root local xy not aligned with world.start_pos: max_err={xy_err.max().item()}"

    base_height = env._compute_base_height()
    assert_finite_tensor("base_height after reset", base_height)
    assert 0.10 <= base_height.mean().item() <= 0.75

    target_dist = env.world.distance_to_target(root_local)
    assert target_dist.min().item() > 0.5

    print_ok(f"max reset xy error = {xy_err.max().item():.8f}")
    print_ok(f"base_height mean = {base_height.mean().item():.6f}")
    print_ok(f"distance_to_goal mean = {target_dist.mean().item():.6f}")
    print_ok("reset root pose 对齐正常")


def check_forced_events(env: Go2Task3Env, cfg: Task3Config) -> None:
    heading("[测试 9] 强制 success / collision / fall / out_of_bounds / timeout 事件检测")

    zero_action = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    # Success
    env.global_steps = 0
    env.reset(seed=args_cli.seed)

    success_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    root_local = env._root_pos_local()
    env.world.target_pos[success_ids] = root_local[success_ids, :2].clone()

    obs, reward, terminated, truncated, info = env.step(zero_action)
    success_hit = int(terminated[success_ids].sum().item())
    assert success_hit > 0, "forced success did not trigger terminated"
    assert "events" in info and "Success_Rate" in info["events"]
    print_ok(f"success 事件触发正常: {success_hit}/{len(success_ids)}")

    # Collision
    env.reset(seed=args_cli.seed)
    collision_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    root_local = env._root_pos_local()

    env.world.static_mask[collision_ids, 0] = True
    env.world.static_obs[collision_ids, 0, 0:2] = root_local[collision_ids, :2]
    env.world.static_obs[collision_ids, 0, 2] = 0.60

    obs, reward, terminated, truncated, info = env.step(zero_action)
    collision_hit = int(terminated[collision_ids].sum().item())
    assert collision_hit > 0, "forced collision did not trigger terminated"
    print_ok(f"collision 事件触发正常: {collision_hit}/{len(collision_ids)}")

    # Low height fall
    env.reset(seed=args_cli.seed)
    low_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    force_root_pose(env, low_ids, height=cfg.fall_height * 0.5)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    low_hit = int(terminated[low_ids].sum().item())
    assert low_hit > 0, "forced low height did not trigger terminated"
    print_ok(f"低高度 fall 事件触发正常: {low_hit}/{len(low_ids)}")

    # Tilt fall
    env.reset(seed=args_cli.seed)
    tilt_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    bad_quat = quat_from_roll_pitch_yaw(roll=1.35, pitch=0.0, yaw=0.0, device=env.device)
    force_root_pose(env, tilt_ids, height=cfg.target_height, quat=bad_quat)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    tilt_hit = int(terminated[tilt_ids].sum().item())
    assert tilt_hit > 0, "forced tilt did not trigger terminated"
    print_ok(f"倾斜 fall 事件触发正常: {tilt_hit}/{len(tilt_ids)}")

    # Out of bounds
    env.reset(seed=args_cli.seed)
    oob_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    local_xy = torch.zeros((len(oob_ids), 2), dtype=torch.float32, device=env.device)
    local_xy[:, 0] = cfg.world_cfg.env_size
    force_root_pose(env, oob_ids, local_xy=local_xy, height=cfg.target_height)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    oob_hit = int(terminated[oob_ids].sum().item())
    assert oob_hit > 0, "forced out_of_bounds did not trigger terminated"
    print_ok(f"out_of_bounds 事件触发正常: {oob_hit}/{len(oob_ids)}")

    # Timeout
    env.reset(seed=args_cli.seed)
    env.episode_steps[:] = cfg.max_episode_length - 1
    env.world.episode_steps[:] = cfg.world_cfg.max_episode_steps - 1

    obs, reward, terminated, truncated, info = env.step(zero_action)
    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "max_episode_length did not trigger truncated"
    print_ok(f"timeout 截断触发正常: truncated={timeout_count}")


def run_tests() -> None:
    heading("Go2 Task3 Navigation / Obstacle Avoidance Env 全量压测启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.steps = min(int(args_cli.steps), 120)
        args_cli.num_envs = min(int(args_cli.num_envs), 16)

    check_project_files()

    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.print_debug_info = bool(args_cli.print_names)

    assert cfg.num_actions == 12
    assert cfg.num_observations == 208
    assert cfg.num_privileged_obs == 276
    assert cfg.world_cfg.num_lidar_rays == 60

    env: Go2Task3Env | None = None

    try:
        heading("[测试 1] Go2Task3Env 初始化 / 名称映射 / 空间维度检测")
        env = Go2Task3Env(cfg)

        print_ok(f"device = {device}")
        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"num_actions = {cfg.num_actions}")
        print_ok(f"num_observations = {cfg.num_observations}")
        print_ok(f"num_privileged_obs = {cfg.num_privileged_obs}")
        print_ok(f"world privileged dim = {env.world.privileged_feature_dim()}")
        print_ok(f"lidar rays = {cfg.world_cfg.num_lidar_rays}")
        print_ok(f"action_joint_ids = {env.action_joint_ids}")
        print_ok(f"foot_body_ids = {env.foot_body_ids}")
        print_ok(f"contact_foot_ids = {env.contact_foot_ids}")

        assert env.robot.num_joints >= 12
        assert len(env.action_joint_ids) == 12
        assert len(env.foot_body_ids) == 4
        assert len(env.contact_foot_ids) == 4
        assert env.observation_space.shape == (cfg.num_observations,)
        assert env.state_space.shape == (cfg.num_privileged_obs,)
        assert env.action_space.shape == (cfg.num_actions,)
        assert env.world.privileged_feature_dim() == 68

        if args_cli.print_names:
            print("\nrobot.joint_names:")
            for i, name in enumerate(env.robot_joint_names):
                print(f"  {i:02d}: {name}")

            print("\nrobot.body_names:")
            for i, name in enumerate(env.robot_body_names):
                print(f"  {i:02d}: {name}")

            print("\ncontact.body_names:")
            for i, name in enumerate(env.contact.body_names):
                print(f"  {i:02d}: {name}")

        heading("[测试 2] reset / actor obs / privileged obs 维度与数值")
        obs, info = env.reset(seed=args_cli.seed)
        priv = env.compute_privileged_obs()

        check_obs_shape_and_values(env, obs)
        check_observation_slices(env, obs)
        check_priv_shape_and_values(env, priv)
        check_privileged_slices(env, priv)

        print_ok(f"reset obs shape = {tuple(obs.shape)}")
        print_ok(f"reset privileged obs shape = {tuple(priv.shape)}")
        print_ok(f"obs range = {obs.min().item():.4f} ~ {obs.max().item():.4f}")
        print_ok(f"priv range = {priv.min().item():.4f} ~ {priv.max().item():.4f}")

        heading("[测试 3] 随机动作控制链路")
        env.global_steps = int(0.20 * cfg.curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        q0 = env.robot.data.joint_pos[:, env.action_joint_ids_t].clone()

        test_action = torch.empty((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device).uniform_(-1.0, 1.0)

        latest_info: Dict[str, Any] = {}

        for _ in range(20):
            obs, reward, terminated, truncated, latest_info = env.step(test_action)

        q1 = env.robot.data.joint_pos[:, env.action_joint_ids_t].clone()
        q_delta = torch.norm(q1 - q0, dim=-1).mean().item()

        assert q_delta > 1e-5, "action did not change joint positions"
        assert reward.shape == (cfg.num_envs,)
        assert terminated.shape == (cfg.num_envs,)
        assert truncated.shape == (cfg.num_envs,)
        assert_finite_tensor("reward", reward)
        check_obs_shape_and_values(env, obs)
        check_observation_slices(env, obs)

        flat = flatten_info(latest_info)
        required_info_keys = [
            "reward_components/Total",
            "reward_components/R_Progress_Step",
            "reward_components/P_Obstacle_Risk",
            "events/Success_Rate",
            "events/Collision_Rate",
            "events/Fall_Rate",
            "telemetry/Command_Stage",
            "telemetry/Distance_To_Goal",
            "telemetry/Progress_Step",
            "telemetry/Collision_Risk",
            "telemetry/Base_Height",
            "telemetry/Contact_Count",
            "debug/Obs_Dim",
            "debug/Privileged_Obs_Dim",
            "debug/World_Priv_Dim",
        ]
        for key in required_info_keys:
            assert key in flat, f"info missing field: {key}"

        print_ok(f"控制链路正常，action joints 平均位移范数 = {q_delta:.6f}")
        print_ok("info 字段结构正常")

        check_command_stage_reset(env, cfg)
        check_reset_alignment(env, cfg)

        heading("[测试 6] contact sensor 检测")
        env.reset(seed=args_cli.seed)
        zero_action = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

        for _ in range(20):
            obs, reward, terminated, truncated, latest_info = env.step(zero_action)

        contact, normal_force = env._get_foot_contact()

        assert contact.shape == (cfg.num_envs, 4)
        assert normal_force.shape == (cfg.num_envs, 4)
        assert_finite_tensor("contact", contact)
        assert_finite_tensor("normal_force", normal_force)

        print_ok(f"contact shape = {tuple(contact.shape)}")
        print_ok(f"normal_force shape = {tuple(normal_force.shape)}")
        print_ok(
            f"contact mean: FL={contact[:, 0].mean().item():.4f}, "
            f"FR={contact[:, 1].mean().item():.4f}, "
            f"RL={contact[:, 2].mean().item():.4f}, "
            f"RR={contact[:, 3].mean().item():.4f}"
        )
        print_ok(f"normal force mean = {normal_force.mean().item():.6f}")

        heading("[测试 7] lidar / risk / target / world privileged 接口检测")
        root_pos_local = env._root_pos_local()
        yaw = env._quat_yaw(env.robot.data.root_quat_w)

        lidar = env.world.compute_lidar_tensors(root_pos_local, yaw, normalize=True)
        lidar_delta = env.world.compute_lidar_delta(lidar, env.prev_lidar)
        risk = env.world.compute_risk_features(root_pos_local, yaw)
        target_obs = env.world.get_target_obs(root_pos_local, yaw)
        world_priv = env.world.make_privileged_features(root_pos_local, yaw)

        assert lidar.shape == (cfg.num_envs, cfg.world_cfg.num_lidar_rays)
        assert lidar_delta.shape == (cfg.num_envs, cfg.world_cfg.num_lidar_rays)
        assert risk.shape == (cfg.num_envs, 8)
        assert target_obs.shape == (cfg.num_envs, 3)
        assert world_priv.shape == (cfg.num_envs, 68)

        assert_finite_tensor("lidar", lidar)
        assert_finite_tensor("lidar_delta", lidar_delta)
        assert_finite_tensor("risk", risk)
        assert_finite_tensor("target_obs", target_obs)
        assert_finite_tensor("world_priv", world_priv)

        assert lidar.min().item() >= -1e-5 and lidar.max().item() <= 1.0 + 1e-5
        assert lidar_delta.min().item() >= -1.0001 and lidar_delta.max().item() <= 1.0001
        assert risk[:, :5].min().item() >= -1e-5 and risk[:, :5].max().item() <= 1.0 + 1e-5
        assert risk[:, 7].min().item() >= -1e-5 and risk[:, 7].max().item() <= 1.0 + 1e-5

        print_ok(f"lidar shape = {tuple(lidar.shape)}")
        print_ok(f"risk shape = {tuple(risk.shape)}")
        print_ok(f"target_obs shape = {tuple(target_obs.shape)}")
        print_ok(f"world_priv shape = {tuple(world_priv.shape)}")
        print_ok("lidar / risk / target / world privileged 接口正常")

        heading("[测试 8] privileged obs 再检测")
        priv = env.compute_privileged_obs()
        check_priv_shape_and_values(env, priv)
        check_privileged_slices(env, priv)
        print_ok("privileged obs layout 正常")

        check_forced_events(env, cfg)

        heading("[测试 10] 随机策略长跑稳定性检测")
        env.global_steps = int(float(args_cli.rollout_k) * cfg.curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        records: List[Dict[str, float]] = []
        total_terminated = 0
        total_truncated = 0

        start_time = time.time()

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
                check_observation_slices(env, obs)
                assert_finite_tensor("rollout_reward", reward)

                priv = env.compute_privileged_obs()
                check_priv_shape_and_values(env, priv)

                flat = flatten_info(info)
                row = {
                    "test/step": float(step),
                    "test/reward_mean": float(reward.detach().mean().cpu().item()),
                    "test/terminated_rate": float(terminated.float().mean().cpu().item()),
                    "test/truncated_rate": float(truncated.float().mean().cpu().item()),
                }
                row.update(flat)
                records.append(row)

                print(
                    f"step={step + 1:>5}/{args_cli.steps} | "
                    f"reward={row.get('test/reward_mean', 0.0):>8.4f} | "
                    f"stage={row.get('telemetry/Command_Stage', 0.0):>4.1f} | "
                    f"dist={row.get('telemetry/Distance_To_Goal', 0.0):>6.3f} | "
                    f"progress={row.get('telemetry/Progress_Step', 0.0):>7.3f} | "
                    f"success={row.get('events/Success_Rate', 0.0):>6.3f} | "
                    f"collision={row.get('events/Collision_Rate', 0.0):>6.3f} | "
                    f"fall={row.get('events/Fall_Rate', 0.0):>6.3f} | "
                    f"h={row.get('telemetry/Base_Height', 0.0):>5.3f} | "
                    f"ct={row.get('telemetry/Contact_Count', 0.0):>4.2f}",
                    flush=True,
                )

        elapsed = time.time() - start_time
        fps = int(args_cli.steps) * int(cfg.num_envs) / max(elapsed, 1e-6)

        print_ok(f"随机策略长跑完成: {args_cli.steps} control steps")
        print_ok(f"总 transitions: {args_cli.steps * cfg.num_envs:,}")
        print_ok(f"吞吐约: {fps:,.2f} env steps/s")
        print_ok(f"累计 terminated: {total_terminated:,}")
        print_ok(f"累计 truncated: {total_truncated:,}")

        heading("[测试 11] 奖励组件 / 事件 / 遥测统计报告")
        print_summary_table(summarize_records(records))

        print("Go2 Task3 training pre-check guide:")
        print("1. actor obs 期望为 208，privileged obs 期望为 276。")
        print("2. lidar 和 lidar_delta 期望为 60 维，risk feature 期望为 8 维。")
        print("3. 随机策略下 collision / fall 属于可接受事件，数值稳定性要求为无 NaN/Inf。")
        print("4. success 在随机策略下很低是正常的。")
        print("5. 训练时重点看 Progress_Step、Distance_To_Goal、Success_Rate、Collision_Rate、Fall_Rate。")
        print("6. Task3 正式训练建议使用 Task1 或 Task2 actor warm-start。")

        heading("Go2 Task3 Navigation / Obstacle Avoidance Env 测试全部通过")

    except Exception as exc:
        print("\n[FAIL] Go2 Task3 环境测试失败：")
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
