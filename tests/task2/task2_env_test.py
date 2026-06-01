# Unitree Go2 Task2 environment test.
#
# Usage:
#   cd <repo_root>
#   python tests/task2/task2_env_test.py --num-envs 32 --steps 240 --headless --device cuda:0
#
# Important:
#   task2_env.py imports task2_world.py, and task2_world.py imports isaaclab.terrains.
#   Therefore AppLauncher must be launched before importing Go2Task2Env.

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


parser = argparse.ArgumentParser(description="Unitree Go2 Task2 Multi-Terrain Env White-Box Test")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rollout-k", type=float, default=0.35)
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--collect-interval", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from go2_rl.tasks.task2.task2_config import Task2Config
from go2_rl.tasks.task2.task2_env import Go2Task2Env


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

    print("\n" + "=" * 160)
    print(" " * 46 + "Unitree Go2 Task2 Environment Statistics")
    print("=" * 160)
    print(f"{'metric':<70} | {'mean':>14} | {'std':>14} | {'min':>14} | {'max':>14}")
    print("-" * 160)
    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<70} | "
            f"{row['mean']:>14.6f} | "
            f"{row['std']:>14.6f} | "
            f"{row['min']:>14.6f} | "
            f"{row['max']:>14.6f}"
        )
    print("=" * 160 + "\n")


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


def check_project_files() -> None:
    heading("[测试 0] Task2 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task2_multiterrain.yaml",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task2" / "task2_config.py",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task2" / "task2_world.py",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task2" / "task2_env.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "Missing required Task2 files:\n" + "\n".join(missing)

    for p in required:
        print_ok(str(p.relative_to(PROJECT_ROOT)))

    print_ok("Task2 工程文件结构正常")


def check_obs_shape_and_values(env: Go2Task2Env, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_observations)
    assert torch.is_tensor(obs), f"obs must be torch.Tensor, got {type(obs)}"
    assert tuple(obs.shape) == expected, f"obs shape wrong: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= 10.0001, f"obs out of clamp range: {obs.abs().max().item():.6f}"


def check_priv_shape_and_values(env: Go2Task2Env, priv: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_privileged_obs)
    assert torch.is_tensor(priv), f"privileged obs must be torch.Tensor, got {type(priv)}"
    assert tuple(priv.shape) == expected, f"privileged obs shape wrong: {tuple(priv.shape)} != {expected}"
    assert_finite_tensor("privileged_obs", priv)
    assert priv.abs().max().item() <= 20.0001, f"privileged obs out of clamp range: {priv.abs().max().item():.6f}"


def check_observation_slices(env: Go2Task2Env, obs: torch.Tensor) -> None:
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
    assert 0.10 <= h_mean <= 0.75, f"base_height mean abnormal: {h_mean:.4f}"


def check_privileged_slices(env: Go2Task2Env, priv: torch.Tensor) -> None:
    actor_dim = int(env.cfg.num_observations)
    terrain_dim = int(env.cfg.terrain_cfg.terrain_priv_dim)

    actor_part = priv[:, :actor_dim]
    terrain_part = priv[:, actor_dim:]

    assert actor_part.shape == (env.cfg.num_envs, actor_dim)
    assert terrain_part.shape == (env.cfg.num_envs, terrain_dim)
    assert terrain_dim == 91, f"terrain privileged dim should be 91, got {terrain_dim}"

    height_scan = terrain_part[:, :81]
    friction = terrain_part[:, 81]
    terrain_onehot = terrain_part[:, 82:86]
    difficulty = terrain_part[:, 86]
    param4 = terrain_part[:, 87:91]

    assert height_scan.shape == (env.cfg.num_envs, 81)
    assert friction.shape == (env.cfg.num_envs,)
    assert terrain_onehot.shape == (env.cfg.num_envs, 4)
    assert difficulty.shape == (env.cfg.num_envs,)
    assert param4.shape == (env.cfg.num_envs, 4)

    assert_finite_tensor("height_scan", height_scan)
    assert_finite_tensor("friction", friction)
    assert_finite_tensor("terrain_onehot", terrain_onehot)
    assert_finite_tensor("difficulty", difficulty)
    assert_finite_tensor("param4", param4)

    assert height_scan.abs().max().item() <= env.cfg.terrain_cfg.height_scan_clip + 1e-5
    assert friction.min().item() >= env.cfg.terrain_cfg.friction_range[0] - 1e-4
    assert friction.max().item() <= env.cfg.terrain_cfg.friction_range[1] + 1e-4
    assert torch.allclose(terrain_onehot.sum(dim=-1), torch.ones(env.cfg.num_envs, device=env.device), atol=1e-5)
    assert difficulty.min().item() >= -1e-5
    assert difficulty.max().item() <= 1.0 + 1e-5


def check_command_stages(env: Go2Task2Env) -> None:
    heading("[测试 4] command stage / terrain restriction 检测")

    cfg = env.cfg
    old_steps = int(env.global_steps)

    stage_rows = []

    checks = [
        (0.00, 0),
        (0.05, 1),
        (0.15, 2),
        (0.30, 3),
        (0.50, 4),
        (0.75, 5),
        (1.00, 5),
    ]

    for k, expected_stage in checks:
        env.global_steps = int(k * cfg.terrain_curriculum_total_steps)
        got_stage = env._command_stage()
        assert got_stage == expected_stage, f"k={k:.2f} stage wrong: got {got_stage}, expected {expected_stage}"

    for k in [0.0, 0.10, 0.20, 0.40, 0.65, 0.90]:
        env.global_steps = int(k * cfg.terrain_curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        stage = env._command_stage()
        max_allowed = env._max_allowed_terrain_level()

        type_min = int(env.env_terrain_types.min().item())
        type_max = int(env.env_terrain_types.max().item())
        level_min = int(env.env_terrain_levels.min().item())
        level_max = int(env.env_terrain_levels.max().item())

        assert level_max <= max_allowed, f"k={k} level_max={level_max} > max_allowed={max_allowed}"
        if stage == 0:
            assert type_min == 0 and type_max == 0, "stage 0 should use only rough_flat"
            assert level_min == 0 and level_max == 0, "stage 0 should use only level 0"

        stage_rows.append(
            {
                "k": k,
                "stage": stage,
                "max_allowed": max_allowed,
                "type_min": type_min,
                "type_max": type_max,
                "level_min": level_min,
                "level_max": level_max,
            }
        )

    env.global_steps = old_steps

    for row in stage_rows:
        print_ok(
            f"k={row['k']:.2f}, stage={row['stage']}, "
            f"type={row['type_min']}~{row['type_max']}, "
            f"level={row['level_min']}~{row['level_max']}, "
            f"max_allowed={row['max_allowed']}"
        )

    print_ok("command stage / terrain restriction 正常")


def check_terrain_reset_spawn(env: Go2Task2Env) -> None:
    heading("[测试 5] terrain reset / spawn origin / material 检测")

    cfg = env.cfg
    old_steps = int(env.global_steps)

    env.global_steps = int(0.60 * cfg.terrain_curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

    check_obs_shape_and_values(env, obs)

    terrain_types = env.env_terrain_types
    terrain_levels = env.env_terrain_levels

    assert terrain_types.min().item() >= 0
    assert terrain_types.max().item() < cfg.terrain_cfg.num_terrain_types
    assert terrain_levels.min().item() >= 0
    assert terrain_levels.max().item() < cfg.terrain_cfg.num_levels

    origins = env.world.get_origins_from_indices(terrain_types, terrain_levels, prefer_scene_origins=True)
    root_xy = env.robot.data.root_pos_w[:, :2]
    xy_delta = torch.abs(root_xy - origins[:, :2])
    max_xy = xy_delta.max().item()

    assert max_xy <= cfg.terrain_cfg.spawn_radius + 1e-4, (
        f"root spawn xy exceeds spawn_radius: {max_xy} > {cfg.terrain_cfg.spawn_radius}"
    )

    base_height = env._compute_base_height()
    assert_finite_tensor("base_height", base_height)
    assert 0.10 <= base_height.mean().item() <= 0.75

    assert env.env_friction.min().item() >= cfg.terrain_cfg.friction_range[0] - 1e-5
    assert env.env_friction.max().item() <= cfg.terrain_cfg.friction_range[1] + 1e-5
    assert torch.allclose(
        env.env_material_onehot.sum(dim=-1),
        torch.ones(cfg.num_envs, device=env.device),
        atol=1e-5,
    )

    env.global_steps = old_steps

    print_ok(f"terrain type range = {terrain_types.min().item()} ~ {terrain_types.max().item()}")
    print_ok(f"terrain level range = {terrain_levels.min().item()} ~ {terrain_levels.max().item()}")
    print_ok(f"max |root_xy - terrain_origin_xy| = {max_xy:.6f}")
    print_ok(f"base_height mean = {base_height.mean().item():.6f}")
    print_ok(f"friction range = {env.env_friction.min().item():.4f} ~ {env.env_friction.max().item():.4f}")
    print_ok("terrain reset / spawn / material 正常")


def force_root_pose(
    env: Go2Task2Env,
    env_ids: torch.Tensor,
    height_offset: float,
    quat: torch.Tensor | None = None,
) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)

    terrain_types = env.env_terrain_types[env_ids]
    terrain_levels = env.env_terrain_levels[env_ids]
    origins = env.world.get_origins_from_indices(terrain_types, terrain_levels, prefer_scene_origins=True)

    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:2] = origins[:, 0:2]

    terrain_h = env._get_terrain_height_under_points(root_state[:, 0:2], terrain_types, terrain_levels).squeeze(-1)
    root_state[:, 2] = terrain_h + float(height_offset)

    if quat is None:
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
    else:
        root_state[:, 3:7] = quat.repeat(len(env_ids), 1)

    root_state[:, 7:13] = 0.0

    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env.scene.update(dt=0.0)


def check_forced_events(env: Go2Task2Env) -> None:
    heading("[测试 9] 强制 fall / tilt / timeout 事件检测")

    cfg = env.cfg
    zero_action = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    env.reset(seed=args_cli.seed)

    low_ids = torch.arange(min(16, cfg.num_envs), device=env.device)
    force_root_pose(env, low_ids, height_offset=cfg.fall_height * 0.5)
    _, _, terminated, _, _ = env.step(zero_action)
    low_hit = int(terminated[low_ids].sum().item())
    assert low_hit > 0, "forced low height did not trigger terminated"
    print_ok(f"低高度摔倒事件触发正常: {low_hit}/{len(low_ids)}")

    env.reset(seed=args_cli.seed)

    tilt_ids = torch.arange(min(16, cfg.num_envs), device=env.device)
    bad_quat = quat_from_roll_pitch_yaw(roll=1.40, device=env.device)
    force_root_pose(env, tilt_ids, height_offset=cfg.target_height, quat=bad_quat)
    _, _, terminated, _, _ = env.step(zero_action)
    tilt_hit = int(terminated[tilt_ids].sum().item())
    assert tilt_hit > 0, "forced tilt did not trigger terminated"
    print_ok(f"倾斜摔倒事件触发正常: {tilt_hit}/{len(tilt_ids)}")

    env.reset(seed=args_cli.seed)

    env.episode_steps[:] = cfg.max_episode_length - 1
    _, _, _, truncated, _ = env.step(zero_action)
    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "max_episode_length did not trigger truncated"
    print_ok(f"超时截断触发正常: truncated={timeout_count}")


def check_curriculum_update_interface(env: Go2Task2Env) -> None:
    heading("[测试 10] terrain curriculum update 接口检测")

    cfg = env.cfg

    old_types = env.terrain_curriculum.env_types.clone()
    old_levels = env.terrain_curriculum.env_levels.clone()

    active_ids = (~env.terrain_curriculum.anchor_mask).nonzero(as_tuple=False).squeeze(-1)
    active_ids = active_ids[: min(64, len(active_ids))]

    if active_ids.numel() == 0:
        print_warn("No active env found. Skip curriculum update interface test.")
        return

    env.terrain_curriculum.env_levels[active_ids] = 3
    env.terrain_curriculum.register_start_positions(active_ids, torch.zeros(len(active_ids), device=env.device))

    current_x = torch.full((len(active_ids),), cfg.terrain_cfg.success_distance + 1.0, device=env.device)
    fall_flags = torch.zeros(len(active_ids), dtype=torch.bool, device=env.device)

    before = env.terrain_curriculum.env_levels[active_ids].clone()
    env.terrain_curriculum.update_curriculum(active_ids, current_x, fall_flags)
    after_success = env.terrain_curriculum.env_levels[active_ids].clone()

    assert (after_success >= before).all(), "success should keep or upgrade terrain level"

    env.terrain_curriculum.register_start_positions(active_ids, torch.zeros(len(active_ids), device=env.device))

    current_x = torch.full((len(active_ids),), cfg.terrain_cfg.failure_distance * 0.25, device=env.device)
    fall_flags = torch.ones(len(active_ids), dtype=torch.bool, device=env.device)

    before_fail = env.terrain_curriculum.env_levels[active_ids].clone()
    env.terrain_curriculum.update_curriculum(active_ids, current_x, fall_flags)
    after_fail = env.terrain_curriculum.env_levels[active_ids].clone()

    downgraded_ratio = (after_fail < before_fail).float().mean().item()

    env.terrain_curriculum.env_types[:] = old_types
    env.terrain_curriculum.env_levels[:] = old_levels

    print_ok(f"成功升级后 mean level = {after_success.float().mean().item():.4f}")
    print_ok(f"失败降级比例 = {downgraded_ratio:.4f}")
    print_ok("terrain curriculum update 接口正常")


def run_tests() -> None:
    heading("Unitree Go2 Task2 Multi-Terrain Env / Reward / Privileged Obs Test")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    check_project_files()

    cfg = Task2Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.print_debug_info = bool(args_cli.print_names)

    assert cfg.num_observations == 87
    assert cfg.num_privileged_obs == 178
    assert cfg.terrain_cfg.terrain_priv_dim == 91

    env: Go2Task2Env | None = None

    try:
        heading("[测试 1] 环境初始化 / Robot / Terrain / Contact 映射")
        env = Go2Task2Env(cfg)

        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"device = {env.device}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"num_actions = {cfg.num_actions}")
        print_ok(f"num_observations = {cfg.num_observations}")
        print_ok(f"num_privileged_obs = {cfg.num_privileged_obs}")
        print_ok(f"terrain_priv_dim = {cfg.terrain_cfg.terrain_priv_dim}")
        print_ok(f"action_joint_ids = {env.action_joint_ids}")
        print_ok(f"foot_body_ids = {env.foot_body_ids}")
        print_ok(f"contact_foot_ids = {env.contact_foot_ids}")

        assert env.action_space.shape == (cfg.num_actions,)
        assert env.observation_space.shape == (cfg.num_observations,)
        assert env.state_space.shape == (cfg.num_privileged_obs,)
        assert len(env.action_joint_ids) == cfg.num_actions
        assert len(env.foot_body_ids) == 4
        assert len(env.contact_foot_ids) == 4

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
        env.global_steps = int(0.20 * cfg.terrain_curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        q0 = env.robot.data.joint_pos[:, env.action_joint_ids_t].clone()

        test_action = torch.empty((cfg.num_envs, cfg.num_actions), device=env.device).uniform_(-1.0, 1.0)
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

        flat = flatten_info(latest_info)
        required_info_keys = [
            "reward_components/Total",
            "reward_components/R_Terrain_Progress",
            "events/Fall_Rate",
            "telemetry/Base_Height",
            "telemetry/Contact_Count",
            "telemetry/Mean_Terrain_Level",
            "telemetry/Mean_Friction",
            "curriculum/Curriculum/Mean_Level_Active",
            "debug/Privileged_Obs_Dim",
            "debug/Terrain_Priv_Dim",
        ]
        for key in required_info_keys:
            assert key in flat, f"info missing field: {key}"

        print_ok(f"控制链路正常，action joints 平均位移范数 = {q_delta:.6f}")
        print_ok("info 字段结构正常")

        check_command_stages(env)
        check_terrain_reset_spawn(env)

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

        heading("[测试 7] terrain height / foot height / height scan 接口检测")
        base_height = env._compute_base_height()
        foot_height = env._compute_foot_heights()
        height_scan = env.world.sample_height_scan(
            base_pos_w=env.robot.data.root_pos_w,
            terrain_types=env.env_terrain_types,
            terrain_levels=env.env_terrain_levels,
            base_quat_wxyz=env.robot.data.root_quat_w,
            prefer_scene_origins=True,
        )

        assert base_height.shape == (cfg.num_envs,)
        assert foot_height.shape == (cfg.num_envs, 4)
        assert height_scan.shape == (cfg.num_envs, 81)

        assert_finite_tensor("base_height", base_height)
        assert_finite_tensor("foot_height", foot_height)
        assert_finite_tensor("height_scan", height_scan)

        assert height_scan.abs().max().item() <= cfg.terrain_cfg.height_scan_clip + 1e-5

        print_ok(f"base_height range = {base_height.min().item():.4f} ~ {base_height.max().item():.4f}")
        print_ok(f"foot_height shape = {tuple(foot_height.shape)}")
        print_ok(f"height_scan shape = {tuple(height_scan.shape)}")
        print_ok("terrain height / height scan 接口正常")

        heading("[测试 8] privileged obs 再检测")
        priv = env.compute_privileged_obs()
        check_priv_shape_and_values(env, priv)
        check_privileged_slices(env, priv)
        print_ok("privileged obs layout 正常")

        check_forced_events(env)
        check_curriculum_update_interface(env)

        heading("[测试 11] 随机策略长跑稳定性检测")
        env.global_steps = int(float(args_cli.rollout_k) * cfg.terrain_curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        records: List[Dict[str, float]] = []
        total_falls = 0
        total_timeouts = 0
        start_time = time.time()

        for step in range(int(args_cli.steps)):
            action = torch.empty(
                (cfg.num_envs, cfg.num_actions),
                dtype=torch.float32,
                device=env.device,
            ).uniform_(-1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            total_falls += int(terminated.sum().item())
            total_timeouts += int(truncated.sum().item())

            if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                check_obs_shape_and_values(env, obs)
                assert_finite_tensor("rollout_reward", reward)

                priv = env.compute_privileged_obs()
                check_priv_shape_and_values(env, priv)

                flat = flatten_info(info)
                row = {
                    "test/step": float(step),
                    "test/reward_mean": float(reward.detach().mean().cpu().item()),
                }
                row.update(flat)
                records.append(row)

                msg = (
                    f"step={step + 1:>5}/{args_cli.steps} | "
                    f"reward={row.get('test/reward_mean', 0.0):>8.4f} | "
                    f"fall={row.get('events/Fall_Rate', 0.0):>7.4f} | "
                    f"stage={row.get('telemetry/Command_Stage', 0.0):>4.0f} | "
                    f"level={row.get('telemetry/Mean_Terrain_Level', 0.0):>5.2f} | "
                    f"fric={row.get('telemetry/Mean_Friction', 0.0):>5.2f} | "
                    f"h={row.get('telemetry/Base_Height', 0.0):>6.3f} | "
                    f"contacts={row.get('telemetry/Contact_Count', 0.0):>5.2f}"
                )
                print(msg, flush=True)

        elapsed = time.time() - start_time
        fps = int(args_cli.steps) * int(cfg.num_envs) / max(elapsed, 1e-6)

        print_ok(f"随机策略长跑完成: {args_cli.steps} steps, {args_cli.steps * cfg.num_envs:,} transitions")
        print_ok(f"吞吐约: {fps:,.2f} env steps/s")
        print_ok(f"累计 terminated: {total_falls:,}")
        print_ok(f"累计 truncated: {total_timeouts:,}")

        heading("[测试 12] 奖励组件 / 遥测 / 事件统计")
        print_summary_table(summarize_records(records))

        print("Go2 Task2 training pre-check guide:")
        print("1. actor obs must be 87, privileged obs must be 178.")
        print("2. terrain privileged part must be 91, height scan must be 81.")
        print("3. Random policy Fall_Rate can be high, but NaN/Inf is not allowed.")
        print("4. Key metrics: Actual_Vx/Cmd_Vx, Fall_Rate, Contact_Count, P_Foot_Slip, Mean_Terrain_Level.")
        print("5. If Mean_Terrain_Level never changes during training, inspect terrain curriculum thresholds.")
        print("\n[OK] Unitree Go2 Task2 environment test completed.")

    except Exception as exc:
        print("\n[FAIL] Unitree Go2 Task2 environment test failed:")
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
