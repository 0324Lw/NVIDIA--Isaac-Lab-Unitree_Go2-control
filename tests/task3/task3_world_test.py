# Copyright (c) 2026
# Unitree Go2 Task3: 解析式导航避障世界模型白盒测试。
#
# 本文件用于检查 Task3World 的目标采样、静态/动态障碍物、解析 lidar、risk features、
# termination 事件和 world privileged features。
#
# 测试入口:
#   bash scripts/ubuntu/test_task3_world.sh
#
# 观测维度:
#   lidar rays = 60
#   world privileged tail = 68
#
# 工程说明:
#   Task3World 是纯 torch 解析世界，不依赖 IsaacLab，不生成 obstacle prim。
#   compute_lidar_tensors 是 full-batch API，因为 obstacle tensor 以 [num_envs, ...] 形式保存。
#
# Unitree Go2 Task3: analytical navigation and obstacle-avoidance world-model white-box test.
#
# This file checks Task3World goal sampling, static/dynamic obstacles, analytical lidar,
# risk features, termination events, and world privileged features.
#
# Test entry:
#   bash scripts/ubuntu/test_task3_world.sh
#
# Observation dimensions:
#   lidar rays = 60
#   world privileged tail = 68
#
# Engineering notes:
#   Task3World is a pure-torch analytical world. It does not depend on IsaacLab and
#   does not spawn obstacle prims. compute_lidar_tensors is a full-batch API because
#   obstacle tensors are stored as [num_envs, ...].

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

from go2_rl.tasks.task3.task3_config import Task3WorldCfg
from go2_rl.tasks.task3.task3_world import Task3World


parser = argparse.ArgumentParser(description="Go2 Task3 Analytical World White-Box Test")
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--print-detail", action="store_true")
args = parser.parse_args()


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


def assert_range(name: str, x: torch.Tensor, lo: float, hi: float, eps: float = 1e-5) -> None:
    assert_finite_tensor(name, x)
    mn = float(x.min().detach().cpu().item())
    mx = float(x.max().detach().cpu().item())
    assert mn >= lo - eps and mx <= hi + eps, f"{name} out of range [{lo}, {hi}]: got {mn} ~ {mx}"


def flatten_stats(stats: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().float().mean().cpu().item())
        elif isinstance(v, (float, int, np.floating, np.integer)):
            out[k] = float(v)
    return out


def print_table(rows: List[Dict[str, Any]], title: str = "") -> None:
    if title:
        print("\n" + title)
    if not rows:
        print("<empty>")
        return

    keys = list(rows[0].keys())
    widths = {}
    for k in keys:
        widths[k] = max(len(str(k)), max(len(str(row.get(k, ""))) for row in rows))

    header = " | ".join(f"{k:<{widths[k]}}" for k in keys)
    print(header)
    print("-" * len(header))

    for row in rows:
        print(" | ".join(f"{str(row.get(k, '')):<{widths[k]}}" for k in keys))


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


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    if not summary:
        print_warn("No summary records collected.")
        return

    print("\n" + "=" * 160)
    print("Go2 Task3 Analytical World Statistics")
    print("=" * 160)
    print(f"{'metric':<72} | {'mean':>14} | {'std':>14} | {'min':>14} | {'max':>14}")
    print("-" * 160)
    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<72} | "
            f"{row['mean']:>14.6f} | "
            f"{row['std']:>14.6f} | "
            f"{row['min']:>14.6f} | "
            f"{row['max']:>14.6f}"
        )
    print("=" * 160 + "\n")


def check_project_files() -> None:
    heading("[测试 0] Task3 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task3_navigation.yaml",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task3" / "task3_config.py",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task3" / "task3_world.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "Missing required Task3 files:\n" + "\n".join(missing)

    for p in required:
        print_ok(str(p.relative_to(PROJECT_ROOT)))

    print_ok("Task3 world 工程文件结构正常")


def test_config(cfg: Task3WorldCfg) -> None:
    heading("[测试 1] Task3WorldCfg 基础配置检测")

    assert cfg.pd_control_freq == 200.0
    assert cfg.rl_policy_freq == 50.0
    assert cfg.decimation == 4
    assert abs(cfg.policy_dt - 0.02) < 1e-8

    assert cfg.env_size == 30.0
    assert cfg.num_lidar_rays == 60
    assert cfg.lidar_max_distance == 6.0
    assert cfg.max_static_obs == 25
    assert cfg.max_dynamic_obs == 8
    assert len(cfg.stage_thresholds) == 6
    assert len(cfg.success_radius_by_stage) == 6
    assert cfg.max_episode_steps == int(cfg.max_episode_length_s * cfg.rl_policy_freq)

    expected_priv = (
        cfg.privileged_static_k * 4
        + cfg.privileged_dynamic_k * 6
        + 8
        + 3
        + len(cfg.stage_thresholds)
        + 3
    )
    assert expected_priv == 68

    print_ok(f"stage_count = {len(cfg.stage_thresholds)}")
    print_ok(f"max_episode_steps = {cfg.max_episode_steps}")
    print_ok(f"lidar rays = {cfg.num_lidar_rays}")
    print_ok(f"max_static_obs = {cfg.max_static_obs}")
    print_ok(f"max_dynamic_obs = {cfg.max_dynamic_obs}")
    print_ok(f"privileged_feature_dim = {expected_priv}")
    print_ok("Task3WorldCfg 基础配置正常")


def test_world_init(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 2] Task3World 初始化张量 shape / finite 检测")

    assert world.num_envs == num_envs
    assert world.device == device
    assert world.stage_count == 6
    assert world.privileged_feature_dim() == 68

    expected_shapes = {
        "start_pos": (num_envs, 2),
        "target_pos": (num_envs, 2),
        "static_obs": (num_envs, cfg.max_static_obs, 3),
        "static_mask": (num_envs, cfg.max_static_obs),
        "dynamic_obs_pos": (num_envs, cfg.max_dynamic_obs, 2),
        "dynamic_obs_vel": (num_envs, cfg.max_dynamic_obs, 2),
        "dynamic_obs_radius": (num_envs, cfg.max_dynamic_obs),
        "dynamic_mask": (num_envs, cfg.max_dynamic_obs),
        "env_stage": (num_envs,),
        "env_target_speed": (num_envs,),
        "episode_steps": (num_envs,),
        "last_distance_to_target": (num_envs,),
        "ray_angles": (cfg.num_lidar_rays,),
    }

    for name, shape in expected_shapes.items():
        value = getattr(world, name)
        assert tuple(value.shape) == shape, f"{name} shape wrong: got {tuple(value.shape)}, expected {shape}"
        if torch.is_floating_point(value):
            assert_finite_tensor(name, value)

    print_ok("所有核心张量 shape 正常")
    print_ok("所有浮点张量 finite 正常")


def test_curriculum(cfg: Task3WorldCfg, world: Task3World) -> None:
    heading("[测试 3] curriculum_k / stage_from_progress 边界检测")

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

    rows = []
    for k, expected in checks:
        got = world.stage_from_progress(k)
        assert got == expected, f"k={k} stage wrong: got {got}, expected {expected}"

        steps = int(k * cfg.curriculum_total_steps)
        got_from_steps = world.stage_from_global_steps(steps)
        assert got_from_steps == expected

        rows.append({"k": k, "steps": steps, "stage": got})

    print_table(rows)
    print_ok("课程阶段边界正常")


def test_reset_by_stage(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 4] reset_envs stage 采样 / 起终点 / 障碍数量检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)

    rows = []
    for stage in range(world.stage_count):
        stages = torch.full((num_envs,), stage, dtype=torch.long, device=device)
        world.reset_envs(env_ids, stages=stages)

        assert (world.env_stage == stage).all()

        start = world.start_pos
        target = world.target_pos
        dist = torch.norm(target - start, dim=-1)

        goal_min, goal_max = cfg.goal_dist_ranges[stage]
        assert dist.min().item() >= goal_min - 1e-4
        assert dist.max().item() <= goal_max + 1e-4

        bound = cfg.env_size * 0.5 - cfg.wall_margin - cfg.safe_zone_radius
        assert torch.abs(start).max().item() <= bound + 1e-4
        assert torch.abs(target).max().item() <= bound + 1e-4

        static_count = world.static_mask.float().sum(dim=-1)
        dynamic_count = world.dynamic_mask.float().sum(dim=-1)

        s_min, s_max = cfg.static_count_ranges[stage]
        d_min, d_max = cfg.dynamic_count_ranges[stage]

        assert static_count.min().item() >= s_min
        assert static_count.max().item() <= s_max
        assert dynamic_count.min().item() >= d_min
        assert dynamic_count.max().item() <= d_max

        speed = world.env_target_speed
        v_min, v_max = cfg.target_speed_ranges[stage]
        assert speed.min().item() >= v_min - 1e-5
        assert speed.max().item() <= v_max + 1e-5

        assert_finite_tensor("start_pos", start)
        assert_finite_tensor("target_pos", target)
        assert_finite_tensor("static_obs", world.static_obs)
        assert_finite_tensor("dynamic_obs_pos", world.dynamic_obs_pos)
        assert_finite_tensor("dynamic_obs_vel", world.dynamic_obs_vel)

        rows.append(
            {
                "stage": stage,
                "goal_dist": f"{dist.mean().item():.3f}",
                "goal_range": f"{dist.min().item():.2f}~{dist.max().item():.2f}",
                "static_count": f"{static_count.mean().item():.2f}",
                "dynamic_count": f"{dynamic_count.mean().item():.2f}",
                "target_speed": f"{speed.mean().item():.2f}",
            }
        )

    print_table(rows)
    print_ok("各 stage reset / 起终点 / 障碍数量 / 目标速度采样正常")


def test_obstacle_safety(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 5] 障碍物安全区 / 起点终点避让 / 障碍间距检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    stages = torch.full((num_envs,), 5, dtype=torch.long, device=device)
    world.reset_envs(env_ids, stages=stages)

    # Static obstacles should avoid start and target safe zones.
    if cfg.max_static_obs > 0:
        stat_pos = world.static_obs[:, :, :2]
        stat_r = world.static_obs[:, :, 2]
        stat_mask = world.static_mask

        dist_start = torch.norm(stat_pos - world.start_pos.unsqueeze(1), dim=-1)
        dist_target = torch.norm(stat_pos - world.target_pos.unsqueeze(1), dim=-1)
        safe_threshold = cfg.safe_zone_radius + stat_r + cfg.obstacle_spawn_buffer

        bad_start = ((dist_start <= safe_threshold) & stat_mask).any()
        bad_target = ((dist_target <= safe_threshold) & stat_mask).any()

        assert not bad_start.item(), "static obstacle overlaps start safe zone"
        assert not bad_target.item(), "static obstacle overlaps target safe zone"

    # Dynamic obstacles should avoid start, target, and static obstacles.
    if cfg.max_dynamic_obs > 0:
        dyn_pos = world.dynamic_obs_pos
        dyn_r = world.dynamic_obs_radius
        dyn_mask = world.dynamic_mask

        dist_start = torch.norm(dyn_pos - world.start_pos.unsqueeze(1), dim=-1)
        dist_target = torch.norm(dyn_pos - world.target_pos.unsqueeze(1), dim=-1)
        safe_threshold = cfg.safe_zone_radius + dyn_r + cfg.obstacle_spawn_buffer

        bad_start = ((dist_start <= safe_threshold) & dyn_mask).any()
        bad_target = ((dist_target <= safe_threshold) & dyn_mask).any()

        assert not bad_start.item(), "dynamic obstacle overlaps start safe zone"
        assert not bad_target.item(), "dynamic obstacle overlaps target safe zone"

        if cfg.max_static_obs > 0:
            stat_pos = world.static_obs[:, :, :2]
            stat_r = world.static_obs[:, :, 2]
            stat_mask = world.static_mask

            dist_ds = torch.norm(dyn_pos.unsqueeze(2) - stat_pos.unsqueeze(1), dim=-1)
            th_ds = dyn_r.unsqueeze(2) + stat_r.unsqueeze(1) + cfg.min_dynamic_spacing
            conflict_ds = ((dist_ds < th_ds) & dyn_mask.unsqueeze(2) & stat_mask.unsqueeze(1)).any()

            assert not conflict_ds.item(), "dynamic obstacle conflicts with static obstacle"

    print_ok("障碍物不压起点 / 不压终点 / 静动态避让正常")


def test_dynamic_kinematics(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 6] 动态障碍运动 / 边界反弹检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    stages = torch.full((num_envs,), 5, dtype=torch.long, device=device)
    world.reset_envs(env_ids, stages=stages)

    old_pos = world.dynamic_obs_pos.clone()
    old_vel = world.dynamic_obs_vel.clone()

    world.step_kinematics(dt=0.10)

    new_pos = world.dynamic_obs_pos
    active = world.dynamic_mask

    if active.any():
        moved = torch.norm(new_pos - old_pos, dim=-1)
        mean_moved = moved[active].mean().item()
        assert mean_moved > 1e-6, "active dynamic obstacles did not move"
    else:
        mean_moved = 0.0

    # Force boundary reflection on first obstacle.
    if cfg.max_dynamic_obs > 0:
        world.dynamic_mask[:, 0] = True
        world.dynamic_obs_radius[:, 0] = 0.40

        half = cfg.env_size * 0.5
        bound = half - cfg.wall_margin - world.dynamic_obs_radius[:, 0]

        world.dynamic_obs_pos[:, 0, 0] = bound - 0.001
        world.dynamic_obs_pos[:, 0, 1] = 0.0
        world.dynamic_obs_vel[:, 0, 0] = 1.0
        world.dynamic_obs_vel[:, 0, 1] = 0.0

        world.step_kinematics(dt=0.10)

        reflected = (world.dynamic_obs_vel[:, 0, 0] < 0.0).float().mean().item()
        assert reflected > 0.99, f"boundary reflection failed: reflected ratio={reflected}"

    assert_finite_tensor("dynamic_obs_pos after kinematics", world.dynamic_obs_pos)
    assert_finite_tensor("dynamic_obs_vel after kinematics", world.dynamic_obs_vel)

    print_ok(f"active dynamic mean displacement = {mean_moved:.6f}")
    print_ok("动态障碍运动 / 边界反弹正常")


def test_target_navigation(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 7] target polar / target obs / progress 检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    stages = torch.full((num_envs,), 0, dtype=torch.long, device=device)
    world.reset_envs(env_ids, stages=stages)

    robot_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    robot_pos[:, :2] = world.start_pos
    robot_yaw = torch.zeros(num_envs, dtype=torch.float32, device=device)

    polar = world.get_target_polar_coords(robot_pos, robot_yaw)
    target_obs = world.get_target_obs(robot_pos, robot_yaw)
    dist = world.distance_to_target(robot_pos)

    assert polar.shape == (num_envs, 2)
    assert target_obs.shape == (num_envs, 3)
    assert dist.shape == (num_envs,)

    assert_finite_tensor("target polar", polar)
    assert_finite_tensor("target obs", target_obs)
    assert_finite_tensor("distance_to_target", dist)

    assert torch.allclose(polar[:, 0], dist, atol=1e-5)
    assert_range("target_obs distance_norm", target_obs[:, 0], 0.0, 2.0)
    assert_range("target_obs sin", target_obs[:, 1], -1.0, 1.0)
    assert_range("target_obs cos", target_obs[:, 2], -1.0, 1.0)

    old_dist = world.last_distance_to_target.clone()

    delta = world.target_pos - world.start_pos
    unit = delta / torch.clamp(torch.norm(delta, dim=-1, keepdim=True), min=1e-6)
    robot_pos[:, :2] = robot_pos[:, :2] + unit * 0.10

    progress = world.compute_progress(robot_pos, dt=0.02)
    assert progress.mean().item() > 0.0, "moving toward target should produce positive progress"

    print_ok(f"initial distance mean = {old_dist.mean().item():.4f}")
    print_ok(f"progress mean after 0.10m toward target = {progress.mean().item():.4f}")
    print_ok("target navigation interfaces 正常")


def test_lidar_risk_collision(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 8] lidar / risk / collision 解析几何检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    stages = torch.zeros(num_envs, dtype=torch.long, device=device)
    world.reset_envs(env_ids, stages=stages)

    # Manually place one obstacle exactly in front of the robot.
    world.static_obs[:] = 0.0
    world.static_mask[:] = False
    world.dynamic_obs_pos[:] = 0.0
    world.dynamic_obs_vel[:] = 0.0
    world.dynamic_obs_radius[:] = 0.0
    world.dynamic_mask[:] = False

    world.static_mask[:, 0] = True
    world.static_obs[:, 0, 0] = 2.0
    world.static_obs[:, 0, 1] = 0.0
    world.static_obs[:, 0, 2] = 0.50

    robot_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    robot_yaw = torch.zeros(num_envs, dtype=torch.float32, device=device)

    lidar = world.compute_lidar_tensors(robot_pos, robot_yaw, normalize=False)
    lidar_norm = world.compute_lidar_tensors(robot_pos, robot_yaw, normalize=True)
    risk = world.compute_risk_features(robot_pos, robot_yaw)

    assert lidar.shape == (num_envs, cfg.num_lidar_rays)
    assert lidar_norm.shape == (num_envs, cfg.num_lidar_rays)
    assert risk.shape == (num_envs, 8)

    assert_finite_tensor("lidar", lidar)
    assert_finite_tensor("lidar_norm", lidar_norm)
    assert_finite_tensor("risk", risk)

    assert_range("lidar", lidar, 0.0, cfg.lidar_max_distance)
    assert_range("lidar_norm", lidar_norm, 0.0, 1.0)
    assert_range("risk normalized distances", risk[:, :5], 0.0, 1.0)
    assert_range("risk sin/cos", risk[:, 5:7], -1.0, 1.0)
    assert_range("risk collision", risk[:, 7], 0.0, 1.0)

    expected_front_hit = 2.0 - 0.50
    front_hit = lidar[:, 0].mean().item()
    assert abs(front_hit - expected_front_hit) < 1e-3, (
        f"front lidar hit wrong: got {front_hit}, expected {expected_front_hit}"
    )

    lidar_delta = world.compute_lidar_delta(lidar_norm, torch.ones_like(lidar_norm))
    assert lidar_delta.shape == lidar_norm.shape
    assert_range("lidar_delta", lidar_delta, -1.0, 1.0)

    signed, signed_static, signed_dyn = world.obstacle_signed_distance(robot_pos)
    expected_signed = 2.0 - (cfg.robot_radius + 0.50)
    assert abs(signed_static.mean().item() - expected_signed) < 1e-3
    assert signed_dyn.min().item() > 1e5

    collision_info = world.check_collision(robot_pos)
    assert collision_info["collision"].float().mean().item() == 0.0

    # Force collision by moving robot inside obstacle influence.
    robot_pos[:, 0] = 1.70
    collision_info = world.check_collision(robot_pos)
    collision_rate = collision_info["collision"].float().mean().item()
    assert collision_rate > 0.99

    print_ok(f"front lidar hit = {front_hit:.6f}")
    print_ok(f"signed distance at origin = {expected_signed:.6f}")
    print_ok(f"forced collision rate = {collision_rate:.4f}")
    print_ok("lidar / risk / collision 正常")


def test_boundary_and_terminations(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 9] boundary / success / collision / fallen / timeout termination 检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    stages = torch.full((num_envs,), 0, dtype=torch.long, device=device)
    world.reset_envs(env_ids, stages=stages)

    robot_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    robot_pos[:, :2] = world.start_pos

    # Success.
    robot_pos[:, :2] = world.target_pos
    terminated, truncated, event_reward, info = world.check_terminations(robot_pos)

    assert terminated.float().mean().item() > 0.99
    assert truncated.float().mean().item() == 0.0
    assert info["success"].float().mean().item() > 0.99
    assert event_reward.mean().item() >= cfg.rew_success - 1e-5

    print_ok("success termination 正常")

    # Collision.
    world.reset_envs(env_ids, stages=torch.full((num_envs,), 1, dtype=torch.long, device=device))
    world.static_obs[:] = 0.0
    world.static_mask[:] = False
    world.static_mask[:, 0] = True
    world.static_obs[:, 0, :2] = world.start_pos
    world.static_obs[:, 0, 2] = 0.60

    robot_pos[:, :2] = world.start_pos
    terminated, truncated, event_reward, info = world.check_terminations(robot_pos)

    assert info["collision"].float().mean().item() > 0.99
    assert terminated.float().mean().item() > 0.99
    assert event_reward.mean().item() <= cfg.rew_collision + 1e-5

    print_ok("collision termination 正常")

    # Fallen.
    world.reset_envs(env_ids, stages=stages)
    robot_pos[:, :2] = world.start_pos
    is_fallen = torch.ones(num_envs, dtype=torch.bool, device=device)
    terminated, truncated, event_reward, info = world.check_terminations(robot_pos, is_fallen=is_fallen)

    assert info["fallen"].float().mean().item() > 0.99
    assert terminated.float().mean().item() > 0.99

    print_ok("fallen termination 正常")

    # Out of bounds.
    world.reset_envs(env_ids, stages=stages)
    robot_pos[:, 0] = cfg.env_size
    robot_pos[:, 1] = 0.0
    boundary = world.boundary_signed_distance(robot_pos)
    terminated, truncated, event_reward, info = world.check_terminations(robot_pos)

    assert boundary.max().item() < 0.0
    assert info["out_of_bounds"].float().mean().item() > 0.99
    assert terminated.float().mean().item() > 0.99

    print_ok("out_of_bounds termination 正常")

    # Timeout.
    world.reset_envs(env_ids, stages=stages)
    robot_pos[:, :2] = world.start_pos
    world.episode_steps[:] = cfg.max_episode_steps
    terminated, truncated, event_reward, info = world.check_terminations(robot_pos)

    assert info["timeout"].float().mean().item() > 0.99
    assert truncated.float().mean().item() > 0.99

    print_ok("timeout truncation 正常")


def test_privileged_features(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int) -> None:
    heading("[测试 10] privileged features layout / dimension 检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    stages = torch.full((num_envs,), 5, dtype=torch.long, device=device)
    world.reset_envs(env_ids, stages=stages)

    robot_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    robot_pos[:, :2] = world.start_pos
    robot_yaw = torch.zeros(num_envs, dtype=torch.float32, device=device)

    feat = world.make_privileged_features(robot_pos, robot_yaw)

    assert feat.shape == (num_envs, 68)
    assert world.privileged_feature_dim() == 68
    assert_finite_tensor("privileged_features", feat)
    assert feat.abs().max().item() <= 10.0001

    cursor = 0
    static_dim = cfg.privileged_static_k * 4
    dynamic_dim = cfg.privileged_dynamic_k * 6

    static_feat = feat[:, cursor:cursor + static_dim]
    cursor += static_dim

    dynamic_feat = feat[:, cursor:cursor + dynamic_dim]
    cursor += dynamic_dim

    risk = feat[:, cursor:cursor + 8]
    cursor += 8

    target_obs = feat[:, cursor:cursor + 3]
    cursor += 3

    stage_oh = feat[:, cursor:cursor + 6]
    cursor += 6

    counts = feat[:, cursor:cursor + 3]
    cursor += 3

    assert cursor == 68

    assert static_feat.shape == (num_envs, 24)
    assert dynamic_feat.shape == (num_envs, 24)
    assert risk.shape == (num_envs, 8)
    assert target_obs.shape == (num_envs, 3)
    assert stage_oh.shape == (num_envs, 6)
    assert counts.shape == (num_envs, 3)

    assert_range("risk dist part", risk[:, :5], 0.0, 1.0)
    assert_range("risk sin/cos", risk[:, 5:7], -1.0, 1.0)
    assert_range("risk collision", risk[:, 7], 0.0, 1.0)
    assert_range("target_obs distance_norm", target_obs[:, 0], 0.0, 2.0)
    assert torch.allclose(stage_oh.sum(dim=-1), torch.ones(num_envs, device=device), atol=1e-5)
    assert_range("counts", counts, 0.0, 1.0)

    print_ok("privileged features shape = [N, 68]")
    print_ok("static 24 + dynamic 24 + risk 8 + target 3 + stage 6 + counts 3 = 68")
    print_ok("privileged feature layout 正常")


def test_random_rollout(cfg: Task3WorldCfg, world: Task3World, device: str, num_envs: int, steps: int) -> None:
    heading("[测试 11] 随机解析 rollout 稳定性检测")

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    stages = torch.randint(0, world.stage_count, (num_envs,), dtype=torch.long, device=device)
    world.reset_envs(env_ids, stages=stages)

    robot_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    robot_pos[:, :2] = world.start_pos
    robot_yaw = torch.rand(num_envs, dtype=torch.float32, device=device) * 2.0 * math.pi - math.pi

    previous_lidar = world.compute_lidar_tensors(robot_pos, robot_yaw, normalize=True)

    records: List[Dict[str, float]] = []
    start_time = time.time()

    for step in range(int(steps)):
        # Random walk with mild target bias to exercise progress and collision interfaces.
        to_goal = world.target_pos - robot_pos[:, :2]
        dist = torch.norm(to_goal, dim=-1, keepdim=True)
        direction = to_goal / torch.clamp(dist, min=1e-6)

        noise = torch.randn((num_envs, 2), dtype=torch.float32, device=device) * 0.35
        velocity = 0.40 * direction + noise

        robot_pos[:, :2] += velocity * cfg.policy_dt
        robot_yaw += torch.randn(num_envs, dtype=torch.float32, device=device) * 0.05
        robot_yaw = torch.atan2(torch.sin(robot_yaw), torch.cos(robot_yaw))

        world.step_kinematics(cfg.policy_dt)

        lidar = world.compute_lidar_tensors(robot_pos, robot_yaw, normalize=True)
        lidar_delta = world.compute_lidar_delta(lidar, previous_lidar)
        risk = world.compute_risk_features(robot_pos, robot_yaw)
        target_obs = world.get_target_obs(robot_pos, robot_yaw)
        progress = world.compute_progress(robot_pos, cfg.policy_dt)
        priv = world.make_privileged_features(robot_pos, robot_yaw)

        terminated, truncated, event_reward, event_info = world.check_terminations(robot_pos)

        assert_finite_tensor("rollout robot_pos", robot_pos)
        assert_finite_tensor("rollout lidar", lidar)
        assert_finite_tensor("rollout lidar_delta", lidar_delta)
        assert_finite_tensor("rollout risk", risk)
        assert_finite_tensor("rollout target_obs", target_obs)
        assert_finite_tensor("rollout progress", progress)
        assert_finite_tensor("rollout privileged", priv)
        assert_finite_tensor("rollout event_reward", event_reward)

        assert lidar.shape == (num_envs, cfg.num_lidar_rays)
        assert lidar_delta.shape == (num_envs, cfg.num_lidar_rays)
        assert risk.shape == (num_envs, 8)
        assert target_obs.shape == (num_envs, 3)
        assert priv.shape == (num_envs, 68)

        previous_lidar = lidar

        if step % max(steps // 5, 1) == 0 or step == steps - 1:
            stats = world.world_stats(robot_pos)
            row = flatten_stats(stats)
            row.update(
                {
                    "rollout/step": float(step),
                    "rollout/lidar_mean": float(lidar.mean().detach().cpu().item()),
                    "rollout/risk_mean": float(risk[:, 7].mean().detach().cpu().item()),
                    "rollout/progress_mean": float(progress.mean().detach().cpu().item()),
                    "rollout/success_rate": float(event_info["success"].float().mean().detach().cpu().item()),
                    "rollout/collision_rate": float(event_info["collision"].float().mean().detach().cpu().item()),
                    "rollout/timeout_rate": float(event_info["timeout"].float().mean().detach().cpu().item()),
                    "rollout/event_reward_mean": float(event_reward.mean().detach().cpu().item()),
                }
            )
            records.append(row)

            print(
                f"step={step + 1:>4}/{steps} | "
                f"stage={row.get('Mean_Stage', 0.0):.2f} | "
                f"dist={row.get('Distance_To_Target_Mean', 0.0):.3f} | "
                f"static={row.get('Mean_Static_Count', 0.0):.2f} | "
                f"dynamic={row.get('Mean_Dynamic_Count', 0.0):.2f} | "
                f"risk={row.get('rollout/risk_mean', 0.0):.3f} | "
                f"collision={row.get('rollout/collision_rate', 0.0):.3f}",
                flush=True,
            )

        # Reset terminated/truncated worlds to keep rollout numerically healthy.
        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            new_stages = torch.randint(0, world.stage_count, (done_ids.numel(),), dtype=torch.long, device=device)
            world.reset_envs(done_ids, stages=new_stages)
            robot_pos[done_ids, :2] = world.start_pos[done_ids]
            robot_pos[done_ids, 2] = 0.0
            robot_yaw[done_ids] = 0.0
            # Task3World stores obstacle tensors as full-batch tensors [num_envs, ...].
            # compute_lidar_tensors is therefore a full-batch API. The refresh path uses
            # robot_pos[done_ids], otherwise ray_dir batch becomes [done_count, ...]
            # while obstacle tensors remain [num_envs, ...], causing torch.bmm batch mismatch.
            refreshed_lidar = world.compute_lidar_tensors(
                robot_pos,
                robot_yaw,
                normalize=True,
            )
            previous_lidar[done_ids] = refreshed_lidar[done_ids]

    elapsed = time.time() - start_time
    throughput = float(num_envs * int(steps)) / max(elapsed, 1e-6)

    print_ok(f"random analytical rollout finished: {steps} steps, {num_envs * int(steps):,} transitions")
    print_ok(f"throughput = {throughput:,.2f} analytical env steps/s")

    print_summary(summarize_records(records))


def main() -> None:
    heading("Go2 Task3 Analytical World 全量白盒测试启动")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    if args.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args.test_device

    cfg = Task3WorldCfg()
    world = Task3World(cfg, num_envs=int(args.num_envs), device=device)

    print_ok(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print_ok(f"device = {device}")
    print_ok(f"num_envs = {args.num_envs}")
    print_ok(f"steps = {args.steps}")

    check_project_files()
    test_config(cfg)
    test_world_init(cfg, world, device, int(args.num_envs))
    test_curriculum(cfg, world)
    test_reset_by_stage(cfg, world, device, int(args.num_envs))
    test_obstacle_safety(cfg, world, device, int(args.num_envs))
    test_dynamic_kinematics(cfg, world, device, int(args.num_envs))
    test_target_navigation(cfg, world, device, int(args.num_envs))
    test_lidar_risk_collision(cfg, world, device, int(args.num_envs))
    test_boundary_and_terminations(cfg, world, device, int(args.num_envs))
    test_privileged_features(cfg, world, device, int(args.num_envs))
    test_random_rollout(cfg, world, device, int(args.num_envs), int(args.steps))

    heading("Go2 Task3 Analytical World 测试全部通过")
    print("重点结论：")
    print("1. Task3World 是纯 torch 解析世界，不依赖 IsaacLab，不创建 obstacle prim。")
    print("2. 课程阶段映射正常：K=0.18 进入 Stage1，K=0.38 进入 Stage2，K=0.96 进入 Stage5。")
    print("3. Stage0 无障碍，Stage1 静态障碍，Stage2+ 动态障碍逐步加入。")
    print("4. 起点/终点采样、障碍物安全区、动态障碍运动、边界反弹均通过。")
    print("5. lidar 输出 [N, 60]，risk features 输出 [N, 8]。")
    print("6. privileged features 输出 [N, 68]，可供后续 Task3 critic 使用。")
    print("7. success / collision / fallen / out_of_bounds / timeout 终止逻辑均通过。")
    print("8. 下一步可以进入 task3_env.py，把解析导航世界接入真实 Go2 IsaacLab 物理环境。")


if __name__ == "__main__":
    main()
