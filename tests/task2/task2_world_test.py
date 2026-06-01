# Unitree Go2 Task2 world / terrain / curriculum test.
#
# Usage:
#   cd /home/lw/unitree_go2_isaaclab_rl
#   python tests/task2/task2_world_test.py --num-envs 1000 --test-device cuda:0 --headless
#
# Important:
#   task2_world.py imports isaaclab.terrains.
#   Therefore AppLauncher must be launched before importing task2_world.py.

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Go2 Task2 World / Terrain / Curriculum White-Box Test")
parser.add_argument("--num-envs", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--print-detail", action="store_true")
parser.add_argument("--scene-test", action="store_true", help="Optional TerrainImporter scene creation test")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from go2_rl.tasks.task2.task2_config import Task2TerrainCfg
from go2_rl.tasks.task2.task2_world import Task2World, TerrainCurriculum


def print_ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def heading(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} must be torch.Tensor, got {type(x)}"
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


def check_close(name: str, value: float, target: float, tol: float = 1e-5) -> None:
    diff = abs(float(value) - float(target))
    assert diff <= tol, f"{name} mismatch: value={value}, target={target}, diff={diff}"


def print_rows(rows: List[Dict], title: str = "") -> None:
    if title:
        print("\n" + title)
    if not rows:
        print("<empty>")
        return

    keys = list(rows[0].keys())
    widths = {k: max(len(str(k)), max(len(str(row.get(k, ""))) for row in rows)) for k in keys}
    line = " | ".join(f"{k:<{widths[k]}}" for k in keys)
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(f"{str(row.get(k, '')):<{widths[k]}}" for k in keys))


def test_config_files_exist() -> None:
    heading("[测试 0] 工程配置文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task1_flat_locomotion.yaml",
        PROJECT_ROOT / "configs" / "task2_multiterrain.yaml",
        PROJECT_ROOT / "configs" / "platform_ubuntu_laptop.yaml",
        PROJECT_ROOT / "configs" / "platform_windows_3090.yaml",
        PROJECT_ROOT / "configs" / "local_paths.example.yaml",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task2" / "task2_config.py",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task2" / "task2_world.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "Missing required files:\n" + "\n".join(missing)

    for p in required:
        print_ok(str(p.relative_to(PROJECT_ROOT)))

    print_ok("configs 与 Task2 world 文件结构正常")


def test_world_config_and_generator(cfg: Task2TerrainCfg, world: Task2World) -> None:
    heading("[测试 1] Terrain config / TerrainGeneratorCfg 基础结构检测")

    assert cfg.num_terrain_types == 4
    assert cfg.num_levels == 10
    assert cfg.height_scan_dim == 81
    assert cfg.terrain_priv_dim == 91

    assert world.generator_cfg.num_rows == cfg.num_levels
    assert world.generator_cfg.num_cols == cfg.num_terrain_types

    expected_names = ["rough_flat", "slopes", "stepping_stones", "stairs"]
    actual_names = list(world.generator_cfg.sub_terrains.keys())

    for name in expected_names:
        assert name in actual_names, f"TerrainGeneratorCfg missing terrain type: {name}"

    print_ok(f"terrain types = {actual_names}")
    print_ok(f"generator rows x cols = {world.generator_cfg.num_rows} x {world.generator_cfg.num_cols}")
    print_ok(f"terrain patch size = {cfg.terrain_length}m x {cfg.terrain_width}m")
    print_ok(f"platform_width = {cfg.platform_width}")
    print_ok(f"height_scan_dim = {cfg.height_scan_dim}")
    print_ok(f"terrain_priv_dim = {cfg.terrain_priv_dim}")
    print_ok("TerrainGeneratorCfg 基础结构正常")


def test_index_mapping(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[测试 2] logical index -> generator flat index 映射检测")

    terrain_types = torch.tensor([0, 1, 2, 3, 0, 3], dtype=torch.long, device=device)
    terrain_levels = torch.tensor([0, 0, 0, 0, 5, 9], dtype=torch.long, device=device)

    flat = world.get_generator_flat_indices(terrain_types, terrain_levels)
    expected = terrain_levels * cfg.num_terrain_types + terrain_types

    assert torch.equal(flat, expected), f"flat index mapping wrong: got={flat}, expected={expected}"

    rows = []
    for i in range(flat.numel()):
        rows.append(
            {
                "terrain_type": int(terrain_types[i].item()),
                "terrain_level": int(terrain_levels[i].item()),
                "flat_index": int(flat[i].item()),
                "expected": int(expected[i].item()),
            }
        )
    print_rows(rows)
    print_ok("logical -> flat index 映射正常")


def test_origin_mapping(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[测试 3] logical origins / spawn origins 检测")

    origins = world.logical_origins
    assert origins.shape == (cfg.num_terrain_types, cfg.num_levels, 3)
    assert_finite_tensor("logical_origins", origins)

    dx = origins[0, 1, 0] - origins[0, 0, 0]
    dy = origins[1, 0, 1] - origins[0, 0, 1]
    check_close("level x spacing", dx.item(), cfg.terrain_length)
    check_close("type y spacing", dy.item(), cfg.terrain_width)

    n = 512
    terrain_types = torch.randint(0, cfg.num_terrain_types, (n,), device=device)
    terrain_levels = torch.randint(0, cfg.num_levels, (n,), device=device)

    base_origins = world.get_origins_from_indices(terrain_types, terrain_levels, prefer_scene_origins=False)
    spawn_origins = world.sample_spawn_origins(
        terrain_types,
        terrain_levels,
        randomize_xy=True,
        prefer_scene_origins=False,
    )

    assert base_origins.shape == (n, 3)
    assert spawn_origins.shape == (n, 3)
    assert_finite_tensor("base_origins", base_origins)
    assert_finite_tensor("spawn_origins", spawn_origins)

    xy_delta = torch.abs(spawn_origins[:, :2] - base_origins[:, :2])
    assert xy_delta.max().item() <= cfg.spawn_radius + 1e-5

    z_delta = spawn_origins[:, 2] - base_origins[:, 2]
    assert torch.allclose(z_delta, torch.full_like(z_delta, cfg.spawn_height_offset), atol=1e-5)

    print_ok(f"logical origins shape = {tuple(origins.shape)}")
    print_ok(f"spawn origins shape = {tuple(spawn_origins.shape)}")
    print_ok(f"max |spawn_xy - origin_xy| = {xy_delta.max().item():.6f}")
    print_ok(f"spawn z offset = {z_delta.mean().item():.6f}")
    print_ok("origin mapping / spawn origin 正常")


def test_scene_origin_override(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[测试 4] scene env_origins override 检测")

    total = cfg.num_terrain_types * cfg.num_levels
    fake_scene_origins = torch.zeros((total, 3), dtype=torch.float32, device=device)

    for level in range(cfg.num_levels):
        for terrain_type in range(cfg.num_terrain_types):
            flat = level * cfg.num_terrain_types + terrain_type
            fake_scene_origins[flat, 0] = 100.0 + level
            fake_scene_origins[flat, 1] = 200.0 + terrain_type
            fake_scene_origins[flat, 2] = 0.5 * level

    world.set_scene_env_origins(fake_scene_origins)

    terrain_types = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    terrain_levels = torch.tensor([0, 3, 6, 9], dtype=torch.long, device=device)
    origins = world.get_origins_from_indices(terrain_types, terrain_levels, prefer_scene_origins=True)

    flat = world.get_generator_flat_indices(terrain_types, terrain_levels)
    expected = fake_scene_origins[flat]

    assert torch.allclose(origins, expected)
    assert_finite_tensor("scene_override_origins", origins)

    # Reset to analytical mode for the rest of tests.
    world.scene_env_origins = None

    print_ok("set_scene_env_origins / prefer_scene_origins 正常")


def test_level_parameters(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[测试 5] terrain level difficulty 参数递增检测")

    terrain_types = torch.arange(cfg.num_terrain_types, device=device).repeat_interleave(cfg.num_levels)
    terrain_levels = torch.arange(cfg.num_levels, device=device).repeat(cfg.num_terrain_types)

    params = world.get_level_parameters(terrain_types, terrain_levels)

    required_keys = [
        "terrain_type",
        "terrain_level",
        "difficulty",
        "rough_amp",
        "slope",
        "stone_height",
        "stone_grid_width",
        "stair_height",
        "stair_width",
    ]

    for key in required_keys:
        assert key in params, f"get_level_parameters missing key: {key}"
        assert_finite_tensor(key, params[key])

    levels = torch.arange(cfg.num_levels, device=device)
    types0 = torch.zeros(cfg.num_levels, dtype=torch.long, device=device)
    p0 = world.get_level_parameters(types0, levels)

    assert torch.all(p0["difficulty"][1:] >= p0["difficulty"][:-1])
    assert p0["rough_amp"][-1] >= p0["rough_amp"][0]
    assert p0["slope"][-1] >= p0["slope"][0]
    assert p0["stone_height"][-1] >= p0["stone_height"][0]
    assert p0["stair_height"][-1] >= p0["stair_height"][0]

    rows = []
    for i in range(cfg.num_levels):
        rows.append(
            {
                "level": i,
                "difficulty": f"{p0['difficulty'][i].item():.4f}",
                "rough_amp": f"{p0['rough_amp'][i].item():.4f}",
                "slope": f"{p0['slope'][i].item():.4f}",
                "stone_height": f"{p0['stone_height'][i].item():.4f}",
                "stone_grid_width": f"{p0['stone_grid_width'][i].item():.4f}",
                "stair_height": f"{p0['stair_height'][i].item():.4f}",
                "stair_width": f"{p0['stair_width'][i].item():.4f}",
            }
        )
    print_rows(rows)
    print_ok("terrain difficulty 参数递增正常")


def test_material_sampling(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[测试 6] 多材质参数采样检测")

    n = 4096
    terrain_types = torch.randint(0, cfg.num_terrain_types, (n,), device=device)
    terrain_levels = torch.randint(0, cfg.num_levels, (n,), device=device)

    mats = world.sample_material_params(terrain_types, terrain_levels)

    required_keys = ["friction", "restitution", "material_id", "material_onehot"]
    for key in required_keys:
        assert key in mats, f"sample_material_params missing key: {key}"

    friction = mats["friction"]
    restitution = mats["restitution"]
    material_id = mats["material_id"]
    material_onehot = mats["material_onehot"]

    assert friction.shape == (n,)
    assert restitution.shape == (n,)
    assert material_id.shape == (n,)
    assert material_onehot.shape == (n, cfg.material_count)

    assert_finite_tensor("friction", friction)
    assert_finite_tensor("restitution", restitution)
    assert_finite_tensor("material_onehot", material_onehot)

    assert friction.min().item() >= cfg.friction_range[0] - 1e-5
    assert friction.max().item() <= cfg.friction_range[1] + 1e-5
    assert restitution.min().item() >= cfg.restitution_range[0] - 1e-5
    assert restitution.max().item() <= cfg.restitution_range[1] + 1e-5
    assert torch.allclose(material_onehot.sum(dim=-1), torch.ones(n, device=device), atol=1e-5)

    counts = torch.bincount(material_id, minlength=cfg.material_count).detach().cpu().numpy()
    ratios = counts / max(counts.sum(), 1)

    print_ok(f"friction range = {friction.min().item():.4f} ~ {friction.max().item():.4f}")
    print_ok(f"restitution range = {restitution.min().item():.4f} ~ {restitution.max().item():.4f}")
    print_ok(f"material counts = {counts.tolist()}")
    print_ok(f"material ratios = {[round(float(x), 4) for x in ratios]}")
    print_ok("多材质采样正常")


def test_height_scan(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[测试 7] height scan / privileged terrain features 检测")

    terrain_types = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        dtype=torch.long,
        device=device,
    )
    terrain_levels = torch.tensor(
        [0, 4, 9, 0, 4, 9, 0, 4, 9, 0, 4, 9],
        dtype=torch.long,
        device=device,
    )

    origins = world.sample_spawn_origins(
        terrain_types,
        terrain_levels,
        randomize_xy=False,
        prefer_scene_origins=False,
    )

    base_pos = origins.clone()
    base_pos[:, 2] += 0.10

    base_quat = torch.zeros((len(terrain_types), 4), device=device)
    base_quat[:, 0] = 1.0

    scan_rows = []
    for terrain_type in range(cfg.num_terrain_types):
        ids = (terrain_types == terrain_type).nonzero(as_tuple=False).squeeze(-1)
        hs = world.sample_height_scan(
            base_pos_w=base_pos[ids],
            terrain_types=terrain_types[ids],
            terrain_levels=terrain_levels[ids],
            base_quat_wxyz=base_quat[ids],
            prefer_scene_origins=False,
        )
        assert hs.shape == (len(ids), cfg.height_scan_dim)
        assert_finite_tensor(f"height_scan_type_{terrain_type}", hs)
        assert hs.abs().max().item() <= cfg.height_scan_clip + 1e-5

        for j in range(len(ids)):
            scan_rows.append(
                {
                    "terrain_type": int(terrain_types[ids[j]].item()),
                    "level": int(terrain_levels[ids[j]].item()),
                    "scan_mean": f"{hs[j].mean().item():.5f}",
                    "scan_std": f"{hs[j].std().item():.5f}",
                    "scan_min": f"{hs[j].min().item():.5f}",
                    "scan_max": f"{hs[j].max().item():.5f}",
                }
            )

    height_scan = world.sample_height_scan(
        base_pos_w=base_pos,
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        base_quat_wxyz=base_quat,
        prefer_scene_origins=False,
    )

    assert height_scan.shape == (len(terrain_types), cfg.height_scan_dim)
    assert_finite_tensor("height_scan_all", height_scan)

    mats = world.sample_material_params(terrain_types, terrain_levels)

    priv = world.make_privileged_terrain_features(
        base_pos_w=base_pos,
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        friction=mats["friction"],
        base_quat_wxyz=base_quat,
        prefer_scene_origins=False,
    )

    assert priv.shape == (len(terrain_types), cfg.terrain_priv_dim)
    assert_finite_tensor("privileged terrain features", priv)

    height_scan_part = priv[:, :81]
    friction_part = priv[:, 81]
    terrain_onehot = priv[:, 82:86]
    difficulty = priv[:, 86]
    param4 = priv[:, 87:91]

    assert height_scan_part.shape == (len(terrain_types), 81)
    assert friction_part.shape == (len(terrain_types),)
    assert terrain_onehot.shape == (len(terrain_types), 4)
    assert difficulty.shape == (len(terrain_types),)
    assert param4.shape == (len(terrain_types), 4)

    assert torch.allclose(terrain_onehot.sum(dim=-1), torch.ones(len(terrain_types), device=device), atol=1e-5)
    assert difficulty.min().item() >= -1e-5 and difficulty.max().item() <= 1.0 + 1e-5

    print_rows(scan_rows)
    print_ok(f"height_scan shape = {tuple(height_scan.shape)}")
    print_ok(f"privileged terrain feature shape = {tuple(priv.shape)}")
    print_ok(f"height_scan range = {height_scan.min().item():.5f} ~ {height_scan.max().item():.5f}")
    print_ok("height scan / privileged terrain feature 正常")


def test_height_scan_yaw_rotation(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[测试 8] height scan yaw rotation 检测")

    terrain_types = torch.tensor([1, 1], dtype=torch.long, device=device)
    terrain_levels = torch.tensor([8, 8], dtype=torch.long, device=device)

    origins = world.sample_spawn_origins(
        terrain_types,
        terrain_levels,
        randomize_xy=False,
        prefer_scene_origins=False,
    )
    base_pos = origins.clone()
    base_pos[:, 2] += 0.15

    q0 = torch.zeros((2, 4), dtype=torch.float32, device=device)
    q0[:, 0] = 1.0

    yaw = math.pi / 2.0
    q90 = torch.zeros((2, 4), dtype=torch.float32, device=device)
    q90[:, 0] = math.cos(yaw * 0.5)
    q90[:, 3] = math.sin(yaw * 0.5)

    hs0 = world.sample_height_scan(base_pos, terrain_types, terrain_levels, q0, prefer_scene_origins=False)
    hs90 = world.sample_height_scan(base_pos, terrain_types, terrain_levels, q90, prefer_scene_origins=False)

    assert_finite_tensor("height_scan_yaw_0", hs0)
    assert_finite_tensor("height_scan_yaw_90", hs90)
    assert hs0.shape == hs90.shape == (2, cfg.height_scan_dim)

    # For slope terrain, yaw rotation should usually change sampled height distribution.
    diff = torch.mean(torch.abs(hs0 - hs90)).item()
    assert diff > 1e-4, f"yaw rotation has almost no effect on height scan: diff={diff}"

    print_ok(f"mean |height_scan(yaw0)-height_scan(yaw90)| = {diff:.6f}")
    print_ok("height scan yaw rotation 正常")


def test_curriculum_initialization(cfg: Task2TerrainCfg, device: str, num_envs: int) -> TerrainCurriculum:
    heading("[测试 9] TerrainCurriculum 初始化 / anchor 防遗忘分组检测")

    curriculum = TerrainCurriculum(num_envs=num_envs, world_cfg=cfg, device=device)

    anchor_count = int(curriculum.anchor_mask.sum().item())
    expected_anchor = int(num_envs * cfg.flat_retention_ratio)
    assert anchor_count == expected_anchor, f"anchor count wrong: {anchor_count} != {expected_anchor}"

    if anchor_count > 0:
        assert (curriculum.env_levels[curriculum.anchor_mask] == 0).all()
        assert (curriculum.env_types[curriculum.anchor_mask] == 0).all()

    assert curriculum.env_types.min().item() >= 0
    assert curriculum.env_types.max().item() < cfg.num_terrain_types
    assert curriculum.env_levels.min().item() >= 0
    assert curriculum.env_levels.max().item() < cfg.num_levels

    stats = curriculum.log_curriculum_stats()

    print_ok(f"num_envs = {num_envs}")
    print_ok(f"anchor_count = {anchor_count}")
    print_ok(f"initial active mean level = {stats['Curriculum/Mean_Level_Active']:.4f}")
    print_ok(
        "terrain type ratios = "
        f"{[round(stats[f'Curriculum/Terrain_Type_{i}_Ratio'], 4) for i in range(cfg.num_terrain_types)]}"
    )
    print_ok("TerrainCurriculum 初始化正常")
    return curriculum


def test_curriculum_upgrade_downgrade(cfg: Task2TerrainCfg, curriculum: TerrainCurriculum, device: str) -> None:
    heading("[测试 10] TerrainCurriculum 升级 / 降级 / 满级回流检测")

    num_envs = curriculum.num_envs
    env_ids = torch.arange(num_envs, device=device)

    curriculum.env_levels[:] = 5
    curriculum.env_types[:] = torch.randint(0, cfg.num_terrain_types, (num_envs,), device=device)
    curriculum.env_levels[curriculum.anchor_mask] = 0
    curriculum.env_types[curriculum.anchor_mask] = 0

    curriculum.register_start_positions(env_ids, torch.zeros(num_envs, device=device))

    current_x = torch.full((num_envs,), cfg.success_distance + 1.0, device=device)
    fall_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)

    before_levels = curriculum.env_levels.clone()
    curriculum.update_curriculum(env_ids, current_x, fall_flags)
    after_success = curriculum.env_levels.clone()

    active = ~curriculum.anchor_mask
    assert (after_success[curriculum.anchor_mask] == 0).all()
    assert (after_success[active] >= before_levels[active]).all()

    print_ok(f"成功回合后 active mean level = {after_success[active].float().mean().item():.4f}")

    curriculum.register_start_positions(env_ids, torch.zeros(num_envs, device=device))

    current_x = torch.full((num_envs,), cfg.failure_distance * 0.25, device=device)
    fall_flags = torch.ones(num_envs, dtype=torch.bool, device=device)

    before_fail = curriculum.env_levels.clone()
    curriculum.update_curriculum(env_ids, current_x, fall_flags)
    after_fail = curriculum.env_levels.clone()

    assert (after_fail[curriculum.anchor_mask] == 0).all()
    downgraded = (after_fail[active] < before_fail[active]).float().mean().item()
    assert downgraded > 0.05, f"downgrade ratio too low: {downgraded}"

    print_ok(f"失败回合后 active 降级比例 = {downgraded:.4f}")

    active_ids = active.nonzero(as_tuple=False).squeeze(-1)
    if active_ids.numel() > 0:
        curriculum.env_levels[active_ids] = cfg.num_levels - 1
        curriculum.register_start_positions(active_ids, torch.zeros(len(active_ids), device=device))

        current_x_active = torch.full((len(active_ids),), cfg.success_distance + 2.0, device=device)
        fall_active = torch.zeros(len(active_ids), dtype=torch.bool, device=device)
        success_active = torch.ones(len(active_ids), dtype=torch.bool, device=device)

        curriculum.update_curriculum(
            active_ids,
            current_x_active,
            fall_active,
            success_flags=success_active,
        )

        levels_after_max = curriculum.env_levels[active_ids]
        assert levels_after_max.min().item() >= cfg.max_level_reset_to_min
        assert levels_after_max.max().item() <= cfg.max_level_reset_to_max

        print_ok(f"满级回流后 level range = {levels_after_max.min().item()} ~ {levels_after_max.max().item()}")

    stats = curriculum.log_curriculum_stats()
    print_ok(f"Upgrade_Total = {stats['Curriculum/Upgrade_Total']:.0f}")
    print_ok(f"Downgrade_Total = {stats['Curriculum/Downgrade_Total']:.0f}")
    print_ok(f"Success_Total = {stats['Curriculum/Success_Total']:.0f}")
    print_ok(f"Fall_Total = {stats['Curriculum/Fall_Total']:.0f}")
    print_ok("课程升级 / 降级 / 满级回流正常")


def test_curriculum_probes(cfg: Task2TerrainCfg, curriculum: TerrainCurriculum, print_detail: bool = False) -> None:
    heading("[测试 11] Curriculum probes / telemetry 统计检测")

    stats = curriculum.log_curriculum_stats()

    required_keys = [
        "Curriculum/Mean_Level_Active",
        "Curriculum/Max_Level_Active",
        "Curriculum/Mean_Terrain_Type_Active",
        "Curriculum/Max_Level_Reached",
        "Curriculum/Anchor_Count",
        "Curriculum/Upgrade_Total",
        "Curriculum/Downgrade_Total",
        "Curriculum/Success_Total",
        "Curriculum/Fall_Total",
    ]

    for key in required_keys:
        assert key in stats, f"log_curriculum_stats missing key: {key}"
        assert math.isfinite(stats[key]), f"{key} is not finite"

    for terrain_type in range(cfg.num_terrain_types):
        key = f"Curriculum/Terrain_Type_{terrain_type}_Ratio"
        assert key in stats

    for level in range(cfg.num_levels):
        key = f"Curriculum/Level_{level}_Ratio"
        assert key in stats

    ratio_sum = sum(stats[f"Curriculum/Level_{i}_Ratio"] for i in range(cfg.num_levels))
    assert abs(ratio_sum - 1.0) < 1e-4, f"level ratio sum != 1: {ratio_sum}"

    print_ok("curriculum telemetry keys 完整")
    print_ok(f"active mean level = {stats['Curriculum/Mean_Level_Active']:.4f}")
    print_ok(f"max level reached = {stats['Curriculum/Max_Level_Reached']:.0f}")

    sample_ids = list(range(min(5, curriculum.num_envs)))
    tail_start = max(0, curriculum.num_envs - 5)
    sample_ids += list(range(tail_start, curriculum.num_envs))
    sample_ids = sorted(set(sample_ids))

    rows = []
    for i in sample_ids:
        rows.append(
            {
                "ID": i,
                "Anchor": bool(curriculum.anchor_mask[i].item()),
                "TerrainType": int(curriculum.env_types[i].item()),
                "Level": int(curriculum.env_levels[i].item()),
                "Upgrades": int(curriculum.probe_upgrades_count[i].item()),
                "Downgrades": int(curriculum.probe_downgrades_count[i].item()),
                "Success": int(curriculum.probe_success_count[i].item()),
                "Falls": int(curriculum.probe_fall_count[i].item()),
                "MaxLevel": int(curriculum.probe_max_level_reached[i].item()),
            }
        )

    print_rows(rows, title="[Probe sample]")

    if print_detail:
        stat_rows = [{"key": k, "value": f"{v:.6f}"} for k, v in sorted(stats.items())]
        print_rows(stat_rows, title="[Full curriculum stats]")


def test_optional_scene_instantiation(cfg: Task2TerrainCfg, world: Task2World, device: str) -> None:
    heading("[可选测试 12] Isaac Terrain scene 实例化检测")

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils import configclass

        @configclass
        class TerrainOnlySceneCfg(InteractiveSceneCfg):
            num_envs: int = 1
            env_spacing: float = 0.0
            terrain: TerrainImporterCfg = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=world.generator_cfg,
                max_init_terrain_level=cfg.num_levels - 1,
                collision_group=-1,
            )

        sim_cfg = sim_utils.SimulationCfg(
            dt=0.005,
            device=device,
            physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
        )

        sim = sim_utils.SimulationContext(sim_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

        scene_cfg = TerrainOnlySceneCfg(num_envs=1, env_spacing=0.0)
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.update(dt=0.0)

        terrain_obj = None
        try:
            terrain_obj = scene["terrain"]
        except Exception:
            terrain_obj = getattr(scene, "terrain", None)

        if terrain_obj is not None and hasattr(terrain_obj, "env_origins"):
            origins = terrain_obj.env_origins
            world.set_scene_env_origins(origins)
            assert origins.reshape(-1, 3).shape[0] >= cfg.num_terrain_types * cfg.num_levels
            assert_finite_tensor("scene.terrain.env_origins", origins)
            print_ok(f"scene terrain env_origins shape = {tuple(origins.shape)}")
            print_ok("world.set_scene_env_origins 正常")
        else:
            print_warn("scene terrain env_origins 未找到；不同 Isaac Lab 版本接口可能不同")

        print_ok("Isaac Terrain scene 可实例化")

    except Exception as exc:
        print_warn(f"可选 scene 实例化测试失败，但不影响 world analytical test: {type(exc).__name__}: {exc}")


def run_tests() -> None:
    heading("Go2 Task2 World / Terrain / Curriculum 全量白盒压测启动")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    cfg = Task2TerrainCfg()
    world = Task2World(cfg, device=device)

    print_ok(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print_ok(f"device = {device}")
    print_ok(f"num_envs for curriculum test = {args_cli.num_envs}")

    test_config_files_exist()
    test_world_config_and_generator(cfg, world)
    test_index_mapping(cfg, world, device)
    test_origin_mapping(cfg, world, device)
    test_scene_origin_override(cfg, world, device)
    test_level_parameters(cfg, world, device)
    test_material_sampling(cfg, world, device)
    test_height_scan(cfg, world, device)
    test_height_scan_yaw_rotation(cfg, world, device)

    curriculum = test_curriculum_initialization(cfg, device, int(args_cli.num_envs))
    test_curriculum_upgrade_downgrade(cfg, curriculum, device)
    test_curriculum_probes(cfg, curriculum, print_detail=bool(args_cli.print_detail))

    if bool(args_cli.scene_test):
        test_optional_scene_instantiation(cfg, world, device)
    else:
        print("\n[WARN] 已跳过 Isaac Terrain scene 实例化测试。需要测试真实 scene 时运行：")
        print("   python tests/task2/task2_world_test.py --scene-test --test-device cuda:0 --headless")

    heading("Go2 Task2 World / Terrain / Curriculum 测试全部通过")
    print("重点结论：")
    print("1. configs/task2_multiterrain.yaml 已存在，configs 不再是空目录。")
    print("2. Task2 配置已独立到 task2_config.py。")
    print("3. TerrainGeneratorCfg 结构正常，4 类地形 × 10 难度等级。")
    print("4. logical index 到 Isaac generator flat index 映射正常。")
    print("5. spawn origin 限制在中央 platform 附近。")
    print("6. height scan 输出为 [N, 81]，critic terrain privileged feature 输出为 [N, 91]。")
    print("7. 多材质 friction / restitution / material_onehot 采样正常。")
    print("8. TerrainCurriculum 的 anchor、防遗忘、升级、降级、满级回流逻辑正常。")


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass
