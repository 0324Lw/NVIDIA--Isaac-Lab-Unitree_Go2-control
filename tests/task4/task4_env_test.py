# Unitree Go2 Task4 Sim2Real / RMA environment test.
#
# Usage:
#   cd /home/lw/unitree_go2_isaaclab_rl
#   bash scripts/ubuntu/test_task4_env.sh
#
# Important:
#   task4_env.py imports IsaacLab / pxr dependent modules.
#   Therefore AppLauncher must be launched before importing Go2Task4Env.

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


parser = argparse.ArgumentParser(description="Unitree Go2 Task4 Sim2Real / RMA Env Test")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--rollout-k", type=float, default=0.30)
parser.add_argument("--collect-interval", type=int, default=40)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from go2_rl.tasks.task4.task4_config import Task4Config
from go2_rl.tasks.task4.task4_env import Go2Task4Env


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
    print(" " * 54 + "Unitree Go2 Task4 Environment Statistics")
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
    heading("[测试 0] Task4 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task4_sim2real_rma.yaml",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task4" / "task4_config.py",
        PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task4" / "task4_env.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "Missing required Task4 files:\n" + "\n".join(missing)

    for p in required:
        print_ok(str(p.relative_to(PROJECT_ROOT)))

    print_ok("Task4 工程文件结构正常")


def check_teacher_obs_shape_and_values(env: Go2Task4Env, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.teacher_obs_dim)
    assert torch.is_tensor(obs), f"obs must be torch.Tensor, got {type(obs)}"
    assert tuple(obs.shape) == expected, f"teacher obs shape wrong: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("teacher_obs", obs)
    assert obs.abs().max().item() <= 10.0001, f"teacher obs out of clamp range: {obs.abs().max().item():.6f}"


def check_actor_obs_shape_and_values(env: Go2Task4Env, actor: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.actor_obs_dim)
    assert torch.is_tensor(actor), f"actor obs must be torch.Tensor, got {type(actor)}"
    assert tuple(actor.shape) == expected, f"actor obs shape wrong: {tuple(actor.shape)} != {expected}"
    assert_finite_tensor("actor_obs", actor)
    assert actor.abs().max().item() <= 10.0001, f"actor obs out of clamp range: {actor.abs().max().item():.6f}"


def check_priv_shape_and_values(env: Go2Task4Env, priv: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.privileged_obs_dim)
    assert torch.is_tensor(priv), f"privileged obs must be torch.Tensor, got {type(priv)}"
    assert tuple(priv.shape) == expected, f"privileged obs shape wrong: {tuple(priv.shape)} != {expected}"
    assert_finite_tensor("privileged_obs", priv)
    assert priv.abs().max().item() <= 10.0001, f"privileged obs out of clamp range: {priv.abs().max().item():.6f}"


def check_single_actor_slices(env: Go2Task4Env, single: torch.Tensor) -> None:
    assert tuple(single.shape) == (env.cfg.num_envs, env.single_actor_obs_dim)

    cursor = 0
    slices: Dict[str, torch.Tensor] = {}

    layout = [
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos_error", 12),
        ("joint_vel", 12),
        ("commands", 3),
        ("last_action", 12),
        ("phase_sin", 1),
        ("phase_cos", 1),
        ("base_height", 1),
    ]

    for name, dim in layout:
        slices[name] = single[:, cursor:cursor + dim]
        cursor += dim

    assert cursor == env.single_actor_obs_dim

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    assert torch.all(slices["projected_gravity"].abs() <= 1.25)
    assert torch.all(slices["last_action"].abs() <= 1.0001)
    assert torch.all(slices["phase_sin"].abs() <= 1.0001)
    assert torch.all(slices["phase_cos"].abs() <= 1.0001)

    base_height_mean = slices["base_height"].mean().item()
    assert 0.10 <= base_height_mean <= 0.75, f"base_height mean abnormal: {base_height_mean:.4f}"


def check_actor_history_layout(env: Go2Task4Env, actor: torch.Tensor) -> None:
    stacked = actor.reshape(env.cfg.num_envs, env.cfg.frame_stack, env.single_actor_obs_dim)
    assert tuple(stacked.shape) == (env.cfg.num_envs, env.cfg.frame_stack, env.single_actor_obs_dim)

    for frame in range(env.cfg.frame_stack):
        check_single_actor_slices(env, stacked[:, frame, :])


def check_privileged_slices(env: Go2Task4Env, priv: torch.Tensor) -> None:
    cursor = 0

    base_lin_vel = priv[:, cursor:cursor + 3]
    cursor += 3

    friction = priv[:, cursor:cursor + 1]
    cursor += 1

    payload = priv[:, cursor:cursor + 1]
    cursor += 1

    com_shift = priv[:, cursor:cursor + 3]
    cursor += 3

    motor_strength = priv[:, cursor:cursor + 12]
    cursor += 12

    push_force_body = priv[:, cursor:cursor + 3]
    cursor += 3

    push_active = priv[:, cursor:cursor + 1]
    cursor += 1

    post_push = priv[:, cursor:cursor + 1]
    cursor += 1

    assert cursor == env.privileged_obs_dim

    for name, value in [
        ("base_lin_vel", base_lin_vel),
        ("friction", friction),
        ("payload", payload),
        ("com_shift", com_shift),
        ("motor_strength", motor_strength),
        ("push_force_body", push_force_body),
        ("push_active", push_active),
        ("post_push", post_push),
    ]:
        assert_finite_tensor(name, value)

    assert torch.all(friction > 0.0)
    assert torch.all(payload >= -1e-5)
    assert torch.all(payload <= 1.0 + 1e-5)
    assert torch.all(motor_strength >= 0.0)
    assert torch.all(motor_strength <= 1.0001)
    assert torch.all(push_active >= -1e-5)
    assert torch.all(push_active <= 1.0 + 1e-5)
    assert torch.all(post_push >= -1e-5)
    assert torch.all(post_push <= 1.0 + 1e-5)


def force_root_pose(
    env: Go2Task4Env,
    env_ids: torch.Tensor,
    height: float | None = None,
    quat: torch.Tensor | None = None,
) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    if env_ids.numel() == 0:
        return

    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.env_origins[env_ids]

    if height is None:
        root_state[:, 2] = float(env.cfg.target_height)
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


def check_config_and_curriculum(cfg: Task4Config) -> None:
    heading("[测试 1] Task4Config 基础配置 / 课程阶段检测")

    assert cfg.frame_stack == 5
    assert cfg.single_actor_obs_dim == 48
    assert cfg.actor_obs_dim == 240
    assert cfg.privileged_obs_dim == 25
    assert cfg.teacher_obs_dim == 265
    assert cfg.num_actions == 12
    assert cfg.teacher_mode is True
    assert cfg.max_episode_length == int(cfg.max_episode_length_s / cfg.policy_dt)

    assert len(cfg.stage_thresholds) == 6
    assert len(cfg.cmd_vx_ranges) == 6
    assert len(cfg.cmd_vy_ranges) == 6
    assert len(cfg.cmd_wz_ranges) == 6
    assert len(cfg.friction_ranges) == 6
    assert len(cfg.payload_mass_ranges) == 6
    assert len(cfg.com_shift_ranges) == 6
    assert len(cfg.motor_strength_ranges) == 6
    assert len(cfg.max_degraded_joints_by_stage) == 6
    assert len(cfg.push_interval_ranges_s) == 6
    assert len(cfg.push_magnitude_ranges) == 6
    assert len(cfg.noise_level_by_stage) == 6

    stage_checks = [
        (0.000, 0),
        (0.099, 0),
        (0.100, 1),
        (0.239, 1),
        (0.240, 2),
        (0.419, 2),
        (0.420, 3),
        (0.619, 3),
        (0.620, 4),
        (0.799, 4),
        (0.800, 5),
        (1.000, 5),
    ]

    tmp = object.__new__(Go2Task4Env)
    tmp.cfg = cfg
    tmp.global_steps = 0

    for k, expected in stage_checks:
        stage = Go2Task4Env.stage_from_progress(tmp, k)
        assert stage == expected, f"k={k} stage wrong: got {stage}, expected {expected}"

    print_ok(f"policy_dt = {cfg.policy_dt}")
    print_ok(f"max_episode_length = {cfg.max_episode_length}")
    print_ok("Task4Config 基础配置正常")
    print_ok("课程阶段边界正常")


def check_command_stage_reset(env: Go2Task4Env, cfg: Task4Config) -> None:
    heading("[测试 4] command curriculum / domain randomization 采样检测")

    old_steps = int(env.global_steps)

    rows = []

    for k in [0.0, 0.11, 0.25, 0.45, 0.65, 0.90]:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        obs, _ = env.reset(seed=args_cli.seed)

        check_teacher_obs_shape_and_values(env, obs)

        stage_expected = env.stage_from_progress(k)
        stage_min = int(env.env_stage.min().item())
        stage_max = int(env.env_stage.max().item())
        assert stage_min == stage_expected and stage_max == stage_expected

        cmd = env.commands
        vx_min, vx_max = cfg.cmd_vx_ranges[stage_expected]
        vy_min, vy_max = cfg.cmd_vy_ranges[stage_expected]
        wz_min, wz_max = cfg.cmd_wz_ranges[stage_expected]

        assert cmd[:, 0].min().item() >= vx_min - 1e-5
        assert cmd[:, 0].max().item() <= vx_max + 1e-5
        assert cmd[:, 1].min().item() >= vy_min - 1e-5
        assert cmd[:, 1].max().item() <= vy_max + 1e-5
        assert cmd[:, 2].min().item() >= wz_min - 1e-5
        assert cmd[:, 2].max().item() <= wz_max + 1e-5

        fr_min, fr_max = cfg.friction_ranges[stage_expected]
        assert env.dr_friction.min().item() >= fr_min - 1e-5
        assert env.dr_friction.max().item() <= fr_max + 1e-5

        payload_min, payload_max = cfg.payload_mass_ranges[stage_expected]
        assert env.dr_payload_mass.min().item() >= payload_min - 1e-5
        assert env.dr_payload_mass.max().item() <= payload_max + 1e-5

        motor_min, motor_max = cfg.motor_strength_ranges[stage_expected]
        assert env.dr_motor_strength.min().item() >= min(motor_min, 0.999) - 1e-5
        assert env.dr_motor_strength.max().item() <= motor_max + 1e-5

        if stage_expected <= 1:
            assert torch.allclose(env.dr_motor_strength, torch.ones_like(env.dr_motor_strength), atol=1e-6)

        if stage_expected == 0:
            assert env.dr_payload_mass.max().item() == 0.0
            assert env.push_active.float().mean().item() == 0.0
            assert env.next_push_time.min().item() > 1e5

        rows.append(
            {
                "k": k,
                "stage": stage_expected,
                "cmd_vx": float(cmd[:, 0].mean().item()),
                "cmd_vy": float(cmd[:, 1].mean().item()),
                "cmd_wz": float(cmd[:, 2].mean().item()),
                "friction": float(env.dr_friction.mean().item()),
                "payload": float(env.dr_payload_mass.mean().item()),
                "motor_min": float(env.dr_motor_strength.min().item()),
            }
        )

    env.global_steps = old_steps

    for row in rows:
        print_ok(
            f"k={row['k']:.2f} | stage={row['stage']} | "
            f"cmd=({row['cmd_vx']:.3f},{row['cmd_vy']:.3f},{row['cmd_wz']:.3f}) | "
            f"friction={row['friction']:.3f} | payload={row['payload']:.3f} | "
            f"motor_min={row['motor_min']:.3f}"
        )

    print_ok("command curriculum / domain randomization 采样正常")


def check_push_disturbance(env: Go2Task4Env, cfg: Task4Config) -> None:
    heading("[测试 7] external push / payload disturbance 检测")

    zero_action = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    env.global_steps = int(0.90 * cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)
    check_teacher_obs_shape_and_values(env, obs)

    env.next_push_time[:] = 0.0
    env.push_active[:] = False
    env.push_time_left[:] = 0.0
    env.post_push_timer[:] = 0.0
    env.push_force_w[:] = 0.0
    env.push_force_b[:] = 0.0

    obs, reward, terminated, truncated, info = env.step(zero_action)

    push_rate = env.push_active.float().mean().item()
    push_force_norm = torch.norm(env.push_force_b, dim=-1).mean().item()

    assert push_rate > 0.90, f"push did not start in most envs: push_rate={push_rate:.4f}"
    assert push_force_norm > 0.01, f"push_force_body too small: {push_force_norm:.6f}"

    priv = env.compute_privileged_obs()
    check_priv_shape_and_values(env, priv)
    check_privileged_slices(env, priv)

    for _ in range(max(int(math.ceil(cfg.push_duration_s / cfg.policy_dt)) + 2, 8)):
        obs, reward, terminated, truncated, info = env.step(zero_action)

    post_push_mean = env.post_push_timer.mean().item()
    assert post_push_mean >= 0.0
    assert env.push_time_left.min().item() >= -1e-4

    print_ok(f"push_active_rate after trigger = {push_rate:.4f}")
    print_ok(f"push_force_body_norm mean = {push_force_norm:.4f}")
    print_ok(f"post_push_timer mean after several steps = {post_push_mean:.4f}")
    print_ok("external push / payload disturbance 正常")


def check_forced_events(env: Go2Task4Env, cfg: Task4Config) -> None:
    heading("[测试 8] 强制 fall / tilt / high jump / timeout 事件检测")

    zero_action = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    # Low height fall.
    env.global_steps = 0
    env.reset(seed=args_cli.seed)

    low_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    force_root_pose(env, low_ids, height=cfg.fall_height * 0.5)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    low_hit = int(terminated[low_ids].sum().item())
    assert low_hit > 0, "forced low height did not trigger fall termination"
    print_ok(f"低高度 fall 事件触发正常: {low_hit}/{len(low_ids)}")

    # Tilt fall.
    env.reset(seed=args_cli.seed)
    tilt_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    bad_quat = quat_from_roll_pitch_yaw(roll=1.35, pitch=0.0, yaw=0.0, device=env.device)
    force_root_pose(env, tilt_ids, height=cfg.target_height, quat=bad_quat)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    tilt_hit = int(terminated[tilt_ids].sum().item())
    assert tilt_hit > 0, "forced tilt did not trigger fall termination"
    print_ok(f"倾斜 fall 事件触发正常: {tilt_hit}/{len(tilt_ids)}")

    # High jump / abnormal height.
    env.reset(seed=args_cli.seed)
    high_ids = torch.arange(min(16, cfg.num_envs), dtype=torch.long, device=env.device)
    force_root_pose(env, high_ids, height=cfg.jump_height + 0.20)

    obs, reward, terminated, truncated, info = env.step(zero_action)
    high_hit = int(terminated[high_ids].sum().item())
    assert high_hit > 0, "forced high height did not trigger termination"
    print_ok(f"高高度 jump/fall 事件触发正常: {high_hit}/{len(high_ids)}")

    # Timeout.
    env.reset(seed=args_cli.seed)
    env.episode_steps[:] = cfg.max_episode_length - 1

    obs, reward, terminated, truncated, info = env.step(zero_action)
    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "max_episode_length did not trigger truncated"

    flat = flatten_info(info)
    assert "events/Timeout_Rate" in flat
    assert "events/Success_Rate" in flat

    print_ok(f"timeout 截断触发正常: truncated={timeout_count}")


def run_tests() -> None:
    heading("Go2 Task4 Sim2Real / RMA Env 全量压测启动")

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

    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.teacher_mode = True
    cfg.print_debug_info = bool(args_cli.print_names)

    check_config_and_curriculum(cfg)

    env: Go2Task4Env | None = None

    try:
        heading("[测试 2] Go2Task4Env 初始化 / 名称映射 / 空间维度检测")
        env = Go2Task4Env(cfg)

        print_ok(f"device = {device}")
        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"num_actions = {cfg.num_actions}")
        print_ok(f"single_actor_obs_dim = {env.single_actor_obs_dim}")
        print_ok(f"actor_obs_dim = {env.actor_obs_dim}")
        print_ok(f"privileged_obs_dim = {env.privileged_obs_dim}")
        print_ok(f"teacher_obs_dim = {env.teacher_obs_dim}")
        print_ok(f"returned_obs_dim = {env.num_observations}")
        print_ok(f"teacher_mode = {env.cfg.teacher_mode}")
        print_ok(f"action_joint_ids = {env.action_joint_ids}")
        print_ok(f"foot_body_ids = {env.foot_body_ids}")
        print_ok(f"contact_foot_ids = {env.contact_foot_ids}")

        assert env.robot.num_joints >= 12
        assert len(env.action_joint_ids) == 12
        assert len(env.foot_body_ids) == 4
        assert len(env.contact_foot_ids) == 4
        assert env.observation_space.shape == (cfg.teacher_obs_dim,)
        assert env.state_space.shape == (cfg.privileged_obs_dim,)
        assert env.action_space.shape == (cfg.num_actions,)

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

        heading("[测试 3] reset / teacher obs / actor obs / privileged obs 维度与数值")
        obs, info = env.reset(seed=args_cli.seed)

        actor = env.compute_actor_obs()
        priv = env.compute_privileged_obs()
        teacher = env.compute_teacher_obs()

        check_teacher_obs_shape_and_values(env, obs)
        check_actor_obs_shape_and_values(env, actor)
        check_actor_history_layout(env, actor)
        check_priv_shape_and_values(env, priv)
        check_privileged_slices(env, priv)
        check_teacher_obs_shape_and_values(env, teacher)

        assert torch.allclose(obs, teacher, atol=1e-6)

        print_ok(f"reset teacher obs shape = {tuple(obs.shape)}")
        print_ok(f"actor obs shape = {tuple(actor.shape)}")
        print_ok(f"privileged obs shape = {tuple(priv.shape)}")
        print_ok(f"teacher obs range = {obs.min().item():.4f} ~ {obs.max().item():.4f}")

        # Student return path check without constructing another SimulationContext.
        old_teacher_mode = env.cfg.teacher_mode
        env.cfg.teacher_mode = False
        student_obs = env._get_return_obs()
        env.cfg.teacher_mode = old_teacher_mode

        check_actor_obs_shape_and_values(env, student_obs)
        print_ok("student_mode 返回 actor_history 路径正常")

        check_command_stage_reset(env, cfg)

        heading("[测试 5] 随机动作控制链路")
        env.global_steps = int(0.24 * cfg.curriculum_total_steps)
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
        check_teacher_obs_shape_and_values(env, obs)

        actor = env.compute_actor_obs()
        priv = env.compute_privileged_obs()
        check_actor_obs_shape_and_values(env, actor)
        check_actor_history_layout(env, actor)
        check_priv_shape_and_values(env, priv)
        check_privileged_slices(env, priv)

        flat = flatten_info(latest_info)
        required_info_keys = [
            "reward_components/Total",
            "reward_components/R_Cmd_Lin",
            "reward_components/R_Upright",
            "reward_components/P_Foot_Slip",
            "events/Success_Rate",
            "events/Fall_Rate",
            "events/Timeout_Rate",
            "telemetry/Command_Stage",
            "telemetry/Cmd_Vx",
            "telemetry/Actual_Vx",
            "telemetry/Tracking_Error",
            "telemetry/Base_Height",
            "telemetry/Friction",
            "telemetry/Payload_Mass",
            "telemetry/Motor_Strength_Mean",
            "debug/Returned_Obs_Dim",
            "debug/Actor_Obs_Dim",
            "debug/Privileged_Obs_Dim",
            "debug/Teacher_Obs_Dim",
        ]
        for key in required_info_keys:
            assert key in flat, f"info missing field: {key}"

        print_ok(f"控制链路正常，action joints 平均位移范数 = {q_delta:.6f}")
        print_ok("info 字段结构正常")

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

        check_push_disturbance(env, cfg)
        check_forced_events(env, cfg)

        heading("[测试 9] 随机策略长跑稳定性检测")
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
                check_teacher_obs_shape_and_values(env, obs)
                assert_finite_tensor("rollout_reward", reward)

                actor = env.compute_actor_obs()
                priv = env.compute_privileged_obs()
                check_actor_obs_shape_and_values(env, actor)
                check_actor_history_layout(env, actor)
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
                    f"cmd_vx={row.get('telemetry/Cmd_Vx', 0.0):>6.3f} | "
                    f"vx={row.get('telemetry/Actual_Vx', 0.0):>7.3f} | "
                    f"err={row.get('telemetry/Tracking_Error', 0.0):>7.3f} | "
                    f"fall={row.get('events/Fall_Rate', 0.0):>6.3f} | "
                    f"timeout={row.get('events/Timeout_Rate', 0.0):>6.3f} | "
                    f"push={row.get('telemetry/Push_Active_Rate', 0.0):>6.3f} | "
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

        heading("[测试 10] 奖励组件 / 事件 / 遥测统计报告")
        print_summary_table(summarize_records(records))

        print("Go2 Task4 training pre-check guide:")
        print("1. teacher obs 必须为 265 = actor_history 240 + privileged 25。")
        print("2. actor single obs 必须为 48，frame_stack=5 后 actor_history=240。")
        print("3. privileged obs 应包含 friction、payload、COM shift、motor strength、push force。")
        print("4. 随机策略下 fall 可以出现，但不能出现 NaN/Inf。")
        print("5. Stage0 没有 push/payload/motor degradation，后续 stage 逐步加入扰动。")
        print("6. 训练时重点看 Cmd_Vx/Actual_Vx、Tracking_Error、Fall_Rate、Push_Active_Rate、Motor_Strength_Min。")

        heading("Go2 Task4 Sim2Real / RMA Env 测试全部通过")

    except Exception as exc:
        print("\n[FAIL] Go2 Task4 环境测试失败：")
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
