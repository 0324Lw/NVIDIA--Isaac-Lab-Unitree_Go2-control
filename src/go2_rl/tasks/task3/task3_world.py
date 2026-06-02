# Copyright (c) 2026
# Unitree Go2 Task3: 解析式导航避障世界模型。
#
# 本文件定义 Task3 的世界级导航与障碍物逻辑，不创建 Go2 机器人、不计算 PPO、不启动训练流程。
# 主要职责:
#   1. 采样起点、目标点、目标速度和课程阶段；
#   2. 采样静态/动态圆形障碍物；
#   3. 推进动态障碍物并处理边界反射；
#   4. 计算解析 2D lidar、lidar delta 和障碍物风险特征；
#   5. 计算目标观测、距离进度、碰撞、越界、成功和超时事件；
#   6. 构造 world privileged features。
#
# 观测维度:
#   lidar rays = 60
#   world privileged tail = 68
#
# 训练入口位于 task3_train.py，模型评估入口位于 task3_model_test.py。
#
# 工程说明:
#   Task3 world 使用 tensor 化解析几何而不是 USD obstacle prim。
#   这样训练阶段可以在大量并行环境中稳定计算 lidar、碰撞和风险特征。
#   评估脚本可以单独用 marker 可视化目标和障碍物，但训练 world 本身保持无 prim 设计。
#
# Unitree Go2 Task3: analytical navigation and obstacle-avoidance world model.
#
# This file defines Task3 world-level navigation and obstacle logic. It does not
# create the Go2 robot, compute PPO, or launch training.
# Main responsibilities:
#   1. Sample starts, goals, target speeds, and curriculum stages;
#   2. Sample static/dynamic circular obstacles;
#   3. Advance dynamic obstacles and handle boundary reflection;
#   4. Compute analytical 2D lidar, lidar delta, and obstacle-risk features;
#   5. Compute target observations, distance progress, collision, out-of-bounds,
#      success, and timeout events;
#   6. Build world privileged features.
#
# Observation dimensions:
#   lidar rays = 60
#   world privileged tail = 68
#
# Training entry is task3_train.py, and model evaluation entry is task3_model_test.py.
#
# Engineering notes:
#   Task3 world uses tensorized analytical geometry instead of USD obstacle prims.
#   This keeps lidar, collision, and risk-feature computation stable across many
#   parallel training environments. Evaluation scripts may visualize targets and
#   obstacles with markers, while the training world itself remains prim-free.

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from go2_rl.tasks.task3.task3_config import Task3WorldCfg


class Task3World:
    """Analytical navigation world for Task3.

    Main tensors:
        start_pos:          [E, 2]
        target_pos:         [E, 2]
        static_obs:         [E, S, 3] -> x, y, radius
        static_mask:        [E, S]
        dynamic_obs_pos:    [E, D, 2]
        dynamic_obs_vel:    [E, D, 2]
        dynamic_obs_radius: [E, D]
        dynamic_mask:       [E, D]
        env_stage:          [E]
        env_target_speed:   [E]

    This world is intentionally independent of IsaacLab. It can be tested
    without launching AppLauncher, making bottom-layer validation fast.
    """

    def __init__(self, cfg: Task3WorldCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = str(device)

        self.stage_count = len(cfg.stage_thresholds)
        self._validate_cfg()

        self.start_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        self.static_obs = torch.zeros(
            (self.num_envs, int(cfg.max_static_obs), 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.static_mask = torch.zeros(
            (self.num_envs, int(cfg.max_static_obs)),
            dtype=torch.bool,
            device=self.device,
        )

        self.dynamic_obs_pos = torch.zeros(
            (self.num_envs, int(cfg.max_dynamic_obs), 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.dynamic_obs_vel = torch.zeros(
            (self.num_envs, int(cfg.max_dynamic_obs), 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.dynamic_obs_radius = torch.zeros(
            (self.num_envs, int(cfg.max_dynamic_obs)),
            dtype=torch.float32,
            device=self.device,
        )
        self.dynamic_mask = torch.zeros(
            (self.num_envs, int(cfg.max_dynamic_obs)),
            dtype=torch.bool,
            device=self.device,
        )

        self.env_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_target_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_distance_to_target = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.ray_angles = torch.linspace(
            0.0,
            2.0 * math.pi * (int(cfg.num_lidar_rays) - 1) / int(cfg.num_lidar_rays),
            int(cfg.num_lidar_rays),
            dtype=torch.float32,
            device=self.device,
        )

    # -------------------------------------------------------------------------
    # Validation / utilities
    # -------------------------------------------------------------------------
    def _validate_cfg(self) -> None:
        c = self.cfg

        if c.env_size <= 10.0:
            raise ValueError("Task3 env_size should be greater than 10m.")
        if c.max_static_obs < 0 or c.max_dynamic_obs < 0:
            raise ValueError("Obstacle counts must be non-negative.")
        if c.num_lidar_rays <= 8:
            raise ValueError("num_lidar_rays is too small.")
        if c.lidar_max_distance <= 0.0:
            raise ValueError("lidar_max_distance must be positive.")
        if len(c.stage_thresholds) != 6:
            raise ValueError("Task3 expects exactly 6 curriculum stages.")
        if len(c.success_radius_by_stage) != self.stage_count:
            raise ValueError("success_radius_by_stage length must equal stage count.")

        range_fields = [
            c.goal_dist_ranges,
            c.static_count_ranges,
            c.dynamic_count_ranges,
            c.dynamic_speed_ranges,
            c.target_speed_ranges,
        ]
        for ranges in range_fields:
            if len(ranges) != self.stage_count:
                raise ValueError("All stage range tuples must match stage_count.")

        for i in range(1, len(c.stage_thresholds)):
            if not c.stage_thresholds[i] > c.stage_thresholds[i - 1]:
                raise ValueError("stage_thresholds must be strictly increasing.")

        expected_priv = (
            int(c.privileged_static_k) * 4
            + int(c.privileged_dynamic_k) * 6
            + 8
            + 3
            + self.stage_count
            + 3
        )
        if expected_priv != 68:
            raise ValueError(f"Task3 privileged_feature_dim must be 68, got {expected_priv}.")

    def _as_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return env_ids

        return torch.clamp(env_ids, 0, self.num_envs - 1)

    @staticmethod
    def _finite_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return torch.nan_to_num(torch.clamp(x, float(lo), float(hi)), nan=0.0, posinf=float(hi), neginf=float(lo))

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------
    def curriculum_k(self, global_steps: int) -> float:
        return min(
            1.0,
            max(0.0, float(global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0)),
        )

    def stage_from_progress(self, k: float) -> int:
        k = float(max(0.0, min(1.0, k)))
        stage = 0
        for i, th in enumerate(self.cfg.stage_thresholds):
            if k >= float(th):
                stage = i
        return int(min(stage, self.stage_count - 1))

    def stage_from_global_steps(self, global_steps: int) -> int:
        return self.stage_from_progress(self.curriculum_k(global_steps))

    def set_stage(self, env_ids: torch.Tensor, stage: int) -> None:
        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        stage = int(max(0, min(self.stage_count - 1, int(stage))))
        self.env_stage[env_ids] = stage

    def success_radius_tensor(self, stages: Optional[torch.Tensor] = None) -> torch.Tensor:
        if stages is None:
            stages = self.env_stage

        stages = torch.as_tensor(stages, dtype=torch.long, device=self.device)
        stages = torch.clamp(stages, 0, self.stage_count - 1)

        table = torch.tensor(
            list(self.cfg.success_radius_by_stage),
            dtype=torch.float32,
            device=self.device,
        )
        return table[stages]

    def _sample_stage_int_range(self, ranges, stages: torch.Tensor) -> torch.Tensor:
        stages = torch.as_tensor(stages, dtype=torch.long, device=self.device)
        stages = torch.clamp(stages, 0, self.stage_count - 1)

        mins = torch.tensor([r[0] for r in ranges], dtype=torch.float32, device=self.device)[stages]
        maxs = torch.tensor([r[1] for r in ranges], dtype=torch.float32, device=self.device)[stages]

        vals = torch.floor(torch.rand_like(mins) * (maxs - mins + 1.0) + mins).long()
        return torch.clamp(vals, mins.long(), maxs.long())

    def _sample_stage_float_range(self, ranges, stages: torch.Tensor) -> torch.Tensor:
        stages = torch.as_tensor(stages, dtype=torch.long, device=self.device)
        stages = torch.clamp(stages, 0, self.stage_count - 1)

        mins = torch.tensor([r[0] for r in ranges], dtype=torch.float32, device=self.device)[stages]
        maxs = torch.tensor([r[1] for r in ranges], dtype=torch.float32, device=self.device)[stages]
        return mins + torch.rand_like(mins) * (maxs - mins)

    def get_stage_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}

        for s in range(self.stage_count):
            stats[f"Stage_{s}_Ratio"] = float((self.env_stage == s).float().mean().detach().cpu().item())

        stats["Mean_Stage"] = float(self.env_stage.float().mean().detach().cpu().item())
        stats["Mean_Target_Speed"] = float(self.env_target_speed.mean().detach().cpu().item())
        stats["Mean_Static_Count"] = float(self.static_mask.float().sum(dim=-1).mean().detach().cpu().item())
        stats["Mean_Dynamic_Count"] = float(self.dynamic_mask.float().sum(dim=-1).mean().detach().cpu().item())

        return stats

    # -------------------------------------------------------------------------
    # Reset / sampling
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def reset_envs(
        self,
        env_ids: torch.Tensor,
        global_steps: Optional[int] = None,
        stages: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset a batch of analytical navigation worlds."""

        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        n = int(env_ids.numel())

        if stages is not None:
            stages = torch.as_tensor(stages, dtype=torch.long, device=self.device).flatten()
            if stages.numel() == 1:
                stages = stages.repeat(n)
            stages = torch.clamp(stages[:n], 0, self.stage_count - 1)

        elif global_steps is not None:
            stage = self.stage_from_global_steps(int(global_steps))
            stages = torch.full((n,), int(stage), dtype=torch.long, device=self.device)

        else:
            stages = self.env_stage[env_ids].clone()
            stages = torch.clamp(stages, 0, self.stage_count - 1)

        self.env_stage[env_ids] = stages

        self.start_pos[env_ids] = 0.0
        self.target_pos[env_ids] = 0.0

        self.static_obs[env_ids] = 0.0
        self.static_mask[env_ids] = False

        self.dynamic_obs_pos[env_ids] = 0.0
        self.dynamic_obs_vel[env_ids] = 0.0
        self.dynamic_obs_radius[env_ids] = 0.0
        self.dynamic_mask[env_ids] = False

        self.episode_steps[env_ids] = 0

        self._sample_start_and_goal(env_ids, stages)
        self._sample_static_obstacles(env_ids, stages)
        self._sample_dynamic_obstacles(env_ids, stages)

        self.env_target_speed[env_ids] = self._sample_stage_float_range(
            self.cfg.target_speed_ranges,
            stages,
        )

        dist = torch.norm(self.target_pos[env_ids] - self.start_pos[env_ids], dim=-1)
        self.last_distance_to_target[env_ids] = dist

    @torch.no_grad()
    def _sample_start_and_goal(self, env_ids: torch.Tensor, stages: torch.Tensor) -> None:
        n = int(env_ids.numel())
        half = float(self.cfg.env_size) * 0.5
        bound = half - float(self.cfg.wall_margin) - float(self.cfg.safe_zone_radius)

        goal_min = torch.tensor([r[0] for r in self.cfg.goal_dist_ranges], dtype=torch.float32, device=self.device)[stages]
        goal_max = torch.tensor([r[1] for r in self.cfg.goal_dist_ranges], dtype=torch.float32, device=self.device)[stages]

        valid = torch.zeros(n, dtype=torch.bool, device=self.device)

        for _ in range(int(self.cfg.max_rejection_iters)):
            pending = ~valid
            if not pending.any():
                break

            pidx = pending.nonzero(as_tuple=False).squeeze(-1)
            m = int(pidx.numel())

            starts = (torch.rand((m, 2), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * bound

            dist = goal_min[pidx] + torch.rand(m, dtype=torch.float32, device=self.device) * (
                goal_max[pidx] - goal_min[pidx]
            )
            angle = torch.rand(m, dtype=torch.float32, device=self.device) * 2.0 * math.pi
            unit = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

            targets = starts + unit * dist.unsqueeze(-1)

            inside = (torch.abs(targets[:, 0]) < bound) & (torch.abs(targets[:, 1]) < bound)
            good = inside

            if good.any():
                good_idx = pidx[good]
                self.start_pos[env_ids[good_idx]] = starts[good]
                self.target_pos[env_ids[good_idx]] = targets[good]
                valid[good_idx] = True

        if not valid.all():
            pidx = (~valid).nonzero(as_tuple=False).squeeze(-1)
            m = int(pidx.numel())

            angle = torch.rand(m, dtype=torch.float32, device=self.device) * 2.0 * math.pi
            dist = 0.5 * (goal_min[pidx] + goal_max[pidx])

            half_vec = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1) * (0.5 * dist).unsqueeze(-1)

            starts = torch.clamp(-half_vec, -bound, bound)
            targets = torch.clamp(half_vec, -bound, bound)

            self.start_pos[env_ids[pidx]] = starts
            self.target_pos[env_ids[pidx]] = targets

    @torch.no_grad()
    def _sample_static_obstacles(self, env_ids: torch.Tensor, stages: torch.Tensor) -> None:
        counts = self._sample_stage_int_range(self.cfg.static_count_ranges, stages)

        half = float(self.cfg.env_size) * 0.5
        r_min, r_max = self.cfg.static_radius_range

        for i in range(int(self.cfg.max_static_obs)):
            active_local = (counts > i).nonzero(as_tuple=False).squeeze(-1)
            if active_local.numel() == 0:
                continue

            active_envs = env_ids[active_local]
            m = int(active_envs.numel())

            radius = torch.empty(m, dtype=torch.float32, device=self.device).uniform_(float(r_min), float(r_max))
            placed = torch.zeros(m, dtype=torch.bool, device=self.device)

            for _ in range(int(self.cfg.max_rejection_iters)):
                pending = ~placed
                if not pending.any():
                    break

                pidx = pending.nonzero(as_tuple=False).squeeze(-1)
                pm = int(pidx.numel())

                r = radius[pidx]
                bound = half - float(self.cfg.wall_margin) - r

                pos = (torch.rand((pm, 2), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * bound.unsqueeze(-1)
                env_sel = active_envs[pidx]

                dist_start = torch.norm(pos - self.start_pos[env_sel], dim=-1)
                dist_target = torch.norm(pos - self.target_pos[env_sel], dim=-1)

                safe_start = dist_start > (float(self.cfg.safe_zone_radius) + r + float(self.cfg.obstacle_spawn_buffer))
                safe_target = dist_target > (float(self.cfg.safe_zone_radius) + r + float(self.cfg.obstacle_spawn_buffer))

                good = safe_start & safe_target

                if i > 0:
                    prev_pos = self.static_obs[env_sel, :i, :2]
                    prev_r = self.static_obs[env_sel, :i, 2]
                    prev_mask = self.static_mask[env_sel, :i]

                    dist_prev = torch.norm(pos.unsqueeze(1) - prev_pos, dim=-1)
                    threshold = r.unsqueeze(1) + prev_r + float(self.cfg.min_static_spacing)
                    conflict = ((dist_prev < threshold) & prev_mask).any(dim=-1)

                    good = good & (~conflict)

                if good.any():
                    good_local = pidx[good]
                    good_envs = active_envs[good_local]

                    self.static_obs[good_envs, i, :2] = pos[good]
                    self.static_obs[good_envs, i, 2] = radius[good_local]
                    self.static_mask[good_envs, i] = True

                    placed[good_local] = True

    @torch.no_grad()
    def _sample_dynamic_obstacles(self, env_ids: torch.Tensor, stages: torch.Tensor) -> None:
        counts = self._sample_stage_int_range(self.cfg.dynamic_count_ranges, stages)

        half = float(self.cfg.env_size) * 0.5
        r_min, r_max = self.cfg.dynamic_radius_range

        speed_min = torch.tensor(
            [r[0] for r in self.cfg.dynamic_speed_ranges],
            dtype=torch.float32,
            device=self.device,
        )[stages]
        speed_max = torch.tensor(
            [r[1] for r in self.cfg.dynamic_speed_ranges],
            dtype=torch.float32,
            device=self.device,
        )[stages]

        for i in range(int(self.cfg.max_dynamic_obs)):
            active_local = (counts > i).nonzero(as_tuple=False).squeeze(-1)
            if active_local.numel() == 0:
                continue

            active_envs = env_ids[active_local]
            m = int(active_envs.numel())

            radius = torch.empty(m, dtype=torch.float32, device=self.device).uniform_(float(r_min), float(r_max))
            placed = torch.zeros(m, dtype=torch.bool, device=self.device)

            for _ in range(int(self.cfg.max_rejection_iters)):
                pending = ~placed
                if not pending.any():
                    break

                pidx = pending.nonzero(as_tuple=False).squeeze(-1)
                pm = int(pidx.numel())

                r = radius[pidx]
                bound = half - float(self.cfg.wall_margin) - r

                pos = (torch.rand((pm, 2), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * bound.unsqueeze(-1)
                env_sel = active_envs[pidx]

                dist_start = torch.norm(pos - self.start_pos[env_sel], dim=-1)
                dist_target = torch.norm(pos - self.target_pos[env_sel], dim=-1)

                safe_start = dist_start > (float(self.cfg.safe_zone_radius) + r + float(self.cfg.obstacle_spawn_buffer))
                safe_target = dist_target > (float(self.cfg.safe_zone_radius) + r + float(self.cfg.obstacle_spawn_buffer))

                good = safe_start & safe_target

                # Avoid static obstacles.
                stat_pos = self.static_obs[env_sel, :, :2]
                stat_r = self.static_obs[env_sel, :, 2]
                stat_mask = self.static_mask[env_sel]

                dist_stat = torch.norm(pos.unsqueeze(1) - stat_pos, dim=-1)
                th_stat = r.unsqueeze(1) + stat_r + float(self.cfg.min_dynamic_spacing)
                conflict_static = ((dist_stat < th_stat) & stat_mask).any(dim=-1)

                good = good & (~conflict_static)

                # Avoid previous dynamic obstacles.
                if i > 0:
                    prev_pos = self.dynamic_obs_pos[env_sel, :i, :]
                    prev_r = self.dynamic_obs_radius[env_sel, :i]
                    prev_mask = self.dynamic_mask[env_sel, :i]

                    dist_dyn = torch.norm(pos.unsqueeze(1) - prev_pos, dim=-1)
                    th_dyn = r.unsqueeze(1) + prev_r + float(self.cfg.min_dynamic_spacing)
                    conflict_dyn = ((dist_dyn < th_dyn) & prev_mask).any(dim=-1)

                    good = good & (~conflict_dyn)

                if good.any():
                    good_local = pidx[good]
                    good_envs = active_envs[good_local]

                    self.dynamic_obs_pos[good_envs, i] = pos[good]
                    self.dynamic_obs_radius[good_envs, i] = radius[good_local]
                    self.dynamic_mask[good_envs, i] = True

                    placed[good_local] = True

            valid_envs = active_envs[placed]
            if valid_envs.numel() > 0:
                valid_local = active_local[placed]

                angle = torch.rand(int(valid_envs.numel()), dtype=torch.float32, device=self.device) * 2.0 * math.pi
                spd_min = speed_min[valid_local]
                spd_max = speed_max[valid_local]
                speed = spd_min + torch.rand_like(spd_min) * (spd_max - spd_min)

                self.dynamic_obs_vel[valid_envs, i, 0] = speed * torch.cos(angle)
                self.dynamic_obs_vel[valid_envs, i, 1] = speed * torch.sin(angle)

    # -------------------------------------------------------------------------
    # Dynamic obstacle kinematics
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def step_kinematics(self, dt: float) -> None:
        """Advance dynamic obstacles with analytical reflection."""

        if self.num_envs == 0 or int(self.cfg.max_dynamic_obs) == 0:
            return

        mask = self.dynamic_mask
        mask_f = mask.unsqueeze(-1).float()

        self.dynamic_obs_pos = self.dynamic_obs_pos + self.dynamic_obs_vel * float(dt) * mask_f

        half = float(self.cfg.env_size) * 0.5
        r = self.dynamic_obs_radius
        bound = half - float(self.cfg.wall_margin) - r

        x = self.dynamic_obs_pos[:, :, 0]
        y = self.dynamic_obs_pos[:, :, 1]

        out_x_hi = x > bound
        out_x_lo = x < -bound
        out_y_hi = y > bound
        out_y_lo = y < -bound

        out_x = (out_x_hi | out_x_lo) & mask
        out_y = (out_y_hi | out_y_lo) & mask

        self.dynamic_obs_vel[:, :, 0] = torch.where(out_x, -self.dynamic_obs_vel[:, :, 0], self.dynamic_obs_vel[:, :, 0])
        self.dynamic_obs_vel[:, :, 1] = torch.where(out_y, -self.dynamic_obs_vel[:, :, 1], self.dynamic_obs_vel[:, :, 1])

        self.dynamic_obs_pos[:, :, 0] = torch.clamp(x, -bound, bound)
        self.dynamic_obs_pos[:, :, 1] = torch.clamp(y, -bound, bound)

        # Dynamic-static collision reflection.
        if int(self.cfg.max_static_obs) > 0:
            dyn_pos = self.dynamic_obs_pos.unsqueeze(2)       # [E, D, 1, 2]
            stat_pos = self.static_obs[:, :, :2].unsqueeze(1) # [E, 1, S, 2]

            dist_ds = torch.norm(dyn_pos - stat_pos, dim=-1)
            th_ds = (
                self.dynamic_obs_radius.unsqueeze(2)
                + self.static_obs[:, :, 2].unsqueeze(1)
                + float(self.cfg.min_dynamic_spacing)
            )
            valid_ds = self.dynamic_mask.unsqueeze(2) & self.static_mask.unsqueeze(1)

            hit_static = ((dist_ds < th_ds) & valid_ds).any(dim=-1)

            self.dynamic_obs_vel = torch.where(
                hit_static.unsqueeze(-1),
                -self.dynamic_obs_vel,
                self.dynamic_obs_vel,
            )

        # Dynamic-dynamic collision reflection.
        if int(self.cfg.max_dynamic_obs) > 1:
            dyn_a = self.dynamic_obs_pos.unsqueeze(2)
            dyn_b = self.dynamic_obs_pos.unsqueeze(1)

            dist_dd = torch.norm(dyn_a - dyn_b, dim=-1)
            th_dd = (
                self.dynamic_obs_radius.unsqueeze(2)
                + self.dynamic_obs_radius.unsqueeze(1)
                + float(self.cfg.min_dynamic_spacing)
            )

            valid_dd = self.dynamic_mask.unsqueeze(2) & self.dynamic_mask.unsqueeze(1)
            eye = torch.eye(int(self.cfg.max_dynamic_obs), dtype=torch.bool, device=self.device).unsqueeze(0)
            valid_dd = valid_dd & (~eye)

            hit_dyn = ((dist_dd < th_dd) & valid_dd).any(dim=-1)

            self.dynamic_obs_vel = torch.where(
                hit_dyn.unsqueeze(-1),
                -self.dynamic_obs_vel,
                self.dynamic_obs_vel,
            )

        # Final boundary-direction correction.
        #
        # Boundary reflection is a hard world constraint. Earlier in this function
        # we first reflect dynamic obstacles at map boundaries, then process
        # dynamic-static and dynamic-dynamic collision reflection. In rare cases,
        # an obstacle can hit the boundary and another obstacle in the same step,
        # causing a second velocity flip and cancelling the boundary reflection.
        #
        # Re-apply directional constraints here so that:
        #   right wall hit -> vx must point left
        #   left wall hit  -> vx must point right
        #   upper wall hit -> vy must point down
        #   lower wall hit -> vy must point up
        #
        # This keeps the analytical world stable and makes boundary behavior
        # deterministic under dense obstacle scenes.
        boundary_x_hi = out_x_hi & mask
        boundary_x_lo = out_x_lo & mask
        boundary_y_hi = out_y_hi & mask
        boundary_y_lo = out_y_lo & mask

        self.dynamic_obs_vel[:, :, 0] = torch.where(
            boundary_x_hi,
            -torch.abs(self.dynamic_obs_vel[:, :, 0]),
            self.dynamic_obs_vel[:, :, 0],
        )
        self.dynamic_obs_vel[:, :, 0] = torch.where(
            boundary_x_lo,
            torch.abs(self.dynamic_obs_vel[:, :, 0]),
            self.dynamic_obs_vel[:, :, 0],
        )
        self.dynamic_obs_vel[:, :, 1] = torch.where(
            boundary_y_hi,
            -torch.abs(self.dynamic_obs_vel[:, :, 1]),
            self.dynamic_obs_vel[:, :, 1],
        )
        self.dynamic_obs_vel[:, :, 1] = torch.where(
            boundary_y_lo,
            torch.abs(self.dynamic_obs_vel[:, :, 1]),
            self.dynamic_obs_vel[:, :, 1],
        )

        self.episode_steps += 1

    # -------------------------------------------------------------------------
    # Target / navigation observations
    # -------------------------------------------------------------------------
    def get_target_polar_coords(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor) -> torch.Tensor:
        """Return [distance, relative_angle]."""

        robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32, device=self.device)
        robot_yaw = torch.as_tensor(robot_yaw, dtype=torch.float32, device=self.device).view(-1)

        delta = self.target_pos - robot_pos[:, :2]
        distance = torch.norm(delta, dim=-1)

        target_angle = torch.atan2(delta[:, 1], delta[:, 0])
        relative_angle = torch.atan2(
            torch.sin(target_angle - robot_yaw),
            torch.cos(target_angle - robot_yaw),
        )

        return torch.stack([distance, relative_angle], dim=-1)

    def get_target_obs(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor) -> torch.Tensor:
        """Actor target observation: [distance_norm, sin(relative_angle), cos(relative_angle)]."""

        polar = self.get_target_polar_coords(robot_pos, robot_yaw)
        dist = polar[:, 0]
        angle = polar[:, 1]

        dist_norm = torch.clamp(dist / (0.5 * float(self.cfg.env_size)), 0.0, 2.0)

        return torch.stack(
            [
                dist_norm,
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=-1,
        )

    def distance_to_target(self, robot_pos: torch.Tensor) -> torch.Tensor:
        robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32, device=self.device)
        return torch.norm(robot_pos[:, :2] - self.target_pos, dim=-1)

    def compute_progress(self, robot_pos: torch.Tensor, dt: float) -> torch.Tensor:
        """Return progress speed toward the target.

        Positive means the robot is getting closer to the target.
        """

        current = self.distance_to_target(robot_pos)
        progress = (self.last_distance_to_target - current) / max(float(dt), 1e-6)
        self.last_distance_to_target = current.clone()
        return progress

    # -------------------------------------------------------------------------
    # Obstacle aggregate / lidar
    # -------------------------------------------------------------------------
    def _all_obstacles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return centers [E, O, 2], radius [E, O], mask [E, O]."""

        static_centers = self.static_obs[:, :, :2]
        static_radius = self.static_obs[:, :, 2]
        static_mask = self.static_mask

        dynamic_centers = self.dynamic_obs_pos
        dynamic_radius = self.dynamic_obs_radius
        dynamic_mask = self.dynamic_mask

        centers = torch.cat([static_centers, dynamic_centers], dim=1)
        radius = torch.cat([static_radius, dynamic_radius], dim=1)
        mask = torch.cat([static_mask, dynamic_mask], dim=1)

        return centers, radius, mask

    def compute_lidar_tensors(
        self,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        max_distance: Optional[float] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Analytical 2D lidar based on ray-circle intersections.

        Args:
            robot_pos: [E, 3]
            robot_yaw: [E]
            max_distance: optional lidar max distance
            normalize: if true, return distance / max_distance

        Returns:
            lidar_dist: [E, num_lidar_rays]
        """

        robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32, device=self.device)
        robot_yaw = torch.as_tensor(robot_yaw, dtype=torch.float32, device=self.device).view(-1)

        max_dist = float(max_distance if max_distance is not None else self.cfg.lidar_max_distance)

        centers, radius, obs_mask = self._all_obstacles()

        origin = robot_pos[:, :2]
        global_angles = robot_yaw.unsqueeze(1) + self.ray_angles.unsqueeze(0)

        ray_dir = torch.stack(
            [
                torch.cos(global_angles),
                torch.sin(global_angles),
            ],
            dim=-1,
        )  # [E, R, 2]

        # f = O - C
        f = origin.unsqueeze(1) - centers  # [E, O, 2]

        # Quadratic with ray_dir normalized:
        # t^2 + 2*b*t + c = 0
        # b = D dot (O-C)
        # c = ||O-C||^2 - r^2
        b = torch.bmm(ray_dir, f.transpose(1, 2))  # [E, R, O]
        c = (torch.sum(f * f, dim=-1) - radius * radius).unsqueeze(1)  # [E, 1, O]

        delta = b * b - c
        sqrt_delta = torch.sqrt(torch.clamp(delta, min=0.0))

        t = -b - sqrt_delta

        inside = c < 0.0
        t = torch.where(inside, torch.zeros_like(t), t)

        valid = (delta >= 0.0) & (t >= 0.0) & obs_mask.unsqueeze(1)
        t = torch.where(valid, t, torch.full_like(t, max_dist))

        min_dist = torch.min(t, dim=-1)[0]
        min_dist = torch.clamp(min_dist, 0.0, max_dist)

        if float(self.cfg.lidar_noise_std) > 0.0:
            noise = torch.randn_like(min_dist) * float(self.cfg.lidar_noise_std)
            min_dist = torch.clamp(min_dist + noise, 0.0, max_dist)

        if normalize:
            return min_dist / max_dist

        return min_dist

    @staticmethod
    def compute_lidar_delta(current_lidar: torch.Tensor, previous_lidar: torch.Tensor) -> torch.Tensor:
        return torch.clamp(current_lidar - previous_lidar, -1.0, 1.0)

    # -------------------------------------------------------------------------
    # Risk / collision / termination
    # -------------------------------------------------------------------------
    def compute_risk_features(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor) -> torch.Tensor:
        """Return 8-D risk features.

        Layout:
            0 front_min_norm
            1 left_min_norm
            2 right_min_norm
            3 rear_min_norm
            4 global_min_norm
            5 nearest_angle_sin
            6 nearest_angle_cos
            7 collision_risk
        """

        lidar = self.compute_lidar_tensors(robot_pos, robot_yaw, normalize=False)

        max_d = float(self.cfg.lidar_max_distance)
        r = int(self.cfg.num_lidar_rays)

        sector = max(1, r // 12)

        front_ids = torch.cat(
            [
                torch.arange(0, sector, dtype=torch.long, device=self.device),
                torch.arange(r - sector, r, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )
        left_ids = torch.arange(r // 8, 3 * r // 8, dtype=torch.long, device=self.device)
        rear_ids = torch.arange(3 * r // 8, 5 * r // 8, dtype=torch.long, device=self.device)
        right_ids = torch.arange(5 * r // 8, 7 * r // 8, dtype=torch.long, device=self.device)

        front_min = lidar[:, front_ids].min(dim=-1)[0]
        left_min = lidar[:, left_ids].min(dim=-1)[0]
        rear_min = lidar[:, rear_ids].min(dim=-1)[0]
        right_min = lidar[:, right_ids].min(dim=-1)[0]

        global_min, global_idx = lidar.min(dim=-1)

        nearest_angle = self.ray_angles[global_idx]
        nearest_angle = torch.atan2(torch.sin(nearest_angle), torch.cos(nearest_angle))

        collision_risk = torch.clamp(
            (float(self.cfg.robot_radius) + float(self.cfg.warning_margin) - global_min)
            / max(float(self.cfg.warning_margin), 1e-6),
            0.0,
            1.0,
        )

        feat = torch.stack(
            [
                front_min / max_d,
                left_min / max_d,
                right_min / max_d,
                rear_min / max_d,
                global_min / max_d,
                torch.sin(nearest_angle),
                torch.cos(nearest_angle),
                collision_risk,
            ],
            dim=-1,
        )

        return self._finite_clamp(feat, -1.0, 1.0)

    def obstacle_signed_distance(self, robot_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return min signed distance to all, static, and dynamic obstacles.

        Positive means safe distance. Negative means penetration.
        """

        robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32, device=self.device)
        xy = robot_pos[:, :2]

        if int(self.cfg.max_static_obs) > 0:
            d_static = torch.norm(xy.unsqueeze(1) - self.static_obs[:, :, :2], dim=-1)
            signed_static = d_static - (float(self.cfg.robot_radius) + self.static_obs[:, :, 2])
            signed_static = torch.where(
                self.static_mask,
                signed_static,
                torch.full_like(signed_static, 1e6),
            )
            min_static = signed_static.min(dim=-1)[0]
        else:
            min_static = torch.full((self.num_envs,), 1e6, dtype=torch.float32, device=self.device)

        if int(self.cfg.max_dynamic_obs) > 0:
            d_dyn = torch.norm(xy.unsqueeze(1) - self.dynamic_obs_pos, dim=-1)
            signed_dyn = d_dyn - (float(self.cfg.robot_radius) + self.dynamic_obs_radius)
            signed_dyn = torch.where(
                self.dynamic_mask,
                signed_dyn,
                torch.full_like(signed_dyn, 1e6),
            )
            min_dyn = signed_dyn.min(dim=-1)[0]
        else:
            min_dyn = torch.full((self.num_envs,), 1e6, dtype=torch.float32, device=self.device)

        min_signed = torch.minimum(min_static, min_dyn)

        return min_signed, min_static, min_dyn

    def check_collision(self, robot_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        min_signed, min_static, min_dyn = self.obstacle_signed_distance(robot_pos)

        static_collision = min_static < float(self.cfg.collision_margin)
        dynamic_collision = min_dyn < float(self.cfg.collision_margin)
        collision = static_collision | dynamic_collision

        return {
            "collision": collision,
            "static_collision": static_collision,
            "dynamic_collision": dynamic_collision,
            "min_signed_distance": min_signed,
            "min_static_signed_distance": min_static,
            "min_dynamic_signed_distance": min_dyn,
        }

    def boundary_signed_distance(self, robot_pos: torch.Tensor) -> torch.Tensor:
        robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32, device=self.device)

        half = float(self.cfg.env_size) * 0.5
        x_margin = half - torch.abs(robot_pos[:, 0])
        y_margin = half - torch.abs(robot_pos[:, 1])

        return torch.minimum(x_margin, y_margin) - float(self.cfg.robot_radius)

    def check_terminations(
        self,
        robot_pos: torch.Tensor,
        is_fallen: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Check success / collision / fall / out-of-bounds / timeout."""

        robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32, device=self.device)

        if is_fallen is None:
            is_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            is_fallen = torch.as_tensor(is_fallen, dtype=torch.bool, device=self.device).view(-1)

        dist_to_goal = self.distance_to_target(robot_pos)
        success_radius = self.success_radius_tensor()
        success = dist_to_goal < success_radius

        collision_info = self.check_collision(robot_pos)
        collision = collision_info["collision"]

        boundary_dist = self.boundary_signed_distance(robot_pos)
        out_of_bounds = boundary_dist < 0.0

        timeout = self.episode_steps >= int(self.cfg.max_episode_steps)

        terminated = success | collision | is_fallen | out_of_bounds
        truncated = timeout & (~terminated)

        event_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        event_reward = torch.where(success, event_reward + float(self.cfg.rew_success), event_reward)
        event_reward = torch.where(collision | out_of_bounds, event_reward + float(self.cfg.rew_collision), event_reward)
        event_reward = torch.where(is_fallen, event_reward + float(self.cfg.rew_fall), event_reward)
        event_reward = torch.where(truncated, event_reward + float(self.cfg.rew_timeout), event_reward)

        info = {
            "success": success,
            "collision": collision,
            "static_collision": collision_info["static_collision"],
            "dynamic_collision": collision_info["dynamic_collision"],
            "fallen": is_fallen,
            "out_of_bounds": out_of_bounds,
            "timeout": timeout,
            "distance_to_goal": dist_to_goal,
            "success_radius": success_radius,
            "min_obstacle_signed_distance": collision_info["min_signed_distance"],
            "min_static_signed_distance": collision_info["min_static_signed_distance"],
            "min_dynamic_signed_distance": collision_info["min_dynamic_signed_distance"],
            "boundary_signed_distance": boundary_dist,
        }

        return terminated, truncated, event_reward, info

    # -------------------------------------------------------------------------
    # Privileged features
    # -------------------------------------------------------------------------
    def privileged_feature_dim(self) -> int:
        """Return critic privileged feature dimension.

        Layout:
            nearest static:  K_s * 4 = dx_body, dy_body, radius, mask
            nearest dynamic: K_d * 6 = dx_body, dy_body, vx_body, vy_body, radius, mask
            risk features:  8
            target obs:     3
            stage onehot:   6
            counts/speed:   3
            total = 6*4 + 4*6 + 8 + 3 + 6 + 3 = 68
        """

        return (
            int(self.cfg.privileged_static_k) * 4
            + int(self.cfg.privileged_dynamic_k) * 6
            + 8
            + 3
            + self.stage_count
            + 3
        )

    def make_privileged_features(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor) -> torch.Tensor:
        """Build privileged critic features.

        Actor should not directly use this tensor.
        """

        robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32, device=self.device)
        robot_yaw = torch.as_tensor(robot_yaw, dtype=torch.float32, device=self.device).view(-1)

        static_feat = self._nearest_static_features(robot_pos, robot_yaw, int(self.cfg.privileged_static_k))
        dynamic_feat = self._nearest_dynamic_features(robot_pos, robot_yaw, int(self.cfg.privileged_dynamic_k))
        risk = self.compute_risk_features(robot_pos, robot_yaw)
        target_obs = self.get_target_obs(robot_pos, robot_yaw)

        stage_oh = torch.zeros((self.num_envs, self.stage_count), dtype=torch.float32, device=self.device)
        stage_oh[torch.arange(self.num_envs, device=self.device), torch.clamp(self.env_stage, 0, self.stage_count - 1)] = 1.0

        counts = torch.stack(
            [
                self.static_mask.float().sum(dim=-1) / max(float(self.cfg.max_static_obs), 1.0),
                self.dynamic_mask.float().sum(dim=-1) / max(float(self.cfg.max_dynamic_obs), 1.0),
                self.env_target_speed / 2.0,
            ],
            dim=-1,
        )

        feat = torch.cat(
            [
                static_feat,
                dynamic_feat,
                risk,
                target_obs,
                stage_oh,
                counts,
            ],
            dim=-1,
        )

        expected = self.privileged_feature_dim()
        if feat.shape[-1] != expected:
            raise RuntimeError(f"Task3 privileged feature dim mismatch: got {feat.shape[-1]}, expected {expected}")

        return self._finite_clamp(feat, -10.0, 10.0)

    def _rotate_world_to_body(self, vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Rotate world-frame 2D vectors into body frame.

        Args:
            vec_w: [E, K, 2]
            yaw: [E]
        """

        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device).view(-1)

        c = torch.cos(-yaw).unsqueeze(-1)
        s = torch.sin(-yaw).unsqueeze(-1)

        x = vec_w[..., 0]
        y = vec_w[..., 1]

        bx = c * x - s * y
        by = s * x + c * y

        return torch.stack([bx, by], dim=-1)

    def _nearest_static_features(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor, k: int) -> torch.Tensor:
        e = self.num_envs
        k = int(k)

        if k <= 0:
            return torch.zeros((e, 0), dtype=torch.float32, device=self.device)

        if int(self.cfg.max_static_obs) == 0:
            return torch.zeros((e, k * 4), dtype=torch.float32, device=self.device)

        xy = robot_pos[:, :2]

        d = torch.norm(xy.unsqueeze(1) - self.static_obs[:, :, :2], dim=-1)
        d = torch.where(self.static_mask, d, torch.full_like(d, 1e6))

        k_eff = min(k, int(self.cfg.max_static_obs))
        _, idx = torch.topk(d, k=k_eff, dim=-1, largest=False)

        gather_idx_xy = idx.unsqueeze(-1).expand(-1, -1, 2)
        centers = torch.gather(self.static_obs[:, :, :2], dim=1, index=gather_idx_xy)
        radius = torch.gather(self.static_obs[:, :, 2], dim=1, index=idx)
        mask = torch.gather(self.static_mask.float(), dim=1, index=idx)

        vec_w = centers - xy.unsqueeze(1)
        vec_b = self._rotate_world_to_body(vec_w, robot_yaw)

        feat = torch.cat(
            [
                vec_b / max(float(self.cfg.env_size), 1e-6),
                radius.unsqueeze(-1),
                mask.unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(e, -1)

        if k_eff < k:
            pad = torch.zeros((e, (k - k_eff) * 4), dtype=torch.float32, device=self.device)
            feat = torch.cat([feat, pad], dim=-1)

        return self._finite_clamp(feat, -10.0, 10.0)

    def _nearest_dynamic_features(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor, k: int) -> torch.Tensor:
        e = self.num_envs
        k = int(k)

        if k <= 0:
            return torch.zeros((e, 0), dtype=torch.float32, device=self.device)

        if int(self.cfg.max_dynamic_obs) == 0:
            return torch.zeros((e, k * 6), dtype=torch.float32, device=self.device)

        xy = robot_pos[:, :2]

        d = torch.norm(xy.unsqueeze(1) - self.dynamic_obs_pos, dim=-1)
        d = torch.where(self.dynamic_mask, d, torch.full_like(d, 1e6))

        k_eff = min(k, int(self.cfg.max_dynamic_obs))
        _, idx = torch.topk(d, k=k_eff, dim=-1, largest=False)

        gather_idx_xy = idx.unsqueeze(-1).expand(-1, -1, 2)
        centers = torch.gather(self.dynamic_obs_pos, dim=1, index=gather_idx_xy)
        vel = torch.gather(self.dynamic_obs_vel, dim=1, index=gather_idx_xy)
        radius = torch.gather(self.dynamic_obs_radius, dim=1, index=idx)
        mask = torch.gather(self.dynamic_mask.float(), dim=1, index=idx)

        vec_w = centers - xy.unsqueeze(1)
        vec_b = self._rotate_world_to_body(vec_w, robot_yaw)
        vel_b = self._rotate_world_to_body(vel, robot_yaw)

        feat = torch.cat(
            [
                vec_b / max(float(self.cfg.env_size), 1e-6),
                vel_b / 2.0,
                radius.unsqueeze(-1),
                mask.unsqueeze(-1),
            ],
            dim=-1,
        ).reshape(e, -1)

        if k_eff < k:
            pad = torch.zeros((e, (k - k_eff) * 6), dtype=torch.float32, device=self.device)
            feat = torch.cat([feat, pad], dim=-1)

        return self._finite_clamp(feat, -10.0, 10.0)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def world_stats(self, robot_pos: Optional[torch.Tensor] = None) -> Dict[str, float]:
        stats = self.get_stage_stats()

        stats["Distance_To_Target_Mean"] = 0.0
        stats["Distance_To_Target_Min"] = 0.0
        stats["Distance_To_Target_Max"] = 0.0
        stats["Success_Radius_Mean"] = float(self.success_radius_tensor().mean().detach().cpu().item())
        stats["Static_Count_Max"] = float(self.static_mask.float().sum(dim=-1).max().detach().cpu().item())
        stats["Dynamic_Count_Max"] = float(self.dynamic_mask.float().sum(dim=-1).max().detach().cpu().item())

        if robot_pos is not None:
            dist = self.distance_to_target(robot_pos)
            stats["Distance_To_Target_Mean"] = float(dist.mean().detach().cpu().item())
            stats["Distance_To_Target_Min"] = float(dist.min().detach().cpu().item())
            stats["Distance_To_Target_Max"] = float(dist.max().detach().cpu().item())

            collision_info = self.check_collision(robot_pos)
            stats["Collision_Rate"] = float(collision_info["collision"].float().mean().detach().cpu().item())
            stats["Min_Obstacle_Signed_Distance"] = float(
                collision_info["min_signed_distance"].min().detach().cpu().item()
            )

            boundary = self.boundary_signed_distance(robot_pos)
            stats["Boundary_Signed_Distance_Min"] = float(boundary.min().detach().cpu().item())

        return stats
