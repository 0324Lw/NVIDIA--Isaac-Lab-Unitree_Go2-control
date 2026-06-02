# Copyright (c) 2026
# Unitree Go2 Task2: 多地形世界模型。
#
# 本文件定义 Task2 的世界级地形逻辑，不创建 Go2 机器人、不计算奖励、不启动训练流程。
# 主要职责:
#   1. 构建 TerrainGeneratorCfg；
#   2. 管理 terrain type / level 与生成器索引的映射；
#   3. 管理逻辑 origin 与 TerrainImporter env_origins；
#   4. 采样地形材料参数和高度扫描；
#   5. 构造 terrain privileged features；
#   6. 更新多地形课程。
#
# 观测维度:
#   terrain privileged tail = 91
#
# 训练入口位于 task2_train.py，模型评估入口位于 task2_model_test.py。
#
# 工程说明:
#   TerrainImporter 生成的 env_origins 是实际仿真地形块的原点。
#   当 env_origins 可用时，height scan 和 reset origin 使用该值；否则回退到逻辑 origin。
#   这样可以同时兼容 IsaacLab 地形生成器和纯解析地形回退逻辑。
#
# Unitree Go2 Task2: multi-terrain world model.
#
# This file defines Task2 world-level terrain logic. It does not create the Go2 robot,
# compute rewards, or launch training.
# Main responsibilities:
#   1. Build TerrainGeneratorCfg;
#   2. Manage terrain type / level to generator-index mapping;
#   3. Manage logical origins and TerrainImporter env_origins;
#   4. Sample terrain material parameters and height scans;
#   5. Build terrain privileged features;
#   6. Update the multi-terrain curriculum.
#
# Observation dimensions:
#   terrain privileged tail = 91
#
# Training entry is task2_train.py, and model evaluation entry is task2_model_test.py.
#
# Engineering notes:
#   TerrainImporter env_origins are the actual origins of simulated terrain tiles.
#   When env_origins are available, height scans and reset origins use them; otherwise
#   the world falls back to logical origins. This keeps compatibility with both
#   IsaacLab terrain generation and analytical terrain fallback logic.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from go2_rl.tasks.task2.task2_config import Task2TerrainCfg


class Task2World:
    """Task2 analytical world model.

    Responsibilities:
    1. Build Isaac Lab TerrainGeneratorCfg.
    2. Map logical terrain type / level to generator flat index.
    3. Provide fallback analytical terrain origins.
    4. Sample spawn origins near central platform.
    5. Sample terrain material parameters.
    6. Provide GPU tensorized analytical height scan for critic.
    7. Build privileged terrain features.
    """

    terrain_type_names = ("rough_flat", "slopes", "stepping_stones", "stairs")

    def __init__(self, cfg: Task2TerrainCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = str(device)

        self.num_types = int(cfg.num_terrain_types)
        self.num_levels = int(cfg.num_levels)

        if self.num_types != 4:
            raise ValueError("Task2World currently expects exactly 4 terrain types.")
        if self.num_levels < 2:
            raise ValueError("Task2World expects num_levels >= 2.")

        self.generator_cfg = self._build_generator_cfg()
        self.logical_origins = self._build_logical_origin_mapping()
        self.scan_offsets = self._build_scan_offsets()

        # Optional: after building TerrainImporter in task2_env.py, call:
        # world.set_scene_env_origins(scene.terrain.env_origins)
        self.scene_env_origins: Optional[torch.Tensor] = None

    # -------------------------------------------------------------------------
    # Terrain Generator
    # -------------------------------------------------------------------------
    def _build_generator_cfg(self) -> TerrainGeneratorCfg:
        c = self.cfg

        sub_terrains = {
            "rough_flat": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25,
                noise_range=(float(c.rough_amplitude_min), float(c.rough_amplitude_max)),
                noise_step=float(c.rough_noise_step),
                border_width=float(c.border_width),
            ),
            "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.25,
                slope_range=(float(c.slope_min), float(c.slope_max)),
                platform_width=float(c.platform_width),
                border_width=float(c.border_width),
            ),
            "stepping_stones": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.25,
                grid_width=float(c.stone_grid_width_min),
                grid_height_range=(float(c.stone_height_min), float(c.stone_height_max)),
                platform_width=float(c.platform_width),
            ),
            "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=(float(c.stair_height_min), float(c.stair_height_max)),
                step_width=float(c.stair_step_width_min),
                platform_width=float(c.platform_width),
            ),
        }

        return TerrainGeneratorCfg(
            size=(float(c.terrain_length), float(c.terrain_width)),
            num_rows=int(c.num_levels),
            num_cols=int(c.num_terrain_types),
            sub_terrains=sub_terrains,
            horizontal_scale=float(c.horizontal_scale),
            vertical_scale=float(c.vertical_scale),
            use_cache=False,
            color_scheme="height",
        )

    def _build_logical_origin_mapping(self) -> torch.Tensor:
        """Return analytical fallback origins with shape [num_types, num_levels, 3]."""

        c = self.cfg
        origins = torch.zeros((self.num_types, self.num_levels, 3), dtype=torch.float32, device=self.device)

        # Match TerrainGenerator layout: x -> level rows, y -> terrain type cols.
        start_x = -0.5 * self.num_levels * c.terrain_length + 0.5 * c.terrain_length
        start_y = -0.5 * self.num_types * c.terrain_width + 0.5 * c.terrain_width

        for terrain_type in range(self.num_types):
            for level in range(self.num_levels):
                origins[terrain_type, level, 0] = start_x + level * c.terrain_length
                origins[terrain_type, level, 1] = start_y + terrain_type * c.terrain_width
                origins[terrain_type, level, 2] = 0.0

        return origins

    def _build_scan_offsets(self) -> torch.Tensor:
        c = self.cfg
        xs = torch.linspace(float(c.scan_x_min), float(c.scan_x_max), int(c.scan_num_x), device=self.device)
        ys = torch.linspace(float(c.scan_y_min), float(c.scan_y_max), int(c.scan_num_y), device=self.device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    # -------------------------------------------------------------------------
    # Index / origin mapping
    # -------------------------------------------------------------------------
    def set_scene_env_origins(self, env_origins: torch.Tensor) -> None:
        self.scene_env_origins = env_origins.to(device=self.device, dtype=torch.float32).reshape(-1, 3)

    def sanitize_indices(
        self,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        terrain_types = torch.as_tensor(terrain_types, dtype=torch.long, device=self.device)
        terrain_levels = torch.as_tensor(terrain_levels, dtype=torch.long, device=self.device)

        terrain_types = torch.clamp(terrain_types, 0, self.num_types - 1)
        terrain_levels = torch.clamp(terrain_levels, 0, self.num_levels - 1)
        return terrain_types, terrain_levels

    def get_generator_flat_indices(
        self,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
    ) -> torch.Tensor:
        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)
        return terrain_levels * self.num_types + terrain_types

    def get_logical_origins(
        self,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
    ) -> torch.Tensor:
        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)
        return self.logical_origins[terrain_types, terrain_levels].clone()

    def get_origins_from_indices(
        self,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
        prefer_scene_origins: bool = True,
    ) -> torch.Tensor:
        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)

        if prefer_scene_origins and self.scene_env_origins is not None:
            flat = self.get_generator_flat_indices(terrain_types, terrain_levels)
            flat = torch.clamp(flat, 0, self.scene_env_origins.shape[0] - 1)
            return self.scene_env_origins[flat].clone()

        return self.get_logical_origins(terrain_types, terrain_levels)

    def sample_spawn_origins(
        self,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
        randomize_xy: bool = True,
        prefer_scene_origins: bool = True,
    ) -> torch.Tensor:
        origins = self.get_origins_from_indices(
            terrain_types=terrain_types,
            terrain_levels=terrain_levels,
            prefer_scene_origins=prefer_scene_origins,
        )

        if randomize_xy:
            n = origins.shape[0]
            radius = float(self.cfg.spawn_radius)
            xy_noise = torch.empty((n, 2), dtype=torch.float32, device=self.device).uniform_(-radius, radius)
            origins[:, 0:2] += xy_noise

        origins[:, 2] += float(self.cfg.spawn_height_offset)
        return origins

    # -------------------------------------------------------------------------
    # Terrain parameters / material
    # -------------------------------------------------------------------------
    def level_to_difficulty(self, terrain_levels: torch.Tensor) -> torch.Tensor:
        terrain_levels = torch.as_tensor(terrain_levels, dtype=torch.float32, device=self.device)
        return torch.clamp(terrain_levels / max(self.num_levels - 1, 1), 0.0, 1.0)

    def get_level_parameters(
        self,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)
        d = self.level_to_difficulty(terrain_levels)
        c = self.cfg

        rough_amp = float(c.rough_amplitude_min) + d * (float(c.rough_amplitude_max) - float(c.rough_amplitude_min))
        slope = float(c.slope_min) + d * (float(c.slope_max) - float(c.slope_min))
        stone_height = float(c.stone_height_min) + d * (float(c.stone_height_max) - float(c.stone_height_min))
        stone_grid_width = float(c.stone_grid_width_min) + d * (
            float(c.stone_grid_width_max) - float(c.stone_grid_width_min)
        )
        stair_height = float(c.stair_height_min) + d * (float(c.stair_height_max) - float(c.stair_height_min))
        stair_width = float(c.stair_step_width_min) + d * (
            float(c.stair_step_width_max) - float(c.stair_step_width_min)
        )

        return {
            "terrain_type": terrain_types.float(),
            "terrain_level": terrain_levels.float(),
            "difficulty": d,
            "rough_amp": rough_amp,
            "slope": slope,
            "stone_height": stone_height,
            "stone_grid_width": torch.clamp(stone_grid_width, min=0.05),
            "stair_height": stair_height,
            "stair_width": torch.clamp(stair_width, min=0.05),
        }

    def sample_material_params(
        self,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)
        n = terrain_types.shape[0]
        c = self.cfg

        material_id = torch.randint(0, int(c.material_count), (n,), dtype=torch.long, device=self.device)

        friction = torch.empty(n, dtype=torch.float32, device=self.device)
        low = material_id == 0
        normal = material_id == 1
        high = material_id == 2

        if low.any():
            friction[low] = torch.empty(int(low.sum()), device=self.device).uniform_(
                float(c.low_friction_range[0]), float(c.low_friction_range[1])
            )
        if normal.any():
            friction[normal] = torch.empty(int(normal.sum()), device=self.device).uniform_(
                float(c.normal_friction_range[0]), float(c.normal_friction_range[1])
            )
        if high.any():
            friction[high] = torch.empty(int(high.sum()), device=self.device).uniform_(
                float(c.high_friction_range[0]), float(c.high_friction_range[1])
            )

        restitution = torch.empty(n, dtype=torch.float32, device=self.device).uniform_(
            float(c.restitution_range[0]), float(c.restitution_range[1])
        )

        material_onehot = torch.zeros((n, int(c.material_count)), dtype=torch.float32, device=self.device)
        material_onehot.scatter_(1, material_id.unsqueeze(-1), 1.0)

        return {
            "friction": friction,
            "restitution": restitution,
            "material_id": material_id,
            "material_onehot": material_onehot,
        }

    # -------------------------------------------------------------------------
    # Analytical height scan
    # -------------------------------------------------------------------------
    @staticmethod
    def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
        # q = [w, x, y, z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _rotate_offsets_by_yaw(self, offsets: torch.Tensor, base_quat_wxyz: Optional[torch.Tensor]) -> torch.Tensor:
        if base_quat_wxyz is None:
            return offsets.unsqueeze(0)

        q = torch.as_tensor(base_quat_wxyz, dtype=torch.float32, device=self.device)
        yaw = self._yaw_from_quat_wxyz(q)
        cy = torch.cos(yaw).unsqueeze(-1)
        sy = torch.sin(yaw).unsqueeze(-1)

        ox = offsets[:, 0].unsqueeze(0)
        oy = offsets[:, 1].unsqueeze(0)

        rx = cy * ox - sy * oy
        ry = sy * ox + cy * oy
        return torch.stack([rx, ry], dim=-1)

    def _analytical_terrain_height(
        self,
        local_xy: torch.Tensor,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
    ) -> torch.Tensor:
        """Return terrain height at local points.

        Args:
            local_xy: [N, M, 2]

        Returns:
            height: [N, M]
        """

        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)
        params = self.get_level_parameters(terrain_types, terrain_levels)

        x = local_xy[..., 0]
        y = local_xy[..., 1]

        n, m = x.shape
        height = torch.zeros((n, m), dtype=torch.float32, device=self.device)

        # 0 rough flat: deterministic smooth pseudo-roughness
        mask = terrain_types == 0
        if mask.any():
            amp = params["rough_amp"][mask].unsqueeze(-1)
            h = amp * (
                0.55 * torch.sin(7.1 * x[mask] + 1.7 * y[mask])
                + 0.45 * torch.cos(5.3 * x[mask] - 2.1 * y[mask])
            )
            height[mask] = h

        # 1 slopes: alternating uphill / downhill by level parity
        mask = terrain_types == 1
        if mask.any():
            slope = params["slope"][mask].unsqueeze(-1)
            levels = terrain_levels[mask]
            direction = torch.where((levels % 2) == 0, 1.0, -1.0).to(self.device).unsqueeze(-1)
            # Keep direction as [N, 1] instead of [N], because [N] broadcasts to [N, N].
            height[mask] = direction * slope * x[mask]

        # 2 stepping stones / random grid
        mask = terrain_types == 2
        if mask.any():
            grid = params["stone_grid_width"][mask].unsqueeze(-1)
            amp = params["stone_height"][mask].unsqueeze(-1)
            cx = torch.floor((x[mask] + 10.0) / grid)
            cy = torch.floor((y[mask] + 10.0) / grid)
            hashed = torch.frac(torch.sin(cx * 12.9898 + cy * 78.233) * 43758.5453)
            height[mask] = amp * (hashed - 0.5) * 2.0

        # 3 stairs
        mask = terrain_types == 3
        if mask.any():
            step_w = params["stair_width"][mask].unsqueeze(-1)
            step_h = params["stair_height"][mask].unsqueeze(-1)
            step_index = torch.floor((x[mask] + 0.5 * float(self.cfg.terrain_length)) / step_w)
            step_index = torch.clamp(step_index, min=0.0, max=24.0)
            height[mask] = step_index * step_h

        # Keep central spawn platform safe and close to zero height.
        platform_half = 0.5 * float(self.cfg.platform_width)
        on_platform = (torch.abs(x) <= platform_half) & (torch.abs(y) <= platform_half)
        height = torch.where(on_platform, torch.zeros_like(height), height)

        return torch.clamp(height, -float(self.cfg.height_scan_clip), float(self.cfg.height_scan_clip))

    def sample_height_scan(
        self,
        base_pos_w: torch.Tensor,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
        base_quat_wxyz: Optional[torch.Tensor] = None,
        prefer_scene_origins: bool = True,
    ) -> torch.Tensor:
        base_pos_w = torch.as_tensor(base_pos_w, dtype=torch.float32, device=self.device)
        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)

        origins = self.get_origins_from_indices(
            terrain_types,
            terrain_levels,
            prefer_scene_origins=prefer_scene_origins,
        )

        offsets = self.scan_offsets
        rotated_offsets = self._rotate_offsets_by_yaw(offsets, base_quat_wxyz)

        if rotated_offsets.shape[0] == 1:
            rotated_offsets = rotated_offsets.expand(base_pos_w.shape[0], -1, -1)

        sample_xy_w = base_pos_w[:, None, 0:2] + rotated_offsets
        local_xy = sample_xy_w - origins[:, None, 0:2]

        terrain_height = self._analytical_terrain_height(local_xy, terrain_types, terrain_levels)
        relative_height = terrain_height + origins[:, None, 2] - base_pos_w[:, None, 2]

        return torch.nan_to_num(
            torch.clamp(relative_height, -float(self.cfg.height_scan_clip), float(self.cfg.height_scan_clip)),
            nan=0.0,
            posinf=float(self.cfg.height_scan_clip),
            neginf=-float(self.cfg.height_scan_clip),
        )

    def make_privileged_terrain_features(
        self,
        base_pos_w: torch.Tensor,
        terrain_types: torch.Tensor,
        terrain_levels: torch.Tensor,
        friction: torch.Tensor,
        base_quat_wxyz: Optional[torch.Tensor] = None,
        prefer_scene_origins: bool = True,
    ) -> torch.Tensor:
        terrain_types, terrain_levels = self.sanitize_indices(terrain_types, terrain_levels)
        friction = torch.as_tensor(friction, dtype=torch.float32, device=self.device).view(-1, 1)

        hs = self.sample_height_scan(
            base_pos_w=base_pos_w,
            terrain_types=terrain_types,
            terrain_levels=terrain_levels,
            base_quat_wxyz=base_quat_wxyz,
            prefer_scene_origins=prefer_scene_origins,
        )

        n = terrain_types.shape[0]
        type_onehot = torch.zeros((n, self.num_types), dtype=torch.float32, device=self.device)
        type_onehot.scatter_(1, terrain_types.unsqueeze(-1), 1.0)

        params = self.get_level_parameters(terrain_types, terrain_levels)
        difficulty = params["difficulty"].view(-1, 1)

        param4 = torch.stack(
            [
                params["rough_amp"],
                params["slope"],
                params["stone_height"],
                params["stair_height"],
            ],
            dim=-1,
        )

        priv = torch.cat([hs, friction, type_onehot, difficulty, param4], dim=-1)

        expected = int(self.cfg.terrain_priv_dim)
        if priv.shape[-1] != expected:
            raise RuntimeError(f"terrain privileged dim mismatch: got {priv.shape[-1]}, expected {expected}")

        return torch.nan_to_num(torch.clamp(priv, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)


class TerrainCurriculum:
    """Vectorized terrain curriculum with flat anchor retention."""

    def __init__(self, num_envs: int, world_cfg: Task2TerrainCfg, device: str = "cuda:0"):
        self.cfg = world_cfg
        self.num_envs = int(num_envs)
        self.device = str(device)

        self.num_types = int(world_cfg.num_terrain_types)
        self.num_levels = int(world_cfg.num_levels)

        self.env_types = torch.randint(0, self.num_types, (self.num_envs,), dtype=torch.long, device=self.device)
        self.env_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self.num_levels > 1:
            self.env_levels[:] = torch.randint(0, min(2, self.num_levels), (self.num_envs,), device=self.device)

        anchor_count = int(self.num_envs * float(world_cfg.flat_retention_ratio))
        self.anchor_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if anchor_count > 0:
            self.anchor_mask[:anchor_count] = True

        self.env_types[self.anchor_mask] = 0
        self.env_levels[self.anchor_mask] = 0

        self.start_x = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.probe_upgrades_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.probe_downgrades_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.probe_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.probe_fall_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.probe_max_level_reached = self.env_levels.clone()

        self.upgrade_total = 0
        self.downgrade_total = 0
        self.success_total = 0
        self.fall_total = 0

    def get_current_grid_indices(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return self.env_types[env_ids].clone(), self.env_levels[env_ids].clone()

    def register_start_positions(self, env_ids: torch.Tensor, start_x: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        start_x = torch.as_tensor(start_x, dtype=torch.float32, device=self.device).view(-1)
        self.start_x[env_ids] = start_x

    def reset_envs_to_current(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.env_types[self.anchor_mask] = 0
        self.env_levels[self.anchor_mask] = 0
        return self.get_current_grid_indices(env_ids)

    def _sample_mid_high_levels(self, n: int) -> torch.Tensor:
        low = int(self.cfg.max_level_reset_to_min)
        high = int(self.cfg.max_level_reset_to_max)
        low = max(0, min(low, self.num_levels - 1))
        high = max(low, min(high, self.num_levels - 1))
        return torch.randint(low, high + 1, (int(n),), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update_curriculum(
        self,
        env_ids: torch.Tensor,
        current_x: torch.Tensor,
        fall_flags: torch.Tensor,
        success_flags: Optional[torch.Tensor] = None,
    ) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        current_x = torch.as_tensor(current_x, dtype=torch.float32, device=self.device).view(-1)
        fall_flags = torch.as_tensor(fall_flags, dtype=torch.bool, device=self.device).view(-1)

        progress = current_x - self.start_x[env_ids]

        if success_flags is None:
            success_flags = (progress >= float(self.cfg.success_distance)) & (~fall_flags)
        else:
            success_flags = torch.as_tensor(success_flags, dtype=torch.bool, device=self.device).view(-1)

        failure_flags = fall_flags | (progress <= float(self.cfg.failure_distance))

        is_anchor = self.anchor_mask[env_ids]
        active = ~is_anchor

        self.probe_success_count[env_ids[success_flags]] += 1
        self.probe_fall_count[env_ids[fall_flags]] += 1
        self.success_total += int(success_flags.sum().item())
        self.fall_total += int(fall_flags.sum().item())

        upgrade_mask = active & success_flags
        if upgrade_mask.any():
            uids = env_ids[upgrade_mask]
            old = self.env_levels[uids].clone()

            at_max = old >= self.num_levels - 1
            normal_ids = uids[~at_max]
            max_ids = uids[at_max]

            if normal_ids.numel() > 0:
                self.env_levels[normal_ids] = torch.clamp(self.env_levels[normal_ids] + 1, 0, self.num_levels - 1)

            if max_ids.numel() > 0:
                self.env_levels[max_ids] = self._sample_mid_high_levels(int(max_ids.numel()))
                self.env_types[max_ids] = torch.randint(0, self.num_types, (int(max_ids.numel()),), device=self.device)

            changed = self.env_levels[uids] != old
            self.probe_upgrades_count[uids[changed]] += 1
            self.upgrade_total += int(changed.sum().item())

        downgrade_candidates = active & failure_flags
        if downgrade_candidates.any():
            cand_ids = env_ids[downgrade_candidates]
            do_down = torch.rand(cand_ids.numel(), device=self.device) < float(self.cfg.downgrade_forgiveness_prob)
            dids = cand_ids[do_down]

            if dids.numel() > 0:
                old = self.env_levels[dids].clone()
                self.env_levels[dids] = torch.clamp(self.env_levels[dids] - 1, 0, self.num_levels - 1)
                changed = self.env_levels[dids] != old
                self.probe_downgrades_count[dids[changed]] += 1
                self.downgrade_total += int(changed.sum().item())

        self.env_types[self.anchor_mask] = 0
        self.env_levels[self.anchor_mask] = 0
        self.probe_max_level_reached = torch.maximum(self.probe_max_level_reached, self.env_levels)

    def log_curriculum_stats(self) -> Dict[str, float]:
        active = ~self.anchor_mask

        if active.any():
            active_levels = self.env_levels[active].float()
            active_types = self.env_types[active].float()
            mean_level = float(active_levels.mean().item())
            max_level_active = float(active_levels.max().item())
            mean_type = float(active_types.mean().item())
        else:
            mean_level = 0.0
            max_level_active = 0.0
            mean_type = 0.0

        stats: Dict[str, float] = {
            "Curriculum/Mean_Level_Active": mean_level,
            "Curriculum/Max_Level_Active": max_level_active,
            "Curriculum/Mean_Terrain_Type_Active": mean_type,
            "Curriculum/Max_Level_Reached": float(self.probe_max_level_reached.max().item()),
            "Curriculum/Anchor_Count": float(self.anchor_mask.sum().item()),
            "Curriculum/Upgrade_Total": float(self.upgrade_total),
            "Curriculum/Downgrade_Total": float(self.downgrade_total),
            "Curriculum/Success_Total": float(self.success_total),
            "Curriculum/Fall_Total": float(self.fall_total),
        }

        for terrain_type in range(self.num_types):
            stats[f"Curriculum/Terrain_Type_{terrain_type}_Ratio"] = float(
                (self.env_types == terrain_type).float().mean().item()
            )

        for level in range(self.num_levels):
            stats[f"Curriculum/Level_{level}_Ratio"] = float((self.env_levels == level).float().mean().item())

        return stats
