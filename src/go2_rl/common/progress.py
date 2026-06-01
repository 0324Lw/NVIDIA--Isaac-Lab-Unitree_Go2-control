# Copyright (c) 2026
# Unitree Go2 Common: 训练进度条显示工具。
#
# 本文件提供 tqdm postfix 字段构造函数。
# 本文件不依赖 IsaacLab，不启动 AppLauncher，也不创建训练环境。
#
# 主要职责:
#   1. 从环境 info 字典中提取低频 telemetry / event 标量；
#   2. 计算当前环境步吞吐量；
#   3. 返回适合 tqdm.set_postfix() 使用的简洁字典。
#
# 工程说明:
#   进度条只用于人工观察训练状态，不参与奖励、观测、reset、课程或 PPO 更新。
#   这里读取的 info 值已经由 info_utils.flat_dict 做 best-effort 标量转换。
#
# Unitree Go2 Common: training progress display utilities.
#
# This file provides helper functions for building tqdm postfix fields.
# It does not depend on IsaacLab, launch AppLauncher, or create training environments.
#
# Main responsibilities:
#   1. Extract low-frequency telemetry / event scalars from the environment info dictionary;
#   2. Compute the current environment-step throughput;
#   3. Return a compact dictionary suitable for tqdm.set_postfix().
#
# Engineering notes:
#   The progress bar is only for human-readable training monitoring. It does not
#   affect rewards, observations, reset logic, curriculum, or PPO updates. The
#   info values are converted by info_utils.flat_dict on a best-effort basis.

from __future__ import annotations

import time
from typing import Dict

from go2_rl.common.info_utils import flat_dict


def go2_progress_postfix(env_steps: int, start_time: float, reward_mean: float, done_count: int, info: Dict):
    """Build a compact tqdm postfix dictionary for Go2 training loops."""

    flat = flat_dict(info)
    fps = env_steps / max(time.time() - start_time, 1e-6)

    return {
        "steps": f"{env_steps:,}",
        "fps": f"{fps:,.0f}",
        "rew": f"{reward_mean:.3f}",
        "done": int(done_count),
        "stage": f"{flat.get('telemetry/Command_Stage', 0.0):.0f}",
        "vx": f"{flat.get('telemetry/Actual_Vx', 0.0):.2f}/{flat.get('telemetry/Cmd_Vx', 0.0):.2f}",
        "h": f"{flat.get('telemetry/Base_Height', 0.0):.2f}",
        "ct": f"{flat.get('telemetry/Contact_Count', 0.0):.2f}",
        "fall": f"{flat.get('events/Fall_Rate', 0.0):.3f}",
    }
