from __future__ import annotations

import time
from typing import Dict

from go2_rl.common.info_utils import flat_dict


def go2_progress_postfix(env_steps: int, start_time: float, reward_mean: float, done_count: int, info: Dict):
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
