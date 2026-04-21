import argparse
import torch
import math

# ===================================================================
# 0. 启动底层引擎 (GUI 模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Control Unitree Go2 with PD loop.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_assets import UNITREE_GO2_CFG

@configclass
class Go2TestSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4 # 开 4 只狗一起测试
    env_spacing: float = 2.0
    
    # 实例化机器狗
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    print("\n" + "="*80)
    print("🐕 唤醒 Unitree Go2: PD 控制器闭环测试")
    print("="*80)

    # 1. 初始化物理引擎
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 显式生成静态地板与全景光源
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)
    
    # 2. 生成交互场景
    scene_cfg = Go2TestSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[✔] 物理世界初始化完毕。")

    # 3. 提取机器人句柄
    robot = scene.articulations["robot"]
    
    # 获取 Go2 预定义的默认关节姿态 (12维张量)
    # 这通常是狗在站立时的完美角度：Hip ~0, Thigh ~0.8, Calf ~-1.5
    default_joint_pos = robot.data.default_joint_pos.clone()
    print(f"🤖 默认关节目标位置: {default_joint_pos[0].cpu().numpy().round(2)}")

    # 4. 控制循环 (以 50Hz 频率下发控制指令)
    control_dt = 0.02 # 50Hz
    sim_dt = sim_cfg.dt
    decimation = int(control_dt / sim_dt) # 2
    
    step_count = 0
    
    print("\n🎬 仿真开始！观察 4 只机器狗如何通过 PD 闭环站立并做正弦运动。")

    while simulation_app.is_running():
        # -----------------------------------------------------
        # 🎯 动作生成：基于默认姿态添加一个正弦扰动，模拟呼吸/深蹲
        # -----------------------------------------------------
        # 用时间流逝计算相位
        phase = step_count * control_dt * 2.0 * math.pi * 0.5 # 0.5Hz 的频率
        
        # 让所有关节都在默认姿态上下浮动 (Hip不动，Thigh和Calf浮动)
        target_pos = default_joint_pos.clone()
        
        # 给 Thigh (大腿) 添加 +0.2 弧度的正弦波动
        target_pos[:, 1::3] += math.sin(phase) * 0.2
        # 给 Calf (小腿) 添加 -0.3 弧度的正弦波动
        target_pos[:, 2::3] += math.sin(phase) * -0.3
        
        # 下发目标关节位置给底层的 PD 控制器
        robot.set_joint_position_target(target_pos)
        
        # -----------------------------------------------------
        # ⚙️ 物理步进
        # -----------------------------------------------------
        # 在这 2 步内，底层的 C++ 会以 100Hz 计算: Tau = Kp*(Target - Curr) - Kd*Vel
        for _ in range(decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            
        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()