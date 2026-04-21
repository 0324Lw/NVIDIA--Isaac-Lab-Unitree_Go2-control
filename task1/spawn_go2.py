import argparse
import torch

# ===================================================================
# 0. 启动底层引擎 (GUI 模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn Native Unitree Go2 in Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False  # 必须开启 GUI 才能看到狗
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

# 直接从 Isaac Lab 官方资产库中导入 Go2 的完美预配置
from isaaclab_assets import UNITREE_GO2_CFG

@configclass
class Go2TestSceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 2.0
    
    # 实例化机器狗！直接使用官方的 CFG，并替换路径
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    print("\n" + "="*80)
    print("🐶 启动 Unitree Go2 原生模型加载测试")
    print("="*80)

    # 1. 初始化物理引擎
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 显式生成静态地板与全景光源 (不要放进 SceneCfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)
    
    # 2. 生成交互场景 (现在只包含狗了)
    scene_cfg = Go2TestSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("\n[✔] 宇树 Go2 模型加载成功！请在 GUI 界面中观察。")

    # 保持仿真运行
    while simulation_app.is_running():
        sim.step()
        scene.update(0.01)

if __name__ == "__main__":
    main()
    simulation_app.close()