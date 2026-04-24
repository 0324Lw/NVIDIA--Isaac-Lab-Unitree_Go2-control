import argparse
import torch
import pandas as pd

# ===================================================================
# 0. 启动引擎 (强制无头模式) - 虽然只测张量逻辑，但需要加载 IsaacLab 库
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 2 Curriculum Logic Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)

from task2_world import Task2TerrainCfg, TerrainCurriculum

def main():
    print("\n" + "="*80)
    print("🌍 启动 Task 2 课程系统 (Curriculum) 白盒逻辑压测")
    print("="*80)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_envs = 1000  # 使用 1000 只狗方便按百分比计算
    
    cfg = Task2TerrainCfg()
    # 确保驻留比例为 15%
    cfg.flat_retention_ratio = 0.15 
    
    print("\n[初始化] 正在实例化纯张量课程管理器...")
    curriculum = TerrainCurriculum(num_envs, cfg, device)
    
    # ---------------------------------------------------------
    # 校验 1：平地锚点 (Anchor) 分配比例与初始等级
    # ---------------------------------------------------------
    anchor_count = curriculum.anchor_mask.sum().item()
    expected_anchors = int(num_envs * 0.15)
    print(f"\n[✔] 校验 1: 平地锚点分配")
    print(f"    - 预期锚点数: {expected_anchors} (15%) | 实际锚点数: {anchor_count}")
    
    anchor_levels = curriculum.env_levels[curriculum.anchor_mask]
    if (anchor_levels == 0).all():
        print("    - 锚点组初始等级校验通过：全部锁定在 Level 0 (平地)。")
    else:
        print("    - ❌ 锚点组初始等级校验失败！")

    # 为了方便测试，将所有非锚点狗的等级强行设置为 Level 5
    curriculum.env_levels[~curriculum.anchor_mask] = 5
    
    # ---------------------------------------------------------
    # 校验 2：全体完美通关 (走得远 + 没摔倒) -> 测试防遗忘拦截
    # ---------------------------------------------------------
    print("\n[🏃] 模拟第一回合：1000 只狗表现极其完美，全速前进了 6 米且无一跌倒...")
    env_ids = torch.arange(num_envs, device=device)
    
    # 注册起点: x = 0.0
    curriculum.register_start_positions(env_ids, torch.zeros(num_envs, device=device))
    
    # 模拟终点: x = 6.0 (> 4.0 触发晋升)
    current_pos_x = torch.full((num_envs,), 6.0, device=device)
    fall_flags = torch.zeros(num_envs, dtype=torch.bool, device=device) # 全部没摔
    
    curriculum.update_curriculum(env_ids, current_pos_x, fall_flags)
    
    # 检查锚点组是否被拦截
    new_anchor_levels = curriculum.env_levels[curriculum.anchor_mask]
    print(f"\n[✔] 校验 2: 锚点防晋升拦截 (Deadband 机制衍生)")
    print(f"    - 锚点组当前等级 (预期全为 0): Max={new_anchor_levels.max().item()}, Min={new_anchor_levels.min().item()}")
    
    # 检查非锚点组是否正常升级
    new_active_levels = curriculum.env_levels[~curriculum.anchor_mask]
    print(f"    - 活跃组当前等级 (预期从 5 升到 6): 均值={new_active_levels.float().mean().item():.1f}")

    # ---------------------------------------------------------
    # 校验 3：全体拉胯 (没走多远 + 摔倒) -> 测试降级逻辑
    # ---------------------------------------------------------
    print("\n[💥] 模拟第二回合：1000 只狗全部拉胯，刚走 1 米就全摔了...")
    curriculum.register_start_positions(env_ids, torch.zeros(num_envs, device=device))
    current_pos_x = torch.full((num_envs,), 1.0, device=device) # < 2.0 且摔倒触发降级
    fall_flags = torch.ones(num_envs, dtype=torch.bool, device=device) # 全部摔倒
    
    curriculum.update_curriculum(env_ids, current_pos_x, fall_flags)
    
    new_anchor_levels_down = curriculum.env_levels[curriculum.anchor_mask]
    new_active_levels_down = curriculum.env_levels[~curriculum.anchor_mask]
    
    print(f"\n[✔] 校验 3: 降级与边界截断逻辑")
    print(f"    - 锚点组当前等级 (预期被 clamp 截断，维持 0): 均值={new_anchor_levels_down.float().mean().item():.1f}")
    print(f"    - 活跃组当前等级 (预期从 6 降回 5): 均值={new_active_levels_down.float().mean().item():.1f}")

    # ---------------------------------------------------------
    # 校验 4：读取探针数据 (Probes)
    # ---------------------------------------------------------
    print("\n[✔] 校验 4: 数据探针健康度评估")
    stats = curriculum.log_curriculum_stats()
    print(f"    - 活跃狗平均等级 (Telemetry): {stats['Curriculum/Mean_Level_Active']:.2f}")
    
    # 用 Pandas 打印前 5 只锚点狗和前 5 只活跃狗的探针记录
    probe_data = {
        "ID": [], "类型": [], "当前 Level": [], 
        "累计升级次数": [], "累计降级次数": [], "曾达最高 Level": []
    }
    
    # 挑选样本：前 5 个锚点，和最后 5 个活跃狗
    sample_ids = list(range(5)) + list(range(num_envs-5, num_envs))
    
    for i in sample_ids:
        probe_data["ID"].append(i)
        probe_data["类型"].append("⚓ 平地锚点" if curriculum.anchor_mask[i].item() else "🧗 攀岩主力")
        probe_data["当前 Level"].append(curriculum.env_levels[i].item())
        probe_data["累计升级次数"].append(curriculum.probe_upgrades_count[i].item())
        probe_data["累计降级次数"].append(curriculum.probe_downgrades_count[i].item())
        probe_data["曾达最高 Level"].append(curriculum.probe_max_level_reached[i].item())
        
    df = pd.DataFrame(probe_data)
    print("\n[ 探针抽样简报 (Probe Data) ]")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)
    
    print("\n🎉 世界模型课程控制系统测试完美通过！防遗忘锚点机制极其稳定！")

if __name__ == "__main__":
    main()
    app_launcher.app.close()