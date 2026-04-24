import argparse
import time
import torch
import numpy as np
import pandas as pd

# ===================================================================
# 0. 启动引擎 (强制无头模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 2 Detailed Reward Components Test by Terrain.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)

# ===================================================================
# 核心库导入
# ===================================================================
from task2_env import QuadrupedRoughEnv, Task2Config

def main():
    print("\n" + "="*95)
    print("🚀 启动 Task 2 [全维度] 分地形/分等级奖励函数专项压测")
    print("="*95)

    cfg = Task2Config()
    cfg.num_envs = 1024
    env = QuadrupedRoughEnv(cfg)

    # 禁用动态课程机制，将狗死死锁在指定的 Level 和 Terrain Type 里
    env.curriculum.update_curriculum = lambda *args, **kwargs: None

    # 设置 Pandas 打印宽度，防止折叠
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)

    # 定义要测试的地形字典与难度等级
    terrain_names = {
        0: "波浪粗糙平地 (Rough Flat)",
        1: "倾斜斜坡 (Slopes)",
        2: "不规则乱石滩 (Stepping Stones)",
        3: "离散台阶 (Stairs)"
    }
    test_levels = [0, 4, 9]  # 抽取 平地/初级、中级、极限 三个难度梯度

    for t_type, t_name in terrain_names.items():
        for level in test_levels:
            print(f"\n" + "▼"*95)
            print(f"🏔️ 开始压测场景: [ {t_name} ] | 难度等级: [ Level {level} ]")
            print("▼"*95)

            # 强制所有环境进入特定地形与特定等级
            env.curriculum.env_types[:] = t_type
            env.curriculum.env_levels[:] = level
            env.reset()

            # 全组件详尽日志字典
            log_data = {
                "总奖励 (Total)": [],
                "跟踪: 前向 (R_Vx)": [],
                "跟踪: 横移 (R_Vy)": [],
                "跟踪: 转向 (R_Wz)": [],
                "步态: 抬腿 (R_Clear)": [],
                "步态: 滞空 (R_Air)": [],
                "惩罚: Z轴弹跳 (P_Z)": [],
                "惩罚: 翻滚俯仰 (P_XY)": [],
                "惩罚: 姿态偏离 (P_Ori)": [],
                "惩罚: 高度偏离 (P_H)": [],
                "惩罚: 扭矩能耗 (P_Tau)": [],
                "惩罚: 关节超速 (P_Vel)": [],
                "惩罚: 动作突变 (P_Act)": [],
                "惩罚: 关节位置 (P_Pos)": [],
            }
            
            stats_fall = 0
            steps = 500  # 压测步数
            start_time = time.time()

            for step in range(steps):
                if step > 0 and step % 250 == 0:
                    print(f"    - 场景 {t_name} (Level {level}) 已推演 {step}/{steps} 步...")
                    
                actions = torch.rand((env.num_envs, 12), device=env.device) * 2.0 - 1.0
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # 记录数据
                log_data["总奖励 (Total)"].append(rewards.mean().item())
                if "reward_components" in info:
                    c = info["reward_components"]
                    log_data["跟踪: 前向 (R_Vx)"].append(c.get("R_Vx_Track", 0))
                    log_data["跟踪: 横移 (R_Vy)"].append(c.get("R_Vy_Track", 0))
                    log_data["跟踪: 转向 (R_Wz)"].append(c.get("R_Wz_Track", 0))
                    log_data["步态: 抬腿 (R_Clear)"].append(c.get("R_Clearance", 0))
                    log_data["步态: 滞空 (R_Air)"].append(c.get("R_Air_Time", 0))
                    
                    log_data["惩罚: Z轴弹跳 (P_Z)"].append(c.get("P_Lin_Vel_Z", 0))
                    log_data["惩罚: 翻滚俯仰 (P_XY)"].append(c.get("P_Ang_Vel_XY", 0))
                    log_data["惩罚: 姿态偏离 (P_Ori)"].append(c.get("P_Ori", 0))
                    log_data["惩罚: 高度偏离 (P_H)"].append(c.get("P_Height", 0))
                    
                    log_data["惩罚: 扭矩能耗 (P_Tau)"].append(c.get("P_Tau", 0))
                    log_data["惩罚: 关节超速 (P_Vel)"].append(c.get("P_Dof_Vel", 0))
                    log_data["惩罚: 关节位置 (P_Pos)"].append(c.get("P_Dof_Pos", 0))
                    log_data["惩罚: 动作突变 (P_Act)"].append(c.get("P_Action_Rate", 0))
                    
                stats_fall += terminated.sum().item()

            cost_time = time.time() - start_time
            
            # 统计分析
            df = pd.DataFrame(log_data)
            summary = df.describe().T
            summary['方差 (Var)'] = summary['std'] ** 2
            
            summary = summary.rename(columns={
                'mean': '平均值', 'min': '最小值', '25%': '25%',
                '50%': '中位数', '75%': '75%', 'max': '最大值'
            })
            final_summary = summary[['平均值', '方差 (Var)', '最小值', '中位数', '最大值']]
            
            print(f"\n[ ✔ {t_name} Level {level} 结算报告 ]  耗时: {cost_time:.2f}s | 跌倒重置次数: {stats_fall}")
            print("-" * 95)
            print(final_summary.to_string())
            print("-" * 95)

    print("\n🎉 全维度、全地形场景奖励指标压测完毕！")

if __name__ == "__main__":
    main()
    app_launcher.app.close()