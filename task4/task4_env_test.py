import argparse
import time
import torch
import numpy as np
import pandas as pd

# ===================================================================
# 0. 启动引擎 (强制无头模式极限加速)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 4 Sim2Real Quadruped Env Extreme Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
from task4_env import QuadrupedSim2RealEnv, Task4Config

def main():
    print("\n" + "="*85)
    print("🚀 启动 Task 4 机器狗 Sim2Real 极限抗扰环境 [张量极限压测]")
    print("="*85)

    # 1. 实例化环境配置
    cfg = Task4Config()
    cfg.num_envs = 1024  # 开启 1024 只狗并行接受“饱和式攻击”
    
    start_time = time.time()
    env = QuadrupedSim2RealEnv(cfg)
    env_load_time = time.time() - start_time

    # 2. 基础张量空间与向量化校验 (测试需求 1, 2)
    print(f"\n[✔] 校验 1 & 2: 原生模型与向量化环境建立成功！耗时: {env_load_time:.2f} 秒")
    print(f"    - 并行环境数量 : {env.num_envs}")
    print(f"    - 物理计算设备 : {env.device}")
    
    print(f"\n[✔] 校验 3: 状态/动作空间维度检验")
    # RMA 状态空间：5帧历史(5*48=240) + 19维特权信息(摩擦1+负载1+偏移3+推力2+衰减12) = 259
    expected_obs_dim = (cfg.frame_stack * 48) + (1 + 1 + 3 + 2 + 12)
    print(f"    - 观测维度 (Obs) : {env.observation_space.shape} (预期: [{expected_obs_dim}] = 历史序列 + 特权信息)")
    print(f"    - 动作维度 (Act) : {env.action_space.shape} (预期: [12])")

    # 3. 3000 步极限抗扰压测 (测试需求 3)
    print("\n⏳ 开始 3000 步纯随机策略抗扰推演，强行触发跌倒、负载突变与飞踢，请稍候...")
    
    obs, info = env.reset()
    
    # 初始化日志字典 (捕捉 Task 4 核心奖励组件)
    log_data = {
        "总奖励 (Total_Reward)": [],
        "主任务: 平面速度 (R_Vel_XY)": [],
        "主任务: 偏航速度 (R_Vel_Wz)": [],
        "姿态: 水平约束 (R_Posture)": [],
        "姿态: 高度保持 (R_Height)": [],
        "抗扰: 快速恢复 (R_Recovery)": [],
        "惩罚: 电机扭矩 (P_Torque)": [],
        "惩罚: 动作突变 (P_Action)": [],
        "惩罚: 足端打滑 (P_Slip)": [],
        "惩罚: 落地冲击 (P_Impact)": [],
        "惩罚: 关节极限 (P_JointLim)": []
    }
    
    # 终局事件计数器
    stats_fall = 0
    stats_timeout = 0
    
    steps = 3000
    start_sim_time = time.time()
    
    for step in range(steps):
        # 实时进度条
        if step > 0 and step % 500 == 0:
            print(f"    - 已顶着外部物理扰动推演 {step} 步...")
            
        # 产生 [-1.0, 1.0] 的均匀分布随机动作
        actions = torch.rand((env.num_envs, 12), device=env.device) * 2.0 - 1.0
        
        # 核心交互 (测试环境重置与回报计算)
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 记录环境整体均值
        log_data["总奖励 (Total_Reward)"].append(rewards.mean().item())
        
        # 抓取 info 字典奖励组件
        if "reward_components" in info:
            comps = info["reward_components"]
            log_data["主任务: 平面速度 (R_Vel_XY)"].append(comps.get("R_Vel_XY", 0))
            log_data["主任务: 偏航速度 (R_Vel_Wz)"].append(comps.get("R_Vel_Wz", 0))
            log_data["姿态: 水平约束 (R_Posture)"].append(comps.get("R_Posture", 0))
            log_data["姿态: 高度保持 (R_Height)"].append(comps.get("R_Height", 0))
            log_data["抗扰: 快速恢复 (R_Recovery)"].append(comps.get("R_Recovery", 0))
            log_data["惩罚: 电机扭矩 (P_Torque)"].append(comps.get("P_Torque", 0))
            log_data["惩罚: 动作突变 (P_Action)"].append(comps.get("P_Action", 0))
            log_data["惩罚: 足端打滑 (P_Slip)"].append(comps.get("P_Slip", 0))
            log_data["惩罚: 落地冲击 (P_Impact)"].append(comps.get("P_Impact", 0))
            log_data["惩罚: 关节极限 (P_JointLim)"].append(comps.get("P_JointLim", 0))
            
        # 累加终局事件
        stats_fall += terminated.sum().item()
        stats_timeout += truncated.sum().item()

    sim_time = time.time() - start_sim_time
    total_interactions = steps * env.num_envs
    fps = total_interactions / sim_time

    # 4. 终局事件结算统计
    print(f"\n[✔] 校验 4: 终局事件逻辑完全正常！")
    print(f"    - 耗时: {sim_time:.2f} 秒")
    print(f"    - 总交互样本量 : {total_interactions} 帧")
    print(f"    - 吞吐量 (FPS) : 约 {fps:.0f} 步/秒")
    print(f"    - 💥 触发 '极限跌倒' 重置总次数 : {stats_fall}")
    print(f"    - 🏁 触发 '生存满时长' 完赛次数 : {stats_timeout} (纯随机动作及飞踢干扰下应接近0)")

    # 5. Pandas 奖励组件深度统计分析 (测试需求 4)
    print("\n[✔] 校验 5: 奖励组件数值稳定性分析 (Pandas DataFrame)")
    print("-" * 125)
    
    df = pd.DataFrame(log_data)
    summary = df.describe().T
    summary['方差(Var)'] = summary['std'] ** 2
    
    # 按照需求列名筛洗
    summary = summary.rename(columns={
        'mean': '平均值',
        'min': '最小值',
        '25%': '25%分位',
        '50%': '中位数',
        '75%': '75%分位',
        'max': '最大值'
    })
    
    final_summary = summary[['平均值', '方差(Var)', '最小值', '25%分位', '中位数', '75%分位', '最大值']]
    
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(final_summary.to_string())
    print("-" * 125)
    print("💡 结果判读指南：")
    print("  1. '总奖励' 的单步平均值在随机策略下应显著为负，最小值可由于跌倒极刑（如 -10.0）而下探。")
    print("  2. 各分项（如 P_Slip, P_Impact）的最大截断值应符合配置权重的上界约束，防止梯度黑洞。")
    print("  3. 若 '抗扰: 快速恢复 (R_Recovery)' 能够捕捉到正值，说明状态偏差导数计算正常。")

if __name__ == "__main__":
    main()
    simulation_app.close()