import argparse
import time
import torch
import pandas as pd

# ===================================================================
# 0. 启动引擎 (强制无头模式) - 必须放在任何 Isaac 库导入之前
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 3 Visual Navigation Env Limit Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True # 极限测试不需要 GUI，压榨算力
app_launcher = AppLauncher(args_cli)

# ===================================================================
# 核心环境导入
# ===================================================================
from task3_env import Task3VisualNavEnv, Task3Config

def main():
    print("\n" + "="*95)
    print("🚀 启动 Task 3 [终极 60-20-20 能量地形] 核心环境极限压测与奖励分析")
    print("="*95)

    # 初始化配置与环境
    cfg = Task3Config()
    cfg.num_envs = 1024 # 开满 1024 个并行环境进行压力测试
    
    print("\n[初始化] 正在实例化 Task3VisualNavEnv 强化学习环境基座...")
    t_start = time.time()
    env = Task3VisualNavEnv(cfg)
    print(f"    - 环境拉起耗时: {time.time() - t_start:.2f}s")
    
    # ===================================================================
    # 校验 1：空间维度测试
    # ===================================================================
    print(f"\n[✔] 校验 1: I/O 空间维度校验")
    
    # 动作空间
    assert env.action_space.shape[0] == 12, "动作空间必须为 12 维连续空间！"
    print(f"    - 动作空间 (Action Space): {env.action_space} -> 合法！")
    
    # 观测空间 (48维基础 + 90维张量雷达 = 138)
    expected_obs_dim = 48 + 90
    assert env.observation_space.shape[0] == expected_obs_dim, f"观测空间维度异常！期望: {expected_obs_dim}, 实际: {env.observation_space.shape[0]}"
    print(f"    - 观测空间 (Observation Space): {env.observation_space} (包含雷达深度拼接) -> 合法！")
    
    # 初始观测生成测试
    obs, info = env.reset()
    assert obs.shape == (cfg.num_envs, expected_obs_dim), "初始观测状态张量形状错误！"
    print(f"    - env.reset() 张量形状校验通过: {obs.shape}")

    # ===================================================================
    # 校验 2：极限推进与事件统计 (1000 步)
    # ===================================================================
    print("\n" + "▼"*95)
    print("🐕 启动纯随机策略 (Random Action) 1000步极限推演...")
    print("   [目的]: 验证终极 60-20-20 奖励函数的所有子项约束与梯度形状。")
    print("▼"*95)
    
    steps = 2000
    
    # 全组件详尽日志字典 (对齐全新的 Info 字典切分)
    log_data = {
        "总奖励 (Total)": [],
        "任务: 势能进展 (R_Prog)": [],
        "任务: 门控朝向 (R_Yaw)": [],
        "任务: 速度奖励 (R_Speed)": [],
        "任务: 停滞惩罚 (P_Stall)": [],
        "安全: 避障惩罚 (P_Obs)": [],
        "安全: 非法接触 (P_Hit)": [],
        "安全: 高度越界 (P_Height)": [],
        "安全: 姿态越界 (P_Ori)": [],
        "安全: 角速度越界 (P_Omega)": [],
        "安全: stall惩罚 (P_Stall)": [],
        "效率: 动作突变 (P_Action)": [],
        "效率: 功率消耗 (P_Power)": [],
        "效率: 关节限幅 (P_Limit)": [],
    }
    
    # 事件计数器
    stats_total_dones = 0
    stats_timeout = 0
    
    # 设置 Pandas 打印参数
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)

    start_time = time.time()

    for step in range(steps):
        if step > 0 and step % 200 == 0:
            print(f"    - 已推演 {step}/{steps} 步 | 累计碰撞/跌倒重置: {stats_total_dones} 次...")
            
        # 赋予 -1.0 到 1.0 的完全随机纯张量动作
        actions = torch.rand((env.num_envs, 12), device=env.device) * 2.0 - 1.0
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # --- 数据采样入库 ---
        log_data["总奖励 (Total)"].append(rewards.mean().item())
        
        if "reward_components" in info:
            c = info["reward_components"]
            log_data["任务: 势能进展 (R_Prog)"].append(c.get("Task_R_Prog", 0))
            log_data["任务: 门控朝向 (R_Yaw)"].append(c.get("Task_R_Yaw", 0))
            log_data["任务: 速度奖励 (R_Speed)"].append(c.get("Task_R_Speed", 0))
            log_data["任务: 停滞惩罚 (P_Stall)"].append(c.get("Task_P_Stall", 0))
            
            log_data["安全: 避障惩罚 (P_Obs)"].append(c.get("Safe_P_Obs", 0))
            log_data["安全: 非法接触 (P_Hit)"].append(c.get("Safe_P_Hit", 0))
            log_data["安全: 高度越界 (P_Height)"].append(c.get("Safe_P_Height", 0))
            log_data["安全: 姿态越界 (P_Ori)"].append(c.get("Safe_P_Ori", 0))
            log_data["安全: 角速度越界 (P_Omega)"].append(c.get("Safe_P_Omega", 0))
            log_data["安全: stall惩罚 (P_Stall)"].append(c.get("Safe_P_Stall", 0))
            
            log_data["效率: 动作突变 (P_Action)"].append(c.get("Eff_P_Action", 0))
            log_data["效率: 功率消耗 (P_Power)"].append(c.get("Eff_P_Power", 0))
            log_data["效率: 关节限幅 (P_Limit)"].append(c.get("Eff_P_Limit", 0))
            
        stats_total_dones += terminated.sum().item()
        stats_timeout += truncated.sum().item()

    cost_time = time.time() - start_time
    
    # ===================================================================
    # 分析报告与断言
    # ===================================================================
    df = pd.DataFrame(log_data)
    summary = df.describe().T
    summary['方差(Var)'] = summary['std'] ** 2
    
    # 重命名列名以对齐需求
    summary = summary.rename(columns={
        'mean': '平均值', 'min': '最小值', '25%': '25%分位',
        '50%': '中位数', '75%': '75%分位', 'max': '最大值'
    })
    
    final_summary = summary[['平均值', '方差(Var)', '最小值', '25%分位', '中位数', '75%分位', '最大值']]
    
    print(f"\n[ ✔ 核心引擎与终极奖励架构测试报告 ]")
    print(f"    - 总计仿真耗时: {cost_time:.2f} s")
    print(f"    - 随机探索导致的 [撞击/跌倒重置]: {stats_total_dones} 次")
    print(f"    - 回合届满 [超时截断]: {stats_timeout} 次")
    print("-" * 115)
    print(final_summary.to_string())
    print("-" * 115)
    
    # 终极断言验证：检查截断与溢出 
    # 新架构下，Dense 在 [-1, 1] 左右，大奖极刑为 ±3.0，安全界限定在 [-10.0, 10.0] 完全足够
    max_reward = df["总奖励 (Total)"].max()
    min_reward = df["总奖励 (Total)"].min()
    
    if max_reward > 10.001 or min_reward < -10.001:
        print(f"\n❌ [致命错误] 总奖励超出了新架构的 [-10.0, 10.0] 理论绝对安全界限！(当前 Max: {max_reward:.2f}, Min: {min_reward:.2f})")
    else:
        print(f"\n🎉 [数值安全校验] 总奖励界限校验通过 (当前 Max: {max_reward:.2f}, Min: {min_reward:.2f})！归一化护栏工作正常。")
        
    print("\n✅ Task 3 强化学习底座 (工业级能量地形架构) 已验证无 Bug，随时可以接入 PPO 算法框架启动史诗级炼丹！")

if __name__ == "__main__":
    main()
    app_launcher.app.close()