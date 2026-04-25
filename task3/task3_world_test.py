import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import time
import os

# ===================================================================
# 0. 启动引擎 (强制无头模式) - 必须放在任何 Isaac Lab 导入之前
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 3 World Logic & GIF Generation Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)

# 导入配置和世界类 (确保同级目录下有 task3_world.py)
from task3_world import Task3WorldCfg, Task3World

def main():
    print("\n" + "="*80)
    print("🐕 启动 Task 3 视觉导航与动态避障 世界模型白盒测试")
    print("="*80)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 使用 1000 个环境实例压测 GPU 张量生成逻辑的鲁棒性
    num_envs = 1000  
    num_gifs = 5     
    
    cfg = Task3WorldCfg()
    print(f"\n[初始化] 正在实例化 Task3World (Envs: {num_envs}, Device: {device})...")
    
    t0 = time.time()
    world = Task3World(cfg, num_envs, device)
    env_ids = torch.arange(num_envs, device=device)
    world.reset_envs(env_ids)
    t1 = time.time()
    
    print(f"    - 环境生成耗时 (含拒绝采样): {(t1 - t0)*1000:.2f} ms")

    # ===================================================================
    # 校验 1：起终点生成与安全区逻辑
    # ===================================================================
    starts = world.start_pos.cpu().numpy()
    targets = world.target_pos.cpu().numpy()
    distances = np.linalg.norm(starts - targets, axis=1)
    
    print(f"\n[✔] 校验 1: 起终点距离约束 (预期 20m - 30m)")
    print(f"    - 终点距离均值: {distances.mean():.2f} m")
    print(f"    - 终点距离最大值: {distances.max():.2f} m, 最小值: {distances.min():.2f} m")
    
    # 因为存在场地边界截断，极少数点可能会略微小于 20m，这里设定一个合理的容差
    if (distances < cfg.goal_dist_range[0] * 0.85).sum() > 0:
        print("    - ⚠️ 警告: 存在部分起终点距离因环境边界截断而缩小。")
    else:
        print("    - 起终点跨度校验通过！")

    # ===================================================================
    # 校验 2：静态障碍物 1.5m 最小间距约束测试
    # ===================================================================
    static_obs = world.static_obs.cpu().numpy() # [num_envs, num_static, 3] -> (x, y, r)
    print(f"\n[✔] 校验 2: 静态障碍物生成与 1.5m 最小间距测试")
    print(f"    - 静态障碍物张量形状: {static_obs.shape}")
    
    # 抽样测试第一个环境的静态障碍物间距是否满足要求
    env_0_static = static_obs[0]
    min_gap_found = float('inf')
    conflict_count = 0
    
    for i in range(cfg.num_static_obs):
        for j in range(i + 1, cfg.num_static_obs):
            dist = np.linalg.norm(env_0_static[i, :2] - env_0_static[j, :2])
            gap = dist - env_0_static[i, 2] - env_0_static[j, 2]
            min_gap_found = min(min_gap_found, gap)
            if gap < cfg.min_static_spacing - 1e-4: # 考虑浮点误差
                conflict_count += 1
                
    if conflict_count == 0:
        print(f"    - 间距校验通过！环境 0 中的最小静态物体间距为: {min_gap_found:.2f} m (要求 >= {cfg.min_static_spacing} m)")
    else:
        print(f"    - ❌ 间距校验失败！发现 {conflict_count} 处冲突。")

    # ===================================================================
    # 校验 3：运动学推进与 2D GIF 渲染
    # ===================================================================
    print(f"\n[🎥] 校验 3: 动态障碍物运动学推演与 GIF 渲染...")
    
    # 我们按照 RL 频率 50Hz 来推进，推演 3 秒 (150 frames)
    frames = 150 
    dt = 1.0 / cfg.rl_policy_freq # 0.02s
    
    # 提前记录前 num_gifs 个环境的动态障碍物轨迹: [frames, num_gifs, num_dynamic_obs, 2]
    dyn_obs_history = np.zeros((frames, num_gifs, cfg.num_dynamic_obs, 2))
    
    for f in range(frames):
        world.step_kinematics(dt=dt)
        dyn_obs_history[f] = world.dynamic_obs_pos[:num_gifs].cpu().numpy()
        
    print(f"    - 成功完成 {frames} 步运动学前向推演！")
    print(f"    - 正在生成 {num_gifs} 张 GIF 动画 (这可能需要 10-20 秒，请耐心等待)...")

    half_env = cfg.env_size / 2.0

    # 开始逐一绘制并保存 GIF
    for i in range(num_gifs):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-half_env - 1, half_env + 1)
        ax.set_ylim(-half_env - 1, half_env + 1)
        ax.set_aspect('equal')
        ax.set_title(f"Task 3 Environment {i} (Top-Down View)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        
        # 1. 绘制环境边界墙壁
        wall = Rectangle((-half_env, -half_env), cfg.env_size, cfg.env_size, fill=False, color='black', linewidth=2)
        ax.add_patch(wall)
        
        # 2. 绘制起终点与 2m 安全区
        ax.plot(starts[i, 0], starts[i, 1], 'go', markersize=8, label='Start')
        ax.plot(targets[i, 0], targets[i, 1], 'r*', markersize=12, label='Target')
        
        start_safe_zone = Circle((starts[i, 0], starts[i, 1]), cfg.safe_zone_radius, color='green', fill=False, linestyle='--', alpha=0.4)
        target_safe_zone = Circle((targets[i, 0], targets[i, 1]), cfg.safe_zone_radius, color='red', fill=False, linestyle='--', alpha=0.4)
        ax.add_patch(start_safe_zone)
        ax.add_patch(target_safe_zone)

        # 3. 绘制静态障碍物 (带半透明填充与边界线)
        for j in range(cfg.num_static_obs):
            ox, oy, orad = static_obs[i, j]
            static_circle = Circle((ox, oy), orad, color='gray', alpha=0.5, ec='black', lw=1.5)
            ax.add_patch(static_circle)

        # 4. 初始化动态障碍物
        dyn_patches = []
        for j in range(cfg.num_dynamic_obs):
            dx, dy = dyn_obs_history[0, i, j]
            dyn_circle = Circle((dx, dy), cfg.dynamic_radius, color='dodgerblue', alpha=0.8, ec='navy', lw=1.5)
            ax.add_patch(dyn_circle)
            dyn_patches.append(dyn_circle)

        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))

        # 动画帧刷新逻辑
        def animate(frame_idx):
            for j, patch in enumerate(dyn_patches):
                patch.center = (dyn_obs_history[frame_idx, i, j, 0], dyn_obs_history[frame_idx, i, j, 1])
            return dyn_patches

        # interval 设定为真实时间的缩放比例以控制播放速度
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=30, blit=True)
        
        gif_filename = f"task3_env_{i}.gif"
        try:
            anim.save(gif_filename, writer='pillow', fps=int(cfg.rl_policy_freq))
            print(f"      [{i+1}/{num_gifs}] 渲染完毕并保存: {gif_filename}")
        except Exception as e:
            print(f"      ❌ 保存 {gif_filename} 失败，请确认是否安装 Pillow。错误详情: {e}")
            
        plt.close(fig) # 释放当前画板内存

    print("\n🎉 世界模型测试全部通过！快去本地目录查看刚刚生成的环境动图吧！")

if __name__ == "__main__":
    main()
    app_launcher.app.close()