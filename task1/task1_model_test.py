import os
import argparse
import torch
import numpy as np

# ===================================================================
# 0. 啟動引擎 (錄製模式：強制無頭，但開啟渲染)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 1 Quadruped Video Recording")
AppLauncher.add_app_launcher_args(parser)
# 開啟無頭模式，並啟用錄製標誌
args_cli = parser.parse_args()
args_cli.headless = True  
args_cli.video = True     # 啟用內置錄製支持
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心庫導入
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.wrappers import RecordVideo # 用於錄製視頻的包裝器

from task1_env import QuadrupedFlatEnv, Task1Config
from task1_train import IsaacLabSb3Wrapper

def main():
    print("\n" + "="*80)
    print("🎥 啟動 Unitree Go2 自動錄製任務 - 視頻將保存至 ./videos")
    print("="*80)

    # 1. 環境配置 (錄製時建議只開 1-2 個環境，畫面更清晰)
    cfg = Task1Config()
    cfg.num_envs = 2 
    cfg.cmd_vx_range = [0.8, 0.8] # 固定向前奔跑速度，方便觀察步態

    raw_env = QuadrupedFlatEnv(cfg)
    sb3_env = IsaacLabSb3Wrapper(raw_env)

    # 2. 載入模型與歸一化字典
    run_dir = "./runs/task1_sb3_ppo"
    model_path = os.path.join(run_dir, "go2_ppo_final.zip")
    norm_path = os.path.join(run_dir, "vec_normalize_final.pkl")

    vec_env = VecNormalize.load(norm_path, sb3_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    # 3. 掛載視頻錄製包裝器
    video_folder = "./videos/task1_eval"
    # 每 1 步開始錄製，錄製長度為 500 步 (約 10 秒)
    vec_env = RecordVideo(
        vec_env, 
        video_folder=video_folder, 
        episode_trigger=lambda x: x == 0, 
        video_length=500 
    )

    model = PPO.load(model_path, env=vec_env)

    # 4. 推理並錄製
    print(f"🎬 正在背景錄製中... 請稍候約 30 秒...")
    obs = vec_env.reset()
    
    # 執行 600 步確保視頻完整寫入
    for step in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        
        if step % 100 == 0:
            print(f"   已處理 {step}/600 幀...")

    print(f"\n🎉 錄製完成！")
    print(f"📂 視頻文件已保存至: {os.path.abspath(video_folder)}")
    print("💡 你可以使用系統自帶播放器或 VLC 查看 mp4 文件。")

if __name__ == "__main__":
    main()
    simulation_app.close()