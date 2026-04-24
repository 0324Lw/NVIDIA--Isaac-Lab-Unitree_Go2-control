import argparse
import os
import glob
import torch
import numpy as np

# ===================================================================
# 1. 啟動 Isaac Sim 底層引擎
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained PPO model for Quadruped")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 🌟【關鍵】測試時必須關閉無頭模式，打開渲染視窗
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 2. 匯入核心庫與環境
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from task2_env import Task2Config, QuadrupedRoughEnv

# ===================================================================
# 3. 內嵌自定義 Wrapper (避免 ImportError)
# ===================================================================
class CustomSb3VecEnvWrapper(VecEnv):
    """自定義 Wrapper：繞過 IsaacLab 類型檢查，直接對接 PyTorch 環境和 SB3"""
    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        self.render_mode = None
        super().__init__(env.num_envs, env.observation_space, env.action_space)

    def reset(self):
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions):
        self.actions = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step(self.actions)
        dones = (terminated | truncated).cpu().numpy()
        list_infos = [{} for _ in range(self.num_envs)]
        if "terminal_observation" in info:
            term_obs = info["terminal_observation"].cpu().numpy()
            reset_idx = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            for i, idx in enumerate(reset_idx):
                list_infos[idx]["terminal_observation"] = term_obs[i]
        return obs.cpu().numpy(), rewards.cpu().numpy(), dones, list_infos

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def get_attr(self, name, indices=None):
        val = getattr(self.env, name, None)
        return [val] * self.num_envs

    def set_attr(self, name, value, indices=None): 
        setattr(self.env, name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): 
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None): 
        return [False] * self.num_envs

# ===================================================================
# 4. 自動尋找最新訓練日誌目錄
# ===================================================================
def get_latest_log_dir(base_dir="./logs"):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"找不到日誌目錄: {base_dir}")
    
    # 尋找所有以 ppo_quadruped 開頭的資料夾
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) and "ppo_quadruped" in d]
    
    if not dirs:
        raise FileNotFoundError(f"在 {base_dir} 中找不到任何訓練日誌資料夾")
        
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

# ===================================================================
# 5. 主推論迴圈
# ===================================================================
def main():
    print("\n" + "="*80)
    print("🚀 啟動模型推論與視覺化測試")
    print("="*80)

    # 1. 自動定位權重文件
    latest_log_dir = get_latest_log_dir()
    model_path = os.path.join(latest_log_dir, "final_model_300M.zip")
    stats_path = os.path.join(latest_log_dir, "vec_normalize.pkl")
    
    # 如果找不到 final model，嘗試尋找最新的 checkpoint
    if not os.path.exists(model_path):
        print(f"[WARN] 找不到 final_model_300M.zip，正在尋找最新 Checkpoint...")
        checkpoints = glob.glob(os.path.join(latest_log_dir, "*.zip"))
        if not checkpoints:
            raise FileNotFoundError("找不到任何 .zip 模型權重檔！")
        model_path = max(checkpoints, key=os.path.getmtime)

    print(f"[INFO] 載入模型權重: {model_path}")
    print(f"[INFO] 載入歸一化狀態: {stats_path}")

    # 2. 初始化環境 (測試時不需要 4096 隻狗，64 隻足以觀察各種地形表現)
    env_cfg = Task2Config()
    env_cfg.num_envs = 64 
    base_env = QuadrupedRoughEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(base_env)

    # 3. 載入狀態歸一化統計資料
    env = VecNormalize.load(stats_path, env)
    # 🌟【關鍵】關閉訓練狀態，凍結均值和方差的更新
    env.training = False
    env.norm_reward = False 

    # 4. 載入 PPO 模型
    model = PPO.load(model_path, env=env)
    print("[SUCCESS] 模型載入完成！開始推論...\n")

    # 5. 推論迴圈
    obs = env.reset()
    
    try:
        while simulation_app.is_running():
            # 🌟【關鍵】deterministic=True 關閉探索噪聲，輸出均值動作
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
    except KeyboardInterrupt:
        print("\n[INFO] 使用者手動中斷推論。")
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()