import os
import argparse
import torch
import numpy as np
from typing import Callable

# ===================================================================
# 0. 启动引擎 (强制无头模式极限加速)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 1 SB3 PPO Training")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # 训练模式强制无头
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import torch.nn as nn

from task1_env import QuadrupedFlatEnv, Task1Config

# ===================================================================
# 1. 适配器：将 Isaac Lab 的 GPU Tensor 映射为 SB3 的 CPU NumPy
# ===================================================================
class IsaacLabSb3Wrapper(VecEnv):
    def __init__(self, env: QuadrupedFlatEnv):
        self.env = env
        # 初始化基类，SB3 会在此处自动调用 get_attr 等方法进行自检
        super().__init__(env.num_envs, env.observation_space, env.action_space)
        self.actions_tensor = None

    def reset(self):
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions):
        # 接收 SB3 下发的 NumPy 动作，转为 GPU Tensor
        self.actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step(self.actions_tensor)
        
        # 转换为 NumPy
        dones = (terminated | truncated).cpu().numpy()
        rewards = rewards.cpu().numpy()
        obs = obs.cpu().numpy()

        infos = [{} for _ in range(self.num_envs)]

        # 正确映射 terminal_observation
        if "terminal_observation" in info:
            term_obs = info["terminal_observation"].cpu().numpy()
            # 提取出所有发生 done 的真实环境 ID
            done_indices = np.nonzero(dones)[0]
            # 遍历这些 ID，将浓缩的 term_obs 逐个准确塞回对应的 infos 字典中
            for idx, env_idx in enumerate(done_indices):
                infos[env_idx]["terminal_observation"] = term_obs[idx]

        # 提取全局遥测数据放入 infos[0]，供 Callback 抓取打印
        if "telemetry" in info:
            infos[0]["telemetry"] = {k: v for k, v in info["telemetry"].items()}

        return obs, rewards, dones, infos

    def close(self): 
        pass
        
    # 正确实现 SB3 要求的环境属性查询接口，返回列表以避免 NoneType 迭代
    def get_attr(self, name, indices=None):
        n = self.num_envs if indices is None else len(indices)
        return [getattr(self.env, name, None) for _ in range(n)]

    def set_attr(self, name, value, indices=None): 
        pass
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        n = self.num_envs if indices is None else len(indices)
        return [None for _ in range(n)]
        
    def env_is_wrapped(self, wrapper_class, indices=None): 
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

# ===================================================================
# 2. 调度器与回调函数 (学习率/持久化/日志)
# ===================================================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """线性衰减调度器 (随着 progress_remaining 从 1 降到 0)"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class CustomLoggingCallback(BaseCallback):
    """自定义控制台输出格式，捕捉遥测数据"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.survival_steps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "telemetry" in infos[0]:
            tel = infos[0]["telemetry"]
            # 注入数据到 SB3 Logger，它会随 ep_rew_mean 一起在控制台打印精美表格
            self.logger.record("custom/Mean_Speed_X", tel["mean_vel_x"])
            self.logger.record("custom/Mean_Height", tel["mean_height"])
            self.logger.record("custom/Fall_Rate", tel["fall_rate"])
            
            # 计算平均存活步数 (用 1 / 跌倒率 估算)
            if tel["fall_rate"] > 0:
                self.logger.record("custom/Avg_Stand_Steps", 1.0 / tel["fall_rate"])
        return True

class SaveNormalizerCallback(BaseCallback):
    """持久化保存 VecNormalize 的均值和方差"""
    def __init__(self, save_freq: int, save_path: str, vec_normalize: VecNormalize):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_normalize = vec_normalize

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            norm_path = os.path.join(self.save_path, f"vec_normalize_{self.num_timesteps}.pkl")
            self.vec_normalize.save(norm_path)
            if self.verbose > 0:
                print(f"💾 状态归一化词典已保存至 {norm_path}")
        return True

# ===================================================================
# 3. 训练主程序
# ===================================================================
def main():
    print("🌍 正在初始化 Task 1 环境与 SB3 适配器...")
    
    # 1. 实例化环境
    env_cfg = Task1Config()
    env_cfg.num_envs = 1024 # 使用 1024 个并行环境
    raw_env = QuadrupedFlatEnv(env_cfg)
    
    # 2. 包装 SB3 适配器与状态归一化
    sb3_env = IsaacLabSb3Wrapper(raw_env)
    # 对状态观测进行 Running Mean/Std 归一化，不归一化 Reward
    vec_env = VecNormalize(sb3_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 3. 训练日志与保存路径
    run_dir = "./runs/task1_sb3_ppo"
    os.makedirs(run_dir, exist_ok=True)
    
    # 4. 构建 PPO 算法与 MLP 网络
    print("🧠 构建 PPO 策略网络...")
    # net_arch 设定策略网络和价值网络均为 [512, 256, 128]，ELU 激活。SB3 默认采用正交初始化。
    policy_kwargs = dict(
        activation_fn=nn.ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
    )
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=linear_schedule(3e-4), # 线性衰减学习率
        n_steps=125,                         # 每次 Rollout 收集 125 步
        batch_size=8000,                     # 大 Batch Size
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.2),     # 线性衰减 PPO 截断幅度
        ent_coef=0.01,                       # 鼓励早期探索
        policy_kwargs=policy_kwargs,
        tensorboard_log=run_dir,
        device="cuda",
        verbose=1                            # 设置为 1 会在控制台输出日志表格
    )
    
    # 5. 回调函数装配
    save_freq = 200 * 1024 * 125 # 约每 200 次网络更新保存一次
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=run_dir, name_prefix="go2_ppo")
    norm_save_callback = SaveNormalizerCallback(save_freq=100_000, save_path=run_dir, vec_normalize=vec_env)
    log_callback = CustomLoggingCallback()
    
    callback_list = CallbackList([checkpoint_callback, norm_save_callback, log_callback])
    
    # 6. 开始训练
    total_timesteps = 50_000_000 # 5000 万步
    print(f"\n🚀 开始 SB3 PPO 训练！目标总步数: {total_timesteps}")
    print("💡 提示：SB3 日志表将在第一次 Rollout 完成后打印在下方...")
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list)
    except KeyboardInterrupt:
        print("\n🛑 用户手动终止训练。")
        
    print("🎉 训练流程结束！正在保存最终模型...")
    model.save(os.path.join(run_dir, "go2_ppo_final.zip"))
    vec_env.save(os.path.join(run_dir, "vec_normalize_final.pkl"))

if __name__ == "__main__":
    main()
    simulation_app.close()