import argparse
import os
import torch
import numpy as np
from datetime import datetime

# ===================================================================
# 1. 启动 Isaac Sim 底层引擎
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO for Quadruped with Curriculum and KL Control")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True # 强制无头模式节省显存
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 2. 导入核心库
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from task2_env import Task2Config, QuadrupedRoughEnv

# ===================================================================
# 3. 自定义Wrapper 类
# ===================================================================
class CustomSb3VecEnvWrapper(VecEnv):
    """自定义 Wrapper：绕过 IsaacLab 类型检查，直接对接 PyTorch 环境和 SB3"""
    def __init__(self, env):
        self.env = env
        # 兼容 Gymnasium / SB3 要求的元数据
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        self.render_mode = None
        super().__init__(env.num_envs, env.observation_space, env.action_space)

    def reset(self):
        obs, _ = self.env.reset()
        return obs.cpu().numpy()  # SB3 需要 numpy 数组

    def step_async(self, actions):
        self.actions = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step(self.actions)
        dones = (terminated | truncated).cpu().numpy()
        
        # 将 PyTorch dict info 转换为 SB3 期望的 list of dicts 格式
        list_infos = [{} for _ in range(self.num_envs)]
        
        if "terminal_observation" in info:
            term_obs = info["terminal_observation"].cpu().numpy()
            reset_idx = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            for i, idx in enumerate(reset_idx):
                list_infos[idx]["terminal_observation"] = term_obs[i]
                
        # 传递遥测数据给 Tensorboard 回调函数
        if "telemetry" in info:
            list_infos[0]["telemetry"] = info["telemetry"]
        if "reward_components" in info:
            list_infos[0]["reward_components"] = info["reward_components"]

        return obs.cpu().numpy(), rewards.cpu().numpy(), dones, list_infos

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    # SB3 的 get_attr 必须返回一个长度为 num_envs 的列表
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
# 4. 高阶机制：自适应 KL 学习率调度器
# ===================================================================
class AdaptiveKLCallback(BaseCallback):
    """
    监控 KL 散度并动态调整学习率。
    原理：保持策略更新在安全范围内，加速平地训练，稳定复杂地形学习。
    """
    def __init__(self, target_kl: float = 0.015, min_lr: float = 1e-5, max_lr: float = 1e-3, verbose=0):
        super().__init__(verbose)
        self.target_kl = target_kl
        self.min_lr = min_lr
        self.max_lr = max_lr

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        approx_kl = self.logger.name_to_value.get("train/approx_kl")
        if approx_kl is not None:
            current_lr = self.model.learning_rate
            new_lr = current_lr
            
            # 根据 KL 散度计算新 LR
            if approx_kl > self.target_kl * 1.5:
                new_lr = max(current_lr / 1.5, self.min_lr)
            elif approx_kl < self.target_kl / 1.5:
                new_lr = min(current_lr * 1.5, self.max_lr)

            if new_lr != current_lr:
                self.model.lr_schedule = lambda _: new_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = new_lr
                self.model.learning_rate = new_lr

# ===================================================================
# 5. 遥测与控制台回调
# ===================================================================
class IsaacTelemetryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "telemetry" in infos[0]:
            t = infos[0]["telemetry"]
            reward_comps = infos[0].get("reward_components", {})
            # 记录到 Tensorboard
            self.logger.record("isaac/mean_vel_x", t.get("mean_vel_x", 0))
            self.logger.record("isaac/fall_rate", t.get("fall_rate", 0))
            self.logger.record("isaac/curriculum_level", t.get("Curriculum/Mean_Level_Active", 0))
            self.logger.record("isaac/deadband_trigger_rate", t.get("probe_height_deadband_rate", 0))
            for key, val in reward_comps.items():
                self.logger.record(f"rewards/{key}", val)
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "telemetry" in infos[0]:
            t = infos[0]["telemetry"]
            print(f"\n🚀 [Rollout {self.rollout_count}] 速度: {t.get('mean_vel_x',0):.2f} m/s | "
                  f"难度: {t.get('Curriculum/Mean_Level_Active',0):.1f} | "
                  f"跌倒率: {t.get('fall_rate',0)*100:.1f}% | "
                  f"LR: {self.model.learning_rate:.2e}")

# ===================================================================
# 6. 主训练逻辑
# ===================================================================
def main():
    set_random_seed(42)
    log_dir = f"./logs/ppo_quadruped_300M_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    # 环境初始化
    env_cfg = Task2Config()
    env_cfg.num_envs = 4096 # 智能体数量 4096
    print(f"[INFO] 正在启动 {env_cfg.num_envs} 个并行环境...")
    
    base_env = QuadrupedRoughEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(base_env)
    
    # 状态与奖励归一化 (启用滑动平均统计)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 网络与算法配置
    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        log_std_init=-1.0 # 初始探索方差，-1.0 对应约 0.36 的标准差
    )

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,     # 初始学习率，随后由 AdaptiveKL 控制
        n_steps=32,             # 每次 rollout 步数：32 * 4096 = 131,072 样本
        batch_size=32768,       # 显存足够建议设大，加速计算
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,           # 依靠 std_init 探索，固定动作分布
        clip_range=0.2,
        max_grad_norm=1.0,      # 梯度裁剪防梯度爆炸
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cuda:0",
        verbose=1
    )

    model = PPO(**ppo_kwargs)

    # 回调函数列表
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="quad_model")
    kl_callback = AdaptiveKLCallback(target_kl=0.015)
    telemetry_callback = IsaacTelemetryCallback()

    total_timesteps = 300_000_000 # 3 亿步
    print(f"\n 训练开始！目标步数: {total_timesteps}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[telemetry_callback, checkpoint_callback, kl_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[WARN] 训练被手动中断")
    finally:
        # 保存最终模型与归一化参数
        model.save(os.path.join(log_dir, "final_model_300M.zip"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        print(f"[SUCCESS] 训练完成，结果已保存至 {log_dir}")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()