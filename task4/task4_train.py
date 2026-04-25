import argparse
import os
import torch
import numpy as np
import logging
from datetime import datetime

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("isaaclab.assets.articulation.articulation").setLevel(logging.ERROR)

# ===================================================================
# 0. 启动底层物理引擎 (强制无头模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train RMA Teacher Policy for Sim2Real Quadruped (PPO)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

# ===================================================================
# 1. 核心算法与环境库导入
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# 导入刚刚编写的 Task4 抗扰环境
from task4_env import Task4Config, QuadrupedSim2RealEnv

# ===================================================================
# 2. SB3 向量化环境桥接器 
# ===================================================================
class CustomSb3VecEnvWrapper(VecEnv):
    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        self.render_mode = None
        super().__init__(env.num_envs, env.observation_space, env.action_space)
        
        self.ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs, _ = self.env.reset()
        self.ep_returns[:] = 0.0
        self.ep_lengths[:] = 0
        return obs.cpu().numpy()

    def step_async(self, actions):
        self.actions = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step(self.actions)
        dones = (terminated | truncated).cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        
        self.ep_returns += rewards_np
        self.ep_lengths += 1
        
        list_infos = [{} for _ in range(self.num_envs)]

        # 提取终端状态用于 PPO Bootstrap
        if "terminal_observation" in info:
            term_obs = info["terminal_observation"].cpu().numpy()
            reset_idx = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            for i, idx in enumerate(reset_idx):
                list_infos[idx]["terminal_observation"] = term_obs[i]
                list_infos[idx]["is_success"] = info["is_success"][idx].item()
                
                # 🚨 [修复 2.3] 注入 SB3 专属的 episode 统计字典
                list_infos[idx]["episode"] = {
                    "r": self.ep_returns[idx], 
                    "l": self.ep_lengths[idx]
                }
                # 重置该环境的累加器
                self.ep_returns[idx] = 0.0
                self.ep_lengths[idx] = 0

        # 提取全图遥测指标给 Logger
        if "telemetry" in info:
            list_infos[0]["telemetry"] = info["telemetry"]
        if "reward_components" in info:
            list_infos[0]["reward_components"] = info["reward_components"]

        return obs.cpu().numpy(), rewards_np, dones, list_infos

    def close(self):
        if hasattr(self.env, "close"): self.env.close()
    def get_attr(self, name, indices=None): return [getattr(self.env, name, None)] * self.num_envs
    def set_attr(self, name, value, indices=None): setattr(self.env, name, value)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

# ===================================================================
# 3. 训练稳定性护航：自适应 KL 散度调度器
# ===================================================================
class AdaptiveKLCallback(BaseCallback):
    """
    通过动态干预学习率，防止在随机化突变（如遭遇大飞踢）时网络梯度爆炸或策略崩溃。
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
            # 策略变化过大，紧急刹车
            if approx_kl > self.target_kl * 1.5:
                new_lr = max(current_lr / 1.5, self.min_lr)
            # 策略过于保守，加大步伐
            elif approx_kl < self.target_kl / 1.5:
                new_lr = min(current_lr * 1.5, self.max_lr)
                
            if new_lr != current_lr:
                self.model.lr_schedule = lambda _: new_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = new_lr
                self.model.learning_rate = new_lr

# ===================================================================
# 4. Sim2Real 专属遥测日志面板
# ===================================================================
class IsaacTelemetryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "reward_components" in infos[0]:
            # 分离并记录 10 项细分抗扰奖励
            for key, val in infos[0]["reward_components"].items():
                self.logger.record(f"rewards/{key}", val)
            
            # 记录核心遥测状态
            t = infos[0].get("telemetry", {})
            self.logger.record("sim2real/mean_vel_err", t.get("mean_vel_err", 0.0))
            self.logger.record("sim2real/fall_rate", t.get("fall_rate", 0.0))
            self.logger.record("sim2real/active_push_ratio", t.get("active_push_ratio", 0.0))
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        infos = self.locals.get("infos", [])
        
        if len(infos) > 0 and "telemetry" in infos[0]:
            t = infos[0]["telemetry"]
            print(f"\n🚀 [Rollout {self.rollout_count} | 步数: {self.num_timesteps}]")
            print(f"   🏃 速度追踪误差: {t.get('mean_vel_err', 0):.2f} m/s | 跌倒极刑率: {t.get('fall_rate', 0)*100:.1f}%")
            print(f"   🌪️ 当前受击比率(飞踢激活): {t.get('active_push_ratio', 0)*100:.1f}%")
            print(f"   ⚙️ LR: {self.model.learning_rate:.2e}")

# ===================================================================
# 5. RMA 教师网络主训练流
# ===================================================================
def main():
    set_random_seed(42)
    log_dir = f"./logs/rma_teacher_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    env_cfg = Task4Config()
    env_cfg.num_envs = 4096 
    print(f"\n[INFO] 正在初始化 {env_cfg.num_envs} 个高强度抗扰并行环境 (Teacher Phase)...")
    
    base_env = QuadrupedSim2RealEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(base_env)
    
    # 状态与奖励滑动归一化 (防梯度爆炸的最后一道防线)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 教师网络结构配置：
    # 鉴于状态空间增加至 259 维(且蕴含高度复杂的历史序列和环境真值)，加宽了隐藏层
    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[512, 256, 128], vf=[768, 512, 256]),
        ortho_init=True, 
        log_std_init=-1.0 
    )

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=64,             # 延长单次探索步数以捕获扰动恢复序列 (64 * 4096 = 262,144 transitions/rollout)
        batch_size=32768,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,           # 连续空间锁死熵增，防止物理乱蹬
        clip_range=0.2,
        max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cuda:0",
        verbose=1
    )

    model = PPO(**ppo_kwargs)

    # 设置自动保存与回调
    checkpoint_callback = CheckpointCallback(save_freq=3000, save_path=log_dir, name_prefix="rma_teacher")
    kl_callback = AdaptiveKLCallback(target_kl=0.015)
    telemetry_callback = IsaacTelemetryCallback()

    total_timesteps = 1000_000_000
    print(f"\n🔥 [Phase 1: Teacher] 炼丹炉点火！目标步数: {total_timesteps}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[telemetry_callback, checkpoint_callback, kl_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[WARN] 接收到手动中断信号，正在安全保存模型...")
    finally:
        # 保存最终成果与归一化标量器 (部署实机时不可或缺)
        model.save(os.path.join(log_dir, "final_rma_teacher.zip"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        print(f"🎉 [SUCCESS] 教师网络训练结束，模型与环境状态器存储于: {log_dir}")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()