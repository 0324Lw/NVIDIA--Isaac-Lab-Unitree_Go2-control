import argparse
import os
import torch
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO for Quadruped Visual Navigation (60-20-20 Architecture)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from task3_env import Task3Config, Task3VisualNavEnv


class CustomSb3VecEnvWrapper(VecEnv):
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
                list_infos[idx]["is_success"] = info["is_success"][idx].item()

        if "telemetry" in info:
            list_infos[0]["telemetry"] = info["telemetry"]
        if "reward_components" in info:
            list_infos[0]["reward_components"] = info["reward_components"]

        return obs.cpu().numpy(), rewards.cpu().numpy(), dones, list_infos

    def close(self):
        if hasattr(self.env, "close"): self.env.close()
    def get_attr(self, name, indices=None): return [getattr(self.env, name, None)] * self.num_envs
    def set_attr(self, name, value, indices=None): setattr(self.env, name, value)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [getattr(self.env, method_name)(*method_args, **method_kwargs)] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs


class AdaptiveKLCallback(BaseCallback):
    def __init__(self, target_kl: float = 0.015, min_lr: float = 1e-5, max_lr: float = 1e-3, verbose=0):
        super().__init__(verbose)
        self.target_kl = target_kl
        self.min_lr = min_lr
        self.max_lr = max_lr

    def _on_step(self) -> bool: return True

    def _on_rollout_end(self):
        approx_kl = self.logger.name_to_value.get("train/approx_kl")
        if approx_kl is not None:
            current_lr = self.model.learning_rate
            new_lr = current_lr
            if approx_kl > self.target_kl * 1.5:
                new_lr = max(current_lr / 1.5, self.min_lr)
            elif approx_kl < self.target_kl / 1.5:
                new_lr = min(current_lr * 1.5, self.max_lr)
            if new_lr != current_lr:
                self.model.lr_schedule = lambda _: new_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = new_lr
                self.model.learning_rate = new_lr


class IsaacTelemetryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "reward_components" in infos[0]:
            for key, val in infos[0]["reward_components"].items():
                self.logger.record(f"rewards/{key}", val)
            
            t = infos[0].get("telemetry", {})
            self.logger.record("nav/mean_distance", t.get("mean_distance", 0))
            self.logger.record("nav/R_Progress_Raw", t.get("R_Progress", 0))
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "telemetry" in infos[0]:
            t = infos[0]["telemetry"]
            print(f"\n🚀 [Rollout {self.rollout_count}] "
                  f"距終點: {t.get('mean_distance', 0):.1f} m | "
                  f"趨近速度: {t.get('R_Progress', 0):.2f} m/s | "
                  f"LR: {self.model.learning_rate:.2e}")


def main():
    set_random_seed(42)
    log_dir = f"./logs/ppo_nav_622_v2_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    env_cfg = Task3Config()
    env_cfg.num_envs = 4096 # 使用 4096 
    print(f"\n[INFO] 正在初始化 {env_cfg.num_envs} 個工業級導航環境...")
    
    base_env = Task3VisualNavEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(base_env)
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

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
        n_steps=32,            
        batch_size=32768,       
        n_epochs=5,             
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,         
        clip_range=0.2,
        max_grad_norm=1.0,      
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cuda:0",
        verbose=1
    )

    model = PPO(**ppo_kwargs)


    checkpoint_callback = CheckpointCallback(save_freq=2500, save_path=log_dir, name_prefix="nav_622_final")
    kl_callback = AdaptiveKLCallback(target_kl=0.015)
    telemetry_callback = IsaacTelemetryCallback()

    total_timesteps = 1000_000_000 
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[telemetry_callback, checkpoint_callback, kl_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[WARN] 接收到中断，正在安全保存...")
    finally:
        # 保存最終成果
        model.save(os.path.join(log_dir, "final_model_622.zip"))
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        print(f"🎉 [SUCCESS] 训练结束，数据存储在: {log_dir}")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()