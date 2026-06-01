# 🐕 基于 NVIDIA Isaac Lab 的 Unitree Go2 四足机器狗强化学习控制项目
 
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-brightgreen)
![skrl](https://img.shields.io/badge/RL-skrl%20PPO-purple)
![OS](https://img.shields.io/badge/OS-Ubuntu%20%7C%20Windows-green)
 
本项目是一个基于 NVIDIA Isaac Lab 的 Unitree Go2 四足机器人强化学习训练项目。项目包含 4 个递进任务：平地速度跟踪、多地形运动、自主导航与避障、Sim2Real / RMA 抗扰训练。
 
这个仓库最开始是我在学习强化学习时做的个人 demo。后来我对代码进行了重新整理和工程化重构：统一了项目目录结构，统一采用 `skrl` 的 PPO 训练流程，增加了环境测试、世界模型测试、模型测试、Ubuntu / Windows 脚本、训练进度条、日志与 checkpoint 管理。希望这个项目能为同样在学习 Isaac Lab、四足机器人控制和强化学习的同学提供一个可参考、可复现、可继续修改的基础工程。
项目重点不是追求完美训练，而是把每个任务从环境、测试、训练到评估尽量拆清楚。代码中仍然会有可以继续改进的地方，欢迎大家根据自己的 Isaac Lab 版本、显卡配置和研究目标继续修改。
 
---
 
## 🎬 训练效果展示
 
| Scene | Preview |
|---|---|
| 平地 / 多地形运动 | ![Go2 locomotion demo](assets/gifs/go2_locomotion_demo.gif) |
| 导航 / 避障 / 抗扰 | ![Go2 navigation demo](assets/gifs/go2_navigation_demo.gif) |
 
---
 
## ✨ 项目特点
 
- 基于 NVIDIA Isaac Lab 和 Unitree Go2 机器人资产。
- 包含 4 个递进任务，从基础平地运动到复杂地形、导航避障和 Sim2Real 抗扰训练。
- 所有任务统一使用 `skrl` PPO 训练框架。
- 每个任务提供独立的环境测试、训练脚本和模型测试脚本。
- Task2 / Task3 将 world 逻辑与 IsaacLab 物理环境分离，方便单独测试地形、障碍物、雷达和课程逻辑。
- 支持 Ubuntu / Windows 本地开发、测试和训练。
- 训练采用 `tqdm` 进度条，方便查看实时进度和日志信息。
 
---
 
## 📁 项目结构
 
```text
unitree_go2_isaaclab_rl/
├── configs/
│   ├── task1_locomotion.yaml
│   ├── task2_terrain.yaml
│   ├── task3_navigation.yaml
│   └── task4_sim2real_rma.yaml
├── src/
│   └── go2_rl/
│       ├── common/
│       │   ├── go2_skrl_models.py
│       │   ├── info_utils.py
│       │   └── paths.py
│       └── tasks/
│           ├── task1/
│           │   ├── task1_config.py
│           │   ├── task1_env.py
│           │   ├── task1_train.py
│           │   └── task1_model_test.py
│           ├── task2/
│           │   ├── task2_config.py
│           │   ├── task2_world.py
│           │   ├── task2_env.py
│           │   ├── task2_train.py
│           │   └── task2_model_test.py
│           ├── task3/
│           │   ├── task3_config.py
│           │   ├── task3_world.py
│           │   ├── task3_env.py
│           │   ├── task3_train.py
│           │   └── task3_model_test.py
│           └── task4/
│               ├── task4_config.py
│               ├── task4_env.py
│               ├── task4_train.py
│               └── task4_model_test.py
├── tests/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── scripts/
│   ├── ubuntu/
│   └── windows/
├── logs/
├── assets/
│   ├── gifs/
│   └── images/
├── LICENSE
└── README.md
```
 
| 目录 | 说明 |
|---|---|
| `configs/` | 每个任务的配置文件，便于统一管理任务参数。 |
| `src/go2_rl/common/` | 通用网络模型、日志工具、路径工具等。 |
| `src/go2_rl/tasks/taskX/` | 每个任务的环境、世界模型、训练脚本和模型测试脚本。 |
| `tests/` | 环境测试和世界模型测试脚本。 |
| `scripts/ubuntu/` | Ubuntu 下的测试、训练、评估脚本。 |
| `scripts/windows/` | Windows / RTX 3090 下的准备检查、训练、评估脚本。 |
| `logs/` | 默认训练日志和 checkpoint 输出目录。 |
| `assets/` | README 图片、GIF 和其他展示素材。 |
 
---
 
## 🛠️ 建议硬件与系统配置
 
### 最低测试配置
 
用于环境测试、world 测试、smoke training 和低并发调试：
 
- Ubuntu 22.04 / 24.04
- NVIDIA GPU，显存 16GB 左右
- Python 3.11
- PyTorch 2.x
- Isaac Sim / Isaac Lab
- `skrl`, `tensorboard`, `tqdm`
 
在 16GB 显存设备上，建议从小并发开始：
 
```bash
--num-envs 16
--num-envs 32
--num-envs 64
--num-envs 128
```
 
### 推荐训练配置
 
用于较大规模训练和长时间实验：
 
- NVIDIA RTX 3090 / 4090 或同级别 GPU
- 显存 24GB 或更高
- Windows 或 Ubuntu 均可，但需要保证 Isaac Lab 环境可正常运行
 
较大显存设备可以尝试：
 
```bash
--num-envs 512
--num-envs 1024
--num-envs 2048
```
 
具体并发数需要根据任务复杂度、显存占用和 Isaac Lab 版本调整。不要一开始直接使用最大并发，建议先运行 smoke training。
 
---
 
## 🚀 基础准备
 
### 1. 安装 Isaac Lab 环境
 
请先按照 NVIDIA Isaac Lab 官方文档安装 Isaac Sim / Isaac Lab，并确认 Isaac Lab 的 Python 环境可以正常导入：
 
```bash
python -c "import isaaclab; print('isaaclab ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
 
### 2. 克隆项目
 
```bash
git clone <your-repo-url> unitree_go2_isaaclab_rl
cd unitree_go2_isaaclab_rl
```
 
如果你保留旧仓库名，也可以直接进入对应目录，只需要保证项目结构与 README 中的结构一致。
 
### 3. 设置 PYTHONPATH
 
```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```
 
也可以直接使用 `scripts/ubuntu/` 下的脚本，这些脚本会自动设置项目路径。
 
### 4. 安装 Python 依赖
 
在 Isaac Lab 对应的 Python 环境中安装必要依赖：
 
```bash
pip install skrl tensorboard tqdm numpy
```
 
如果你的 Isaac Lab 安装方式已经包含部分依赖，可以按需跳过。
 
---
 
## ⚡ 快速开始
 
### 1. 环境测试
 
建议先从 Task1 开始测试，再进入后续任务。
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/test_task2_world.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/test_task4_env.sh
```
 
如果显存不足，可以打开对应脚本，降低 `--num-envs`。
 
### 2. Smoke 训练
 
Smoke training 用于确认训练管线可以启动、日志可以写入、checkpoint 可以保存，不用于评估最终效果。
 
```bash
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
```
 
### 3. 模型测试
 
训练完成后，可以使用 eval 脚本加载 checkpoint 做推理测试。
 
```bash
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt 0.12
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt 0.30
```
 
---
 
## 🧩 任务设计总览
 
| Task | 目标 | 环境特点 | 训练重点 | 主要脚本 |
|---|---|---|---|---|
| Task1 | 平地速度跟踪 | 平坦地面、随机速度指令 | 基础站立、行走、小跑、速度跟踪 | `task1_env.py`, `task1_train.py`, `task1_model_test.py` |
| Task2 | 多地形运动 | rough flat / slopes / stepping stones / stairs | 本体感知下的多地形 locomotion | `task2_world.py`, `task2_env.py`, `task2_train.py` |
| Task3 | 自主导航与避障 | 虚拟目标、静态/动态障碍、90 维 lidar | 到达目标、避障、保持运动稳定 | `task3_world.py`, `task3_env.py`, `task3_train.py` |
| Task4 | Sim2Real / RMA 抗扰训练 | 摩擦、负载、重心偏移、电机衰减、外力扰动 | 鲁棒运动和 RMA Teacher 训练 | `task4_env.py`, `task4_train.py`, `task4_model_test.py` |
 
---
 
## ➡️ Task 1：平地速度跟踪
 
Task1 是最基础的 locomotion 任务，用于训练 Unitree Go2 在平地上稳定站立、前进和跟踪速度指令。
 
### 任务目标
 
- 机器狗在平坦地面上保持稳定姿态。
- 跟踪随机给定的线速度和角速度指令。
- 学习基础步态，为 Task2 / Task3 / Task4 提供可 warm-start 的 checkpoint。
 
### 环境设计
 
- 使用 Isaac Lab 中的 Unitree Go2 资产。
- 控制频率采用低频 RL 策略 + 高频 PD 控制。
- 动作输出为 12 个关节的目标位置残差。
- 观测包含机身速度、角速度、重力投影、关节状态、历史动作、足端接触等信息。
- 训练代码统一采用 `skrl` PPO。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task1_skrl_laptop.sh
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
```
 
### 训练时重点观察
 
- `Actual_Vx` 是否逐步接近 `Cmd_Vx`
- `Base_Height` 是否稳定在目标高度附近
- `Fall_Rate` 是否接近 0
- `Contact_Count` 是否处在合理范围
- `P_Foot_Slip` 是否过大
- PPO 的 `approx_kl`、`clip_fraction` 是否稳定
 
---
 
## ➡️ Task 2：多地形运动
 
Task2 在 Task1 的基础上加入多地形训练，用于提升 Go2 在不同地形上的通过能力。
 
### 任务目标
 
- 在多种地形上保持稳定运动。
- 训练机器狗通过 rough flat、slopes、stepping stones、stairs 等地形。
- 使用课程学习逐步提升地形难度。
- 支持从 Task1 checkpoint warm-start。
 
### 环境设计
 
Task2 将地形和环境拆开：
 
- `task2_world.py`：负责地形类型、地形等级、课程逻辑、地形高度采样和 privileged terrain features。
- `task2_env.py`：负责真实 Go2 物理控制、观测、奖励、终止条件和与 IsaacLab 的交互。
- `task2_world_test.py`：不启动完整训练，用于检查地形世界逻辑。
- `task2_env_test.py`：用于检查 IsaacLab 环境和观测、奖励、reset、contact 等逻辑。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task2_world.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_laptop.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
```
 
### 训练时重点观察
 
- `Actual_Vx` 与 `Cmd_Vx` 的差距
- `Fall_Rate`
- `Mean_Terrain_Level`
- `Success_Total` / `Upgrade_Total`
- `Contact_Count`
- `P_Foot_Slip`
- 不同地形类型上的稳定性
 
---
 
## ➡️ Task 3：自主导航与避障
 
Task3 在真实 Go2 物理控制的基础上加入解析导航世界。机器狗需要根据目标点、lidar 和风险特征，在存在静态/动态障碍物的环境中到达目标。
 
### 任务目标
 
- 根据虚拟目标点进行自主导航。
- 使用 90 维 lidar 与 risk features 感知障碍物。
- 避免静态和动态障碍物碰撞。
- 在保持身体稳定的同时尽量向目标前进。
 
### 环境设计
 
Task3 使用“真实机器人物理 + 解析导航世界”的结构：
 
- `task3_world.py` 是纯 torch 解析世界，不依赖 IsaacLab。
- 静态障碍、动态障碍、lidar、碰撞、目标点、risk features 都由 GPU tensor 计算。
- `task3_env.py` 接入 IsaacLab 的 Go2 物理环境，动作仍然控制真实 Go2 关节。
- 障碍物不生成大量真实 prim，这样可以保持较高并发和较低仿真开销。
 
### 观测结构
 
- 单帧 actor observation：208 维
- 5 帧堆叠后 actor input：1040 维
- world privileged features：68 维
- critic input：1108 维
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_laptop.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt 0.12
```
 
### 训练时重点观察
 
- `Progress`
- `Distance_To_Goal`
- `Success_Rate`
- `Collision_Rate`
- `Fall_Rate`
- `Collision_Risk`
- `Front_Clearance_Norm`
- `Actual_Along_Goal`
 
Task3 训练难度比 Task1 / Task2 更高，建议优先使用 Task1 或 Task2 checkpoint warm-start。
 
---
 
## ➡️ Task 4：Sim2Real / RMA 抗扰训练
 
Task4 面向 Sim2Real 和鲁棒运动控制。当前实现的是 RMA Teacher 阶段，用于训练带 privileged information 的 teacher policy。后续可以继续扩展 Student / adaptation module。
 
### 任务目标
 
- 在摩擦变化、负载变化、重心偏移、电机输出衰减和外部扰动下保持运动稳定。
- 训练一个使用 privileged information 的 Teacher policy。
- 为后续 Student 模仿学习、RMA adaptation module 和真机部署做准备。
 
### 环境设计
 
Task4 当前采用 teacher 训练结构：
 
```text
teacher_obs = actor_history + privileged_obs
teacher_obs = 240 + 25 = 265
```
 
其中：
 
- `actor_history`：5 帧历史观测，每帧 48 维，总共 240 维。
- `privileged_obs`：25 维，包括摩擦、负载、重心偏移、电机强度、外力扰动等信息。
- Teacher policy 可以使用 privileged information。
- Student policy 和 adaptation module 属于后续扩展方向。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task4_env.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_laptop.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt 0.30
```
 
### 训练时重点观察
 
- `Cmd_Vx` / `Actual_Vx`
- `Tracking_Error`
- `Fall_Rate`
- `Push_Active_Rate`
- `Motor_Strength_Min`
- `Payload_Mass`
- `Friction`
- `Base_Height`
 
---
 
## 📊 日志与模型保存
 
训练日志默认保存在：
 
```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```
 
每个训练 run 通常包含：
 
```text
checkpoint_<env_steps>/
final_checkpoint/
train_metadata.pt
```
 
可以使用 TensorBoard 查看训练过程：
 
```bash
tensorboard --logdir logs
```
 
训练过程中会记录以下类型的信息：
 
- `reward_components`：各奖励项。
- `events`：成功、摔倒、碰撞、超时等事件。
- `telemetry`：速度、高度、距离、课程阶段等训练指标。
- `debug`：观测维度、reward 范围、异常值检查等。
- `ppo`：PPO 更新信息，例如 KL、loss、学习率等。
 
---
 
## 💻 Ubuntu / Windows 使用说明
 
### Ubuntu
 
Ubuntu 用于：
 
- 代码开发
- 环境测试
- world 测试
- smoke training
- 训练验证
 
常用脚本在：
 
```text
scripts/ubuntu/
```
 
### Windows 
 
Windows 脚本在：
 
```text
scripts/windows/
```
 
建议先运行 readiness check：
 
```powershell
.\scripts\windows\check_task1_windows_ready.ps1
.\scripts\windows\check_task2_windows_ready.ps1
.\scripts\windows\check_task3_windows_ready.ps1
.\scripts\windows\check_task4_windows_ready.ps1
```
 
Windows 训练脚本通常带有审批环境变量，避免误启动长时间训练。例如：
 
```powershell
$env:GO2_TASK3_WINDOWS_SMOKE_APPROVED = "1"
.\scripts\windows\train_task3_skrl_smoke_3090.ps1
```
 
正式训练前建议先运行 smoke 版本，确认路径、IsaacLab Python、显卡和日志输出都正常。
 
---
 
## 🧭 推荐训练顺序
 
推荐顺序：
 
1. 先训练 Task1，获得基础平地 locomotion checkpoint。
2. Task2 从 Task1 warm-start，训练多地形运动。
3. Task3 从 Task1 或 Task2 warm-start，训练导航与避障。
4. Task4 从 Task2 warm-start，训练 Sim2Real / RMA Teacher。
 
也可以每个任务从零开始训练，但训练时间会更长，早期调参也会更困难。
 
---
 
## 📌 当前状态与限制
 
- 本项目主要用于学习、复现实验和开源交流。
- 当前代码完成了四个任务的 IsaacLab 环境、测试、`skrl` PPO 训练和模型测试脚本。
- Task4 当前实现的是 RMA Teacher 训练阶段，Student 蒸馏和 adaptation module 还需要后续继续扩展。
- 不同 Isaac Lab / Isaac Sim 版本之间可能存在 API 差异，需要根据本地环境做少量适配。
- 训练效果会受到 GPU、并发数、随机种子、训练步数和超参数影响。
- Windows 脚本中的默认路径可能需要根据自己的机器修改。
- 本项目不是官方 Unitree 或 NVIDIA 项目，只是个人学习和开源整理。
 
---
 
## ❓ 常见问题
 
### 1. `ModuleNotFoundError: No module named torch`
 
通常是没有进入 Isaac Lab 对应的 Python / conda 环境。请先确认：
 
```bash
which python
python -c "import torch; print(torch.__version__)"
```
 
### 2. IsaacLab / `pxr` 导入报错
 
涉及 IsaacLab、USD、`pxr` 的文件需要在 Isaac Sim / Isaac Lab 环境中运行。测试脚本中如果需要 AppLauncher，应保证先启动 AppLauncher，再导入依赖 IsaacLab 的环境文件。
 
### 3. 训练启动后显存不足怎么办?
 
先降低并发数：
 
```bash
--num-envs 16
--num-envs 32
--num-envs 64
--num-envs 128
```
 
确认能跑通后再逐步增加。
 
### 4. Smoke training
 
正常。Smoke training 只用于检查训练流程是否能启动和保存模型，不代表最终策略效果。
 
### 5. Task3 / Task4 为什么推荐 warm-start?
 
Task3 加入导航和障碍物，Task4 加入扰动和域随机化，直接从零训练会更难。使用 Task1 / Task2 checkpoint 可以先继承基础步态，再学习更复杂的任务。
 
### 6. Windows 路径需要怎么改?
 
打开 `scripts/windows/` 下的 `.ps1` 文件，修改：
 
```powershell
$ProjectRoot
$IsaacLabRoot
$LogRoot
```
 
确保它们对应你本机的项目路径、IsaacLab 路径和日志路径。
 
### 7. 为什么要先跑环境测试?
 
四足机器人训练中的很多问题不是 PPO 本身造成的，而是 reset、观测维度、坐标系、接触传感器、奖励项或终止条件有问题。先跑测试可以减少后续训练调参的时间。
 
---
 
## 📄 License
 
This project is released under the MIT License.
 
See the `LICENSE` file for details.
 
---
 
## 🙏 Acknowledgements
 
感谢以下开源项目和工具：
 
- NVIDIA Isaac Sim / Isaac Lab
- Unitree Go2 robot asset in Isaac Lab
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- 机器人强化学习和 Isaac Lab 开源社区
 
如果这个项目对你有帮助，欢迎参考、修改和继续完善。也欢迎指出代码或文档中的问题。
联系邮箱：2559906288@qq.com 小红书账号：574661219

