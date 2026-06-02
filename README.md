# 🐕 基于 NVIDIA Isaac Lab 的 Unitree Go2 四足机器狗强化学习控制项目

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-brightgreen)
![skrl](https://img.shields.io/badge/RL-skrl%20PPO-purple)
![OS](https://img.shields.io/badge/OS-Ubuntu%20%7C%20Windows-green)

本项目是一个基于 NVIDIA Isaac Lab 的 Unitree Go2 四足机器人强化学习训练工程。Isaac Lab 基于 Isaac Sim，提供高并发物理仿真、机器人资产、传感器接口和强化学习任务构建能力；Unitree Go2 是 12 自由度四足机器人，适合作为运动控制、导航避障和 Sim2Real 研究对象。项目包含 4 个递进任务：平地速度跟踪、多地形运动、自主导航与避障、Sim2Real / RMA 抗扰训练。

本仓库重点关注完整训练链路的可复现组织方式：从任务配置、环境构建、世界模型、测试脚本、训练入口、评估入口到日志与 checkpoint 管理，尽量将每个模块的职责拆清楚。项目不追求单一固定最优策略，而是提供一个可继续扩展的 Isaac Lab 四足机器人强化学习工程基础，便于开展 locomotion、terrain curriculum、navigation、obstacle avoidance、domain randomization 和 RMA teacher policy 等方向实验。

---

## 🎬 训练效果展示

| Scene | Preview |
|---|---|
| 平地 / 多地形运动 | 可将训练录制的 GIF 放入 `assets/gifs/` 后在此处展示。 |
| 导航 / 避障 / 抗扰 | 可将评估录制的 GIF 放入 `assets/gifs/` 后在此处展示。 |

---

## ✨ 项目特点

- **基于 NVIDIA Isaac Lab 与 Unitree Go2 资产。** 项目使用 Isaac Lab 的环境管理、资产加载、并行仿真和 AppLauncher 入口，围绕 Unitree Go2 的 12 个执行关节构建四足机器人强化学习任务，便于在统一仿真框架下完成测试、训练和评估。
- **包含 4 个递进任务。** Task1 训练平地速度跟踪，Task2 加入多地形与课程学习，Task3 加入目标导航、静态 / 动态障碍物和 lidar 风险感知，Task4 面向 Sim2Real / RMA teacher 阶段，加入摩擦、负载、重心、电机强度和外力扰动。
- **统一使用 `skrl` PPO 训练流程。** Actor / Critic 模型、normalizer 保存加载、checkpoint 选择、训练元数据、模型评估兼容逻辑集中放在 `src/go2_rl/common/` 中，四个任务共享相似的训练与评估组织方式。
- **模块化工程框架。** 每个任务独立维护 `config / env / train / model_test`，Task2 和 Task3 额外提供 `world` 模块，将 terrain、curriculum、navigation、obstacle、lidar 等逻辑从 IsaacLab 环境中分离，方便白盒测试和后续扩展。
- **完整的测试入口。** Ubuntu 端提供 world 测试、env 测试和 smoke training 脚本，覆盖地形逻辑、解析世界逻辑、Gymnasium `reset()` / `step()` API、观测维度、奖励输出、终止条件和训练链路启动。
- **清晰的日志与进度输出。** 训练过程中使用进度条展示 env steps、FPS、reward、阶段、速度、高度、接触、摔倒率等关键指标；日志结构按任务划分，checkpoint、normalizer、训练元数据和 TensorBoard 数据集中保存，便于定位训练状态。
- **支持参数化运行。** 常用脚本支持通过命令行参数调整并发数、训练步数、checkpoint 路径、headless / GUI 等运行选项；配置文件放在 `configs/` 中，脚本内公共路径逻辑放在 `_common` 文件中，便于在不同机器上迁移。
- **Ubuntu / Windows 双系统脚本入口。** 两个系统都提供 check、test、smoke、train、eval、visualize 等控制脚本，脚本命名保持一致，便于在不同系统下使用相同的任务流程。
- **预留未来扩展接口。** Task4 当前实现 RMA teacher 训练结构，后续可继续扩展 student policy、adaptation module、真机部署接口；Task3 的解析 world 也可继续扩展更复杂障碍物、地图和任务阶段。

---

## 📁 项目结构

```text
unitree_go2_isaaclab_rl/
├── assets/
│   ├── gifs/
│   │   └── README.md
│   ├── motions/
│   │   └── README.md
│   ├── usd/
│   │   └── README.md
│   └── README.md
├── configs/
│   ├── .gitkeep
│   ├── local_paths.example.yaml
│   ├── platform_ubuntu.example.yaml
│   ├── platform_windows.example.yaml
│   ├── task1_flat_locomotion.yaml
│   ├── task2_multiterrain.yaml
│   ├── task3_navigation.yaml
│   └── task4_sim2real_rma.yaml
├── docs/
│   ├── project_overview.md
│   ├── results_and_checkpoints.md
│   ├── task1_design.md
│   ├── task2_design.md
│   ├── task3_design.md
│   ├── task4_design.md
│   ├── troubleshooting.md
│   ├── ubuntu_training.md
│   ├── ubuntu_validation.md
│   ├── windows_path_config.md
│   └── windows_training.md
├── scripts/
│   ├── ubuntu/
│   │   ├── _common.sh
│   │   ├── check_env.sh
│   │   ├── test_task1_env.sh
│   │   ├── test_task2_world.sh
│   │   ├── test_task2_env.sh
│   │   ├── test_task3_world.sh
│   │   ├── test_task3_env.sh
│   │   ├── test_task4_env.sh
│   │   ├── smoke_task1.sh
│   │   ├── smoke_task2.sh
│   │   ├── smoke_task3.sh
│   │   ├── smoke_task4.sh
│   │   ├── train_task1.sh
│   │   ├── train_task2.sh
│   │   ├── train_task3.sh
│   │   ├── train_task4.sh
│   │   ├── eval_task1.sh
│   │   ├── eval_task2.sh
│   │   ├── eval_task3.sh
│   │   ├── eval_task4.sh
│   │   ├── visualize_task1.sh
│   │   ├── visualize_task2.sh
│   │   ├── visualize_task3.sh
│   │   └── visualize_task4.sh
│   └── windows/
│       ├── _common.ps1
│       ├── check_env.ps1
│       ├── check_task1.ps1
│       ├── check_task2.ps1
│       ├── check_task3.ps1
│       ├── check_task4.ps1
│       ├── smoke_task1.ps1
│       ├── smoke_task2.ps1
│       ├── smoke_task3.ps1
│       ├── smoke_task4.ps1
│       ├── train_task1.ps1
│       ├── train_task2.ps1
│       ├── train_task3.ps1
│       ├── train_task4.ps1
│       ├── eval_task1.ps1
│       ├── eval_task2.ps1
│       ├── eval_task3.ps1
│       ├── eval_task4.ps1
│       ├── visualize_task1.ps1
│       ├── visualize_task2.ps1
│       ├── visualize_task3.ps1
│       └── visualize_task4.ps1
├── src/
│   └── go2_rl/
│       ├── __init__.py
│       ├── common/
│       │   ├── __init__.py
│       │   ├── checkpoint_utils.py
│       │   ├── eval_curriculum_utils.py
│       │   ├── go2_skrl_models.py
│       │   ├── go2_skrl_wrappers.py
│       │   ├── info_utils.py
│       │   ├── model_eval_utils.py
│       │   ├── normalizer_utils.py
│       │   ├── paths.py
│       │   ├── progress.py
│       │   └── train_metadata.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── README.md
│       └── tasks/
│           ├── __init__.py
│           ├── task1/
│           │   ├── __init__.py
│           │   ├── task1_config.py
│           │   ├── task1_env.py
│           │   ├── task1_train.py
│           │   └── task1_model_test.py
│           ├── task2/
│           │   ├── __init__.py
│           │   ├── task2_config.py
│           │   ├── task2_world.py
│           │   ├── task2_env.py
│           │   ├── task2_train.py
│           │   └── task2_model_test.py
│           ├── task3/
│           │   ├── __init__.py
│           │   ├── task3_config.py
│           │   ├── task3_world.py
│           │   ├── task3_env.py
│           │   ├── task3_train.py
│           │   └── task3_model_test.py
│           └── task4/
│               ├── __init__.py
│               ├── task4_config.py
│               ├── task4_env.py
│               ├── task4_train.py
│               └── task4_model_test.py
├── tests/
│   ├── __init__.py
│   ├── task1/
│   │   ├── __init__.py
│   │   └── task1_env_test.py
│   ├── task2/
│   │   ├── __init__.py
│   │   ├── task2_world_test.py
│   │   └── task2_env_test.py
│   ├── task3/
│   │   ├── __init__.py
│   │   ├── task3_world_test.py
│   │   └── task3_env_test.py
│   └── task4/
│       ├── __init__.py
│       └── task4_env_test.py
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

| 目录 / 文件 | 说明 |
|---|---|
| `assets/` | 展示素材目录，包含 GIF、运动文件、USD 资源说明。 |
| `configs/` | 任务配置和平台示例配置。`local_paths.example.yaml` 用于说明本地路径写法，`platform_*.example.yaml` 用于说明 Ubuntu / Windows 平台配置。 |
| `docs/` | 专题文档，包括项目概览、任务设计、训练说明、测试记录、路径配置和故障排查。 |
| `scripts/ubuntu/` | Ubuntu 下的环境检查、world/env 测试、smoke training、正式训练、模型评估和可视化脚本。 |
| `scripts/windows/` | Windows 下的环境检查、任务检查、smoke training、正式训练、模型评估和可视化脚本。 |
| `src/go2_rl/common/` | 公共工具模块，包括 skrl 模型、frame stack wrapper、checkpoint、normalizer、日志进度、路径解析和训练元数据。 |
| `src/go2_rl/tasks/` | 四个任务的核心代码。每个任务独立维护 config、env、train、model_test；Task2 / Task3 额外维护 world。 |
| `tests/` | 运行测试目录。Task2 / Task3 包含 world 白盒测试，Task1 ~ Task4 包含环境测试。 |
| `logs/` | 默认训练输出目录，由训练脚本运行时自动生成，不作为源码文件提交。 |

---

## 🛠️ 建议硬件与系统配置

### 基础运行配置

用于环境检查、world 测试、env 测试、smoke training 和低并发调试：

- 操作系统：Ubuntu 22.04 / 24.04，或 Windows 11；
- Python：3.11；
- GPU：支持 RTX 的 NVIDIA GPU；
- 显存：建议 16GB 或以上；
- 内存：建议 32GB 或以上；
- 存储：建议使用 SSD，并为 Isaac Sim / Isaac Lab、缓存和训练日志预留足够空间；
- 软件：NVIDIA 驱动、Isaac Sim、Isaac Lab、PyTorch、skrl、TensorBoard、tqdm。

### 训练配置建议

用于较大并发训练、长时间实验和复杂可视化：

- 更高显存和内存会显著提升并行环境数量和训练稳定性；
- headless 训练通常比 GUI 可视化占用更少资源；
- rendering、传感器数量、并行环境数量、地形复杂度都会影响显存占用；
- 初次运行建议先使用 `smoke_task*.sh` / `smoke_task*.ps1`，再逐步提高 `num_envs` 和训练步数。

常用并发参考：

```bash
--num-envs 8
--num-envs 16
--num-envs 32
--num-envs 64
--num-envs 128
--num-envs 512
```

并发数应根据显存、任务复杂度和 Isaac Lab 版本调整。world 测试和 smoke training 通过后，再进入正式训练。

---

## 🚀 基础准备

### 1. 安装 Isaac Sim / Isaac Lab

先按照 NVIDIA Isaac Sim 与 Isaac Lab 官方文档安装仿真环境。安装完成后，进入 Isaac Lab 对应的 conda 环境或 Python 环境，并检查 Python、PyTorch、CUDA、IsaacLab 是否可用。

Ubuntu 示例：

```bash
conda activate isaaclab

which python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import isaaclab; print('isaaclab ok')"
```

Windows PowerShell 示例：

```powershell
conda activate isaaclab

Get-Command python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import isaaclab; print('isaaclab ok')"
```

如果 `torch`、`isaaclab` 或 `pxr` 导入失败，通常说明当前 Python 环境不是 Isaac Lab 环境。

### 2. 克隆项目并进入工程目录

```bash
git clone <your-repo-url> unitree_go2_isaaclab_rl
cd unitree_go2_isaaclab_rl
```

确认当前目录是项目根目录：

```bash
pwd
git rev-parse --show-toplevel
ls
```

应能看到：

```text
assets  configs  docs  scripts  src  tests  README.md  pyproject.toml
```

### 3. 理解运行入口

本项目建议通过 `scripts/` 下的脚本运行。脚本会设置项目根目录、`PYTHONPATH`、日志目录和 Python 入口，减少手动配置错误。

Ubuntu 常用入口：

```text
scripts/ubuntu/check_env.sh
scripts/ubuntu/test_task*.sh
scripts/ubuntu/smoke_task*.sh
scripts/ubuntu/train_task*.sh
scripts/ubuntu/eval_task*.sh
scripts/ubuntu/visualize_task*.sh
```

Windows 常用入口：

```text
scripts/windows/check_env.ps1
scripts/windows/check_task*.ps1
scripts/windows/smoke_task*.ps1
scripts/windows/train_task*.ps1
scripts/windows/eval_task*.ps1
scripts/windows/visualize_task*.ps1
```

如果在 VS Code、PyCharm 等编辑器中运行 Python 文件，需要确认三项设置：

```text
Working Directory = 项目根目录
Python Interpreter = Isaac Lab 对应的 Python
Environment Variable: PYTHONPATH = <项目根目录>/src
```

Ubuntu 中可用下面命令查看项目路径：

```bash
cd /path/to/unitree_go2_isaaclab_rl
pwd
realpath .
```

Windows PowerShell 中可用下面命令查看项目路径：

```powershell
cd C:\path\to\unitree_go2_isaaclab_rl
Get-Location
Resolve-Path .
```

### 4. 设置 `PYTHONPATH`

使用 `scripts/ubuntu/` 和 `scripts/windows/` 脚本时，通常不需要手动设置 `PYTHONPATH`。如果直接运行 Python 入口，需要手动设置。

Ubuntu：

```bash
cd /path/to/unitree_go2_isaaclab_rl
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Windows PowerShell：

```powershell
cd C:\path\to\unitree_go2_isaaclab_rl
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

验证导入：

```bash
python -c "import go2_rl; print('go2_rl ok')"
```

### 5. 配置本地路径文件

项目提供示例路径文件，不直接提交个人绝对路径。可复制示例文件为本地私有配置：

Ubuntu：

```bash
cd /path/to/unitree_go2_isaaclab_rl

cp configs/local_paths.example.yaml configs/local_paths.yaml
nano configs/local_paths.yaml
```

Windows PowerShell：

```powershell
cd C:\path\to\unitree_go2_isaaclab_rl

Copy-Item configs\local_paths.example.yaml configs\local_paths.yaml
notepad configs\local_paths.yaml
```

`configs/local_paths.yaml` 建议填写：

```yaml
ubuntu:
  project_root: "/path/to/unitree_go2_isaaclab_rl"
  isaaclab_root: "/path/to/IsaacLab"
  log_root: "/path/to/unitree_go2_isaaclab_rl/logs"

windows:
  project_root: "C:\\path\\to\\unitree_go2_isaaclab_rl"
  isaaclab_root: "C:\\path\\to\\IsaacLab"
  isaac_python: "C:\\path\\to\\IsaacLab\\_isaac_sim\\python.bat"
  log_root: "C:\\path\\to\\unitree_go2_isaaclab_rl\\logs"
```

路径查找方法：

Ubuntu 查看项目路径：

```bash
cd /path/to/unitree_go2_isaaclab_rl
pwd
```

Ubuntu 查找 IsaacLab 目录：

```bash
find $HOME -maxdepth 4 -type d -name "IsaacLab" 2>/dev/null
```

Ubuntu 查看当前 Python：

```bash
which python
python -c "import sys; print(sys.executable)"
```

Windows 查看项目路径：

```powershell
Get-Location
Resolve-Path .
```

Windows 查找 IsaacLab 目录，可在常用代码目录下执行：

```powershell
Get-ChildItem -Path C:\ -Directory -Filter IsaacLab -Recurse -ErrorAction SilentlyContinue
```

Windows 查看当前 Python：

```powershell
Get-Command python
python -c "import sys; print(sys.executable)"
```

### 6. 平台配置示例

`configs/platform_ubuntu.example.yaml` 和 `configs/platform_windows.example.yaml` 是平台运行配置示例，可作为本机配置参考。

Ubuntu：

```bash
cp configs/platform_ubuntu.example.yaml configs/platform_ubuntu.local.yaml
nano configs/platform_ubuntu.local.yaml
```

Windows：

```powershell
Copy-Item configs\platform_windows.example.yaml configs\platform_windows.local.yaml
notepad configs\platform_windows.local.yaml
```

这些 `.local.yaml` 文件用于本机记录，不作为公共配置提交。

### 7. 安装 Python 依赖

在 Isaac Lab Python 环境中安装强化学习与日志相关依赖：

```bash
pip install skrl tensorboard tqdm numpy
```

如果 Isaac Lab 环境已经包含部分依赖，可以按需跳过。安装完成后运行：

```bash
python -c "import skrl; print('skrl ok')"
python -c "import tqdm; print('tqdm ok')"
```

### 8. 运行环境检查

Ubuntu：

```bash
bash scripts/ubuntu/check_env.sh
```

Windows PowerShell：

```powershell
.\scripts\windows\check_env.ps1
```

检查通过后，再进入 world 测试、env 测试和 smoke training。

---

## ⚡ 快速开始

### 1. Ubuntu 测试入口

先运行 world 和环境测试，确认基础逻辑、观测维度和 IsaacLab 环境构建正常。

```bash
bash scripts/ubuntu/test_task2_world.sh
bash scripts/ubuntu/test_task3_world.sh

bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/test_task4_env.sh
```

带参数示例：

```bash
bash scripts/ubuntu/test_task2_world.sh --num-envs 512
bash scripts/ubuntu/test_task3_world.sh --num-envs 512

bash scripts/ubuntu/test_task1_env.sh --num-envs 8 --steps 16
bash scripts/ubuntu/test_task2_env.sh --num-envs 8 --steps 16
bash scripts/ubuntu/test_task3_env.sh --num-envs 8 --steps 16
bash scripts/ubuntu/test_task4_env.sh --num-envs 8 --steps 16
```

### 2. Ubuntu smoke training

Smoke training 用于确认训练入口、日志写入、模型保存和 checkpoint 管理能正常工作，不用于代表最终策略效果。

```bash
bash scripts/ubuntu/smoke_task1.sh
bash scripts/ubuntu/smoke_task2.sh
bash scripts/ubuntu/smoke_task3.sh
bash scripts/ubuntu/smoke_task4.sh
```

### 3. Ubuntu 正式训练

Task1 从零开始训练：

```bash
bash scripts/ubuntu/train_task1.sh
```

带参数示例：

```bash
bash scripts/ubuntu/train_task1.sh --num-envs 512 --total-env-steps 100000000
```

Task2 可从 Task1 checkpoint warm-start：

```bash
bash scripts/ubuntu/train_task2.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
```

带参数示例：

```bash
bash scripts/ubuntu/train_task2.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt --num-envs 512 --total-env-steps 100000000
```

Task3 可从 Task1 或 Task2 checkpoint warm-start：

```bash
bash scripts/ubuntu/train_task3.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
```

带参数示例：

```bash
bash scripts/ubuntu/train_task3.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt --num-envs 512 --total-env-steps 100000000
```

Task4 可从 Task2 checkpoint warm-start：

```bash
bash scripts/ubuntu/train_task4.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
```

带参数示例：

```bash
bash scripts/ubuntu/train_task4.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt --num-envs 512 --total-env-steps 100000000
```

### 4. Ubuntu 模型评估

```bash
bash scripts/ubuntu/eval_task1.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
bash scripts/ubuntu/eval_task2.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/eval_task3.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt
bash scripts/ubuntu/eval_task4.sh logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt
```

### 5. Ubuntu GUI 可视化

```bash
bash scripts/ubuntu/visualize_task1.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
bash scripts/ubuntu/visualize_task2.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/visualize_task3.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt
bash scripts/ubuntu/visualize_task4.sh logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt
```

---

## 🧩 任务设计总览

| Task | 目标 | 环境特点 | 训练重点 | 主要脚本 |
|---|---|---|---|---|
| Task1 | 平地速度跟踪 | 平坦地面、随机速度指令 | 基础站立、行走、速度跟踪 | `task1_env.py`, `task1_train.py`, `task1_model_test.py` |
| Task2 | 多地形运动 | rough flat / slopes / stepping stones / stairs | 地形课程学习、多地形稳定运动 | `task2_world.py`, `task2_env.py`, `task2_train.py`, `task2_model_test.py` |
| Task3 | 自主导航与避障 | 虚拟目标、静态 / 动态障碍、60 维 lidar | 目标到达、避障、运动稳定性 | `task3_world.py`, `task3_env.py`, `task3_train.py`, `task3_model_test.py` |
| Task4 | Sim2Real / RMA 抗扰训练 | 摩擦、负载、重心、电机强度、外力扰动 | 鲁棒运动与 RMA Teacher 训练 | `task4_env.py`, `task4_train.py`, `task4_model_test.py` |

---

## ➡️ Task 1：平地速度跟踪

Task1 是基础 locomotion 任务，用于训练 Unitree Go2 在平地上保持稳定姿态，并跟踪随机速度指令。

### 任务目标

- 在平坦地面上保持稳定站立和运动；
- 跟踪随机给定的线速度和角速度指令；
- 学习基础步态，为 Task2 / Task3 / Task4 提供 warm-start checkpoint。

### 环境设计

- 使用 Isaac Lab 中的 Unitree Go2 资产；
- 动作维度为 12，对应 12 个关节目标位置残差；
- actor observation 为 87 维；
- 不使用 privileged observation；
- 控制结构采用低频 RL policy + 高频 PD control；
- 训练代码统一采用 `skrl` PPO。

### 常用命令

```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/smoke_task1.sh
bash scripts/ubuntu/train_task1.sh
bash scripts/ubuntu/eval_task1.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
bash scripts/ubuntu/visualize_task1.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
```

### 训练时重点观察

- `Actual_Vx` 是否逐步接近 `Cmd_Vx`；
- `Base_Height` 是否稳定在目标高度附近；
- `Fall_Rate` 是否接近 0；
- `Contact_Count` 是否处在合理范围；
- `P_Foot_Slip` 是否过大；
- PPO 的 `approx_kl`、`clip_fraction`、loss 是否稳定。

---

## ➡️ Task 2：多地形运动

Task2 在 Task1 的基础上加入多地形训练，用于提升 Go2 在不同地形上的通过能力。

### 任务目标

- 在 rough flat、slopes、stepping stones、stairs 等地形上保持稳定运动；
- 使用课程学习逐步提升地形难度；
- 支持从 Task1 checkpoint warm-start；
- 通过 terrain privileged features 辅助 Critic 学习地形相关价值估计。

### 环境设计

Task2 将地形世界和 IsaacLab 环境拆开：

- `task2_world.py`：负责地形类型、地形等级、课程逻辑、height scan 和 terrain privileged features；
- `task2_env.py`：负责 Go2 物理控制、观测、奖励、终止条件和 IsaacLab 交互；
- `task2_world_test.py`：用于检查地形世界逻辑；
- `task2_env_test.py`：用于检查 IsaacLab 环境、观测、奖励、reset 和 contact 逻辑。

### 观测结构

- actor observation：87 维；
- privileged observation：178 维；
- terrain privileged tail：91 维；
- action dim：12。

### 常用命令

```bash
bash scripts/ubuntu/test_task2_world.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/smoke_task2.sh
bash scripts/ubuntu/train_task2.sh logs/task1/<run_name>/final_checkpoint/go2_task1_model.pt
bash scripts/ubuntu/eval_task2.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/visualize_task2.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
```

### 训练时重点观察

- `Actual_Vx` 与 `Cmd_Vx` 的差距；
- `Fall_Rate`；
- `Mean_Terrain_Level`；
- `Success_Total` / `Upgrade_Total`；
- `Contact_Count`；
- `P_Foot_Slip`；
- 不同地形类型上的稳定性。

---

## ➡️ Task 3：自主导航与避障

Task3 在真实 Go2 物理控制基础上加入解析导航世界。机器狗需要根据目标点、lidar 和风险特征，在存在静态 / 动态障碍物的环境中到达目标。

### 任务目标

- 根据虚拟目标点进行自主导航；
- 使用 60 维 lidar 与 risk features 感知障碍物；
- 避免静态和动态障碍物碰撞；
- 在保持身体稳定的同时向目标前进。

### 环境设计

Task3 使用“真实机器人物理 + 解析导航世界”的结构：

- `task3_world.py` 是纯 torch 解析世界，负责目标、障碍物、lidar、碰撞、边界和风险特征；
- `task3_env.py` 接入 IsaacLab 的 Go2 物理环境，动作控制真实 Go2 关节；
- 障碍物不生成大量真实 prim，训练阶段主要使用 GPU tensor 计算，从而降低仿真开销；
- 评估脚本可根据需要显示目标和障碍物标记。

### 观测结构

- 单帧 actor observation：208 维；
- 5 帧堆叠后 actor input：1040 维；
- world privileged tail：68 维；
- critic input：1108 维；
- lidar rays：60；
- action dim：12。

### 常用命令

```bash
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/smoke_task3.sh
bash scripts/ubuntu/train_task3.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/eval_task3.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt
bash scripts/ubuntu/visualize_task3.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt
```

### 训练时重点观察

- `Progress`；
- `Distance_To_Goal`；
- `Success_Rate`；
- `Collision_Rate`；
- `Fall_Rate`；
- `Collision_Risk`；
- `Front_Clearance_Norm`；
- `Actual_Along_Goal`。

Task3 训练难度高于 Task1 / Task2，推荐优先使用 Task1 或 Task2 checkpoint warm-start。

---

## ➡️ Task 4：Sim2Real / RMA 抗扰训练

Task4 面向 Sim2Real 和鲁棒运动控制。当前实现 RMA Teacher 阶段，用于训练带 privileged information 的 teacher policy。后续可继续扩展 Student / adaptation module。

### 任务目标

- 在摩擦变化、负载变化、重心偏移、电机强度变化和外部扰动下保持运动稳定；
- 训练使用 privileged information 的 Teacher policy；
- 为后续 Student 模仿学习、RMA adaptation module 和真机部署做准备。

### 环境设计

Task4 当前采用 teacher 训练结构：

```text
teacher_obs = actor_history + privileged_obs
teacher_obs = 240 + 25 = 265
```

其中：

- `single actor obs`：48 维；
- `actor_history`：5 帧历史观测，总共 240 维；
- `privileged_obs`：25 维，包括摩擦、负载、重心偏移、电机强度、外力扰动等信息；
- `action dim`：12；
- Teacher policy 可以使用 privileged information；
- Student policy 和 adaptation module 属于后续扩展方向。

### 常用命令

```bash
bash scripts/ubuntu/test_task4_env.sh
bash scripts/ubuntu/smoke_task4.sh
bash scripts/ubuntu/train_task4.sh logs/task2/<run_name>/final_checkpoint/go2_task2_model.pt
bash scripts/ubuntu/eval_task4.sh logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt
bash scripts/ubuntu/visualize_task4.sh logs/task4/<run_name>/final_checkpoint/go2_task4_teacher_model.pt
```

### 训练时重点观察

- `Cmd_Vx` / `Actual_Vx`；
- `Tracking_Error`；
- `Fall_Rate`；
- `Push_Active_Rate`；
- `Motor_Strength_Min`；
- `Payload_Mass`；
- `Friction`；
- `Base_Height`。

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

- `reward_components`：各奖励项；
- `events`：成功、摔倒、碰撞、超时等事件；
- `telemetry`：速度、高度、距离、课程阶段等训练指标；
- `debug`：观测维度、reward 范围、异常值检查等；
- `ppo`：PPO 更新信息，例如 KL、loss、学习率等；
- `normalizer`：观测归一化统计量，用于训练恢复和模型评估。

---

## 💻 Ubuntu / Windows 使用说明

### Ubuntu

Ubuntu 脚本位于：

```text
scripts/ubuntu/
```

常用流程：

```bash
conda activate isaaclab
cd /path/to/unitree_go2_isaaclab_rl

bash scripts/ubuntu/check_env.sh

bash scripts/ubuntu/test_task2_world.sh
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/test_task4_env.sh

bash scripts/ubuntu/smoke_task1.sh
bash scripts/ubuntu/train_task1.sh
```

Ubuntu 脚本适合 headless 训练、环境测试、world 测试、smoke training、正式训练、模型评估和 GUI 可视化。

### Windows

Windows 脚本位于：

```text
scripts/windows/
```

常用流程：

```powershell
conda activate isaaclab
cd C:\path\to\unitree_go2_isaaclab_rl

.\scripts\windows\check_env.ps1
.\scripts\windows\check_task1.ps1
.\scripts\windows\check_task2.ps1
.\scripts\windows\check_task3.ps1
.\scripts\windows\check_task4.ps1

.\scripts\windows\smoke_task1.ps1
.\scripts\windows\train_task1.ps1
```

Windows 和 Ubuntu 均提供 check、smoke、train、eval、visualize 等控制入口。两个系统的脚本职责一致，只是命令语法不同：Ubuntu 使用 Bash，Windows 使用 PowerShell。

---

## 🧭 推荐训练顺序

推荐顺序：

1. 先训练 Task1，获得基础平地 locomotion checkpoint；
2. Task2 从 Task1 warm-start，训练多地形运动；
3. Task3 从 Task1 或 Task2 warm-start，训练导航与避障；
4. Task4 从 Task2 warm-start，训练 Sim2Real / RMA Teacher。

也可以每个任务从零开始训练，但训练时间会更长，早期调参难度更高。

---

## 📌 当前状态与限制

- 本项目主要用于学习、复现实验和开源交流；
- 当前代码完成了四个任务的 IsaacLab 环境、测试、`skrl` PPO 训练和模型测试脚本；
- Ubuntu 下已完成 world 测试、env 测试和小批量 smoke training 验证，验证记录见 `docs/ubuntu_validation.md`；
- Task4 当前实现 RMA Teacher 训练阶段，Student 蒸馏和 adaptation module 需要后续继续扩展；
- 不同 Isaac Lab / Isaac Sim 版本之间可能存在 API 差异，需要根据本地环境做少量适配；
- 训练效果会受到 GPU、并发数、随机种子、训练步数和超参数影响；
- 本项目不是官方 Unitree 或 NVIDIA 项目。

---

## ❓ 常见问题

### 1. `ModuleNotFoundError: No module named torch`

通常是当前 Python 环境不是 Isaac Lab 对应环境。先确认：

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
```

### 2. IsaacLab / `pxr` 导入报错

涉及 IsaacLab、USD、`pxr` 的文件需要在 Isaac Sim / Isaac Lab 环境中运行。环境测试和训练脚本会通过 AppLauncher 管理 IsaacLab 运行入口，直接运行单个 Python 文件时需要保证环境和导入顺序正确。

### 3. 训练启动后显存不足怎么办?

先降低并发数：

```bash
--num-envs 8
--num-envs 16
--num-envs 32
--num-envs 64
```

确认能跑通后再逐步增加。

### 4. Smoke training 有什么作用?

Smoke training 用于检查训练流程是否能启动、日志是否能写入、checkpoint 是否能保存，不代表最终策略效果。

### 5. Task3 / Task4 为什么推荐 warm-start?

Task3 加入导航和障碍物，Task4 加入扰动和域随机化，直接从零训练更难。使用 Task1 / Task2 checkpoint 可以先继承基础步态，再学习更复杂的任务。

### 6. Windows 路径需要怎么改?

优先参考：

```text
configs/local_paths.example.yaml
configs/platform_windows.example.yaml
docs/windows_path_config.md
scripts/windows/_common.ps1
```

常用检查命令：

```powershell
Get-Location
Resolve-Path .
Get-Command python
python -c "import sys; print(sys.executable)"
```

### 7. 为什么要先跑环境测试?

四足机器人训练中的很多问题来自 reset、观测维度、坐标系、接触传感器、奖励项或终止条件。先运行测试可以在正式训练前定位基础工程问题。

### 8. Task2 world 测试通过后进程退出异常怎么办?

Task2 world 测试主要验证 terrain 与 curriculum 的 tensor-level 白盒逻辑。如果日志中已经出现：

```text
Go2 Task2 World / Terrain / Curriculum 测试全部通过
```

说明功能测试已完成。部分本地 Isaac / Kit 环境可能在进程退出阶段需要 timeout 处理，该现象属于测试进程生命周期问题，不代表 terrain / curriculum 逻辑失败。

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

如果这个项目对相关工作有帮助，欢迎参考、修改和继续完善。也欢迎指出代码或文档中的问题。

联系邮箱：2559906288@qq.com  
小红书账号：574661219
