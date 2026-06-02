# Ubuntu 验证记录 / Ubuntu Validation Record

## 一、验证范围

本文档记录开源清理重构后，在 Ubuntu 系统下执行的验证结果。本文档只记录专题验证信息，不修改根目录 `README.md`。

## 1. Scope

This document records the Ubuntu-only validation performed after the open-source cleanup refactor. This document only records focused validation information and does not update the root `README.md`.

## 二、验证类别

本次验证覆盖三个必要类别：

1. 世界模型 / 场景逻辑测试；
2. IsaacLab 环境测试；
3. 四个任务的小批量 smoke training 测试。

## 2. Validation Categories

The validation covers three required categories:

1. World-model / scene-logic tests;
2. IsaacLab environment tests;
3. Small-batch smoke training tests for all four tasks.

## 三、验证环境

验证应在 IsaacLab conda 环境中运行。基础环境检查包括：

- Python 能够导入 `torch`；
- CUDA 可用；
- IsaacLab 可导入；
- `skrl` 可导入；
- `go2_rl` 可从项目源码树导入。

## 3. Environment

Validation is expected to run inside the IsaacLab conda environment. Baseline environment checks include:

- Python can import `torch`;
- CUDA is available;
- IsaacLab can be imported;
- `skrl` can be imported;
- `go2_rl` can be imported from the project source tree.

## 四、静态检查

运行实际测试前，先执行以下静态检查：

- 对 `src/go2_rl` 和 `tests` 下所有 Python 文件执行 `py_compile`；
- 对 active Ubuntu shell 脚本执行 `bash -n` 语法检查。

## 4. Static Checks

Before runtime validation, the following static checks are executed:

- `py_compile` for all Python files under `src/go2_rl` and `tests`;
- `bash -n` syntax checks for active Ubuntu shell scripts.

## 五、世界模型 / 场景逻辑测试

Ubuntu 世界模型测试入口包括：

- `scripts/ubuntu/test_task2_world.sh`
- `scripts/ubuntu/test_task3_world.sh`

Task2 world 测试覆盖多地形生成器、logical terrain index 到 Isaac generator flat index 的映射、spawn origin、地形难度参数、多材质采样、height scan、terrain privileged features 和 terrain curriculum telemetry。

Task3 world 测试覆盖解析式导航世界、目标采样、静态 / 动态障碍物、lidar tensor、risk features、termination events 和 world privileged features。

## 5. World-Model / Scene-Logic Tests

Ubuntu world-model test entries include:

- `scripts/ubuntu/test_task2_world.sh`
- `scripts/ubuntu/test_task3_world.sh`

Task2 world validation checks the multi-terrain generator, logical terrain index to Isaac generator flat-index mapping, spawn origins, terrain difficulty parameters, material sampling, height scan, terrain privileged features, and terrain curriculum telemetry.

Task3 world validation checks analytical navigation-world logic, target sampling, static / dynamic obstacles, lidar tensors, risk features, termination events, and world privileged features.

## 六、环境测试

Ubuntu 环境测试入口包括：

- `scripts/ubuntu/test_task1_env.sh`
- `scripts/ubuntu/test_task2_env.sh`
- `scripts/ubuntu/test_task3_env.sh`
- `scripts/ubuntu/test_task4_env.sh`

这些测试用于验证 IsaacLab 环境构建、Gymnasium `reset()` / `step()` API、观测维度、privileged observation 维度、动作维度、reward / info 输出、强制事件和随机 rollout 数值稳定性。

## 6. Environment Tests

Ubuntu environment test entries include:

- `scripts/ubuntu/test_task1_env.sh`
- `scripts/ubuntu/test_task2_env.sh`
- `scripts/ubuntu/test_task3_env.sh`
- `scripts/ubuntu/test_task4_env.sh`

These tests validate IsaacLab environment construction, the Gymnasium `reset()` / `step()` API, observation dimensions, privileged observation dimensions, action dimensions, reward / info outputs, forced events, and random-rollout numerical stability.

## 七、小批量训练测试

Ubuntu smoke training 测试入口包括：

- `scripts/ubuntu/smoke_task1.sh`
- `scripts/ubuntu/smoke_task2.sh`
- `scripts/ubuntu/smoke_task3.sh`
- `scripts/ubuntu/smoke_task4.sh`

这些测试用于验证四个任务在重构后能够进入训练链路，并能完成小批量训练流程。

## 7. Small-Batch Smoke Training Tests

Ubuntu smoke training test entries include:

- `scripts/ubuntu/smoke_task1.sh`
- `scripts/ubuntu/smoke_task2.sh`
- `scripts/ubuntu/smoke_task3.sh`
- `scripts/ubuntu/smoke_task4.sh`

These tests validate that all four tasks can enter the training path and complete a small-batch training workflow after the refactor.

## 八、验证结果

Ubuntu 验证状态：通过。

在 Ubuntu 下，必要的世界模型测试、环境测试和小批量 smoke training 测试均已完成后，本轮开源清理重构可以进入最终根目录 `README.md` 更新阶段。

## 8. Result

Ubuntu validation status: PASSED.

After the required Ubuntu world-model tests, environment tests, and small-batch smoke training tests have been completed, this open-source cleanup refactor is ready for the final root `README.md` update stage.

## 九、备注

Task2 world 测试主要是 terrain 与 curriculum 逻辑的 tensor-level 白盒验证。如果本地机器上 Isaac / Kit 进程退出需要 timeout 处理，应以日志中的显式完成标记作为功能测试结论：

`Go2 Task2 World / Terrain / Curriculum 测试全部通过`

该现象属于测试进程退出生命周期问题，不代表 Task2 world terrain / curriculum 逻辑失败。

## 9. Notes

Task2 world testing is primarily a tensor-level white-box validation for terrain and curriculum logic. If Isaac / Kit process shutdown requires timeout handling on a local machine, the functional test result should be interpreted from the explicit completion marker in the log:

`Go2 Task2 World / Terrain / Curriculum 测试全部通过`

This behavior is a test-process lifecycle issue and does not indicate a failure in the Task2 world terrain / curriculum logic.
