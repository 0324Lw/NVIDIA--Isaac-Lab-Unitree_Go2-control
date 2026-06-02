# Ubuntu Validation Record

## Scope

This document records the Ubuntu-only validation performed after the open-source cleanup refactor.

The root README is intentionally not updated in this step.

## Validation Categories

The validation covers three required categories:

1. World / scene-level tests
2. IsaacLab environment tests
3. Small-batch smoke training tests for all tasks

## Environment

Validation is expected to run inside the IsaacLab conda environment.

Required baseline checks:

- Python can import `torch`
- CUDA is available
- IsaacLab can be imported
- `skrl` can be imported
- `go2_rl` can be imported from the project source tree

## Static Checks

The following static checks are included before runtime tests:

- Python compilation for `src/go2_rl` and `tests`
- Bash syntax check for active Ubuntu scripts

## World Tests

Expected Ubuntu world tests:

- `scripts/ubuntu/test_task2_world.sh`
- `scripts/ubuntu/test_task3_world.sh`

Task2 world validation checks the multi-terrain generator, logical-to-generator terrain mapping, spawn origins, terrain-level parameters, material sampling, height scan, privileged terrain features, and terrain curriculum telemetry.

Task3 world validation checks analytical navigation-world logic, target sampling, static/dynamic obstacles, lidar tensors, risk features, termination events, and world privileged features.

## Environment Tests

Expected Ubuntu environment tests:

- `scripts/ubuntu/test_task1_env.sh`
- `scripts/ubuntu/test_task2_env.sh`
- `scripts/ubuntu/test_task3_env.sh`
- `scripts/ubuntu/test_task4_env.sh`

These tests validate IsaacLab environment construction, reset/step API behavior, observation dimensions, privileged observation dimensions where applicable, action dimensions, reward/info outputs, forced events, and random rollout numerical stability.

## Smoke Training Tests

Expected Ubuntu smoke training tests:

- `scripts/ubuntu/smoke_task1.sh`
- `scripts/ubuntu/smoke_task2.sh`
- `scripts/ubuntu/smoke_task3.sh`
- `scripts/ubuntu/smoke_task4.sh`

These tests validate that each task can enter the training path and run a small training job after the refactor.

## Result

Ubuntu validation status: PASSED.

The refactor is considered ready for final README update after the required Ubuntu world tests, environment tests, and smoke training tests have passed.

## Notes

Task2 world testing is primarily a tensor-level white-box validation for terrain and curriculum logic. If Isaac/Kit process shutdown requires timeout handling on a local machine, the functional test result should be interpreted from the explicit test completion marker in the log:

`Go2 Task2 World / Terrain / Curriculum 测试全部通过`

