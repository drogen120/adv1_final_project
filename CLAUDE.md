# CLAUDE.md

Physical AI 课程期末作业（个人独立完成）：MuJoCo 仿真 SO-101 机械臂 → ROS 2 采集示教数据 → LeRobot 训练 ACT 策略 → 策略在仿真中闭环控制。
整体计划见 `physical_ai_final_project_plan.md`，使用方法与接口见 `README.md`。用户使用中文交流。

## 当前进度

- M0（环境）、M1（`sim_node`：`/joint_states`、`/joint_command`、`/box_pose`、`/clock`、`/reset`）已完成并验证。
- 下一步 M2：数据采集。先确认 `scene_box.xml` 中方块初始位置（x=0.5 m）是否在 SO-101 工作空间内——尚未验证，可能需要移近并随机化。

## 环境与命令

- 开发机为 Apple M1 Mac，无 NVIDIA GPU。环境由 pixi 管理（`pixi.toml`），ROS 2 Jazzy 来自 RoboStack，Python 固定为 3.12。
- 构建：`pixi run build`（= 在 `ros2_ws/` 下 `colcon build --symlink-install`）。
- 运行 ROS 命令前需要：`pixi shell` 后 `source ros2_ws/install/setup.zsh`（非交互脚本中用 `pixi run bash script.sh` 并在脚本里 `source install/setup.bash`）。
- 唯一的包是 `ros2_ws/src/so101_mujoco`（ament_python）。SO-101 模型在包内 `models/robotstudio_so101/`，通过 `setup.py` 的 `model_files()` 安装到 share 目录。

## 已确认的坑（不要重复踩）

- **osx-arm64 上没有 `ros-jazzy-mujoco-ros2-control`**（RoboStack 中只有 Linux 版本），DFKI 版本也不在 RoboStack 中。因此仿真接口采用自写的 rclpy + mujoco 节点，不使用 ros2_control。`webots-ros2` 也不可用。
- **mjpython / python**：`launch_passive`（`sim_node` 的 viewer 模式）必须用 `mjpython`（`ros2 run --prefix mjpython ...`）；独立 viewer `python -m mujoco.viewer` 必须用普通 `python`，用 mjpython 会报 `Caught an unknown exception!`。
- **测试时停止节点**：`ros2 run` 不转发 SIGINT，脚本中需 `set -m` 并 `kill -INT -<pgid>`（发给进程组）。非交互 bash 的后台任务默认忽略 SIGINT。
- macOS 没有 `timeout` 命令；`ros2 topic pub/echo/hz` 在脚本中需要自己加超时，否则会挂住。`ros2 topic pub --once` 要加 `-w 1` 等待订阅者。
- 在 `sim_node.run()` 等函数内部写 `import mujoco.viewer` 会把 `mujoco` 变成局部变量导致 `UnboundLocalError`，导入放在文件顶部。
- spin 线程在关闭时会抛 `ExternalShutdownException`，属正常现象，需捕获。

## 约定

- `sim_node` 中 `self.lock` 保护 `self.data`：ROS 回调在 executor 线程，物理步进在主（脚本）线程；访问 `data` 必须持锁。
- 仿真时间 = `time_offset + data.time`，重置时累加 offset，保证 `/clock` 单调递增（rosbag 依赖这一点）。
- 依赖用 `pixi add ros-jazzy-<pkg>` / `pixi add --pypi <pkg>` 添加，不用 rosdep。LeRobot（含 torch）暂不装进 ROS 环境，避免与 RoboStack 的 numpy 等版本冲突；训练在 Colab 上进行。
