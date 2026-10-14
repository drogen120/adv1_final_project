# Physical AI 课程期末作业：计划与方向整理

> 整理日期：2026-10-13（2026-10-14 更新）
> 状态：M0（环境）、M1（ROS2 接口）已完成；下一步 M2（数据采集）

---

## 1. 作业要求（原始需求）

- 用 **ROS2** 与一个**机器人模拟器**连接。
- 用该模拟器**生成训练数据**。
- 用这些数据**训练一个模型**。
- 最后用训练好的模型**控制机器人完成某个任务**。
- 机器人类型不限（广义机器人：人形、机器狗、机械臂等）。
- **个人独立完成**，因此要求模拟器**易于上手、不要太复杂**。

## 2. 约束与环境

| 项目 | 情况 |
|---|---|
| 开发机 | MacBook Pro（Apple M1） |
| ROS2 | Jazzy，已通过 RoboStack + pixi 安装（Python 3.12） |
| MuJoCo | 3.15（PyPI `mujoco`） |
| NVIDIA GPU | 无（Mac），因此无法使用 Isaac Sim / Isaac Lab |
| 人力 | 1 人 |
| 机器人 | **SO-101 机械臂**（已决定） |

## 3. 模拟器选型结论

| 模拟器 | 上手难度 | ROS2 连接 | 学习生态 | 在 M1 Mac 上 | 结论 |
|---|---|---|---|---|---|
| **MuJoCo** | 低 | 自写 rclpy 节点（mujoco_ros2_control 在 Mac 上不可用） | 强（LeRobot 等） | 原生支持 | **首选（已采用）** |
| Isaac Sim + Isaac Lab | 中到高 | 官方 ROS 2 Bridge | 最强 | 不可用（需 NVIDIA GPU） | 排除 |
| Gazebo | 中 | 最原生（RoboStack 有 osx-arm64 版 `ros-gz`） | 弱（需自建数据管线） | 可安装 | 备选 |
| Webots | 低 | 官方 webots_ros2 | 较弱 | RoboStack 无 `webots-ros2` | 排除 |

**决定：MuJoCo + ROS2 + SO-101 + LeRobot（ACT 策略）。** ROS2 接口采用自写的 rclpy + mujoco 节点（`so101_mujoco/sim_node`），不使用 ros2_control。

## 4. 总体技术路线

```
[MuJoCo 仿真（SO-101 + 方块）]
        │  ROS2 话题（/joint_states, /joint_command, /box_pose, /clock；服务 /reset；可选相机图像）
        ▼
[ROS2 接口层] ──► [数据采集节点：遥操作 / 脚本策略 → rosbag]
                          │
                          ▼
              [rosbag → LeRobot 数据集转换]
                          │
                          ▼
        [训练（Colab / 云 GPU）：ACT 等策略]
                          │
                          ▼
        [策略推理节点（Mac 上运行）] ──► ROS2 话题 ──► MuJoCo 中的 SO-101 完成任务
```

### 各阶段说明

1. **仿真**：MuJoCo 中加载 SO-101（MuJoCo Menagerie 的 `robotstudio_so101`，使用其 `scene_box.xml` 场景）与一个待抓取的方块。✅
2. **ROS2 接口**：把 MuJoCo 封装为 ROS2 节点，发布关节状态与方块位姿，订阅关节指令，提供重置服务。✅（接口细节见 README）
3. **数据采集**：通过键盘 / 手柄遥操作，或编写脚本策略自动生成示教轨迹，用 rosbag 录制。
4. **数据转换**：将 rosbag 转为 LeRobot 数据集格式。
5. **训练**：使用 LeRobot 训练 ACT 策略（可选 SmolVLA）。训练放在 Colab / 云 GPU，不在 M1 上硬训。
6. **闭环控制**：策略节点订阅观测、发布动作，在仿真中完成任务。

## 5. 任务设计（初案）

- **主任务**：SO-101 抓取方块并放置到目标位置（pick-and-place）。
- **降级方案**（若抓取过难）：让末端到达目标点（reaching），或推动方块。
- **观测**：先使用**关节状态**为主；图像观测作为加分项（M1 上离屏渲染较慢）。
- **评价**：固定随机种子下跑 N 个回合，统计成功率。

## 6. 环境搭建方案（Mac / M1）

ROS2 在 macOS 上不是官方主要支持平台，这曾是最大风险点。**已采用 RoboStack 方案并安装成功。**

| 方案                                          | 优点                       | 缺点                         |
| ------------------------------------------- | ------------------------ | -------------------------- |
| **RoboStack（pixi，ROS2 Jazzy）**（已采用） | 原生运行，与 MuJoCo 在同一系统，调试简单 | 部分包缺少 osx-arm64 版本（见下） |
| UTM + Ubuntu 24.04 arm64 虚拟机（退路）            | 接近标准 Linux，教程可直接照搬       | 图形性能一般                     |
| Docker                                      | 环境可复现                    | Mac 上 GUI 与 DDS 网络配置麻烦，不推荐 |

- **ROS2 版本**：Jazzy（LTS，支持至 2029 年 5 月）。
- **包管理**：pixi（`pixi.toml` / `pixi.lock`），频道为 `robostack-jazzy` + `conda-forge`。RoboStack Jazzy 固定 Python 3.12、NumPy 2，要求 macOS ≥ 14。
- **osx-arm64 包核实结果（2026-10-13）**：`desktop`、`ros2_control`/`ros2_controllers`、`rosbag2`（含 mcap、Python 接口）、`xacro`、`joy`、`teleop_twist_keyboard`、`cv_bridge`、`moveit`、`ros_gz` 等均可用；**`mujoco_ros2_control` 仅有 Linux 版本**，`webots_ros2` 全平台都没有。
- **MuJoCo**：`pixi add --pypi mujoco`。macOS 上 `launch_passive` 必须用 `mjpython` 启动，独立 viewer（`python -m mujoco.viewer`）则必须用普通 `python`。
- **LeRobot**：暂不装入 ROS 环境（torch 等依赖可能与 RoboStack 冲突）；M4 需要在 Mac 上推理时再评估，必要时把策略放到独立进程中。

## 7. 参考资源

- SO101 ROS 2 Workspace（含遥操作、rosbag 录制、LeRobot 数据集转换、策略节点）：https://so101-ros2.readthedocs.io/latest/
- mujoco_ros2_control（DFKI，ROS2 与 MuJoCo 的 ros2_control 接口；Mac 上不可用，仅作参考）：https://github.com/dfki-ric/mujoco_ros2_control
- MujocoROS2Control 论文（JOSS）：https://joss.theoj.org/papers/10.21105/joss.09140
- LeRobot SO-101 + MuJoCo 的 ACT 仿真示例：https://huggingface.co/ddduk/so101_act35_real_sim_v1
- 仿真预训练 + 少量真机数据微调的 ACT 示例：https://huggingface.co/omiya239532/so101_act_cotrain
- DeepMind MuJoCo Menagerie（机器人模型库，SO-101 模型取自其 `robotstudio_so101`）：https://github.com/google-deepmind/mujoco_menagerie
- RoboStack Jazzy 包列表：https://robostack.github.io/jazzy.html

## 8. 主要风险与对策

| 风险 | 对策 |
|---|---|
| ~~Mac 上 ROS2 安装困难~~ | 已解决：RoboStack + pixi 安装成功 |
| ~~Mac 上没有 mujoco_ros2_control~~ | 已解决：自写 rclpy + mujoco 仿真节点 |
| LeRobot 与 ROS 环境依赖冲突 | 训练放 Colab；Mac 推理时用独立环境 / 独立进程 |
| 方块初始位置（x=0.5 m）可能超出 SO-101 工作空间 | M2 开始前验证，必要时移近并随机化 |
| M1 无法高效训练 | 在 Colab / 云 GPU 训练，Mac 只做采集与推理 |
| 图像观测渲染慢 | 先用关节状态；图像作为可选扩展 |
| 抓取任务示教数据质量差 | 先用脚本策略生成干净示教；任务可降级为 reaching |
| SO-101 的 ROS2 工作区面向 Isaac / 真机，MuJoCo 适配需自己做 | 只借鉴其节点结构与数据转换工具，仿真侧自己封装（已完成） |
| 作业时间有限 | 先打通最小闭环（reaching），再逐步升级到 pick-and-place |

## 9. 里程碑（建议）

1. ✅ **M0 环境**：Mac 上跑通 ROS2（`ros2 topic list`）与 MuJoCo（加载 SO-101）。（2026-10-13）
2. ✅ **M1 接口**：MuJoCo 节点发布 `/joint_states`，能接收 `/joint_command` 控制机械臂。（2026-10-14；实测 50 Hz，另有 `/box_pose`、`/clock`、`/reset`）
3. ⏳ **M2 数据**：脚本策略 / 遥操作录制 rosbag，转换为 LeRobot 数据集。
4. **M3 训练**：在 Colab 上训练出第一个可用策略（先 reaching）。
5. **M4 闭环**：策略节点在仿真中控制 SO-101 完成任务，并统计成功率。
6. **M5 收尾**：升级任务到 pick-and-place（如时间允许）、整理报告与演示视频。

## 10. 待决事项

- [x] RoboStack 在本机能否装上 ROS2 Jazzy 及所需包（可以；仅 mujoco_ros2_control 缺失，已绕开）
- [ ] 方块初始位置是否在 SO-101 工作空间内、是否随机化
- [ ] 最终任务选 reaching 还是 pick-and-place
- [ ] 观测是否包含相机图像
- [ ] 训练用 Colab 还是其他云 GPU
- [ ] 作业的提交截止日期与评分要点（需对照课程要求）
