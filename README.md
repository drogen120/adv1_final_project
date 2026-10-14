# SO-101 MuJoCo × ROS 2：模仿学习课程大作业

Physical AI 课程期末作业。在 MuJoCo 中仿真 SO-101 机械臂，通过 ROS 2 连接仿真、采集示教数据，用 LeRobot（ACT 策略）训练模型，最后用训练好的模型在仿真中控制机械臂完成任务（目标：pick-and-place，降级方案：reaching）。

详细的方案选型与计划见 [physical_ai_final_project_plan.md](physical_ai_final_project_plan.md)。

## 进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 环境 | Mac 上跑通 ROS 2 与 MuJoCo，加载 SO-101 | ✅ 完成 |
| M1 接口 | MuJoCo 节点发布 `/joint_states`，接收 `/joint_command` | ✅ 完成 |
| M2 数据 | 脚本策略 / 遥操作录制 rosbag，转换为 LeRobot 数据集 | ⏳ 下一步 |
| M3 训练 | 在 Colab 上训练策略 | — |
| M4 闭环 | 策略节点控制仿真中的 SO-101，统计成功率 | — |
| M5 收尾 | 升级任务、整理报告与演示视频 | — |

## 环境

| 项目 | 版本 |
|---|---|
| 平台 | macOS（Apple Silicon，osx-arm64），需 macOS ≥ 14 |
| 包管理 | [pixi](https://pixi.sh)（依赖与锁定版本见 `pixi.toml` / `pixi.lock`） |
| ROS 2 | Jazzy，来自 [RoboStack](https://robostack.github.io/jazzy.html)（`robostack-jazzy` 频道） |
| Python | 3.12（RoboStack Jazzy 固定版本） |
| MuJoCo | 3.15（PyPI 的 `mujoco` 包） |

## 工程结构

```
adv1_final_project/
├── README.md
├── CLAUDE.md                              # 给 Claude Code 的项目说明
├── physical_ai_final_project_plan.md      # 方案选型与计划
├── pixi.toml / pixi.lock                  # 环境定义与锁定版本
└── ros2_ws/                               # colcon 工作区
    └── src/
        └── so101_mujoco/                  # ament_python 包
            ├── package.xml
            ├── setup.py / setup.cfg
            ├── resource/so101_mujoco
            ├── models/robotstudio_so101/  # SO-101 的 MJCF 模型（来自 MuJoCo Menagerie）
            │   ├── so101.xml              # 机械臂本体
            │   ├── scene.xml              # 机械臂 + 地面
            │   ├── scene_box.xml          # 机械臂 + 地面 + 方块（默认场景）
            │   └── assets/*.stl
            └── so101_mujoco/
                ├── sim_node.py            # MuJoCo 仿真节点
                └── wave_demo.py           # 发送正弦关节指令的演示节点
```

## 安装

```bash
# 1. 安装 pixi
brew install pixi

# 2. 在工程根目录安装全部依赖（ROS 2 Jazzy、colcon、编译工具、mujoco）
pixi install

# 3. 构建 ROS 2 工作区
pixi run build
```

## 使用方法

每个新终端都需要先进入环境并 source 工作区：

```bash
pixi shell
source ros2_ws/install/setup.zsh
```

### 启动仿真

```bash
# 带可视化窗口（macOS 上必须通过 mjpython 启动）
ros2 run --prefix mjpython so101_mujoco sim_node --ros-args -p use_viewer:=true

# 无界面（采集数据时使用）
ros2 run so101_mujoco sim_node
```

### 让机械臂动起来

```bash
# 演示：多个关节做正弦运动、夹爪开合
ros2 run so101_mujoco wave_demo

# 手动发送一次指令（单位：弧度；可以只指定部分关节）
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
  "{name: [shoulder_pan, elbow_flex], position: [0.5, 0.8]}"

# 查看状态
ros2 topic echo /joint_states
ros2 topic echo /box_pose

# 重置场景
ros2 service call /reset std_srvs/srv/Trigger
```

### 只看模型（不启动 ROS）

```bash
pixi run python -m mujoco.viewer --mjcf=ros2_ws/src/so101_mujoco/models/robotstudio_so101/scene_box.xml
```

## `sim_node` 接口

| 类型 | 名称 | 消息类型 | 说明 |
|---|---|---|---|
| 发布 | `/joint_states` | `sensor_msgs/JointState` | 6 个关节的位置 / 速度 / 力矩 |
| 发布 | `/box_pose` | `geometry_msgs/PoseStamped` | 方块位姿（场景中有名为 `box` 的 body 时），frame 为 `world` |
| 发布 | `/clock` | `rosgraph_msgs/Clock` | 仿真时间，重置后继续递增，不会回退 |
| 订阅 | `/joint_command` | `sensor_msgs/JointState` | 关节目标角度（弧度）。带 `name` 时可只控制部分关节；不带 `name` 时需按顺序给出全部 6 个值。超出范围会被截断 |
| 服务 | `/reset` | `std_srvs/Trigger` | 将场景重置到初始状态 |

关节顺序：`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`。每个关节都是位置控制执行器，指令即目标角度。

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_path` | 包内 `scene_box.xml` | MJCF 场景文件路径 |
| `keyframe` | `""` | 初始姿态使用的 keyframe 名（如 `pickup`），为空则使用模型默认姿态 |
| `publish_rate` | `50.0` | `/joint_states`、`/box_pose` 的发布频率（Hz） |
| `realtime_factor` | `1.0` | 仿真速度相对实时的倍率，`0` 表示尽可能快 |
| `use_viewer` | `false` | 是否打开 MuJoCo 可视化窗口（需 mjpython） |
| `publish_clock` | `true` | 是否发布 `/clock` |

## 注意事项

- **mjpython 与 python 的区别**（macOS）：使用 `launch_passive` 的程序（如 `sim_node` 开启 viewer 时）必须用 `mjpython` 启动；而独立 viewer `python -m mujoco.viewer` 必须用普通 `python`，用 `mjpython` 会报 `Caught an unknown exception!`。
- **Ctrl-C**：在终端中直接 Ctrl-C 可以正常退出。在脚本中用 `kill -INT <pid>` 停止 `ros2 run` 启动的节点**无效**（`ros2 run` 不会转发信号），应发给整个进程组：`kill -INT -<pgid>`。
- **话题互相看不到时**：设置 `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`，或切换中间件 `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。
- **添加依赖**：用 `pixi add ros-jazzy-<包名>`（不要用 `rosdep`）；PyPI 包用 `pixi add --pypi <包名>`，需有 Python 3.12 的版本。
- **重新构建**：使用了 `--symlink-install`，修改 `.py` 文件无需重新构建；新增文件、入口或模型文件后需要再次 `pixi run build`。

## 致谢

SO-101 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（`robotstudio_so101`，commit `0059d43`），其许可证见模型目录下的 `LICENSE`。
