# Physical AI 课程期末作业：计划与方向整理

> 整理日期：2026-10-13（2026-10-18 更新）
> 状态：M0（环境）、M1（ROS2 接口）、M2（数据采集）、M3（训练）、M4（闭环评估）已完成；M5（报告与演示视频）进行中

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
3. **数据采集**：用 IK 脚本专家自动生成示教轨迹，用 rosbag（mcap）逐回合录制。✅
4. **数据转换**：rosbag → npz（ROS 环境中）→ LeRobot 数据集（LeRobot 环境中）。✅
5. **训练**：使用 LeRobot 训练 ACT 策略（状态输入）作为基线；SmolVLA 放到扩展阶段（需要先加图像观测，见第 10 节）。训练放在 Colab Pro，不在 M1 上硬训。✅
6. **闭环控制**：策略在独立 venv 的推理服务（`scripts/policy_server.py`）中运行，ROS 节点 `policy_eval` 通过本地 TCP 获取动作，在 `sync_mode` 下控制仿真并统计成功率。✅（结果见“实验结果”一节）

## 5. 任务设计

- **主任务（已定）**：SO-101 把 4 cm 绿色立方体抓起，放到红色目标方框上（pick-and-place）。由于脚本专家可以稳定完成抓取放置，不再需要降级到 reaching。
- **方块初始位置**（2026-10-15 验证）：Menagerie 原场景中方块在 x=0.5 m，超出了末端最大水平伸展（约 0.48 m），**不可达**。新场景 `scene_pick_place.xml` 中，每次重置时方块在 r ∈ [0.18, 0.25] m、方位角 ∈ [5°, 40°] 内随机生成（最初为 0.17 m，见下），yaw 也随机；目标固定在 (0.22, −0.12)。范围的依据：
  - 俯视姿态的末端只能到 z≈0.06–0.09 m（`wrist_flex` 限位），所以用 4 cm 立方体代替 6 cm 高的方块，否则张开的夹爪无法越过方块；
  - r < 0.16 m 时，夹爪上的相机支架与 `shoulder` 自碰撞；r > 0.26 m 时，俯视姿态够不到悬停高度；
  - 离线测试中，脚本专家在此范围内 600/600 成功（外扩一圈后为 297/300）。
  - 2026-10-17：按 ACT 失败分析，专家抓取时把夹爪朝固定爪一侧平移 9 mm（`expert.GRASP_OFFSET`）。`scripts/check_expert.py` 发现 r ≈ 0.17–0.175 m 的部分朝向下，相机支架会与 `shoulder` 碰撞（偏移前就有少量，偏移后变多），r ≥ 0.18 m 全部成功，因此下限提高到 0.18 m。之前的回合 0–199 和 seed 100 评估都是在 0.17 m 下限下得到的。
- **观测**：关节角（6）+ 方块特征 `[x, y, z, cos 4ψ, sin 4ψ]`（5，作为 `observation.environment_state`）；暂不使用图像（场景中已预留 `overview` 相机，可作为扩展）。
- **动作**：6 个关节的目标角，50 Hz。
- **成功判据**：方块中心与目标的水平距离 < 3 cm，且方块立在地面上。
- **评价**：`sim_node` 固定 `seed`（评估用 100；示教数据用了 0 和 1），在 `sync_mode` 下跑 50 个回合，统计成功率；用闭环成功率挑选 checkpoint。

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
- **LeRobot**：不装入 ROS 环境。本机推理用工程下单独的 uv venv（`.venv-lerobot`，`lerobot[dataset,training]==0.6.1`），作为独立进程运行。从 pixi shell 启动时，pixi 的 `PYTHONPATH` 会把 ROS 环境的 numpy/libomp 带进来（`OMP: Error #15`），`policy_server.py` 已在导入 torch 前把它们从 `sys.path` 中去掉。

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
| ~~LeRobot 与 ROS 环境依赖冲突~~ | 已解决：训练放 Colab；Mac 推理用独立 venv 中的推理服务 + 本地 TCP |
| ~~方块初始位置（x=0.5 m）可能超出 SO-101 工作空间~~ | 已解决：确认不可达，新建 `scene_pick_place.xml` 并随机化 |
| 控制端延迟 / 卡顿导致数据中动作与状态错位 | 已解决：`sim_node` 增加 `sync_mode`，逐帧握手；M4 推理也使用该模式 |
| M1 无法高效训练 | 在 Colab / 云 GPU 训练，Mac 只做采集与推理 |
| 图像观测渲染慢 | 先用关节状态；图像作为可选扩展 |
| 抓取任务示教数据质量差 | 已解决：脚本专家 200/200 成功，不需要降级为 reaching |
| 示教“正确但不稳”：抓取余量太小，模仿误差导致抓取失败 | 已解决：专家抓取点朝固定爪偏移 9 mm（失败分析见 `docs/eval_act_200_failure_analysis.md`） |
| 专家在相对抓取角 ±45° 处切换抓法，动作不连续，策略在边界附近抓偏 | 未解决：在观测中加入专家选用的抓取角，或限制方块朝向（见 `docs/eval_act_offset.md` 第 6 节） |
| SO-101 的 ROS2 工作区面向 Isaac / 真机，MuJoCo 适配需自己做 | 只借鉴其节点结构与数据转换工具，仿真侧自己封装（已完成） |
| ~~作业时间有限~~ | 已解决：直接以 pick-and-place 打通了完整闭环 |

## 9. 里程碑（建议）

1. ✅ **M0 环境**：Mac 上跑通 ROS2（`ros2 topic list`）与 MuJoCo（加载 SO-101）。（2026-10-13）
2. ✅ **M1 接口**：MuJoCo 节点发布 `/joint_states`，能接收 `/joint_command` 控制机械臂。（2026-10-14；实测 50 Hz，另有 `/box_pose`、`/clock`、`/reset`）
3. ✅ **M2 数据**：脚本专家录制 rosbag，转换为 LeRobot 数据集。（2026-10-15；50 回合，ACT 冒烟训练通过；2026-10-16 补采到 200 回合，全部成功）
4. ✅ **M3 训练**：在 Colab Pro 上训练 ACT（pick-and-place，状态输入），对比 50 / 200 回合两个数据量（`notebooks/train_act_colab.ipynb`，60k 步，每 10k 步存档）。（2026-10-17；2026-10-18 用加抓取偏移后的数据重新训练）
5. ✅ **M4 闭环**：`policy_eval` + `policy_server.py` 在仿真中闭环控制 SO-101，seed 100 下评估 50 回合（约 5 s/回合）。最好的结果为 49/50（偏移数据，40k）。（2026-10-18）
6. ⏳ **M5 收尾**：整理报告与演示视频（失败 / 成功回合的视频可用 `scripts/render_eval_cases.py` 生成）。时间允许的话做第 10 节中的改进实验。

## 实验结果

ACT（关节角 + 方块特征，无图像，`chunk_size = n_action_steps = 50`），seed 100，50 回合：

| 模型 | 训练数据 | 成功率 | 主要失败原因 |
|---|---|---|---|
| act_50（60k） | 专家无偏移，50 回合 | 30/50 = 60% | 固定爪压在方块顶面（专家余量约 0.6 mm） |
| act_200（60k） | 专家无偏移，200 回合 | 48/50 = 96% | 同上 |
| act_200 偏移（40k） | 专家偏移 9 mm，200 回合 | 49/50 = 98% | 相对抓取角 ≈ +45° 处抓法切换，手腕少转约 30° |
| act_200 偏移（60k） | 同上 | 47/50 = 94% | 同上 |

报告中可以讨论的几点：

1. **数据量**：50 → 200 回合，成功率从 60% 提升到 96%，配对比较 18:0，差异显著；提升全部来自抓取阶段，放置精度不变（`docs/eval_act_50_vs_200.md`）。
2. **示教要对模仿误差鲁棒**：专家本身 100% 成功，但抓取余量只有 0.6 mm，策略约 ±4.5 mm 的误差足以让抓取失败。给专家留出余量后，这类失败消失了（`docs/eval_act_200_failure_analysis.md`）。
3. **示教中的不连续会被模仿学习平滑掉**：专家的抓取方向在 ±45° 处跳 90°，而观测是连续的，策略在边界附近把两种抓法平均了。这是偏移之后剩下的唯一失败模式（`docs/eval_act_offset.md`）。
4. 相邻 checkpoint 的成功率差异不显著，系统偏差有 ±0.5 cm 的波动；50 回合的评估分辨不出 94% 和 98% 的差别。

## 10. 待决事项

- [x] RoboStack 在本机能否装上 ROS2 Jazzy 及所需包（可以；仅 mujoco_ros2_control 缺失，已绕开）
- [x] 方块初始位置是否在 SO-101 工作空间内、是否随机化（x=0.5 m 不可达；改为随机化，范围见第 5 节）
- [x] 最终任务选 reaching 还是 pick-and-place（pick-and-place）
- [x] 观测是否包含相机图像（基线不用；扩展阶段加 overview + wrist 图像，对比 ACT 图像版与 SmolVLA 微调）
- [x] 训练用 Colab 还是其他云 GPU（Colab Pro）
- [ ] 作业的提交截止日期与评分要点（需对照课程要求）

改进实验（可选，按优先级排列）：

- [ ] 在观测中加入专家选用的相对抓取角，消除 ±45° 处的不连续，重新训练并评估
- [ ] 针对 |g| ∈ [35°, 45°] 做定向评估；把评估增加到 100–200 回合
- [ ] 用偏移数据训练的 act_50 做评估，完成偏移后的 50 / 200 对比（模型已训练，尚未评估）
- [ ] 执行方式对比：时间集成（`--temporal-ensemble 0.01`）、`--n-action-steps 25`
- [ ] 扩展：给 `sim_node` 加图像观测，对比 ACT 图像版与 SmolVLA 微调
