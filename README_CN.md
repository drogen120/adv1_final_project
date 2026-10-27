# SO-101 MuJoCo × ROS 2：模仿学习课程大作业

[日本語](README.md) | 中文

Physical AI 课程期末作业。在 MuJoCo 中仿真 SO-101 机械臂，通过 ROS 2 连接仿真、采集示教数据，用 LeRobot（ACT 策略）训练模型，最后用训练好的模型在仿真中闭环控制机械臂完成 pick-and-place 任务。

详细的方案选型与计划见 [physical_ai_final_project_plan.md](physical_ai_final_project_plan.md)。

## 进度

| 里程碑   | 内容                                              | 状态                     |
| ----- | ----------------------------------------------- | ---------------------- |
| M0 环境 | Mac 上跑通 ROS 2 与 MuJoCo，加载 SO-101                | ✅ 完成                   |
| M1 接口 | MuJoCo 节点发布 `/joint_states`，接收 `/joint_command` | ✅ 完成                   |
| M2 数据 | 脚本策略录制 rosbag，转换为 LeRobot 数据集                   | ✅ 完成                   |
| M3 训练 | 在 Colab 上训练 ACT（状态输入），对比 50 / 200 回合            | ✅ 完成                   |
| M4 闭环 | 策略节点控制仿真中的 SO-101，统计成功率                         | ✅ 完成（最好 50/50，见下文“结果”） |
| 改进    | 抓取偏移、方块特征归一化、失败后重新抓取的示教数据                       | ✅ 完成                   |
| 演示    | 交互演示界面：在面板上摆放方块，实时观看策略                          | ✅ 完成（见“交互演示”）          |
| M5 收尾 | 整理报告与演示视频；可选改进与扩展                               | ⏳ 进行中                  |

## 结果

ACT（关节角 + 方块特征，无图像），seed 100 的 50 个回合，`sim_node` 同步模式下闭环评估。**当前最好的模型是 `act_291_envnorm_retry`（60k），50/50 成功**：

| 模型 | 训练数据 | checkpoint | 成功率 | 成功回合的放置误差（平均） |
|---|---|---|---|---|
| act_50 | 专家无抓取偏移，前 50 回合 | 60k | 30/50 = 60% | 0.69 cm |
| act_200 | 专家无抓取偏移，200 回合 | 60k | 48/50 = 96% | 0.70 cm |
| act_200（偏移） | 专家抓取偏移 9 mm，200 回合 | 40k | **49/50 = 98%** | 0.88 cm |
| act_200（偏移） | 同上 | 60k | 47/50 = 94% | 0.66 cm |
| act_200_envnorm | 同上，方块特征加 `MEAN_STD` 归一化 | 40k | 49/50 = 98% | 0.58 cm |
| **act_291_envnorm_retry** | 偏移数据 200 回合 + 重新抓取示教 91 回合，方块特征归一化 | 60k | **50/50 = 100%** | **0.48 cm** |

- 所有失败都发生在抓取阶段（方块没有被抬起）；只要抓起来，都能放到目标上。
- 数据量从 50 增加到 200 回合，成功率从 60% 提升到 96%，逐回合配对的差异显著（18:0）。
- 失败分析发现，专家抓取时固定爪离方块只有约 0.6 mm，没有给模仿误差留余量。专家加了 9 mm 偏移后，这类失败消失了；剩下的失败集中在相对抓取角接近 ±45° 的摆放上：专家在这里换一对侧面抓，抓取方向跳 90°，策略把两种抓法“平均”了。
- 偏移模型的 40k 与 60k 在成功率上的差异不显著（p = 0.5），60k 的放置更准。
- 偏移模型的 r 下限是 0.18 m（之前是 0.17 m），所以评估摆放与前两行只是近似配对。
- 加入“第一次抓取失败 → 重新抓取”的示教（`data/demos_retry`，seed 3，采集 100 回合，重抓成功的 91 回合只导出重抓之后的片段）后，`act_291_envnorm_retry` 在 seed 100 的 50 回合上全部成功，而且都是第一次就抓起（`regrasped` = 0，全部在 600 帧内完成），放置误差也是各模型中最小的。在 seed 101 的 100 个回合上（其中 24 个 |g| ≥ 35°）也是 100/100。
- 对照：不含重抓数据的偏移模型（act_200，60k）在边界评估（|g| ∈ [35°, 45°]）上 49/50，但其中 14 回合需要重抓（它没学过重抓，是把整个任务从头再做一遍），600 帧内只成功 36 个。

详细分析：

- [docs/eval_act_200_failure_analysis.md](docs/eval_act_200_failure_analysis.md)：act_200 的失败原因（固定爪压在方块顶面）与改进建议
- [docs/eval_act_50_vs_200.md](docs/eval_act_50_vs_200.md)：50 / 200 回合的对比
- [docs/eval_act_offset.md](docs/eval_act_offset.md)：加抓取偏移后的评估，以及 ±45° 抓法切换处的新失败机制

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
├── README.md                              # 日语版 README（默认）
├── README_CN.md                           # 中文版 README（本文件）
├── CLAUDE.md                              # 给 Claude Code 的项目说明
├── physical_ai_final_project_plan.md      # 方案选型与计划
├── pixi.toml / pixi.lock                  # 环境定义与锁定版本
├── docs/
│   ├── eval_act_200_failure_analysis.md   # ACT 200 回合评估的失败分析与改进建议
│   ├── eval_act_50_vs_200.md              # ACT 50 / 200 回合训练的闭环评估对比
│   └── eval_act_offset.md                 # 加抓取偏移后重新训练的 ACT（40k / 60k）评估与新的失败机制
├── notebooks/
│   └── train_act_colab.ipynb              # 在 Colab 上训练 ACT（M3）
├── scripts/
│   ├── check_expert.py                    # 不用 ROS，在整个随机范围上离线验证脚本专家（可模拟沿开合方向的误差）
│   ├── demo.sh                            # 交互演示：一条命令启动推理服务、带 viewer 的仿真和控制面板
│   ├── check_retry.py                     # 不用 ROS，离线验证失败注入与重新抓取（选注入范围用）
│   ├── grasp_metrics.py                   # 评估的抓取指标（开合方向偏差、朝向误差、首次接触）与逐回合配对比较
│   ├── npz_to_lerobot.py                  # npz → LeRobot 数据集（在装有 LeRobot 的环境中运行）
│   ├── policy_server.py                   # 推理服务：在 LeRobot venv 中加载策略，通过本地 TCP 提供动作
│   ├── render_eval_cases.py               # 把评估中的失败回合和几个成功回合渲染成 mp4（带抓取特写）
│   ├── render_policy_cameras.py           # 重新渲染图像策略在某个评估回合中看到的相机画面（overview / wrist_cam）
│   └── visualize_demos.py                 # 把采集的数据回放成动画（mp4 或窗口）
├── data/                                  # 采集的数据（不纳入 git）
└── ros2_ws/                               # colcon 工作区
    └── src/
        └── so101_mujoco/                  # ament_python 包
            ├── package.xml
            ├── setup.py / setup.cfg
            ├── resource/so101_mujoco
            ├── models/robotstudio_so101/  # SO-101 的 MJCF 模型（来自 MuJoCo Menagerie）
            │   ├── so101.xml              # 机械臂本体
            │   ├── scene.xml              # 机械臂 + 地面
            │   ├── scene_box.xml          # 机械臂 + 地面 + 方块（Menagerie 原版，方块够不着）
            │   ├── scene_pick_place.xml   # 抓取放置场景（默认场景，本项目新增）
            │   └── assets/*.stl
            └── so101_mujoco/
                ├── sim_node.py            # MuJoCo 仿真节点
                ├── wave_demo.py           # 发送正弦关节指令的演示节点
                ├── kinematics.py          # 俯视抓取的数值 IK
                ├── expert.py              # 脚本专家：规划整条抓取放置轨迹
                ├── collect_demos.py       # 采集节点：专家示教 → 每回合一个 rosbag
                ├── export_dataset.py      # rosbag → 对齐的训练帧（.npz）
                ├── policy_eval.py         # 闭环评估节点：调用推理服务控制仿真，统计成功率
                └── policy_demo.py         # 交互演示节点：Tk 控制面板摆放方块，实时运行策略
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

### 采集示教数据（M2）

数据流：`collect_demos`（脚本专家 + rosbag 录制）→ `export_dataset`（rosbag → npz）→ `scripts/npz_to_lerobot.py`（npz → LeRobot 数据集）。

```bash
# 终端 1：同步模式 + 全速运行仿真（同步模式见下文）
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0

# 终端 2：采集 50 个回合（每回合约 10 s 仿真时间，在 M1 上约 4.5 s）
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=50 -p output_dir:=data/demos

# 追加采集时要换一个 seed，否则会重复已有回合的方块摆放（编号会自动接着往下）
# 无偏移的旧数据：回合 0–49 用 seed 0，50–199 用 seed 1（data/demos → data/demos.npz）；评估用 seed 100（补充评估用 seed 101）
# 加抓取偏移后的数据（当前使用）：data/demos_offset（seed 2，200 回合）→ data/demos_offset.npz

# 专家抓取时把夹爪朝固定爪一侧平移 grasp_offset（默认 0.009 m，见 expert.GRASP_OFFSET），
# 给固定爪留出余量；回合 0–199 是 offset 为 0、r ∈ [0.17, 0.25] m 时采集的。改动后先离线验证：
#   pixi run python scripts/check_expert.py --grasp-offset 0.0 0.009 --tool-error -0.005 0 0.005

# 导出为 npz（只导出成功的回合）
ros2 run so101_mujoco export_dataset data/demos data/demos.npz
```

**重新抓取（恢复）示教**：专家每次抓取后都会检查方块是否被抬起（z > 3 cm）；没抬起时，从当前位置回到方块（以当前位姿为准）上方重新抓取，最多重试 2 次。要采集这类数据，需要让一部分回合的第一次抓取故意失败（`inject_failure_prob`）。注入有两种方式，每回合随机选一种：

- `wrist`：开合方向偏转 ±25–35°，同时朝活动爪平移 10–16 mm。这模仿 ACT 在 g ≈ 45° 处的失败；只转手腕不会失败，因为夹爪闭合时会把方块转正。
- `offset`：朝活动爪平移 12–20 mm，固定爪会压到方块顶面上。
- `midway`（边界数据用，不在默认的 `inject_modes` 里）：模仿图像版 ACT 在 |g| ≳ 35° 处的失败。手腕转回到两种抓法的中间（偏差 = −g × 0.7–1.0，`wrist_roll` ≈ 1.6），同时朝活动爪平移 7–11 mm。只转手腕的话，98% 仍能抓起方块；图像模型的失败回合也都是平移约 0.7–1.1 cm 的（见 `docs/eval_act_images.md`）。

抓空后如果方块没有平放在某个面上（多数是被斜着卡在闭合的夹爪里，张开后还会斜靠在固定爪上），专家先张开夹爪，再把末端竖直抬高 2.5 cm（远处够不到时逐级降到 1 cm），让方块倒平，然后再检查一次，还不平才放弃。立方体侧躺也算平放（`expert.is_flat`；`box_yaw` 此时取水平的那条轴）。方块直立时的路径和以前逐帧一致。离线验证（`midway`，|g| ∈ [30°, 45°]，两个 seed 各 100 回合）：抓空 93 + 88，重抓成功 93 + 86。

注入失败后，专家会继续朝目标运动 0–0.8 s 才“发现”失败，这是在模仿 ACT 按动作块开环执行时的反应延迟。这类回合只有在重抓成功后才保留；`episodes.jsonl` 中记录 `clip_start`（开始重抓的帧号），`export_dataset` 只导出从这一帧开始的片段。这样策略只学恢复，不学故意抓偏。

```bash
# 先离线验证注入范围（无需 ROS）：注入是否真的失败、重抓的成功率
pixi run python scripts/check_retry.py

# 采集恢复示教（sim_node 与 collect_demos 用同一个新 seed）
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=3
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 -p output_dir:=data/demos_retry \
    -p inject_failure_prob:=1.0 -p seed:=3

# 边界附近（|g| ∈ [30°, 45°]）的补充数据：box_rel_yaw_range 设为 [30, 60]（rel 30–45 → g +30…+45，45–60 → g −45…−30）
#   离线验证：pixi run python scripts/check_retry.py --modes midway --rel-yaw 30 60 --delay 0 0.3 -n 60
#            （见上文，可用率约 90%）；专家本身：pixi run python scripts/check_expert.py --rel-yaw 30 60 --grid 5 8 12
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=4 -p "box_rel_yaw_range:=[30.0,60.0]"
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 -p output_dir:=data/demos_boundary -p seed:=4
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=5 -p "box_rel_yaw_range:=[30.0,60.0]"
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 -p output_dir:=data/demos_boundary_retry -p seed:=5 \
    -p inject_failure_prob:=1.0 -p "inject_modes:=['midway']" -p "inject_delay:=[0.0,0.3]"

# 与正常示教合并导出
ros2 run so101_mujoco export_dataset data/demos_offset data/demos_retry data/demos_offset_retry.npz
```

在装有 LeRobot 的环境（Colab 等，不要装进 pixi 环境）中转换为 LeRobot 数据集（已在 lerobot 0.6.1 上验证，ACT 可以直接训练）：

```bash
pip install 'lerobot[dataset,training]==0.6.1'
python scripts/npz_to_lerobot.py data/demos.npz --repo-id local/so101_sim_pick_place \
    --root data/lerobot/so101_sim_pick_place
```

**检查数据**：`scripts/visualize_demos.py` 读取 `data/demos.npz`，把选中的回合回放成动画。左边是按记录的关节角和方块位姿重放的 MuJoCo 画面（只算运动学，不跑物理）；右边是各关节的状态（实线）与动作（虚线）曲线以及方块高度，带一条随播放移动的光标。不需要 ROS。

```bash
pixi run python scripts/visualize_demos.py --list         # 列出所有回合（帧数、方块初始位置和朝向等）
pixi run python scripts/visualize_demos.py 0 3 10-12       # 每个回合生成一个 data/viz/episode_NNNN.mp4（约 13 s/回合）
pixi run python scripts/visualize_demos.py 5 --show        # 直接在窗口中播放
pixi run python scripts/visualize_demos.py 0-9 --stride 2 --speed 2   # 隔帧绘制、2 倍速，生成更快
```

想看专家演示时，也可以用带 viewer 的 sim（`--prefix mjpython ... -p use_viewer:=true -p sync_mode:=true`，`realtime_factor` 保持 1.0），再运行 `collect_demos`。

**任务**：把绿色立方体（4 cm）抓起，放到红色方框（中心 (0.22, −0.12)）上。每次 `/reset` 时，方块在极坐标 r ∈ [0.18, 0.25] m、方位角 ∈ [5°, 40°] 的范围内随机生成，yaw 也随机。成功判据：方块中心与目标的水平距离 < 3 cm，且方块立在地面上（z < 3 cm）。

**专家**（`expert.py`）：根据方块位姿，用数值 IK（MuJoCo 雅可比 + 阻尼最小二乘）规划轨迹：关节空间移动到方块上方 → 竖直下降 → 夹紧 → 抬起 → 平移到目标上方 → 下降 → 松开 → 抬起 → 回到初始姿态，共约 10 s（506 帧）。之后 `collect_demos` 再录 `settle_time`（默认 2.0 s，100 帧）手臂停在初始姿态、方块在目标上的保持帧，比 ACT 的 1 s 动作块长，策略才能学到完成后保持不动；2026-10 之前的数据用的是 0.5 s，策略回到初始姿态后会再去方块那里抓一次。`collect_demos` 通过 `PickPlaceEpisode` 分两段规划：抬起之后检查方块，没抬起来就重新抓取（见上文）。

**rosbag 内容**：每回合一个 mcap bag（`data/demos/episode_NNNN`），包含 `/joint_states`、`/box_pose`、`/joint_command`；`data/demos/episodes.jsonl` 记录每个回合的初始位姿与成败。每条指令的时间戳等于它所回应的那条关节状态的时间戳，导出时据此把观测和动作配对。

**数据集字段**（50 fps）：

| 字段 | 维度 | 内容 |
|---|---|---|
| `observation.state` | 6 | 关节角（rad），顺序同下文关节顺序 |
| `observation.environment_state` | 5 | 方块 `[x, y, z, cos 4ψ, sin 4ψ]`（ψ 为 yaw；立方体每 90° 对称，抓取方向只取决于 ψ mod 90°） |
| `action` | 6 | 下发的关节目标角（rad） |

### 训练策略（M3）

在 Colab（GPU）上运行 `notebooks/train_act_colab.ipynb`：把 `data/demos.npz` 和 `scripts/npz_to_lerobot.py` 上传到 Google Drive 的 `so101_act/` 目录，按顺序运行即可。notebook 会转换出两个数据集（全部 200 回合 / 前 50 回合），各训练一个 ACT：

| 设置 | 值 |
|---|---|
| 输入 | `observation.state` + `observation.environment_state`（无图像） |
| `chunk_size` / `n_action_steps` | 50 / 50（1 s 动作块） |
| batch / 步数 / 存档间隔 | 64 / 60k / 10k |
| 归一化 | STATE / ENV / ACTION 均为 `MEAN_STD`（`--policy.normalization_mapping`） |
| 其余 | LeRobot 0.6.1 默认（lr 1e-5，KL 权重 10） |

当前最好的模型 `outputs/act_291_envnorm_retry` 用偏移数据 + 重抓数据合并导出的 `data/demos_offset_retry.npz`（291 回合）训练，方块特征归一化，60k 步（交互演示现在默认用图像版的 `act_img_all_hold` 100k）。此前的模型是用加抓取偏移后的数据 `data/demos_offset.npz` 训练的（notebook 读取的文件名仍是 `demos.npz`，上传时改名即可）。本机 `outputs/act_200`、`outputs/act_50` 是用偏移数据训练的模型，`outputs/*_backup` 是用无偏移旧数据训练的模型；对应的评估结果在 `data/eval/` 下，命名方式相同。

`act_50` / `act_200` 及其 `_backup` 都是在方块特征**未归一化**时训练的（`act_200_envnorm`、`act_291_envnorm_retry` 已归一化）：ACT 默认的 `normalization_mapping` 只有 `VISUAL / STATE / ACTION`，`observation.environment_state` 的类型是 `ENV`，被当作 `IDENTITY` 处理（统计量照样会写进 checkpoint，只是没有用上）。notebook 现在显式加上了 `ENV`，200 回合的 run 改名为 `act_200_envnorm`，不会覆盖 Drive 上旧的 `act_200`。归一化方式存在 checkpoint 的 `policy_preprocessor.json` 里（`norm_map`），`policy_server.py` 会照它处理，新旧模型都不用改推理代码；但旧权重不能靠改这个 json 切换成归一化输入，必须重新训练。

训练曲线记录到 W&B 的 `so101-act` 项目（run 名为 `act_200_envnorm` / `act_50`，此前的 run 为 `act_200`；需在 Colab Secrets 中设置 `WANDB_API_KEY`，不用时把 notebook 中的 `USE_WANDB` 改为 `False`）。checkpoint 不上传 W&B，输出写在 Drive 的 `so101_act/outputs/<run>/checkpoints/NNNNNN/pretrained_model`，最后一个单元把所有 `pretrained_model` 打包下载。在本机解压到工程根目录的 `outputs/`（不纳入 git）。

训练好的模型参数（checkpoint）已在 Google Drive 共享：[https://drive.google.com/drive/folders/10yvLc3Go5icyDsADPANyEJj__5-9XT9s?usp=drive_link](https://drive.google.com/drive/folders/10yvLc3Go5icyDsADPANyEJj__5-9XT9s?usp=drive_link)。下载后解压到工程根目录的 `outputs/` 即可直接用于评估和演示。

### 闭环评估（M4）

LeRobot 和 torch 不装进 pixi 环境：策略在独立 venv 中由 `scripts/policy_server.py` 加载，ROS 节点 `policy_eval` 通过本地 TCP（每行一个 JSON）向它请求动作。

```bash
# 一次性：建 LeRobot venv（工程根目录下的 .venv-lerobot，不纳入 git）
uv venv -p 3.12 .venv-lerobot
VIRTUAL_ENV=.venv-lerobot uv pip install 'lerobot[dataset,training]==0.6.1'

# 终端 1：推理服务（可以在 pixi shell 里启动，脚本会屏蔽 pixi 的 PYTHONPATH）
.venv-lerobot/bin/python scripts/policy_server.py outputs/act_200/checkpoints/060000/pretrained_model
#   时间集成：加 --temporal-ensemble 0.01；改每块执行步数：--n-action-steps 25

# 终端 2：仿真，同步模式，用训练没用过的 seed
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=100

# 终端 3：评估 50 个回合（每回合 900 帧 = 18 s 仿真时间，留出重抓的时间；之前的评估是 600 帧）
ros2 run so101_mujoco policy_eval --ros-args -p num_episodes:=50 -p output_dir:=data/eval/act_200_60k
```

**边界评估**：sim 加 `-p "box_rel_yaw_range:=[35.0,55.0]"`，只在相对抓取角 |g| ∈ [35°, 45°] 内摆放方块（立方体每 90° 对称，[35, 55] 就是 [35, 45] ∪ [−45, −35]）。这是专家切换抓法的位置，ACT 的失败集中在这里，用来比较重抓数据的效果。

`policy_eval` 对每条 `/joint_states` 等到同一时间戳的 `/box_pose` 后，按与 `export_dataset` 相同的方式构造观测（关节角 + `expert.box_features`），请求动作并以相同时间戳发布 `/joint_command`。输出：

- `episodes.jsonl`：每回合的方块初始/最终位置、是否成功、离目标距离、方块是否被抬起（区分“没抓起来”和“抓起后掉落/放偏”）、抓取次数 `grasp_attempts`（离目标 5 cm 以外夹爪由开到合的次数，大于 1 表示重抓过）、平均推理耗时；
- `summary.json`：成功率、重抓过的回合数（`regrasped`）及其中成功的回合数（`recovered`）；
- `rollouts.npz`：全部帧，格式同 `demos.npz`，可以直接回放：`pixi run python scripts/visualize_demos.py --data data/eval/act_200_60k/rollouts.npz 0-4`。
  失败回合和对照的成功回合可以用 `pixi run python scripts/render_eval_cases.py data/eval/act_200_60k` 渲染成带抓取特写的 mp4（输出在 `videos/` 子目录）。
  图像策略在某个回合中看到的相机画面（评估时不保存图像）可以用 `scripts/render_policy_cameras.py` 重新渲染：用 sim_node 同一个 `CameraRenderer`，按每帧的关节角和方块位姿渲染 overview / wrist_cam，左右并排输出 mp4 和一张联系表 PNG（输出在 `cameras/` 子目录）。方块在桌上时与策略当时的输入逐像素一致；rollouts 里只有方块 yaw（模 90°），夹在爪中倾斜的方块会画成正的。
  ```bash
  pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7                        # 默认放大 2 倍，联系表均匀取 12 帧
  pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7 --speed 0.5 --scale 3
  pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7 --frames 200 240 260 300   # 指定联系表的帧
  ```
  抓取指标和两次评估的逐回合配对比较：`pixi run python scripts/grasp_metrics.py data/eval/act_200_40k data/eval/act_200_60k`。

同一 seed 下每次评估的方块摆放序列相同，不同 checkpoint / 执行方式的成功率可以直接对比（改了 `box_radius_range` 后，各回合的方位角和朝向不变，只有 r 会变）。

### 交互演示

用户在控制面板上摆放方块，策略在 viewer 中实时完成抓取和放置。前提：已按上文建好 `.venv-lerobot`、`pixi run build` 过，并且 `outputs/` 下有 checkpoint（默认是图像版 ACT `outputs/act_img_all_hold/checkpoints/100000/pretrained_model`，见 `docs/eval_act_images_all_hold.md`；sim 需要的相机按 checkpoint 的 `config.json` 自动打开）。在工程根目录运行（不需要先 `pixi shell`）：

```bash
pixi run bash scripts/demo.sh                 # 1 倍速，默认用 480 回合训练的图像版 ACT（100k）
pixi run bash scripts/demo.sh --checkpoint outputs/act_291_envnorm_retry/checkpoints/060000/pretrained_model   # 状态版（用方块坐标）
pixi run bash scripts/demo.sh --speed 0.5     # 半速，方便看清抓取
pixi run bash scripts/demo.sh --checkpoint outputs/act_200/checkpoints/040000/pretrained_model
#   推理服务端口默认 5555，被占用时用 --port N 改
#   -- 之后的参数交给 policy_server.py，例如：-- --temporal-ensemble 0.01
```

典型用法：启动后等 viewer 和控制面板出现 → 用滑块（或 *Random*）摆好方块 → 点 *Start* → 在 viewer 中观看抓取、放置和回到初始姿态 → 换个位置再点 *Start*。

脚本依次启动 `policy_server.py`、带 viewer 的 `sim_node`（`sync_mode`，`realtime_factor` = 速度）和 `policy_demo` 控制面板；关闭面板或 viewer、或在终端按 Ctrl-C，三个进程会一起退出。日志在 `data/demo/logs/`。

控制面板：

- **滑块**（*Distance r* / *Azimuth* / *Yaw*）：方块到基座的距离 r、方位角、朝向。每次改动都会通过 `/place_box` 重置场景：机械臂回到初始姿态，方块放到指定位置。默认只能在训练范围内调（r ∈ [18, 25] cm，方位角 ∈ [5°, 40°]，viewer 地面上的白框）。勾选 *Allow outside the training range* 后范围放宽到 r ∈ [10, 40] cm、方位角 ∈ [−90°, 90°]，超出时方块变成黄色，策略可能失败。
- **Start / Stop**：运行策略。方块在目标上停稳 1 s、并且机械臂回到初始姿态后，这次运行结束（在目标上停够 5 s 也会结束，最多 `max_steps` = 900 帧）。运行结束后要重新摆放方块才能再次开始。
- **Random / Place again**：在训练范围内随机摆放；按当前滑块的值重新摆放（机械臂回到初始姿态）。
- **Speed**：0.25× – 2×，运行中也可以改（设置 `sim_node` 的 `realtime_factor` 参数）。
- 运行过程中可以在 viewer 里双击选中方块，按住 Ctrl 加鼠标右键拖动，推动方块，观察闭环策略的反应。

每次运行的结果都追加到 `data/demo/<启动时间>/episodes.jsonl` 和 `rollouts.npz`，格式与 `policy_eval` 相同，可以用 `render_eval_cases.py` / `visualize_demos.py` 离线渲染成视频。

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
| 订阅 | `/place_box` | `geometry_msgs/Pose2D` | 和 `/reset` 一样重置场景，但把方块放在 (x, y)（m）、yaw = theta（rad）处，不随机摆放；超出 `box_radius_range` × `box_azimuth_range` 时方块显示为黄色。`policy_demo` 用它摆放方块 |
| 订阅 | `/joint_command` | `sensor_msgs/JointState` | 关节目标角度（弧度）。带 `name` 时可只控制部分关节；不带 `name` 时需按顺序给出全部 6 个值；`name` 和 `position` 都为空表示“保持当前目标”（同步模式下用作应答）。超出范围会被截断 |
| 服务 | `/reset` | `std_srvs/Trigger` | 将场景重置到初始状态（开启 `randomize_box` 时随机摆放方块） |

关节顺序：`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`。每个关节都是位置控制执行器，指令即目标角度。

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_path` | 包内 `scene_pick_place.xml` | MJCF 场景文件路径 |
| `keyframe` | `""` | 初始姿态使用的 keyframe 名（如 `pickup`），为空则使用模型默认姿态 |
| `publish_rate` | `50.0` | `/joint_states`、`/box_pose` 的发布频率（Hz） |
| `realtime_factor` | `1.0` | 仿真速度相对实时的倍率，`0.0` 表示尽可能快（必须写成浮点数，写 `0` 会报类型错误）。运行中可以用 `ros2 param set /so101_mujoco_sim realtime_factor 0.5` 修改 |
| `use_viewer` | `false` | 是否打开 MuJoCo 可视化窗口（需 mjpython） |
| `publish_clock` | `true` | 是否发布 `/clock` |
| `randomize_box` | `true` | 重置时随机摆放方块（位置 + yaw） |
| `box_radius_range` | `[0.18, 0.25]` | 方块到基座的水平距离范围（m） |
| `box_azimuth_range` | `[5.0, 40.0]` | 方块方位角范围（度，从 +x 轴起算） |
| `box_rel_yaw_range` | `[-45.0, 45.0]` | 方块相对抓取角 g 的范围（度）。默认值表示不限制，随机数序列和以前相同；`[35.0, 55.0]` 用于边界评估 |
| `seed` | `0` | 随机种子（评估时固定它，便于复现） |
| `sync_mode` | `false` | 同步模式：每发布一帧状态，就等待一条时间戳相同的 `/joint_command` 后再继续步进 |
| `sync_timeout` | `1.0` | 同步模式下最多等待多久（墙钟秒）；超时则照常步进，因此没有控制端时仿真也能继续运行 |
| `show_workspace` | `true` | viewer 中在地面上画出方块的随机范围（`box_radius_range` × `box_azimuth_range`，即训练范围） |

**同步模式**：控制端（`collect_demos`、之后的策略节点）对每一条 `/joint_states` 都回复一条带相同时间戳的指令，没有新指令时回复空指令。这样无论控制端多慢、是否卡顿，指令都会在下一步之前生效，记录下来的 (状态, 动作) 严格对应，结果也可以复现。异步模式下，控制端的任何卡顿都会让机械臂停留在旧目标上，而计划仍在继续推进，导致数据中出现“动作在变、状态冻结”的片段（实测出现过）。

## 注意事项

- **方块工作空间**：Menagerie 原版 `scene_box.xml` 的方块在 x=0.5 m，而 SO-101 末端的最大水平伸展只有约 0.48 m，够不着。俯视姿态的末端只能到 z≈0.06–0.09 m（再高 `wrist_flex` 就到限位了），因此改用 4 cm 立方体；r < 0.16 m 时夹爪上的相机支架会与 `shoulder` 自碰撞。默认随机范围在这些限制之内，离线测试 600/600 成功。
- **mjpython 与 python 的区别**（macOS）：使用 `launch_passive` 的程序（如 `sim_node` 开启 viewer 时）必须用 `mjpython` 启动；而独立 viewer `python -m mujoco.viewer` 必须用普通 `python`，用 `mjpython` 会报 `Caught an unknown exception!`。
- **Ctrl-C**：在终端中直接 Ctrl-C 可以正常退出。在脚本中用 `kill -INT <pid>` 停止 `ros2 run` 启动的节点**无效**（`ros2 run` 不会转发信号），应发给整个进程组：`kill -INT -<pgid>`。
- **话题互相看不到时**：设置 `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`，或切换中间件 `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。
- **添加依赖**：用 `pixi add ros-jazzy-<包名>`（不要用 `rosdep`）；PyPI 包用 `pixi add --pypi <包名>`，需有 Python 3.12 的版本。
- **重新构建**：使用了 `--symlink-install`，修改 `.py` 文件无需重新构建；新增文件、入口或模型文件后需要再次 `pixi run build`。

## 致谢

SO-101 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（`robotstudio_so101`，commit `0059d43`），其许可证见模型目录下的 `LICENSE`。
