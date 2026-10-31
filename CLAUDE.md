# CLAUDE.md

Physical AI 课程期末作业（个人独立完成）：MuJoCo 仿真 SO-101 机械臂 → ROS 2 采集示教数据 → LeRobot 训练 ACT 策略 → 策略在仿真中闭环控制。
整体计划见 `physical_ai_final_project_plan.md`，使用方法与接口见 `README_CN.md`（默认的 `README.md` 是日语版，两者内容保持一致）。用户使用中文交流。

## 当前进度

- M0（环境）、M1（`sim_node`：`/joint_states`、`/joint_command`、`/box_pose`、`/clock`、`/reset`）已完成并验证。
- M2 已完成：`scene_pick_place.xml`（4 cm 立方体，重置时随机摆放）、IK 脚本专家（`expert.py`）、`collect_demos`（同步模式下录制 rosbag）、`export_dataset`（bag → npz）、`scripts/npz_to_lerobot.py`（npz → LeRobot 0.6.1 数据集，ACT 冒烟训练已通过）。数据在 `data/`（不纳入 git）。
- 数据已补采到 200 回合（0–49 用 seed 0，50–199 用 seed 1；追加采集必须换 seed，评估用 seed 100，补充评估用 seed 101）。
- 按失败分析 4.1 改了专家（抓取点朝固定爪偏移 9 mm，`box_radius_range` 下限改为 0.18 m），用 seed 2 重新采集了 200 回合：`data/demos_offset/` → `data/demos_offset.npz`（200/200 成功）。旧数据 `data/demos*` 是偏移 0、r 从 0.17 m 起采集的，两者不要混用。
- M3、M4、M5 已完成（报告：`docs/report.md` / `report_ja.md` 及对应 PDF；演示视频已做完）。M3：ACT 基线（state + environment_state，无图像），用 `notebooks/train_act_colab.ipynb` 在 Colab Pro 上训练，对比 50 / 200 回合。SmolVLA 只作为扩展阶段（需要先给 `sim_node` 加图像观测）。
- 偏移数据训练的 ACT（seed 100 评估）：40k 49/50、60k 47/50，差异不显著。剩下的失败都在相对抓取角 g ≈ +45° 处（专家在这里换一对侧面抓，抓取方向跳 90°，策略把两种抓法平均，手腕少转约 30°），见 `docs/eval_act_offset.md`；指标用 `scripts/grasp_metrics.py` 计算。
- 重新抓取（已完成）：`expert.PickPlaceEpisode` 分段规划，抬起后检查方块，没抬起就重抓；`collect_demos` 可以用 `inject_failure_prob` 让第一次抓取故意失败（`wrist` / `offset` 两种，范围用 `scripts/check_retry.py` 离线选定），`episodes.jsonl` 的 `clip_start` 让 `export_dataset` 只导出重抓开始之后的帧。恢复数据：`data/demos_retry`（seed 3，100 回合中重抓成功 91 回合），与偏移数据合并为 `data/demos_offset_retry.npz`（291 回合）。边界评估：sim 加 `box_rel_yaw_range:=[35.0,55.0]`（|g| ∈ [35°, 45°]）。不注入时 `plan()` / 采集结果和以前逐帧一致。
- **当前最好的模型**：图像版 `outputs/act_img_all_hold`（480 回合，关节角 + `overview` + `wrist_cam`，无方块坐标）100k：seed 100 普通 47/50、边界 50/50、seed 101 100 回合 99/100，合计 196/200；成功后不再乱动（0/196）。失败是搬运途中掉落（2）、手腕停在两种抓法中间（1）、g ≈ 26° 手腕转过头（1），见 `docs/eval_act_images_all_hold.md`。交互演示默认用它。状态版对照 `outputs/act_291_envnorm_retry`（291 回合，ENV 归一化）60k：三组 200/200（`data/eval/act_291_envnorm_retry_60k*`）。
- 交互演示：`scripts/demo.sh`（`pixi run bash scripts/demo.sh [--speed 0.5]`）启动 policy_server、带 viewer 的 sim 和 `policy_demo`（Tk 面板：滑块摆放方块 → `/place_box`；开始/停止；速度 → 运行时设置 sim 的 `realtime_factor`）。`policy_demo` 继承 `PolicyEvaluator`（`begin_episode` / `end_episode` / `_episode_over`）。目标仍是固定的（观测里没有目标）。
- 完成后的保持：`collect_demos` 的 `settle_time` 从 0.5 s 改为 2.0 s（回合末尾手臂停在初始姿态、方块在目标上，比 1 s 的动作块长）。之前的数据只有 25 帧保持，策略回到初始姿态后会再去方块那里抓一次。同一 seed 重采时前 506 帧与旧数据一致，只多出末尾 75 帧。现有 `data/` 的数据和模型都是 0.5 s 时的，需要重采、重训。
- 边界补充数据（代码已改好，尚未正式采集）：新注入模式 `midway`（手腕转到两种抓法中间 + 朝活动爪平移 7–11 mm），以及抓空后方块没平放时先张开夹爪、竖直抬高让方块倒平再判断（立方体侧躺也算平放，`is_flat`）。第一次采集 seed 5 时只成功 72/100（当时只张开不抬高，方块斜靠在固定爪上），修改后离线可用率约 90%，需要重采。计划用 seed 4（正常示教）和 seed 5（`midway` 恢复示教），两批都用 `box_rel_yaw_range:=[30.0,60.0]`（|g| ∈ [30°, 45°]），各 100 回合，命令见 README。`check_retry.py` / `check_expert.py` 加了 `--rel-yaw`。旧模式下，以前成功的回合规划逐帧不变。
- 新数据已采完并渲染：`data/demos_all_hold.npz` → `data/lerobot/so101_sim_pick_place_images_all_hold`（480 回合 = 正常 200 + 重抓 94 + 边界 100 + 边界 `midway` 重抓 86，274,790 帧；打包为 `data/lerobot/so101_images_all_hold.tar`）。在 Colab 上训练 `act_img_all_hold`（`notebooks/train_act_images_colab.ipynb` 2.3 节，100k 步），评估任务已写进 `scripts/run_evals.sh`。`scripts/after_success.py` 统计成功后手臂回到初始姿态之后有没有再动：旧图像模型几乎每回合都会再动（seed 100 49/49），状态模型 0/150。
- M4：`scripts/policy_server.py`（`.venv-lerobot` 中运行，本地 TCP + 每行一个 JSON）+ `policy_eval` 节点（sync_mode，等同一时间戳的 `/box_pose` 到齐后再构造观测）。

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
- 从 pixi 环境启动 venv 的 Python 时，pixi 的 `PYTHONPATH` 会混入 ROS 环境的 numpy/libomp，torch 报 `OMP: Error #15`（`policy_server.py` 只剔除 `.pixi` 下的 site-packages；venv 可能是基于 pixi 的 Python 建的，标准库也在 `.pixi` 下，整段剔除会报 `No module named 'ctypes'`）。输出重定向到文件时，print 要加 flush，否则脚本等不到日志里的就绪信息。
- **SO-101 工作空间**：末端（`gripperframe` site，x 轴为接近方向，z 轴为开合方向）最大水平伸展约 0.48 m；俯视姿态只能到 z≈0.06–0.09 m（受 `wrist_flex` 限位约束）；r < 0.16 m 时相机支架会与 `shoulder` 自碰撞（r ≈ 0.17 m 的部分朝向下也会，抓取偏移后更明显，所以随机范围下限是 0.18 m）。IK 不检查碰撞，改动随机范围或专家后要用 `scripts/check_expert.py` 离线重新验证。
- `mj_jacSite` 依赖 `mj_comPos`：只调用 `mj_kinematics` 时，雅可比全为零。
- 仿真时间转 `Time` 时，要先四舍五入到整数纳秒再 `divmod`；先取 `int(seconds)` 再对小数部分四舍五入，会在 x.9999999 处倒退 1 秒。
- 不要在步进物理时持有回调也需要的锁：Python 的 `Lock` 不保证公平，仿真跟不上目标速度时回调会被饿死（指令几秒不生效）。
- 采集/评估要用 `sync_mode`；异步模式下，控制端卡顿会产生“动作变化、状态冻结”的坏数据。
- **LeRobot ACT 默认不归一化 `observation.environment_state`**：默认 `normalization_mapping` 里没有 `ENV`，这类特征会按 `IDENTITY` 处理。训练时要传 `--policy.normalization_mapping='{"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ENV": "MEAN_STD", "ACTION": "MEAN_STD"}'`（会整体替换默认映射，所以要写全）。现有 `outputs/` 里的模型都是没归一化时训练的。可以查 checkpoint 中 `policy_preprocessor.json` 的 `norm_map` 确认。
- 现有策略没抓起来时会空手去目标、回到初始姿态，然后把整个任务**从头再做一遍**（初始姿态 + 方块在原处 = 回合开始时的观测）。所以评估的成功率取决于 `max_steps`（现在默认 900，以前是 600），比较时要看 `success_frame` / `successes_by_frame_600`。
- ROS 参数类型严格：浮点参数要写 `0.0`，写 `0` 会报 `InvalidParameterTypeException`。

## 约定

- `sim_node` 中只有主（仿真）线程访问 `self.data`。ROS 回调（executor 线程）把指令和重置请求放进 `self.pending`（`Condition`，只在拷贝时短暂持有），由主循环在每次步进前通过 `_apply_requests()` 应用。
- 控制端发布的 `/joint_command` 时间戳等于它所回应的 `/joint_states` 的时间戳（同步模式和数据导出都依赖这一点）；没有新指令时发空指令作为应答。
- 仿真时间 = `time_offset + data.time`，重置时累加 offset，保证 `/clock` 单调递增（rosbag 依赖这一点）。
- 依赖用 `pixi add ros-jazzy-<pkg>` / `pixi add --pypi <pkg>` 添加，不用 rosdep。LeRobot（含 torch）暂不装进 ROS 环境，避免与 RoboStack 的 numpy 等版本冲突；训练在 Colab 上进行（本机 MPS 上 ACT 约 5 step/s @ batch 8，可作备选）。本地测试 / 推理用工程下的 `.venv-lerobot`（uv 建的 venv，`lerobot[dataset,training]==0.6.1`，不纳入 git）。
