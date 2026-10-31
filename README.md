# SO-101 MuJoCo × ROS 2：模倣学習の講義課題

日本語 | [中文](README_CN.md)

Physical AI 講義の期末課題。MuJoCo 上で SO-101 ロボットアームをシミュレーションし、ROS 2 でシミュレーションと接続してデモデータを収集する。LeRobot（ACT 方策）でモデルを学習させ、最後に学習済みモデルでシミュレーション内のアームを閉ループ制御し、ピック・アンド・プレースのタスクを行う。

方式の選定と計画の詳細は [physical_ai_final_project_plan.md](physical_ai_final_project_plan.md)（中国語）を参照。

## 進捗

| マイルストーン | 内容 | 状態 |
|---|---|---|
| M0 環境 | Mac 上で ROS 2 と MuJoCo を動かし、SO-101 を読み込む | ✅ 完了 |
| M1 インタフェース | MuJoCo ノードが `/joint_states` を配信し、`/joint_command` を受け取る | ✅ 完了 |
| M2 データ | スクリプト方策で rosbag を記録し、LeRobot データセットに変換する | ✅ 完了 |
| M3 学習 | Colab 上で ACT（状態入力）を学習し、50 / 200 エピソードを比較する | ✅ 完了 |
| M4 閉ループ | 方策ノードでシミュレーション内の SO-101 を制御し、成功率を集計する | ✅ 完了（下記「結果」を参照） |
| 改善 | 把持オフセット、キューブ特徴量の正規化、把持失敗後に掴み直すデモデータ | ✅ 完了 |
| 画像入力 | 観測からキューブ座標を外し、関節角 + カメラ 2 台（`overview` / `wrist_cam`）の画像だけで ACT を学習する。完了後の保持と境界付近のデータを補充 | ✅ 完了（196/200、最終モデル） |
| デモ | 対話型デモ画面：パネルでキューブを配置し、方策の動作をリアルタイムで見る | ✅ 完了（「対話型デモ」を参照） |
| M5 仕上げ | レポートとデモ動画の作成 | ✅ 完了（レポート：[docs/report_ja.pdf](docs/report_ja.pdf)、中国語版 [docs/report.pdf](docs/report.pdf)） |

## 結果

学習に使っていないキューブ配置で、`sim_node` の同期モードにより閉ループ評価した。テストは 3 組・合計 200 エピソード：seed 100 の通常配置 50、seed 100 の境界配置 50（`box_rel_yaw_range:=[35.0,55.0]`、|g| ∈ [35°, 45°]、掴み方が切り替わる難しい配置）、seed 101 の通常配置 100。

**最終モデルは画像入力の `act_img_all_hold`（100k）で、キューブ座標を与えずに 196/200 成功**：

| モデル | 観測 | 学習データ | seed 100 通常 | 境界 | seed 101 | 合計 | 成功時の配置誤差（平均） |
|---|---|---|---|---|---|---|---|
| **act_img_all_hold（100k、最終モデル）** | 関節角 + `overview` + `wrist_cam`（240×320 RGB）、キューブ座標なし | 480 エピソード（完了後 2 s の保持 + 境界データ） | 47/50 | **50/50** | 99/100 | **196/200 = 98%** | 0.64–0.69 cm |
| act_img_291（60k） | 同上 | 291 エピソード（下記の act_291_envnorm_retry と同じデモ） | 49/50 | 47/50 | 98/100 | 194/200 = 97% | 0.54–0.63 cm |
| act_291_envnorm_retry（60k、比較用） | 関節角 + キューブの位置姿勢 | 291 エピソード | 50/50 | 50/50 | 100/100 | 200/200 = 100% | 0.48–0.60 cm |

- 画像入力モデルでも、キューブを持ち上げさえすれば配置精度と速度は状態入力モデルとほぼ同じである。推論は約 7–8 ms/ステップ（2 枚の画像に対する ResNet18 を含む、CPU）で、50 Hz の制御に間に合う。
- act_img_291 は、成功して初期姿勢に戻った後、ほぼ毎回もう一度キューブの方へ手を伸ばしていた（seed 100 で 49/49）。デモ末尾の保持が 0.5 s しかなく、1 s のアクションチャンクより短かったためである。保持を 2 s にした act_img_all_hold では、これが 0/196 になった。
- act_img_291 の失敗は境界付近（|g| ≳ 35°）に集中していた。境界データ（通常 100 + 掴み直し `midway` 86 エピソード）を加えると境界評価は 47/50 → 50/50 になった（エピソードごとの対応比較で 0 対 3、p = 0.25 でまだ有意ではない）。
- act_img_all_hold に残る 4 件の失敗：搬送中の落下 2、手首が 2 種類の掴み方の中間で止まる 1、g ≈ 26° で手首を回しすぎる 1。

状態入力モデルの改善の経緯（seed 100 の 50 エピソード）：

| モデル | 学習データ | checkpoint | 成功率 | 成功エピソードの配置誤差（平均） |
|---|---|---|---|---|
| act_50 | エキスパートに把持オフセットなし、最初の 50 エピソード | 60k | 30/50 = 60% | 0.69 cm |
| act_200 | エキスパートに把持オフセットなし、200 エピソード | 60k | 48/50 = 96% | 0.70 cm |
| act_200（オフセット） | エキスパートの把持オフセット 9 mm、200 エピソード | 40k | **49/50 = 98%** | 0.88 cm |
| act_200（オフセット） | 同上 | 60k | 47/50 = 94% | 0.66 cm |
| act_200_envnorm | 同上、キューブ特徴量に `MEAN_STD` 正規化を追加 | 40k | 49/50 = 98% | 0.58 cm |
| **act_291_envnorm_retry** | オフセットデータ 200 エピソード + 再把持デモ 91 エピソード、キューブ特徴量を正規化 | 60k | **50/50 = 100%** | **0.48 cm** |

- 失敗はすべて把持の段階（キューブを持ち上げられない）で起きている。持ち上げさえすれば、すべて目標に置けている。
- データ量を 50 から 200 エピソードに増やすと、成功率は 60% から 96% に上がった。エピソードごとに対応をとった差は有意である（18:0）。
- 失敗分析により、エキスパートが把持するときの固定ジョーとキューブの隙間が約 0.6 mm しかなく、模倣誤差を許容する余裕がないことがわかった。エキスパートに 9 mm のオフセットを加えると、この種の失敗はなくなった。残った失敗は、相対把持角が ±45° に近い配置に集中している。エキスパートはここで掴む側面の組を切り替え、把持方向が 90° 跳ぶため、方策が 2 種類の掴み方を「平均」してしまう。
- オフセットモデルの 40k と 60k の成功率の差は有意ではない（p = 0.5）。60k のほうが配置は正確である。
- オフセットモデルの r の下限は 0.18 m（以前は 0.17 m）なので、評価時の配置は上の 2 行とは近似的にしか対応していない。
- 「1 回目の把持に失敗 → 掴み直す」デモ（`data/demos_retry`、seed 3、100 エピソードを収集し、掴み直しに成功した 91 エピソードについて掴み直し以降の区間だけを書き出したもの）を加えると、`act_291_envnorm_retry` は seed 100 の 50 エピソードですべて成功し、しかもすべて 1 回目で持ち上げた（`regrasped` = 0、すべて 600 フレーム以内に完了）。配置誤差も各モデルの中で最小である。seed 101 の 100 エピソード（うち 24 エピソードが |g| ≥ 35°）でも 100/100 だった。
- 比較：再把持データを含まないオフセットモデル（act_200、60k）は境界評価（|g| ∈ [35°, 45°]）で 49/50 だったが、そのうち 14 エピソードで掴み直しが必要だった（掴み直しを学習していないため、タスク全体を最初からやり直している）。600 フレーム以内の成功は 36 エピソードだけである。

レポートと詳細な分析：

- [docs/report_ja.pdf](docs/report_ja.pdf)（[Markdown 版](docs/report_ja.md)）：最終レポート。中国語版は [docs/report.pdf](docs/report.pdf)
- [docs/eval_act_200_failure_analysis.md](docs/eval_act_200_failure_analysis.md)（中国語）：act_200 の失敗原因（固定ジョーがキューブの上面に乗る）と改善案
- [docs/eval_act_50_vs_200.md](docs/eval_act_50_vs_200.md)（中国語）：50 / 200 エピソードの比較
- [docs/eval_act_offset.md](docs/eval_act_offset.md)（中国語）：把持オフセット追加後の評価と、±45° で掴み方が切り替わる箇所での新たな失敗メカニズム
- [docs/eval_act_images.md](docs/eval_act_images.md)（中国語）：画像入力 ACT（291 エピソード）と状態入力モデルの比較
- [docs/eval_act_images_all_hold.md](docs/eval_act_images_all_hold.md)（中国語）：480 エピソードの画像入力 ACT（最終モデル）の評価と失敗分析

## 環境

| 項目 | バージョン |
|---|---|
| プラットフォーム | macOS（Apple Silicon、osx-arm64）、macOS 14 以上が必要 |
| パッケージ管理 | [pixi](https://pixi.sh)（依存関係と固定バージョンは `pixi.toml` / `pixi.lock` を参照） |
| ROS 2 | Jazzy、[RoboStack](https://robostack.github.io/jazzy.html) から導入（`robostack-jazzy` チャンネル） |
| Python | 3.12（RoboStack Jazzy の固定バージョン） |
| MuJoCo | 3.15（PyPI の `mujoco` パッケージ） |

## ディレクトリ構成

```
adv1_final_project/
├── README.md                              # 日本語版の README（本ファイル）
├── README_CN.md                           # 中国語版の README
├── CLAUDE.md                              # Claude Code 向けのプロジェクト説明
├── physical_ai_final_project_plan.md      # 方式の選定と計画
├── pixi.toml / pixi.lock                  # 環境の定義と固定バージョン
├── docs/
│   ├── eval_act_200_failure_analysis.md   # ACT 200 エピソード評価の失敗分析と改善案
│   ├── eval_act_50_vs_200.md              # 50 / 200 エピソードで学習した ACT の閉ループ評価の比較
│   ├── eval_act_offset.md                 # 把持オフセット追加後に再学習した ACT（40k / 60k）の評価と新たな失敗メカニズム
│   ├── eval_act_images.md                 # 画像入力 ACT（291 エピソード）の評価と状態入力モデルとの比較
│   ├── eval_act_images_all_hold.md        # 画像入力 ACT（480 エピソード、最終モデル）の評価
│   ├── report_ja.md / report_ja.pdf       # 最終レポート（日本語）
│   ├── report.md / report.pdf             # 最終レポート（中国語）
│   └── figures_ja/, figures/              # レポートの図
├── notebooks/
│   └── train_act_colab.ipynb              # Colab で ACT を学習する（M3）
├── scripts/
│   ├── check_expert.py                    # ROS を使わず、ランダム範囲全体でエキスパートをオフライン検証する（開閉方向の誤差も模擬できる）
│   ├── demo.sh                            # 対話型デモ：推論サーバ、viewer 付きシミュレーション、操作パネルを 1 コマンドで起動する
│   ├── check_retry.py                     # ROS を使わず、失敗の注入と掴み直しをオフライン検証する（注入範囲の選定用）
│   ├── grasp_metrics.py                   # 評価の把持指標（開閉方向のずれ、向きの誤差、最初の接触）とエピソードごとの対応比較
│   ├── npz_to_lerobot.py                  # npz → LeRobot データセット（LeRobot を入れた環境で実行する）
│   ├── policy_server.py                   # 推論サーバ：LeRobot の venv で方策を読み込み、ローカル TCP で行動を返す
│   ├── render_eval_cases.py               # 評価中の失敗エピソードといくつかの成功エピソードを mp4 に描画する（把持部分の拡大付き）
│   ├── render_policy_cameras.py           # 画像入力方策が評価エピソードで見たカメラ画像（overview / wrist_cam）を描画し直す
│   └── visualize_demos.py                 # 収集したデータをアニメーションとして再生する（mp4 またはウィンドウ）
├── data/                                  # 収集したデータ（git 管理外）
└── ros2_ws/                               # colcon ワークスペース
    └── src/
        └── so101_mujoco/                  # ament_python パッケージ
            ├── package.xml
            ├── setup.py / setup.cfg
            ├── resource/so101_mujoco
            ├── models/robotstudio_so101/  # SO-101 の MJCF モデル（MuJoCo Menagerie 由来）
            │   ├── so101.xml              # アーム本体
            │   ├── scene.xml              # アーム + 床
            │   ├── scene_box.xml          # アーム + 床 + キューブ（Menagerie の元のシーン、キューブに届かない）
            │   ├── scene_pick_place.xml   # ピック・アンド・プレースのシーン（既定のシーン、本プロジェクトで追加）
            │   └── assets/*.stl
            └── so101_mujoco/
                ├── sim_node.py            # MuJoCo シミュレーションノード
                ├── wave_demo.py           # 正弦波の関節指令を送るデモノード
                ├── kinematics.py          # 下向き把持のための数値 IK
                ├── expert.py              # スクリプト化されたエキスパート：ピック・アンド・プレースの軌道全体を計画する
                ├── collect_demos.py       # 収集ノード：エキスパートのデモ → エピソードごとに 1 つの rosbag
                ├── export_dataset.py      # rosbag → 揃えた学習フレーム（.npz）
                ├── policy_eval.py         # 閉ループ評価ノード：推論サーバを呼んでシミュレーションを制御し、成功率を集計する
                └── policy_demo.py         # 対話型デモノード：Tk の操作パネルでキューブを配置し、方策をリアルタイムで実行する
```

## インストール

```bash
# 1. pixi をインストールする
brew install pixi

# 2. プロジェクトのルートで全依存関係をインストールする（ROS 2 Jazzy、colcon、ビルドツール、mujoco）
pixi install

# 3. ROS 2 ワークスペースをビルドする
pixi run build
```

## 使い方

新しいターミナルを開くたびに、まず環境に入ってワークスペースを source する：

```bash
pixi shell
source ros2_ws/install/setup.zsh
```

### シミュレーションの起動

```bash
# 可視化ウィンドウ付き（macOS では mjpython で起動する必要がある）
ros2 run --prefix mjpython so101_mujoco sim_node --ros-args -p use_viewer:=true

# 画面なし（データ収集時に使う）
ros2 run so101_mujoco sim_node
```

### アームを動かす

```bash
# デモ：複数の関節が正弦波で動き、グリッパが開閉する
ros2 run so101_mujoco wave_demo

# 指令を手動で 1 回送る（単位はラジアン。一部の関節だけを指定してもよい）
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
  "{name: [shoulder_pan, elbow_flex], position: [0.5, 0.8]}"

# 状態を確認する
ros2 topic echo /joint_states
ros2 topic echo /box_pose

# シーンをリセットする
ros2 service call /reset std_srvs/srv/Trigger
```

### デモデータの収集（M2）

データの流れ：`collect_demos`（エキスパート + rosbag 記録）→ `export_dataset`（rosbag → npz）→ `scripts/npz_to_lerobot.py`（npz → LeRobot データセット）。

```bash
# ターミナル 1：同期モード + 全速でシミュレーションを実行（同期モードは後述）
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0

# ターミナル 2：50 エピソードを収集する（1 エピソードはシミュレーション時間で約 10 s、M1 上で約 4.5 s）
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=50 -p output_dir:=data/demos

# 追加で収集するときは seed を変えること。変えないと既存エピソードと同じキューブ配置が繰り返される（番号は自動で続きから振られる）
# オフセットなしの旧データ：エピソード 0–49 は seed 0、50–199 は seed 1（data/demos → data/demos.npz）。評価は seed 100（補足評価は seed 101）
# 把持オフセット追加後のデータ（現在使用中）：data/demos_offset（seed 2、200 エピソード）→ data/demos_offset.npz

# エキスパートは把持時にグリッパを固定ジョー側へ grasp_offset（既定 0.009 m、expert.GRASP_OFFSET を参照）だけずらし、
# 固定ジョーに余裕を持たせる。エピソード 0–199 はオフセット 0、r ∈ [0.17, 0.25] m のときに収集したもの。変更したらまずオフラインで検証する：
#   pixi run python scripts/check_expert.py --grasp-offset 0.0 0.009 --tool-error -0.005 0 0.005

# npz に書き出す（成功したエピソードだけを書き出す）
ros2 run so101_mujoco export_dataset data/demos data/demos.npz
```

**掴み直し（リカバリ）のデモ**：エキスパートは把持のたびにキューブが持ち上がったか（z > 3 cm）を確認する。持ち上がっていなければ、現在位置からキューブ（現在の位置姿勢を基準とする）の上方に戻って掴み直す。再試行は最大 2 回。このデータを収集するには、一部のエピソードで 1 回目の把持をわざと失敗させる（`inject_failure_prob`）。注入の方式は次のとおりで、エピソードごとにランダムに 1 つ選ぶ：

- `wrist`：開閉方向を ±25–35° 回転させ、同時に可動ジョー側へ 10–16 mm ずらす。ACT の g ≈ 45° での失敗を模したもの。手首を回すだけでは失敗しない。グリッパが閉じるときにキューブがまっすぐに戻されるからである。
- `offset`：可動ジョー側へ 12–20 mm ずらす。固定ジョーがキューブの上面に乗る。
- `midway`（境界データ用。既定の `inject_modes` には含まれない）：画像入力の ACT の |g| ≳ 35° での失敗を模したもの。手首を 2 種類の掴み方の中間まで戻し（ずれ = −g × 0.7–1.0、`wrist_roll` ≈ 1.6）、同時に可動ジョー側へ 7–11 mm ずらす。手首を回すだけでは、98% はまだキューブを持ち上げられる。画像入力モデルの失敗エピソードも、いずれも約 0.7–1.1 cm ずれていた（`docs/eval_act_images.md` を参照）。

掴み損ねた後、キューブがどれかの面を下にして平らに置かれていない場合（多くは閉じたグリッパに斜めに挟まり、開いた後も固定ジョーに斜めにもたれかかっている）、エキスパートはまずグリッパを開き、エンドエフェクタを鉛直に 2.5 cm 持ち上げて（遠くて届かないときは 1 cm まで段階的に下げる）キューブを倒して平らにし、もう一度確認する。それでも平らでなければ諦める。立方体が横倒しになっていても平らに置かれているとみなす（`expert.is_flat`。このとき `box_yaw` は水平な軸から求める）。キューブが直立している場合の経路は以前とフレーム単位で同じである。オフライン検証（`midway`、|g| ∈ [30°, 45°]、2 つの seed で各 100 エピソード）：掴み損ね 93 + 88、掴み直し成功 93 + 86。

失敗を注入した後、エキスパートは 0–0.8 s 目標に向かって動き続けてから失敗に「気づく」。これは、ACT がアクションチャンク単位で開ループに実行するときの反応の遅れを模したものである。こうしたエピソードは掴み直しに成功した場合だけ残す。`episodes.jsonl` に `clip_start`（掴み直しを始めたフレーム番号）を記録し、`export_dataset` はこのフレーム以降の区間だけを書き出す。こうすることで、方策はリカバリだけを学び、わざと掴み損ねる動作は学ばない。

```bash
# まず注入範囲をオフラインで検証する（ROS 不要）：注入で本当に失敗するか、掴み直しの成功率はどうか
pixi run python scripts/check_retry.py

# リカバリのデモを収集する（sim_node と collect_demos で同じ新しい seed を使う）
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=3
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 -p output_dir:=data/demos_retry \
    -p inject_failure_prob:=1.0 -p seed:=3

# 境界付近（|g| ∈ [30°, 45°]）の追加データ：box_rel_yaw_range を [30, 60] にする（rel 30–45 → g +30…+45、45–60 → g −45…−30）
#   オフライン検証：pixi run python scripts/check_retry.py --modes midway --rel-yaw 30 60 --delay 0 0.3 -n 60
#                  （上記のとおり、使える割合は約 90%）。エキスパート自体：pixi run python scripts/check_expert.py --rel-yaw 30 60 --grid 5 8 12
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=4 -p "box_rel_yaw_range:=[30.0,60.0]"
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 -p output_dir:=data/demos_boundary -p seed:=4
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=5 -p "box_rel_yaw_range:=[30.0,60.0]"
ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 -p output_dir:=data/demos_boundary_retry -p seed:=5 \
    -p inject_failure_prob:=1.0 -p "inject_modes:=['midway']" -p "inject_delay:=[0.0,0.3]"

# 通常のデモとまとめて書き出す
ros2 run so101_mujoco export_dataset data/demos_offset data/demos_retry data/demos_offset_retry.npz
```

LeRobot を入れた環境（Colab など。pixi 環境には入れないこと）で LeRobot データセットに変換する（lerobot 0.6.1 で確認済み。ACT をそのまま学習できる）：

```bash
pip install 'lerobot[dataset,training]==0.6.1'
python scripts/npz_to_lerobot.py data/demos.npz --repo-id local/so101_sim_pick_place \
    --root data/lerobot/so101_sim_pick_place
```

**データの確認**：`scripts/visualize_demos.py` は `data/demos.npz` を読み込み、選んだエピソードをアニメーションとして再生する。左側は記録した関節角とキューブの位置姿勢から再現した MuJoCo の画面（運動学だけを計算し、物理シミュレーションは行わない）。右側は各関節の状態（実線）と行動（破線）の曲線、およびキューブの高さで、再生に合わせて動くカーソルが付く。ROS は不要。

```bash
pixi run python scripts/visualize_demos.py --list         # 全エピソードを一覧表示する（フレーム数、キューブの初期位置と向きなど）
pixi run python scripts/visualize_demos.py 0 3 10-12       # エピソードごとに data/viz/episode_NNNN.mp4 を生成する（1 エピソードあたり約 13 s）
pixi run python scripts/visualize_demos.py 5 --show        # ウィンドウで直接再生する
pixi run python scripts/visualize_demos.py 0-9 --stride 2 --speed 2   # 1 フレームおきに描画し 2 倍速にして、生成を速くする
```

エキスパートの動作を見たいときは、viewer 付きの sim（`--prefix mjpython ... -p use_viewer:=true -p sync_mode:=true`、`realtime_factor` は 1.0 のまま）を起動してから `collect_demos` を実行してもよい。

**タスク**：緑の立方体（4 cm）を持ち上げ、赤い枠（中心 (0.22, −0.12)）の上に置く。`/reset` のたびに、キューブは極座標 r ∈ [0.18, 0.25] m、方位角 ∈ [5°, 40°] の範囲にランダムに置かれ、yaw もランダムになる。成功の判定：キューブ中心と目標の水平距離が 3 cm 未満で、かつキューブが床の上に立っている（z < 3 cm）。

**エキスパート**（`expert.py`）：キューブの位置姿勢から、数値 IK（MuJoCo のヤコビアン + 減衰最小二乗法）で軌道を計画する。関節空間でキューブの上方へ移動 → 鉛直に下降 → 把持 → 持ち上げ → 目標の上方へ水平移動 → 下降 → 解放 → 持ち上げ → 初期姿勢に戻る、の流れで、合計約 10 s（506 フレーム）。その後 `collect_demos` が、アームが初期姿勢で止まりキューブが目標の上にある静止フレームを `settle_time`（既定 2.0 s、100 フレーム）だけ記録する。ACT の 1 s のアクションチャンクより長くすることで、方策はタスク完了後に静止することを学べる。2026-10 より前のデータは 0.5 s で、方策は初期姿勢に戻った後、もう一度キューブを掴みに行っていた。`collect_demos` は `PickPlaceEpisode` で 2 段階に分けて計画する。持ち上げた後にキューブを確認し、持ち上がっていなければ掴み直す（上記を参照）。

**rosbag の内容**：エピソードごとに 1 つの mcap bag（`data/demos/episode_NNNN`）で、`/joint_states`、`/box_pose`、`/joint_command` を含む。`data/demos/episodes.jsonl` には各エピソードの初期位置姿勢と成否を記録する。各指令のタイムスタンプは、それが応答する関節状態のタイムスタンプと同じで、書き出し時にはこれを使って観測と行動を対応づける。

**データセットのフィールド**（50 fps）：

| フィールド | 次元 | 内容 |
|---|---|---|
| `observation.state` | 6 | 関節角（rad）。順序は後述の関節順序と同じ |
| `observation.environment_state` | 5 | キューブの `[x, y, z, cos 4ψ, sin 4ψ]`（ψ は yaw。立方体は 90° ごとに対称なので、把持方向は ψ mod 90° だけで決まる） |
| `action` | 6 | 送った関節目標角（rad） |

### 方策の学習（M3）

Colab（GPU）上で `notebooks/train_act_colab.ipynb` を実行する。`data/demos.npz` と `scripts/npz_to_lerobot.py` を Google Drive の `so101_act/` ディレクトリにアップロードし、順番に実行すればよい。notebook は 2 つのデータセット（全 200 エピソード / 最初の 50 エピソード）に変換し、それぞれ ACT を 1 つ学習させる：

| 設定 | 値 |
|---|---|
| 入力 | `observation.state` + `observation.environment_state`（画像なし） |
| `chunk_size` / `n_action_steps` | 50 / 50（1 s のアクションチャンク） |
| batch / ステップ数 / 保存間隔 | 64 / 60k / 10k |
| 正規化 | STATE / ENV / ACTION いずれも `MEAN_STD`（`--policy.normalization_mapping`） |
| その他 | LeRobot 0.6.1 の既定値（lr 1e-5、KL の重み 10） |

現時点で最良のモデル `outputs/act_291_envnorm_retry` は、オフセットデータと再把持データをまとめて書き出した `data/demos_offset_retry.npz`（291 エピソード）で、キューブ特徴量を正規化して 60k ステップ学習したものである（対話型デモの既定は現在、画像入力の `act_img_all_hold` 100k）。それ以前のモデルは、把持オフセット追加後のデータ `data/demos_offset.npz` で学習した（notebook が読むファイル名は `demos.npz` のままなので、アップロード時に名前を変えればよい）。ローカルの `outputs/act_200`、`outputs/act_50` はオフセットデータで学習したモデル、`outputs/*_backup` はオフセットなしの旧データで学習したモデルである。対応する評価結果は `data/eval/` に同じ命名で置いてある。

`act_50` / `act_200` とその `_backup` は、いずれもキューブ特徴量を**正規化していない**状態で学習したものである（`act_200_envnorm`、`act_291_envnorm_retry` は正規化済み）。ACT の既定の `normalization_mapping` には `VISUAL / STATE / ACTION` しかなく、`observation.environment_state` の種類は `ENV` なので、`IDENTITY` として扱われる（統計量は checkpoint に書き込まれるが、使われない）。notebook では現在 `ENV` を明示的に追加しており、200 エピソードの run は `act_200_envnorm` に名前を変えたので、Drive 上の旧 `act_200` は上書きされない。正規化の方式は checkpoint の `policy_preprocessor.json`（`norm_map`）に保存されており、`policy_server.py` はそれに従って処理するため、新旧どちらのモデルでも推論コードを変える必要はない。ただし、旧い重みをこの json の書き換えだけで正規化入力に切り替えることはできず、再学習が必要である。

学習曲線は W&B の `so101-act` プロジェクトに記録する（run 名は `act_200_envnorm` / `act_50`、以前の run は `act_200`。Colab の Secrets に `WANDB_API_KEY` を設定する必要があり、使わない場合は notebook の `USE_WANDB` を `False` にする）。checkpoint は W&B にはアップロードせず、Drive の `so101_act/outputs/<run>/checkpoints/NNNNNN/pretrained_model` に出力する。最後のセルですべての `pretrained_model` をまとめてダウンロードできる。ローカルではプロジェクトのルートの `outputs/`（git 管理外）に展開する。

学習済みモデルのパラメータ（checkpoint）は Google Drive で共有している：[https://drive.google.com/drive/folders/10yvLc3Go5icyDsADPANyEJj__5-9XT9s?usp=drive_link](https://drive.google.com/drive/folders/10yvLc3Go5icyDsADPANyEJj__5-9XT9s?usp=drive_link)。ダウンロードしたものをプロジェクトのルートの `outputs/` に展開すれば、そのまま評価やデモに使える。

### 閉ループ評価（M4）

LeRobot と torch は pixi 環境に入れない。方策は別の venv で `scripts/policy_server.py` が読み込み、ROS ノード `policy_eval` はローカル TCP（1 行に 1 つの JSON）でそこに行動を問い合わせる。

```bash
# 初回のみ：LeRobot の venv を作る（プロジェクトのルートの .venv-lerobot、git 管理外）
uv venv -p 3.12 .venv-lerobot
VIRTUAL_ENV=.venv-lerobot uv pip install 'lerobot[dataset,training]==0.6.1'

# ターミナル 1：推論サーバ（pixi shell の中から起動してもよい。スクリプトが pixi の PYTHONPATH を除外する）
.venv-lerobot/bin/python scripts/policy_server.py outputs/act_200/checkpoints/060000/pretrained_model
#   時間的アンサンブル：--temporal-ensemble 0.01 を付ける。チャンクごとの実行ステップ数を変える：--n-action-steps 25

# ターミナル 2：シミュレーション、同期モード、学習に使っていない seed
ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=100

# ターミナル 3：50 エピソードを評価する（1 エピソード 900 フレーム = シミュレーション時間 18 s、掴み直しの時間を見込む。以前の評価は 600 フレーム）
ros2 run so101_mujoco policy_eval --ros-args -p num_episodes:=50 -p output_dir:=data/eval/act_200_60k
```

**境界評価**：sim に `-p "box_rel_yaw_range:=[35.0,55.0]"` を付けると、相対把持角 |g| ∈ [35°, 45°] の範囲にだけキューブを置く（立方体は 90° ごとに対称なので、[35, 55] は [35, 45] ∪ [−45, −35] と同じ）。ここはエキスパートが掴み方を切り替える位置で、ACT の失敗はここに集中するため、再把持データの効果の比較に使う。

`policy_eval` は `/joint_states` を受け取るたびに同じタイムスタンプの `/box_pose` を待ち、`export_dataset` と同じ方法で観測（関節角 + `expert.box_features`）を構成し、行動を問い合わせて、同じタイムスタンプで `/joint_command` を配信する。出力：

- `episodes.jsonl`：エピソードごとのキューブの初期 / 最終位置、成否、目標までの距離、キューブが持ち上がったか（「持ち上げられなかった」と「持ち上げた後に落とした / ずれて置いた」を区別する）、把持回数 `grasp_attempts`（目標から 5 cm 以上離れた場所でグリッパが開から閉になった回数。1 より大きければ掴み直したことを表す）、平均推論時間。
- `summary.json`：成功率、掴み直したエピソード数（`regrasped`）とそのうち成功した数（`recovered`）。
- `rollouts.npz`：全フレーム。形式は `demos.npz` と同じで、そのまま再生できる：`pixi run python scripts/visualize_demos.py --data data/eval/act_200_60k/rollouts.npz 0-4`。
  失敗エピソードと比較用の成功エピソードは、`pixi run python scripts/render_eval_cases.py data/eval/act_200_60k` で把持部分の拡大付きの mp4 に描画できる（出力は `videos/` サブディレクトリ）。
  画像入力方策があるエピソードで見たカメラ画像（評価時には画像を保存しない）は、`scripts/render_policy_cameras.py` で描画し直せる。sim_node と同じ `CameraRenderer` を使い、各フレームの関節角とキューブの位置姿勢から overview / wrist_cam を描画して、左右に並べた mp4 と一覧用の PNG を出力する（出力は `cameras/` サブディレクトリ）。キューブが机の上にある間は、方策が当時受け取った入力とピクセル単位で一致する。rollouts にはキューブの yaw（mod 90°）しかないため、グリッパに挟まれて傾いたキューブはまっすぐに描かれる。
  ```bash
  pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7                        # 既定で 2 倍に拡大し、一覧には均等に 12 フレームを選ぶ
  pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7 --speed 0.5 --scale 3
  pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7 --frames 200 240 260 300   # 一覧に載せるフレームを指定する
  ```
  把持指標と、2 つの評価のエピソードごとの対応比較：`pixi run python scripts/grasp_metrics.py data/eval/act_200_40k data/eval/act_200_60k`。

同じ seed なら評価ごとのキューブ配置の列は同じなので、異なる checkpoint や実行方式の成功率を直接比較できる（`box_radius_range` を変えた場合、各エピソードの方位角と向きは変わらず、r だけが変わる）。

### 対話型デモ

ユーザが操作パネルでキューブを配置すると、方策が viewer の中でリアルタイムに把持と配置を行う。前提：上記のとおり `.venv-lerobot` を作成し、`pixi run build` を済ませ、`outputs/` に checkpoint があること（既定は画像入力の ACT `outputs/act_img_all_hold/checkpoints/100000/pretrained_model`。`docs/eval_act_images_all_hold.md` を参照。sim に必要なカメラは checkpoint の `config.json` に従って自動で有効になる）。プロジェクトのルートで実行する（先に `pixi shell` に入る必要はない）：

```bash
pixi run bash scripts/demo.sh                 # 等倍速。既定では 480 エピソードで学習した画像入力の ACT（100k）を使う
pixi run bash scripts/demo.sh --checkpoint outputs/act_291_envnorm_retry/checkpoints/060000/pretrained_model   # 状態入力版（キューブ座標を使う）
pixi run bash scripts/demo.sh --speed 0.5     # 半分の速さ。把持の様子を見やすい
pixi run bash scripts/demo.sh --checkpoint outputs/act_200/checkpoints/040000/pretrained_model
#   推論サーバのポートは既定で 5555。使用中なら --port N で変える
#   -- の後の引数は policy_server.py に渡される。例：-- --temporal-ensemble 0.01
```

典型的な使い方：起動後、viewer と操作パネルが現れるのを待つ → スライダー（または *Random*）でキューブを配置する → *Start* を押す → viewer で把持、配置、初期姿勢への復帰を見る → 位置を変えて再び *Start* を押す。

スクリプトは `policy_server.py`、viewer 付きの `sim_node`（`sync_mode`、`realtime_factor` = 速度）、`policy_demo` の操作パネルを順に起動する。パネルか viewer を閉じるか、ターミナルで Ctrl-C を押すと、3 つのプロセスがまとめて終了する。ログは `data/demo/logs/` にある。

操作パネル：

- **スライダー**（*Distance r* / *Azimuth* / *Yaw*）：キューブの基部からの距離 r、方位角、向き。値を変えるたびに `/place_box` でシーンをリセットし、アームは初期姿勢に戻り、キューブは指定の位置に置かれる。既定では学習範囲内（r ∈ [18, 25] cm、方位角 ∈ [5°, 40°]、viewer の床に描かれた白枠）でしか調整できない。*Allow outside the training range* にチェックを入れると、範囲が r ∈ [10, 40] cm、方位角 ∈ [−90°, 90°] に広がる。範囲外ではキューブが黄色になり、方策が失敗することがある。
- **Start / Stop**：方策を実行する。キューブが目標の上で 1 s 静止し、かつアームが初期姿勢に戻ると、その実行は終了する（目標の上で 5 s 静止した場合も終了し、最大 `max_steps` = 900 フレーム）。終了後は、キューブを置き直してから再び開始する。
- **Random / Place again**：学習範囲内にランダムに配置する / 現在のスライダーの値で置き直す（アームは初期姿勢に戻る）。
- **Speed**：0.25× – 2×。実行中でも変えられる（`sim_node` の `realtime_factor` パラメータを設定する）。
- 実行中に viewer でキューブをダブルクリックして選び、Ctrl を押しながらマウスの右ボタンでドラッグすると、キューブを押して閉ループ方策の反応を観察できる。

各実行の結果は `data/demo/<起動時刻>/episodes.jsonl` と `rollouts.npz` に追記される。形式は `policy_eval` と同じで、`render_eval_cases.py` / `visualize_demos.py` でオフラインに動画として描画できる。

### モデルだけを見る（ROS を起動しない）

```bash
pixi run python -m mujoco.viewer --mjcf=ros2_ws/src/so101_mujoco/models/robotstudio_so101/scene_box.xml
```

## `sim_node` のインタフェース

| 種類 | 名前 | メッセージ型 | 説明 |
|---|---|---|---|
| 配信 | `/joint_states` | `sensor_msgs/JointState` | 6 関節の位置 / 速度 / トルク |
| 配信 | `/box_pose` | `geometry_msgs/PoseStamped` | キューブの位置姿勢（シーンに `box` という名前の body がある場合）。frame は `world` |
| 配信 | `/clock` | `rosgraph_msgs/Clock` | シミュレーション時刻。リセット後も増え続け、巻き戻らない |
| 購読 | `/place_box` | `geometry_msgs/Pose2D` | `/reset` と同様にシーンをリセットするが、キューブはランダムに置かず (x, y)（m）、yaw = theta（rad）の位置に置く。`box_radius_range` × `box_azimuth_range` の範囲外ではキューブが黄色で表示される。`policy_demo` がキューブの配置に使う |
| 購読 | `/joint_command` | `sensor_msgs/JointState` | 関節の目標角度（ラジアン）。`name` を付ければ一部の関節だけを制御できる。`name` がない場合は 6 つの値をすべて順番どおりに与える。`name` と `position` がどちらも空なら「現在の目標を維持する」という意味になる（同期モードでの応答に使う）。範囲外の値は切り詰められる |
| サービス | `/reset` | `std_srvs/Trigger` | シーンを初期状態にリセットする（`randomize_box` が有効ならキューブをランダムに置く） |

関節の順序：`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`。各関節は位置制御のアクチュエータで、指令はそのまま目標角度となる。

### パラメータ

| パラメータ | 既定値 | 説明 |
|---|---|---|
| `model_path` | パッケージ内の `scene_pick_place.xml` | MJCF シーンファイルのパス |
| `keyframe` | `""` | 初期姿勢に使う keyframe の名前（`pickup` など）。空ならモデルの既定の姿勢を使う |
| `publish_rate` | `50.0` | `/joint_states`、`/box_pose` の配信周波数（Hz） |
| `realtime_factor` | `1.0` | 実時間に対するシミュレーション速度の倍率。`0.0` は可能な限り速く（浮動小数点数で書く必要があり、`0` と書くと型エラーになる）。実行中でも `ros2 param set /so101_mujoco_sim realtime_factor 0.5` で変更できる |
| `use_viewer` | `false` | MuJoCo の可視化ウィンドウを開くかどうか（mjpython が必要） |
| `publish_clock` | `true` | `/clock` を配信するかどうか |
| `randomize_box` | `true` | リセット時にキューブをランダムに置く（位置 + yaw） |
| `box_radius_range` | `[0.18, 0.25]` | キューブの基部からの水平距離の範囲（m） |
| `box_azimuth_range` | `[5.0, 40.0]` | キューブの方位角の範囲（度、+x 軸から測る） |
| `box_rel_yaw_range` | `[-45.0, 45.0]` | キューブの相対把持角 g の範囲（度）。既定値は制限なしを表し、乱数列も以前と同じになる。`[35.0, 55.0]` は境界評価に使う |
| `seed` | `0` | 乱数シード（評価時は固定して再現しやすくする） |
| `sync_mode` | `false` | 同期モード：状態を 1 フレーム配信するたびに、同じタイムスタンプの `/joint_command` を 1 つ待ってから次のステップに進む |
| `sync_timeout` | `1.0` | 同期モードで待つ最大時間（実時間の秒）。タイムアウトしたら通常どおりステップを進めるため、制御側がなくてもシミュレーションは動き続ける |
| `show_workspace` | `true` | viewer の床に、キューブのランダム配置範囲（`box_radius_range` × `box_azimuth_range`、つまり学習範囲）を描く |

**同期モード**：制御側（`collect_demos`、その後の方策ノード）は `/joint_states` の 1 つ 1 つに、同じタイムスタンプの指令で応答する。新しい指令がなければ空の指令で応答する。こうすれば、制御側がどれだけ遅くても、一時的に止まっても、指令は必ず次のステップの前に反映され、記録される（状態, 行動）は厳密に対応し、結果も再現できる。非同期モードでは、制御側が少しでも止まるとアームは古い目標にとどまったまま計画だけが進み、データに「行動は変わっているのに状態が止まっている」区間が生じる（実際に発生した）。

## 注意点

- **キューブの作業空間**：Menagerie の元の `scene_box.xml` ではキューブが x=0.5 m にあるが、SO-101 のエンドエフェクタの最大水平到達距離は約 0.48 m しかなく、届かない。下向き姿勢のエンドエフェクタは z≈0.06–0.09 m までしか下がらない（それより上では `wrist_flex` が可動限界に達する）ため、4 cm の立方体を使っている。r < 0.16 m ではグリッパのカメラ支持部が `shoulder` と自己干渉する。既定のランダム範囲はこれらの制約の内側にあり、オフライン検証で 600/600 成功している。
- **mjpython と python の違い**（macOS）：`launch_passive` を使うプログラム（viewer を有効にした `sim_node` など）は `mjpython` で起動しなければならない。一方、単独の viewer `python -m mujoco.viewer` は通常の `python` で起動する必要があり、`mjpython` を使うと `Caught an unknown exception!` というエラーになる。
- **Ctrl-C**：ターミナルで直接 Ctrl-C を押せば正常に終了する。スクリプトで `ros2 run` から起動したノードを `kill -INT <pid>` で止めようとしても**効かない**（`ros2 run` はシグナルを転送しない）。プロセスグループ全体に送ること：`kill -INT -<pgid>`。
- **トピックが互いに見えないとき**：`export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` を設定するか、ミドルウェアを `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` に切り替える。
- **依存関係の追加**：`pixi add ros-jazzy-<パッケージ名>` を使う（`rosdep` は使わない）。PyPI のパッケージは `pixi add --pypi <パッケージ名>` で追加し、Python 3.12 向けのバージョンが必要である。
- **再ビルド**：`--symlink-install` を使っているため、`.py` ファイルを変更しても再ビルドは不要。ファイル、エントリポイント、モデルファイルを追加したときは、もう一度 `pixi run build` を実行する。

## 謝辞

SO-101 のモデルは [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（`robotstudio_so101`、commit `0059d43`）のものであり、そのライセンスはモデルのディレクトリにある `LICENSE` を参照。
