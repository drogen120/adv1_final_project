"""Draw the figures of the report: docs/report.md (Chinese) or docs/report_ja.md (Japanese).

    pixi run python scripts/make_report_figures.py              # -> docs/figures/
    pixi run python scripts/make_report_figures.py --lang ja    # -> docs/figures_ja/

Scene images are re-rendered from a closed-loop evaluation episode of the final image policy
(data/eval/act_img_all_hold_100k, episode 0); diagrams and charts are drawn with matplotlib.
Result numbers in the charts are copied from the evaluation docs (docs/eval_*.md).
"""

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.cameras import CameraRenderer  # noqa: E402

MODEL = os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco', 'models', 'robotstudio_so101',
                     'scene_pick_place.xml')
EVAL_DIR = os.path.join(REPO, 'data', 'eval', 'act_img_all_hold_100k')
OUT = None  # set in main
T = None    # the strings of the chosen language, set in main

# Chinese and Japanese need different fonts: the same characters have different glyphs.
FONTS = {'zh': ['Hiragino Sans GB', 'Heiti SC', 'sans-serif'],
         'ja': ['Hiragino Kaku Gothic ProN', 'Hiragino Sans', 'sans-serif']}
OUT_DIRS = {'zh': 'figures', 'ja': 'figures_ja'}
plt.rcParams['axes.unicode_minus'] = False

TEXT = {
    'zh': {
        'scene': ['0.0 s  初始姿态', '2.4 s  移到方块上方', '3.6 s  下降并夹紧', '5.4 s  抬起并搬运',
                  '7.2 s  放到目标上', '12.0 s  回到初始姿态'],
        'cam_when': ['回合开始（0 s）', '方块上方（2.4 s）'],
        'cam_title': '{name}，{when}',
        'arch_1': '① 数据采集（ROS 2 / pixi 环境，M1 Mac）',
        'sim': 'sim_node\nMuJoCo 仿真\nSO-101 + 4 cm 方块',
        'collect': 'collect_demos\nIK 脚本专家\n（可注入抓取失败）',
        'bag': 'rosbag (mcap)\n每回合一个',
        'record': '录制',
        'arch_2': '② 训练（LeRobot 0.6.1，Colab Pro GPU）',
        'to_lerobot': 'npz_to_lerobot\n离线渲染相机图像',
        'dataset': 'LeRobot 数据集\n（parquet + mp4）',
        'train': 'ACT 策略训练\n60k–100k 步，batch 64',
        'arch_3': '③ 闭环控制与评估',
        'eval': 'policy_eval /\npolicy_demo（ROS 2）',
        'server': 'policy_server\n（LeRobot venv，加载 checkpoint）',
        'obs': '观测', 'actions': '50 步动作', 'tcp': '本地 TCP，每行一个 JSON',
        'sync': '同步模式\n↑ /joint_states\n   /box_pose\n↓ /joint_command',
        'img_overview': 'overview 图像\n3×240×320', 'img_wrist': 'wrist_cam 图像\n3×240×320',
        'joints': '关节角 6 维\n（状态版另加方块 5 维）',
        'resnet': 'ResNet18\n（ImageNet 预训练）\n每张 → 8×10×512',
        'linear': '线性层 → 512', 'latent': '隐变量 z（32 维）\n线性层 → 512',
        'encoder': 'Transformer\n编码器\n4 层，8 头，d=512\n\n输入 162 个 token\n（2×80 图像 + 状态 + z）',
        'decoder': 'Transformer\n解码器\n1 层\n\n50 个可学习的\n查询向量',
        'cross': '交叉\n注意力', 'head': '线性层\n↓\n动作块\n50×6',
        'chunk_note': '50 帧 = 1 s 的\n关节目标角\n（执行完整 50 步\n再重新推理）',
        'cvae': '仅训练时使用：CVAE 编码器',
        'cvae_in': '[CLS] + 关节角\n+ 示教动作序列 50×6',
        'cvae_enc': 'Transformer 编码器（4 层）\n→ μ, σ → 采样 z',
        'loss': '损失 = L1(预测动作, 示教动作) + 10 × KL(z ‖ N(0, I))\n推理时没有示教动作，z 取 0（先验均值）。',
        'stages': ['状态版  50 回合', '状态版  200 回合', '+ 抓取偏移 9 mm', '+ 方块特征归一化',
                   '+ 重抓示教（291 回合）', '图像版  291 回合', '图像版  480 回合（最终）'],
        'totals': ['状态版  291 回合', '图像版  291 回合', '图像版  480 回合（最终）'],
        'res_left': '各阶段：seed 100 普通摆放，50 回合',
        'res_right': '最终对比：三组测试合计 200 回合\n（普通 50 + 边界 50 + seed 101 普通 100）',
        'legend': ['状态版（关节角 + 方块坐标）', '图像版（关节角 + 两路相机，无方块坐标）'],
        'ok': '成功', 'fail': '没抓起来',
        'jaw_note': '> +0.25 cm：固定爪压到\n方块顶面，全部没抓起来',
        'jaw_x': '下降时夹爪中心沿开合方向的偏差（cm；正值 = 偏向活动爪一侧，固定爪更靠近方块）',
        'jaw_y': '回合数',
        'jaw_title': '无偏移专家训练的 act_50 + act_200，seed 100 共 100 回合（第 105 帧测量）',
    },
    'ja': {
        'fs': 0.86,  # Japanese labels are longer; shrink the diagram text
        'scene': ['0.0 s  初期姿勢', '2.4 s  キューブの上へ移動', '3.6 s  下降して把持',
                  '5.4 s  持ち上げて搬送', '7.2 s  目標エリアに配置', '12.0 s  初期姿勢に戻る'],
        'cam_when': ['エピソード開始時（0 s）', 'キューブの上方（2.4 s）'],
        'cam_title': '{name}：{when}',
        'arch_1': '① データ収集（ROS 2 / pixi 環境、M1 Mac）',
        'sim': 'sim_node\nMuJoCo シミュレーション\nSO-101 + 4 cm キューブ',
        'collect': 'collect_demos\nIK スクリプトエキスパート\n（把持失敗を注入可能）',
        'bag': 'rosbag (mcap)\nエピソードごとに 1 つ',
        'record': '記録',
        'arch_2': '② 学習（LeRobot 0.6.1、Colab Pro GPU）',
        'to_lerobot': 'npz_to_lerobot\nカメラ画像を\nオフラインで描画',
        'dataset': 'LeRobot データセット\n（parquet + mp4）',
        'train': 'ACT 方策の学習\n60k～100k ステップ\nbatch 64',
        'arch_3': '③ 閉ループ制御と評価',
        'eval': 'policy_eval /\npolicy_demo（ROS 2）',
        'server': 'policy_server\n（LeRobot venv で\ncheckpoint を読み込む）',
        'obs': '観測', 'actions': '行動 50 ステップ', 'tcp': 'ローカル TCP、1 行 1 JSON',
        'sync': '同期モード\n↑ /joint_states\n   /box_pose\n↓ /joint_command',
        'img_overview': 'overview 画像\n3×240×320', 'img_wrist': 'wrist_cam 画像\n3×240×320',
        'joints': '関節角 6 次元\n（状態入力版はキューブ\n5 次元を追加）',
        'resnet': 'ResNet18\n（ImageNet 事前学習）\n各画像 → 8×10×512',
        'linear': '線形層 → 512', 'latent': '潜在変数 z（32 次元）\n線形層 → 512',
        'encoder': 'Transformer\nエンコーダ\n4 層、8 ヘッド、d=512\n\n入力 162 トークン\n（画像 2×80 + 状態 + z）',
        'decoder': 'Transformer\nデコーダ\n1 層\n\n学習可能な\nクエリ 50 個',
        'cross': 'クロス\nアテン\nション', 'head': '線形層\n↓\nアクション\nチャンク\n50×6',
        'chunk_note': '50 フレーム = 1 s 分の\n関節目標角\n（50 ステップを\n実行してから\n再び推論）',
        'cvae': '学習時のみ使用：CVAE エンコーダ',
        'cvae_in': '[CLS] + 関節角\n+ デモの行動列 50×6',
        'cvae_enc': 'Transformer エンコーダ（4 層）\n→ μ, σ → z をサンプリング',
        'loss': '損失 = L1(予測行動, デモの行動) + 10 × KL(z ‖ N(0, I))\n推論時はデモの行動がないため、z = 0（事前分布の平均）とする。',
        'stages': ['状態入力  50 エピソード', '状態入力  200 エピソード', '+ 把持オフセット 9 mm',
                   '+ キューブ特徴量の正規化', '+ 再把持デモ（291 エピソード）',
                   '画像入力  291 エピソード', '画像入力  480 エピソード（最終）'],
        'totals': ['状態入力  291 エピソード', '画像入力  291 エピソード',
                   '画像入力  480 エピソード（最終）'],
        'res_left': '各段階：seed 100 通常配置、50 エピソード',
        'res_right': '最終比較：3 つのテストの合計 200 エピソード\n（通常 50 + 境界 50 + seed 101 通常 100）',
        'legend': ['状態入力（関節角 + キューブ座標）', '画像入力（関節角 + カメラ 2 台、キューブ座標なし）'],
        'ok': '成功', 'fail': '持ち上げられず',
        'jaw_note': '> +0.25 cm：固定ジョーが\nキューブ上面に乗り、\nすべて失敗',
        'jaw_x': '下降中のグリッパ中心の開閉方向のずれ（cm、正の値は可動ジョー側で、固定ジョーがキューブに近づく）',
        'jaw_y': 'エピソード数',
        'jaw_title': 'オフセットなしのエキスパートで学習した act_50 + act_200、seed 100 の計 100 エピソード（第 105 フレームで計測）',
    },
}

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
# Light tints of the same hues, for box fills in the diagrams.
TINT = {BLUE: '#e3eefb', ORANGE: '#fde8df', AQUA: '#dcf4ea', VIOLET: '#e8e5fa', MUTED: '#f1f0ed'}


def box_quat(env):
    yaw = np.arctan2(env[4], env[3]) / 4
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def load_episode(episode):
    z = np.load(os.path.join(EVAL_DIR, 'rollouts.npz'))
    rows = np.flatnonzero(z['episode_index'] == episode)
    names = [str(n) for n in z['joint_names']]
    return (z['observation_state'][rows].astype(np.float64),
            z['observation_environment_state'][rows].astype(np.float64), names)


def scene_sequence():
    """Six moments of one successful episode, from a free camera."""
    state, env, names = load_episode(0)
    model = mujoco.MjModel.from_xml_path(MODEL)
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    rend = CameraRenderer(model, ['overview'], joint_names=names)
    big = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.16, 0.0, 0.06)
    cam.distance, cam.azimuth, cam.elevation = 0.72, 120.0, -30.0
    frames = list(zip((0, 120, 180, 270, 360, 600), T['scene']))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (i, label) in zip(axes.flat, frames):
        rend.render_state(state[i], env[i, :3], box_quat(env[i]))  # sets rend.data
        mujoco.mj_forward(model, rend.data)
        big.update_scene(rend.data, cam)
        ax.imshow(big.render())
        ax.set_title(label, fontsize=12, color=INK)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'scene_sequence.png'), dpi=150)
    plt.close(fig)
    big.close()
    rend.close()


def camera_inputs():
    """What the image policy sees: both cameras at 240x320, at the start and above the box."""
    state, env, names = load_episode(0)
    model = mujoco.MjModel.from_xml_path(MODEL)
    rend = CameraRenderer(model, ['overview', 'wrist_cam'], joint_names=names)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    for row, (i, when) in enumerate(zip((0, 120), T['cam_when'])):
        imgs = rend.render_state(state[i], env[i, :3], box_quat(env[i]))
        for col, name in enumerate(['overview', 'wrist_cam']):
            ax = axes[row, col]
            ax.imshow(imgs[name])
            ax.set_title(T['cam_title'].format(name=name, when=when), fontsize=11, color=INK)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'camera_inputs.png'), dpi=150)
    plt.close(fig)
    rend.close()


def node(ax, x, y, w, h, text, colour, size=10.5, bold=False):
    size *= T.get('fs', 1.0)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=TINT[colour], ec=colour, lw=1.4))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=INK,
            fontweight='bold' if bold else 'normal', linespacing=1.45)


def arrow(ax, p, q, text='', colour=INK2, offset=(0, 0.13), rad=0.0, size=9, style='-|>'):
    size *= T.get('fs', 1.0)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3,
                                 color=colour, connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text,
                ha='center', va='center', fontsize=size, color=INK2, linespacing=1.4)


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis('off')

    # Top row: data collection (ROS 2 environment).
    ax.text(0.15, 6.4, T['arch_1'], fontsize=11.5, color=BLUE,
            fontweight='bold')
    node(ax, 1.4, 5.15, 2.2, 1.3, T['sim'], BLUE)
    node(ax, 5.1, 5.15, 2.2, 1.3, T['collect'], BLUE)
    node(ax, 8.1, 5.15, 2.0, 1.3, T['bag'], BLUE)
    node(ax, 10.8, 5.15, 1.9, 1.3, 'export_dataset\n→ npz', BLUE)
    arrow(ax, (2.5, 5.4), (4.0, 5.4), '/joint_states\n/box_pose  /clock', offset=(0, 0.45),
          size=8.5)
    arrow(ax, (4.0, 4.9), (2.5, 4.9), '/joint_command\n/reset', offset=(0, -0.45), size=8.5)
    arrow(ax, (6.2, 5.15), (7.1, 5.15), T['record'])
    arrow(ax, (9.1, 5.15), (9.85, 5.15))

    # Middle row: training (LeRobot / Colab).
    ax.text(3.0, 3.65, T['arch_2'], fontsize=11.5, color=ORANGE,
            fontweight='bold')
    node(ax, 10.8, 2.75, 1.9, 1.2, T['to_lerobot'], ORANGE)
    node(ax, 7.5, 2.75, 2.6, 1.2, T['dataset'], ORANGE)
    node(ax, 4.3, 2.75, 2.6, 1.2, T['train'], ORANGE)
    arrow(ax, (10.8, 4.5), (10.8, 3.35))
    arrow(ax, (9.85, 2.75), (8.8, 2.75))
    arrow(ax, (6.2, 2.75), (5.6, 2.75))

    # Bottom row: closed loop.
    ax.text(3.0, 1.45, T['arch_3'], fontsize=11.5, color=AQUA, fontweight='bold')
    node(ax, 1.4, 0.6, 2.2, 0.95, T['eval'], AQUA, size=9.5)
    node(ax, 5.6, 0.6, 2.9, 0.95, T['server'], AQUA,
         size=9.5)
    arrow(ax, (4.3, 2.15), (4.95, 1.08), 'checkpoint', offset=(1.05, 0.2))
    arrow(ax, (2.5, 0.78), (4.15, 0.78), T['obs'], offset=(0, 0.17))
    arrow(ax, (4.15, 0.42), (2.5, 0.42), T['actions'], offset=(0, -0.17))
    ax.text(7.2, 0.6, T['tcp'], ha='left', va='center', fontsize=8.5,
            color=MUTED)
    # Closed loop between the policy node and the sim, straight up the left side.
    arrow(ax, (1.1, 1.08), (1.1, 4.5), colour=AQUA, style='<|-|>')
    ax.text(1.25, 2.8, T['sync'],
            fontsize=8.5, color=INK2, va='center', linespacing=1.45)
    fig.savefig(os.path.join(OUT, 'architecture.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def act_structure():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.2)
    ax.axis('off')

    # Inputs.
    node(ax, 1.25, 6.1, 2.1, 0.8, T['img_overview'], ORANGE, size=9.5)
    node(ax, 1.25, 4.95, 2.1, 0.8, T['img_wrist'], ORANGE, size=9.5)
    node(ax, 1.25, 3.8, 2.1, 0.8, T['joints'], BLUE, size=9)
    node(ax, 3.9, 5.5, 2.0, 1.3, T['resnet'], ORANGE,
         size=9.5)
    node(ax, 3.9, 3.8, 2.0, 0.8, T['linear'], BLUE, size=9.5)
    node(ax, 3.9, 2.75, 2.0, 0.8, T['latent'], VIOLET, size=9)
    arrow(ax, (2.3, 6.1), (2.9, 5.75))
    arrow(ax, (2.3, 4.95), (2.9, 5.25))
    arrow(ax, (2.3, 3.8), (2.9, 3.8))

    # Transformer.
    node(ax, 6.85, 4.4, 2.3, 2.3, T['encoder'],
         MUTED, size=9.5)
    for y in (5.5, 3.8, 2.75):
        arrow(ax, (4.9, y), (5.7, 4.4 + (y - 4.0) * 0.35))
    node(ax, 9.65, 4.4, 2.1, 2.3, T['decoder'], MUTED,
         size=9.5)
    arrow(ax, (8.0, 4.4), (8.6, 4.4), T['cross'], offset=(0, 0.45), size=8.5)
    node(ax, 11.6, 4.4, 1.5, 1.6, T['head'], AQUA, size=10, bold=False)
    arrow(ax, (10.7, 4.4), (10.85, 4.4))
    ax.text(11.6, 3.3, T['chunk_note'], ha='center',
            va='top', fontsize=8.5, color=INK2, linespacing=1.4)

    # CVAE encoder (training only).
    ax.add_patch(FancyBboxPatch((0.15, 0.2), 6.0, 1.75, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc='none', ec=VIOLET, lw=1.2, ls=(0, (4, 3))))
    ax.text(0.35, 1.72, T['cvae'], fontsize=10, color=VIOLET,
            fontweight='bold')
    node(ax, 1.5, 0.85, 2.3, 0.85, T['cvae_in'], VIOLET, size=9)
    node(ax, 4.4, 0.85, 2.6, 0.85, T['cvae_enc'], VIOLET,
         size=9)
    arrow(ax, (2.65, 0.85), (3.1, 0.85))
    arrow(ax, (4.4, 1.28), (3.9, 2.35), colour=VIOLET)
    ax.text(6.6, 1.25, T['loss'], fontsize=9.5, color=INK2,
            linespacing=1.6, va='center')
    fig.savefig(os.path.join(OUT, 'act_structure.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def style_axes(ax):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def results_chart():
    """Left: seed 100, 50 episodes, every stage. Right: all three test sets (200 episodes)."""
    # (label, successes, colour): seed 100, 50 episodes / all three test sets, 200 episodes.
    stages = list(zip(T['stages'], (30, 48, 49, 49, 50, 49, 47), [BLUE] * 5 + [ORANGE] * 2))
    totals = list(zip(T['totals'], (200, 194, 196), (BLUE, ORANGE, ORANGE)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={'width_ratios': [1.35, 1]})
    for ax, rows, n, title in (
            (a1, stages, 50, T['res_left']),
            (a2, totals, 200, T['res_right'])):
        y = np.arange(len(rows))[::-1]
        rates = [100 * k / n for _, k, _ in rows]
        ax.barh(y, rates, height=0.56, color=[c for *_, c in rows], edgecolor='white', lw=2)
        for yi, (label, k, _), r in zip(y, rows, rates):
            ax.text(r + 1.0, yi, f'{k}/{n}', va='center', fontsize=9.5, color=INK)
        ax.set_yticks(y, [label for label, *_ in rows], fontsize=10, color=INK)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100%'])
        ax.set_title(title, fontsize=11, color=INK, loc='left')
        style_axes(ax)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                        plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
               labels=T['legend'],
               loc='lower center', ncol=2, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, 'results.png'), dpi=150)
    plt.close(fig)


def jaw_offset_chart():
    """Success vs the tool's offset along the closing direction (no grasp offset, 100 episodes)."""
    bins = ['< −1.0', '−1.0 ~ −0.5', '−0.5 ~ −0.25', '−0.25 ~ 0', '0 ~ +0.25',
            '+0.25 ~ +0.5', '+0.5 ~ +1.0', '> +1.0']
    ok = [10, 21, 9, 18, 20, 0, 0, 0]
    total = [10, 21, 9, 18, 20, 10, 8, 4]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(bins))
    ax.bar(x, ok, width=0.62, color=BLUE, edgecolor='white', lw=2, label=T['ok'])
    ax.bar(x, np.subtract(total, ok), bottom=ok, width=0.62, color='#c9c8c3', edgecolor='white',
           lw=2, label=T['fail'])
    for xi, k, t in zip(x, ok, total):
        ax.text(xi, t + 0.6, f'{k}/{t}', ha='center', fontsize=9.5, color=INK)
    ax.axvline(4.5, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(4.6, 18.5, T['jaw_note'], fontsize=9.5, color=INK2, va='top')
    ax.set_xticks(x, bins, fontsize=9.5, color=INK)
    ax.set_xlabel(T['jaw_x'],
                  fontsize=9.5, color=INK2)
    ax.set_ylabel(T['jaw_y'], fontsize=9.5, color=INK2)
    ax.set_ylim(0, 24)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc='upper right')
    ax.set_title(T['jaw_title'],
                 fontsize=10.5, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'jaw_offset.png'), dpi=150)
    plt.close(fig)


def main():
    global OUT, T
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--lang', choices=sorted(TEXT), default='zh')
    args = parser.parse_args()
    T = TEXT[args.lang]
    plt.rcParams['font.family'] = FONTS[args.lang]
    OUT = os.path.join(REPO, 'docs', OUT_DIRS[args.lang])
    os.makedirs(OUT, exist_ok=True)
    architecture()
    act_structure()
    results_chart()
    jaw_offset_chart()
    scene_sequence()
    camera_inputs()
    print('\n'.join(sorted(os.path.join(OUT, f) for f in os.listdir(OUT))))


if __name__ == '__main__':
    main()
