"""Animate episodes of the exported demonstration data, to check what the policy will see.

Replays the recorded joint positions and box pose in MuJoCo (kinematics only, no physics)
next to plots of each joint's state (solid) and commanded action (dashed), with a cursor
at the current frame. Reads the .npz written by `ros2 run so101_mujoco export_dataset`.

    pixi run python scripts/visualize_demos.py --list            # what is in the data
    pixi run python scripts/visualize_demos.py 0 3 10-12          # mp4s in data/viz/
    pixi run python scripts/visualize_demos.py 5 --show           # play in a window

The box is drawn upright, with its yaw recovered modulo 90 degrees from the cos/sin 4yaw
features; for a cube that looks the same.
"""

import argparse
import os
import shutil
import subprocess

import cv2
import matplotlib
import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco', 'models',
                             'robotstudio_so101', 'scene_pick_place.xml')


def parse_episodes(specs):
    """'3', '0-4' -> sorted episode indices."""
    episodes = set()
    for spec in specs:
        if '-' in spec:
            first, last = spec.split('-')
            episodes.update(range(int(first), int(last) + 1))
        else:
            episodes.add(int(spec))
    return sorted(episodes)


def box_yaw(env_state):
    """Yaw (mod 90 deg) from the [x, y, z, cos 4yaw, sin 4yaw] box features."""
    return np.arctan2(env_state[..., 4], env_state[..., 3]) / 4


def list_episodes(data):
    episodes = data['episode_index']
    fps = float(data['fps'])
    print(f'{len(np.unique(episodes))} episodes, {len(episodes)} frames at {fps:g} fps\n')
    print(' episode  frames  seconds   box x    box y   yaw(deg, mod 90)  max box z')
    for ep in np.unique(episodes):
        env = data['observation_environment_state'][episodes == ep]
        print(f'{ep:8d} {len(env):7d} {len(env) / fps:8.2f} {env[0, 0]:8.3f} {env[0, 1]:8.3f}'
              f' {np.degrees(box_yaw(env[0])):14.1f} {env[:, 2].max():12.3f}')


class Replayer:
    """Renders the scene for recorded states (no physics is stepped)."""

    def __init__(self, model_path, joint_names, camera, width, height):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        self.data = mujoco.MjData(self.model)
        self.joint_adr = [self.model.joint(n).qposadr[0] for n in joint_names]
        box = self.model.body('box')
        self.box_adr = self.model.jnt_qposadr[box.jntadr[0]]
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.camera = camera

    def render(self, state, env_state):
        q = self.data.qpos
        q[self.joint_adr] = state
        yaw = box_yaw(env_state)
        q[self.box_adr:self.box_adr + 3] = env_state[:3]
        q[self.box_adr + 3:self.box_adr + 7] = np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
        # mj_forward, not mj_kinematics: camera poses are only computed in mj_camlight.
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()


# Figure layout in pixels: the MuJoCo frame on the left, the plots on the right.
FIG_W, FIG_H, DPI = 1440, 680, 100
IMG_X, IMG_Y, IMG_W, IMG_H = 20, 40, 800, 600   # from the top-left corner


def build_figure(t, state, action, env, names):
    """Figure with an empty image axes and the plots; returns (fig, image axes, plot axes)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(FIG_W / DPI, FIG_H / DPI), dpi=DPI)
    ax_img = fig.add_axes([IMG_X / FIG_W, 1 - (IMG_Y + IMG_H) / FIG_H, IMG_W / FIG_W, IMG_H / FIG_H])
    ax_img.axis('off')
    grid = fig.add_gridspec(len(names) + 1, 1, left=(IMG_X + IMG_W + 130) / FIG_W, right=0.985,
                            top=1 - IMG_Y / FIG_H, bottom=0.08, hspace=0.15)
    axes = []
    for i, name in enumerate(names):
        ax = fig.add_subplot(grid[i], sharex=axes[0] if axes else None)
        ax.plot(t, state[:, i], color='tab:blue', lw=1.2, label='state')
        ax.plot(t, action[:, i], color='tab:orange', lw=1.2, ls='--', label='action')
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=8)
        ax.tick_params(labelsize=7, labelbottom=False)
        axes.append(ax)
    ax = fig.add_subplot(grid[len(names)], sharex=axes[0])
    ax.plot(t, env[:, 2], color='tab:green', lw=1.2)
    ax.set_ylabel('box z', rotation=0, ha='right', va='center', fontsize=8)
    ax.set_xlabel('time (s)', fontsize=8)
    ax.tick_params(labelsize=7)
    axes.append(ax)
    axes[0].legend(loc='upper right', fontsize=7, ncols=2)
    axes[0].set_xlim(t[0], t[-1])
    return fig, ax_img, axes


def save_video(path, fig, axes, frames, render, label, t, fps):
    """Draw the plots once, then per frame paste the rendering and a cursor and pipe the
    result to ffmpeg; redrawing the whole figure every frame is ~10x slower."""
    fig.canvas.draw()
    background = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    height = background.shape[0]
    # Cursor geometry in buffer pixels (rows count from the top).
    spans = []
    for ax in axes:
        box = ax.get_window_extent()
        spans.append((int(height - box.y1), int(height - box.y0)))
    to_px = axes[0].transData.transform

    ffmpeg = subprocess.Popen(
        [shutil.which('ffmpeg') or 'ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
         '-pix_fmt', 'rgb24', '-s', f'{background.shape[1]}x{height}', '-r', f'{fps:g}',
         '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', path],
        stdin=subprocess.PIPE)
    for frame in frames:
        canvas = background.copy()
        image = render(frame)
        cv2.putText(image, label(frame), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        canvas[IMG_Y:IMG_Y + IMG_H, IMG_X:IMG_X + IMG_W] = image
        x = int(round(to_px((t[frame], 0))[0]))
        for top, bottom in spans:
            canvas[top:bottom, x - 1:x + 1] = 0
        ffmpeg.stdin.write(canvas.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit(f'ffmpeg failed writing {path}')


def show_window(fig, ax_img, axes, frames, render, label, t, fps):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    image = ax_img.imshow(render(frames[0]), animated=True)
    text = ax_img.text(12, 28, '', color='white', fontsize=11, animated=True)
    cursors = [ax.axvline(t[0], color='k', lw=0.8, animated=True) for ax in axes]

    def update(frame):
        image.set_data(render(frame))
        text.set_text(label(frame))
        for cursor in cursors:
            cursor.set_xdata([t[frame], t[frame]])
        return [image, text, *cursors]

    # Keep a reference, or the animation is garbage collected before it plays.
    fig._animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
    plt.show()


def animate(data, episode, replayer, args):
    import matplotlib.pyplot as plt

    rows = np.flatnonzero(data['episode_index'] == episode)
    if len(rows) == 0:
        raise SystemExit(f'episode {episode} is not in the data (see --list)')
    state = data['observation_state'][rows]
    action = data['action'][rows]
    env = data['observation_environment_state'][rows]
    fps = float(data['fps'])
    t = np.arange(len(rows)) / fps
    frames = list(range(0, len(rows), args.stride))
    playback_fps = fps * args.speed / args.stride

    def render(frame):
        return replayer.render(state[frame], env[frame])

    def label(frame):
        return f'episode {episode}  frame {frame}/{len(rows) - 1}  t={t[frame]:.2f}s'

    fig, ax_img, axes = build_figure(t, state, action, env, [str(n) for n in data['joint_names']])
    if args.show:
        show_window(fig, ax_img, axes, frames, render, label, t, playback_fps)
        return
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f'episode_{episode:04d}.mp4')
    save_video(path, fig, axes, frames, render, label, t, playback_fps)
    plt.close(fig)
    print(f'episode {episode}: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split('\n', 2)[2])
    parser.add_argument('episodes', nargs='*', help="episode indices or ranges, e.g. 0 3 10-12")
    parser.add_argument('--data', default=os.path.join(REPO, 'data', 'demos.npz'))
    parser.add_argument('--list', action='store_true', help='list the episodes and exit')
    parser.add_argument('--show', action='store_true', help='play in a window instead of saving')
    parser.add_argument('--output-dir', default=os.path.join(REPO, 'data', 'viz'))
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed factor')
    parser.add_argument('--stride', type=int, default=1, help='draw every n-th frame')
    parser.add_argument('--camera', default='overview', help='camera name in the scene')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()

    data = np.load(args.data)
    if args.list or not args.episodes:
        list_episodes(data)
        if not args.list:
            print('\nGive episode indices to animate, e.g. "0 3 10-12" (see --help).')
        return
    if not args.show:
        matplotlib.use('Agg')

    replayer = Replayer(args.model, [str(n) for n in data['joint_names']], args.camera,
                        IMG_W, IMG_H)
    for episode in parse_episodes(args.episodes):
        animate(data, episode, replayer, args)


if __name__ == '__main__':
    main()
