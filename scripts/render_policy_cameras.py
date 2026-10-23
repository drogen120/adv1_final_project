"""Show what an image-based policy saw in a closed-loop evaluation episode.

Re-renders the policy's cameras (overview, wrist_cam, 240x320) for every frame of an episode
from rollouts.npz with cameras.CameraRenderer, the renderer sim_node used, and writes them
side by side to an mp4, plus a contact sheet PNG of evenly spaced frames.

    pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7
    pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7 --scale 3 --speed 0.5
    pixi run python scripts/render_policy_cameras.py data/eval/act_img_291_60k 7 --frames 0 150 300

The rollouts only store the joints (float32) and the box features [x, y, z, cos 4yaw, sin 4yaw],
so the box is drawn upright with its yaw modulo 90 degrees. That is pixel-exact while the box
sits on the table; a box tilted in the gripper is drawn untilted.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.cameras import CAMERAS, HEIGHT, WIDTH, CameraRenderer  # noqa: E402

import mujoco  # noqa: E402

DEFAULT_MODEL = os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco', 'models',
                             'robotstudio_so101', 'scene_pick_place.xml')
BANNER = 34   # px above the images (at --scale 1 it is still this tall)


def box_quat(env):
    yaw = np.arctan2(env[4], env[3]) / 4
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def compose(images, cameras, scale, text, colour):
    """Cameras side by side, upscaled (nearest, to keep the policy's pixels), with a banner."""
    tiles = []
    for name in cameras:
        img = cv2.resize(images[name], (WIDTH * scale, HEIGHT * scale),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        tiles.append(img)
    gap = np.full((HEIGHT * scale, 4, 3), 255, np.uint8)
    row = tiles[0]
    for tile in tiles[1:]:
        row = np.hstack([row, gap, tile])
    banner = np.zeros((BANNER, row.shape[1], 3), np.uint8)
    font = 0.6 if scale > 1 else 0.42
    cv2.putText(banner, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, font, colour, 1, cv2.LINE_AA)
    frame = np.vstack([banner, row])
    # libx264 + yuv420p needs even sizes.
    return frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split('\n', 2)[2])
    parser.add_argument('eval_dir', help='output_dir of policy_eval')
    parser.add_argument('episode', type=int)
    parser.add_argument('--cameras', nargs='+', default=list(CAMERAS))
    parser.add_argument('--scale', type=int, default=2, help='upscale factor (default 2)')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed factor')
    parser.add_argument('--stride', type=int, default=1, help='use every n-th frame')
    parser.add_argument('--frames', type=int, nargs='+',
                        help='frames for the contact sheet (default: 12 evenly spaced)')
    parser.add_argument('--output-dir', help='default: <eval_dir>/cameras')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()

    with open(os.path.join(args.eval_dir, 'episodes.jsonl')) as f:
        results = {r['episode']: r for r in map(json.loads, f)}
    if args.episode not in results:
        raise SystemExit(f'episode {args.episode} is not in {args.eval_dir}')
    result = results[args.episode]
    data = np.load(os.path.join(args.eval_dir, 'rollouts.npz'))
    rows = np.flatnonzero(data['episode_index'] == args.episode)
    state = data['observation_state'][rows].astype(np.float64)
    env = data['observation_environment_state'][rows].astype(np.float64)
    action = data['action'][rows]
    fps = float(data['fps'])
    joint_names = [str(n) for n in data['joint_names']]

    model = mujoco.MjModel.from_xml_path(args.model)
    renderer = CameraRenderer(model, args.cameras, joint_names=joint_names)

    outcome = 'SUCCESS' if result['success'] else ('FAIL (dropped/misplaced)' if result['lifted']
                                                  else 'FAIL (not lifted)')
    colour = (80, 220, 80) if result['success'] else (255, 90, 90)

    def frame_image(i):
        images = renderer.render_state(state[i], env[i, :3], box_quat(env[i]))
        text = (f'ep {args.episode} {outcome}  frame {i}/{len(rows) - 1}  t={i / fps:5.2f}s  '
                f'box z={100 * env[i, 2]:.1f}cm  gripper={state[i, -1]:+.2f} '
                f'(cmd {action[i, -1]:+.2f})')
        return compose(images, args.cameras, args.scale, text, colour)

    out_dir = args.output_dir or os.path.join(args.eval_dir, 'cameras')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'success' if result['success'] else 'fail'
    stem = os.path.join(out_dir, f'episode_{args.episode:04d}_{tag}')

    first = frame_image(0)
    h, w = first.shape[:2]
    ffmpeg = subprocess.Popen(
        [shutil.which('ffmpeg') or 'ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
         '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', f'{fps * args.speed / args.stride:g}',
         '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', stem + '.mp4'],
        stdin=subprocess.PIPE)
    for i in range(0, len(rows), args.stride):
        ffmpeg.stdin.write(frame_image(i).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit('ffmpeg failed')

    # Contact sheet, at scale 1 so it stays a manageable size.
    picks = args.frames or np.linspace(0, len(rows) - 1, 12).round().astype(int).tolist()
    scale, args.scale = args.scale, 1
    sheet = [frame_image(min(i, len(rows) - 1)) for i in picks]
    args.scale = scale
    sheet = np.vstack([np.hstack(sheet[k:k + 2]) if k + 1 < len(sheet)
                       else np.hstack([sheet[k], np.zeros_like(sheet[k])])
                       for k in range(0, len(sheet), 2)])
    cv2.imwrite(stem + '_sheet.png', cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    renderer.close()
    print(f'{stem}.mp4\n{stem}_sheet.png')


if __name__ == '__main__':
    main()
