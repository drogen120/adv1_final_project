"""Render selected closed-loop evaluation episodes (all failures plus a few successes) as mp4s.

Reads the output directory of `ros2 run so101_mujoco policy_eval` (rollouts.npz and
episodes.jsonl) and replays the episodes like visualize_demos.py: the overview camera with a
close-up of the grasp inset in a corner, next to plots of the joints and the box height. The
banner says whether the episode succeeded and how far the box ended from the target.

    pixi run python scripts/render_eval_cases.py data/eval/act_200_60k            # mp4s in <eval>/videos/
    pixi run python scripts/render_eval_cases.py data/eval/act_200_60k --num-success 2
    pixi run python scripts/render_eval_cases.py data/eval/act_200_60k --episodes 36 49

By default the successes are the ones whose box started closest to a failed episode's box,
so each failure can be compared with a success from the same part of the workspace, plus the
success with the median placement error as a typical case.
"""

import argparse
import json
import os

import cv2
import matplotlib
import mujoco
import numpy as np

from visualize_demos import (DEFAULT_MODEL, IMG_H, IMG_W, Replayer, build_figure,
                             save_video)

# Close-up inset in the top-right corner of the rendering.
INSET_W, INSET_H, INSET_MARGIN = 320, 240, 10


def load_results(eval_dir):
    with open(os.path.join(eval_dir, 'episodes.jsonl')) as f:
        results = {r['episode']: r for r in map(json.loads, f)}
    return results, np.load(os.path.join(eval_dir, 'rollouts.npz'))


def pick_episodes(results, num_success):
    """All failures, then the successes closest to them, then the median success."""
    failures = [e for e, r in results.items() if not r['success']]
    successes = [e for e, r in results.items() if r['success']]
    start = {e: np.array(r['box_init'][:2]) for e, r in results.items()}
    picked = []
    # Round-robin over the failures, nearest unused success each time.
    while failures and len(picked) < num_success - 1 and len(picked) < len(successes):
        for f in failures:
            free = [e for e in successes if e not in picked]
            if not free or len(picked) >= num_success - 1:
                break
            picked.append(min(free, key=lambda e: np.linalg.norm(start[e] - start[f])))
    free = sorted((e for e in successes if e not in picked), key=lambda e: results[e]['target_dist'])
    while free and len(picked) < num_success:
        picked.append(free.pop(len(free) // 2))
    return failures + picked


class InsetReplayer(Replayer):
    """Replayer that adds a close-up of the box's starting position as an inset."""

    def __init__(self, model_path, joint_names, camera, width, height):
        super().__init__(model_path, joint_names, camera, width, height)
        self.inset = mujoco.Renderer(self.model, INSET_H, INSET_W)
        self.closeup = mujoco.MjvCamera()
        self.closeup.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.closeup.distance = 0.26
        self.closeup.elevation = -15.0

    def aim(self, box_pos):
        # Look at the box from outside its radius, towards the base: the jaws close across
        # the arm, so their sides (and a jaw landing on the box) are seen side-on.
        self.closeup.lookat[:] = box_pos[0], box_pos[1], 0.045
        self.closeup.azimuth = np.degrees(np.arctan2(box_pos[1], box_pos[0])) + 180.0

    def render(self, state, env_state):
        image = super().render(state, env_state).copy()   # also runs mj_forward
        self.inset.update_scene(self.data, self.closeup)
        inset = self.inset.render()
        x0 = image.shape[1] - INSET_W - INSET_MARGIN
        image[INSET_MARGIN:INSET_MARGIN + INSET_H, x0:x0 + INSET_W] = inset
        cv2.rectangle(image, (x0 - 1, INSET_MARGIN - 1),
                      (x0 + INSET_W, INSET_MARGIN + INSET_H), (255, 255, 255), 1)
        return image


def render_episode(data, result, replayer, output_dir, stride, speed):
    import matplotlib.pyplot as plt

    episode = result['episode']
    rows = np.flatnonzero(data['episode_index'] == episode)
    state = data['observation_state'][rows]
    action = data['action'][rows]
    env = data['observation_environment_state'][rows]
    fps = float(data['fps'])
    t = np.arange(len(rows)) / fps
    frames = list(range(0, len(rows), stride))
    replayer.aim(env[0, :3])

    if result['success']:
        outcome = 'SUCCESS'
    else:
        outcome = 'FAIL (dropped/misplaced)' if result['lifted'] else 'FAIL (not lifted)'
    colour = (80, 220, 80) if result['success'] else (255, 80, 80)

    def render(frame):
        image = replayer.render(state[frame], env[frame])
        cv2.putText(image, f'{outcome}, {100 * result["target_dist"]:.1f} cm from target',
                    (12, IMG_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        return image

    def label(frame):
        return f'episode {episode}  frame {frame}/{len(rows) - 1}  t={t[frame]:.2f}s'

    fig, _, axes = build_figure(t, state, action, env, [str(n) for n in data['joint_names']])
    tag = 'success' if result['success'] else 'fail'
    path = os.path.join(output_dir, f'episode_{episode:04d}_{tag}.mp4')
    save_video(path, fig, axes, frames, render, label, t, fps * speed / stride)
    plt.close(fig)
    print(f'episode {episode} ({outcome}): {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split('\n', 2)[2])
    parser.add_argument('eval_dir', help='output_dir of policy_eval')
    parser.add_argument('--episodes', type=int, nargs='+',
                        help='render these episodes instead of the automatic selection')
    parser.add_argument('--num-success', type=int, default=3,
                        help='successes to add to the failures (default 3)')
    parser.add_argument('--output-dir', help='default: <eval_dir>/videos')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed factor')
    parser.add_argument('--stride', type=int, default=1, help='draw every n-th frame')
    parser.add_argument('--camera', default='overview', help='camera name in the scene')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()

    results, data = load_results(args.eval_dir)
    episodes = args.episodes or pick_episodes(results, args.num_success)
    missing = [e for e in episodes if e not in results]
    if missing:
        raise SystemExit(f'episodes {missing} are not in {args.eval_dir}')
    output_dir = args.output_dir or os.path.join(args.eval_dir, 'videos')
    os.makedirs(output_dir, exist_ok=True)

    matplotlib.use('Agg')
    replayer = InsetReplayer(args.model, [str(n) for n in data['joint_names']], args.camera,
                             IMG_W, IMG_H)
    for episode in episodes:
        render_episode(data, results[episode], replayer, output_dir, args.stride, args.speed)


if __name__ == '__main__':
    main()
