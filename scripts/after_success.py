"""What a policy does after it has finished the task, from closed-loop evaluations.

The demonstrations used to end 0.5 s after the arm got back to its start pose, so a policy
never saw "box on the target, arm at home": back home, the image-based policy went for the
box again (the state-based one did not). The demos now hold the start pose for 2 s at the
end. For every successful episode of a
policy_eval output directory this finds the first frame after success_frame at which the
arm is back at its start pose (all 5 arm joints within --home-tol of frame 0), and from
there to the end of the episode (policy_eval always runs max_steps frames) measures:

- moved: the largest arm joint deviation from the start pose (rad); counted as moving
  again when above --move-tol;
- box: how far the box moved (cm) from where it was when the arm got home;
- whether the box is still on the target at the end (expert.is_success).

    pixi run python scripts/after_success.py data/eval/act_img_291_60k data/eval/act_img_all_hold_100k
    pixi run python scripts/after_success.py data/eval/act_img_291_60k -v     # every episode
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.expert import is_success  # noqa: E402

MODEL = os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco', 'models', 'robotstudio_so101',
                     'scene_pick_place.xml')


def analyse(eval_dir, target, home_tol):
    data = np.load(os.path.join(eval_dir, 'rollouts.npz'))
    with open(os.path.join(eval_dir, 'episodes.jsonl')) as f:
        episodes = [json.loads(line) for line in f]
    rows = []
    for ep in episodes:
        if not ep['success']:
            continue
        mask = data['episode_index'] == ep['episode']
        arm = data['observation_state'][mask][:, :5]
        box = data['observation_environment_state'][mask][:, :3]
        dev = np.abs(arm - arm[0]).max(axis=1)
        after = np.flatnonzero(dev[ep['success_frame']:] < home_tol)
        if len(after) == 0:
            rows.append({'episode': ep['episode'], 'home_frame': None})
            continue
        home = ep['success_frame'] + after[0]
        rows.append({'episode': ep['episode'], 'home_frame': int(home),
                     'idle_frames': int(len(arm) - home),
                     'moved': float(dev[home:].max()),
                     'box_moved_cm': float(100 * np.linalg.norm(box[home:, :2] - box[home, :2],
                                                                axis=1).max()),
                     'on_target_at_end': is_success(box[-1], target)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('eval_dirs', nargs='+')
    parser.add_argument('--home-tol', type=float, default=0.05, help='rad')
    parser.add_argument('--move-tol', type=float, default=0.2, help='rad')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print(f'{"evaluation":40s} {"success":>7s} {"home":>5s} {"idle":>5s} {"moved again":>11s} '
          f'{"max move (rad)":>17s} {"box moved":>9s} {"off target":>10s}')
    target = mujoco.MjModel.from_xml_path(MODEL).site('target').pos.copy()
    for d in args.eval_dirs:
        rows = analyse(d, target, args.home_tol)
        home = [r for r in rows if r['home_frame'] is not None]
        moved = [r for r in home if r['moved'] > args.move_tol]
        box = [r for r in home if r['box_moved_cm'] > 1.0]
        off = [r for r in home if not r['on_target_at_end']]
        m = np.array([r['moved'] for r in home]) if home else np.zeros(1)
        idle = np.median([r['idle_frames'] for r in home]) if home else 0
        print(f'{os.path.basename(os.path.normpath(d)):40s} {len(rows):7d} {len(home):5d} '
              f'{idle:5.0f} {len(moved):11d} {np.median(m):8.2f} / {m.max():5.2f} '
              f'{len(box):9d} {len(off):10d}')
        if args.verbose:
            for r in rows:
                print('   ', r)


if __name__ == '__main__':
    main()
