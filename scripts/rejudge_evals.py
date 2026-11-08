"""Re-judge saved closed-loop evaluations with the current so101_mujoco.rollouts.

For output directories of policy_eval (ROS) or scripts/sim_eval.py written before a field
was added to episode_result (e.g. strict_success), recomputes every episode from
rollouts.npz and the box positions in episodes.jsonl, and prints successes and strict
successes per directory. With --write, merges the new fields into episodes.jsonl and
summary.json (existing fields such as policy_ms are kept).

    .venv-lerobot/bin/python scripts/rejudge_evals.py data/eval_py/act_img_all_hold_100k_n30*
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.kinematics import TopDownIK  # noqa: E402
from so101_mujoco.pick_place_sim import DEFAULT_MODEL  # noqa: E402
from so101_mujoco.rollouts import episode_result, summarize  # noqa: E402


def rejudge(eval_dir, kinematics, target_pos):
    with open(os.path.join(eval_dir, 'episodes.jsonl')) as f:
        episodes = [json.loads(line) for line in f]
    data = np.load(os.path.join(eval_dir, 'rollouts.npz'))
    updated = []
    for old in episodes:
        i = data['episode_index'] == old['episode']
        frames = [(s, e, a, 0) for s, e, a in zip(data['observation_state'][i],
                                                  data['observation_environment_state'][i],
                                                  data['action'][i])]
        new = episode_result(frames, old['box_init'], old['box_final'], target_pos,
                             kinematics, 0.0)
        new.pop('policy_ms')
        updated.append({**old, **new})
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('eval_dirs', nargs='+')
    parser.add_argument('--write', action='store_true',
                        help='update episodes.jsonl and summary.json in place')
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(DEFAULT_MODEL)
    kinematics = TopDownIK(model)
    target_pos = model.site('target').pos.copy()
    for eval_dir in args.eval_dirs:
        episodes = rejudge(eval_dir, kinematics, target_pos)
        summary = summarize(episodes)
        print(f'{eval_dir}: {summary["successes"]}/{summary["episodes"]} successes, '
              f'{summary["strict_successes"]} strict')
        if args.write:
            with open(os.path.join(eval_dir, 'episodes.jsonl'), 'w') as f:
                f.writelines(json.dumps(e) + '\n' for e in episodes)
            path = os.path.join(eval_dir, 'summary.json')
            old = json.load(open(path)) if os.path.exists(path) else {}
            with open(path, 'w') as f:
                json.dump({**old, **summary}, f, indent=2)


if __name__ == '__main__':
    main()
