"""Compare two closed-loop evaluations of the same checkpoint and seed episode by episode.

Meant for checking scripts/sim_eval.py against policy_eval (ROS): for each episode, whether
the box started at the same place, whether both succeeded, the success frames, and how long
the two trajectories stay together (first frame where a joint position differs by more than
--tol rad). Only needs numpy.

    python scripts/compare_evals.py data/eval/act_img_all_hold_100k data/eval_py/act_img_all_hold_100k
"""

import argparse
import json
import os

import numpy as np


def load(eval_dir):
    with open(os.path.join(eval_dir, 'episodes.jsonl')) as f:
        episodes = [json.loads(line) for line in f]
    rollouts = np.load(os.path.join(eval_dir, 'rollouts.npz'))
    states = [rollouts['observation_state'][rollouts['episode_index'] == i]
              for i in range(len(episodes))]
    return episodes, states


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('reference')
    parser.add_argument('other')
    parser.add_argument('--tol', type=float, default=1e-4,
                        help='joint position difference (rad) counted as diverged')
    parser.add_argument('--quiet', action='store_true', help='only the summary')
    args = parser.parse_args()

    ref_eps, ref_states = load(args.reference)
    oth_eps, oth_states = load(args.other)
    n = min(len(ref_eps), len(oth_eps))
    print(f'{args.reference}: {sum(e["success"] for e in ref_eps[:n])}/{n} successes, '
          f'{args.other}: {sum(e["success"] for e in oth_eps[:n])}/{n} '
          f'(first {n} episodes)')

    box_diffs, diverge, lengths, agree, frame_diffs = [], [], [], 0, []
    if not args.quiet:
        print(f'{"ep":>3} {"box Δ mm":>8} {"success":>9} {"success_frame":>15} '
              f'{"same until":>10} {"max Δ rad":>9}')
    for i in range(n):
        a, b = ref_eps[i], oth_eps[i]
        box_diff = 1e3 * np.abs(np.subtract(a['box_init'][:2], b['box_init'][:2])).max()
        sa, sb = ref_states[i], oth_states[i]
        m = min(len(sa), len(sb))
        diff = np.abs(sa[:m] - sb[:m]).max(axis=1)
        over = np.flatnonzero(diff > args.tol)
        same_until = int(over[0]) if len(over) else m
        box_diffs.append(box_diff)
        diverge.append(same_until)
        lengths.append(m)
        agree += a['success'] == b['success']
        if a['success_frame'] is not None and b['success_frame'] is not None:
            frame_diffs.append(b['success_frame'] - a['success_frame'])
        if not args.quiet:
            mark = '' if a['success'] == b['success'] else '  <- differs'
            print(f'{i:>3} {box_diff:>8.2f} {str(a["success"])[0]}/{str(b["success"])[0]:>7} '
                  f'{str(a["success_frame"]):>7}/{str(b["success_frame"]):<7} '
                  f'{same_until:>10} {diff.max():>9.4f}{mark}')

    diverge = np.array(diverge)
    print(f'box start: max difference {max(box_diffs):.2f} mm')
    print(f'outcome agrees in {agree}/{n} episodes')
    if frame_diffs:
        print(f'success_frame (other - reference), both successful: '
              f'mean {np.mean(frame_diffs):+.1f}, median {np.median(frame_diffs):+.0f}, '
              f'|Δ| <= 5 in {np.sum(np.abs(frame_diffs) <= 5)}/{len(frame_diffs)}')
    print(f'trajectories identical (tol {args.tol} rad) for: median {np.median(diverge):.0f} '
          f'frames, min {diverge.min()}; whole episode in {np.sum(diverge >= np.array(lengths))}/{n}')


if __name__ == '__main__':
    main()
