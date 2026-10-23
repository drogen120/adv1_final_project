"""Offline check of the expert's failed-grasp recovery (expert.PickPlaceEpisode), without ROS.

Runs episodes in plain MuJoCo the way collect_demos drives sim_node in sync_mode, with the
first grasp made to miss on purpose, at random box poses from the sim_node randomization
range. For each injection mode it reports how often the injected grasp really missed, how
often the episode then succeeded after regrasping, and why the others failed. Use it to pick
the injection ranges of collect_demos, and after changing the expert.

    pixi run python scripts/check_retry.py                      # collect_demos defaults
    pixi run python scripts/check_retry.py --yaw-error 15 25 --tool-error 0.010 0.015 -n 100
    pixi run python scripts/check_retry.py --modes midway --rel-yaw 30 60 --delay 0 0.3 -n 100
"""

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_expert import DEFAULT_MODEL, RATE, SETTLE_TIME, Checker  # noqa: E402
from so101_mujoco.expert import (PickPlaceEpisode, PickPlaceExpert, PlanningError,  # noqa: E402
                                 is_success, sample_injection)


def run_episode(checker, expert, x, y, yaw, injection):
    checker._reset(x, y, yaw)
    for _ in range(5):  # collect_demos waits for 5 states after the reset
        checker._step()
    d = checker.d
    episode = PickPlaceEpisode(expert, d.qpos[checker.qpos_adr].copy(), checker.target, injection)
    plan = np.zeros((0, 6))
    i = 0
    injected_wrist = None
    try:
        while True:
            if i == len(plan):
                rows = episode.next_rows(d.xpos[checker.box_id].copy(), d.xquat[checker.box_id].copy())
                if rows is None:
                    break
                if injected_wrist is None:
                    injected_wrist = rows[-1][4]  # wrist_roll of the injected grasp
                plan = np.vstack([plan, rows])
            checker._step(plan[i])
            i += 1
    except PlanningError as e:
        return {'success': False, 'reason': f'planning: {e}', 'attempts': episode.attempts}
    if episode.failure is None:
        for _ in range(int(round(SETTLE_TIME * RATE))):
            checker._step()
    success = episode.failure is None and is_success(d.xpos[checker.box_id], checker.target)
    reason = episode.failure or ('' if success else 'not placed')
    return {'success': success, 'reason': reason, 'attempts': episode.attempts, 'frames': i,
            'clip': i - (episode.retry_frame or 0), 'injected_wrist': injected_wrist}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--modes', nargs='+', default=['wrist', 'offset'])
    parser.add_argument('--yaw-error', type=float, nargs=2, default=[25.0, 35.0], help='deg')
    parser.add_argument('--wrist-tool-error', type=float, nargs=2, default=[0.010, 0.016],
                        help='m, tool error added to the wrist injections')
    parser.add_argument('--tool-error', type=float, nargs=2, default=[0.012, 0.02], help='m')
    parser.add_argument('--midway-fraction', type=float, nargs=2, default=[0.7, 1.0],
                        help='share of g the jaws are turned back by, midway injections')
    parser.add_argument('--midway-tool-error', type=float, nargs=2, default=[0.007, 0.011],
                        help='m, tool error of the midway injections')
    parser.add_argument('--delay', type=float, nargs=2, default=[0.0, 0.8], help='s')
    parser.add_argument('--radius', type=float, nargs=2, default=[0.18, 0.25])
    parser.add_argument('--azimuth', type=float, nargs=2, default=[5.0, 40.0], help='deg')
    parser.add_argument('--rel-yaw', type=float, nargs=2, default=None,
                        help='deg, box yaw relative to the arm-perpendicular direction, as '
                             'sim_node box_rel_yaw_range (default: any yaw)')
    parser.add_argument('-n', type=int, default=50, help='episodes per mode')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true', help='print every failure')
    args = parser.parse_args()

    checker = Checker(args.model)
    expert = PickPlaceExpert(checker.m, rate=RATE)
    rng = np.random.default_rng(args.seed)
    for mode in args.modes:
        results = []
        for _ in range(args.n):
            r = rng.uniform(*args.radius)
            a = np.radians(rng.uniform(*args.azimuth))
            if args.rel_yaw is None:
                yaw = rng.uniform(-np.pi, np.pi)
            else:
                yaw = a + np.pi / 2 + np.radians(rng.uniform(*args.rel_yaw))
            injection = sample_injection(rng, mode, args.yaw_error, args.wrist_tool_error,
                                         args.tool_error, args.delay, args.midway_fraction,
                                         args.midway_tool_error)
            res = run_episode(checker, expert, r * np.cos(a), r * np.sin(a), yaw, injection)
            results.append(res)
            if args.verbose and not res['success']:
                print(f'  {mode} FAIL r={r:.3f} az={np.degrees(a):.1f} '
                      f'rel_yaw={np.degrees((yaw - a + np.pi / 4) % (np.pi / 2) - np.pi / 4):.1f} '
                      f'{injection}: {res["reason"]}')
        ok = [x for x in results if x['success']]
        missed = [x for x in results if x['reason'] != 'injected grasp lifted the box']
        reasons = collections.Counter(x['reason'].split(':')[0] for x in results if not x['success'])
        print(f'{mode:>6}: injected grasp missed {len(missed)}/{len(results)}, '
              f'recovered {len(ok)}/{len(missed)}, '
              f'attempts {collections.Counter(x["attempts"] for x in ok)}, '
              f'clip frames {np.mean([x["clip"] for x in ok]):.0f} '
              f'(episode {np.mean([x["frames"] for x in ok]):.0f})' if ok else '')
        for reason, count in reasons.most_common():
            print(f'        {count:3d} x {reason}')
        wrist = np.array([x['injected_wrist'] for x in results if x.get('injected_wrist') is not None])
        print(f'        wrist_roll of the injected grasp: {wrist.mean():.2f} '
              f'[{wrist.min():.2f}, {wrist.max():.2f}]')


if __name__ == '__main__':
    main()
