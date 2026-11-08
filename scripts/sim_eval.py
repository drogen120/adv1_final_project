"""Closed-loop evaluation of a trained policy without ROS (so101_mujoco.pick_place_sim).

Does what sim_node (sync_mode) + policy_server.py + policy_eval do, in one process of the
LeRobot venv: same random boxes for a seed, same episode timing, same observations (images
rendered by the same CameraRenderer), same judging and output files (episodes.jsonl,
summary.json, rollouts.npz, see policy_eval.py). Images are only rendered on the frames where
the policy predicts a new chunk, so an image policy runs ~10x faster than through ROS.

    .venv-lerobot/bin/python scripts/sim_eval.py \
        outputs/act_img_all_hold/checkpoints/100000/pretrained_model \
        --seed 100 --episodes 50 --output-dir data/eval_py/act_img_all_hold_100k
    # boundary evaluation (|g| in [35, 45] deg), like sim_node -p box_rel_yaw_range:=[35.0,55.0]
    ... --rel-yaw 35 55

Compare with a ROS evaluation of the same checkpoint and seed episode by episode with
scripts/compare_evals.py.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lerobot_policy import Policy  # noqa: E402  (first: keeps pixi's numpy out of sys.path)

sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.kinematics import TopDownIK  # noqa: E402
from so101_mujoco.pick_place_sim import PickPlaceEnv  # noqa: E402
from so101_mujoco.rollouts import episode_result, save_rollouts, summarize  # noqa: E402


def _mm(meters):
    return '-' if meters is None else f'{1e3 * meters:+.1f} mm'


class ChunkRunner:
    """Executes a policy's chunks like LeRobot's select_action without temporal ensembling:
    predict a chunk, execute its first n_action_steps actions open loop, predict again. Asks
    the env for images only when it predicts."""

    def __init__(self, policy, n_action_steps=None):
        self.policy = policy
        self.n_action_steps = n_action_steps or policy.config.n_action_steps
        if policy.config.temporal_ensemble_coeff is not None:
            raise ValueError('temporal ensembling is not supported here; use policy_eval')
        self.queue = deque()

    def reset(self):
        self.queue.clear()

    def act(self, env, obs):
        if not self.queue:
            images = env.render() if self.policy.cameras else None
            chunk = self.policy.predict_chunk(obs['state'], obs['environment_state'], images)
            self.queue.extend(chunk[:self.n_action_steps])
        return self.queue.popleft()


def run_episode(env, runner, max_steps, box_pose=None):
    """One episode; returns (frames, box_init, box_final, policy_time) as policy_eval records
    them: frames are (state, environment_state, action, stamp_ns) lists."""
    obs = env.reset(box_pose)
    runner.reset()
    box_init = obs['box_pos']
    frames, policy_time = [], 0.0
    for _ in range(max_steps):
        start = time.perf_counter()
        action = [float(a) for a in runner.act(env, obs)]
        policy_time += time.perf_counter() - start
        frames.append((obs['state'].tolist(), obs['environment_state'].tolist(), action,
                       int(round(obs['time'] * 1e9))))
        obs = env.step(action)
    _, box_final = env.judge()
    return frames, box_init, box_final, policy_time


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('checkpoint', help='a pretrained_model directory written by lerobot-train')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=900)
    parser.add_argument('--rel-yaw', type=float, nargs=2, default=(-45.0, 45.0),
                        metavar=('LOW', 'HIGH'), help='box_rel_yaw_range (deg), see sim_node.py')
    parser.add_argument('--n-action-steps', type=int,
                        help='actions executed per predicted chunk (default: as trained)')
    parser.add_argument('--start-frame', type=int, default=7,
                        help='frames between the reset and the first observation')
    parser.add_argument('--device', help='torch device (default: cuda if available, else cpu)')
    args = parser.parse_args()

    policy = Policy(args.checkpoint, args.device)
    runner = ChunkRunner(policy, args.n_action_steps)
    env = PickPlaceEnv(cameras=policy.cameras, seed=args.seed, box_rel_yaw_range=args.rel_yaw,
                       start_frame=args.start_frame)
    kinematics = TopDownIK(env.sim.model)
    c = policy.config
    print(f'{c.type} policy on {policy.device}: chunk_size={c.chunk_size}, executing '
          f'{runner.n_action_steps} actions per chunk; inputs {policy.info()}', flush=True)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'checkpoint': os.path.abspath(args.checkpoint),
                   'n_action_steps': runner.n_action_steps}, f, indent=2)

    results, episode_frames = [], []
    wall_start = time.perf_counter()
    with open(os.path.join(out_dir, 'episodes.jsonl'), 'w') as f:
        for episode in range(args.episodes):
            start = time.perf_counter()
            frames, box_init, box_final, policy_time = run_episode(env, runner, args.max_steps)
            result = episode_result(frames, box_init, box_final, env.target_pos, kinematics,
                                    policy_time)
            results.append(result)
            episode_frames.append(frames)
            f.write(json.dumps({'episode': episode, **result}) + '\n')
            f.flush()
            outcome = ('strict success' if result['strict_success'] else 'success') \
            if result['success'] else \
                'FAILED (' + ('dropped/misplaced' if result['lifted'] else 'not lifted') + ')'
            print(f'episode {episode}: {outcome}, {100 * result["target_dist"]:.1f} cm from '
                  f'target, {result["grasp_attempts"]} grasp(s) (first at frame {result["grasp_frame"]}, '
                  f'lateral {_mm(result["grasp_lateral"])}), success_frame '
                  f'{result["success_frame"]}, {time.perf_counter() - start:.1f} s', flush=True)
    env.close()

    save_rollouts(os.path.join(out_dir, 'rollouts.npz'), episode_frames)
    summary = summarize(results)
    summary['wall_time_s'] = round(time.perf_counter() - wall_start, 1)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Success rate: {summary["successes"]}/{summary["episodes"]} (strict '
          f'{summary["strict_successes"]}), box lifted in '
          f'{summary["lifted"]}, regrasped in {summary["regrasped"]} '
          f'({summary["recovered"]} succeeded), {summary["successes_by_frame_600"]} succeeded '
          f'within 600 frames; {summary["wall_time_s"]} s')


if __name__ == '__main__':
    main()
