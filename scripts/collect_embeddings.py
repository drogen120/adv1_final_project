"""Collect ACT encoder embeddings along closed-loop rollouts, to train the RL token.

Runs the policy like scripts/sim_eval.py (chunks executed n_action_steps at a time, default
as trained) and, every --slot frames up to --max-frame, also takes the ACT transformer
encoder's output for the current observation (the online RL actor/critic decide every slot
frames and see this observation through the RL token). Per slot it saves:

  embeddings.npy   (N, n_tokens, dim_model) float16, the encoder output
  slots.npz        episode, frame, state (JOINT_NAMES order), reference (the next slot
                   actions of the executed chunk, i.e. what the base policy does next),
                   and privileged labels for probing the token (never policy inputs):
                   box_pos, box_quat, box_in_tool (box centre in the gripperframe axes:
                   approach, lateral, jaw; m) and yaw_err (deg, jaw axis vs box face)

plus episodes.jsonl / summary.json / rollouts.npz of the whole episodes, as sim_eval writes.
The image renders and encoder passes of the extra slots do not change the rollout.

    .venv-lerobot/bin/python scripts/collect_embeddings.py \\
        outputs/act_img_all_hold/checkpoints/100000/pretrained_model \\
        --seed 1000 --episodes 100 --output-dir data/rlt/emb_normal
    ... --seed 1001 --rel-yaw 30 60 --output-dir data/rlt/emb_boundary   # |g| in [30, 45]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lerobot_policy import Policy  # noqa: E402  (first: keeps pixi's numpy out of sys.path)
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.kinematics import TopDownIK  # noqa: E402
from so101_mujoco.pick_place_sim import PickPlaceEnv  # noqa: E402
from so101_mujoco.rollouts import episode_result, save_rollouts, summarize  # noqa: E402


def box_yaw(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tool_labels(kinematics, state, box_pos, box_quat):
    """(box centre in the tool axes (3,), yaw error deg) for a frame."""
    tool, rot = kinematics.forward(np.asarray(state[:5]))
    box_in_tool = rot.T @ (np.asarray(box_pos) - tool)
    jaw_yaw = math.degrees(math.atan2(rot[1, 2], rot[0, 2]))
    yaw_err = (jaw_yaw - math.degrees(box_yaw(box_quat)) + 45) % 90 - 45
    return box_in_tool, yaw_err


def run_episode(env, policy, kinematics, n_action_steps, slot, max_frame, max_steps):
    obs = env.reset()
    box_init = obs['box_pos']
    queue = deque()
    frames, slots, policy_time = [], [], 0.0
    for t in range(max_steps):
        start = time.perf_counter()
        need_chunk = not queue
        take_slot = t % slot == 0 and t < max_frame
        if need_chunk or take_slot:
            images = env.render()
            chunk, emb = policy.predict_chunk_and_embedding(obs['state'], None, images)
            if need_chunk:
                queue.extend(chunk[:n_action_steps])
            if take_slot:
                if len(queue) < slot:
                    raise ValueError('slot must divide n_action_steps')
                box_in_tool, yaw_err = tool_labels(kinematics, obs['state'], obs['box_pos'],
                                                   obs['box_quat'])
                slots.append(dict(frame=t, emb=emb.astype(np.float16), state=obs['state'],
                                  reference=np.array(list(queue)[:slot]),
                                  box_pos=obs['box_pos'], box_quat=obs['box_quat'],
                                  box_in_tool=box_in_tool, yaw_err=yaw_err))
        action = [float(a) for a in queue.popleft()]
        policy_time += time.perf_counter() - start
        frames.append((obs['state'].tolist(), obs['environment_state'].tolist(), action,
                       int(round(obs['time'] * 1e9))))
        obs = env.step(action)
    _, box_final = env.judge()
    return frames, slots, box_init, box_final, policy_time


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=900)
    parser.add_argument('--rel-yaw', type=float, nargs=2, default=(-45.0, 45.0),
                        metavar=('LOW', 'HIGH'))
    parser.add_argument('--n-action-steps', type=int)
    parser.add_argument('--slot', type=int, default=10, help='frames between embeddings')
    parser.add_argument('--max-frame', type=int, default=300,
                        help='take embeddings in frames [0, max-frame)')
    parser.add_argument('--device')
    args = parser.parse_args()

    policy = Policy(args.checkpoint, args.device)
    n_action_steps = args.n_action_steps or policy.config.n_action_steps
    env = PickPlaceEnv(cameras=policy.cameras, seed=args.seed, box_rel_yaw_range=args.rel_yaw)
    kinematics = TopDownIK(env.sim.model)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'checkpoint': os.path.abspath(args.checkpoint),
                   'n_action_steps': n_action_steps, 'start_frame': env.start_frame}, f, indent=2)

    n_slots = len(range(0, min(args.max_frame, args.max_steps), args.slot))
    # latent + state (+ env) + one token per cell of each camera's ResNet feature map (stride 32)
    n_tokens = 2 + policy.uses_env + sum(math.ceil(h / 32) * math.ceil(w / 32)
                                         for _, h, w in policy.image_shapes.values())
    dim = policy.config.dim_model
    embeddings = np.lib.format.open_memmap(
        os.path.join(out_dir, 'embeddings.npy'), mode='w+', dtype=np.float16,
        shape=(args.episodes * n_slots, n_tokens, dim))
    columns = {k: [] for k in ['episode', 'frame', 'state', 'reference', 'box_pos', 'box_quat',
                               'box_in_tool', 'yaw_err']}
    results, episode_frames, i = [], [], 0
    wall = time.perf_counter()
    with open(os.path.join(out_dir, 'episodes.jsonl'), 'w') as f:
        for episode in range(args.episodes):
            start = time.perf_counter()
            frames, slots, box_init, box_final, policy_time = run_episode(
                env, policy, kinematics, n_action_steps, args.slot, args.max_frame,
                args.max_steps)
            for s in slots:
                embeddings[i] = s.pop('emb')
                i += 1
                for k, v in s.items():
                    columns[k].append(v)
                columns['episode'].append(episode)
            result = episode_result(frames, box_init, box_final, env.target_pos, kinematics,
                                    policy_time)
            results.append(result)
            episode_frames.append(frames)
            f.write(json.dumps({'episode': episode, **result}) + '\n')
            f.flush()
            print(f'episode {episode}: success {result["success"]}, strict '
                  f'{result["strict_success"]}, grasp frame {result["grasp_frame"]}, '
                  f'{time.perf_counter() - start:.1f} s', flush=True)
    env.close()
    embeddings.flush()
    np.savez(os.path.join(out_dir, 'slots.npz'),
             **{k: np.array(v, dtype=np.int32 if k in ('episode', 'frame') else np.float32)
                for k, v in columns.items()})
    save_rollouts(os.path.join(out_dir, 'rollouts.npz'), episode_frames)
    summary = summarize(results)
    summary['wall_time_s'] = round(time.perf_counter() - wall, 1)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'{i} slots; success {summary["successes"]}/{summary["episodes"]}, strict '
          f'{summary["strict_successes"]}; {summary["wall_time_s"]} s')


if __name__ == '__main__':
    main()
