"""Online RL of a residual actor + TD3 critics on top of the frozen image ACT (scripts/rlt.py).

Trains on random boxes of --train-seed / --train-rel-yaw (default |g| in [30, 45] deg) and
evaluates the deterministic actor every --eval-every episodes on fixed boxes (default the
boundary evaluation: seed 100, rel yaw [35, 55]), counting strict successes. Writes
train.jsonl (one line per training episode), eval.jsonl and agent_<episode>.pt to
--output-dir.

    .venv-lerobot/bin/python scripts/train_rlt.py \\
        outputs/act_img_all_hold/checkpoints/100000/pretrained_model \\
        --token outputs/rlt/rl_token_v0.pt --output-dir outputs/rlt/run0
    # the base policy through the RL wrapper (actor = reference), full 900-frame episodes:
    ... --eval-only --full-eval --output-dir outputs/rlt/identity
    # a trained actor:
    ... --eval-only --agent outputs/rlt/run0/agent_1000.pt --full-eval --output-dir ...
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lerobot_policy import Policy  # noqa: E402  (first: keeps pixi's numpy out of sys.path)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_token import RLToken  # noqa: E402
from rlt import TD3, Replay, RLTEnv  # noqa: E402
from so101_mujoco.kinematics import TopDownIK  # noqa: E402
from so101_mujoco.pick_place_sim import PickPlaceEnv  # noqa: E402
from so101_mujoco.rollouts import summarize  # noqa: E402


def run_eval(renv, agent, seed, episodes, full):
    """Deterministic actor (agent None: the base policy) on the boxes of `seed`."""
    sim = renv.env.sim
    sim.rng = np.random.default_rng(seed)
    sim.reset()   # PickPlaceSim draws once at construction, like sim_node at start-up
    rows = []
    for _ in range(episodes):
        o, deltas = renv.reset(), []
        while True:
            ref_rel = (o['ref'] - o['state']).ravel()
            delta = agent.act(o['x'], ref_rel) if agent else np.zeros(6, np.float32)
            deltas.append(np.abs(delta).mean())
            o, r, done, info = renv.step(o['ref'] + delta.reshape(-1, 6))
            if done:
                break
        row = {**info, 'mean_abs_delta': float(np.mean(deltas))}
        if full:
            row['result'] = renv.finish_episode()
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--token', required=True, help='RL token from train_rl_token.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-seed', type=int, default=2000)
    parser.add_argument('--train-rel-yaw', type=float, nargs=2, default=(30.0, 60.0))
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=30,
                        help='episodes with actor = reference (+ noise) before actor updates')
    parser.add_argument('--utd', type=int, default=5, help='critic updates per transition')
    parser.add_argument('--actor-every', type=int, default=2,
                        help='critic updates per actor update')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--noise', type=float, default=0.02,
                        help='exploration noise (rad), one draw per joint per chunk')
    parser.add_argument('--scale', type=float, default=0.1, help='max |Δ| (rad)')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--action', choices=['chunk', 'offset'], default='chunk',
                        help='Δ over the whole chunk (paper) or one offset per joint')
    parser.add_argument('--reward', choices=['strict', 'shaped'], default='strict',
                        help='training reward (evaluation always counts strict successes)')
    parser.add_argument('--phase', type=int, nargs=2, default=(100, 240))
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-seed', type=int, default=100)
    parser.add_argument('--eval-rel-yaw', type=float, nargs=2, default=(35.0, 55.0))
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--full-eval', action='store_true',
                        help='run evaluation episodes to 900 frames and judge like sim_eval')
    parser.add_argument('--agent', help='agent checkpoint to start from / evaluate')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    policy = Policy(args.checkpoint, 'cpu')
    token, _ = RLToken.load(args.token)
    eval_env = PickPlaceEnv(cameras=policy.cameras, seed=args.eval_seed,
                            box_rel_yaw_range=args.eval_rel_yaw)
    kinematics = TopDownIK(eval_env.sim.model)
    eval_renv = RLTEnv(eval_env, policy, token, kinematics, phase=tuple(args.phase))
    agent = TD3(eval_renv.obs_dim, eval_renv.slot, args.scale, action=args.action,
                gamma=args.gamma, beta=args.beta)
    if args.agent:
        agent.load_state_dict(torch.load(args.agent, weights_only=False)['agent'])

    def evaluate(episode, use_agent):
        start = time.perf_counter()
        rows = run_eval(eval_renv, agent if use_agent else None, args.eval_seed,
                        args.eval_episodes, args.full_eval)
        line = {'episode': episode, 'strict': sum(r['strict'] for r in rows),
                'grasp_ok': sum(r['grasp_ok'] for r in rows), 'n': len(rows),
                'mean_abs_delta': float(np.mean([r['mean_abs_delta'] for r in rows])),
                'time_s': round(time.perf_counter() - start, 1)}
        if args.full_eval:
            line['summary'] = summarize([r['result'] for r in rows])
        with open(os.path.join(out_dir, 'eval.jsonl'), 'a') as f:
            f.write(json.dumps({**line, 'episodes': rows}) + '\n')
        print(f'[eval @ {episode}] strict {line["strict"]}/{line["n"]}, grasp ok '
              f'{line["grasp_ok"]}, mean |Δ| {line["mean_abs_delta"]:.4f} rad'
              + (f', success {line["summary"]["successes"]}, strict (900 frames) '
                 f'{line["summary"]["strict_successes"]}' if args.full_eval else '')
              + f', {line["time_s"]} s', flush=True)

    if args.eval_only:
        evaluate(0, bool(args.agent))
        return

    train_env = PickPlaceEnv(cameras=policy.cameras, seed=args.train_seed,
                             box_rel_yaw_range=args.train_rel_yaw)
    renv = RLTEnv(train_env, policy, token, kinematics, phase=tuple(args.phase),
                  reward=args.reward)
    replay = Replay(renv.obs_dim, agent.ref_dim, agent.delta_dim)
    evaluate(0, False)
    n_updates, recent, stats = 0, [], []
    train_log = open(os.path.join(out_dir, 'train.jsonl'), 'a')
    wall = time.perf_counter()
    for episode in range(1, args.episodes + 1):
        o = renv.reset()
        while True:
            ref_rel = (o['ref'] - o['state']).ravel()
            delta = agent.act(o['x'], ref_rel) if episode > args.warmup else \
                np.zeros(agent.delta_dim, np.float32)
            noise = rng.normal(0, args.noise, 6).astype(np.float32)   # one draw per chunk
            delta = delta + np.tile(noise, agent.delta_dim // 6)
            o2, r, done, info = renv.step(agent.chunk(o['ref'], delta))
            replay.add(x=o['x'], ref=ref_rel, delta=delta, r=r, done=float(done),
                       x2=o2['x'] if o2 else 0, ref2=(o2['ref'] - o2['state']).ravel() if o2 else 0)
            if len(replay) >= args.batch_size:
                for _ in range(args.utd):
                    b = replay.sample(args.batch_size, rng, agent.device)
                    c_loss, q = agent.update_critic(b)
                    n_updates += 1
                    a_loss = penalty = None
                    if episode > args.warmup and n_updates % args.actor_every == 0:
                        a_loss, penalty = agent.update_actor(b)
                stats.append((c_loss, q, penalty))
            if done:
                break
            o = o2
        recent = (recent + [info['strict']])[-50:]
        train_log.write(json.dumps({'episode': episode, 'reward': r, **info}) + '\n')
        train_log.flush()
        if episode % 10 == 0 and stats:
            s = np.array([(c, q, p if p is not None else np.nan) for c, q, p in stats[-200:]])
            print(f'episode {episode}: strict (last {len(recent)}) {np.mean(recent):.2f}, '
                  f'critic loss {np.nanmean(s[:, 0]):.4f}, Q {np.nanmean(s[:, 1]):.3f}, '
                  f'actor penalty {np.nanmean(s[:, 2]) if episode > args.warmup else 0:.4f}, '
                  f'{len(replay)} transitions, {time.perf_counter() - wall:.0f} s', flush=True)
        if episode % args.eval_every == 0 or episode == args.episodes:
            torch.save({'agent': agent.state_dict(), 'episode': episode, 'args': vars(args)},
                       os.path.join(out_dir, f'agent_{episode}.pt'))
            evaluate(episode, True)


if __name__ == '__main__':
    main()
