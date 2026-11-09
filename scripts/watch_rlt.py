"""Watch the online RL loop of scripts/train_rlt.py in a MuJoCo viewer.

Same loop as train_rlt.py (env step -> replay -> critic/actor updates), but the training
env is shown in a viewer and every decision is printed. Nothing is written to outputs/.

    .venv-lerobot/bin/mjpython scripts/watch_rlt.py [--speed 0.5] [--episodes 20]
    # a fresh actor (starts as the base ACT):
    .venv-lerobot/bin/mjpython scripts/watch_rlt.py --agent ''

The viewer (launch_passive) needs mjpython. Closing the window stops the run.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))   # default paths are relative to the project root
from lerobot_policy import Policy  # noqa: E402  (first, as in train_rlt.py)
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_token import RLToken  # noqa: E402
from rlt import TD3, Replay, RLTEnv  # noqa: E402
from so101_mujoco.kinematics import TopDownIK  # noqa: E402
from so101_mujoco.pick_place_sim import PickPlaceEnv  # noqa: E402

JOINTS = ['pan', 'lift', 'elbow', 'wflex', 'wroll', 'grip']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='outputs/act_img_all_hold/checkpoints/100000/pretrained_model')
    p.add_argument('--token', default='outputs/rlt/rl_token_v0.pt')
    p.add_argument('--agent', default='outputs/rlt/run1/agent_600.pt',
                   help="start from this agent ('' = fresh actor, i.e. the base ACT)")
    p.add_argument('--speed', type=float, default=1.0, help='1 = real time')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64,
                   help='updates start once the replay has this many transitions '
                        '(train_rlt uses 256; smaller so they start sooner here)')
    p.add_argument('--noise', type=float, default=0.005)
    p.add_argument('--seed', type=int, default=2000)
    args = p.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)

    # --- same objects as train_rlt.py (run1 settings: offset action, shaped reward) ---
    policy = Policy(args.checkpoint, 'cpu')
    token, _ = RLToken.load(args.token)
    env = PickPlaceEnv(cameras=policy.cameras, seed=args.seed, box_rel_yaw_range=(30.0, 60.0))
    kin = TopDownIK(env.sim.model)
    renv = RLTEnv(env, policy, token, kin, phase=(100, 240), reward='shaped')
    agent = TD3(renv.obs_dim, renv.slot, 0.05, action='offset')
    if args.agent:
        agent.load_state_dict(torch.load(args.agent, weights_only=False)['agent'])
    replay = Replay(renv.obs_dim, agent.ref_dim, agent.delta_dim)

    # --- viewer: hook env.step so every simulated frame is drawn ---
    viewer = mujoco.viewer.launch_passive(env.sim.model, env.sim.data,
                                          show_left_ui=False, show_right_ui=False)
    viewer.cam.lookat[:] = (0.2, 0.0, 0.05)
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.9, 200, -35
    hud = {'episode': 0, 'mode': '', 'delta': '', 'reward': '', 'updates': 0, 'replay': 0,
           'q': ''}
    dt = 1.0 / 50.0 / args.speed
    step_frame = env.step

    def draw():
        left = 'episode\nframe\nwho acts\nDelta (mrad)\nreplay / updates\ncritic Q\nlast reward'
        right = (f"{hud['episode']}\n{renv.t}\n{hud['mode']}\n{hud['delta']}\n"
                 f"{hud['replay']} / {hud['updates']}\n{hud['q']}\n{hud['reward']}")
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          left, right))

    def mode_for(t):
        if t < renv.phase_start:
            return 'ACT (frozen)'
        if t < renv.phase_end:
            return 'RL actor = ACT + Delta'
        return 'ACT -> judge @ 500'

    def watched_step(action):
        start = time.perf_counter()
        obs = step_frame(action)
        hud['mode'] = mode_for(renv.t + 1)
        draw()
        viewer.sync()
        if not viewer.is_running():
            raise SystemExit
        time.sleep(max(0.0, dt - (time.perf_counter() - start)))
        return obs

    env.step = watched_step

    n_updates = 0
    for episode in range(1, args.episodes + 1):
        hud.update(episode=episode, delta='', reward='')
        print(f'\n=== episode {episode}: reset, ACT runs frames 0-99 alone ===', flush=True)
        o = renv.reset()                       # frames 0..99: frozen ACT
        while True:
            # 1) actor decides: Δ from (z_rl, joint state, phase) and the ACT reference
            ref_rel = (o['ref'] - o['state']).ravel()
            delta = agent.act(o['x'], ref_rel)
            noise = rng.normal(0, args.noise, 6).astype(np.float32)
            delta = delta + noise
            hud['delta'] = ' '.join(f'{d * 1e3:+.1f}' for d in delta)
            t0 = renv.t
            # 2) env executes the 10 actions (ref + Δ) -> next obs, reward
            o2, r, done, info = renv.step(agent.chunk(o['ref'], delta))
            # 3) store the transition
            replay.add(x=o['x'], ref=ref_rel, delta=delta, r=r, done=float(done),
                       x2=o2['x'] if o2 else 0,
                       ref2=(o2['ref'] - o2['state']).ravel() if o2 else 0)
            # 4) learn from random past transitions (5 critic updates, actor every 2nd)
            msg = 'replay too small, no update yet'
            if len(replay) >= args.batch_size:
                for _ in range(5):
                    c_loss, q = agent.update_critic(replay.sample(args.batch_size, rng, 'cpu'))
                    n_updates += 1
                    if n_updates % 2 == 0:
                        agent.update_actor(replay.sample(args.batch_size, rng, 'cpu'))
                hud['q'] = f'{q:.3f}'
                msg = f'5 critic + actor updates, critic loss {c_loss:.4f}, Q {q:.3f}'
            hud.update(updates=n_updates, replay=len(replay))
            span = f'{t0:3d}-{t0 + 9:3d}' + (f' (+ACT to {renv.t - 1})' if renv.t > t0 + 10 else '')
            print(f'  frames {span}  Delta(mrad) '
                  + ' '.join(f'{j}{d * 1e3:+5.1f}' for j, d in zip(JOINTS, delta))
                  + f'  r={r:.2f}  | {msg}', flush=True)
            if done:
                break
            o = o2
        lat = info['grasp_lateral']
        hud['reward'] = f'{r:.2f}'
        print(f'  -> grasp frame {info["grasp_frame"]}, lateral '
              f'{lat * 1e3 if lat is not None else float("nan"):.1f} mm, grasp ok '
              f'{info["grasp_ok"]}, success {info["success"]}, reward {r:.2f}', flush=True)
        draw()
        viewer.sync()
        time.sleep(1.5)
    viewer.close()


if __name__ == '__main__':
    main()
