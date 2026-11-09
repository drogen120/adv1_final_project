"""Side-by-side videos of the base image ACT and the RL actor (train_rlt.py) on the same boxes.

Replays the box draws of an evaluation seed to get the boxes of the chosen episodes, runs
each episode once with actor = reference (the base policy) and once with the trained agent,
exactly as `train_rlt.py --eval-only --full-eval` does, and checks the grasp frame, lateral
offset and outcome against that run's eval.jsonl. Each video shows the overview camera (top)
and a close-up of the box (bottom) for both runs, with the live lateral offset (box centre
minus tool along the jaw axis) while the tool is near the box before frame 260; frames --slow-from..--slow-to
(the grasp) play at 1/--slow-factor speed.

    .venv-lerobot/bin/python scripts/render_rlt_compare.py \\
        outputs/act_img_all_hold/checkpoints/100000/pretrained_model \\
        --token outputs/rlt/rl_token_v0.pt --agent outputs/rlt/run1/agent_600.pt \\
        --base-eval outputs/rlt/eval_base_b101 --rl-eval outputs/rlt/eval_run1_b101 \\
        --episodes 1 22 49 55 60 61 68 75 73 --output-dir outputs/rlt/videos_b101
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lerobot_policy import Policy  # noqa: E402  (first: keeps pixi's numpy out of sys.path)
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_token import RLToken  # noqa: E402
from rlt import TD3, RLTEnv  # noqa: E402
from so101_mujoco.kinematics import TopDownIK  # noqa: E402
from so101_mujoco.pick_place_sim import PickPlaceEnv  # noqa: E402
from so101_mujoco.rollouts import STRICT_LATERAL, grasp_lateral  # noqa: E402

W, TOP_H, BOT_H, HEAD_H = 640, 480, 360, 70


def box_poses(sim, seed, n):
    """(x, y, yaw) of the first n episodes of `seed`, as run_eval draws them."""
    poses, place = [], sim.place_box
    sim.place_box = lambda x, y, yaw: (poses.append((x, y, yaw)), place(x, y, yaw))
    sim.rng = np.random.default_rng(seed)
    for _ in range(n + 1):   # the first reset is the one at construction
        sim.reset()
    sim.place_box = place
    return poses[1:]


def run(renv, agent, pose):
    """One evaluation episode; returns (qpos per frame, info, result)."""
    qpos = []
    apply = renv._apply

    def recording_apply(action):
        qpos.append(renv.env.sim.data.qpos.copy())
        apply(action)

    renv._apply = recording_apply
    try:
        o = renv.reset(pose)
        while True:
            ref_rel = (o['ref'] - o['state']).ravel()
            delta = agent.act(o['x'], ref_rel) if agent else np.zeros(6, np.float32)
            o, _, done, info = renv.step(o['ref'] + delta.reshape(-1, 6))
            if done:
                break
        result = renv.finish_episode()
    finally:
        renv._apply = apply
    return np.array(qpos), list(renv.frames), info, result


class SceneRenderer:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.vis.global_.offwidth = max(W, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(TOP_H, self.model.vis.global_.offheight)
        self.data = mujoco.MjData(self.model)
        self.top = mujoco.Renderer(self.model, TOP_H, W)
        self.bottom = mujoco.Renderer(self.model, BOT_H, W)
        self.closeup = mujoco.MjvCamera()
        self.closeup.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.closeup.distance = 0.17
        self.closeup.elevation = -12.0

    def aim(self, box_pos):
        # From outside the box's radius towards the base: the jaws close across the arm,
        # so a sideways offset of the box between them is seen side-on.
        self.closeup.lookat[:] = box_pos[0], box_pos[1], 0.04
        self.closeup.azimuth = np.degrees(np.arctan2(box_pos[1], box_pos[0])) + 180.0

    def render(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.top.update_scene(self.data, 'overview')
        self.bottom.update_scene(self.data, self.closeup)
        return np.vstack([self.top.render(), self.bottom.render()])


def put(img, text, xy, colour=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def panel(renderer, run_data, frame, title, kinematics):
    qpos, frames, result = run_data
    f = min(frame, len(qpos) - 1)
    img = renderer.render(qpos[f]).copy()
    head = np.full((HEAD_H, W, 3), 30, np.uint8)
    strict = result['strict_success']
    lat = result['grasp_lateral']
    verdict = ('STRICT OK' if strict else 'strict FAIL') + \
        (', success' if result['success'] else ', task FAIL')
    colour = (80, 220, 80) if strict else (255, 90, 90)
    put(head, title, (10, 24), scale=0.65, thick=2)
    put(head, f'{verdict} | grasp @ frame {result["grasp_frame"]}, lateral '
              f'{1000 * lat:+.1f} mm (limit {1000 * STRICT_LATERAL:.0f})' if lat is not None
        else verdict, (10, 54), colour, scale=0.55)
    state, env = frames[f][0], frames[f][1]
    tool, _ = kinematics.forward(np.array(state[:5]))
    near = f < 260 and np.linalg.norm(np.asarray(env[:3]) - tool) < 0.04   # grasp only
    live = f'lateral now {1000 * grasp_lateral(frames[f], kinematics):+.1f} mm' if near else ''
    put(img, f'frame {f}', (10, 26))
    if f == result['grasp_frame']:
        put(img, 'GRASP (gripper starts closing)', (10, 54), (255, 220, 0))
    put(img, live, (10, TOP_H + 26), (255, 220, 0))
    return np.vstack([head, img])


def write_video(path, renderer, base, rl, kinematics, frames, fps, title_rl, slow_text):
    first = base[0][0]
    box = first[renderer.model.jnt_qposadr[renderer.model.body('box').jntadr[0]]:][:3]
    renderer.aim(box)
    h = HEAD_H + TOP_H + BOT_H
    ffmpeg = subprocess.Popen(
        [shutil.which('ffmpeg') or 'ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
         '-pix_fmt', 'rgb24', '-s', f'{2 * W + 4}x{h}', '-r', f'{fps:g}', '-i', '-',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', path], stdin=subprocess.PIPE)
    for frame, slow in frames:
        left = panel(renderer, base, frame, 'ACT baseline', kinematics)
        right = panel(renderer, rl, frame, title_rl, kinematics)
        canvas = np.hstack([left, np.full((h, 4, 3), 255, np.uint8), right])
        if slow:
            for x in (W - 150, 2 * W + 4 - 150):
                put(canvas, slow_text, (x, HEAD_H + 26), (255, 220, 0))
        ffmpeg.stdin.write(np.ascontiguousarray(canvas).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit(f'ffmpeg failed writing {path}')


def check(name, ep, info, result, eval_rows):
    ref = eval_rows[ep]['result']
    keys = ('success', 'strict_success', 'grasp_frame', 'grasp_lateral', 'success_frame')
    got = {k: (round(result[k], 4) if isinstance(result[k], float) else result[k]) for k in keys}
    want = {k: ref[k] for k in keys}
    same = got == want
    print(f'  {name}: {got}' + ('  (= eval.jsonl)' if same else f'  != eval.jsonl {want}'),
          flush=True)
    return same


def load_eval(path):
    with open(os.path.join(path, 'eval.jsonl')) as f:
        return json.loads(f.readlines()[-1])['episodes']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--token', required=True)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--base-eval', required=True, help='eval dir of the base policy')
    parser.add_argument('--rl-eval', required=True, help='eval dir of the agent')
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--slow-from', type=int, default=130)
    parser.add_argument('--slow-to', type=int, default=200)
    parser.add_argument('--slow-factor', type=int, default=4)
    parser.add_argument('--end-frame', type=int, default=500,
                        help='last frame shown (the box is on the target by ~350)')
    args = parser.parse_args()

    cfg = json.load(open(os.path.join(args.rl_eval, 'config.json')))
    base_rows, rl_rows = load_eval(args.base_eval), load_eval(args.rl_eval)
    torch.set_num_threads(4)
    policy = Policy(args.checkpoint, 'cpu')
    token, _ = RLToken.load(args.token)
    env = PickPlaceEnv(cameras=policy.cameras, seed=cfg['eval_seed'],
                       box_rel_yaw_range=cfg['eval_rel_yaw'])
    kinematics = TopDownIK(env.sim.model)
    renv = RLTEnv(env, policy, token, kinematics, phase=tuple(cfg['phase']))
    agent = TD3(renv.obs_dim, renv.slot, cfg['scale'], action=cfg['action'],
                gamma=cfg['gamma'], beta=cfg['beta'])
    agent.load_state_dict(torch.load(args.agent, weights_only=False)['agent'])
    poses = box_poses(env.sim, cfg['eval_seed'], max(args.episodes) + 1)

    os.makedirs(args.output_dir, exist_ok=True)
    renderer = SceneRenderer(env.sim.model_path)
    frames = []
    for f in range(args.end_frame + 1):
        slow = args.slow_from <= f < args.slow_to
        frames += [(f, slow)] * (args.slow_factor if slow else 1)
    title_rl = f'ACT + RL ({os.path.basename(os.path.dirname(args.agent))}/' \
               f'{os.path.splitext(os.path.basename(args.agent))[0]})'
    for ep in args.episodes:
        print(f'episode {ep}: box {np.round(poses[ep], 4).tolist()}', flush=True)
        q_b, fr_b, info_b, res_b = run(renv, None, poses[ep])
        ok_b = check('base', ep, info_b, res_b, base_rows)
        q_r, fr_r, info_r, res_r = run(renv, agent, poses[ep])
        ok_r = check('RL  ', ep, info_r, res_r, rl_rows)
        tag = ('rl_only' if res_r['strict_success'] else 'base_only') \
            if res_r['strict_success'] != res_b['strict_success'] else 'same'
        path = os.path.join(args.output_dir, f'episode_{ep:03d}_{tag}.mp4')
        write_video(path, renderer, (q_b, fr_b, res_b), (q_r, fr_r, res_r), kinematics, frames,
                    50, title_rl, f'{1 / args.slow_factor:g}x slow')
        print(f'  -> {path}' + ('' if ok_b and ok_r else '  (WARNING: differs from eval)'),
              flush=True)


if __name__ == '__main__':
    main()
