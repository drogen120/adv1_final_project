"""Checks for rendering camera images from recorded states (step 1 of the image policy).

    pixi run python scripts/check_render.py consistency   # replay vs physical state, determinism
    pixi run python scripts/check_render.py sheet         # contact sheets of recorded episodes
    pixi run python scripts/check_render.py visibility    # how much of the box each camera sees
    pixi run python scripts/check_render.py timing        # render cost per frame

consistency runs the expert in plain MuJoCo the way sim_node steps (check_expert.Checker) and,
at every published state, compares the image rendered from the published values (joints =
data.qpos, box = data.xpos/xquat, what the bags contain) with the image of the full physical
state, and with a second, independent renderer (must be identical).

sheet renders key frames of a few recorded episodes, chosen at the corners of the box
randomization range, for both cameras, into data/vision_check/ (new; nothing else is written).
"""

import argparse
import json
import os
import sys
import time

import cv2
import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.cameras import CAMERAS, CameraRenderer  # noqa: E402

DEFAULT_MODEL = os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco', 'models',
                             'robotstudio_so101', 'scene_pick_place.xml')
OUT_DIR = os.path.join(REPO, 'data', 'vision_check')


def consistency(args):
    from check_expert import RATE, SETTLE_TIME, Checker
    from so101_mujoco.expert import PickPlaceExpert

    checker = Checker(args.model)
    m, d = checker.m, checker.d
    expert = PickPlaceExpert(m, rate=RATE)
    replay = CameraRenderer(m)
    replay2 = CameraRenderer(m)
    true_data = mujoco.MjData(m)
    true_renderer = mujoco.Renderer(m, replay.renderer.height, replay.renderer.width)

    poses = [(0.18, 5.0, -40.0), (0.25, 40.0, 30.0), (0.21, 20.0, 44.0)]
    diffs = {c: [] for c in CAMERAS}
    identical = True
    for r, az, rel in poses:
        a = np.radians(az)
        checker._reset(r * np.cos(a), r * np.sin(a), a + np.radians(rel))
        for _ in range(5):
            checker._step()
        plan = expert.plan(d.qpos[checker.qpos_adr].copy(), d.xpos[checker.box_id].copy(),
                           d.xquat[checker.box_id].copy(), checker.target)
        for i in range(len(plan) + int(round(SETTLE_TIME * RATE))):
            checker._step(plan[min(i, len(plan) - 1)])
            published = (d.qpos[checker.qpos_adr].copy(), d.xpos[checker.box_id].copy(),
                         d.xquat[checker.box_id].copy())
            images = replay.render_state(*published)
            if i % 10 == 0:
                identical &= all(np.array_equal(images[c], v)
                                 for c, v in replay2.render_state(*published).items())
            mujoco.mj_copyData(true_data, m, d)
            mujoco.mj_forward(m, true_data)
            for c in CAMERAS:
                true_renderer.update_scene(true_data, camera=c, scene_option=replay.option)
                diff = np.abs(true_renderer.render().astype(int) - images[c]).max(axis=2)
                diffs[c].append(((diff > 0).mean(), (diff > 16).mean(), diff.max(), i,
                                 float(d.xpos[checker.box_id][2])))
        print(f'r={r} az={az} rel_yaw={rel}: {len(plan)} plan rows, '
              f'success={checker.d.xpos[checker.box_id][2] < 0.03}')
    print(f'\nsecond renderer gives identical pixels: {identical}')
    print('published-state image vs physical-state image, per frame:')
    for c in CAMERAS:
        v = np.array(diffs[c])
        print(f'  {c:10s} pixels differing: mean {v[:, 0].mean() * 100:.3f}% max {v[:, 0].max() * 100:.3f}%'
              f' | by >16/255: mean {v[:, 1].mean() * 100:.3f}% max {v[:, 1].max() * 100:.3f}%'
              f' | max abs diff {int(v[:, 2].max())}')
        w = v[np.argmax(v[:, 0])]
        print(f'  {"":10s} worst frame: step {int(w[3])} of its episode, box z {w[4]:.3f} m')


def load_episodes(demo_dirs):
    entries = []
    for demo_dir in demo_dirs:
        with open(os.path.join(demo_dir, 'episodes.jsonl')) as f:
            entries += [{**json.loads(line), 'dir': demo_dir} for line in f if line.strip()]
    return [e for e in entries if e['success'] and e['bag']]


def pick_corners(entries):
    """Episodes closest to the corners of (radius, azimuth) and to relative yaw +-45 / 0."""
    def polar(e):
        x, y = e['box_init'][:2]
        w, _, _, z = e['box_quat_init']
        az = np.arctan2(y, x)
        rel = (np.degrees(2 * np.arctan2(z, w) - az) + 45) % 90 - 45
        return np.hypot(x, y), np.degrees(az), rel
    feats = np.array([polar(e) for e in entries])
    goals = [(0.18, 5, None), (0.18, 40, None), (0.25, 5, None), (0.25, 40, None),
             (None, None, 44), (None, None, -44), (None, None, 0)]
    chosen = []
    for goal in goals:
        cost = sum(((feats[:, k] - g) / s) ** 2 for k, (g, s) in enumerate(zip(goal, (0.07, 35, 45)))
                   if g is not None)
        for i in np.argsort(cost):
            if i not in chosen:
                chosen.append(int(i))
                break
    return [(entries[i], feats[i]) for i in chosen]


def sheet(args):
    from so101_mujoco.collect_demos import pose_to_array
    from so101_mujoco.export_dataset import ordered, read_bag

    model = mujoco.MjModel.from_xml_path(args.model)
    renderer = CameraRenderer(model)
    os.makedirs(OUT_DIR, exist_ok=True)
    picked = pick_corners(load_episodes(args.demo_dirs))
    rows = []
    for entry, (r, az, rel) in picked:
        topics = read_bag(os.path.join(entry['dir'], entry['bag']))
        commands = {t for t, _ in topics['joint_command']}
        states = [(t, m) for t, m in topics['joint_states'] if t in commands]
        states = states[entry.get('clip_start', 0):]
        boxes = topics['box_pose']
        box_stamps = np.array([t for t, _ in boxes])
        n = len(states)
        frames = [int(f * (n - 1)) for f in (0.0, 0.3, 0.42, 0.5, 0.65, 0.85, 1.0)]
        tiles = []
        for c in CAMERAS:
            row = []
            for f in frames:
                t, msg = states[f]
                k = max(0, np.searchsorted(box_stamps, t, side='right') - 1)
                assert boxes[k][0] == t, 'box pose and joint state stamps differ'
                img = renderer.render_state(ordered(msg), *pose_to_array(boxes[k][1]))[c]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img, f'{c} f{f}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                row.append(img)
            tiles.append(np.hstack(row))
        label = (f'{os.path.basename(entry["dir"])}/{entry["bag"]}  r={r:.3f} az={az:.1f} '
                 f'rel_yaw={rel:.1f}  frames={n}')
        bar = np.full((26, tiles[0].shape[1], 3), 40, np.uint8)
        cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        rows.append(np.vstack([bar, *tiles]))
        path = os.path.join(OUT_DIR, f'{os.path.basename(entry["dir"])}_{entry["bag"]}.png')
        cv2.imwrite(path, rows[-1])
        print(label, '->', path)


def visibility(args):
    """Fraction of the box visible in each camera at the first exported frame of every
    episode: box pixels in a segmentation render, over the same with the arm hidden."""
    from so101_mujoco.collect_demos import pose_to_array
    from so101_mujoco.export_dataset import ordered, read_bag

    model = mujoco.MjModel.from_xml_path(args.model)
    renderer = CameraRenderer(model)
    seg = mujoco.Renderer(model, renderer.renderer.height, renderer.renderer.width)
    seg.enable_segmentation_rendering()
    box_geom = model.geom('box').id
    robot = [g for g in range(model.ngeom) if g not in (box_geom, model.geom('floor').id)]
    alpha = model.geom_rgba[robot, 3].copy()

    def box_pixels(cam, hide_arm):
        model.geom_rgba[robot, 3] = 0.0 if hide_arm else alpha
        seg.update_scene(renderer.data, camera=cam, scene_option=renderer.option)
        ids = seg.render()
        return int(((ids[..., 0] == box_geom) & (ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)).sum())

    rows = []
    for entry in load_episodes(args.demo_dirs):
        topics = read_bag(os.path.join(entry['dir'], entry['bag']))
        commands = {t for t, _ in topics['joint_command']}
        states = [(t, m) for t, m in topics['joint_states'] if t in commands]
        t, msg = states[entry.get('clip_start', 0)]
        box = dict(topics['box_pose'])[t]
        renderer.render_state(ordered(msg), *pose_to_array(box))
        row = []
        for cam in CAMERAS:
            full = box_pixels(cam, True)
            row += [box_pixels(cam, False) / full if full else 0.0, full]
        x, y = entry['box_init'][:2]
        rows.append([np.hypot(x, y), np.degrees(np.arctan2(y, x)), 'clip_start' in entry, *row])
    model.geom_rgba[robot, 3] = alpha
    v = np.array(rows, dtype=float)
    for k, cam in enumerate(CAMERAS):
        frac, full = v[:, 3 + 2 * k], v[:, 4 + 2 * k]
        print(f'{cam}: box pixels without arm median {np.median(full):.0f}; visible fraction '
              f'min {frac.min():.2f}, <0.5 in {(frac < 0.5).sum()}, <0.9 in {(frac < 0.9).sum()}, '
              f'=0 in {(frac == 0).sum()} of {len(v)} episodes')
    worst = v[np.argsort(v[:, 3])[:8]]
    print('least visible in overview (r, az, retry, overview frac):')
    for r, az, retry, f, *_ in worst:
        print(f'  r={r:.3f} az={az:5.1f} retry={bool(retry)} {f:.2f}')
    np.save(os.path.join(OUT_DIR, 'visibility.npy'), v)


def timing(args):
    model = mujoco.MjModel.from_xml_path(args.model)
    renderer = CameraRenderer(model)
    joints = np.zeros(6)
    pos, quat = np.array([0.2, 0.1, 0.02]), np.array([1.0, 0, 0, 0])
    renderer.render_state(joints, pos, quat)
    n = 200
    start = time.perf_counter()
    for i in range(n):
        joints[0] = 0.002 * i
        renderer.render_state(joints, pos, quat)
    per = (time.perf_counter() - start) / n
    print(f'{len(CAMERAS)} cameras at {renderer.renderer.height}x{renderer.renderer.width}: '
          f'{per * 1e3:.1f} ms per frame ({1 / per:.1f} frames/s)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('check', choices=['consistency', 'sheet', 'visibility', 'timing'])
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--demo-dirs', nargs='+',
                        default=[os.path.join(REPO, 'data', 'demos_offset'),
                                 os.path.join(REPO, 'data', 'demos_retry')])
    args = parser.parse_args()
    {'consistency': consistency, 'sheet': sheet, 'visibility': visibility,
     'timing': timing}[args.check](args)


if __name__ == '__main__':
    main()
