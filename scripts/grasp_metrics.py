"""Grasp metrics of closed-loop evaluations (or demonstrations), from MuJoCo FK and contacts.

Every failure so far has been a missed grasp, decided during the descent onto the box. This
replays the recorded joint angles and box poses (kinematics only, no physics) and measures,
per episode:

- g: the box's relative grasp angle, i.e. the expert's grasp yaw minus the direction
  perpendicular to the arm, in [-45, 45) deg. The expert switches to the other pair of box
  faces at +-45 deg, so its grasp yaw jumps by 90 deg there;
- jaw: the offset of gripperframe from the box centre along the jaw closing axis (the tool
  z axis, from the fixed to the moving jaw), in cm. Positive moves the fixed jaw onto the box;
- yaw err: the angle between the closing axis and the nearest box face normal, in deg;
- the first body to touch the box during the descent and grasp, and the tool height when
  the expert would close the jaws.

    pixi run python scripts/grasp_metrics.py data/eval/act_200_40k data/eval/act_200_60k
    pixi run python scripts/grasp_metrics.py data/demos_offset.npz      # the expert, for reference
    pixi run python scripts/grasp_metrics.py data/eval/act_200_60k --csv data/eval/act_200_60k/grasp.csv

An argument is an output directory of policy_eval (rollouts.npz + episodes.jsonl) or an npz in
the demos.npz format (every episode then counts as a success). With several evaluations, the
later ones are also compared episode by episode with the first, if their box placements match.
"""

import argparse
import csv
import json
import math
import os

import mujoco
import numpy as np

from visualize_demos import DEFAULT_MODEL, box_yaw

# Frames at 50 fps of PickPlaceExpert.plan: 2 s approach, then a 1 s descent from z = 6 cm.
FRAME_DESCENT = 105    # just after the descent starts, well before any contact
FRAME_CONTACT = 120    # tool at z ~ 5 cm, just above the box top (4 cm)
FRAME_CLOSE = 150      # the expert's tool is down at Z_GRASP and starts closing
CONTACT_FRAMES = range(100, 200)
G_BINS = [-45, -35, -15, 15, 35, 45]


def rel_grasp_deg(env_state):
    """Relative grasp angle g (deg) of expert.grasp_yaw for [x, y, z, cos 4yaw, sin 4yaw]."""
    natural = math.atan2(env_state[1], env_state[0]) + math.pi / 2
    yaw = box_yaw(env_state)
    k = round((natural - yaw) / (math.pi / 2))
    return math.degrees(yaw + k * math.pi / 2 - natural)


class Kinematics:

    def __init__(self, joint_names, model_path=DEFAULT_MODEL):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_adr = [self.model.joint(n).qposadr[0] for n in joint_names]
        self.site = self.model.site('gripperframe').id
        box = self.model.body('box')
        self.box_adr = self.model.jnt_qposadr[box.jntadr[0]]
        self.box_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == box.id}
        self.floor_body = 0   # world: the box resting on the floor is not a grasp contact

    def set(self, state, env_state):
        q = self.data.qpos
        q[self.joint_adr] = state
        yaw = box_yaw(env_state)
        q[self.box_adr:self.box_adr + 3] = env_state[:3]
        q[self.box_adr + 3:self.box_adr + 7] = math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
        mujoco.mj_forward(self.model, self.data)   # contacts need the full forward pass

    def tool(self, state, env_state):
        """(jaw offset in cm, yaw error in deg, tool height in cm)."""
        self.set(state, env_state)
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        pos = self.data.site_xpos[self.site]
        jaw = (pos - env_state[:3]) @ rot[:, 2]
        jaw_yaw = math.degrees(math.atan2(rot[1, 2], rot[0, 2]))
        yaw_err = (jaw_yaw - math.degrees(box_yaw(env_state)) + 45) % 90 - 45
        return 100 * jaw, yaw_err, 100 * pos[2]

    def first_contact(self, states, env_states):
        """(frame, body name) of the first robot body touching the box, or None."""
        for f in CONTACT_FRAMES:
            if f >= len(states):
                break
            self.set(states[f], env_states[f])
            for c in self.data.contact[:self.data.ncon]:
                if c.geom1 in self.box_geoms or c.geom2 in self.box_geoms:
                    other = c.geom2 if c.geom1 in self.box_geoms else c.geom1
                    body = self.model.geom_bodyid[other]
                    if body != self.floor_body:
                        return f, self.model.body(body).name
        return None


def load(path):
    """(label, rollouts npz, {episode: episodes.jsonl row or None})."""
    if os.path.isdir(path):
        data = np.load(os.path.join(path, 'rollouts.npz'))
        with open(os.path.join(path, 'episodes.jsonl')) as f:
            results = {r['episode']: r for r in map(json.loads, f)}
        return os.path.basename(os.path.normpath(path)), data, results
    data = np.load(path)
    return os.path.basename(path), data, {int(e): None for e in np.unique(data['episode_index'])}


def episode_metrics(data, results):
    kin = Kinematics([str(n) for n in data['joint_names']])
    episodes = data['episode_index']
    rows = []
    for e in np.unique(episodes):
        idx = np.flatnonzero(episodes == e)
        state = data['observation_state'][idx]
        env = data['observation_environment_state'][idx]
        jaw_descent, _, _ = kin.tool(state[FRAME_DESCENT], env[FRAME_DESCENT])
        jaw_contact, yaw_err, _ = kin.tool(state[FRAME_CONTACT], env[FRAME_CONTACT])
        _, _, z_close = kin.tool(state[FRAME_CLOSE], env[FRAME_CLOSE])
        first = kin.first_contact(state, env)
        result = results.get(int(e))
        rows.append(dict(
            episode=int(e),
            success=True if result is None else result['success'],
            target_dist_cm=None if result is None else 100 * result['target_dist'],
            g=rel_grasp_deg(env[0]),
            jaw_descent=jaw_descent, jaw_contact=jaw_contact, yaw_err=yaw_err,
            z_close=z_close, wrist_roll=float(state[FRAME_CONTACT][4]),
            contact_frame=first[0] if first else None,
            contact_body=first[1] if first else None,
            box_init=env[0][:2].tolist()))
    return rows


def binom_two_sided(k, n):
    """Exact two-sided p-value of k successes out of n under p = 0.5 (sign / McNemar test)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def report(label, rows):
    ok = np.array([r['success'] for r in rows])
    jaw = np.array([r['jaw_descent'] for r in rows])
    print(f'== {label}: {ok.sum()}/{len(rows)} succeeded')
    print(f'   jaw offset at frame {FRAME_DESCENT}: {jaw.mean():+.2f} +- {jaw.std():.2f} cm'
          f' (successes {jaw[ok].min():+.2f} .. {jaw[ok].max():+.2f})')
    dist = [r['target_dist_cm'] for r in rows if r['success'] and r['target_dist_cm'] is not None]
    if dist:
        print(f'   placement error of successes: mean {np.mean(dist):.2f}, median {np.median(dist):.2f},'
              f' max {np.max(dist):.2f} cm')
    for outcome, sel in (('successes', ok), ('failures', ~ok)):
        bodies = {}
        for r, s in zip(rows, sel):
            if s:
                bodies[r['contact_body']] = bodies.get(r['contact_body'], 0) + 1
        if bodies:
            print(f'   first body touching the box, {outcome}: {bodies}')

    print(f'   {"g (deg)":>12} {"n":>3} {"succ":>4}  {"jaw@" + str(FRAME_DESCENT):>13}'
          f'  {"jaw@" + str(FRAME_CONTACT):>13}  {"yaw err@" + str(FRAME_CONTACT):>13}')
    g = np.array([r['g'] for r in rows])
    jaw_c = np.array([r['jaw_contact'] for r in rows])
    yaw = np.array([r['yaw_err'] for r in rows])
    for lo, hi in zip(G_BINS[:-1], G_BINS[1:]):
        sel = (g >= lo) & (g < hi)
        if sel.any():
            print(f'   [{lo:+4d}, {hi:+4d}) {sel.sum():3d} {ok[sel].sum():4d}'
                  f'  {jaw[sel].mean():+6.2f}+-{jaw[sel].std():4.2f}'
                  f'  {jaw_c[sel].mean():+6.2f}+-{jaw_c[sel].std():4.2f}'
                  f'  {yaw[sel].mean():+6.1f}+-{yaw[sel].std():4.1f}')

    # Episode list for evaluations only: the demos all succeed and fill the screen.
    listed = [r for r in rows if not r['success'] or abs(r['g']) >= 35]
    if listed and rows[0]['target_dist_cm'] is not None:
        print('   failures and |g| >= 35 deg:')
        for r in listed:
            x, y = r['box_init']
            contact = f'{r["contact_body"]}@{r["contact_frame"]}' if r['contact_body'] else '-'
            print(f'     ep {r["episode"]:3d} {"ok  " if r["success"] else "FAIL"}'
                  f' r={math.hypot(x, y):.3f} az={math.degrees(math.atan2(y, x)):5.1f}'
                  f' g={r["g"]:+5.1f} jaw={r["jaw_contact"]:+.2f} yaw_err={r["yaw_err"]:+5.1f}'
                  f' wrist_roll={r["wrist_roll"]:.2f} z@{FRAME_CLOSE}={r["z_close"]:.1f} first={contact}')
    print()


def compare(label_a, rows_a, label_b, rows_b):
    a = {r['episode']: r for r in rows_a}
    b = {r['episode']: r for r in rows_b}
    common = sorted(set(a) & set(b))
    same = [e for e in common if np.allclose(a[e]['box_init'], b[e]['box_init'], atol=1e-3)]
    if len(same) < len(common):
        print(f'== {label_b} vs {label_a}: box placements differ ({len(same)}/{len(common)} match),'
              ' no paired comparison\n')
        return
    only_a = [e for e in common if a[e]['success'] and not b[e]['success']]
    only_b = [e for e in common if b[e]['success'] and not a[e]['success']]
    print(f'== {label_b} vs {label_a}, paired over {len(common)} episodes')
    print(f'   only {label_a} succeeded: {only_a}; only {label_b} succeeded: {only_b};'
          f' exact McNemar p = {binom_two_sided(len(only_a), len(only_a) + len(only_b)):.3g}')
    both = [e for e in common if a[e]['success'] and b[e]['success']
            and a[e]['target_dist_cm'] is not None and b[e]['target_dist_cm'] is not None]
    if both:
        worse = sum(b[e]['target_dist_cm'] > a[e]['target_dist_cm'] for e in both)
        print(f'   placement error where both succeeded: {np.mean([a[e]["target_dist_cm"] for e in both]):.2f}'
              f' vs {np.mean([b[e]["target_dist_cm"] for e in both]):.2f} cm;'
              f' {label_b} worse in {worse}/{len(both)} (sign test p = {binom_two_sided(worse, len(both)):.3g})')
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='policy_eval output directories or demos-format npz files')
    parser.add_argument('--csv', help='write the per-episode metrics of the (single) input to this file')
    args = parser.parse_args()

    evaluated = []
    for path in args.paths:
        label, data, results = load(path)
        rows = episode_metrics(data, results)
        report(label, rows)
        evaluated.append((label, rows))
    for label, rows in evaluated[1:]:
        compare(*evaluated[0], label, rows)

    if args.csv:
        rows = evaluated[0][1]
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
