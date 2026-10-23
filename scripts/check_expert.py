"""Offline check of the scripted expert over the whole box randomization range, without ROS.

Runs PickPlaceExpert in plain MuJoCo the way collect_demos drives sim_node in sync_mode
(one plan row per 1/rate s, then settle_time of holding), on a grid of box poses: radius x
azimuth x yaw relative to the arm direction (the cube repeats every 90 degrees, so -45..45
covers all of it). For each pose it reports whether planning and the pick-and-place
succeed, whether the gripper touches the box before the jaws start closing, the gaps between
each jaw and the box at the bottom of the descent, and where the box sits in the gripper once
lifted.

--tool-error emulates a policy that is off along the jaw closing direction (+ = towards the
moving jaw): the plan is made for a box shifted by that much, so the tool misses its grasp
point by the same distance. This is the error that made ACT fail, see
docs/eval_act_200_failure_analysis.md.

    pixi run python scripts/check_expert.py                                  # default offset
    pixi run python scripts/check_expert.py --grasp-offset 0.0 0.009 --tool-error -0.005 0 0.005
"""

import argparse
import os
import sys

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco'))
from so101_mujoco.expert import (GRASP_OFFSET, PickPlaceExpert, PlanningError,  # noqa: E402
                                 box_yaw, grasp_yaw, is_success)

DEFAULT_MODEL = os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco', 'models',
                             'robotstudio_so101', 'scene_pick_place.xml')
RATE = 50.0
SETTLE_TIME = 2.0      # s, as collect_demos
PRE_CLOSE_TIME = 3.0   # s, approach (2.0) + descend (1.0) in PickPlaceExpert.plan
LIFT_END_TIME = 4.6    # s, ... + grasp (0.6) + lift (1.0)


class Checker:

    def __init__(self, model_path):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        m = self.m
        joint_ids = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joint_ids])
        self.box_id = m.body('box').id
        self.box_adr = m.jnt_qposadr[m.body_jntadr[self.box_id]]
        self.box_geom = m.geom('box').id
        self.site = m.site('gripperframe').id
        self.target = m.site('target').pos.copy()
        self.steps = max(1, round(1.0 / (RATE * m.opt.timestep)))
        gripper = m.body('gripper').id
        moving = m.body('moving_jaw_so101_v1').id
        collide = (m.geom_contype | m.geom_conaffinity) > 0
        self.fixed_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == gripper and collide[g]]
        self.moving_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == moving and collide[g]]
        self.hand_geoms = set(self.fixed_geoms + self.moving_geoms)

    def _reset(self, x, y, yaw):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.box_adr:self.box_adr + 2] = x, y
        d.qpos[self.box_adr + 3:self.box_adr + 7] = np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
        d.ctrl[:] = np.clip(d.qpos[self.qpos_adr], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(m, d)

    def _step(self, ctrl=None):
        if ctrl is not None:
            self.d.ctrl[:] = ctrl
        for _ in range(self.steps):
            mujoco.mj_step(self.m, self.d)

    def _touching(self):
        for c in self.d.contact[:self.d.ncon]:
            pair = {c.geom1, c.geom2}
            if self.box_geom in pair and pair & self.hand_geoms:
                return True
        return False

    def _gap(self, geoms):
        """Smallest distance (m) between the box and any of geoms; negative = penetration."""
        fromto = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.m, self.d, g, self.box_geom, 0.1, fromto)
                   for g in geoms)

    def _box_in_tool(self):
        """Box centre in the gripperframe, along the jaw closing (z) axis."""
        rot = self.d.site_xmat[self.site].reshape(3, 3)
        return float(rot[:, 2] @ (self.d.xpos[self.box_id] - self.d.site_xpos[self.site]))

    def run(self, expert, x, y, yaw, tool_error=0.0):
        self._reset(x, y, yaw)
        for _ in range(5):  # collect_demos waits for 5 states after the reset
            self._step()
        q_start = self.d.qpos[self.qpos_adr].copy()
        box_pos = self.d.xpos[self.box_id].copy()
        box_quat = self.d.xquat[self.box_id].copy()
        g = grasp_yaw(box_pos, box_yaw(box_quat))
        shifted = box_pos + tool_error * np.array([np.cos(g), np.sin(g), 0.0])
        try:
            plan = expert.plan(q_start, shifted, box_quat, self.target)
        except PlanningError as e:
            return {'planned': False, 'success': False, 'reason': str(e)}

        pre_close = int(round(PRE_CLOSE_TIME * RATE))
        lift_end = int(round(LIFT_END_TIME * RATE))
        touched, gaps, in_tool, lifted = False, None, None, False
        for i in range(len(plan) + int(round(SETTLE_TIME * RATE))):
            self._step(plan[min(i, len(plan) - 1)])
            if i <= pre_close:
                touched |= self._touching()
            if i == pre_close:
                gaps = (self._gap(self.fixed_geoms), self._gap(self.moving_geoms))
            if i == lift_end:
                in_tool = self._box_in_tool()
                lifted = self.d.xpos[self.box_id][2] > 0.03
        box_final = self.d.xpos[self.box_id].copy()
        return {'planned': True, 'success': is_success(box_final, self.target),
                'touched': touched, 'fixed_gap': gaps[0], 'moving_gap': gaps[1],
                'lifted': lifted, 'box_in_tool': in_tool,
                'place_error': float(np.linalg.norm(box_final[:2] - self.target[:2]))}


def mm(values):
    v = np.asarray(values) * 1e3
    return f'{v.mean():6.1f} [{v.min():6.1f}, {v.max():6.1f}]' if len(v) else '     -'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--grasp-offset', type=float, nargs='+', default=[GRASP_OFFSET],
                        help='m, expert grasp offsets to compare')
    parser.add_argument('--tool-error', type=float, nargs='+', default=[0.0],
                        help='m, emulated tool errors along the jaw axis (+ = towards the moving jaw)')
    parser.add_argument('--radius', type=float, nargs=2, default=[0.18, 0.25])
    parser.add_argument('--azimuth', type=float, nargs=2, default=[5.0, 40.0], help='deg')
    parser.add_argument('--grid', type=int, nargs=3, default=[5, 8, 9],
                        help='number of radii, azimuths and relative yaws')
    parser.add_argument('-v', '--verbose', action='store_true', help='print every failure')
    args = parser.parse_args()

    checker = Checker(args.model)
    radii = np.linspace(*args.radius, args.grid[0])
    azimuths = np.radians(np.linspace(*args.azimuth, args.grid[1]))
    # Relative yaw in [-45, 45) deg: 45 is the same cube as -45.
    rel_yaws = np.radians(np.linspace(-45.0, 45.0, args.grid[2], endpoint=False))
    poses = [(r, a, a + ry) for r in radii for a in azimuths for ry in rel_yaws]

    print(f'{len(poses)} box poses; gaps and box-in-tool in mm as mean [min, max]')
    print(f'{"offset":>7} {"error":>6} | {"plan":>7} {"success":>7} {"touch":>5} {"lifted":>6} | '
          f'{"fixed jaw gap":>22} {"moving jaw gap":>22} {"box in tool (z)":>22} {"place error":>22}')
    for offset in args.grasp_offset:
        expert = PickPlaceExpert(checker.m, rate=RATE, grasp_offset=offset)
        for error in args.tool_error:
            results = []
            for r, a, yaw in poses:
                res = checker.run(expert, r * np.cos(a), r * np.sin(a), yaw, error)
                results.append(res)
                if args.verbose and not res['success']:
                    print(f'  FAIL r={r:.3f} az={np.degrees(a):.1f} rel_yaw={np.degrees(yaw - a):.1f}: '
                          + (res.get('reason') or f'touched={res["touched"]} lifted={res["lifted"]} '
                             f'fixed_gap={res["fixed_gap"] * 1e3:.1f} mm'))
            ok = [x for x in results if x['planned']]
            n = len(results)
            print(f'{offset * 1e3:5.1f}mm {error * 1e3:4.1f}mm | '
                  f'{len(ok):3d}/{n:<3d} {sum(x["success"] for x in results):3d}/{n:<3d} '
                  f'{sum(x["touched"] for x in ok):5d} {sum(x["lifted"] for x in ok):6d} | '
                  f'{mm([x["fixed_gap"] for x in ok]):>22} {mm([x["moving_gap"] for x in ok]):>22} '
                  f'{mm([x["box_in_tool"] for x in ok if x["lifted"]]):>22} '
                  f'{mm([x["place_error"] for x in ok if x["success"]]):>22}')


if __name__ == '__main__':
    main()
