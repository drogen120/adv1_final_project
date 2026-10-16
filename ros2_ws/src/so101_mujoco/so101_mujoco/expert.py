"""Scripted pick-and-place expert for the SO-101: plans a whole episode of joint targets.

The plan is open loop: given the box pose at the start of the episode it returns joint
position targets sampled at a fixed rate, which are then streamed to /joint_command.
Cartesian segments are straight lines of the tool (gripperframe) pointing straight down,
solved frame by frame with TopDownIK.
"""

import numpy as np

from .kinematics import TopDownIK, top_down_rotation

GRIPPER_OPEN = 1.0     # rad, about 9 cm between the jaw tips
GRIPPER_CLOSED = 0.0   # rad, past contact with the 4 cm cube, so the actuator squeezes

Z_HOVER = 0.06    # m, tool height while moving; top-down poses reach only ~0.065-0.09 m
                  #    for 0.14 m < r < 0.26 m (wrist_flex hits its limit above that)
Z_GRASP = 0.015   # m, tool (jaw tip) height when closing on the box
Z_PLACE = 0.02    # m, tool height when releasing the box (drops it ~5 mm)


def smoothstep(n):
    s = np.linspace(0.0, 1.0, n)
    return s * s * (3 - 2 * s)


def box_yaw(quat):
    """Yaw of a box from its (w, x, y, z) quaternion, assuming it stands upright."""
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def box_features(pos, quat):
    """Box observation for the policy: [x, y, z, cos 4yaw, sin 4yaw].

    The cube looks the same every 90 degrees and the expert's grasp only depends on yaw
    modulo 90 degrees, so 4*yaw is the smooth way to encode it.
    """
    yaw = box_yaw(quat)
    return np.array([pos[0], pos[1], pos[2], np.cos(4 * yaw), np.sin(4 * yaw)])


def grasp_yaw(box_pos, yaw):
    """Direction for the jaws to close: across a pair of box faces, as close as possible to
    the direction perpendicular to the arm (the wrist_roll ~1.58 rad pose)."""
    natural = np.arctan2(box_pos[1], box_pos[0]) + np.pi / 2
    # The box is square, so face normals repeat every 90 degrees.
    k = np.round((natural - yaw) / (np.pi / 2))
    return yaw + k * np.pi / 2


class PlanningError(RuntimeError):
    pass


class PickPlaceExpert:

    def __init__(self, model, rate=50.0, max_ik_error=3e-3):
        self.ik = TopDownIK(model)
        self.dt = 1.0 / rate
        self.max_ik_error = max_ik_error

    def _frames(self, seconds):
        return max(2, int(round(seconds / self.dt)) + 1)

    def _cartesian(self, q, start, end, yaw, seconds, end_yaw=None):
        """Joint path for a straight tool line from start to end (turning from yaw to
        end_yaw on the way), warm-starting from q."""
        end_yaw = yaw if end_yaw is None else end_yaw
        path = []
        for s in smoothstep(self._frames(seconds)):
            rot = top_down_rotation(yaw + s * (end_yaw - yaw))
            q, pos_err, ang_err = self.ik.solve(start + s * (end - start), rot, q)
            if pos_err > self.max_ik_error or ang_err > 0.05:
                raise PlanningError(f'IK failed near {np.round(start + s * (end - start), 3)}: '
                                    f'{pos_err * 1e3:.1f} mm, {np.degrees(ang_err):.1f} deg')
            path.append(q)
        return np.array(path)

    def _ik(self, pos, yaw, q_init):
        q, pos_err, ang_err = self.ik.solve(pos, top_down_rotation(yaw), q_init, iters=500)
        if pos_err > self.max_ik_error or ang_err > 0.05:
            raise PlanningError(f'IK failed at {np.round(pos, 3)}: '
                                f'{pos_err * 1e3:.1f} mm, {np.degrees(ang_err):.1f} deg')
        return q

    def plan(self, q_start, box_pos, box_quat, target_pos):
        """Joint targets (N x 6: 5 arm joints + gripper), one row per 1/rate seconds.

        q_start: current 6 joint positions; the episode starts and ends there.
        """
        q_start = np.asarray(q_start, dtype=float)
        box_pos = np.asarray(box_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        yaw = grasp_yaw(box_pos, box_yaw(box_quat))
        # Carry the box turning with the arm, so wrist_roll stays put on the way.
        place_yaw = yaw + np.arctan2(target_pos[1], target_pos[0]) - np.arctan2(box_pos[1], box_pos[0])

        above_box = np.array([box_pos[0], box_pos[1], Z_HOVER])
        at_box = np.array([box_pos[0], box_pos[1], Z_GRASP])
        above_target = np.array([target_pos[0], target_pos[1], Z_HOVER])
        at_target = np.array([target_pos[0], target_pos[1], Z_PLACE])

        # Seed the first solve with the arm folded down towards the box (from the
        # Menagerie "pickup" keyframe); from the upright start pose IK can get stuck.
        seed = np.array([np.arctan2(box_pos[1], box_pos[0]), 0.0, 0.47, 1.18, 1.58])
        q_above_box = self._ik(above_box, yaw, seed)

        segments = []  # (arm path, gripper start, gripper end)

        def joint_move(q_from, q_to, seconds):
            s = smoothstep(self._frames(seconds))[:, None]
            return q_from + s * (q_to - q_from)

        def add(arm, g_from, g_to=None):
            g_to = g_from if g_to is None else g_to
            segments.append((arm, g_from, g_to))
            return arm[-1]

        def hold(q, seconds):
            return np.repeat(q[None], self._frames(seconds), axis=0)

        q = add(joint_move(q_start[:5], q_above_box, 2.0), q_start[5], GRIPPER_OPEN)
        q = add(self._cartesian(q, above_box, at_box, yaw, 1.0), GRIPPER_OPEN)
        q = add(hold(q, 0.6), GRIPPER_OPEN, GRIPPER_CLOSED)
        q = add(self._cartesian(q, at_box, above_box, yaw, 1.0), GRIPPER_CLOSED)
        q = add(self._cartesian(q, above_box, above_target, yaw, 1.6, place_yaw), GRIPPER_CLOSED)
        q = add(self._cartesian(q, above_target, at_target, place_yaw, 0.8), GRIPPER_CLOSED)
        q = add(hold(q, 0.5), GRIPPER_CLOSED, GRIPPER_OPEN)
        q = add(self._cartesian(q, at_target, above_target, place_yaw, 0.6), GRIPPER_OPEN)
        add(joint_move(q, q_start[:5], 1.5), GRIPPER_OPEN, q_start[5])

        rows = []
        for i, (arm, g_from, g_to) in enumerate(segments):
            grip = g_from + smoothstep(len(arm)) * (g_to - g_from)
            seg = np.column_stack([arm, grip])
            rows.append(seg if i == 0 else seg[1:])  # segments share their boundary frame
        return np.vstack(rows)


def is_success(box_pos, target_pos, tolerance=0.03):
    """Box resting upright on the floor with its centre within tolerance of the target (xy)."""
    box_pos = np.asarray(box_pos)
    return bool(np.linalg.norm(box_pos[:2] - np.asarray(target_pos)[:2]) < tolerance
                and box_pos[2] < 0.03)
