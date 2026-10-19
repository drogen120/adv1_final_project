"""Scripted pick-and-place expert for the SO-101: plans joint targets for a whole episode.

PickPlaceExpert.plan is open loop: given the box pose at the start of the episode it returns
joint position targets sampled at a fixed rate, which are then streamed to /joint_command.
PickPlaceEpisode runs the same motions phase by phase instead, checks after each grasp that
the box was lifted and grasps again if it was not; it can also make the first grasp miss on
purpose (Injection), to record demonstrations of recovering from a failed grasp.
Cartesian segments are straight lines of the tool (gripperframe) pointing straight down,
solved frame by frame with TopDownIK.
"""

from typing import NamedTuple

import numpy as np

from .kinematics import TopDownIK, top_down_rotation

GRIPPER_OPEN = 1.0     # rad, about 9 cm between the jaw tips
GRIPPER_CLOSED = 0.0   # rad, past contact with the 4 cm cube, so the actuator squeezes

Z_HOVER = 0.06    # m, tool height while moving; top-down poses reach only ~0.065-0.09 m
                  #    for 0.14 m < r < 0.26 m (wrist_flex hits its limit above that)
Z_GRASP = 0.015   # m, tool (jaw tip) height when closing on the box
Z_PLACE = 0.02    # m, tool height when releasing the box (drops it ~5 mm)
GRASP_OFFSET = 0.009  # m, tool shift from the box centre towards the fixed jaw when grasping.
                      #    With the tool centred, the open fixed jaw clears the box face by
                      #    only ~0.6 mm, so a policy a few mm off lands that jaw on the box
                      #    top. Closing pushes the box against the fixed jaw, so once held it
                      #    is centred on the tool again and the place pose needs no shift.
LIFTED_Z = 0.03   # m, box centre height above which a grasp counts as lifted (rests at 0.02)
RETURN_SPEED = 0.12   # m/s, average tool speed when going back above the box to grasp again


def smoothstep(n):
    s = np.linspace(0.0, 1.0, n)
    return s * s * (3 - 2 * s)


def box_yaw(quat):
    """Yaw of a box from its (w, x, y, z) quaternion, assuming it stands upright."""
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def is_upright(quat, max_tilt_deg=10.0):
    """Whether the box's z axis is within max_tilt_deg of vertical."""
    w, x, y, z = quat
    return 1 - 2 * (x * x + y * y) > np.cos(np.radians(max_tilt_deg))


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


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PlanningError(RuntimeError):
    pass


class Grasp(NamedTuple):
    box_pos: np.ndarray   # box position the grasp was planned for
    xy: np.ndarray        # tool position (x, y) when closing
    yaw: float            # jaw closing direction


def _assemble(segments, include_first=True):
    """Rows (N x 6) of a list of (arm path, gripper start, gripper end) segments that share
    their boundary frames. Without include_first, the first frame (the pose the rows start
    from, already sent) is left out."""
    rows = []
    for i, (arm, g_from, g_to) in enumerate(segments):
        grip = g_from + smoothstep(len(arm)) * (g_to - g_from)
        seg = np.column_stack([arm, grip])
        rows.append(seg if i == 0 and include_first else seg[1:])
    return np.vstack(rows)


class PickPlaceExpert:

    def __init__(self, model, rate=50.0, max_ik_error=3e-3, grasp_offset=GRASP_OFFSET):
        self.ik = TopDownIK(model)
        self.dt = 1.0 / rate
        self.max_ik_error = max_ik_error
        self.grasp_offset = grasp_offset

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

    def _joint_move(self, q_from, q_to, seconds):
        s = smoothstep(self._frames(seconds))[:, None]
        return q_from + s * (q_to - q_from)

    def _hold(self, q, seconds):
        return np.repeat(q[None], self._frames(seconds), axis=0)

    def grasp(self, box_pos, box_quat, yaw_error=0.0, tool_error=0.0):
        """Grasp for a box pose. yaw_error turns the jaws away from the box faces and
        tool_error shifts the tool along the jaw closing direction (+ = towards the moving
        jaw); both are for making a grasp miss on purpose."""
        box_pos = np.asarray(box_pos, dtype=float)
        yaw = grasp_yaw(box_pos, box_yaw(box_quat)) + yaw_error
        # The tool z axis (jaw closing direction) points from the fixed to the moving jaw.
        xy = box_pos[:2] + (tool_error - self.grasp_offset) * np.array([np.cos(yaw), np.sin(yaw)])
        return Grasp(box_pos, xy, yaw)

    def _grasp_segments(self, q_above, grasp):
        """Descend from above the grasp point (arm at q_above), close the jaws, lift."""
        above = np.array([*grasp.xy, Z_HOVER])
        at = np.array([*grasp.xy, Z_GRASP])
        down = self._cartesian(q_above, above, at, grasp.yaw, 1.0)
        close = self._hold(down[-1], 0.6)
        up = self._cartesian(close[-1], at, above, grasp.yaw, 1.0)
        return [(down, GRIPPER_OPEN, GRIPPER_OPEN), (close, GRIPPER_OPEN, GRIPPER_CLOSED),
                (up, GRIPPER_CLOSED, GRIPPER_CLOSED)]

    def _place_yaw(self, grasp, target_pos):
        # Carry the box turning with the arm, so wrist_roll stays put on the way.
        return grasp.yaw + np.arctan2(target_pos[1], target_pos[0]) \
            - np.arctan2(grasp.box_pos[1], grasp.box_pos[0])

    def plan_pick(self, q_start, box_pos, box_quat, yaw_error=0.0, tool_error=0.0):
        """From the 6 joint positions q_start: move above the box, open, descend, close and
        lift. Returns (rows, grasp); the first row is q_start's arm pose."""
        q_start = np.asarray(q_start, dtype=float)
        grasp = self.grasp(box_pos, box_quat, yaw_error, tool_error)
        # Seed the first solve with the arm folded down towards the box (from the
        # Menagerie "pickup" keyframe); from the upright start pose IK can get stuck.
        seed = np.array([np.arctan2(box_pos[1], box_pos[0]), 0.0, 0.47, 1.18, 1.58])
        q_above = self._ik(np.array([*grasp.xy, Z_HOVER]), grasp.yaw, seed)
        segments = [(self._joint_move(q_start[:5], q_above, 2.0), q_start[5], GRIPPER_OPEN)]
        return _assemble(segments + self._grasp_segments(q_above, grasp)), grasp

    def plan_regrasp(self, q, box_pos, box_quat):
        """After a grasp that missed, from the last commanded joint targets q (6): go back
        above the box (wherever it is now), opening the jaws on the way, and grasp again.
        Returns (rows, grasp), without the starting frame."""
        q = np.asarray(q, dtype=float)
        pos, rot = self.ik.forward(q[:5])
        grasp = self.grasp(box_pos, box_quat)
        # Turn the jaws the short way round from where they point now.
        yaw_now = grasp.yaw + wrap_angle(np.arctan2(rot[1, 2], rot[0, 2]) - grasp.yaw)
        above = np.array([*grasp.xy, Z_HOVER])
        seconds = max(1.0, np.linalg.norm(above - pos) / RETURN_SPEED)
        back = self._cartesian(q[:5], pos, above, yaw_now, seconds, grasp.yaw)
        segments = [(back, q[5], GRIPPER_OPEN)] + self._grasp_segments(back[-1], grasp)
        return _assemble(segments, include_first=False), grasp

    def plan_carry(self, q, grasp, target_pos):
        """From above the grasp point (arm at q[:5]) to above the target, as when carrying
        the box; without the starting frame."""
        start = np.array([*grasp.xy, Z_HOVER])
        above_target = np.array([target_pos[0], target_pos[1], Z_HOVER])
        carry = self._cartesian(q[:5], start, above_target, grasp.yaw, 1.6,
                                self._place_yaw(grasp, target_pos))
        return _assemble([(carry, q[5], q[5])], include_first=False)

    def plan_place(self, q, grasp, target_pos, q_home):
        """With the box lifted by grasp (arm at q[:5]): carry it to the target, put it down,
        release it and go back to the 6 joint positions q_home. Without the starting frame."""
        target_pos = np.asarray(target_pos, dtype=float)
        place_yaw = self._place_yaw(grasp, target_pos)
        start = np.array([*grasp.xy, Z_HOVER])
        above_target = np.array([target_pos[0], target_pos[1], Z_HOVER])
        at_target = np.array([target_pos[0], target_pos[1], Z_PLACE])

        segments = []

        def add(arm, g_from, g_to=None):
            segments.append((arm, g_from, g_from if g_to is None else g_to))
            return arm[-1]

        q = add(self._cartesian(q[:5], start, above_target, grasp.yaw, 1.6, place_yaw), GRIPPER_CLOSED)
        q = add(self._cartesian(q, above_target, at_target, place_yaw, 0.8), GRIPPER_CLOSED)
        q = add(self._hold(q, 0.5), GRIPPER_CLOSED, GRIPPER_OPEN)
        q = add(self._cartesian(q, at_target, above_target, place_yaw, 0.6), GRIPPER_OPEN)
        add(self._joint_move(q, q_home[:5], 1.5), GRIPPER_OPEN, q_home[5])
        return _assemble(segments, include_first=False)

    def plan(self, q_start, box_pos, box_quat, target_pos):
        """Joint targets (N x 6: 5 arm joints + gripper), one row per 1/rate seconds.

        q_start: current 6 joint positions; the episode starts and ends there.
        """
        q_start = np.asarray(q_start, dtype=float)
        pick, grasp = self.plan_pick(q_start, box_pos, box_quat)
        return np.vstack([pick, self.plan_place(pick[-1], grasp, target_pos, q_start)])


class Injection(NamedTuple):
    """A deliberate error in the first grasp of an episode, so that it misses the box."""
    yaw_error: float = 0.0    # rad, jaws turned away from the box faces
    tool_error: float = 0.0   # m, tool shifted towards the moving jaw
    delay: float = 0.0        # s of carrying the empty gripper towards the target before
                              #   noticing the miss (the policy runs open loop for up to a
                              #   whole action chunk, so it may notice late too)


def sample_injection(rng, mode, yaw_error, wrist_tool_error, tool_error, delay):
    """Random Injection of the given mode, with each error uniform in its (low, high) range.

    'wrist' mimics ACT's failures at relative grasp angles near 45 degrees: jaws turned by
    yaw_error (deg, either way) and shifted by wrist_tool_error (m) towards the moving jaw,
    which lands the fixed jaw on the box top (turning alone does not miss: closing turns the
    box square to the jaws). 'offset' is a pure shift by tool_error (m), the earlier ACT
    failure. See docs/eval_act_offset.md.
    """
    if mode == 'wrist':
        injection = Injection(yaw_error=rng.choice([-1, 1]) * np.radians(rng.uniform(*yaw_error)),
                              tool_error=rng.uniform(*wrist_tool_error))
    elif mode == 'offset':
        injection = Injection(tool_error=rng.uniform(*tool_error))
    else:
        raise ValueError(f'unknown injection mode {mode!r}')
    return injection._replace(delay=rng.uniform(*delay))


class PickPlaceEpisode:
    """Closed-loop expert for one episode, planned one phase at a time.

    Call next_rows(box_pos, box_quat) with the latest box pose whenever all rows returned so
    far have been sent (the first time right after the reset); it returns the next joint
    targets, or None when the episode is over. After each grasp it checks that the box was
    lifted, and if not, grasps again from the box's new pose, up to max_retries times.
    PlanningError propagates. Afterwards:
        failure      None, or why the episode stopped early
        attempts     number of grasps
        retry_frame  index of the first row of the first regrasp, or None
    """

    def __init__(self, expert, q_start, target_pos, injection=None, max_retries=2):
        self.expert = expert
        self.q_start = np.asarray(q_start, dtype=float)
        self.target_pos = np.asarray(target_pos, dtype=float)
        self.injection = injection
        self.max_retries = max_retries
        self.phase = 'start'
        self.grasp = None
        self.last_row = self.q_start
        self.rows_sent = 0
        self.attempts = 0
        self.retry_frame = None
        self.failure = None

    def next_rows(self, box_pos, box_quat):
        rows = self._next(np.asarray(box_pos, dtype=float), np.asarray(box_quat, dtype=float))
        if rows is not None:
            self.last_row = rows[-1]
            self.rows_sent += len(rows)
        return rows

    def _stop(self, reason):
        self.failure = reason
        return None

    def _regrasp(self, box_pos, box_quat):
        if not is_upright(box_quat):
            return self._stop('box knocked over')
        rows, self.grasp = self.expert.plan_regrasp(self.last_row, box_pos, box_quat)
        if self.retry_frame is None:
            self.retry_frame = self.rows_sent
        self.attempts += 1
        self.phase = 'grasped'
        return rows

    def _next(self, box_pos, box_quat):
        if self.phase == 'start':
            injection = self.injection or Injection()
            rows, self.grasp = self.expert.plan_pick(self.q_start, box_pos, box_quat,
                                                     injection.yaw_error, injection.tool_error)
            self.attempts = 1
            self.phase = 'grasped'
            return rows
        if self.phase == 'grasped':
            if box_pos[2] > LIFTED_Z:
                if self.attempts == 1 and self.injection is not None:
                    return self._stop('injected grasp lifted the box')
                self.phase = 'placing'
                return self.expert.plan_place(self.last_row, self.grasp, self.target_pos,
                                              self.q_start)
            if self.attempts > self.max_retries:
                return self._stop(f'box not lifted after {self.attempts} grasps')
            n = int(round(self.injection.delay / self.expert.dt)) if self.injection else 0
            if self.attempts == 1 and n > 0:
                self.phase = 'missed'
                return self.expert.plan_carry(self.last_row, self.grasp, self.target_pos)[:n]
            return self._regrasp(box_pos, box_quat)
        if self.phase == 'missed':
            return self._regrasp(box_pos, box_quat)
        return None  # placed


def is_success(box_pos, target_pos, tolerance=0.03):
    """Box resting upright on the floor with its centre within tolerance of the target (xy)."""
    box_pos = np.asarray(box_pos)
    return bool(np.linalg.norm(box_pos[:2] - np.asarray(target_pos)[:2]) < tolerance
                and box_pos[2] < 0.03)
