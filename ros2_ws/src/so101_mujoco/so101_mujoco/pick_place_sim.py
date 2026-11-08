"""The pick-and-place simulation without ROS: physics core (PickPlaceSim) and episode env
(PickPlaceEnv).

PickPlaceSim is what sim_node steps: reset (with the random box placement), joint targets,
one frame of physics. sim_node adds the ROS interface around it; PickPlaceEnv adds the
episode loop of policy_eval and the cameras, so a policy can be run in closed loop from plain
Python (scripts/sim_eval.py, and the online RL experiments) without the ROS round trips.

The two run the same physics on the same random boxes: for a given seed, PickPlaceEnv's
episode k gets the box of the k-th /reset of a sim_node started with that seed, and a
policy sees the same observations it sees through policy_eval (see PickPlaceEnv).

Only depends on mujoco and numpy, so it can be imported from the LeRobot venv.
"""

import os

import mujoco
import numpy as np

from .cameras import HEIGHT, JOINT_NAMES, WIDTH, CameraRenderer
from .expert import box_features, is_success

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'models', 'robotstudio_so101', 'scene_pick_place.xml')


class PickPlaceSim:
    """MuJoCo model and data of the scene, the box placement on reset, and the frame step.

    With randomize_box, every reset puts the box at a random pose, see sim_node.py for the
    ranges. The constructor resets once, like sim_node does at start-up, so the k-th reset()
    after it gets the k-th box of a sim_node with the same seed.
    """

    def __init__(self, model_path=DEFAULT_MODEL, keyframe='', frame_rate=50.0,
                 randomize_box=True, box_radius_range=(0.18, 0.25),
                 box_azimuth_range=(5.0, 40.0), box_rel_yaw_range=(-45.0, 45.0), seed=0):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.key_id = -1
        if keyframe:
            self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if self.key_id < 0:
                raise ValueError(f'Keyframe "{keyframe}" not found in model')
        self.steps_per_frame = max(1, round(1.0 / (frame_rate * self.model.opt.timestep)))

        # The controllable joints are the ones driven by actuators, in actuator order.
        m = self.model
        joint_ids = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
        self.joint_names = [m.joint(j).name for j in joint_ids]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in joint_ids])
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        self.box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'box')
        self.box_qpos_adr = None
        if self.box_id >= 0:
            self.box_qpos_adr = m.jnt_qposadr[m.body_jntadr[self.box_id]]
            self.box_geom_id = int(m.body_geomadr[self.box_id])
            self.box_rgba = m.geom_rgba[self.box_geom_id].copy()

        self.randomize_box = randomize_box
        self.box_radius_range = tuple(box_radius_range)
        self.box_azimuth_range = tuple(box_azimuth_range)
        self.box_rel_yaw_range = tuple(box_rel_yaw_range)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, box_pose=None):
        """Reset physics and hold the initial pose; put the box at box_pose (x, y, yaw) if
        given, else at a random pose (randomize_box) or the model's."""
        if self.key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if box_pose is not None:
            self.place_box(*box_pose)
        elif self.box_qpos_adr is not None and self.randomize_box:
            self.randomize_box_pose()
        if self.box_qpos_adr is not None:
            # Shown in sim_node's viewer; the cameras render their own copy of the model.
            x, y = self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 2]
            in_range = self.in_workspace(x, y)
            self.model.geom_rgba[self.box_geom_id] = self.box_rgba if in_range else (1, 0.8, 0, 1)
        self.data.ctrl[:] = np.clip(self.data.qpos[self.qpos_adr], self.ctrl_low, self.ctrl_high)
        mujoco.mj_forward(self.model, self.data)

    def in_workspace(self, x, y, eps=1e-6):
        """Whether (x, y) is in the box sampling sector (box_radius_range x box_azimuth_range)."""
        r_low, r_high = self.box_radius_range
        az_low, az_high = self.box_azimuth_range
        r, az = np.hypot(x, y), np.degrees(np.arctan2(y, x))
        return bool(r_low - eps <= r <= r_high + eps and az_low - eps <= az <= az_high + eps)

    def randomize_box_pose(self):
        """Place the (free-floating) box upright at a random pose; height is kept from the model."""
        r = self.rng.uniform(*self.box_radius_range)
        azimuth = np.radians(self.rng.uniform(*self.box_azimuth_range))
        if list(self.box_rel_yaw_range) == [-45.0, 45.0]:
            # Same draws as before box_rel_yaw_range existed, so old seeds give the same boxes.
            yaw = self.rng.uniform(-np.pi, np.pi)
        else:
            yaw = azimuth + np.pi / 2 + np.radians(self.rng.uniform(*self.box_rel_yaw_range))
        self.place_box(r * np.cos(azimuth), r * np.sin(azimuth), yaw)

    def place_box(self, x, y, yaw):
        """Put the box upright at (x, y) with the given yaw; height is kept from the model."""
        adr = self.box_qpos_adr
        self.data.qpos[adr:adr + 2] = x, y
        self.data.qpos[adr + 3:adr + 7] = np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)

    def clip_targets(self, targets, idx=slice(None)):
        """Joint targets (actuator order, or the joints idx) clipped to the actuator ranges."""
        return np.clip(np.asarray(targets, dtype=float), self.ctrl_low[idx], self.ctrl_high[idx])

    def step_frame(self):
        """Advance one frame (1 / frame_rate s) with the current data.ctrl."""
        for _ in range(self.steps_per_frame):
            mujoco.mj_step(self.model, self.data)

    def joint_positions(self):
        """Joint positions in actuator order, as published on /joint_states."""
        return self.data.qpos[self.qpos_adr].copy()

    def joint_velocities(self):
        return self.data.qvel[self.dof_adr].copy()

    def box_pose(self):
        """Box position and quaternion (w, x, y, z), as published on /box_pose. (data.xpos /
        xquat: after mj_step they lag qpos by one physics step, see cameras.py.)"""
        return self.data.xpos[self.box_id].copy(), self.data.xquat[self.box_id].copy()


class PickPlaceEnv:
    """Closed-loop episodes of a policy on PickPlaceSim, timed like policy_eval.

    reset() resets the sim and lets start_frame frames pass with the arm holding its pose,
    then returns the first observation. (policy_eval waits for 5 /joint_states after /reset
    and answers the next one; through ROS that turned out to be frame 7, sometimes 6: of the
    200 episodes of three ROS evaluations, the trajectories of 162 equal PickPlaceEnv's with
    start_frame=7 exactly, 25 with start_frame=6; 7 of the other 13 went apart where sim_node
    logged a sync_mode timeout, the run of the other 6 kept no logs.) step(action) applies the joint targets (JOINT_NAMES order, rad;
    clipped like sim_node does), advances one frame and returns the next observation. An
    observation is the published state of that frame:

        {'state': joint positions (JOINT_NAMES order), 'velocity': joint velocities,
         'box_pos', 'box_quat': box pose, 'environment_state': expert.box_features,
         'time': sim time (s)}

    render() gives the camera images of the current frame, {camera: (H, W, 3) uint8},
    rendered by cameras.CameraRenderer on its own copy of the model, as sim_node does. Images
    are rendered only when asked for, since they cost ~18 ms a camera and a chunked policy
    only looks at every n-th frame. judge() lets the box settle for settle_frames frames on
    the last targets (policy_eval waits for 2 more states) and says whether it ended on the
    target.
    """

    def __init__(self, cameras=(), seed=0, box_rel_yaw_range=(-45.0, 45.0), start_frame=7,
                 settle_frames=2, camera_height=HEIGHT, camera_width=WIDTH, **sim_kwargs):
        self.sim = PickPlaceSim(seed=seed, box_rel_yaw_range=box_rel_yaw_range, **sim_kwargs)
        self.start_frame = start_frame
        self.settle_frames = settle_frames
        self.target_pos = self.sim.model.site('target').pos.copy()
        # Observations and actions in JOINT_NAMES order; the sim works in actuator order.
        index = {n: i for i, n in enumerate(self.sim.joint_names)}
        self.order = np.array([index[n] for n in JOINT_NAMES])
        self.cameras = tuple(cameras)
        self.renderer = None
        if self.cameras:
            model = mujoco.MjModel.from_xml_path(self.sim.model_path)
            self.renderer = CameraRenderer(model, self.cameras, camera_height, camera_width,
                                           joint_names=self.sim.joint_names)

    def reset(self, box_pose=None):
        self.sim.reset(box_pose)
        for _ in range(self.start_frame):
            self.sim.step_frame()
        return self.observe()

    def step(self, action):
        targets = np.empty(len(self.order))
        targets[self.order] = action
        self.sim.data.ctrl[:] = self.sim.clip_targets(targets)
        self.sim.step_frame()
        return self.observe()

    def observe(self):
        box_pos, box_quat = self.sim.box_pose()
        return {
            'state': self.sim.joint_positions()[self.order],
            'velocity': self.sim.joint_velocities()[self.order],
            'box_pos': box_pos,
            'box_quat': box_quat,
            'environment_state': box_features(box_pos, box_quat),
            'time': self.sim.data.time,
        }

    def render(self):
        box_pos, box_quat = self.sim.box_pose()
        return self.renderer.render_state(self.sim.joint_positions(), box_pos, box_quat)

    def judge(self):
        """Let the box settle, then (success, final box position)."""
        for _ in range(self.settle_frames):
            self.sim.step_frame()
        box_pos, _ = self.sim.box_pose()
        return is_success(box_pos, self.target_pos), box_pos

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
