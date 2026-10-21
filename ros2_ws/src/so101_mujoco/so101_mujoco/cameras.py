"""Camera images for the image-based policy, rendered the same way everywhere.

An image is defined by the published state: the joint positions of /joint_states and the box
pose of /box_pose. Training data renders it offline from the recorded bags, sim_node renders
it live from the values it is about to publish; both go through CameraRenderer.render_state,
so the two give identical pixels for identical states.

(sim_node publishes data.qpos for the joints but data.xpos/xquat for the box. After mj_step
the latter lag the former by one physics step, so this is not exactly the physical state;
it is within a pixel and the same in training and evaluation.)

Only depends on mujoco and numpy, so it can also be imported outside ROS.
"""

import mujoco
import numpy as np

CAMERAS = ('overview', 'wrist_cam')
HEIGHT, WIDTH = 240, 320
JOINT_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


class CameraRenderer:
    """Renders the cameras for a given (joints, box pose) on its own MjData.

    Uses an OpenGL context, so create and use it on one thread (sim_node: the sim thread).
    """

    def __init__(self, model, cameras=CAMERAS, height=HEIGHT, width=WIDTH,
                 joint_names=JOINT_NAMES):
        self.model = model
        self.cameras = tuple(cameras)
        for name in self.cameras:
            model.camera(name)  # raises KeyError for an unknown camera
        g = model.vis.global_
        g.offwidth, g.offheight = max(width, g.offwidth), max(height, g.offheight)
        self.data = mujoco.MjData(model)
        self.joint_adr = np.array([model.joint(n).qposadr[0] for n in joint_names])
        box = model.body('box')
        self.box_adr = int(model.jnt_qposadr[box.jntadr[0]])
        self.renderer = mujoco.Renderer(model, height, width)
        self.option = mujoco.MjvOption()  # fixed defaults, independent of any viewer

    def render_state(self, joints, box_pos, box_quat):
        """{camera: (H, W, 3) uint8} for joint positions (joint_names order) and the box
        position / quaternion (w, x, y, z)."""
        q = self.data.qpos
        q[self.joint_adr] = joints
        q[self.box_adr:self.box_adr + 3] = box_pos
        q[self.box_adr + 3:self.box_adr + 7] = box_quat
        # mj_forward, not mj_kinematics: camera poses are computed in mj_camlight.
        mujoco.mj_forward(self.model, self.data)
        images = {}
        for name in self.cameras:
            self.renderer.update_scene(self.data, camera=name, scene_option=self.option)
            images[name] = self.renderer.render().copy()
        return images

    def close(self):
        self.renderer.close()
