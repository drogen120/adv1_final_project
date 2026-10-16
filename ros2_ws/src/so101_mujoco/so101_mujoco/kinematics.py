"""Inverse kinematics for top-down grasps with the SO-101, using MuJoCo's Jacobians.

The tool frame is the "gripperframe" site: its x axis points out of the jaws (approach
direction) and its z axis is the direction in which the jaws open and close.
"""

import mujoco
import numpy as np

ARM_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']


def top_down_rotation(yaw):
    """Tool orientation pointing straight down, jaws closing along the horizontal direction yaw."""
    x = np.array([0.0, 0.0, -1.0])
    z = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


class TopDownIK:

    def __init__(self, model):
        # A private MjData so solving never touches the simulation state.
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site('gripperframe').id
        joint_ids = [model.joint(n).id for n in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.low = model.jnt_range[joint_ids, 0]
        self.high = model.jnt_range[joint_ids, 1]

    def forward(self, q):
        """Tool position and rotation matrix for the 5 arm joint angles q."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # needed by mj_jacSite
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def solve(self, pos, rot, q_init, iters=200, tol=1e-4, damping=1e-3, rot_weight=0.1):
        """Damped least squares towards (pos, rot). Returns (q, position error in m, angle error in rad)."""
        q = np.clip(np.asarray(q_init, dtype=float), self.low, self.high)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        quat_cur, quat_des, quat_err = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(quat_des, rot.flatten())
        for _ in range(iters):
            p, r = self.forward(q)
            mujoco.mju_mat2Quat(quat_cur, r.flatten())
            mujoco.mju_negQuat(quat_err, quat_cur)
            mujoco.mju_mulQuat(quat_err, quat_des, quat_err)
            rot_err = np.zeros(3)
            mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)
            err = np.concatenate([pos - p, rot_weight * rot_err])
            if np.linalg.norm(pos - p) < tol and np.linalg.norm(rot_err) < 10 * tol:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = np.vstack([jacp[:, self.dof_adr], rot_weight * jacr[:, self.dof_adr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            q = np.clip(q + dq, self.low, self.high)
        p, r = self.forward(q)
        angle = np.arccos(np.clip((np.trace(rot.T @ r) - 1) / 2, -1.0, 1.0))
        return q, float(np.linalg.norm(pos - p)), float(angle)
