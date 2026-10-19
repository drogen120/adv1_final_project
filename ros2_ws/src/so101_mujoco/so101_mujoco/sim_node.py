"""MuJoCo simulation of the SO-101 arm, exposed as a ROS 2 node.

Topics
  publishes   /joint_states   sensor_msgs/JointState      arm joint position/velocity/effort
              /box_pose       geometry_msgs/PoseStamped   pose of the "box" body, if the scene has one
              /clock          rosgraph_msgs/Clock         simulation time (if publish_clock is true)
  subscribes  /joint_command  sensor_msgs/JointState      position targets in rad; names optional;
                                                          no positions = keep the current targets
              /place_box      geometry_msgs/Pose2D        reset the scene with the box at (x, y) m,
                                                          yaw theta rad, instead of a random pose
Services
              /reset          std_srvs/Trigger            reset the scene to its initial state

With randomize_box (default), every reset puts the box at a random position and yaw on the
floor, sampled in polar coordinates around the arm base from box_radius_range (m) and
box_azimuth_range (deg). The default ranges are the part of the top-down grasp workspace
where the scripted expert succeeds (see expert.py). box_rel_yaw_range (deg) narrows the yaw
to a range of the relative grasp angle g, the angle between the expert's jaw closing
direction and the direction perpendicular to the arm (see expert.grasp_yaw and
scripts/grasp_metrics.py). The cube repeats every 90 degrees, so [-45, 45] (the default,
which leaves the yaw sampling as it was) is every pose, and [35, 55] is |g| >= 35, the
boundary where the expert switches between two pairs of faces.

/place_box resets like /reset but puts the box where it is told (used by policy_demo). In
the viewer, the sector of box_radius_range x box_azimuth_range is drawn on the floor
(show_workspace), and a box placed outside it turns yellow.

realtime_factor can be changed while the sim runs (ros2 param set), e.g. to slow a demo down.

With sync_mode, after publishing the state stamped t the sim waits (up to sync_timeout s of
wall time) for a /joint_command stamped t before stepping on, so a controller that answers
every /joint_states message is applied with no latency, however slow it is. Controllers
with nothing new to say answer with an empty command. Used for collecting demonstrations
and for evaluating policies.

Run headless:     ros2 run so101_mujoco sim_node
Run with viewer:  ros2 run --prefix mjpython so101_mujoco sim_node --ros-args -p use_viewer:=true
(On macOS the passive viewer only works under mjpython.)
"""

import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose2D, PoseStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


def to_time_msg(seconds):
    # Round to whole nanoseconds first: splitting off int(seconds) and rounding the fraction
    # turns e.g. 3.9999999999 into sec=3, nanosec=0, a 1 s jump backwards.
    sec, nanosec = divmod(int(round(seconds * 1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nanosec)


class MujocoSim(Node):

    def __init__(self):
        super().__init__('so101_mujoco_sim')
        default_model = os.path.join(get_package_share_directory('so101_mujoco'),
                                     'models', 'robotstudio_so101', 'scene_pick_place.xml')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('keyframe', '')           # empty = model's default pose
        self.declare_parameter('publish_rate', 50.0)     # Hz, /joint_states and /box_pose
        self.declare_parameter('realtime_factor', 1.0)   # 0 = run as fast as possible
        self.declare_parameter('use_viewer', False)
        self.declare_parameter('publish_clock', True)
        self.declare_parameter('randomize_box', True)
        self.declare_parameter('box_radius_range', [0.18, 0.25])
        self.declare_parameter('box_azimuth_range', [5.0, 40.0])
        self.declare_parameter('box_rel_yaw_range', [-45.0, 45.0])
        self.declare_parameter('seed', 0)
        self.declare_parameter('sync_mode', False)
        self.declare_parameter('sync_timeout', 1.0)      # s of wall time
        self.declare_parameter('show_workspace', True)   # viewer: draw the box sampling sector

        model_path = self.get_parameter('model_path').value
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.get_logger().info(f'Loaded {model_path}')

        keyframe = self.get_parameter('keyframe').value
        self.key_id = -1
        if keyframe:
            self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if self.key_id < 0:
                raise ValueError(f'Keyframe "{keyframe}" not found in model')

        # The controllable joints are the ones driven by actuators, in actuator order.
        m = self.model
        joint_ids = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
        self.joint_names = [m.joint(j).name for j in joint_ids]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in joint_ids])
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        self.name_to_index = {n: i for i, n in enumerate(self.joint_names)}
        self.box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'box')
        self.box_qpos_adr = None
        self.randomize_box = self.get_parameter('randomize_box').value
        if self.box_id >= 0:
            self.box_qpos_adr = m.jnt_qposadr[m.body_jntadr[self.box_id]]
            self.box_geom_id = int(m.body_geomadr[self.box_id])
            self.box_rgba = m.geom_rgba[self.box_geom_id].copy()
        self.rng = np.random.default_rng(self.get_parameter('seed').value)

        # Sim time keeps increasing across resets so /clock never jumps backwards.
        self.time_offset = 0.0
        # Only the sim (main) thread touches self.data. ROS callbacks run on the executor
        # thread and leave requests here for it, under self.pending, which is only held for
        # a copy. (Locking self.data while stepping starved the callbacks whenever the sim
        # could not keep up with realtime_factor: commands went unapplied for seconds.)
        self.pending = threading.Condition()
        # (threading.Event or None, (x, y, yaw) or None): the event is set once the reset has
        # happened, the pose is where /place_box wants the box (None = random).
        self.reset_requests = []
        self.command_stamp = -1    # ns, stamp of the latest /joint_command (for sync_mode)
        self._reset_state()
        self.ctrl_target = self.data.ctrl.copy()
        self.rtf = self.get_parameter('realtime_factor').value
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.box_pub = self.create_publisher(PoseStamped, 'box_pose', 10) if self.box_id >= 0 else None
        self.clock_pub = (self.create_publisher(Clock, 'clock', 10)
                          if self.get_parameter('publish_clock').value else None)
        self.create_subscription(JointState, 'joint_command', self._on_command, 10)
        self.create_service(Trigger, 'reset', self._on_reset)
        if self.box_qpos_adr is not None:
            self.create_subscription(Pose2D, 'place_box', self._on_place_box, 10)

        self.get_logger().info(f'Joints: {self.joint_names}')

    def sim_time(self):
        return self.time_offset + self.data.time

    def _reset_state(self, box_pose=None):
        """Reset physics and hold the initial pose; put the box at box_pose (x, y, yaw) if
        given, else at a random pose (randomize_box) or the model's. Sim thread only."""
        self.time_offset += self.data.time
        if self.key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if box_pose is not None:
            self._place_box(*box_pose)
        elif self.box_qpos_adr is not None and self.randomize_box:
            self._randomize_box()
        if self.box_qpos_adr is not None:
            x, y = self.data.qpos[self.box_qpos_adr:self.box_qpos_adr + 2]
            in_range = self._in_workspace(x, y)
            self.model.geom_rgba[self.box_geom_id] = self.box_rgba if in_range else (1, 0.8, 0, 1)
        self.data.ctrl[:] = np.clip(self.data.qpos[self.qpos_adr], self.ctrl_low, self.ctrl_high)
        mujoco.mj_forward(self.model, self.data)

    def _in_workspace(self, x, y, eps=1e-6):
        """Whether (x, y) is in the box sampling sector (box_radius_range x box_azimuth_range)."""
        r_low, r_high = self.get_parameter('box_radius_range').value
        az_low, az_high = self.get_parameter('box_azimuth_range').value
        r, az = np.hypot(x, y), np.degrees(np.arctan2(y, x))
        return bool(r_low - eps <= r <= r_high + eps and az_low - eps <= az <= az_high + eps)

    def _randomize_box(self):
        """Place the (free-floating) box upright at a random pose; height is kept from the model."""
        r = self.rng.uniform(*self.get_parameter('box_radius_range').value)
        azimuth = np.radians(self.rng.uniform(*self.get_parameter('box_azimuth_range').value))
        rel_yaw_range = list(self.get_parameter('box_rel_yaw_range').value)
        if rel_yaw_range == [-45.0, 45.0]:
            # Same draws as before box_rel_yaw_range existed, so old seeds give the same boxes.
            yaw = self.rng.uniform(-np.pi, np.pi)
        else:
            yaw = azimuth + np.pi / 2 + np.radians(self.rng.uniform(*rel_yaw_range))
        self._place_box(r * np.cos(azimuth), r * np.sin(azimuth), yaw)

    def _place_box(self, x, y, yaw):
        """Put the box upright at (x, y) with the given yaw; height is kept from the model."""
        adr = self.box_qpos_adr
        self.data.qpos[adr:adr + 2] = x, y
        self.data.qpos[adr + 3:adr + 7] = np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)

    def _on_command(self, msg):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if not msg.name and not msg.position:
            idx, targets = np.arange(0), np.zeros(0)  # just an acknowledgement
        elif msg.name:
            if len(msg.name) != len(msg.position):
                self.get_logger().warn('joint_command: name and position lengths differ',
                                       throttle_duration_sec=2.0)
                return
            unknown = [n for n in msg.name if n not in self.name_to_index]
            if unknown:
                self.get_logger().warn(f'joint_command: unknown joints {unknown}',
                                       throttle_duration_sec=2.0)
                return
            idx = np.array([self.name_to_index[n] for n in msg.name], dtype=int)
            targets = np.asarray(msg.position, dtype=float)
        elif len(msg.position) == len(self.joint_names):
            idx = np.arange(len(self.joint_names))
            targets = np.asarray(msg.position, dtype=float)
        else:
            self.get_logger().warn(
                f'joint_command without names must have {len(self.joint_names)} positions',
                throttle_duration_sec=2.0)
            return
        targets = np.clip(targets, self.ctrl_low[idx], self.ctrl_high[idx])
        with self.pending:
            self.ctrl_target[idx] = targets
            self.command_stamp = max(self.command_stamp, stamp)
            self.pending.notify_all()

    def _on_reset(self, request, response):
        done = threading.Event()
        with self.pending:
            self.reset_requests.append((done, None))
            self.pending.notify_all()
        response.success = done.wait(timeout=5.0)
        response.message = 'Simulation reset' if response.success else 'Sim loop not running'
        return response

    def _on_place_box(self, msg):
        with self.pending:
            self.reset_requests.append((None, (msg.x, msg.y, msg.theta)))
            self.pending.notify_all()

    def _on_set_parameters(self, params):
        for p in params:
            if p.name == 'realtime_factor':
                if p.type_ != Parameter.Type.DOUBLE or p.value < 0:
                    return SetParametersResult(successful=False,
                                               reason='realtime_factor must be a float >= 0')
                self.rtf = p.value  # picked up by run() on its next step
        return SetParametersResult(successful=True)

    def _apply_requests(self):
        """Apply resets and joint targets received since the last call. Sim thread only."""
        with self.pending:
            resets, self.reset_requests = self.reset_requests, []
            if resets:
                # Commands sent before the reset are dropped along with the old state. Only
                # the last request counts: a /place_box after a /reset keeps its box pose.
                self._reset_state(box_pose=resets[-1][1])
                self.ctrl_target = self.data.ctrl.copy()
            self.data.ctrl[:] = self.ctrl_target
        for done, _ in resets:
            if done is not None:
                done.set()

    def _wait_for_command(self, stamp, timeout):
        """sync_mode: block until the command answering the state stamped stamp (ns) arrives.
        Returns False on timeout."""
        with self.pending:
            return self.pending.wait_for(
                lambda: self.command_stamp >= stamp or self.reset_requests or not rclpy.ok(),
                timeout=timeout)

    def _publish(self):
        """Publish the current state, return its stamp in ns. Sim thread only."""
        stamp = to_time_msg(self.sim_time())
        if self.clock_pub:
            self.clock_pub.publish(Clock(clock=stamp))

        js = JointState()
        js.header.stamp = stamp
        js.name = self.joint_names
        js.position = self.data.qpos[self.qpos_adr].tolist()
        js.velocity = self.data.qvel[self.dof_adr].tolist()
        js.effort = self.data.actuator_force.tolist()
        self.joint_pub.publish(js)

        if self.box_pub:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = 'world'
            x, y, z = self.data.xpos[self.box_id]
            qw, qx, qy, qz = self.data.xquat[self.box_id]
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
            pose.pose.orientation.w, pose.pose.orientation.x = qw, qx
            pose.pose.orientation.y, pose.pose.orientation.z = qy, qz
            self.box_pub.publish(pose)
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _draw_workspace(self, viewer):
        """Outline the box sampling sector on the floor of the viewer."""
        r_low, r_high = self.get_parameter('box_radius_range').value
        az_low, az_high = np.radians(self.get_parameter('box_azimuth_range').value)
        z = 0.001
        arc = np.linspace(az_low, az_high, 24)
        points = ([(r_low * np.cos(a), r_low * np.sin(a)) for a in arc]
                  + [(r_high * np.cos(a), r_high * np.sin(a)) for a in arc[::-1]])
        with viewer.lock():
            scn = viewer.user_scn
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
                if scn.ngeom >= scn.maxgeom:
                    break
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                    np.zeros(3), np.eye(3).ravel(),
                                    np.array([1.0, 1.0, 1.0, 0.8], dtype=np.float32))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015,
                                     np.array([x0, y0, z]), np.array([x1, y1, z]))
                scn.ngeom += 1

    def run(self):
        """Step physics, publish at publish_rate, pace to realtime_factor. Blocks until shutdown."""
        rate = self.get_parameter('publish_rate').value
        steps_per_publish = max(1, round(1.0 / (rate * self.model.opt.timestep)))
        sync = self.get_parameter('sync_mode').value
        sync_timeout = self.get_parameter('sync_timeout').value

        viewer = None
        if self.get_parameter('use_viewer').value:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.box_qpos_adr is not None and self.get_parameter('show_workspace').value:
                self._draw_workspace(viewer)

        rtf = None
        try:
            while rclpy.ok() and (viewer is None or viewer.is_running()):
                if self.rtf != rtf:
                    # (Re)start pacing from here, so a new speed does not try to catch up.
                    rtf = self.rtf
                    wall_start = time.perf_counter()
                    sim_start = self.sim_time()
                self._apply_requests()
                for _ in range(steps_per_publish):
                    mujoco.mj_step(self.model, self.data)
                stamp = self._publish()
                if viewer:
                    viewer.sync()
                if sync and not self._wait_for_command(stamp, sync_timeout):
                    self.get_logger().warn('sync_mode: no /joint_command answered the last state, '
                                           'stepping on', throttle_duration_sec=10.0)
                sim_elapsed = self.sim_time() - sim_start
                if rtf > 0:
                    ahead = wall_start + sim_elapsed / rtf - time.perf_counter()
                    if ahead > 0:
                        time.sleep(ahead)
        finally:
            if viewer:
                viewer.close()


def spin_quietly(executor):
    try:
        executor.spin()
    except ExternalShutdownException:
        pass  # normal on Ctrl-C: rclpy shut the context down


def main():
    rclpy.init()
    node = MujocoSim()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=spin_quietly, args=(executor,), daemon=True)
    spin_thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
