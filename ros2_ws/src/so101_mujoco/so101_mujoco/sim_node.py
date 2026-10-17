"""MuJoCo simulation of the SO-101 arm, exposed as a ROS 2 node.

Topics
  publishes   /joint_states   sensor_msgs/JointState      arm joint position/velocity/effort
              /box_pose       geometry_msgs/PoseStamped   pose of the "box" body, if the scene has one
              /clock          rosgraph_msgs/Clock         simulation time (if publish_clock is true)
  subscribes  /joint_command  sensor_msgs/JointState      position targets in rad; names optional;
                                                          no positions = keep the current targets
Services
              /reset          std_srvs/Trigger            reset the scene to its initial state

With randomize_box (default), every reset puts the box at a random position and yaw on the
floor, sampled in polar coordinates around the arm base from box_radius_range (m) and
box_azimuth_range (deg). The default ranges are the part of the top-down grasp workspace
where the scripted expert succeeds (see expert.py).

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
from geometry_msgs.msg import PoseStamped
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
        self.declare_parameter('seed', 0)
        self.declare_parameter('sync_mode', False)
        self.declare_parameter('sync_timeout', 1.0)      # s of wall time

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
        if self.box_id >= 0 and self.get_parameter('randomize_box').value:
            self.box_qpos_adr = m.jnt_qposadr[m.body_jntadr[self.box_id]]
        self.rng = np.random.default_rng(self.get_parameter('seed').value)

        # Sim time keeps increasing across resets so /clock never jumps backwards.
        self.time_offset = 0.0
        # Only the sim (main) thread touches self.data. ROS callbacks run on the executor
        # thread and leave requests here for it, under self.pending, which is only held for
        # a copy. (Locking self.data while stepping starved the callbacks whenever the sim
        # could not keep up with realtime_factor: commands went unapplied for seconds.)
        self.pending = threading.Condition()
        self.reset_requests = []   # threading.Events, set once the reset has happened
        self.command_stamp = -1    # ns, stamp of the latest /joint_command (for sync_mode)
        self._reset_state()
        self.ctrl_target = self.data.ctrl.copy()

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.box_pub = self.create_publisher(PoseStamped, 'box_pose', 10) if self.box_id >= 0 else None
        self.clock_pub = (self.create_publisher(Clock, 'clock', 10)
                          if self.get_parameter('publish_clock').value else None)
        self.create_subscription(JointState, 'joint_command', self._on_command, 10)
        self.create_service(Trigger, 'reset', self._on_reset)

        self.get_logger().info(f'Joints: {self.joint_names}')

    def sim_time(self):
        return self.time_offset + self.data.time

    def _reset_state(self):
        """Reset physics and hold the initial pose. Sim thread only."""
        self.time_offset += self.data.time
        if self.key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if self.box_qpos_adr is not None:
            self._randomize_box()
        self.data.ctrl[:] = np.clip(self.data.qpos[self.qpos_adr], self.ctrl_low, self.ctrl_high)
        mujoco.mj_forward(self.model, self.data)

    def _randomize_box(self):
        """Place the (free-floating) box upright at a random pose; height is kept from the model."""
        r = self.rng.uniform(*self.get_parameter('box_radius_range').value)
        azimuth = np.radians(self.rng.uniform(*self.get_parameter('box_azimuth_range').value))
        yaw = self.rng.uniform(-np.pi, np.pi)
        adr = self.box_qpos_adr
        self.data.qpos[adr:adr + 2] = r * np.cos(azimuth), r * np.sin(azimuth)
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
            self.reset_requests.append(done)
            self.pending.notify_all()
        response.success = done.wait(timeout=5.0)
        response.message = 'Simulation reset' if response.success else 'Sim loop not running'
        return response

    def _apply_requests(self):
        """Apply resets and joint targets received since the last call. Sim thread only."""
        with self.pending:
            resets, self.reset_requests = self.reset_requests, []
            if resets:
                # Commands sent before the reset are dropped along with the old state.
                self._reset_state()
                self.ctrl_target = self.data.ctrl.copy()
            self.data.ctrl[:] = self.ctrl_target
        for done in resets:
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

    def run(self):
        """Step physics, publish at publish_rate, pace to realtime_factor. Blocks until shutdown."""
        rate = self.get_parameter('publish_rate').value
        rtf = self.get_parameter('realtime_factor').value
        steps_per_publish = max(1, round(1.0 / (rate * self.model.opt.timestep)))
        sync = self.get_parameter('sync_mode').value
        sync_timeout = self.get_parameter('sync_timeout').value

        viewer = None
        if self.get_parameter('use_viewer').value:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

        wall_start = time.perf_counter()
        sim_start = self.sim_time()
        try:
            while rclpy.ok() and (viewer is None or viewer.is_running()):
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
