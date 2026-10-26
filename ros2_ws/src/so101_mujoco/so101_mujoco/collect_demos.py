"""Record scripted pick-and-place demonstrations from sim_node into rosbags.

For each episode: call /reset (sim_node randomizes the box), then run the scripted expert
(expert.PickPlaceEpisode) from the first /joint_states and /box_pose after the reset,
streaming its targets to /joint_command, one per /joint_states message. The expert plans one
phase at a time: once the grasp's targets have been sent, it checks the latest /box_pose
and either carries the box to the target or, if the box was not lifted, grasps again. Every episode is written to its
own bag, <output_dir>/episode_NNNN (mcap), containing /joint_states, /box_pose and
/joint_command. Commands carry the stamp of the joint state they answer, which is how the
exporter pairs observations with actions. <output_dir>/episodes.jsonl gets one line per
episode with its outcome.

Recovery demonstrations: with inject_failure_prob > 0, that share of the episodes get a first
grasp that misses on purpose (expert.sample_injection, mode drawn from inject_modes). They
are kept only if the regrasp then succeeds, and their episodes.jsonl line has clip_start, the
frame where the regrasp starts: export_dataset drops the frames before it, so the policy
learns to recover from a missed grasp but not to miss. Use a new seed (here and in sim_node)
and a separate output_dir:

    ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 \
        -p output_dir:=data/demos_retry -p inject_failure_prob:=1.0 -p seed:=3

Near the grasp switching boundary (|g| in [30, 45] deg; sim_node with the same seed and
-p box_rel_yaw_range:=[30.0,60.0]), the 'midway' mode mimics the image-based policy:

    ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=100 \
        -p output_dir:=data/demos_boundary_retry -p inject_failure_prob:=1.0 -p seed:=5 \
        -p "inject_modes:=['midway']" -p inject_delay:=[0.0,0.3]

Run the sim in sync_mode, so that it waits for each command; otherwise any hiccup in this
node leaves the arm holding an old target while the plan moves on, and the recorded
actions no longer match the states:

    ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0
    ros2 run so101_mujoco collect_demos --ros-args -p num_episodes:=50 -p output_dir:=data/demos

The sim's publish_rate must equal control_rate (both default to 50 Hz).
"""

import json
import os
import shutil
import threading
import time

import mujoco
import numpy as np
import rclpy
import rosbag2_py
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .expert import (GRASP_OFFSET, PickPlaceEpisode, PickPlaceExpert, PlanningError, box_yaw,
                     is_success, relative_grasp_angle, sample_injection)

TOPICS = {
    'joint_states': 'sensor_msgs/msg/JointState',
    'box_pose': 'geometry_msgs/msg/PoseStamped',
    'joint_command': 'sensor_msgs/msg/JointState',
}


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def pose_to_array(msg):
    p, o = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z])


class DemoCollector(Node):

    def __init__(self):
        super().__init__('so101_demo_collector')
        default_model = os.path.join(get_package_share_directory('so101_mujoco'),
                                     'models', 'robotstudio_so101', 'scene_pick_place.xml')
        self.declare_parameter('model_path', default_model)   # same scene as sim_node
        self.declare_parameter('num_episodes', 10)
        self.declare_parameter('output_dir', 'demos')
        self.declare_parameter('control_rate', 50.0)          # Hz, must match sim publish_rate
        # s recorded after the plan ends, holding the arm at home with the box on the target.
        # Longer than ACT's 1 s action chunk, so the policy learns to stay put once done
        # (with 0.5 s it went for the box again). Data before 2026-10 used 0.5.
        self.declare_parameter('settle_time', 2.0)
        self.declare_parameter('keep_failures', False)        # keep bags of failed episodes
        self.declare_parameter('grasp_offset', GRASP_OFFSET)  # m, see expert.GRASP_OFFSET
        self.declare_parameter('max_retries', 2)              # regrasps after a missed grasp
        self.declare_parameter('inject_failure_prob', 0.0)    # share of episodes whose first grasp misses
        self.declare_parameter('inject_modes', ['wrist', 'offset'])
        self.declare_parameter('inject_yaw_error', [25.0, 35.0])          # deg, 'wrist'
        self.declare_parameter('inject_wrist_tool_error', [0.010, 0.016])  # m, 'wrist'
        self.declare_parameter('inject_tool_error', [0.012, 0.020])        # m, 'offset'
        self.declare_parameter('inject_midway_fraction', [0.7, 1.0])       # of g, 'midway'
        self.declare_parameter('inject_midway_tool_error', [0.007, 0.011])  # m, 'midway'
        self.declare_parameter('inject_delay', [0.0, 0.8])     # s carrying the empty gripper
        self.declare_parameter('seed', 0)                     # for the injections

        model = mujoco.MjModel.from_xml_path(self.get_parameter('model_path').value)
        self.target_pos = model.site('target').pos.copy()
        self.expert = PickPlaceExpert(model, rate=self.get_parameter('control_rate').value,
                                      grasp_offset=self.get_parameter('grasp_offset').value)
        self.rng = np.random.default_rng(self.get_parameter('seed').value)
        self.settle_frames = int(round(self.get_parameter('settle_time').value
                                       * self.get_parameter('control_rate').value))

        self.cmd_pub = self.create_publisher(JointState, 'joint_command', 10)
        # Deep queues: a dropped /joint_states message is a missing frame in the episode.
        self.create_subscription(JointState, 'joint_states', self._on_joint_states, 100)
        self.create_subscription(PoseStamped, 'box_pose', self._on_box_pose, 100)
        self.reset_client = self.create_client(Trigger, 'reset')

        # Shared with the main thread; callbacks run on the executor thread.
        self.lock = threading.Lock()
        self.state_count = 0
        self.last_state = None
        self.last_box = None
        self.writer = None
        self.episode = None   # PickPlaceEpisode; None once it has no more targets
        self.plan = None      # targets handed out so far, one row per frame
        self.step = 0
        self.end_step = None  # frame at which recording stops
        self.error = None     # PlanningError raised mid-episode
        self.done = threading.Event()

    # --- callbacks (executor thread) ---

    def _on_box_pose(self, msg):
        with self.lock:
            self.last_box = msg
            if self.writer:
                self.writer.write('box_pose', serialize_message(msg), stamp_ns(msg.header.stamp))

    def _on_joint_states(self, msg):
        with self.lock:
            self.last_state = msg
            self.state_count += 1
            cmd = JointState()
            cmd.header.stamp = msg.header.stamp
            if self.plan is None:
                self.cmd_pub.publish(cmd)  # empty: keep the targets, lets a sync_mode sim go on
                return
            if self.step >= len(self.plan) and self.episode is not None:
                self._extend_plan()
            if self.end_step is not None and self.step >= self.end_step:
                self.cmd_pub.publish(cmd)
                self.plan = None
                self.done.set()
                return
            t = stamp_ns(msg.header.stamp)
            self.writer.write('joint_states', serialize_message(msg), t)
            cmd.name = list(msg.name)
            cmd.position = self.plan[min(self.step, len(self.plan) - 1)].tolist()
            self.cmd_pub.publish(cmd)
            self.writer.write('joint_command', serialize_message(cmd), t)
            self.step += 1

    def _extend_plan(self):
        """All targets sent: ask the expert for the next phase, given the latest box pose
        (from the previous frame; this frame's arrives after its joint state). Planning here
        holds up the sim, which waits for this frame's command in sync_mode."""
        try:
            rows = self.episode.next_rows(*pose_to_array(self.last_box))
        except PlanningError as e:
            self.error, rows = e, None
        if rows is not None:
            self.plan = np.vstack([self.plan, rows])
            return
        failed = self.error is not None or self.episode.failure is not None
        self.end_step = self.step + (0 if failed else self.settle_frames)
        self.episode = None

    # --- episode loop (main thread) ---

    def _wait_states(self, n, timeout=5.0):
        """Block until n more /joint_states messages have arrived."""
        with self.lock:
            target = self.state_count + n
        deadline = time.monotonic() + timeout
        while rclpy.ok():
            with self.lock:
                if self.state_count >= target:
                    return
            if time.monotonic() > deadline:
                raise TimeoutError('no /joint_states from the simulator')
            time.sleep(0.01)

    def _open_bag(self, path):
        writer = rosbag2_py.SequentialWriter()
        # A cache makes write() hand messages to a background thread; with the default of 0
        # the callback blocks while mcap compresses chunks, and /joint_states messages drop.
        writer.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap',
                                              max_cache_size=100 * 1024 * 1024),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        for i, (name, msg_type) in enumerate(TOPICS.items()):
            writer.create_topic(rosbag2_py.TopicMetadata(i, name, msg_type, 'cdr'))
        return writer

    def run_episode(self, path):
        """Record one episode into the bag at path. Returns a dict describing the outcome."""
        if not self.reset_client.call(Trigger.Request()).success:
            raise RuntimeError('/reset failed')
        # Messages published before the reset may still be queued; let them drain.
        self._wait_states(5)
        with self.lock:
            q_start = np.array(self.last_state.position)
            box_pos, box_quat = pose_to_array(self.last_box)
        info = {'box_init': box_pos.round(4).tolist(), 'box_quat_init': box_quat.round(4).tolist()}
        injection = None
        if self.rng.uniform() < self.get_parameter('inject_failure_prob').value:
            p = lambda name: self.get_parameter(name).value  # noqa: E731
            mode = str(self.rng.choice(p('inject_modes')))
            injection = sample_injection(self.rng, mode,
                                         p('inject_yaw_error'), p('inject_wrist_tool_error'),
                                         p('inject_tool_error'), p('inject_delay'),
                                         p('inject_midway_fraction'), p('inject_midway_tool_error'))
            yaw_error = injection.yaw_error - injection.midway * relative_grasp_angle(
                box_pos, box_yaw(box_quat))
            info['injection'] = {'mode': mode,
                                 'yaw_error_deg': round(float(np.degrees(yaw_error)), 2),
                                 'tool_error': round(float(injection.tool_error), 4),
                                 'delay': round(float(injection.delay), 3)}
        episode = PickPlaceEpisode(self.expert, q_start, self.target_pos, injection,
                                   self.get_parameter('max_retries').value)
        try:
            plan = episode.next_rows(box_pos, box_quat)
        except PlanningError as e:
            return {**info, 'success': False, 'reason': f'planning: {e}', 'frames': 0}

        writer = self._open_bag(path)
        self.done.clear()
        with self.lock:
            self.writer, self.step, self.plan = writer, 0, plan
            self.episode, self.end_step, self.error = episode, None, None
        try:
            while not self.done.wait(0.1):
                if not rclpy.ok():
                    raise KeyboardInterrupt
        finally:
            with self.lock:
                self.writer, self.plan, self.episode = None, None, None
                box_final, _ = pose_to_array(self.last_box)
                frames, error = self.step, self.error
            writer.close()
        result = {**info, 'box_final': box_final.round(4).tolist(), 'frames': frames,
                  'attempts': episode.attempts}
        if error is not None:
            result['reason'] = f'planning: {error}'
        elif episode.failure is not None:
            result['reason'] = episode.failure
        result['success'] = 'reason' not in result and is_success(box_final, self.target_pos)
        if injection is not None and episode.retry_frame is not None:
            result['clip_start'] = episode.retry_frame
        return result

    def run(self):
        out_dir = os.path.abspath(self.get_parameter('output_dir').value)
        os.makedirs(out_dir, exist_ok=True)
        index_path = os.path.join(out_dir, 'episodes.jsonl')
        # Continue numbering after any episodes already in output_dir.
        start = 0
        if os.path.exists(index_path):
            with open(index_path) as f:
                start = max((json.loads(line)['episode'] + 1 for line in f if line.strip()), default=0)

        if not self.reset_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('/reset service not available; is sim_node running?')
        self.get_logger().info(f'Recording {self.get_parameter("num_episodes").value} '
                               f'episodes into {out_dir}')
        successes = 0
        for episode in range(start, start + self.get_parameter('num_episodes').value):
            path = os.path.join(out_dir, f'episode_{episode:04d}')
            result = self.run_episode(path)
            successes += result['success']
            if not result['success'] and os.path.exists(path) \
                    and not self.get_parameter('keep_failures').value:
                shutil.rmtree(path)
            bag = os.path.basename(path) if os.path.exists(path) else None
            with open(index_path, 'a') as f:
                f.write(json.dumps({'episode': episode, 'bag': bag, **result}) + '\n')
            self.get_logger().info(f'episode {episode}: '
                                   f'{"success" if result["success"] else "FAILED"} '
                                   f'({result["frames"]} frames) {result.get("reason", "")}')
        self.get_logger().info(f'Done: {successes}/{self.get_parameter("num_episodes").value} succeeded')


def spin_quietly(executor):
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def main():
    rclpy.init()
    node = DemoCollector()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=spin_quietly, args=(executor,), daemon=True).start()
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
