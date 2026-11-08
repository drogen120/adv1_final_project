"""Run a trained policy in closed loop on sim_node and measure its success rate.

The policy itself runs in scripts/policy_server.py, in a separate venv with LeRobot; this
node only turns ROS messages into its observations and its actions into /joint_command,
over a local TCP socket (newline-delimited JSON, see policy_server.py).

For each episode: call /reset, reset the policy, then answer every /joint_states with the
policy's action for that frame, for max_steps frames; the episode succeeds if the box ends
up on the target (expert.is_success). The observation is built exactly like the training
frames in export_dataset: the joint positions plus expert.box_features of the /box_pose with
the same stamp. Run the sim in sync_mode, with a seed the demonstrations did not use:

    .venv-lerobot/bin/python scripts/policy_server.py <checkpoint>/pretrained_model
    ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 -p seed:=100
    ros2 run so101_mujoco policy_eval --ros-args -p num_episodes:=50 -p output_dir:=data/eval/act_200

<output_dir>/episodes.jsonl gets one line per episode, <output_dir>/rollouts.npz all the
frames in the format of export_dataset (so scripts/visualize_demos.py --data can replay them).
grasp_attempts counts how often the policy closed the gripper low down next to the box, so
regrasps after a missed grasp show up as more than one; success_frame is the frame from
which the box stays on the target (None if it does not end there). After a missed grasp the
policies trained on the plain demos carry on to the target and back to the start pose, which
looks like the start of an episode, so they do the whole task again: success_frame shows
whether that still fits in a shorter episode.

A policy trained on camera images (scripts/npz_to_lerobot.py --cameras) gets the images of
/camera/<name>/image_raw with the stamp of the joint state instead of the box features: the
node asks the server which inputs the policy has (policy_server.py, "info") and waits for
those. sim_node must then publish the cameras (-p cameras:=overview,wrist_cam). /box_pose is
still used, but only to judge success and for the metrics:

    .venv-lerobot/bin/python scripts/policy_server.py outputs/act_img_291/checkpoints/060000/pretrained_model
    ros2 run so101_mujoco sim_node --ros-args -p sync_mode:=true -p realtime_factor:=0.0 \
        -p seed:=100 -p cameras:=overview,wrist_cam
    ros2 run so101_mujoco policy_eval --ros-args -p num_episodes:=50 -p output_dir:=data/eval/act_img_291

To evaluate on the boundary where the expert switches grasps (relative grasp angle |g| in
[35, 45] deg), add -p box_rel_yaw_range:=[35.0,55.0] to sim_node, see sim_node.py.
"""

import base64
import json
import os
import socket
import threading
import time

import mujoco
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger

from .collect_demos import pose_to_array, stamp_ns
from .expert import box_features
from .export_dataset import JOINT_NAMES, ordered
from .kinematics import TopDownIK
from .rollouts import episode_result, save_rollouts, summarize


class PolicyClient:
    """Blocking client for scripts/policy_server.py."""

    def __init__(self, host, port, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.file = self.sock.makefile('rwb')

    def _call(self, request):
        self.file.write((json.dumps(request) + '\n').encode())
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise ConnectionError('policy server closed the connection')
        return json.loads(line)

    def info(self):
        """Which inputs the policy takes: {'environment_state': bool, 'cameras': [names]}."""
        return self._call({'info': True})

    def reset(self):
        self._call({'reset': True})

    def act(self, state, environment_state=None, images=None):
        """images: {camera: sensor_msgs/Image (rgb8)}."""
        request = {'state': list(state)}
        if environment_state is not None:
            request['environment_state'] = list(environment_state)
        if images:
            request['images'] = {
                cam: {'shape': [msg.height, msg.width, 3],
                      'data': base64.b64encode(bytes(msg.data)).decode('ascii')}
                for cam, msg in images.items()}
        return self._call(request)['action']

    def close(self):
        self.file.close()
        self.sock.close()


class PolicyEvaluator(Node):

    def __init__(self, node_name='so101_policy_eval'):
        super().__init__(node_name)
        default_model = os.path.join(get_package_share_directory('so101_mujoco'),
                                     'models', 'robotstudio_so101', 'scene_pick_place.xml')
        self.declare_parameter('model_path', default_model)   # same scene as sim_node
        self.declare_parameter('num_episodes', 50)
        self.declare_parameter('max_steps', 900)              # frames; the demos take 506,
                                                              #   ~700 with a regrasp
        self.declare_parameter('output_dir', 'eval')
        self.declare_parameter('server_host', '127.0.0.1')
        self.declare_parameter('server_port', 5555)

        model = mujoco.MjModel.from_xml_path(self.get_parameter('model_path').value)
        self.target_pos = model.site('target').pos.copy()
        self.kinematics = TopDownIK(model)
        self.max_steps = self.get_parameter('max_steps').value
        self.policy = PolicyClient(self.get_parameter('server_host').value,
                                   self.get_parameter('server_port').value)
        info = self.policy.info()
        self.uses_env = info['environment_state']
        self.cameras = info['cameras']
        self.get_logger().info(f'Policy inputs: joint state'
                               + (', box features' if self.uses_env else '')
                               + ''.join(f', camera {c}' for c in self.cameras))

        self.cmd_pub = self.create_publisher(JointState, 'joint_command', 10)
        self.create_subscription(JointState, 'joint_states', self._on_joint_states, 100)
        self.create_subscription(PoseStamped, 'box_pose', self._on_box_pose, 100)
        for cam in self.cameras:
            self.create_subscription(Image, f'camera/{cam}/image_raw',
                                     lambda msg, cam=cam: self._on_image(cam, msg), 10)
        self.reset_client = self.create_client(Trigger, 'reset')

        # Shared with the main thread; callbacks run on the executor thread.
        self.lock = threading.Lock()
        self.state_count = 0
        self.last_box = None
        self.last_images = {}       # camera -> latest sensor_msgs/Image
        self.pending_state = None   # /joint_states waiting for the /box_pose with its stamp
        self.active = False
        self.frames = None          # recorded (state, env, action, stamp) of the episode
        self.policy_time = 0.0
        self.done = threading.Event()

    # --- callbacks (executor thread) ---

    def _on_box_pose(self, msg):
        with self.lock:
            self.last_box = msg
            self._maybe_act()

    def _on_image(self, cam, msg):
        if msg.encoding != 'rgb8':
            self.get_logger().error(f'{cam}: expected rgb8 images, got {msg.encoding}',
                                    throttle_duration_sec=5.0)
            return
        with self.lock:
            self.last_images[cam] = msg
            self._maybe_act()

    def _on_joint_states(self, msg):
        with self.lock:
            self.state_count += 1
            if not self.active:
                cmd = JointState()
                cmd.header.stamp = msg.header.stamp
                self.cmd_pub.publish(cmd)  # empty: keep the targets, lets a sync_mode sim go on
                return
            self.pending_state = msg
            self._maybe_act()

    def _maybe_act(self):
        """Answer the pending joint state once the box pose and the camera images with the
        same stamp are in."""
        state_msg, box_msg = self.pending_state, self.last_box
        if not self.active or state_msg is None or box_msg is None:
            return
        t = stamp_ns(state_msg.header.stamp)
        if stamp_ns(box_msg.header.stamp) != t:
            return
        images = {cam: self.last_images.get(cam) for cam in self.cameras}
        if any(msg is None or stamp_ns(msg.header.stamp) != t for msg in images.values()):
            return
        self.pending_state = None
        box_pos, box_quat = pose_to_array(box_msg)
        state = ordered(state_msg)
        env = box_features(box_pos, box_quat).tolist()  # recorded for the metrics in any case
        start = time.perf_counter()
        action = self.policy.act(state, env if self.uses_env else None, images)
        self.policy_time += time.perf_counter() - start

        cmd = JointState()
        cmd.header.stamp = state_msg.header.stamp
        cmd.name = JOINT_NAMES
        cmd.position = [float(a) for a in action]
        self.cmd_pub.publish(cmd)

        self.frames.append((state, env, cmd.position, t))
        if self._episode_over():
            self.active = False
            self.done.set()

    def _episode_over(self):
        """Called with the lock held after each frame; True ends the episode."""
        return len(self.frames) >= self.max_steps

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

    def run_episode(self):
        if not self.reset_client.call(Trigger.Request()).success:
            raise RuntimeError('/reset failed')
        # Messages published before the reset may still be queued; let them drain.
        self._wait_states(5)
        box_init = self.begin_episode()
        start = time.monotonic()
        while not self.done.wait(0.1):
            if not rclpy.ok():
                raise KeyboardInterrupt
            with self.lock:
                stuck = not self.frames and time.monotonic() - start > 10.0
            if stuck:
                missing = [c for c in self.cameras if c not in self.last_images]
                raise RuntimeError(
                    'no observation complete after 10 s'
                    + (f'; no images from {missing}: start sim_node with '
                       f'-p cameras:={",".join(self.cameras)}' if missing else ''))
        return self.end_episode(box_init)

    def begin_episode(self):
        """Reset the policy and start answering /joint_states with it, from the current
        state of the sim. Returns the box position at the start."""
        self.policy.reset()
        with self.lock:
            box_init, _ = pose_to_array(self.last_box)
            self.frames, self.policy_time = [], 0.0
            self.pending_state = None
            self.done.clear()
            self.active = True
        return box_init

    def stop_episode(self):
        """End the running episode early (done is set as when it ends by itself)."""
        with self.lock:
            self.active = False
            self.done.set()

    def end_episode(self, box_init):
        """Judge the episode that just ended. Returns (result, frames)."""
        # Let the box come to rest after the last command before judging.
        self._wait_states(2)
        with self.lock:
            box_final, _ = pose_to_array(self.last_box)
            frames = self.frames
        result = episode_result(frames, box_init, box_final, self.target_pos, self.kinematics,
                                self.policy_time)
        return result, frames

    def run(self):
        out_dir = os.path.abspath(self.get_parameter('output_dir').value)
        os.makedirs(out_dir, exist_ok=True)
        num_episodes = self.get_parameter('num_episodes').value
        if not self.reset_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('/reset service not available; is sim_node running?')
        self.get_logger().info(f'Evaluating {num_episodes} episodes into {out_dir}')

        results, episode_frames = [], []
        with open(os.path.join(out_dir, 'episodes.jsonl'), 'w') as f:
            for episode in range(num_episodes):
                result, frames = self.run_episode()
                results.append(result)
                episode_frames.append(frames)
                f.write(json.dumps({'episode': episode, **result}) + '\n')
                f.flush()
                outcome = 'success' if result['success'] else \
                    'FAILED (' + ('dropped/misplaced' if result['lifted'] else 'not lifted') + ')'
                self.get_logger().info(f'episode {episode}: {outcome}, '
                                       f'{100 * result["target_dist"]:.1f} cm from target, '
                                       f'{result["grasp_attempts"]} grasp(s), '
                                       f'{result["policy_ms"]} ms/step')

        save_rollouts(os.path.join(out_dir, 'rollouts.npz'), episode_frames)

        summary = summarize(results)
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        n_success, n_recovered = summary['successes'], summary['recovered']
        n_lifted, n_regrasped = summary['lifted'], summary['regrasped']
        n_by_600 = summary['successes_by_frame_600']
        self.get_logger().info(f'Success rate: {n_success}/{num_episodes} '
                               f'({100 * n_success / num_episodes:.0f}%; strict '
                               f'{summary["strict_successes"]}), '
                               f'box lifted in {n_lifted}/{num_episodes}, '
                               f'regrasped in {n_regrasped} ({n_recovered} succeeded), '
                               f'{n_by_600} succeeded within 600 frames')


def spin_quietly(executor):
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def main():
    rclpy.init()
    node = PolicyEvaluator()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=spin_quietly, args=(executor,), daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.policy.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
