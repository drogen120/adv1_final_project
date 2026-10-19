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

To evaluate on the boundary where the expert switches grasps (relative grasp angle |g| in
[35, 45] deg), add -p box_rel_yaw_range:=[35.0,55.0] to sim_node, see sim_node.py.
"""

import json
import os
import socket
import threading
import time

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .collect_demos import pose_to_array, stamp_ns
from .expert import box_features, is_success
from .export_dataset import JOINT_NAMES, ordered
from .kinematics import TopDownIK


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

    def reset(self):
        self._call({'reset': True})

    def act(self, state, environment_state):
        return self._call({'state': list(state), 'environment_state': list(environment_state)})['action']

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

        self.cmd_pub = self.create_publisher(JointState, 'joint_command', 10)
        self.create_subscription(JointState, 'joint_states', self._on_joint_states, 100)
        self.create_subscription(PoseStamped, 'box_pose', self._on_box_pose, 100)
        self.reset_client = self.create_client(Trigger, 'reset')

        # Shared with the main thread; callbacks run on the executor thread.
        self.lock = threading.Lock()
        self.state_count = 0
        self.last_box = None
        self.pending_state = None   # /joint_states waiting for the /box_pose with its stamp
        self.active = False
        self.frames = None          # recorded (state, env, action, stamp) of the episode
        self.max_box_z = 0.0
        self.policy_time = 0.0
        self.done = threading.Event()

    # --- callbacks (executor thread) ---

    def _on_box_pose(self, msg):
        with self.lock:
            self.last_box = msg
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
        """Answer the pending joint state once the box pose with the same stamp is in."""
        state_msg, box_msg = self.pending_state, self.last_box
        if not self.active or state_msg is None or box_msg is None:
            return
        t = stamp_ns(state_msg.header.stamp)
        if stamp_ns(box_msg.header.stamp) != t:
            return
        self.pending_state = None
        box_pos, box_quat = pose_to_array(box_msg)
        state = ordered(state_msg)
        env = box_features(box_pos, box_quat).tolist()
        start = time.perf_counter()
        action = self.policy.act(state, env)
        self.policy_time += time.perf_counter() - start

        cmd = JointState()
        cmd.header.stamp = state_msg.header.stamp
        cmd.name = JOINT_NAMES
        cmd.position = [float(a) for a in action]
        self.cmd_pub.publish(cmd)

        self.frames.append((state, env, cmd.position, t))
        self.max_box_z = max(self.max_box_z, box_pos[2])
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
        while not self.done.wait(0.1):
            if not rclpy.ok():
                raise KeyboardInterrupt
        return self.end_episode(box_init)

    def begin_episode(self):
        """Reset the policy and start answering /joint_states with it, from the current
        state of the sim. Returns the box position at the start."""
        self.policy.reset()
        with self.lock:
            box_init, _ = pose_to_array(self.last_box)
            self.frames, self.max_box_z, self.policy_time = [], 0.0, 0.0
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
        success = is_success(box_final, self.target_pos)
        result = {
            'box_init': box_init.round(4).tolist(),
            'box_final': box_final.round(4).tolist(),
            'success': success,
            'target_dist': round(float(np.linalg.norm(box_final[:2] - self.target_pos[:2])), 4),
            'lifted': bool(self.max_box_z > 0.04),   # box centre rests at 0.02 m
            'grasp_attempts': self._grasp_attempts(frames),
            'success_frame': self._success_frame(frames),
            'frames': len(frames),
            'policy_ms': round(1e3 * self.policy_time / max(1, len(frames)), 2),
        }
        return result, frames

    def _grasp_attempts(self, frames, threshold=0.5, max_dist=0.04, max_height=0.05):
        """Times the gripper command went from open to closed (across threshold, rad) with
        the tool within max_dist (m) of the box horizontally and below max_height (m), i.e.
        not when the gripper closes on the way back to the start pose."""
        grip = np.array([action[5] for _, _, action, _ in frames])
        attempts = 0
        for i in np.flatnonzero((grip[:-1] >= threshold) & (grip[1:] < threshold)) + 1:
            state, env, _, _ = frames[i]
            tool, _ = self.kinematics.forward(np.array(state[:5]))
            attempts += bool(np.linalg.norm(tool[:2] - env[:2]) < max_dist and tool[2] < max_height)
        return attempts

    def _success_frame(self, frames):
        """First frame from which the box is on the target (expert.is_success) to the end."""
        ok = [is_success(env[:3], self.target_pos) for _, env, _, _ in frames]
        if not ok[-1]:
            return None
        return len(ok) - int(np.argmin(ok[::-1])) if not all(ok) else 0

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

        n_success = sum(r['success'] for r in results)
        n_lifted = sum(r['lifted'] for r in results)
        n_regrasped = sum(r['grasp_attempts'] > 1 for r in results)
        n_recovered = sum(r['grasp_attempts'] > 1 and r['success'] for r in results)
        n_by_600 = sum(r['success'] and r['success_frame'] < 600 for r in results)
        summary = {'episodes': num_episodes, 'success_rate': n_success / num_episodes,
                   'successes': n_success, 'successes_by_frame_600': n_by_600,
                   'lifted': n_lifted, 'regrasped': n_regrasped, 'recovered': n_recovered}
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        self.get_logger().info(f'Success rate: {n_success}/{num_episodes} '
                               f'({100 * n_success / num_episodes:.0f}%), '
                               f'box lifted in {n_lifted}/{num_episodes}, '
                               f'regrasped in {n_regrasped} ({n_recovered} succeeded), '
                               f'{n_by_600} succeeded within 600 frames')


def save_rollouts(path, episode_frames):
    """Write the frames of each episode (lists of (state, env, action, stamp)) to path, in the
    format of export_dataset."""
    columns = {k: [] for k in ['observation_state', 'observation_environment_state', 'action',
                               'episode_index', 'frame_index', 'timestamp']}
    for episode, frames in enumerate(episode_frames):
        state, env, action, stamps = map(np.array, zip(*frames))
        columns['observation_state'].append(state)
        columns['observation_environment_state'].append(env)
        columns['action'].append(action)
        columns['episode_index'].append(np.full(len(frames), episode))
        columns['frame_index'].append(np.arange(len(frames)))
        columns['timestamp'].append((stamps - stamps[0]) * 1e-9)
    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    for k in ['observation_state', 'observation_environment_state', 'action', 'timestamp']:
        arrays[k] = arrays[k].astype(np.float32)
    np.savez_compressed(path, **arrays, fps=50.0, joint_names=np.array(JOINT_NAMES))


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
