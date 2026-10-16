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

    def __init__(self):
        super().__init__('so101_policy_eval')
        default_model = os.path.join(get_package_share_directory('so101_mujoco'),
                                     'models', 'robotstudio_so101', 'scene_pick_place.xml')
        self.declare_parameter('model_path', default_model)   # same scene as sim_node
        self.declare_parameter('num_episodes', 50)
        self.declare_parameter('max_steps', 600)              # frames; the demos take 506
        self.declare_parameter('output_dir', 'eval')
        self.declare_parameter('server_host', '127.0.0.1')
        self.declare_parameter('server_port', 5555)

        model = mujoco.MjModel.from_xml_path(self.get_parameter('model_path').value)
        self.target_pos = model.site('target').pos.copy()
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
        if len(self.frames) >= self.max_steps:
            self.active = False
            self.done.set()

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
        self.policy.reset()
        with self.lock:
            box_init, _ = pose_to_array(self.last_box)
            self.frames, self.max_box_z, self.policy_time = [], 0.0, 0.0
            self.pending_state = None
            self.done.clear()
            self.active = True
        while not self.done.wait(0.1):
            if not rclpy.ok():
                raise KeyboardInterrupt
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
            'frames': len(frames),
            'policy_ms': round(1e3 * self.policy_time / max(1, len(frames)), 2),
        }
        return result, frames

    def run(self):
        out_dir = os.path.abspath(self.get_parameter('output_dir').value)
        os.makedirs(out_dir, exist_ok=True)
        num_episodes = self.get_parameter('num_episodes').value
        if not self.reset_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('/reset service not available; is sim_node running?')
        self.get_logger().info(f'Evaluating {num_episodes} episodes into {out_dir}')

        columns = {k: [] for k in ['observation_state', 'observation_environment_state', 'action',
                                   'episode_index', 'frame_index', 'timestamp']}
        results = []
        with open(os.path.join(out_dir, 'episodes.jsonl'), 'w') as f:
            for episode in range(num_episodes):
                result, frames = self.run_episode()
                results.append(result)
                f.write(json.dumps({'episode': episode, **result}) + '\n')
                f.flush()
                state, env, action, stamps = map(np.array, zip(*frames))
                columns['observation_state'].append(state)
                columns['observation_environment_state'].append(env)
                columns['action'].append(action)
                columns['episode_index'].append(np.full(len(frames), episode))
                columns['frame_index'].append(np.arange(len(frames)))
                columns['timestamp'].append((stamps - stamps[0]) * 1e-9)
                outcome = 'success' if result['success'] else \
                    'FAILED (' + ('dropped/misplaced' if result['lifted'] else 'not lifted') + ')'
                self.get_logger().info(f'episode {episode}: {outcome}, '
                                       f'{100 * result["target_dist"]:.1f} cm from target, '
                                       f'{result["policy_ms"]} ms/step')

        arrays = {k: np.concatenate(v) for k, v in columns.items()}
        for k in ['observation_state', 'observation_environment_state', 'action', 'timestamp']:
            arrays[k] = arrays[k].astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, 'rollouts.npz'), **arrays,
                            fps=50.0, joint_names=np.array(JOINT_NAMES))

        n_success = sum(r['success'] for r in results)
        n_lifted = sum(r['lifted'] for r in results)
        summary = {'episodes': num_episodes, 'success_rate': n_success / num_episodes,
                   'successes': n_success, 'lifted': n_lifted}
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        self.get_logger().info(f'Success rate: {n_success}/{num_episodes} '
                               f'({100 * n_success / num_episodes:.0f}%), '
                               f'box lifted in {n_lifted}/{num_episodes}')


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
