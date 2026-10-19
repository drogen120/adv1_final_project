"""Interactive demo: place the box from a control panel, then watch the policy pick and place it.

Like policy_eval, the policy runs in scripts/policy_server.py and this node answers every
/joint_states with its action, but the episodes are started by hand from a small Tk window:

  - sliders for the box's distance r and azimuth from the arm base and its yaw; every change
    resets the scene with the box there (/place_box on sim_node), the arm in its start pose;
  - by default the sliders only cover the range the policy was trained on (train_radius_range
    x train_azimuth_range, drawn on the floor of the viewer); "allow outside" widens them, and
    sim_node shows a box outside the range in yellow;
  - Start runs the policy until the box has been on the target for a while and the arm is
    back in its start pose (or for max_steps frames); Stop ends a run early;
  - the speed menu sets sim_node's realtime_factor.

Every run is appended to <output_dir>/episodes.jsonl and <output_dir>/rollouts.npz in the
format of policy_eval, so scripts/render_eval_cases.py and visualize_demos.py can render them.
scripts/demo.sh starts everything (policy server, sim with viewer, this panel):

    pixi run bash scripts/demo.sh [--speed 0.5] [--checkpoint <dir>/pretrained_model]
"""

import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor

from .collect_demos import pose_to_array
from .expert import is_success
from .policy_eval import PolicyEvaluator, save_rollouts, spin_quietly

SPEEDS = [0.25, 0.5, 1.0, 2.0]
WIDE_RADIUS = (0.10, 0.40)     # m, slider range with "allow outside"
WIDE_AZIMUTH = (-90.0, 90.0)   # deg


class PolicyDemo(PolicyEvaluator):

    def __init__(self):
        super().__init__('so101_policy_demo')
        self.declare_parameter('train_radius_range', [0.18, 0.25])   # as sim_node's defaults
        self.declare_parameter('train_azimuth_range', [5.0, 40.0])
        self.declare_parameter('speed', 1.0)                # realtime_factor sim_node starts with
        self.declare_parameter('sim_node', 'so101_mujoco_sim')
        self.declare_parameter('success_hold', 1.0)         # s on the target before the arm
                                                            #   counts as done once back home
        self.declare_parameter('home_tolerance', 0.1)       # rad, max joint error to the start pose
        self.declare_parameter('max_success_hold', 5.0)     # s on the target that end a run anyway

        self.place_pub = self.create_publisher(Pose2D, 'place_box', 10)
        sim_node = self.get_parameter('sim_node').value
        self.param_client = self.create_client(SetParameters, f'/{sim_node}/set_parameters')
        self.success_run = 0   # frames in a row with the box on the target
        self.stopped = False

    def _episode_over(self):
        if super()._episode_over():
            return True
        state, env, _, _ = self.frames[-1]
        self.success_run = self.success_run + 1 if is_success(env[:3], self.target_pos) else 0
        held = self.success_run / 50.0   # frames are 50 Hz
        home = np.max(np.abs(np.subtract(state[:5], self.frames[0][0][:5])))
        return bool((held >= self.get_parameter('success_hold').value
                     and home < self.get_parameter('home_tolerance').value)
                    or held >= self.get_parameter('max_success_hold').value)

    def begin_episode(self):
        self.success_run, self.stopped = 0, False
        return super().begin_episode()

    def stop_episode(self):
        self.stopped = True
        super().stop_episode()

    def place_box(self, x, y, yaw):
        self.place_pub.publish(Pose2D(x=float(x), y=float(y), theta=float(yaw)))

    def set_speed(self, speed):
        if not self.param_client.service_is_ready():
            self.get_logger().warn('sim_node parameter service not available')
            return
        value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(speed))
        request = SetParameters.Request(parameters=[Parameter(name='realtime_factor', value=value)])
        self.param_client.call_async(request)

    def box_position(self):
        with self.lock:
            return None if self.last_box is None else pose_to_array(self.last_box)[0]

    def running(self):
        with self.lock:
            return self.active

    def frame_count(self):
        with self.lock:
            return len(self.frames) if self.frames else 0


class Panel:
    """The Tk control window. Runs on the main thread; ROS spins on another."""

    def __init__(self, node, out_dir):
        self.node, self.out_dir = node, out_dir
        self.results, self.episode_frames = [], []
        self.box_init = None
        self.place_after = None
        self.r_train = node.get_parameter('train_radius_range').value
        self.az_train = node.get_parameter('train_azimuth_range').value

        root = self.root = tk.Tk()
        root.title('Physical AI Final Project')
        root.resizable(False, False)
        root.protocol('WM_DELETE_WINDOW', self.close)
        frame = ttk.Frame(root, padding=12)
        frame.grid()

        box = ttk.LabelFrame(frame, text='Initial box pose', padding=8)
        box.grid(row=0, column=0, sticky='ew')
        self.r = tk.DoubleVar(value=100 * np.mean(self.r_train))
        self.az = tk.DoubleVar(value=np.mean(self.az_train))
        self.yaw = tk.DoubleVar(value=0.0)
        self.sliders = [
            self._slider(box, 0, 'Distance r (cm)', self.r, 100 * self.r_train[0], 100 * self.r_train[1]),
            self._slider(box, 1, 'Azimuth (°)', self.az, *self.az_train),
            self._slider(box, 2, 'Yaw (°)', self.yaw, -45.0, 45.0),
        ]
        self.allow_outside = tk.BooleanVar(value=False)
        self.outside_check = ttk.Checkbutton(box, text='Allow outside the training range (white outline)',
                                             variable=self.allow_outside,
                                             command=self._on_allow_outside)
        self.outside_check.grid(row=3, column=0, columnspan=3, sticky='w', pady=(6, 0))
        self.range_label = ttk.Label(box, text='')
        self.range_label.grid(row=4, column=0, columnspan=3, sticky='w')

        buttons = ttk.Frame(frame, padding=(0, 10))
        buttons.grid(row=1, column=0, sticky='ew')
        self.random_button = ttk.Button(buttons, text='Random', command=self.randomize)
        self.start_button = ttk.Button(buttons, text='Start', command=self.start)
        self.stop_button = ttk.Button(buttons, text='Stop', command=self.stop, state='disabled')
        self.reset_button = ttk.Button(buttons, text='Place again', command=self.place)
        for i, b in enumerate([self.random_button, self.start_button, self.stop_button,
                               self.reset_button]):
            b.grid(row=0, column=i, padx=2)

        speed_row = ttk.Frame(frame)
        speed_row.grid(row=2, column=0, sticky='w')
        ttk.Label(speed_row, text='Speed').grid(row=0, column=0, padx=(0, 6))
        speed = node.get_parameter('speed').value
        self.speed = tk.StringVar(value=self._speed_label(speed))
        labels = [self._speed_label(s) for s in sorted(set(SPEEDS) | {speed})]
        speed_menu = ttk.OptionMenu(speed_row, self.speed, self.speed.get(), *labels,
                                    command=lambda label: node.set_speed(float(label[:-1])))
        speed_menu.grid(row=0, column=1)

        self.status = ttk.Label(frame, text='Waiting for the simulator…', width=60)
        self.status.grid(row=3, column=0, sticky='w', pady=(10, 0))
        self.result_label = ttk.Label(frame, text='')
        self.result_label.grid(row=4, column=0, sticky='w')
        self.tally = ttk.Label(frame, text='')
        self.tally.grid(row=5, column=0, sticky='w')

        self._update_range_label()
        self.root.after(100, self._wait_for_sim)
        self.root.after(200, self._heartbeat)

    @staticmethod
    def _speed_label(speed):
        return f'{speed:g}×'

    def _slider(self, parent, row, text, var, low, high):
        ttk.Label(parent, text=text, width=15).grid(row=row, column=0, sticky='w')
        scale = ttk.Scale(parent, from_=low, to=high, variable=var, length=260,
                          command=lambda _: self._on_slider())
        scale.grid(row=row, column=1)
        value = ttk.Label(parent, width=6, anchor='e')
        value.grid(row=row, column=2)
        var.trace_add('write', lambda *_: value.config(text=f'{var.get():.1f}'))
        value.config(text=f'{var.get():.1f}')
        return scale

    # --- box placement ---

    def _box_pose(self):
        r, az = self.r.get() / 100, np.radians(self.az.get())
        return r * np.cos(az), r * np.sin(az), np.radians(self.yaw.get())

    def _in_train_range(self):
        return (self.r_train[0] - 1e-6 <= self.r.get() / 100 <= self.r_train[1] + 1e-6
                and self.az_train[0] - 1e-6 <= self.az.get() <= self.az_train[1] + 1e-6)

    def _update_range_label(self):
        if self._in_train_range():
            self.range_label.config(text='Inside the training range', foreground='green')
        else:
            self.range_label.config(text='Outside the training range (yellow box): the policy may fail',
                                    foreground='#b07000')

    def _on_slider(self):
        self._update_range_label()
        # Sliders fire on every pixel of a drag; place the box once it pauses.
        if self.place_after is not None:
            self.root.after_cancel(self.place_after)
        self.place_after = self.root.after(80, self.place)

    def _on_allow_outside(self):
        if self.allow_outside.get():
            r_range = [100 * v for v in WIDE_RADIUS]
            az_range = WIDE_AZIMUTH
        else:
            r_range = [100 * v for v in self.r_train]
            az_range = self.az_train
            self.r.set(float(np.clip(self.r.get(), *r_range)))
            self.az.set(float(np.clip(self.az.get(), *az_range)))
        self.sliders[0].config(from_=r_range[0], to=r_range[1])
        self.sliders[1].config(from_=az_range[0], to=az_range[1])
        self._update_range_label()
        self.place()

    def place(self):
        self.place_after = None
        if self.node.running():
            return
        self.node.place_box(*self._box_pose())
        self.start_button.state(['!disabled'])
        self.status.config(text='Box placed, press Start to run the policy')

    def randomize(self):
        rng = np.random.default_rng()
        self.r.set(100 * rng.uniform(*self.r_train))
        self.az.set(rng.uniform(*self.az_train))
        self.yaw.set(rng.uniform(-45.0, 45.0))
        self._update_range_label()
        self.place()

    # --- runs ---

    def _wait_for_sim(self):
        if self.node.box_position() is None:
            self.root.after(200, self._wait_for_sim)
            return
        self.place()

    def _set_running(self, running):
        idle = 'disabled' if running else '!disabled'
        for widget in self.sliders + [self.outside_check, self.random_button,
                                      self.start_button, self.reset_button]:
            widget.state([idle])
        self.stop_button.state(['!disabled' if running else 'disabled'])

    def start(self):
        if self.node.running() or self.node.box_position() is None:
            return
        if self.place_after is not None:
            return   # a placement is still pending; wait for it
        self.box_init = self.node.begin_episode()
        self._set_running(True)
        self.result_label.config(text='')
        self.root.after(100, self._poll_run)

    def stop(self):
        self.node.stop_episode()

    def _poll_run(self):
        if not self.node.done.is_set():
            n = self.node.frame_count()
            self.status.config(text=f'Running: frame {n} ({n / 50:.1f} s sim time)')
            self.root.after(100, self._poll_run)
            return
        self._set_running(False)
        # The arm is wherever the run left it: place the box again before the next run.
        self.start_button.state(['disabled'])
        if self.node.frame_count() == 0:
            self.status.config(text='Stopped, place the box again')
            return
        self._record(*self.node.end_episode(self.box_init))

    def _record(self, result, frames):
        result['stopped'] = self.node.stopped
        result['in_train_range'] = self._in_train_range()
        episode = len(self.results)
        self.results.append(result)
        self.episode_frames.append(frames)
        with open(os.path.join(self.out_dir, 'episodes.jsonl'), 'a') as f:
            f.write(json.dumps({'episode': episode, **result}) + '\n')
        save_rollouts(os.path.join(self.out_dir, 'rollouts.npz'), self.episode_frames)

        if result['success']:
            text, colour = f'Success: {100 * result["target_dist"]:.1f} cm from the target centre', 'green'
        elif result['stopped']:
            text, colour = 'Stopped by hand', 'gray'
        else:
            text = 'Failed: ' + ('dropped / misplaced' if result['lifted'] else 'box not lifted')
            colour = 'red'
        if result['grasp_attempts'] > 1:
            text += f' ({result["grasp_attempts"]} grasps)'
        self.result_label.config(text=text, foreground=colour)
        self.status.config(text=f'Run {episode} finished after {len(frames)} frames'
                                f' ({len(frames) / 50:.1f} s); place the box again')
        n_success = sum(r['success'] for r in self.results)
        self.tally.config(text=f'Total: {n_success}/{len(self.results)} successful')
        self.node.get_logger().info(f'run {episode}: {text}')

    def _heartbeat(self):
        # Python only handles signals (Ctrl-C) when it runs, which an idle Tk loop never does.
        if not rclpy.ok():
            self.close()
            return
        self.root.after(200, self._heartbeat)

    def close(self):
        if self.node.running():
            self.node.stop_episode()
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = PolicyDemo()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=spin_quietly, args=(executor,), daemon=True).start()
    out_dir = os.path.join(os.path.abspath(node.get_parameter('output_dir').value),
                           time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    node.get_logger().info(f'Recording runs into {out_dir}')
    try:
        Panel(node, out_dir).mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.policy.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
