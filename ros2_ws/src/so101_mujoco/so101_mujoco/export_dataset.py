"""Turn the episode bags written by collect_demos into one .npz of aligned training frames.

One frame per recorded /joint_states message:
    observation_state              (N, 6)  joint positions, rad (JOINT_NAMES order)
    observation_environment_state  (N, 5)  box features, see expert.box_features
    action                         (N, 6)  /joint_command answering that joint state
    episode_index, frame_index     (N,)
    timestamp                      (N,)    s since the start of the episode
Only successful episodes are exported. Episodes with a clip_start in episodes.jsonl (the
recovery demonstrations of collect_demos, whose first grasp missed on purpose) start at that
frame. Several demo directories can be merged into one .npz. The .npz has no ROS
dependencies, so it can be converted to a LeRobot dataset elsewhere
(scripts/npz_to_lerobot.py).

    ros2 run so101_mujoco export_dataset data/demos data/demos.npz
    ros2 run so101_mujoco export_dataset data/demos_offset data/demos_retry data/demos_offset_retry.npz
"""

import argparse
import json
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from .collect_demos import pose_to_array, stamp_ns
from .expert import box_features

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def read_bag(path):
    """{topic: [(stamp_ns, msg), ...]} with each list sorted by header stamp."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    topics = {name: [] for name in types}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, types[topic])
        topics[topic].append((stamp_ns(msg.header.stamp), msg))
    for msgs in topics.values():
        msgs.sort(key=lambda m: m[0])
    return topics


def ordered(msg):
    """Joint positions of a JointState in JOINT_NAMES order."""
    by_name = dict(zip(msg.name, msg.position))
    return [by_name[n] for n in JOINT_NAMES]


def episode_frames(topics, start=0):
    states = topics['joint_states']
    commands = {t: m for t, m in topics['joint_command']}
    boxes = topics['box_pose']
    box_stamps = np.array([t for t, _ in boxes])

    state, env, action, stamps = [], [], [], []
    for t, msg in [s for s in states if s[0] in commands][start:]:
        # Latest box pose at or before this state (sim_node publishes both with one stamp).
        i = max(0, np.searchsorted(box_stamps, t, side='right') - 1)
        state.append(ordered(msg))
        env.append(box_features(*pose_to_array(boxes[i][1])))
        action.append(ordered(commands[t]))
        stamps.append(t)
    stamps = (np.array(stamps) - stamps[0]) * 1e-9
    return np.array(state), np.array(env), np.array(action), stamps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('demo_dirs', nargs='+', help='output_dir(s) of collect_demos')
    parser.add_argument('output', help='.npz file to write')
    args = parser.parse_args()

    entries = []
    for demo_dir in args.demo_dirs:
        with open(os.path.join(demo_dir, 'episodes.jsonl')) as f:
            entries += [{**json.loads(line), 'dir': demo_dir} for line in f if line.strip()]
    entries = [e for e in entries if e['success'] and e['bag']]

    columns = {k: [] for k in ['observation_state', 'observation_environment_state', 'action',
                               'episode_index', 'frame_index', 'timestamp']}
    for new_index, entry in enumerate(entries):
        bag = os.path.join(entry['dir'], entry['bag'])
        state, env, action, stamps = episode_frames(read_bag(bag), entry.get('clip_start', 0))
        n = len(state)
        columns['observation_state'].append(state)
        columns['observation_environment_state'].append(env)
        columns['action'].append(action)
        columns['episode_index'].append(np.full(n, new_index))
        columns['frame_index'].append(np.arange(n))
        columns['timestamp'].append(stamps)
        dt = np.diff(stamps)
        irregular = np.abs(dt - np.median(dt)) > 0.5 * np.median(dt)
        note = f', WARNING: {irregular.sum()} irregular frame intervals' if irregular.any() else ''
        clip = f' (from frame {entry["clip_start"]})' if 'clip_start' in entry else ''
        print(f'{bag}: {n} frames{clip}, {stamps[-1]:.2f} s{note}')

    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    for k in ['observation_state', 'observation_environment_state', 'action', 'timestamp']:
        arrays[k] = arrays[k].astype(np.float32)
    dt = np.diff(arrays['timestamp'])[np.diff(arrays['episode_index']) == 0]
    fps = float(np.round(1.0 / np.median(dt)))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **arrays, fps=fps, joint_names=np.array(JOINT_NAMES))
    print(f'Wrote {len(entries)} episodes, {len(arrays["action"])} frames at {fps:g} fps '
          f'(dt min {dt.min() * 1e3:.1f} ms, max {dt.max() * 1e3:.1f} ms) to {args.output}')


if __name__ == '__main__':
    main()
