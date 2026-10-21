"""Convert the .npz written by `ros2 run so101_mujoco export_dataset` into a LeRobot dataset.

Runs outside the ROS environment, wherever LeRobot is installed (e.g. Colab); it only needs
numpy and lerobot. Tested with lerobot 0.6.1.

    pip install lerobot
    python scripts/npz_to_lerobot.py data/demos.npz --repo-id local/so101_sim_pick_place \
        --root data/lerobot/so101_sim_pick_place

Add --push to upload it to the Hugging Face Hub (needs `huggingface-cli login` and a
repo id under your user name).

With --cameras, every frame is also rendered from its recorded joint positions and box pose
(so101_mujoco.cameras, the same renderer sim_node uses live; needs mujoco, and an .npz with
joint_positions / box_pose, i.e. exported after they were added) and stored as
observation.images.<camera> videos. --no-env-state leaves out observation.environment_state,
so a policy trained on the dataset only sees the joints and the images:

    python scripts/npz_to_lerobot.py data/demos_offset_retry_pose.npz \
        --repo-id local/so101_sim_pick_place_images --root data/lerobot/so101_sim_pick_place_images \
        --cameras overview wrist_cam --no-env-state
"""

import argparse
import os
import sys
import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco', 'models',
                             'robotstudio_so101', 'scene_pick_place.xml')

TASK = 'Pick up the green cube and place it on the red square.'
BOX_FEATURES = ['box_x', 'box_y', 'box_z', 'box_cos4yaw', 'box_sin4yaw']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('npz')
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--root', help='output directory (default: $HF_LEROBOT_HOME/<repo-id>)')
    parser.add_argument('--num-episodes', type=int,
                        help='only convert the first N episodes (default: all)')
    parser.add_argument('--push', action='store_true', help='push to the Hugging Face Hub')
    parser.add_argument('--cameras', nargs='+', default=[],
                        help='render these cameras into observation.images.<camera>')
    parser.add_argument('--image-size', type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help='rendered image size (default: cameras.HEIGHT, cameras.WIDTH)')
    parser.add_argument('--crf', type=float, default=30,
                        help='video quality, lower = better and larger (LeRobot default 30)')
    parser.add_argument('--image-writer-threads', type=int, default=4,
                        help='threads writing the rendered frames to disk before encoding')
    parser.add_argument('--no-env-state', action='store_true',
                        help='leave out observation.environment_state (the box features)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='scene to render')
    args = parser.parse_args()

    # Load every array once: indexing an NpzFile decompresses the whole array on each access.
    data = dict(np.load(args.npz))
    joints = [str(n) for n in data['joint_names']]
    env_dim = data['observation_environment_state'].shape[1]
    assert env_dim == len(BOX_FEATURES), f'unexpected environment state size {env_dim}'

    features = {
        'observation.state': {'dtype': 'float32', 'shape': (len(joints),), 'names': joints},
        'action': {'dtype': 'float32', 'shape': (len(joints),), 'names': joints},
    }
    if not args.no_env_state:
        features['observation.environment_state'] = {'dtype': 'float32', 'shape': (env_dim,),
                                                     'names': BOX_FEATURES}

    renderer, extra = None, {}
    if args.cameras:
        import mujoco
        from lerobot.configs.video import RGBEncoderConfig
        sys.path.insert(0, os.path.join(REPO, 'ros2_ws', 'src', 'so101_mujoco'))
        from so101_mujoco import cameras

        for key in ['joint_positions', 'box_pose']:
            if key not in data:
                raise SystemExit(f'{args.npz} has no {key}: re-export it with export_dataset')
        height, width = args.image_size or (cameras.HEIGHT, cameras.WIDTH)
        renderer = cameras.CameraRenderer(mujoco.MjModel.from_xml_path(args.model), args.cameras,
                                          height, width, joint_names=joints)
        for cam in args.cameras:
            features[f'observation.images.{cam}'] = {
                'dtype': 'video', 'shape': (height, width, 3),
                'names': ['height', 'width', 'channels']}
        extra['rgb_encoder'] = RGBEncoderConfig(crf=args.crf)
        extra['image_writer_threads'] = args.image_writer_threads

    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=int(data['fps']), features=features,
                                    root=args.root, robot_type='so101_mujoco',
                                    use_videos=bool(args.cameras), **extra)

    episodes = data['episode_index']
    keep = np.unique(episodes)[:args.num_episodes]
    start = time.perf_counter()
    for n, ep in enumerate(keep):
        rows = np.flatnonzero(episodes == ep)
        assert np.all(np.diff(data['frame_index'][rows]) == 1), f'episode {ep} has gaps'
        for i in rows:
            frame = {
                'observation.state': data['observation_state'][i],
                'action': data['action'][i],
                'task': TASK,
            }
            if not args.no_env_state:
                frame['observation.environment_state'] = data['observation_environment_state'][i]
            if renderer:
                pose = data['box_pose'][i]
                images = renderer.render_state(data['joint_positions'][i], pose[:3], pose[3:])
                for cam, image in images.items():
                    frame[f'observation.images.{cam}'] = image
            dataset.add_frame(frame)
        dataset.save_episode()
        if renderer:
            elapsed = time.perf_counter() - start
            print(f'episode {ep}: {len(rows)} frames, {n + 1}/{len(keep)} done, '
                  f'{elapsed / 60:.1f} min, ~{elapsed / (n + 1) * (len(keep) - n - 1) / 60:.0f} min left',
                  flush=True)
    dataset.finalize()
    print(f'{len(keep)} episodes, {np.isin(episodes, keep).sum()} frames -> {dataset.root}')

    if args.push:
        dataset.push_to_hub()


if __name__ == '__main__':
    main()
