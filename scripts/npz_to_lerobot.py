"""Convert the .npz written by `ros2 run so101_mujoco export_dataset` into a LeRobot dataset.

Runs outside the ROS environment, wherever LeRobot is installed (e.g. Colab); it only needs
numpy and lerobot. Tested with lerobot 0.6.1.

    pip install lerobot
    python scripts/npz_to_lerobot.py data/demos.npz --repo-id local/so101_sim_pick_place \
        --root data/lerobot/so101_sim_pick_place

Add --push to upload it to the Hugging Face Hub (needs `huggingface-cli login` and a
repo id under your user name).
"""

import argparse

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
    args = parser.parse_args()

    data = np.load(args.npz)
    joints = [str(n) for n in data['joint_names']]
    env_dim = data['observation_environment_state'].shape[1]
    assert env_dim == len(BOX_FEATURES), f'unexpected environment state size {env_dim}'

    features = {
        'observation.state': {'dtype': 'float32', 'shape': (len(joints),), 'names': joints},
        'observation.environment_state': {'dtype': 'float32', 'shape': (env_dim,),
                                          'names': BOX_FEATURES},
        'action': {'dtype': 'float32', 'shape': (len(joints),), 'names': joints},
    }
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=int(data['fps']), features=features,
                                    root=args.root, robot_type='so101_mujoco', use_videos=False)

    episodes = data['episode_index']
    keep = np.unique(episodes)[:args.num_episodes]
    for ep in keep:
        rows = np.flatnonzero(episodes == ep)
        assert np.all(np.diff(data['frame_index'][rows]) == 1), f'episode {ep} has gaps'
        for i in rows:
            dataset.add_frame({
                'observation.state': data['observation_state'][i],
                'observation.environment_state': data['observation_environment_state'][i],
                'action': data['action'][i],
                'task': TASK,
            })
        dataset.save_episode()
    dataset.finalize()
    print(f'{len(keep)} episodes, {np.isin(episodes, keep).sum()} frames -> {dataset.root}')

    if args.push:
        dataset.push_to_hub()


if __name__ == '__main__':
    main()
