import os

from setuptools import find_packages, setup

package_name = 'so101_mujoco'


def model_files():
    """Install everything under models/ into share/, keeping the directory layout."""
    entries = []
    for root, _, files in os.walk('models'):
        if files:
            entries.append((os.path.join('share', package_name, root),
                            [os.path.join(root, f) for f in files]))
    return entries


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *model_files(),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wang Qianlong',
    maintainer_email='wangqianlong120@gmail.com',
    description='MuJoCo simulation of the SO-101 arm exposed over ROS 2 topics.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'sim_node = so101_mujoco.sim_node:main',
            'wave_demo = so101_mujoco.wave_demo:main',
            'collect_demos = so101_mujoco.collect_demos:main',
            'export_dataset = so101_mujoco.export_dataset:main',
            'policy_eval = so101_mujoco.policy_eval:main',
        ],
    },
)
