"""Sends a slow sinusoidal /joint_command so you can check the sim responds.

Run: ros2 run so101_mujoco wave_demo
"""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


class WaveDemo(Node):

    def __init__(self):
        super().__init__('so101_wave_demo')
        self.pub = self.create_publisher(JointState, 'joint_command', 10)
        self.t = 0.0
        self.dt = 0.05
        self.create_timer(self.dt, self._tick)

    def _tick(self):
        self.t += self.dt
        s = math.sin(2 * math.pi * 0.2 * self.t)  # 0.2 Hz
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'gripper']
        msg.position = [0.8 * s, -0.3 + 0.3 * s, 0.5 - 0.4 * s, 0.6 + 0.6 * s]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = WaveDemo()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
