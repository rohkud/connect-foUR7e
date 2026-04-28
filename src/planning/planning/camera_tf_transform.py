"""
================================================================================
Static Transform Broadcaster (static_tf_transform.py)
================================================================================

PURPOSE:
    Publishes static coordinate frame transformation from robot wrist to camera.
    Enables coordinate conversions between the UR7e gripper frame and RealSense
    camera frame, essential for connecting 3D perception to motion control.
================================================================================
"""

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseArray
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from ros2_aruco_interfaces.msg import ArucoMarkers
import numpy as np

class ConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('constant_tf_publisher')
        self.br = TransformBroadcaster(self)

        self.markerSub = self.create_subscription(
            ArucoMarkers,
            'aruco_markers',
            self.aruco_marker_callback,
            5,
        )

        self.get_logger().info('Constant Transform Publisher initialized')

    def aruco_marker_callback(self, msg):
        self.get_logger().info(f"Marker callback triggered with {len(msg.marker_ids)} markers")
        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == 6:
                pose = msg.poses[i]
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'camera1'
                t.child_frame_id = 'ar_marker_6_camera'
                t.transform.translation.x = pose.position.x
                t.transform.translation.y = pose.position.y
                t.transform.translation.z = pose.position.z
                t.transform.rotation = pose.orientation
                self.br.sendTransform(t)
                self.get_logger().info(f'Broadcasting transform from camera to ar_marker_6')

def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
