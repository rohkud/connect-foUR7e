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
from geometry_msgs.msg import TransformStamped, PoseArray, PointStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from rclpy.time import Time

class ConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('constant_tf_publisher')
        self.br = TransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.markerSub = self.create_subscription(
            ArucoMarkers,
            'aruco_markers',
            self.aruco_marker_callback,
            5,
        )
        self.get_logger().info('Constant Transform Publisher initialized')

    def tf_matrix(self, tf):
        # Convert geometry_msgs/Transform to 4x4 homogeneous transformation matrix
        translation = (tf.transform.translation.x,
                       tf.transform.translation.y,
                       tf.transform.translation.z)
        rotation_quat = (tf.transform.rotation.x,
                         tf.transform.rotation.y,
                         tf.transform.rotation.z,
                         tf.transform.rotation.w)
        # Convert quaternion to rotation matrix using scipy
        rot = R.from_quat(rotation_quat)
        rot_matrix = rot.as_matrix()
        
        # Create 4x4 homogeneous matrix
        T = np.eye(4)
        T[0:3, 0:3] = rot_matrix
        T[0:3, 3] = translation
        return T
    
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

        # tf = None
        # source_frame = "camera1"
        # target_frame = "ar_marker_6_camera"
        # try:
        #     tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        # except TransformException as ex:
        #     self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
        #     return

        # g_tag_camera = self.tf_matrix(tf)

        # source_frame = "ar_marker_6"
        # target_frame = "base_link"
        # try:
        #     tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        # except TransformException as ex:
        #     self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
        #     return
    
        # g_base_tag = self.tf_matrix(tf)
        # g = g_base_tag @ g_tag_camera
        # x, y, z = g[0, 3], g[1, 3], g[2, 3]
        # point = PointStamped()
        # point.header.stamp = self.get_clock().now().to_msg()
        # point.header.frame_id = 'camera1'
        # point.point.x = x
        # point.point.y = y
        # point.point.z = z
        # self.camera_pose_pub.publish(point)
        # self.get_logger().info(f'Published camera pose: x={x:.3f}, y={y:.3f}, z={z:.3f}')

def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
