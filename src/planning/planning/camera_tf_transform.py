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
        self.get_logger().debug('Constant Transform Publisher initialized')
        self.g_base_ar = np.array([[-1, 0, 0, 0],
                      [0, 0, 1, 0.16],
                      [0, 1, 0, -0.13],
                      [0, 0, 0, 1.0]
        ])

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
    
    def invert_transform(self, pose):
        """Invert a pose transformation"""
        # Invert rotation
        rot = R.from_quat([pose.orientation.x, pose.orientation.y, 
                        pose.orientation.z, pose.orientation.w])
        rot_inv = rot.inv()
        
        # Invert translation
        trans = np.array([pose.position.x, pose.position.y, pose.position.z])
        trans_inv = -rot_inv.apply(trans)
        
        return trans_inv, rot_inv

    def aruco_marker_callback(self, msg):
        self.get_logger().debug(f"Marker callback triggered with {len(msg.marker_ids)} markers")
        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == 7:
                pose = msg.poses[i]
                trans_inv, rot_inv = self.invert_transform(pose)
                g_ar_camera = np.eye(4)
                g_ar_camera[0:3, 0:3] = rot_inv.as_matrix()
                g_ar_camera[0:3, 3] = trans_inv
                g_base_camera = self.g_base_ar @ g_ar_camera
                
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'base_link'
                t.child_frame_id = 'camera1'
                t.transform.translation.x = g_base_camera[0, 3]
                t.transform.translation.y = g_base_camera[1, 3]
                t.transform.translation.z = g_base_camera[2, 3]
                rot = R.from_matrix(g_base_camera[0:3, 0:3])
                q = rot.as_quat()
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]
                self.br.sendTransform(t)

def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
