import rclpy
from rclpy.node import Node

from game_msgs.msg import GameBoard
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from tf2_ros import Buffer, TransformListener, TransformException
import sympy as sp
import numpy as np
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

class BoardLocalizer(Node):
    def __init__(self):
        super().__init__('board_localizer')

        self.board_height = 0.26
        self.table_z = -0.28
        self.board_z = self.board_height + self.table_z

        self.camera_frame = 'camera1'
        self.cam_info_ready = False
        self.K = None
        self.K_inv = None

        self.board_sub = self.create_subscription(
            GameBoard,
            '/board_data',
            self.board_callback,
            10,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera1/camera_info',
            self.camera_info_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.top_left_pub = self.create_publisher(PointStamped, '/board_corner_tl_3d', 10)
        self.top_right_pub = self.create_publisher(PointStamped, '/board_corner_tr_3d', 10)

        self.cam_model = PinholeCameraModel()

        self.get_logger().info(f'Board Localizer initialized with board_z={self.board_z}')

    def camera_info_callback(self, msg: CameraInfo):
        self.cam_model.fromCameraInfo(msg)
        self.K = np.array(msg.k).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)
        self.camera_frame = msg.header.frame_id or self.camera_frame
        self.cam_info_ready = True

        self.get_logger().info(
            f'Received camera intrinsics. frame={self.camera_frame}, K=\n{self.K}'
        )

    def undistort_point(self, u, v):
        if not self.cam_info_ready:
            raise RuntimeError('CameraInfo not received yet')

        return self.cam_model.rectifyPoint((u, v))
    
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

    def depth_estimation(self, u, v):
            d = sp.symbols('d')
            point = sp.Matrix(np.array([u, v, 1.0]))
            ray = self.K_inv @ point
            depth_ray = d * ray

            tf = None
            source_frame = "camera1"
            target_frame = f"ar_marker_6_camera"
            try:
                tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
            except TransformException as ex:
                self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
                return

            g_tag_camera = sp.Matrix(self.tf_matrix(tf))

            source_frame = f"ar_marker_6"
            target_frame = "base_link"
            try:
                tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
            except TransformException as ex:
                self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
                return
            
            g_base_tag = sp.Matrix(self.tf_matrix(tf))
            g = g_base_tag * g_tag_camera

            self.g = g

            point_base = g * sp.Matrix([depth_ray[0], depth_ray[1], depth_ray[2], 1])
            point_base_z = point_base[2]

            depth = sp.solve(sp.Eq(point_base_z, self.board_z), d)
            
            point_base_3d = point_base.subs(d, depth[0])

            return point_base_3d

    def publish_corner(self, u, v, publisher, name):
        try:
            u_rect, v_rect = self.undistort_point(u, v)
        except RuntimeError as ex:
            self.get_logger().warn(str(ex))
            return

        point_3d = self.depth_estimation(u_rect, v_rect)
        if point_3d is None:
            self.get_logger().warn(f'Failed to estimate depth for {name} corner')
            return
        x, y, z = point_3d[0], point_3d[1], point_3d[2]
        point = PointStamped()
        point.header.frame_id = self.camera_frame or 'camera1'
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = float(x)
        point.point.y = float(y)
        point.point.z = float(z)

        publisher.publish(point)
        self.get_logger().info(
            f'Published {name} corner 3D position: x={x:.4f}, y={y:.4f}, z={z:.4f}'
        )

    def board_callback(self, msg: GameBoard):
        if len(msg.corner_x) != 4 or len(msg.corner_y) != 4:
            self.get_logger().warn('Invalid board corner message, expected 4 corners')
            return

        tl_u = msg.corner_x[0]
        tl_v = msg.corner_y[0]
        tr_u = msg.corner_x[1]
        tr_v = msg.corner_y[1]

        self.publish_corner(tl_u, tl_v, self.top_left_pub, 'top-left')
        self.publish_corner(tr_u, tr_v, self.top_right_pub, 'top-right')


def main(args=None):
    rclpy.init(args=args)
    node = BoardLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
