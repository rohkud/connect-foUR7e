import rclpy
from rclpy.node import Node

from game_msgs.msg import DiscLoc2d, GameBoard
from ros2_aruco_interfaces.msg import ArucoMarkers

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo

import numpy as np
import cv2
import sympy as sp
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from image_geometry import PinholeCameraModel

class PieceLocalizer(Node):
    def __init__(self):
        super().__init__('piece_localizer')

        self.declare_parameter('table_marker_id', 12)

        self.table_marker_id = (
            self.get_parameter('table_marker_id')
            .get_parameter_value()
            .integer_value
        )

        self.disc_sub = self.create_subscription(
            DiscLoc2d,
            '/disc_data',
            self.disc_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera1/camera_info',
            self.camera_info_callback,
            10
        )

        self.board_sub = self.create_subscription(
            GameBoard,
            '/board_data',
            self.board_callback,
            10
        )

        self.localized_pub = self.create_publisher(
            PointStamped,
            '/localized_pieces',
            10
        )

        self.K = None
        self.K_inv = None
        self.sympy_K_inv = None
        self.homography = None
        self.table_z = -0.25

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cam_model = PinholeCameraModel()

        self.get_logger().info('Piece Localizer initialized')

    def undistort_point(self, u, v):
        if not self.cam_info_ready:
            raise RuntimeError("CameraInfo not received yet")

        return self.cam_model.rectifyPoint((u, v))

    def camera_info_callback(self, msg: CameraInfo):
        self.cam_model.fromCameraInfo(msg)
        self.K = np.array(msg.k).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)
        self.sympy_K_inv = sp.Matrix(self.K_inv)

        self.get_logger().info(f'Received camera intrinsics:\n{self.K}')
        self.cam_info_ready = True

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
        target_frame = "ar_marker_6_camera"
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return

        g_tag_camera = sp.Matrix(self.tf_matrix(tf))

        source_frame = "ar_marker_6"
        target_frame = "base_link"
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return
        
        g_base_tag = sp.Matrix(self.tf_matrix(tf))

        point_base = g_base_tag * g_tag_camera * sp.Matrix([depth_ray[0], depth_ray[1], depth_ray[2], 1])
        point_base_z = point_base[2]

        depth = sp.solve(sp.Eq(point_base_z, self.table_z), d)
        point_base_3d = point_base.subs(d, depth[0])

        return point_base_3d

    def board_callback(self, msg: GameBoard):
        if len(msg.corner_x) == 4 and len(msg.corner_y) == 4:
            src_points = np.array(
                [self.undistort_point(msg.corner_x[i], msg.corner_y[i]) for i in range(4)],
                dtype=np.float32
            )

            # Assuming order: TL, TR, BR, BL
            dst_points = np.array(
                [[0, 0], [700, 0], [700, 600], [0, 600]],
                dtype=np.float32
            )

            self.homography = cv2.getPerspectiveTransform(src_points, dst_points)
            self.get_logger().info('Computed homography from board corners')
        else:
            self.homography = None
            self.get_logger().warn('Invalid board corners, homography not computed')

    def apply_homography(self, H, x, y):
        point = np.array([x, y, 1.0], dtype=np.float32)
        warped = H.dot(point)

        if warped[2] == 0:
            return None

        return float(warped[0] / warped[2]), float(warped[1] / warped[2])

    def disc_callback(self, msg: DiscLoc2d):
        if self.K is None:
            self.get_logger().warn('Camera intrinsics not received yet')
            return

        if self.homography is None:
            self.get_logger().warn('Board homography not received yet')
            return

        for i, (x, y, color) in enumerate(zip(msg.x, msg.y, msg.color)):
            x, y = self.undistort_point(x, y)
            transformed = self.apply_homography(self.homography, x, y)

            if transformed is None:
                continue

            tx, ty = transformed

            inside_board = (0 <= tx < 700) and (0 <= ty < 600)
            # We only want loose pieces outside the board
            if inside_board:
                self.get_logger().info(
                    f'Skipping disc {i}: inside board at ({tx:.1f}, {ty:.1f})'
                )
                continue

            point_3d = self.depth_estimation(x, y)

            if point_3d is None:
                self.get_logger().warn(f'Could not project disc {i} to table plane')
                continue

            localized_point = PointStamped()
            localized_point.header.frame_id = 'base_link'
            localized_point.header.stamp = self.get_clock().now().to_msg()

            localized_point.point.x = float(point_3d[0])
            localized_point.point.y = float(point_3d[1])
            localized_point.point.z = float(point_3d[2])

            self.localized_pub.publish(localized_point)

            self.get_logger().info(
                f'Localized {color} disc at camera frame: '
                f'x={point_3d[0]:.3f}, y={point_3d[1]:.3f}, z={point_3d[2]:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PieceLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()