import rclpy
from rclpy.node import Node

from game_msgs.msg import DiscLoc2d, GameBoard
from ros2_aruco_interfaces.msg import ArucoMarkers
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped

import numpy as np
import cv2


BASE_MARKER_ID = 6

# Table is 25 cm below base_link
TABLE_Z_IN_BASE = -0.25

# Professor-provided static transform:
# parent = ar_marker_6
# child  = base_link
# This maps marker frame -> base_link frame
T_MARKER_TO_BASE = np.array([
    [-1, 0, 0,  0.0],
    [ 0, 0, 1,  0.16],
    [ 0, 1, 0, -0.13],
    [ 0, 0, 0,  1.0]
])


def quat_to_R(q):
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y]
    ])


def pose_to_T(pose):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(pose.orientation)
    T[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z
    ]
    return T


def transform_point(T, p):
    p_h = np.array([p[0], p[1], p[2], 1.0])
    return (T @ p_h)[:3]


class PieceLocalizer(Node):
    def __init__(self):
        super().__init__('piece_localizer')

        self.K = None
        self.K_inv = None
        self.homography = None

        # From /aruco_markers:
        # pose of ar_marker_6 in camera frame
        self.T_MARKER_TO_CAMERA = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera1/camera_info',
            self.camera_info_callback,
            10
        )

        self.aruco_sub = self.create_subscription(
            ArucoMarkers,
            '/aruco_markers',
            self.aruco_callback,
            10
        )

        self.disc_sub = self.create_subscription(
            DiscLoc2d,
            '/disc_data',
            self.disc_callback,
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
            '/localized_pieces_base',
            10
        )

        self.get_logger().info('Piece localizer initialized')

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)
        self.get_logger().info('Received camera intrinsics')

    def aruco_callback(self, msg):
        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            if marker_id == BASE_MARKER_ID:
                self.T_MARKER_TO_CAMERA = pose_to_T(pose)
                self.get_logger().info('Updated base ArUco pose')

    def board_callback(self, msg):
        if len(msg.corner_x) == 4 and len(msg.corner_y) == 4:
            src_points = np.array(
                [[msg.corner_x[i], msg.corner_y[i]] for i in range(4)],
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

    def get_T_camera_to_base(self):
        if self.T_MARKER_TO_CAMERA is None:
            return None

        # ArUco detection gives:
        # T_MARKER_TO_CAMERA = marker frame -> camera frame
        #
        # Professor transform gives:
        # T_MARKER_TO_BASE = marker frame -> base_link frame
        #
        # Need:
        # T_CAMERA_TO_BASE = camera frame -> base_link frame

        T_CAMERA_TO_MARKER = np.linalg.inv(self.T_MARKER_TO_CAMERA)
        T_CAMERA_TO_BASE = T_MARKER_TO_BASE @ T_CAMERA_TO_MARKER

        return T_CAMERA_TO_BASE

    def pixel_ray_in_camera(self, u, v):
        ray = self.K_inv @ np.array([u, v, 1.0])
        ray = ray / np.linalg.norm(ray)
        return ray

    def pixel_to_table_point_in_base(self, u, v):
        T_CAMERA_TO_BASE = self.get_T_camera_to_base()

        if T_CAMERA_TO_BASE is None:
            return None

        ray_cam = self.pixel_ray_in_camera(u, v)

        # Camera origin in camera frame
        cam_origin_cam = np.array([0.0, 0.0, 0.0])

        # Transform camera origin into base_link frame
        cam_origin_base = transform_point(T_CAMERA_TO_BASE, cam_origin_cam)

        # Rotate ray into base_link frame
        R_CAMERA_TO_BASE = T_CAMERA_TO_BASE[:3, :3]
        ray_base = R_CAMERA_TO_BASE @ ray_cam
        ray_base = ray_base / np.linalg.norm(ray_base)

        # Intersect ray with table plane:
        # z = TABLE_Z_IN_BASE

        if abs(ray_base[2]) < 1e-6:
            return None

        t = (TABLE_Z_IN_BASE - cam_origin_base[2]) / ray_base[2]

        if t < 0:
            return None

        point_base = cam_origin_base + t * ray_base
        return point_base

    def disc_callback(self, msg):
        if self.K_inv is None:
            self.get_logger().warn('No camera intrinsics yet')
            return

        if self.T_MARKER_TO_CAMERA is None:
            self.get_logger().warn('No base ArUco detected yet')
            return

        if self.homography is None:
            self.get_logger().warn('No board homography yet')
            return

        for i, (u, v, color) in enumerate(zip(msg.x, msg.y, msg.color)):
            transformed = self.apply_homography(self.homography, u, v)

            if transformed is None:
                continue

            tx, ty = transformed

            # Board bounds in canonical board coordinates
            inside_board = (0 <= tx < 700) and (0 <= ty < 600)

            # We only want loose pieces outside the board
            if inside_board:
                self.get_logger().info(
                    f'Skipping disc {i}: inside board at ({tx:.1f}, {ty:.1f})'
                )
                continue

            point_base = self.pixel_to_table_point_in_base(u, v)

            if point_base is None:
                self.get_logger().warn(f'Could not localize disc {i}')
                continue

            out = PointStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = 'base_link'

            out.point.x = float(point_base[0])
            out.point.y = float(point_base[1])
            out.point.z = float(point_base[2])

            self.localized_pub.publish(out)

            self.get_logger().info(
                f'{color} loose disc {i} in base_link: '
                f'x={point_base[0]:.3f}, '
                f'y={point_base[1]:.3f}, '
                f'z={point_base[2]:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = PieceLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()