#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from image_geometry import PinholeCameraModel

import numpy as np
from scipy.spatial.transform import Rotation as R

from piece_localization_interfaces.srv import PixelToPoint


class PixelToPointService(Node):
    def __init__(self):
        super().__init__('pixel_to_point_service')

        self.declare_parameter('camera_frame', 'camera1')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('table_z', -0.28)

        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.table_z = self.get_parameter('table_z').value

        self.K = None
        self.K_inv = None
        self.cam_info_ready = False

        self.cam_model = PinholeCameraModel()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera1/camera_info',
            self.camera_info_callback,
            10
        )

        self.localized_pub = self.create_publisher(
            PointStamped,
            '/localized_pixel_3d',
            10
        )

        self.service = self.create_service(
            PixelToPoint,
            '/pixel_to_point',
            self.pixel_to_point_callback
        )

        self.get_logger().debug('Pixel-to-3D service ready on /pixel_to_point')

    def camera_info_callback(self, msg: CameraInfo):
        self.cam_model.fromCameraInfo(msg)
        self.K = np.array(msg.k).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)

        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

        self.cam_info_ready = True

    def tf_matrix(self, tf):
        translation = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        ])

        quat = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w
        ]

        rot = R.from_quat(quat).as_matrix()

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = translation

        return T

    def undistort_point(self, u, v):
        return self.cam_model.rectifyPoint((u, v))

    def project_pixel_to_table(self, u, v):
        if not self.cam_info_ready:
            raise RuntimeError('CameraInfo not received yet')

        u_rect, v_rect = self.undistort_point(u, v)

        pixel = np.array([u_rect, v_rect, 1.0])
        ray_camera = self.K_inv @ pixel

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                Time()
            )
        except TransformException as ex:
            raise RuntimeError(
                f'Could not transform {self.camera_frame} to {self.target_frame}: {ex}'
            )

        T_base_camera = self.tf_matrix(tf)

        R_base_camera = T_base_camera[:3, :3]
        camera_origin_base = T_base_camera[:3, 3]
        ray_base = R_base_camera @ ray_camera

        if abs(ray_base[2]) < 1e-6:
            raise RuntimeError('Ray is parallel to table plane')

        d = (self.table_z - camera_origin_base[2]) / ray_base[2]

        if d <= 0:
            raise RuntimeError('Intersection is behind camera')

        point_base = camera_origin_base + d * ray_base

        return point_base

    def pixel_to_point_callback(self, request, response):
        try:
            point_3d = self.project_pixel_to_table(request.u, request.v)

            msg = PointStamped()
            msg.header.frame_id = self.target_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = float(point_3d[0])
            msg.point.y = float(point_3d[1])
            msg.point.z = float(point_3d[2])

            self.localized_pub.publish(msg)

            response.success = True
            response.message = 'Projected pixel to 3D table point'
            response.point = msg

            self.get_logger().debug(
                f'Pixel ({request.u:.1f}, {request.v:.1f}) -> '
                f'3D ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})'
            )

        except Exception as e:
            response.success = False
            response.message = str(e)
            response.point = PointStamped()

            self.get_logger().warn(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PixelToPointService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()