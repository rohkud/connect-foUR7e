import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from rcl_interfaces.msg import SetParametersResult

import struct

def rgb_float_to_tuple(rgb_float):
    rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
    r = (rgb_int >> 16) & 0xFF
    g = (rgb_int >> 8) & 0xFF
    b = rgb_int & 0xFF
    return r, g, b

def is_yellow(r, g, b):
    return r > 200 and g < 80 and b < 80

class RealSensePCSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')
        self.target_frame = self.declare_parameter('target_frame', 'base_link').value
        self.max_y = float(self.declare_parameter('max_y', 0.79).value)

        self.min_z = float(self.declare_parameter('min_z', -0.18).value)
        self.max_z = float(self.declare_parameter('max_z', -0.15).value)
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # Publishers
        self.cube_pose_pub = self.create_publisher(PointStamped, '/cube_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info("Subscribed to PointCloud2 topic and marker publisher ready")

    def pointcloud_callback(self, msg: PointCloud2):

        # Transform the pointcloud from its original frame to base_link
        # Lookup Transform and use library function to transform cloud

        # Filter points between z coords between min_z and max_z and max_y
        # Call the numpy array filtered_points

        source_frame = "camera_depth_optical_frame" # TODO: Fill in the source frame based on what you implemented in your static TF broadcaster 
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, source_frame, Time()) # TODO: the entire tf lookup params should be filled in
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {self.target_frame}: {ex}')
            return

        transformed_cloud = do_transform_cloud(msg, tf) # TODO: look what do_transform_cloud takes in and outputs

        raw_points = pc2.read_points(
            transformed_cloud,
            field_names=('x', 'y', 'z', 'rgb'),
            skip_nans=True,
        )
        # print("Raw points [rgb]:", raw_points["rgb"])
        colors = [rgb_float_to_tuple(rgb) for rgb in raw_points["rgb"]]

        points_base = np.column_stack(
            (raw_points['x'], raw_points['y'], raw_points['z'], raw_points['rgb'])
        ).astype(np.float32, copy=False)

        yellow_mask = np.array([is_yellow(r, g, b) for r, g, b in colors])
        yellow_points = points_base[yellow_mask]

        # points_base = np.column_stack(
        #         (raw_points['x'], raw_points['y'], raw_points['z'])
        #     ).astype(np.float32, copy=False)

        # # TODO: Create masks based on the specified min, max y and z parameters above in order to filter points
        # filtered_points = points_base[
        #     (points_base[:, 2] >= self.min_z) &
        #     (points_base[:, 2] <= self.max_z) &
        #     (points_base[:, 1] <= self.max_y)
        # ]

        # if filtered_points.size == 0:
        #     self.get_logger().warn(
        #         f'No points after filters: z in [{self.min_z:.3f}, {self.max_z:.3f}] m, y <= {self.max_y:.3f} m'
        #     )
        #     return

        # filtered_cloud = pc2.create_cloud_xyz32(
        #     transformed_cloud.header,
        #     yellow_points.tolist(),
        # )
        # self.filtered_points_pub.publish(filtered_cloud)

        # TODO: Compute cube position in base_link frame using filtered_points.
        cube_x = np.mean(yellow_points[:, 0])
        cube_y = np.mean(yellow_points[:, 1])
        cube_z = np.mean(yellow_points[:, 2])

        # TODO: Publish the cube pose message with the cube position information
        cube_pose = PointStamped()
        cube_pose.header.frame_id = self.target_frame
        cube_pose.header.stamp = self.get_clock().now().to_msg()
        cube_pose.point.x = float(cube_x)
        cube_pose.point.y = float(cube_y)
        cube_pose.point.z = float(cube_z)

        self.cube_pose_pub.publish(cube_pose)

    def cluster_points(self, points, cluster_distance=0.02):
        """Simple distance-based clustering to separate individual discs"""
        if len(points) == 0:
            return []

        clusters = []
        visited = set()

        for i, point in enumerate(points):
            if i in visited:
                continue

            # Start new cluster
            cluster = [point]
            visited.add(i)
            queue = [i]

            # Grow cluster by finding nearby points
            while queue:
                current_idx = queue.pop(0)
                current_point = points[current_idx]

                for j, other_point in enumerate(points):
                    if j in visited:
                        continue

                    # Check Euclidean distance
                    distance = np.linalg.norm(current_point - other_point)
                    if distance <= cluster_distance:
                        cluster.append(other_point)
                        visited.add(j)
                        queue.append(j)

            clusters.append(cluster)

        return clusters

    def _on_parameter_update(self, params):

        for param in params:
            if param.name == 'min_z' and param.type_ == Parameter.Type.DOUBLE:
                new_min_z = float(param.value)
            elif param.name == 'max_z' and param.type_ == Parameter.Type.DOUBLE:
                new_max_z = float(param.value)

        if new_min_z > new_max_z:
            return SetParametersResult(
                successful=False,
                reason='min_z must be <= max_z',
            )

        self.min_z = new_min_z
        self.max_z = new_max_z
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
