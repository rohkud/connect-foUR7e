#!/usr/bin/env python3
"""
Simple arm movement to hardcoded point
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoveToPoint(Node):
    def __init__(self):
        super().__init__('move_to_point')

        self.current_joint_state = None
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            1
        )

        # MoveIt IK client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for /compute_ik service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        # Trajectory publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        self.get_logger().info("Move to Point Node initialized")

    def joint_state_callback(self, msg):
        """Store current joint state"""
        self.current_joint_state = msg

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        """
        Compute IK for a given workspace pose using MoveIt.
        """
        if self.current_joint_state is None:
            self.get_logger().error("No joint state available")
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = 'ur_manipulator'
        ik_req.ik_request.robot_state.joint_state = self.current_joint_state
        ik_req.ik_request.ik_link_name = 'wrist_3_link'
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.timeout = Duration(sec=2)
        ik_req.ik_request.avoid_collisions = True

        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service failed.')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val}')
            return None

        return result.solution.joint_state

    def execute_move(self, x, y, z):
        """Execute movement to target point"""
        self.get_logger().info(f"Computing IK for target: ({x}, {y}, {z})")

        joint_solution = self.compute_ik(x, y, z)

        if joint_solution is None:
            self.get_logger().error("IK computation failed!")
            return False

        # Create trajectory
        joint_traj = JointTrajectory()
        joint_traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        point = JointTrajectoryPoint()

        # Reorder joints from IK solution
        ik_joint_dict = {}
        for i, name in enumerate(joint_solution.name):
            if i < 6:
                ik_joint_dict[name] = joint_solution.position[i]

        point.positions = [
            ik_joint_dict['shoulder_pan_joint'],
            ik_joint_dict['shoulder_lift_joint'],
            ik_joint_dict['elbow_joint'],
            ik_joint_dict['wrist_1_joint'],
            ik_joint_dict['wrist_2_joint'],
            ik_joint_dict['wrist_3_joint']
        ]

        point.velocities = [0.0] * 6
        point.time_from_start.sec = 3
        point.time_from_start.nanosec = 0

        joint_traj.points.append(point)

        self.get_logger().info("Publishing trajectory...")
        self.traj_pub.publish(joint_traj)
        
        # Wait for execution
        time.sleep(4)
        self.get_logger().info("Movement complete!")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = MoveToPoint()

    # ===== HARDCODED TARGET POINT - CHANGE THESE VALUES =====
    target_x = 0.073
    target_y = 0.635
    target_z = 0.2

    # ========================================================

    node.get_logger().info(f"Target position: ({target_x}, {target_y}, {target_z})")
    success = node.execute_move(target_x, target_y, target_z)

    if not success:
        node.get_logger().error("Movement failed!")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()