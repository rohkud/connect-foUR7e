"""
================================================================================
Main Planning & Control Node (main.py)
================================================================================

PURPOSE:
    Orchestrates robot manipulation for Connect Four disc placement. Converts
    high-level game moves (column index) into physical robot actions via
    inverse kinematics and motion planning. Implements a job queue for
    sequential execution of grasp, move, and release actions.
================================================================================
"""

# ROS Libraries
from std_srvs.srv import Trigger
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

from planning.ik import IKPlanner

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.localized_piece_sub = self.create_subscription(
            PointStamped,
            '/localized_pieces',
            self.localized_piece_callback,
            10
        )

        self.tl_board_sub = self.create_subscription(
            PointStamped,
            '/board_corner_tr_3d',
            self.tr_board_callback,
            10
        )

        self.tr = None

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.cube_pose = None
        self.current_plan = None
        self.joint_state = None

        self.ik_planner = IKPlanner()
        self.goal = None

        self.job_queue = [] # Entries should be of type either JointState or String('toggle_grip')
        while self.joint_state is None:
            self.get_logger().info("Waiting for initial joint state...")
            rclpy.spin_once(self, timeout_sec=1.0)

        while self.goal is None:
            self.get_logger().info("Waiting for goal state...")
            rclpy.spin_once(self, timeout_sec=1.0)

        while self.tr is None:
            self.get_logger().info("Waiting for tr state...")
            rclpy.spin_once(self, timeout_sec=1.0)

        self.cube_callback()

    def tr_board_callback(self, msg: PointStamped):
        self.get_logger().info(f"Received board corner position: {msg.point}")
        self.tr = msg.point

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def localized_piece_callback(self, msg: PointStamped):
        if self.goal is not None:
            return
    
        self.goal = msg.point
        self.get_logger().info(f"Received localized piece position: {msg.point}")   

    def cube_callback(self):
        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")

        # -----------------------------------------------------------
        # TODO: In the following section you will add joint angles to the job queue. 
        # Entries of the job queue should be of type either JointState or String('toggle_grip')
        # Think about you will leverage the IK planner to get joint configurations for the cube grasping task.
        # To understand how the queue works, refer to the execute_jobs() function below.
        # -----------------------------------------------------------

        # 1) Move to Pre-Grasp Position (gripper above the cube)
        '''
        Use the following offsets for pre-grasp position:
        z offset: +0.185 (to be above the cube by accounting for gripper length)
        '''
        x = self.goal.x
        y = self.goal.y
        # -------------------Baymax OFFSETS---------------
        # dx = 0.055
        # dy = 0.0
        # ------------------------------------------------
        dx = -.055
        dy = 0.0
        # x = 0.2
        # y = 0.16

        table_height = -0.28
        tool_height = 0.217
        tool_width = 0.06
        safe_position_job = self.ik_planner.compute_ik(self.joint_state,
                                            x + dx,
                                            y + dy,
                                            0.5)
        self.job_queue.append(safe_position_job)

        grasp_position_job = self.ik_planner.compute_ik(safe_position_job,
                                    x + dx,
                                    y + dy,
                                    table_height + tool_height) # -0.058
        self.job_queue.append(grasp_position_job)

        self.job_queue.append('toggle_grip')

        post_position_job = self.ik_planner.compute_ik(grasp_position_job,
                                    x + dx,
                                    y + dy,
                                    0.5)
        self.job_queue.append(post_position_job)

        neutral_position_job = self.ik_planner.compute_ik(post_position_job,
                                    0.0,
                                    .6,
                                    0.4)
        self.job_queue.append(neutral_position_job)
        
        # Rotate the gripper 90 degrees to the side before placing the piece.
        side_down_quat = R.from_euler('z', 90, degrees=True) * R.from_quat([0.0, 1.0, 0.0, 0.0])

        #------------------------------- If baymax, use------------------------- 
        # side_down_quat = R.from_euler('y', -90, degrees=True) * side_down_quat
        #-----------------------------------------------------------------------
        side_down_quat = R.from_euler('y', -90, degrees=True) * side_down_quat

        qx, qy, qz, qw = side_down_quat.as_quat()
        rotate_job = self.ik_planner.compute_ik(neutral_position_job,
                                            0.0,
                                            .6,
                                            0.35, qx=qx, qy=qy, qz=qz, qw=qw)
        self.job_queue.append(rotate_job)

        x = self.tr.x
        y = self.tr.y
        z = self.tr.z
        board_position_job = self.ik_planner.compute_ik(rotate_job,
                                            # x + dx,
                                            0.05,
                                            y + tool_width,
                                            z + tool_width, qx=qx, qy=qy, qz=qz, qw=qw) # -0.05

        self.job_queue.append(board_position_job)

        slot_position_job = self.ik_planner.compute_ik(board_position_job,
                                    # x + dx,
                                    0.05,
                                    y + tool_width,
                                    z, qx=qx, qy=qy, qz=qz, qw=qw) # -0.05

        self.job_queue.append(slot_position_job)

        self.job_queue.append('toggle_grip')


        self.execute_jobs()


    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):

            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                return

            self.get_logger().info("Planned to position")

            self._execute_joint_trajectory(traj.joint_trajectory)
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next job

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        if not future.done():
            self.get_logger().error("Gripper service call timed out")
            rclpy.shutdown()
            return

        response = future.result()

        if response is None:
            self.get_logger().error("Gripper service returned no response")
            rclpy.shutdown()
            return

        if response.success:
            self.get_logger().info(f"Gripper activated: {response.message}")
        else:
            self.get_logger().error(f"Gripper failed: {response.message}")
            rclpy.shutdown()
            return

        self.execute_jobs()

            
    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
