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

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.current_plan = None
        self.joint_state = None

        self.ik_planner = IKPlanner()
        self.goal = None

        self.job_queue = [] # Entries should be of type either JointState or String('toggle_grip')
        while self.joint_state is None:
            self.get_logger().info("Waiting for initial joint state...")
            rclpy.spin_once(self, timeout_sec=1.0)

        self.cube_callback()

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

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
        x = -0.2
        y = 0.6

        board_x = 0.1
        board_center_y = 0.5
        board_width = 0.21
        board_min_y = board_center_y - board_width / 2.0
        board_max_y = board_center_y + board_width / 2.0

        board_height = 0.30
        table_z = -0.275
        board_z = board_height + table_z

        slot = 6
        test_slot = True
        board_slot_test = np.array([
            board_x,
            board_min_y + (slot / 6.0) * (board_max_y - board_min_y),
            board_z
        ])


        board_tl = np.array([board_x, board_center_y - board_width / 2.0, board_z])
        board_tr = np.array([board_x, board_center_y + board_width / 2.0, board_z])

        table_height = -0.28
        tool_height = 0.225
        tool_width = 0.06
        safe_position_job = self.ik_planner.compute_ik(self.joint_state,
                                            x,
                                            y,
                                            0.5)
        self.job_queue.append(safe_position_job)

        grasp_position_job = self.ik_planner.compute_ik(safe_position_job,
                                    x,
                                    y,
                                    table_height + tool_height) # -0.058
        self.job_queue.append(grasp_position_job)

        self.job_queue.append('toggle_grip')

        post_position_job = self.ik_planner.compute_ik(grasp_position_job,
                                    x,
                                    y,
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

        board_position_job = self.ik_planner.compute_ik(rotate_job,
                                            board_tl[0],
                                            board_tl[1] + tool_width,
                                            board_tl[2] + tool_width, qx=qx, qy=qy, qz=qz, qw=qw)

        self.job_queue.append(board_position_job)

        slot_position_job = None
        if test_slot:
            slot_position_job = self.ik_planner.compute_ik(board_position_job,
                                        board_slot_test[0],
                                        board_slot_test[1] + tool_width,
                                        board_slot_test[2] + tool_width / 2.0, qx=qx, qy=qy, qz=qz, qw=qw)
            self.job_queue.append(slot_position_job)

            self.job_queue.append('toggle_grip')
        else:

            slot_position_job = self.ik_planner.compute_ik(board_position_job,
                                        board_tl[0],
                                        board_tl[1] + tool_width,
                                        board_tl[2] + tool_width / 2.0, qx=qx, qy=qy, qz=qz, qw=qw)

            self.job_queue.append(slot_position_job)

            slot_position_job = self.ik_planner.compute_ik(slot_position_job,
                                        board_tr[0],
                                        board_tr[1] + tool_width,
                                        board_tr[2] + tool_width / 2.0, qx=qx, qy=qy, qz=qz, qw=qw)
            self.job_queue.append(slot_position_job)

        reset_position_job = self.ik_planner.compute_ik(slot_position_job,
                                    0.0,
                                    .6,
                                    0.35, qx=qx, qy=qy, qz=qz, qw=qw)
        self.job_queue.append(reset_position_job)

        neutral_position_job = self.ik_planner.compute_ik(reset_position_job,
                                    0.0,
                                    .6,
                                    0.4)
        self.job_queue.append(neutral_position_job)

        if not test_slot:
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

            traj = self.ik_planner.plan_to_joints(self.joint_state, next_job)
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
        # wait for 2 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

            
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
