"""
================================================================================
Main Planning & Control Node as Service
================================================================================
"""

from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R

from planning.ik import IKPlanner
from planning.srv import RunPlacement


class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp_service')

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            1
        )

        self.exec_ac = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.place_srv = self.create_service(
            RunPlacement,
            '/run_piece_placement',
            self.run_piece_placement_callback
        )

        self.joint_state = None
        self.ik_planner = IKPlanner()
        self.job_queue = []
        self.running = False

        self.get_logger().info("Cube grasp service ready: /run_piece_placement")

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def run_piece_placement_callback(self, request, response):
        if self.running:
            response.success = False
            response.message = "Robot is already running a placement job"
            return response

        if self.joint_state is None:
            response.success = False
            response.message = "No joint state received yet"
            return response

        self.running = True
        self.job_queue = []

        piece = request.piece_position
        board = request.board_position

        self.get_logger().info(
            f"Received placement request: "
            f"piece=({piece.x:.3f}, {piece.y:.3f}, {piece.z:.3f}), "
            f"board=({board.x:.3f}, {board.y:.3f}, {board.z:.3f})"
        )

        success = self.build_job_queue(piece, board)

        if not success:
            self.running = False
            response.success = False
            response.message = "Failed to build IK job queue"
            return response

        self.execute_jobs()

        response.success = True
        response.message = "Placement job started"
        return response

    def build_job_queue(self, piece, board):
        x = piece.x
        y = piece.y

        dx = 0.0
        dy = 0.0

        table_height = -0.28
        tool_height = 0.217
        tool_width = 0.06

        safe_z = 0.5
        grasp_z = table_height + tool_height

        try:
            # 1. Move above piece
            safe_position_job = self.ik_planner.compute_ik(
                self.joint_state,
                x + dx,
                y + dy,
                safe_z
            )
            if safe_position_job is None:
                return False
            self.job_queue.append(safe_position_job)

            # 2. Lower to grasp
            grasp_position_job = self.ik_planner.compute_ik(
                safe_position_job,
                x + dx,
                y + dy,
                grasp_z
            )
            if grasp_position_job is None:
                return False
            self.job_queue.append(grasp_position_job)

            # 3. Close gripper
            self.job_queue.append('toggle_grip')

            # 4. Lift piece
            post_position_job = self.ik_planner.compute_ik(
                grasp_position_job,
                x + dx,
                y + dy,
                safe_z
            )
            if post_position_job is None:
                return False
            self.job_queue.append(post_position_job)

            # 5. Move to neutral
            neutral_position_job = self.ik_planner.compute_ik(
                post_position_job,
                0.0,
                0.6,
                0.4
            )
            if neutral_position_job is None:
                return False
            self.job_queue.append(neutral_position_job)

            # 6. Rotate gripper
            side_down_quat = R.from_euler('z', 90, degrees=True) * R.from_quat(
                [0.0, 1.0, 0.0, 0.0]
            )

            side_down_quat = R.from_euler('y', -90, degrees=True) * side_down_quat

            qx, qy, qz, qw = side_down_quat.as_quat()

            rotate_job = self.ik_planner.compute_ik(
                neutral_position_job,
                0.0,
                0.6,
                0.35,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw
            )
            if rotate_job is None:
                return False
            self.job_queue.append(rotate_job)

            # 7. Move above board
            board_position_job = self.ik_planner.compute_ik(
                rotate_job,
                board.x,
                board.y + tool_width,
                board.z + tool_width,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw
            )
            if board_position_job is None:
                return False
            self.job_queue.append(board_position_job)

            # 8. Lower into board slot
            slot_position_job = self.ik_planner.compute_ik(
                board_position_job,
                board.x,
                board.y + tool_width,
                board.z,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw
            )
            if slot_position_job is None:
                return False
            self.job_queue.append(slot_position_job)

            # 9. Release piece
            self.job_queue.append('toggle_grip')

            # 10. Retreat
            retreat_job = self.ik_planner.compute_ik(
                slot_position_job,
                board.x,
                board.y + tool_width,
                board.z + tool_width,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw
            )
            if retreat_job is None:
                return False
            self.job_queue.append(retreat_job)

            return True

        except Exception as e:
            self.get_logger().error(f"Error while building job queue: {e}")
            return False

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            self.running = False
            return

        self.get_logger().info(
            f"Executing job queue, {len(self.job_queue)} jobs remaining."
        )

        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            traj = self.ik_planner.plan_to_joints(next_job)

            if traj is None:
                self.get_logger().error("Failed to plan to position")
                self.running = False
                return

            self.get_logger().info("Planned to position")
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()

        else:
            self.get_logger().error("Unknown job type")
            self.running = False

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Gripper service not available")
            self.running = False
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)

        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            self.get_logger().error("Gripper service call timed out")
            self.running = False
            return

        response = future.result()

        if response is None:
            self.get_logger().error("Gripper service returned no response")
            self.running = False
            return

        if response.success:
            self.get_logger().info(f"Gripper toggled: {response.message}")
            self.execute_jobs()
        else:
            self.get_logger().error(f"Gripper failed: {response.message}")
            self.running = False

    def _execute_joint_trajectory(self, joint_traj: JointTrajectory):
        self.get_logger().info("Waiting for controller action server...")
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info("Sending trajectory to controller...")
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            self.running = False
            return

        self.get_logger().info("Executing...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            future.result().result
            self.get_logger().info("Execution complete.")
            self.execute_jobs()

        except Exception as e:
            self.get_logger().error(f"Execution failed: {e}")
            self.running = False


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()