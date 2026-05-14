import rclpy
from control_msgs.action import FollowJointTrajectory
from planning_interfaces.srv import RunPlacement
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from planning.ik import IKPlanner


class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__("cube_grasp_service")

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 1
        )

        self.exec_ac = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )

        self.gripper_cli = self.create_client(Trigger, "/toggle_gripper")

        self.place_srv = self.create_service(
            RunPlacement, "/run_piece_placement", self.run_piece_placement_callback
        )

        self.robot_done_pub = self.create_publisher(Bool, "/robot_done", 10)

        self.joint_state = None
        self.ik_planner = IKPlanner()
        self.job_queue = []
        self.running = False

        self.get_logger().info("Cube grasp service ready: /run_piece_placement")

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def run_piece_placement_callback(self, request, response):
        self.get_logger().warn(
            f"RUN_PLACEMENT CALLED | "
            f"piece=({request.piece_position.x:.3f}, "
            f"{request.piece_position.y:.3f}, "
            f"{request.piece_position.z:.3f})"
        )

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

        table_height = -0.27
        tool_height = 0.225
        tool_width = 0.06

        safe_z = 0.5
        grasp_z = table_height + tool_height

        try:
            safe_position_job = self.ik_planner.compute_ik(
                self.joint_state, x, y, safe_z
            )
            if safe_position_job is None:
                self.get_logger().error("IK failed at safe_position_job")
                return False
            self.job_queue.append(safe_position_job)

            grasp_position_job = self.ik_planner.compute_ik(
                safe_position_job, x, y, grasp_z
            )
            if grasp_position_job is None:
                self.get_logger().error("IK failed at grasp_position_job")
                return False
            self.job_queue.append(grasp_position_job)

            self.job_queue.append("toggle_grip")

            post_position_job = self.ik_planner.compute_ik(
                grasp_position_job, x, y, safe_z
            )
            if post_position_job is None:
                self.get_logger().error("IK failed at post_position_job")
                return False
            self.job_queue.append(post_position_job)

            neutral_position_job = self.ik_planner.compute_ik(
                post_position_job, 0.0, 0.6, 0.4
            )
            if neutral_position_job is None:
                self.get_logger().error("IK failed at neutral_position_job")
                return False
            self.job_queue.append(neutral_position_job)

            side_down_quat = R.from_euler("z", 90, degrees=True) * R.from_quat(
                [0.0, 1.0, 0.0, 0.0]
            )
            side_down_quat = R.from_euler("y", -90, degrees=True) * side_down_quat
            qx, qy, qz, qw = side_down_quat.as_quat()

            rotate_job = self.ik_planner.compute_ik(
                neutral_position_job, 0.0, 0.4, 0.1, qx=qx, qy=qy, qz=qz, qw=qw
            )
            if rotate_job is None:
                self.get_logger().error("IK failed at rotate_job")
                return False
            self.job_queue.append(rotate_job)

            board_position_job = self.ik_planner.compute_ik(
                rotate_job,
                board.x,
                board.y + tool_width,
                board.z + tool_width,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
            )
            if board_position_job is None:
                self.get_logger().error("IK failed at board_position_job")
                return False
            self.job_queue.append(board_position_job)

            slot_position_job = self.ik_planner.compute_ik(
                board_position_job,
                board.x,
                board.y + tool_width,
                board.z + tool_width / 2.0,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
            )
            if slot_position_job is None:
                self.get_logger().error("IK failed at slot_position_job")
                return False
            self.job_queue.append(slot_position_job)

            self.job_queue.append("toggle_grip")

            retreat_job = self.ik_planner.compute_ik(
                slot_position_job,
                board.x,
                board.y + tool_width,
                board.z + tool_width,
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
            )
            if retreat_job is None:
                self.get_logger().error("IK failed at retreat_job")
                return False
            self.job_queue.append(retreat_job)

            reset_position_job = self.ik_planner.compute_ik(
                retreat_job, 0.0, 0.4, 0.1, qx=qx, qy=qy, qz=qz, qw=qw
            )
            if reset_position_job is None:
                self.get_logger().error("IK failed at reset_position_job")
                return False
            self.job_queue.append(reset_position_job)

            final_neutral_job = self.ik_planner.compute_ik(
                reset_position_job, 0.0, 0.6, 0.4
            )
            if final_neutral_job is None:
                self.get_logger().error("IK failed at final_neutral_job")
                return False
            self.job_queue.append(final_neutral_job)

            return True

        except Exception as e:
            self.get_logger().error(f"Error while building job queue: {e}")
            return False

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            self.running = False

            done_msg = Bool()
            done_msg.data = True
            self.robot_done_pub.publish(done_msg)

            self.get_logger().warn("Published /robot_done = True")
            return

        self.get_logger().info(
            f"Executing job queue, {len(self.job_queue)} jobs remaining."
        )

        next_job = self.job_queue.pop(0)

        self.get_logger().warn(
            f"NEXT JOB | type={type(next_job)} | remaining={len(self.job_queue)}"
        )

        if isinstance(next_job, JointState):
            self.get_logger().warn("PLANNING JOINT TRAJECTORY")

            traj = self.ik_planner.plan_to_joints(next_job)

            if traj is None:
                self.get_logger().error("Failed to plan to position")
                self.running = False
                return

            self.get_logger().info("Planned to position")
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif next_job == "toggle_grip":
            self.get_logger().warn("NEXT JOB: TOGGLE_GRIP")
            self._toggle_gripper()

        else:
            self.get_logger().error("Unknown job type")
            self.running = False

    def _toggle_gripper(self):
        self.get_logger().warn("CALLING GRIPPER SERVICE")

        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Gripper service not available")
            self.running = False
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)

        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

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
            self.get_logger().warn(f"GRIPPER DONE | message={response.message}")
            self.execute_jobs()
        else:
            self.get_logger().error(f"Gripper failed: {response.message}")
            self.running = False

    def _execute_joint_trajectory(self, joint_traj: JointTrajectory):
        self.get_logger().warn("WAITING FOR CONTROLLER ACTION SERVER")
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().warn("SENDING TRAJECTORY TO CONTROLLER")
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("TRAJECTORY GOAL REJECTED")
            self.running = False
            return

        self.get_logger().warn("TRAJECTORY ACCEPTED BY CONTROLLER")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            wrapped_result = future.result()
            result = wrapped_result.result
            status = wrapped_result.status

            self.get_logger().warn(
                f"TRAJECTORY DONE | action_status={status} | "
                f"error_code={result.error_code} | "
                f"error_string={result.error_string}"
            )

            if result.error_code != 0:
                self.get_logger().error("Trajectory did not finish successfully")
                self.running = False
                return

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


if __name__ == "__main__":
    main()

