#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8MultiArray, Bool
from geometry_msgs.msg import Point

from planning_interfaces.srv import RunPlacement, SolveMove


class Connect4Main(Node):
    def __init__(self):
        super().__init__('connect4_main')

        self.declare_parameter('robot_color', 1)   # 1=red, 2=yellow
        self.declare_parameter('human_color', 2)   # 1=red, 2=yellow

        self.robot_color = self.get_parameter('robot_color').value
        self.human_color = self.get_parameter('human_color').value

        self.last_stable_board = None
        self.latest_move = None

        self.robot_busy = False
        self.solver_busy = False

        self.board_sub = self.create_subscription(
            Int8MultiArray,
            '/game_planner/stable_board',
            self.board_callback,
            10
        )

        self.robot_done_sub = self.create_subscription(
            Bool,
            '/robot_done',
            self.robot_done_topic_callback,
            10
        )

        self.solve_client = self.create_client(
            SolveMove,
            '/solve_move'
        )

        self.place_client = self.create_client(
            RunPlacement,
            '/run_piece_placement'
        )

        while not self.solve_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /solve_move service...')

        while not self.place_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /run_piece_placement service...')

        self.get_logger().warn('amongus. Connect4 main orchestrator started')

    def robot_done_topic_callback(self, msg):
        if not msg.data:
            return

        self.get_logger().warn('amongus. Robot truly done, accepting new moves')
        self.robot_busy = False
        self.latest_move = None

    def board_callback(self, msg):
        if len(msg.data) != 42:
            return

        current_board = tuple(msg.data)

        if self.last_stable_board is None:
            self.last_stable_board = current_board
            self.get_logger().warn('amongus. Initial board saved')
            return

        if current_board == self.last_stable_board:
            return

        if self.robot_busy:
            self.get_logger().warn('amongus. Board changed but robot is busy; ignoring')
            self.last_stable_board = current_board
            return

        changes = self.get_new_pieces(self.last_stable_board, current_board)

        if len(changes) != 1:
            self.get_logger().warn(
                f'amongus. Ignoring board change because {len(changes)} new pieces were detected'
            )
            self.last_stable_board = current_board
            return

        index, color = changes[0]

        if color != self.human_color:
            self.get_logger().warn('amongus. Board changed, but it was not the human color')
            self.last_stable_board = current_board
            return

        row = index // 7
        col = index % 7

        self.get_logger().warn(
            f'amongus. Human move detected at row={row}, col={col}. Calling solver.'
        )

        self.last_stable_board = current_board
        self.request_solver_move(current_board)

    def get_new_pieces(self, old_board, new_board):
        changes = []

        for i, (old, new) in enumerate(zip(old_board, new_board)):
            if old == 0 and new != 0:
                changes.append((i, new))

        return changes

    def request_solver_move(self, board):
        if self.robot_busy:
            self.get_logger().warn('amongus. Robot busy, not solving')
            return

        if self.solver_busy:
            self.get_logger().warn('amongus. Solver already busy')
            return

        req = SolveMove.Request()
        req.board = list(board)
        req.player = int(self.robot_color)

        self.solver_busy = True

        self.get_logger().warn('amongus. Calling /solve_move service')

        future = self.solve_client.call_async(req)
        future.add_done_callback(self.solve_done_callback)

    def solve_done_callback(self, future):
        self.solver_busy = False

        try:
            result = future.result()

            if result is None:
                self.get_logger().error('amongus. Solver returned no result')
                return

            if not result.success:
                self.get_logger().error(f'amongus. Solver failed: {result.message}')
                return

            self.latest_move = int(result.column)

            self.get_logger().warn(
                f'amongus. Solver chose column {self.latest_move}'
            )

            self.try_run_robot()

        except Exception as e:
            self.get_logger().error(f'amongus. Solver service failed: {e}')

    def try_run_robot(self):
        self.get_logger().warn('amongus. try_run_robot called')

        if self.robot_busy:
            self.get_logger().warn('amongus. Robot already busy')
            return

        if self.latest_move is None:
            self.get_logger().warn('amongus. No solver move available')
            return

        self.robot_busy = True

        piece_position = Point()
        piece_position.x = -0.2
        piece_position.y = 0.6
        piece_position.z = 0.0

        board_position = self.get_hardcoded_board_position(self.latest_move)

        self.call_robot_service(piece_position, board_position)

    def get_hardcoded_board_position(self, col):
        board_x = 0.1
        board_center_y = 0.5
        board_width = 0.22
        board_min_y = board_center_y - board_width / 2.0
        board_max_y = board_center_y + board_width / 2.0

        board_height = 0.30
        table_z = -0.28
        board_z = board_height + table_z

        p = Point()
        p.x = board_x
        p.y = board_min_y + (float(col) / 6.0) * (board_max_y - board_min_y)
        p.z = board_z

        self.get_logger().warn(
            f'amongus. Hardcoded board position for col {col}: '
            f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
        )

        return p

    def call_robot_service(self, piece_position, board_position):
        req = RunPlacement.Request()
        req.piece_position = piece_position
        req.board_position = board_position

        self.get_logger().warn(
            f'amongus. Calling robot service: '
            f'piece=({piece_position.x:.3f}, {piece_position.y:.3f}, {piece_position.z:.3f}), '
            f'board=({board_position.x:.3f}, {board_position.y:.3f}, {board_position.z:.3f})'
        )

        future = self.place_client.call_async(req)
        future.add_done_callback(self.robot_service_done_callback)

    def robot_service_done_callback(self, future):
        try:
            result = future.result()

            if result is None:
                self.get_logger().error('amongus. Robot service returned no result')
                self.robot_busy = False
                return

            if result.success:
                self.get_logger().warn(
                    f'amongus. Robot placement started: {result.message}'
                )
                self.get_logger().warn(
                    'amongus. Waiting for /robot_done before accepting new moves'
                )
            else:
                self.get_logger().error(
                    f'amongus. Robot placement failed: {result.message}'
                )
                self.robot_busy = False
                return

        except Exception as e:
            self.get_logger().error(f'amongus. Robot service call failed: {e}')
            self.robot_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = Connect4Main()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()