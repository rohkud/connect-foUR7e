#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8MultiArray, Int8
from geometry_msgs.msg import Point
from game_msgs.msg import DiscLoc2d

from planning_interfaces.srv import RunPlacement


class Connect4Main(Node):
    def __init__(self):
        super().__init__('connect4_main')

        self.declare_parameter('settle_time', 0.3)
        self.declare_parameter('robot_color', 1)   # 1=red, 2=yellow
        self.declare_parameter('human_color', 2)   # 1=red, 2=yellow

        self.settle_time = self.get_parameter('settle_time').value
        self.robot_color = self.get_parameter('robot_color').value
        self.human_color = self.get_parameter('human_color').value

        self.last_seen_board = None
        self.last_stable_board = None
        self.pending_board = None
        self.last_change_time = None

        self.latest_move = None
        self.latest_disc_data = None

        self.robot_busy = False
        self.waiting_for_solver = False

        self.board_sub = self.create_subscription(
            Int8MultiArray,
            '/game_solver/board',
            self.board_callback,
            10
        )

        self.move_sub = self.create_subscription(
            Int8,
            '/game_planner/move',
            self.move_callback,
            10
        )

        self.disc_sub = self.create_subscription(
            DiscLoc2d,
            '/disc_data',
            self.disc_callback,
            10
        )

        self.place_client = self.create_client(
            RunPlacement,
            '/run_piece_placement'
        )

        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info('amongus.Connect4 main orchestrator started')

    def board_callback(self, msg):
        if len(msg.data) != 42:
            return

        board = tuple(msg.data)

        if self.last_seen_board != board:
            self.last_seen_board = board
            self.last_change_time = self.get_clock().now()
            self.pending_board = board

    def move_callback(self, msg):
        self.latest_move = msg.data
        self.get_logger().info(f'Received solver move: column {self.latest_move}')

        if self.waiting_for_solver and not self.robot_busy:
            self.try_run_robot()

    def disc_callback(self, msg):
        self.latest_disc_data = msg

    def timer_callback(self):
        if self.robot_busy:
            return

        if self.pending_board is None or self.last_change_time is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_change_time).nanoseconds / 1e9

        if elapsed < self.settle_time:
            self.get_logger().warn(
                f'amongus. Waiting for board to settle... {elapsed:.2f}s elapsed'
            )
            return

        stable_board = self.pending_board

        if self.last_stable_board is None:
            self.last_stable_board = stable_board
            return

        if stable_board == self.last_stable_board:
            return

        changes = self.get_new_pieces(self.last_stable_board, stable_board)

        if len(changes) != 1:
            self.get_logger().warn(
                f'amongus. Ignoring board change because {len(changes)} new pieces were detected'
            )
            self.last_stable_board = stable_board
            return

        index, color = changes[0]

        if color != self.human_color:
            self.get_logger().warn('amongus. Board changed, but it was not the human color')
            self.last_stable_board = stable_board
            return

        row = index // 7
        col = index % 7

        self.get_logger().warn(
            f'amongus. Human move detected at row={row}, col={col}. Waiting for solver.'
        )

        self.last_stable_board = stable_board
        self.waiting_for_solver = True

        if self.latest_move is not None:
            self.try_run_robot()

    def get_new_pieces(self, old_board, new_board):
        changes = []

        for i, (old, new) in enumerate(zip(old_board, new_board)):
            if old == 0 and new != 0:
                changes.append((i, new))

        return changes

    def try_run_robot(self):
        if self.latest_move is None:
            self.get_logger().warn('No solver move available yet')
            return

        if self.latest_disc_data is None:
            self.get_logger().warn('No disc data available yet')
            return

        piece_position = self.choose_ground_piece()
        if piece_position is None:
            self.get_logger().warn('Could not find a ground piece to pick up')
            return

        board_position = self.column_to_board_position(self.latest_move)
        if board_position is None:
            self.get_logger().warn('Could not compute board placement position')
            return

        self.call_robot_service(piece_position, board_position)

    def choose_ground_piece(self):
        """
        Simple version:
        pick the first detected disc of the robot's color.

        NOTE:
        This assumes /disc_data gives pixel positions, not 3D positions.
        If your robot service needs 3D base_link points, replace this with
        your localized piece topic or 2D-to-3D localizer output.
        """

        target_color = 'red' if self.robot_color == 1 else 'yellow'

        for x, y, color in zip(
            self.latest_disc_data.x,
            self.latest_disc_data.y,
            self.latest_disc_data.color
        ):
            if color == target_color:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                return p

        return None

    def column_to_board_position(self, col):
        """
        Placeholder board position.

        Replace this with your real board localizer / slot localization logic.
        For now, this maps column number to an approximate physical x offset.
        """

        p = Point()

        base_x = 0.05
        base_y = 0.60
        base_z = 0.04
        slot_spacing = 0.035

        p.x = base_x + (col - 3) * slot_spacing
        p.y = base_y
        p.z = base_z

        return p

    def call_robot_service(self, piece_position, board_position):
        if not self.place_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/run_piece_placement service not available')
            return

        req = RunPlacement.Request()
        req.piece_position = piece_position
        req.board_position = board_position

        self.robot_busy = True
        self.waiting_for_solver = False

        self.get_logger().info(
            f'Calling robot service: '
            f'piece=({piece_position.x:.3f}, {piece_position.y:.3f}, {piece_position.z:.3f}), '
            f'board=({board_position.x:.3f}, {board_position.y:.3f}, {board_position.z:.3f})'
        )

        future = self.place_client.call_async(req)
        future.add_done_callback(self.robot_done_callback)

    def robot_done_callback(self, future):
        try:
            result = future.result()

            if result.success:
                self.get_logger().info(f'Robot placement started: {result.message}')
            else:
                self.get_logger().error(f'Robot placement failed: {result.message}')

        except Exception as e:
            self.get_logger().error(f'Robot service call failed: {e}')

        self.robot_busy = False
        self.latest_move = None


def main(args=None):
    rclpy.init(args=args)
    node = Connect4Main()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()