#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int8MultiArray, Int8
from geometry_msgs.msg import Point, PointStamped
from game_msgs.msg import DiscLoc2d
from piece_localization_interfaces.srv import PixelToPoint

from planning_interfaces.srv import RunPlacement


class Connect4Main(Node):
    def __init__(self):
        super().__init__('connect4_main')

        self.declare_parameter('robot_color', 1)   # 1=red, 2=yellow
        self.declare_parameter('human_color', 2)   # 1=red, 2=yellow

        self.robot_color = self.get_parameter('robot_color').value
        self.human_color = self.get_parameter('human_color').value

        self.last_stable_board = None

        self.latest_move = None
        self.latest_disc_data = None

        self.robot_busy = False
        self.waiting_for_solver = False
        self.pixel_request_in_progress = False

        self.tl = None
        self.tr = None

        self.board_sub = self.create_subscription(
            Int8MultiArray,
            '/game_planner/stable_board',
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

        self.top_left_sub = self.create_subscription(
            PointStamped,
            '/board_corner_tl_3d',
            self.top_left_callback,
            10
        )

        self.top_right_sub = self.create_subscription(
            PointStamped,
            '/board_corner_tr_3d',
            self.top_right_callback,
            10
        )

        self.place_client = self.create_client(
            RunPlacement,
            '/run_piece_placement'
        )

        self.pixel_client = self.create_client(
            PixelToPoint,
            '/pixel_to_point'
        )

        self.get_logger().warn('amongus. Connect4 main orchestrator started')

    def top_left_callback(self, msg):
        self.tl = msg.point

    def top_right_callback(self, msg):
        self.tr = msg.point

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
            f'amongus. Human move detected at row={row}, col={col}. Waiting for solver.'
        )

        self.last_stable_board = current_board
        self.waiting_for_solver = True

        if self.latest_move is not None and not self.robot_busy:
            self.try_run_robot()

    def move_callback(self, msg):
        self.latest_move = msg.data
        self.get_logger().warn(f'amongus. Received solver move: column {self.latest_move}')

        if self.waiting_for_solver and not self.robot_busy:
            self.try_run_robot()

    def disc_callback(self, msg):
        self.latest_disc_data = msg

    def get_new_pieces(self, old_board, new_board):
        changes = []

        for i, (old, new) in enumerate(zip(old_board, new_board)):
            if old == 0 and new != 0:
                changes.append((i, new))

        return changes

    def try_run_robot(self):
        self.get_logger().warn('amongus. try_run_robot called')

        if self.robot_busy:
            self.get_logger().warn('amongus. Robot already busy')
            return

        if self.latest_move is None:
            self.get_logger().warn('amongus. No solver move available yet')
            return

        if self.latest_disc_data is None:
            self.get_logger().warn('amongus. No disc data available yet')
            return

        self.robot_busy = True
        self.pixel_request_in_progress = True

        success = self.choose_ground_piece_async()

        if not success:
            self.robot_busy = False
            self.pixel_request_in_progress = False

    def choose_ground_piece_async(self):
        target_color = 'red' if self.robot_color == 1 else 'yellow'

        for x, y, color in zip(
            self.latest_disc_data.x,
            self.latest_disc_data.y,
            self.latest_disc_data.color
        ):
            if color == target_color:
                self.get_logger().warn(
                    f'amongus. Found {target_color} piece at pixel ({x:.1f}, {y:.1f})'
                )

                if not self.pixel_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().error('amongus. PixelToPoint service not available')
                    return False

                req = PixelToPoint.Request()
                req.u = float(x)
                req.v = float(y)

                future = self.pixel_client.call_async(req)
                future.add_done_callback(self.pixel_to_point_done_callback)

                self.get_logger().warn('amongus. Sent pixel_to_point request')
                return True

        self.get_logger().warn(f'amongus. No {target_color} pieces found on ground')
        return False

    def pixel_to_point_done_callback(self, future):
        self.pixel_request_in_progress = False

        try:
            result = future.result()

            if result is None:
                self.get_logger().warn('amongus. PixelToPoint returned no result')
                self.robot_busy = False
                return

            if not result.success:
                self.get_logger().warn(
                    f'amongus. PixelToPoint failed: {result.message}'
                )
                self.robot_busy = False
                return

            piece_position = Point()
            # piece_position.x = result.point.point.x
            # piece_position.y = result.point.point.y
            # piece_position.z = result.point.point.z
            piece_position.x = .2
            piece_position.y = 0.6
            piece_position.z = 0.0

            self.get_logger().warn(
                f'amongus. Pixel converted to 3D: '
                f'({piece_position.x:.3f}, {piece_position.y:.3f}, {piece_position.z:.3f})'
            )

            board_position = self.column_to_board_position(self.latest_move)

            board_position.x = -0.2
            board_position.y = 0.6
            board_position.z = 0.0

            if board_position is None:
                self.get_logger().warn('amongus. Could not compute board placement position')
                self.robot_busy = False
                return

            self.call_robot_service(piece_position, board_position)

        except Exception as e:
            self.get_logger().error(f'amongus. Pixel service callback failed: {e}')
            self.robot_busy = False

    def column_to_board_position(self, col):
        if self.tl is None or self.tr is None:
            self.get_logger().error('amongus. Board corners not available for positioning')
            return None

        p = Point()

        # Interpolate along top edge from TL to TR
        p.x = self.tl.x + col * (self.tr.x - self.tl.x) / 6.0
        p.y = self.tl.y + col * (self.tr.y - self.tl.y) / 6.0

        # Small offset toward board/drop direction
        slot_spacing = 0.035
        p.y += slot_spacing

        # Adjust this if your board localizer gives a better z
        p.z = 0.0

        self.get_logger().warn(
            f'amongus. Board position for col {col}: '
            f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
        )

        return p

    def call_robot_service(self, piece_position, board_position):
        if not self.place_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('amongus. /run_piece_placement service not available')
            self.robot_busy = False
            return

        req = RunPlacement.Request()
        req.piece_position = piece_position
        req.board_position = board_position

        self.waiting_for_solver = False

        self.get_logger().warn(
            f'amongus. Calling robot service: '
            f'piece=({piece_position.x:.3f}, {piece_position.y:.3f}, {piece_position.z:.3f}), '
            f'board=({board_position.x:.3f}, {board_position.y:.3f}, {board_position.z:.3f})'
        )

        future = self.place_client.call_async(req)
        future.add_done_callback(self.robot_done_callback)

    def robot_done_callback(self, future):
        try:
            result = future.result()

            if result is None:
                self.get_logger().error('amongus. Robot service returned no result')
            elif result.success:
                self.get_logger().warn(f'amongus. Robot placement started: {result.message}')
            else:
                self.get_logger().error(f'amongus. Robot placement failed: {result.message}')

        except Exception as e:
            self.get_logger().error(f'amongus. Robot service call failed: {e}')

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