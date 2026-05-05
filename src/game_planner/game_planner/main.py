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

        self.declare_parameter('settle_time', 0.5)
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

        self.pixel_client = self.create_client(
            PixelToPoint,
            '/pixel_to_point'
        )

        self.top_left_sub = self.create_subscription(PointStamped, '/board_corner_tl_3d', self.top_left_callback, 10)
        self.top_right_sub = self.create_subscription(PointStamped, '/board_corner_tr_3d', self.top_right_callback, 10)
        self.tr = None
        self.tl = None

        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info('amongus.Connect4 main orchestrator started')

    def top_left_callback(self, msg):
        self.get_logger().debug(f'Received top-left corner: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})')
        self.tl = msg.point

    def top_right_callback(self, msg):
        self.get_logger().debug(f'Received top-right corner: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})')
        self.tr = msg.point

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
        
        self.eror

        self.call_robot_service(piece_position, board_position)

    def choose_ground_piece(self):
        """
        Find a ground piece of the robot's color and convert its pixel coordinates to 3D.
        """
        if self.latest_disc_data is None:
            return None

        target_color = 'red' if self.robot_color == 1 else 'yellow'

        for x, y, color in zip(
            self.latest_disc_data.x,
            self.latest_disc_data.y,
            self.latest_disc_data.color
        ):
            if color == target_color:
                self.get_logger().info(
                    f'Found {target_color} piece at pixel ({x:.1f}, {y:.1f}), converting to 3D'
                )

                # Call the pixel to point service
                if not self.pixel_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().error('PixelToPoint service not available')
                    return None

                req = PixelToPoint.Request()
                req.u = float(x)
                req.v = float(y)

                future = self.pixel_client.call_async(req)
                
                # Wait for result (this is blocking, but okay for now)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                
                if future.done() and future.result() is not None:
                    result = future.result()
                    if result.success:
                        p = Point()
                        p.x = result.point.point.x
                        p.y = result.point.point.y
                        p.z = result.point.point.z
                        self.get_logger().info(f'Converted to 3D: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
                        return p
                    else:
                        self.get_logger().warn(f'Pixel to 3D conversion failed: {result.message}')
                else:
                    self.get_logger().warn('Pixel to 3D service call failed or timed out')

        self.get_logger().warn(f'No {target_color} pieces found on ground')
        return None

    def column_to_board_position(self, col):
        """
        Calculate the 3D position for dropping a piece into the specified column.
        
        Assumes we have board corners: tl (top-left), tr (top-right)
        Interpolates along the top edge of the board for the column position.
        """

        if self.tl is None or self.tr is None:
            self.get_logger().error('Board corners not available for positioning')
            return None

        p = Point()

        # For 7 columns (0-6), there are 6 intervals
        # Interpolate x position along the top edge from left to right
        p.x = self.tl.x + col * (self.tr.x - self.tl.x) / 6.0
        
        # Interpolate y position along the top edge
        p.y = self.tl.y + col * (self.tr.y - self.tl.y) / 6.0
        
        # Add small offset downward for piece placement (adjust as needed)
        slot_spacing = 0.035
        p.y += slot_spacing
        
        p.z = 0.0  # Assume board is at z=0, adjust if needed
        
        self.get_logger().debug(f'Column {col} position: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
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