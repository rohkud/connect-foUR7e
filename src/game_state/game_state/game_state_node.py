"""
================================================================================
Game State Aggregator Node (game_state_node.py)
================================================================================

PURPOSE:
    Aggregates perception data from board_node and disc_detector to maintain
    a 6x7 game board state matrix. Performs perspective transformation
    from camera pixel coordinates to logical board grid indices.

CORE ALGORITHM - Perspective Transform:
    1. Receive 4 board corner pixel coordinates (top-left, top-right, bottom-right, bottom-left)
    2. Define canonical board in 700x600 pixel space (100 pixels per cell)
    3. Compute homography H via cv2.getPerspectiveTransform()
    4. For each detected disc:
       a. Apply homography: (tx, ty) = H * (px, py)
       b. Map to grid: col = int(tx/100), row = int(ty/100)
       c. Clamp indices to [0-6] and [0-5] respectively
       d. Flip row (image coords → board coords): row = 5 - row
       e. Write to board array: board[row][col] = color_code
================================================================================
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, MultiArrayDimension
from game_msgs.msg import GameBoard, DiscLoc2d
import numpy as np
import cv2


class GameStateNode(Node):
    def __init__(self):
        super().__init__('game_state')

        self.board = None
        self.disc_data = None
        self.homography = None

        self.board_sub = self.create_subscription(
            GameBoard,
            '/board_data',
            self.board_callback,
            5,   # smaller queue is better for perception
        )

        self.disc_sub = self.create_subscription(
            DiscLoc2d,
            '/disc_data',
            self.disc_callback,
            5,
        )

        self.board_pub = self.create_publisher(
            Int8MultiArray,
            '/game_state/board',
            5
        )

        self.get_logger().info('Game state node started')

    def board_callback(self, msg):
        self.board = msg
        # Compute homography from board corners to canonical 6x7 board
        self.get_logger().info('Received board corners, computing homography...')
        self.get_logger().info(f"Board corners: {list(zip(msg.corner_x, msg.corner_y))}")
        if len(msg.corner_x) == 4 and len(msg.corner_y) == 4:
            src_points = np.array([[msg.corner_x[i], msg.corner_y[i]] for i in range(4)], dtype=np.float32)
            # Assuming order: TL, TR, BR, BL
            dst_points = np.array([[0, 0], [700, 0], [700, 600], [0, 600]], dtype=np.float32)
            self.homography = cv2.getPerspectiveTransform(src_points, dst_points)
            self.get_logger().info('Computed homography from board corners')
        else:
            self.homography = None
            self.get_logger().warn('Invalid board corners, homography not computed')
        self.update_game_state()

    def disc_callback(self, msg):
        self.disc_data = msg
        self.update_game_state()

    def update_game_state(self):
        if self.board is None or self.disc_data is None or self.homography is None:
            self.get_logger().debug("Waiting for board, disc data, and homography...")
            return

        # 6 rows (height), 7 columns (width)
        board_array = [[0 for _ in range(7)] for _ in range(6)]

        for x, y, color in zip(self.disc_data.x, self.disc_data.y, self.disc_data.color):
            # Apply homography to transform to canonical board coordinates
            transformed = self.apply_homography(self.homography, x, y)
            if transformed is None:
                continue
            tx, ty = transformed

            # Check if within canonical board bounds
            if tx < 0 or tx >= 700 or ty < 0 or ty >= 600:
                self.get_logger().warn(f"Transformed point out of bounds: (tx, ty)=({tx:.1f}, {ty:.1f})")
                continue

            # Compute grid indices (cell size 100)
            col = int(tx / 100.0)
            row = int(ty / 100.0)

            # Clamp indices
            col = max(0, min(6, col))
            row = max(0, min(5, row))

            # Flip row (image origin top-left → board bottom-left)
            row = 5 - row

            # Normalize color
            color = str(color).lower()

            if color == 'red':
                board_array[row][col] = 1
            elif color == 'yellow':
                board_array[row][col] = 2

            # Debug mapping
            self.get_logger().debug(
                f"(x,y)=({x:.1f},{y:.1f}) → (tx,ty)=({tx:.1f},{ty:.1f}) → (row,col)=({row},{col}) color={color}"
            )

        board_array = board_array[::-1]

        # Print board nicely
        self.get_logger().info(f"\nCurrent board state:\n{np.array(board_array)}")

        # Publish as Int8MultiArray
        msg = Int8MultiArray()

        msg.layout.dim.append(
            MultiArrayDimension(label='rows', size=6, stride=42)
        )
        msg.layout.dim.append(
            MultiArrayDimension(label='cols', size=7, stride=7)
        )

        msg.data = [int(cell) for row in board_array for cell in row]

        self.board_pub.publish(msg)
        self.get_logger().info('Published game state board')

    def apply_homography(self, H, x, y):
        point = np.array([x, y, 1.0], dtype=np.float32)
        warped = H.dot(point)
        if warped[2] == 0:
            return None
        return float(warped[0] / warped[2]), float(warped[1] / warped[2])


def main(args=None):
    rclpy.init(args=args)
    node = GameStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()