import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, MultiArrayDimension
from game_msgs.msg import GameBoard, DiscLoc2d, HomographyMatrix
import numpy as np


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

        self.homography_sub = self.create_subscription(
            HomographyMatrix,
            '/board_homography',
            self.homography_callback,
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
        self.update_game_state()

    def disc_callback(self, msg):
        self.disc_data = msg
        self.update_game_state()

    def homography_callback(self, msg):
        matrix = np.array(msg.data, dtype=np.float32)
        if matrix.size != 9:
            self.get_logger().warn('Received invalid homography matrix size')
            return
        self.homography = matrix.reshape((3, 3))
        self.update_game_state()

    def update_game_state(self):
        if self.board is None or self.disc_data is None:
            self.get_logger().debug("Waiting for both board and disc data...")
            return

        # 6 rows (height), 7 columns (width)
        board_array = [[0 for _ in range(7)] for _ in range(6)]

        x0 = self.board.x
        y0 = self.board.y
        w = self.board.w
        h = self.board.h

        if w <= 0 or h <= 0:
            self.get_logger().warn('Invalid board bounding box')
            return

        if self.homography is not None:
            target_w = float(w)
            target_h = float(h)
            cell_w = target_w / 7.0
            cell_h = target_h / 6.0

            for x, y, color in zip(self.disc_data.x, self.disc_data.y, self.disc_data.color):
                rel_x = x - x0
                rel_y = y - y0
                transformed = self.apply_homography(self.homography, rel_x, rel_y)
                if transformed is None:
                    continue
                tx, ty = transformed

                if tx < 0 or tx > target_w or ty < 0 or ty > target_h:
                    continue

                col = int(tx / cell_w)
                row = int(ty / cell_h)
                col = max(0, min(6, col))
                row = max(0, min(5, row))
                row = 5 - row

                color = str(color).lower()
                if color == 'red':
                    board_array[row][col] = 1
                elif color == 'yellow':
                    board_array[row][col] = 2

                self.get_logger().debug(
                    f"(x,y)=({x:.1f},{y:.1f}) rel=({rel_x:.1f},{rel_y:.1f}) → (tx,ty)=({tx:.1f},{ty:.1f}) → (row,col)=({row},{col}) color={color}"
                )
        else:
            cell_w = w / 7.0
            cell_h = h / 6.0

            for x, y, color in zip(self.disc_data.x, self.disc_data.y, self.disc_data.color):
                if x < x0 or x > x0 + w or y < y0 or y > y0 + h:
                    continue

                col = int((x - x0) / cell_w)
                row = int((y - y0) / cell_h)
                col = max(0, min(6, col))
                row = max(0, min(5, row))
                row = 5 - row

                color = str(color).lower()
                if color == 'red':
                    board_array[row][col] = 1
                elif color == 'yellow':
                    board_array[row][col] = 2

                self.get_logger().debug(
                    f"(x,y)=({x:.1f},{y:.1f}) → (row,col)=({row},{col}) color={color}"
                )

        # Print board nicely
        print("\nCurrent board state:")
        print(np.array(board_array))

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