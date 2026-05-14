import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from game_msgs.msg import DiscLoc2d, GameBoard
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray, MultiArrayDimension


class BoardStateNode(Node):
    def __init__(self):
        super().__init__("board_state")

        self.bridge = CvBridge()
        self.board = None
        self.last_board = None
        self.disc_data = None
        self.homography = None
        self.latest_image = None

        self.board_sub = self.create_subscription(
            GameBoard,
            "/board_data",
            self.board_callback,
            5,
        )

        self.disc_sub = self.create_subscription(
            DiscLoc2d,
            "/disc_data",
            self.disc_callback,
            5,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/camera1/image_raw",
            self.image_callback,
            5,
        )

        self.board_pub = self.create_publisher(Int8MultiArray, "/game_state/board", 5)

        self.warped_image_pub = self.create_publisher(
            Image, "/game_state/warped_image", 5
        )

        self.get_logger().debug("Game state node started")

    def board_callback(self, msg):
        current_corners = list(zip(msg.corner_x, msg.corner_y))

        if self.last_board == current_corners:
            return

        if self.last_board:
            return

        self.last_board = current_corners
        self.board = msg
        self.get_logger().debug("Received board corners, computing homography...")
        if len(msg.corner_x) == 4 and len(msg.corner_y) == 4:
            src_points = np.array(
                [[msg.corner_x[i], msg.corner_y[i]] for i in range(4)], dtype=np.float32
            )
            dst_points = np.array(
                [[0, 0], [700, 0], [700, 600], [0, 600]], dtype=np.float32
            )
            self.homography = cv2.getPerspectiveTransform(src_points, dst_points)
            self.get_logger().debug("Computed homography from board corners")
        else:
            self.homography = None
            self.get_logger().debug("Invalid board corners, homography not computed")
        self.update_game_state()

    def disc_callback(self, msg):
        self.disc_data = msg
        self.update_game_state()

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().debug(f"Failed to convert image: {e}")
        self.update_game_state()

    def update_game_state(self):
        if self.board is None or self.disc_data is None or self.homography is None:
            self.get_logger().debug("Waiting for board, disc data, and homography...")
            return

        board_array = [[0 for _ in range(7)] for _ in range(6)]

        for x, y, color in zip(
            self.disc_data.x, self.disc_data.y, self.disc_data.color
        ):
            transformed = self.apply_homography(self.homography, x, y)
            if transformed is None:
                continue
            tx, ty = transformed

            if tx < 0 or tx >= 700 or ty < 0 or ty >= 600:
                self.get_logger().debug(
                    f"Transformed point out of bounds: (tx, ty)=({tx:.1f}, {ty:.1f})"
                )
                continue

            col = int(tx / 100.0)
            row = int(ty / 100.0)

            col = max(0, min(6, col))
            row = max(0, min(5, row))

            row = 5 - row

            color = str(color).lower()

            if color == "red":
                board_array[row][col] = 1
            elif color == "yellow":
                board_array[row][col] = 2

            self.get_logger().debug(
                f"(x,y)=({x:.1f},{y:.1f}) → (tx,ty)=({tx:.1f},{ty:.1f}) → (row,col)=({row},{col}) color={color}"
            )

        board_array = board_array[::-1]

        self.get_logger().debug(f"\nCurrent board state:\n{np.array(board_array)}")

        msg = Int8MultiArray()

        msg.layout.dim.append(MultiArrayDimension(label="rows", size=6, stride=42))
        msg.layout.dim.append(MultiArrayDimension(label="cols", size=7, stride=7))

        msg.data = [int(cell) for row in board_array for cell in row]

        self.board_pub.publish(msg)
        self.get_logger().debug("Published game state board")

        if self.latest_image is not None and self.homography is not None:
            try:
                warped_image = cv2.warpPerspective(
                    self.latest_image,
                    self.homography,
                    (700, 600),  # Canonical board size
                )

                warped_msg = self.bridge.cv2_to_imgmsg(warped_image, encoding="bgr8")
                warped_msg.header.stamp = self.get_clock().now().to_msg()
                warped_msg.header.frame_id = "board_canonical"

                self.warped_image_pub.publish(warped_msg)
                self.get_logger().debug("Published warped image for debugging")
            except Exception as e:
                self.get_logger().debug(f"Failed to warp and publish image: {e}")

    def apply_homography(self, H, x, y):
        point = np.array([x, y, 1.0], dtype=np.float32)
        warped = H.dot(point)
        if warped[2] == 0:
            return None
        return float(warped[0] / warped[2]), float(warped[1] / warped[2])


def main(args=None):
    rclpy.init(args=args)
    node = BoardStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

