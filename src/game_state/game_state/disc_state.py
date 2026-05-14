import cv2
import numpy as np
import rclpy
from game_msgs.msg import DiscLoc2d, GameBoard
from rclpy.node import Node


class DiscStateNode(Node):
    def __init__(self):
        super().__init__("disc_state")

        self.board = None
        self.last_board = None
        self.disc_data = None
        self.homography = None

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

        self.disc_pub = self.create_publisher(DiscLoc2d, "/active_disc_data", 5)

        self.get_logger().debug("Disc state node started")

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

    def disc_callback(self, msg):
        self.disc_data = msg

        new_disc_data = DiscLoc2d()
        for x, y, color in zip(
            self.disc_data.x, self.disc_data.y, self.disc_data.color
        ):
            if self.homography is None:
                return

            transformed = self.apply_homography(self.homography, x, y)
            if transformed is None:
                continue
            tx, ty = transformed

            if not (tx < 0 or tx >= 700 or ty < 0 or ty >= 600):
                self.get_logger().debug(
                    f"Transformed point out of bounds: (tx, ty)=({tx:.1f}, {ty:.1f})"
                )
                continue

            new_disc_data.x.append(x)
            new_disc_data.y.append(y)
            new_disc_data.color.append(color)

        self.disc_pub.publish(new_disc_data)
        self.get_logger().debug(
            "Published active disc data with transformed coordinates"
        )

    def apply_homography(self, H, x, y):
        point = np.array([x, y, 1.0], dtype=np.float32)
        warped = H.dot(point)
        if warped[2] == 0:
            return None
        return float(warped[0] / warped[2]), float(warped[1] / warped[2])


def main(args=None):
    rclpy.init(args=args)
    node = DiscStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

