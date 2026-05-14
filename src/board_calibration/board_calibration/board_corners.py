#!/usr/bin/env python3
import cv2
import rclpy
from cv_bridge import CvBridge
from game_msgs.msg import GameBoard
from rclpy.node import Node
from sensor_msgs.msg import Image


def make_mouse_callback(state):
    """Create a mouse callback function for selecting board corners."""

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state["corners"]) < 4:
            state["corners"].append((x, y))
            print(f"Selected corner {len(state['corners'])}/4: ({x}, {y})")

            if len(state["corners"]) == 4:
                state["selected"] = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            state["corners"] = []
            print("Cleared corner selection. Start again.")

    return mouse_callback


class ColorPickerNode(Node):
    def __init__(self):
        super().__init__("color_picker")
        self.bridge = CvBridge()
        self.state = {
            "image": None,
            "selected": False,
            "corners": [],
        }
        self.window_created = False

        self.image_sub = self.create_subscription(
            Image, "/camera1/image_raw", self.image_callback, 10
        )

        self.board_data_pub = self.create_publisher(GameBoard, "/board_data", 10)

        # Publish the corners once at startup
        self.timer = self.create_timer(1.0, self.publish_corners)
        self.corners = None
        self.get_logger().info(
            "Board Corner Picker started. Click four board corners in order: TL, TR, BR, BL."
        )

    def image_callback(self, msg):

        if len(self.state["corners"]) == 4:
            self.corners = self.state["corners"]
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.get_logger().info(f"Image_size: {cv_image.shape}")
            self.state["image"] = cv_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        if not self.window_created:
            window_name = "Select board corners"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, make_mouse_callback(self.state))
            self.window_created = True
            self.get_logger().info(
                "Window created. Click four board corners in order: TL, TR, BR, BL."
            )

        display_image = cv_image.copy()
        for idx, (cx, cy) in enumerate(self.state["corners"], start=1):
            cv2.circle(display_image, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(
                display_image,
                str(idx),
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if len(self.state["corners"]) < 4:
            remaining = 4 - len(self.state["corners"])
            cv2.putText(
                display_image,
                f"Click {remaining} corner(s) remaining",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Select board corners", display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or self.state["selected"]:
            cv2.destroyAllWindows()

    def publish_corners(self):

        if self.corners:
            board_state_msg = GameBoard()
            board_state_msg.corner_x = [float(point[0]) for point in self.corners]
            board_state_msg.corner_y = [float(point[1]) for point in self.corners]
            self.board_data_pub.publish(board_state_msg)


def main():
    rclpy.init()
    node = ColorPickerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

