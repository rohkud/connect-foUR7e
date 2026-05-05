#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from game_msgs.msg import HsvColor
from cv_bridge import CvBridge


CLICK_ORDER = [
    "RED TABLE",
    "RED BOARD",
    "YELLOW TABLE",
    "YELLOW BOARD",
]


def bgr_to_hsv_range(bgr_color, h_tolerance=10, s_tolerance=50, v_tolerance=30):
    bgr_array = np.uint8([[[bgr_color[0], bgr_color[1], bgr_color[2]]]])
    hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_array[0][0]

    lower_h = max(0, int(h) - h_tolerance)
    upper_h = min(180, int(h) + h_tolerance)

    lower_s = max(0, int(s) - s_tolerance)
    upper_s = min(255, int(s) + s_tolerance)

    lower_v = max(0, int(v) - v_tolerance)
    upper_v = min(255, int(v) + v_tolerance)

    lower = (lower_h, lower_s, lower_v)
    upper = (upper_h, upper_s, upper_v)
    center = (int(h), int(s), int(v))

    return lower, upper, center


def make_mouse_callback(state):
    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if state["image"] is None:
            return

        if len(state["colors"]) >= 4:
            return

        bgr = state["image"][y, x]
        label = CLICK_ORDER[len(state["colors"])]

        hsv_range = bgr_to_hsv_range(
            bgr,
            h_tolerance=10,
            s_tolerance=50,
            v_tolerance=30,
        )

        lower_hsv, upper_hsv, hsv_center = hsv_range

        print("\n" + "=" * 70)
        print(f"Selected {label} at pixel ({x}, {y})")
        print("=" * 70)
        print(f"BGR: B={int(bgr[0])}, G={int(bgr[1])}, R={int(bgr[2])}")
        print(f"HSV center: H={hsv_center[0]}, S={hsv_center[1]}, V={hsv_center[2]}")
        print(f"HSV lower:  {lower_hsv}")
        print(f"HSV upper:  {upper_hsv}")
        print("=" * 70)

        state["colors"].append((label, bgr, hsv_range))

        if len(state["colors"]) == 4:
            state["selected"] = True
            print("All 4 colors selected. Publishing HSV ranges.")
        else:
            print(f"Next click: {CLICK_ORDER[len(state['colors'])]}")

    return mouse_callback


class ColorPickerNode(Node):
    def __init__(self):
        super().__init__("color_picker")

        self.bridge = CvBridge()
        self.state = {
            "image": None,
            "selected": False,
            "colors": [],
        }

        self.window_created = False

        self.image_sub = self.create_subscription(
            Image,
            "/camera1/image_raw",
            self.image_callback,
            10,
        )

        self.red_color_pub = self.create_publisher(
            HsvColor,
            "/disc_color_red",
            10,
        )

        self.yellow_color_pub = self.create_publisher(
            HsvColor,
            "/disc_color_yellow",
            10,
        )

        self.red_timer = self.create_timer(0.5, self.publish_red_color)
        self.yellow_timer = self.create_timer(0.5, self.publish_yellow_color)

        self.get_logger().info(
            "Color Picker started. Click: RED TABLE, RED BOARD, YELLOW TABLE, YELLOW BOARD."
        )

    def combine_hsv_ranges(self, hsv_range_1, hsv_range_2):
        msg = HsvColor()

        lower1, upper1, _ = hsv_range_1
        lower2, upper2, _ = hsv_range_2

        lower = np.minimum(np.array(lower1), np.array(lower2))
        upper = np.maximum(np.array(upper1), np.array(upper2))

        msg.lower = [int(lower[0]), int(lower[1]), int(lower[2])]
        msg.upper = [int(upper[0]), int(upper[1]), int(upper[2])]

        return msg

    def publish_red_color(self):
        if len(self.state["colors"]) == 4:
            red_table_hsv = self.state["colors"][0][2]
            red_board_hsv = self.state["colors"][1][2]

            msg = self.combine_hsv_ranges(red_table_hsv, red_board_hsv)
            self.red_color_pub.publish(msg)

    def publish_yellow_color(self):
        if len(self.state["colors"]) == 4:
            yellow_table_hsv = self.state["colors"][2][2]
            yellow_board_hsv = self.state["colors"][3][2]

            msg = self.combine_hsv_ranges(yellow_table_hsv, yellow_board_hsv)
            self.yellow_color_pub.publish(msg)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.state["image"] = cv_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        if not self.window_created:
            cv2.namedWindow("Disc Color Picker", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(
                "Disc Color Picker",
                make_mouse_callback(self.state),
            )
            self.window_created = True

        display_image = cv_image.copy()

        for i, (label, bgr, _) in enumerate(self.state["colors"]):
            swatch_y = 60 + i * 50

            cv2.rectangle(
                display_image,
                (10, swatch_y),
                (60, swatch_y + 40),
                [int(bgr[0]), int(bgr[1]), int(bgr[2])],
                -1,
            )

            cv2.putText(
                display_image,
                f"{label}: B={int(bgr[0])}, G={int(bgr[1])}, R={int(bgr[2])}",
                (70, swatch_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        if len(self.state["colors"]) < 4:
            next_label = CLICK_ORDER[len(self.state["colors"])]

            cv2.putText(
                display_image,
                f"Click {next_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                display_image,
                "Publishing HSV ranges. Press q to quit.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Disc Color Picker", display_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            rclpy.shutdown()

        elif key == ord("r"):
            self.state["colors"] = []
            self.state["selected"] = False
            print("Reset color selection.")
            print(f"Next click: {CLICK_ORDER[0]}")


def main():
    rclpy.init()
    node = ColorPickerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()