# capture_node_cv460.py

import os
import glob

import rclpy
from rclpy.node import Node

import cv2
import cv2.aruco as aruco

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CharucoCapture(Node):
    def __init__(self):
        super().__init__("charuco_capture")

        # --- save setup ---
        self.save_dir = "images"
        os.makedirs(self.save_dir, exist_ok=True)

        existing = glob.glob("images/img_*.jpg")
        self.count = len(existing)

        # --- CV bridge ---
        self.bridge = CvBridge()

        # --- ArUco setup (OpenCV 4.6 safe API) ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        # IMPORTANT: 4.6-compatible Charuco creation
        self.board = aruco.CharucoBoard_create(
            5, 7, 0.04, 0.03, self.aruco_dict
        )

        # detector (works in 4.6, but we also keep fallback safe)
        self.parameters = aruco.DetectorParameters_create()

        # in 4.6, this class may or may not exist depending on build
        try:
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
            self.use_detector_class = True
        except Exception:
            self.detector = None
            self.use_detector_class = False

        # --- ROS subscriber ---
        self.sub = self.create_subscription(
            Image,
            "/camera1/image_raw",
            self.image_callback,
            10,
        )

        self.get_logger().info("Listening to /camera1/image_raw")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- detect markers (4.6-safe branching) ---
        if self.use_detector_class:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )

        display = frame.copy()
        valid = False

        if ids is not None and len(ids) > 0:
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )

            # 4.6 behavior: retval is int count or None-safe check
            if retval is not None and retval > 10:
                valid = True
                aruco.drawDetectedMarkers(display, corners, ids)

        # --- overlay ---
        cv2.putText(
            display,
            "VALID" if valid else "NO BOARD",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if valid else (0, 0, 255),
            2,
        )

        cv2.imshow("capture", display)
        key = cv2.waitKey(1) & 0xFF

        # save frame
        if key == ord("s") and valid:
            path = f"{self.save_dir}/img_{self.count:03d}.jpg"
            cv2.imwrite(path, frame)
            self.get_logger().info(f"Saved: {path}")
            self.count += 1

        if key == ord("q"):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = CharucoCapture()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()