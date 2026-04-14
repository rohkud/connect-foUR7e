import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class BoardDetector(Node):
    def __init__(self):
        super().__init__('board_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image_callback,
            10
        )
        self.board_pub = self.create_publisher(Image, '/board_image', 10)
        self.debug_pub = self.create_publisher(Image, '/board_debug_image', 10)
        self.get_logger().info("Board detector node started")

    def image_callback(self, msg):
        self.get_logger().info("Image received")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Blue color range (allow more range in illumination)
        lower_blue = np.array([90, 50, 80])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply Gaussian blur
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f"Found {len(contours)} contours")

        # Publish debug image with contours
        if contours:
            debug_image = cv_image.copy()
            cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_pub.publish(debug_msg)

        if contours:
            # Find the largest rectangular contour
            best_contour = None
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2500:  # Minimum area
                    # Approximate the contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if area > max_area:  # Rectangular
                        best_contour = contour
                        max_area = area

            if best_contour is not None:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(best_contour)
                # Crop the image
                cropped = cv_image[y:y+h, x:x+w]
                # Publish cropped image
                board_msg = self.bridge.cv2_to_imgmsg(cropped, encoding='bgr8')
                self.board_pub.publish(board_msg)
                self.get_logger().info(f"Board detected and cropped: {w}x{h}")

def main(args=None):
    rclpy.init(args=args)
    node = BoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()