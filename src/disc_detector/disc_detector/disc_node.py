import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class DiscDetector(Node):
    def __init__(self):
        super().__init__('disc_detector')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image_callback,
            10
        )
        self.pose_pub = self.create_publisher(PointStamped, '/disc_pose', 10)
        self.image_pub = self.create_publisher(Image, '/disc_image', 10)
        self.mask_pub = self.create_publisher(Image, '/disc_mask', 10)
        self.declare_parameter('color', 'red')
        self.get_logger().info("Disc detector node started")

    def image_callback(self, msg):
        self.get_logger().info("Image received")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Get color parameter
        color = self.get_parameter('color').value
        if color == 'red':
            lower1 = np.array([0, 120, 120])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 120, 120])
            upper2 = np.array([180, 255, 255])
        elif color == 'yellow':
            lower1 = np.array([25, 100, 100])
            upper1 = np.array([35, 255, 255])
            lower2 = None  # No second range
            upper2 = None
        else:
            self.get_logger().warn(f"Color {color} not supported, using red")
            lower1 = np.array([0, 120, 120])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 120, 120])
            upper2 = np.array([180, 255, 255])

        if lower2 is not None:
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, lower1, upper1)

        # # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Publish the mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        self.mask_pub.publish(mask_msg)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f"Found {len(contours)} contours")

        if contours:
            # Find the most circular contour
            best_contour = None
            best_circularity = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    if circularity > 0.7 and circularity > best_circularity:
                        best_contour = contour
                        best_circularity = circularity

            if best_contour is not None:
                # Draw contours on the image
                cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 2)

                # Compute center
                M = cv2.moments(best_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Draw center
                    cv2.circle(cv_image, (cx, cy), 10, (255, 0, 0), -1)

                    # Publish pose
                    pose_msg = PointStamped()
                    pose_msg.header = msg.header
                    pose_msg.point.x = float(cx)
                    pose_msg.point.y = float(cy)
                    pose_msg.point.z = 0.0  # Assuming 2D
                    self.pose_pub.publish(pose_msg)
                    self.get_logger().info(f"Disc detected at ({cx}, {cy}) with circularity {best_circularity:.2f}")

                # Publish the annotated image
                img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                self.image_pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DiscDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()