import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
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
        self.points_pub = self.create_publisher(PolygonStamped, '/disc_points', 10)
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
            points_published = 0
            candidate_points = []
            cv2.drawContours(cv_image, contours, -1, (0, 255, 0), 2)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 500:
                    continue

                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if circularity <= 0.5:
                    continue

                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                cv2.circle(cv_image, (cx, cy), 6, (255, 0, 0), -1)
                point = Point32(x=float(cx), y=float(cy), z=0.0)
                candidate_points.append(point)
                points_published += 1

                self.get_logger().info(
                    f"Disc candidate at ({cx}, {cy}) with circularity {circularity:.2f}")

            if points_published > 0:
                polygon_msg = PolygonStamped()
                polygon_msg.header = msg.header
                polygon_msg.polygon.points = candidate_points
                self.points_pub.publish(polygon_msg)

                img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                self.image_pub.publish(img_msg)
            else:
                self.get_logger().info("No circular disc candidates above threshold were published")

def main(args=None):
    rclpy.init(args=args)
    node = DiscDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()