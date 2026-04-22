"""
================================================================================
Disc Detection Node (disc_node.py)
================================================================================

PURPOSE:
    Real-time detection of red and yellow Connect Four discs from camera feed.
    Uses HSV color segmentation to identify disc positions in pixel coordinates.
    This is the 1st stage of the perception pipeline.

KEY PROCESSING STEPS:
    1. Load color configuration from color_config.json
    2. Convert BGR image to HSV color space
    3. Apply HSV range masks for target colors
    4. Combine masks (bitwise OR for red's dual ranges)
    5. Apply Gaussian blur (5x5 kernel) for noise reduction
    6. Find contours on masked image
    7. Calculate centroid for each contour (disc position)

OUTPUT DATA STRUCTURE (DiscLoc2d):
    - x[]: Array of x-coordinates (pixels) for all detected discs
    - y[]: Array of y-coordinates (pixels) for all detected discs
    - color[]: Array of color labels ("red" or "yellow") matching x,y order

================================================================================
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from game_msgs.msg import DiscLoc2d
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os

from ament_index_python.packages import get_package_share_directory


def load_color_config(config_file):
    """Load color configuration from JSON file or return defaults."""
    default_config = {
        'red': {
            'lower_hsv': [0, 70, 70],
            'upper_hsv': [10, 255, 255]
        },
        'yellow': {
            'lower_hsv': [25, 50, 50],
            'upper_hsv': [35, 255, 255]
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Validate config has required fields
            for color in ['red', 'yellow']:
                if color not in config or 'lower_hsv' not in config[color] or 'upper_hsv' not in config[color]:
                    self.get_logger().warn(f"Invalid color config for {color}, using defaults")
                    config[color] = default_config[color]
            return config
        except Exception as e:
            print(f"Error reading color config: {e}")
            return default_config
    
    return default_config


class DiscDetector(Node):
    def __init__(self):
        super().__init__('disc_detector')
        self.bridge = CvBridge()
        
        # Load color configuration
        config_file = os.path.join(
            get_package_share_directory('board_calibration'),
            'color_config.json'
        )
        self.color_config = load_color_config(config_file)
        self.get_logger().info(f"Loaded color config: {self.color_config}")
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image_callback,
            10
        )
        self.red_points_pub = self.create_publisher(PolygonStamped, f'/disc_points_red', 10)
        self.yellow_points_pub = self.create_publisher(PolygonStamped, f'/disc_points_yellow', 10)
        self.red_image_pub = self.create_publisher(Image, f'/disc_image_red', 10)
        self.yellow_image_pub = self.create_publisher(Image, f'/disc_image_yellow', 10)
        self.red_mask_pub = self.create_publisher(Image, f'/disc_mask_red', 10)
        self.yellow_mask_pub = self.create_publisher(Image, f'/disc_mask_yellow', 10)
        self.disc_data_pub = self.create_publisher(DiscLoc2d, f'/disc_data', 10)
        self.get_logger().info("Disc detector node started")

    def image_callback(self, msg):
        red_candidates = self.image_callback_color(msg, 'red')
        yellow_candidates = self.image_callback_color(msg, 'yellow')

        disc_data_msg = DiscLoc2d()
        x_red = [p.x for p in red_candidates]
        y_red = [p.y for p in red_candidates]

        x_yellow = [p.x for p in yellow_candidates]
        y_yellow = [p.y for p in yellow_candidates]

        x = x_red + x_yellow
        y = y_red + y_yellow
        disc_data_msg.x = x
        disc_data_msg.y = y
        disc_data_msg.color = ['red'] * len(red_candidates) + ['yellow'] * len(yellow_candidates)
        self.disc_data_pub.publish(disc_data_msg)

    def image_callback_color(self, msg, color):
        self.get_logger().info("Image received")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Get color from config
        if color in self.color_config and 'lower_hsv' in self.color_config[color] and 'upper_hsv' in self.color_config[color]:
            lower_hsv = self.color_config[color]['lower_hsv']
            upper_hsv = self.color_config[color]['upper_hsv']
            
            # Handle red's dual range (wrap around hue 180)
            if color == 'red' and lower_hsv[0] > upper_hsv[0]:
                # Red wraps around, need two ranges
                lower1 = np.array([0, lower_hsv[1], lower_hsv[2]])
                upper1 = np.array([upper_hsv[0], upper_hsv[1], upper_hsv[2]])
                lower2 = np.array([lower_hsv[0], lower_hsv[1], lower_hsv[2]])
                upper2 = np.array([180, upper_hsv[1], upper_hsv[2]])
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower1 = np.array(lower_hsv)
                upper1 = np.array(upper_hsv)
                mask = cv2.inRange(hsv, lower1, upper1)
        else:
            # Fallback to defaults
            self.get_logger().warn(f"Color config not found for {color}, using defaults")
            if color == 'red':
                lower1 = np.array([0, 120, 120])
                upper1 = np.array([10, 255, 255])
                lower2 = np.array([170, 120, 120])
                upper2 = np.array([180, 255, 255])
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            elif color == 'yellow':
                lower1 = np.array([25, 100, 100])
                upper1 = np.array([35, 255, 255])
                mask = cv2.inRange(hsv, lower1, upper1)
            else:
                lower1 = np.array([0, 120, 120])
                upper1 = np.array([10, 255, 255])
                mask = cv2.inRange(hsv, lower1, upper1)

        # # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Publish the mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        if color == 'red':
            self.red_mask_pub.publish(mask_msg)
        else:
            self.yellow_mask_pub.publish(mask_msg)

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

                img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                if color == 'red':
                    self.red_image_pub.publish(img_msg)
                else:
                    self.yellow_image_pub.publish(img_msg)

            else:
                self.get_logger().info("No circular disc candidates above threshold were published")
            
            return candidate_points
        return []

def main(args=None):
    rclpy.init(args=args)
    node = DiscDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()