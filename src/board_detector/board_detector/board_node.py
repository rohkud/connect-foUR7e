import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from cv_bridge import CvBridge
from game_msgs.msg import GameBoard
import cv2
import numpy as np
import json
import os
from pathlib import Path


def load_color_config(config_file):
    """Load color configuration from JSON file. Returns defaults if file doesn't exist."""
    default_config = {
        'lower_h': 90,
        'lower_s': 50,
        'lower_v': 80,
        'upper_h': 130,
        'upper_s': 255,
        'upper_v': 255,
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading config file {config_file}: {e}")
            return default_config
    else:
        return default_config

class BoardDetector(Node):
    def __init__(self):
        super().__init__('board_detector')
        self.bridge = CvBridge()
        
        # Load color config from file
        config_dir = os.path.dirname(__file__)
        config_file = os.path.join(config_dir, 'color_config.json')
        color_config = load_color_config(config_file)
        
        # Declare parameters
        self.declare_parameter('lower_h', color_config['lower_h'])
        self.declare_parameter('lower_s', color_config['lower_s'])
        self.declare_parameter('lower_v', color_config['lower_v'])
        self.declare_parameter('upper_h', color_config['upper_h'])
        self.declare_parameter('upper_s', color_config['upper_s'])
        self.declare_parameter('upper_v', color_config['upper_v'])
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image_callback,
            10
        )
        self.board_pub = self.create_publisher(Image, '/board_image', 10)
        self.debug_pub = self.create_publisher(Image, '/board_debug_image', 10)
        self.board_data_pub = self.create_publisher(GameBoard, '/board_data', 10)
        
        h_range = f"[{color_config['lower_h']}, {color_config['upper_h']}]"
        s_range = f"[{color_config['lower_s']}, {color_config['upper_s']}]"
        v_range = f"[{color_config['lower_v']}, {color_config['upper_v']}]"
        
        self.get_logger().info("Board Detector initialized")
        self.get_logger().info(f"Color config: H{h_range} S{s_range} V{v_range}")
        self.get_logger().info(f"To pick a new color, run: python3 color_picker.py")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Get color range from parameters
        lower_h = self.get_parameter('lower_h').value
        lower_s = self.get_parameter('lower_s').value
        lower_v = self.get_parameter('lower_v').value
        upper_h = self.get_parameter('upper_h').value
        upper_s = self.get_parameter('upper_s').value
        upper_v = self.get_parameter('upper_v').value
        
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
        
        mask = cv2.inRange(hsv, lower_color, upper_color)
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
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if area > max_area:
                        best_contour = contour
                        max_area = area

            if best_contour is not None:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(best_contour)

                # Trim the lower portion of the bounding box to crop out the legs
                leg_trim = int(h * 0.2)
                cropped_height = max(1, h - leg_trim)
                cropped = cv_image[y:y+cropped_height, x:x+w]

                # Publish cropped image
                board_msg = self.bridge.cv2_to_imgmsg(cropped, encoding='bgr8')
                self.board_pub.publish(board_msg)

                board_state_msg = GameBoard()
                board_state_msg.x = float(x)
                board_state_msg.y = float(y)
                board_state_msg.w = float(w)
                board_state_msg.h = float(h)
                self.board_data_pub.publish(board_state_msg)

                self.get_logger().info(
                    f"Board detected and cropped: {w}x{cropped_height} (trimmed {leg_trim}px for legs)")

def main(args=None):
    rclpy.init(args=args)
    node = BoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()