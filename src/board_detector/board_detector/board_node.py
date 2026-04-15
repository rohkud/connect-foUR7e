import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from game_msgs.msg import GameBoard, DiscLoc2d
import cv2
import numpy as np
import json
import os


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
    return default_config


class BoardDetector(Node):
    def __init__(self):
        super().__init__('board_detector')
        self.bridge = CvBridge()
        
        config_dir = os.path.dirname(__file__)
        config_file = os.path.join(config_dir, 'color_config.json')
        color_config = load_color_config(config_file)
        
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
            10,
        )
        self.board_pub = self.create_publisher(Image, '/board_image', 10)
        self.debug_pub = self.create_publisher(Image, '/board_debug_image', 10)
        self.slot_contour_pub = self.create_publisher(Image, '/board_slot_contours', 10)
        self.board_data_pub = self.create_publisher(GameBoard, '/board_data', 10)
        self.slot_data_pub = self.create_publisher(DiscLoc2d, '/board_slot_data', 10)
        
        h_range = f"[{color_config['lower_h']}, {color_config['upper_h']}]"
        s_range = f"[{color_config['lower_s']}, {color_config['upper_s']}]"
        v_range = f"[{color_config['lower_v']}, {color_config['upper_v']}]"
        self.get_logger().info('Board Detector initialized')
        self.get_logger().info(f'Color config: H{h_range} S{s_range} V{v_range}')
        self.get_logger().info('To pick a new color, run: python3 color_picker.py')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f'Found {len(contours)} contours')

        if contours:
            debug_image = cv_image.copy()
            cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_pub.publish(debug_msg)

        best_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2500:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if area > max_area:
                    best_contour = contour
                    max_area = area

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            leg_trim = int(h * 0.2)
            cropped_height = max(1, h - leg_trim)
            cropped = cv_image[y:y+cropped_height, x:x+w]

            board_msg = self.bridge.cv2_to_imgmsg(cropped, encoding='bgr8')
            self.board_pub.publish(board_msg)

            board_state_msg = GameBoard()
            board_state_msg.x = float(x)
            board_state_msg.y = float(y)
            board_state_msg.w = float(w)
            board_state_msg.h = float(h)
            self.board_data_pub.publish(board_state_msg)

            self.board_slots(cv_image)
            self.get_logger().info(
                f'Board detected and cropped: {w}x{cropped_height} (trimmed {leg_trim}px for legs)')

    def board_slots(self, board_image):
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f'Found {len(contours)} contours')

        output = board_image.copy()
        slot_points = []

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

            cv2.circle(output, (cx, cy), 6, (255, 0, 0), -1)
            point = (float(cx), float(cy))
            slot_points.append(point)
            self.get_logger().info(f'Disc candidate at ({cx}, {cy}) with circularity {circularity:.2f}')

        slot_msg = DiscLoc2d()
        slot_msg.x = [p[0] for p in slot_points]
        slot_msg.y = [p[1] for p in slot_points]
        slot_msg.color = ['slot'] * len(slot_points)
        self.slot_data_pub.publish(slot_msg)

        contour_msg = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        self.slot_contour_pub.publish(contour_msg)

        if len(slot_points) == 0:
            self.get_logger().info('No circular slot candidates above threshold were published')

        self.get_logger().info(f'Published {len(slot_points)} board slots')
        return slot_points


def main(args=None):
    rclpy.init(args=args)
    node = BoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
