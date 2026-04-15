#!/usr/bin/env python3
"""
Color picker tool for board detector.
Displays the camera feed and saves selected color to config file.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


def bgr_to_hsv_range(bgr_color, h_tolerance=10, s_tolerance=50, v_tolerance=50):
    """Convert BGR color to HSV range with tolerance."""
    bgr_array = np.uint8([[[bgr_color[2], bgr_color[1], bgr_color[0]]]])
    hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_array[0][0]
    
    lower_h = max(0, int(h) - h_tolerance)
    upper_h = min(180, int(h) + h_tolerance)
    lower_s = max(0, int(s) - s_tolerance)
    upper_s = min(255, int(s) + s_tolerance)
    lower_v = max(0, int(v) - v_tolerance)
    upper_v = min(255, int(v) + v_tolerance)
    
    return (lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v), (int(h), int(s), int(v))


def save_color_config(lower_hsv, upper_hsv, config_file):
    """Save color configuration to JSON file."""
    config = {
        'lower_h': lower_hsv[0],
        'lower_s': lower_hsv[1],
        'lower_v': lower_hsv[2],
        'upper_h': upper_hsv[0],
        'upper_s': upper_hsv[1],
        'upper_v': upper_hsv[2],
    }
    
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nColor config saved to: {config_file}")


def make_mouse_callback(state, config_file):
    """Create a mouse callback function for color picking."""
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr = state['image'][y, x]
            
            # Convert to HSV and get range
            (lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v), (h, s, v) = bgr_to_hsv_range(
                bgr, h_tolerance=10, s_tolerance=50, v_tolerance=30
            )
            
            print("\n" + "="*70)
            print(f"Color selected at pixel: ({x}, {y})")
            print("="*70)
            print(f"BGR values: B={int(bgr[0])}, G={int(bgr[1])}, R={int(bgr[2])}")
            print(f"HSV center: H={h}, S={s}, V={v}")
            print(f"HSV Range:")
            print(f"  Hue:        [{lower_h:3d}, {upper_h:3d}]")
            print(f"  Saturation: [{lower_s:3d}, {upper_s:3d}]")
            print(f"  Value:      [{lower_v:3d}, {upper_v:3d}]")
            print("="*70)
            
            # Save to file
            save_color_config((lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v), config_file)
            
            state['selected'] = True
    
    return mouse_callback


def main():
    """Main color picker function."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Import rclpy here for ROS2 camera feed
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    
    config_file = os.path.join(os.path.dirname(__file__), 'color_config.json')
    
    class ColorPickerNode(Node):
        def __init__(self):
            super().__init__('color_picker')
            self.bridge = CvBridge()
            self.state = {'image': None, 'selected': False}
            self.window_created = False
            
            self.image_sub = self.create_subscription(
                Image,
                '/camera1/image_raw',
                self.image_callback,
                10
            )
            self.get_logger().info("Color Picker started. Click on a board color to save.")
        
        def image_callback(self, msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.state['image'] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
            
            # Create window on first image
            if not self.window_created:
                window_name = "Click on the board color"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window_name, make_mouse_callback(self.state, config_file))
                self.window_created = True
                self.get_logger().info("Window created. Click on the board color.")
            
            cv2.imshow("Click on the board color", cv_image)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.state['selected']:
                cv2.destroyAllWindows()
                rclpy.shutdown()
    
    rclpy.init()
    node = ColorPickerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
