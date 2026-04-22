#!/usr/bin/env python3
"""
Color picker tool for board detector.
Displays the camera feed and saves selected red and yellow disc colors to config file.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


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


def save_color_config(colors, config_file):
    """Save red and yellow color configuration to JSON file."""
    config = {}
    for color_name, (bgr, hsv_range) in colors.items():
        lower_hsv, upper_hsv, center_hsv = hsv_range
        config[color_name] = {
            'bgr': [int(bgr[0]), int(bgr[1]), int(bgr[2])],
            'hsv_center': list(center_hsv),
            'lower_hsv': list(lower_hsv),
            'upper_hsv': list(upper_hsv),
        }
    
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nColor config saved to: {config_file}")
    for color_name, data in config.items():
        print(f"  {color_name}: BGR={data['bgr']}, HSV center={data['hsv_center']}")


def make_mouse_callback(state, config_file):
    """Create a mouse callback function for color picking."""
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state['colors']) < 2:
            bgr = state['image'][y, x]
            color_name = ['red', 'yellow'][len(state['colors'])]
            
            # Convert to HSV and get range
            hsv_range = bgr_to_hsv_range(
                bgr, h_tolerance=10, s_tolerance=50, v_tolerance=30
            )
            lower_hsv, upper_hsv, hsv_center = hsv_range
            
            print("\n" + "="*70)
            print(f"Selected {color_name.upper()} at pixel: ({x}, {y})")
            print("="*70)
            print(f"BGR values: B={int(bgr[0])}, G={int(bgr[1])}, R={int(bgr[2])}")
            print(f"HSV center: H={hsv_center[0]}, S={hsv_center[1]}, V={hsv_center[2]}")
            print(f"HSV Range:")
            print(f"  Hue:        [{lower_hsv[0]:3d}, {upper_hsv[0]:3d}]")
            print(f"  Saturation: [{lower_hsv[1]:3d}, {upper_hsv[1]:3d}]")
            print(f"  Value:      [{lower_hsv[2]:3d}, {upper_hsv[2]:3d}]")
            print("="*70)
            
            state['colors'].append((bgr, hsv_range))
            
            if len(state['colors']) == 2:
                # Save both colors
                colors_dict = {
                    'red': state['colors'][0],
                    'yellow': state['colors'][1]
                }
                save_color_config(colors_dict, config_file)
                state['selected'] = True
            else:
                print(f"{color_name.capitalize()} saved! Now click on a YELLOW disc.")
    
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
    
    config_file = os.path.join(
        get_package_share_directory('board_calibration'),
        'color_config.json'
    )
    
    class ColorPickerNode(Node):
        def __init__(self):
            super().__init__('color_picker')
            self.bridge = CvBridge()
            self.state = {'image': None, 'selected': False, 'colors': []}
            self.window_created = False
            
            self.image_sub = self.create_subscription(
                Image,
                '/camera1/image_raw',
                self.image_callback,
                10
            )
            self.get_logger().info("Color Picker started. Click on a RED disc, then a YELLOW disc.")
        
        def image_callback(self, msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.state['image'] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
            
            # Create window on first image
            if not self.window_created:
                window_name = "Disc Color Picker"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window_name, make_mouse_callback(self.state, config_file))
                self.window_created = True
                self.get_logger().info("Window created. Click on a RED disc, then a YELLOW disc.")
            
            display_image = cv_image.copy()
            
            # Draw color swatches if colors selected
            for i, (bgr, _) in enumerate(self.state['colors']):
                color_name = ['RED', 'YELLOW'][i]
                swatch_y = 60 + i * 50
                cv2.rectangle(display_image, (10, swatch_y), (60, swatch_y + 40), [int(bgr[0]), int(bgr[1]), int(bgr[2])], -1)
                cv2.putText(
                    display_image,
                    f"{color_name}: B={int(bgr[0])}, G={int(bgr[1])}, R={int(bgr[2])}",
                    (70, swatch_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            
            # Display instructions
            if len(self.state['colors']) < 2:
                remaining = 2 - len(self.state['colors'])
                color_to_pick = ['RED', 'YELLOW'][len(self.state['colors'])]
                cv2.putText(
                    display_image,
                    f"Click {remaining} more ({color_to_pick} disc)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    display_image,
                    "CALIBRATION COMPLETE! Press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            
            # Draw crosshair at mouse position
            # Note: This would require trackMouse, skipping for simplicity
            
            cv2.imshow("Disc Color Picker", display_image)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.state['selected']:
                cv2.destroyAllWindows()
                rclpy.shutdown()
            elif key == ord('r'):
                # Reset
                self.state['colors'] = []
                print("Reset color selection. Start again.")
    
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