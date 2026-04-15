#!/usr/bin/env python3
"""
Board corner picker tool for board detector.
Displays the camera feed and saves four selected board corners to the config file.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


def load_config(config_file):
    """Load existing board corners config file or return defaults."""
    default_config = {
        'board_corners': [[0.0, 0.0] for _ in range(4)],
    }

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error reading config file {config_file}: {e}")
            return default_config

        corners = config.get('board_corners', default_config['board_corners'])
        if not isinstance(corners, list) or len(corners) != 4:
            corners = default_config['board_corners']
        else:
            cleaned = []
            for corner in corners:
                if isinstance(corner, (list, tuple)) and len(corner) == 2:
                    try:
                        cleaned.append([float(corner[0]), float(corner[1])])
                    except (TypeError, ValueError):
                        cleaned.append([0.0, 0.0])
                else:
                    cleaned.append([0.0, 0.0])
            corners = cleaned

        return {'board_corners': corners}

    return default_config


def save_config(corners, config_file):
    """Save board corner points to JSON file."""
    config = {'board_corners': [[float(x), float(y)] for x, y in corners]}
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nBoard corner config saved to: {config_file}")


def make_mouse_callback(state, config_file):
    """Create a mouse callback function for selecting board corners."""
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state['corners']) < 4:
            state['corners'].append((x, y))
            print(f"Selected corner {len(state['corners'])}/4: ({x}, {y})")

            if len(state['corners']) == 4:
                save_config(state['corners'], config_file)
                state['selected'] = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            state['corners'] = []
            print("Cleared corner selection. Start again.")

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
    
    config_file = os.path.join(os.path.dirname(__file__), 'board_corners.json')
    config = load_config(config_file)
    
    class ColorPickerNode(Node):
        def __init__(self):
            super().__init__('color_picker')
            self.bridge = CvBridge()
            self.state = {
                'image': None,
                'selected': False,
                'corners': [],
            }
            self.window_created = False
            
            self.image_sub = self.create_subscription(
                Image,
                '/camera1/image_raw',
                self.image_callback,
                10
            )
            self.get_logger().info("Board Corner Picker started. Click four board corners in order: TL, TR, BR, BL.")
        
        def image_callback(self, msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.state['image'] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
            
            # Create window on first image
            if not self.window_created:
                window_name = "Select board corners"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window_name, make_mouse_callback(self.state, config_file))
                self.window_created = True
                self.get_logger().info("Window created. Click four board corners in order: TL, TR, BR, BL.")

            display_image = cv_image.copy()
            for idx, (cx, cy) in enumerate(self.state['corners'], start=1):
                cv2.circle(display_image, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(
                    display_image,
                    str(idx),
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if len(self.state['corners']) < 4:
                remaining = 4 - len(self.state['corners'])
                cv2.putText(
                    display_image,
                    f"Click {remaining} corner(s) remaining",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Select board corners", display_image)
            
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
