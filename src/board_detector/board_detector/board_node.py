import rclpy
from rclpy.node import Node
from game_msgs.msg import GameBoard
import json
import os


def load_board_corners(config_file):
    """Load board corners from JSON file. Returns defaults if file doesn't exist."""
    default_corners = [[0.0, 0.0] for _ in range(4)]

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error reading config file {config_file}: {e}")
            return default_corners

        corners = config.get('board_corners', default_corners)
        if not isinstance(corners, list) or len(corners) != 4:
            return default_corners
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
            return cleaned

    return default_corners


class BoardDetector(Node):
    def __init__(self):
        super().__init__('board_detector')
        
        config_dir = os.path.dirname(__file__)
        board_corners_file = os.path.join(config_dir, 'board_corners.json')
        self.board_corners = load_board_corners(board_corners_file)
        
        self.board_data_pub = self.create_publisher(GameBoard, '/board_data', 10)
        
        # Publish the corners once at startup
        self.publish_corners()
        
        self.get_logger().info('Board Detector initialized')
        self.get_logger().info('Published board corners')

    def publish_corners(self):
        board_state_msg = GameBoard()
        board_state_msg.corner_x = [float(point[0]) for point in self.board_corners]
        board_state_msg.corner_y = [float(point[1]) for point in self.board_corners]
        self.board_data_pub.publish(board_state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BoardDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
