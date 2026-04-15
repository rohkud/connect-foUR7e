import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from game_msgs.msg import GameBoard, DiscLoc2d, HomographyMatrix
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
        self.homography_pub = self.create_publisher(HomographyMatrix, '/board_homography', 10)

        self.prev_corners = None
        self.corner_smoothing_alpha = 0.10
        self.corner_jump_threshold = 35.0
        
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
            cropped_mask = mask[y:y+cropped_height, x:x+w]

            warped, homography = self.rectify_board(cropped, cropped_mask)
            if warped is None:
                warped = cropped

            board_msg = self.bridge.cv2_to_imgmsg(warped, encoding='bgr8')
            self.board_pub.publish(board_msg)

            board_state_msg = GameBoard()
            board_state_msg.x = float(x)
            board_state_msg.y = float(y)
            board_state_msg.w = float(w)
            board_state_msg.h = float(h)
            self.board_data_pub.publish(board_state_msg)

            self.get_logger().info(
                f'Board detected and cropped: {w}x{cropped_height} (trimmed {leg_trim}px for legs)')

    def rectify_board(self, board_image, mask_image):
        corners = self.detect_board_corners(mask_image)
        if corners is None:
            if self.prev_corners is None:
                self.get_logger().warn('Board corners not found; skipping homography')
                return None, None
            self.get_logger().info('Using last stable corners because current detection failed')
            corners = self.prev_corners
        else:
            corners = self.stabilize_corners(corners)

        target_cols = 7
        target_rows = 6
        cell_size = 100
        target_w = target_cols * cell_size
        target_h = target_rows * cell_size

        dst = np.array([
            [0.0, 0.0],
            [target_w - 1.0, 0.0],
            [target_w - 1.0, target_h - 1.0],
            [0.0, target_h - 1.0],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(board_image, M, (target_w, target_h))

        debug_corners = board_image.copy()
        cv2.polylines(debug_corners, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_corners, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

        self.publish_homography(M)
        self.get_logger().info(f'Board corners found and rectified to {target_w}x{target_h}')
        return warped, M

    def detect_board_corners(self, mask_image):
        kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < 2500:
            return None

        x, y, w, h = cv2.boundingRect(best)
        roi = clean[y:y+h, x:x+w]
        if roi.size == 0:
            return None

        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=32,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
            useHarrisDetector=True,
            k=0.04,
        )

        if corners is not None and len(corners) >= 4:
            pts = np.squeeze(corners, axis=1).astype(np.float32)
            pts[:, 0] += x
            pts[:, 1] += y

            center = np.mean(pts, axis=0)
            quadrants = {
                'tl': [],
                'tr': [],
                'br': [],
                'bl': [],
            }
            for px, py in pts:
                if px < center[0] and py < center[1]:
                    quadrants['tl'].append((px, py))
                elif px >= center[0] and py < center[1]:
                    quadrants['tr'].append((px, py))
                elif px >= center[0] and py >= center[1]:
                    quadrants['br'].append((px, py))
                else:
                    quadrants['bl'].append((px, py))

            if all(quadrants.values()):
                strong = []
                for key, pts_in_quad in quadrants.items():
                    best_pt = max(
                        pts_in_quad,
                        key=lambda p: np.hypot(p[0] - center[0], p[1] - center[1]),
                    )
                    strong.append(best_pt)

                corners = np.array([
                    strong[0],
                    strong[1],
                    strong[2],
                    strong[3],
                ], dtype=np.float32)
                return self.order_corners(corners)

        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return self.order_corners(corners)

        hull = cv2.convexHull(best)
        if len(hull) == 4:
            corners = hull.reshape(4, 2).astype(np.float32)
            return self.order_corners(corners)

        rect = cv2.minAreaRect(best)
        box = cv2.boxPoints(rect)
        return self.order_corners(np.array(box, dtype=np.float32))

    def stabilize_corners(self, corners):
        if self.prev_corners is None:
            self.prev_corners = corners
            return corners

        jump = np.linalg.norm(corners - self.prev_corners, axis=1)
        max_jump = np.max(jump)
        if max_jump > self.corner_jump_threshold:
            self.get_logger().info(
                f'Corner detection jump too large ({max_jump:.1f}px); keeping previous corners')
            return self.prev_corners

        smoothed = self.prev_corners * (1.0 - self.corner_smoothing_alpha) + corners * self.corner_smoothing_alpha
        self.prev_corners = smoothed
        return smoothed

    def publish_homography(self, matrix):
        msg = HomographyMatrix()
        msg.data = matrix.reshape(9).tolist()
        self.homography_pub.publish(msg)

    def order_corners(self, corners):
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).reshape(-1)
        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        tr = corners[np.argmin(diff)]
        bl = corners[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def board_slots(self, board_image):
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            slot_points.append((float(cx), float(cy)))
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
