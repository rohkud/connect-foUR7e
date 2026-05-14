#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, MultiArrayDimension


class GamePlanner(Node):
    def __init__(self):
        super().__init__("game_planner")

        self.declare_parameter("alpha", 0.3)
        self.declare_parameter("publish_interval", 1.0)

        self.alpha = self.get_parameter("alpha").value
        self.publish_interval = self.get_parameter("publish_interval").value

        self.prob_board = np.zeros((6, 7, 3), dtype=np.float32)
        self.prob_board[:, :, 0] = 1.0

        self.observation_count = 0
        self.latest_board = None

        self.board_sub = self.create_subscription(
            Int8MultiArray, "/game_state/board", self.board_callback, 5
        )

        self.stable_board_pub = self.create_publisher(
            Int8MultiArray, "/game_planner/stable_board", 5
        )

        self.stable_board_timer = self.create_timer(
            self.publish_interval, self.stable_board_callback
        )

        self.get_logger().info(
            f"Game planner started: publishing stable board every {self.publish_interval}s, alpha={self.alpha}"
        )

    def board_callback(self, msg):
        if len(msg.data) != 42:
            self.get_logger().error(f"Invalid board size: {len(msg.data)}")
            return

        self.latest_board = list(msg.data)
        obs_board = np.array(msg.data, dtype=np.int8).reshape(6, 7)

        self.observation_count += 1

        for row in range(6):
            for col in range(7):
                cell_value = int(obs_board[row, col])

                obs_probs = np.zeros(3, dtype=np.float32)

                if cell_value == 0:
                    obs_probs[0] = 1.0
                elif cell_value == 1:
                    obs_probs[1] = 1.0
                elif cell_value == 2:
                    obs_probs[2] = 1.0
                else:
                    self.get_logger().warn(
                        f"Invalid cell value {cell_value} at row={row}, col={col}"
                    )
                    obs_probs[0] = 1.0

                self.prob_board[row, col] = (
                    self.alpha * obs_probs
                    + (1.0 - self.alpha) * self.prob_board[row, col]
                )

    def get_deterministic_board(self):
        deterministic = np.zeros(42, dtype=np.int8)

        for row in range(6):
            for col in range(7):
                probs = self.prob_board[row, col]
                max_idx = int(np.argmax(probs))
                max_prob = float(probs[max_idx])

                if max_prob > 0.5:
                    deterministic[row * 7 + col] = max_idx
                else:
                    deterministic[row * 7 + col] = 0

        return deterministic.tolist()

    def stable_board_callback(self):
        if self.observation_count == 0:
            return

        stable_board = self.get_deterministic_board()
        self.get_logger().info(
            f"Generated stable board:\n{np.array(stable_board).reshape(6, 7)}"
        )
        msg = Int8MultiArray()

        msg.layout.dim.append(MultiArrayDimension(label="rows", size=6, stride=42))
        msg.layout.dim.append(MultiArrayDimension(label="cols", size=7, stride=7))

        msg.data = stable_board
        self.stable_board_pub.publish(msg)

        avg_confidence = float(np.mean(np.max(self.prob_board, axis=2)))
        self.get_logger().debug(
            f"Published stable board | avg confidence={avg_confidence:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GamePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

