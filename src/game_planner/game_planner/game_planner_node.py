#!/usr/bin/env python3
"""
================================================================================
Game Planner Node (game_planner_node.py)
================================================================================

PURPOSE:
    Reads the current game board state from game_state and maintains a 
    probabilistic board state using weighted averaging over time. Sends 
    the board to solver on a timer (default 3 seconds).

SUBSCRIPTIONS:
    - /game_state/board: Int8MultiArray containing 6x7 board state
        (0=empty, 1=red, 2=yellow)

PUBLISHES:
    - /game_solver/board: Int8MultiArray containing board for solver to evaluate
    - /game_planner/move: Int8 - The recommended column to play (0-6)

PARAMETERS:
    - player_color: "red" or "yellow" (default: "red")
    - solve_interval: Timer interval in seconds (default: 3.0)
    - alpha: Exponential moving average weight (default: 0.3)
        Higher = more weight on recent observations

================================================================================
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, Int8
import numpy as np


class GamePlanner(Node):
    def __init__(self):
        super().__init__('game_planner')
        
        self.declare_parameter('player_color', 'red')
        self.declare_parameter('solve_interval', 3.0)
        self.declare_parameter('alpha', 0.3)
        
        self.player_color = self.get_parameter('player_color').value
        self.solve_interval = self.get_parameter('solve_interval').value
        self.alpha = self.get_parameter('alpha').value
        
        # Probabilistic board state: 3 channels (empty, red, yellow) for each cell
        # Shape: (6, 7, 3) - probabilities for each cell being empty/red/yellow
        self.prob_board = np.zeros((6, 7, 3), dtype=np.float32)
        self.prob_board[:, :, 0] = 1.0  # Start with all cells empty
        
        # Raw observation count for confidence tracking
        self.observation_count = 0
        
        # Latest deterministic board (for comparison)
        self.latest_board = None
        
        # Subscribe to game state board
        self.board_sub = self.create_subscription(
            Int8MultiArray,
            '/game_state/board',
            self.board_callback,
            5
        )
        
        # Publish to game solver
        self.solver_pub = self.create_publisher(
            Int8MultiArray,
            '/game_solver/board',
            5
        )
        
        # Subscribe to solver solution
        self.solution_sub = self.create_subscription(
            Int8,
            '/game_solver/move',
            self.solution_callback,
            5
        )
        
        # Publish recommended move
        self.move_pub = self.create_publisher(
            Int8,
            '/game_planner/move',
            5
        )
        
        # Timer to send board to solver periodically
        self.solve_timer = self.create_timer(
            self.solve_interval,
            self.timer_callback
        )
        
        self.get_logger().info(f'Game planner started for {self.player_color}')
        self.get_logger().info(f'Solve interval: {self.solve_interval}s, alpha: {self.alpha}')
        self.latest_solution = None

    def board_callback(self, msg):
        """Receive board state from game_state and update probabilistic state."""
        if len(msg.data) != 42:
            self.get_logger().error(f'Invalid board size: {len(msg.data)}')
            return
        
        self.latest_board = list(msg.data)
        
        # Convert flat array to 2D
        obs_board = np.array(msg.data).reshape(6, 7)
        
        # Update probabilistic board using exponential moving average
        self.observation_count += 1
        
        for row in range(6):
            for col in range(7):
                cell_value = obs_board[row, col]  # 0=empty, 1=red, 2=yellow
                
                # Create one-hot encoding of observation
                obs_probs = np.zeros(3, dtype=np.float32)
                if cell_value == 0:
                    obs_probs[0] = 1.0
                elif cell_value == 1:
                    obs_probs[1] = 1.0
                elif cell_value == 2:
                    obs_probs[2] = 1.0
                
                # Exponential moving average update
                self.prob_board[row, col] = (
                    self.alpha * obs_probs + 
                    (1 - self.alpha) * self.prob_board[row, col]
                )
        
        self.get_logger().debug(f'Updated probabilistic board (obs #{self.observation_count})')

    def get_deterministic_board(self):
        """Convert probabilistic board to deterministic (most likely state)."""
        # For each cell, pick the most likely state
        # But apply confidence threshold - if no state exceeds threshold, keep empty
        deterministic = np.zeros(42, dtype=np.int8)
        
        for row in range(6):
            for col in range(7):
                probs = self.prob_board[row, col]
                max_idx = np.argmax(probs)
                max_prob = probs[max_idx]
                
                # Only commit if confidence is high enough (> 0.5)
                if max_prob > 0.5:
                    # Map index to cell value: 0=empty, 1=red, 2=yellow
                    deterministic[row * 7 + col] = max_idx
                else:
                    # Default to empty if uncertain
                    deterministic[row * 7 + col] = 0
        
        return deterministic.tolist()

    def timer_callback(self):
        """Timer callback - send current probabilistic board to solver."""
        if self.observation_count == 0:
            self.get_logger().debug('No observations yet, skipping solver request')
            return
        
        # Get deterministic board from probabilistic state
        board_data = self.get_deterministic_board()
        
        # Log confidence info
        avg_confidence = np.mean(np.max(self.prob_board, axis=2))
        self.get_logger().info(f'Sending to solver (avg confidence: {avg_confidence:.2f})')
        
        # Publish to solver
        solver_msg = Int8MultiArray()
        solver_msg.data = board_data
        self.solver_pub.publish(solver_msg)

    def solution_callback(self, msg):
        """Receive solution from game_solver and publish."""
        self.latest_solution = msg.data
        self.get_logger().info(f'Received solution: play in column {msg.data}')
        
        # Publish the move
        move_msg = Int8()
        move_msg.data = msg.data
        self.move_pub.publish(move_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GamePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()