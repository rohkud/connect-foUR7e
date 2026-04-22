"""
================================================================================
Game Solver Node (game_solver_node.py)
================================================================================

PURPOSE:
    Determines optimal Connect Four moves for the robot using minimax algorithm
    with alpha-beta pruning. Serves as the AI decision engine for the robot.

SUBSCRIPTIONS:
    - /game_solver/board: Int8MultiArray containing 6x7 board state

PUBLISHES:
    - /game_solver/move: Int8 - The recommended column to play (0-6)

METHODS:
    - get_row(board, col): Returns lowest empty row in column, None if full
    - check_win(board, player): Detects if player has 4-in-a-row
    - is_winning_move(board, col, player): Tests if move wins the game
    - get_best_move(board, player, depth): Main minimax interface
    - minimax(board, depth, maximizing): Recursive game tree search
    - score_position(board): Heuristic evaluation function

OUTPUTS:
    - Publishes to /game_solver/move: "Optimal move for {color}: column {move}"

================================================================================
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, Int8


class GameSolver(Node):
    def __init__(self):
        super().__init__('game_solver')
        self.declare_parameter('color', 'red')
        self.color = self.get_parameter('color').value
        self.player = 1 if self.color == 'red' else 2
        
        # Current board state (received from game_planner)
        self.board = None
        
        # Subscribe to board from game_planner
        self.board_sub = self.create_subscription(
            Int8MultiArray,
            '/game_solver/board',
            self.board_callback,
            5
        )
        
        # Publish solution
        self.solution_pub = self.create_publisher(
            Int8,
            '/game_solver/move',
            5
        )
        
        self.get_logger().info(f'Game solver node started for {self.color}')

    def board_callback(self, msg):
        """Receive board from game_planner and compute best move."""
        # Convert flat array to 2D board
        data = msg.data
        if len(data) != 42:  # 6x7 = 42
            self.get_logger().error(f'Invalid board size: {len(data)}, expected 42')
            return
        
        # Reshape to 6x7 (row-major)
        self.board = [data[i*7:(i+1)*7] for i in range(6)]
        
        self.get_logger().info('Received board, computing best move...')
        
        # Compute best move
        move = self.get_best_move(self.board, self.player)
        
        if move is not None:
            self.get_logger().info(f"Optimal move for {self.color}: column {move}")
            
            # Publish solution
            move_msg = Int8()
            move_msg.data = move
            self.solution_pub.publish(move_msg)
        else:
            self.get_logger().warn('No valid move found (board may be full)')

    def get_row(self, board, col):
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                return r
        return None

    def check_win(self, board, player):
        # Horizontal
        for r in range(6):
            for c in range(4):
                if all(board[r][c + i] == player for i in range(4)):
                    return True
        # Vertical
        for c in range(7):
            for r in range(3):
                if all(board[r + i][c] == player for i in range(4)):
                    return True
        # Diagonal /
        for r in range(3):
            for c in range(4):
                if all(board[r + i][c + i] == player for i in range(4)):
                    return True
        # Diagonal \
        for r in range(3):
            for c in range(3, 7):
                if all(board[r + i][c - i] == player for i in range(4)):
                    return True
        return False

    def is_winning_move(self, board, col, player):
        row = self.get_row(board, col)
        if row is None:
            return False
        board[row][col] = player
        win = self.check_win(board, player)
        board[row][col] = 0
        return win
    # TODO: Make depth = 6
    def get_best_move(self, board, player, depth=2):
        valid_moves = [c for c in range(7) if board[0][c] == 0]
        opponent = 3 - player

        def score_position(board):
            # simple heuristic
            score = 0

            # center preference
            center_col = [board[r][3] for r in range(6)]
            score += center_col.count(player) * 3

            return score

        def minimax(board, depth, maximizing):
            valid_moves = [c for c in range(7) if board[0][c] == 0]

            if self.check_win(board, player):
                return (None, 100000)
            if self.check_win(board, opponent):
                return (None, -100000)
            if depth == 0 or not valid_moves:
                return (None, score_position(board))

            if maximizing:
                best_score = -float('inf')
                best_col = valid_moves[0]

                for col in valid_moves:
                    row = self.get_row(board, col)
                    board[row][col] = player
                    _, score = minimax(board, depth - 1, False)
                    board[row][col] = 0

                    if score > best_score:
                        best_score = score
                        best_col = col

                return best_col, best_score

            else:
                best_score = float('inf')
                best_col = valid_moves[0]

                for col in valid_moves:
                    row = self.get_row(board, col)
                    board[row][col] = opponent
                    _, score = minimax(board, depth - 1, True)
                    board[row][col] = 0

                    if score < best_score:
                        best_score = score
                        best_col = col

                return best_col, best_score

        best_col, _ = minimax(board, depth, True)
        return best_col

def main(args=None):
    rclpy.init(args=args)
    node = GameSolver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()