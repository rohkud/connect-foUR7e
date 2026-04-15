import rclpy
from rclpy.node import Node

class GameSolver(Node):
    def __init__(self):
        super().__init__('game_solver')
        self.declare_parameter('color', 'red')
        self.color = self.get_parameter('color').value
        self.player = 1 if self.color == 'red' else 2
        # Hardcoded board: 0=empty, 1=red, 2=yellow
        self.board = [
            [0, 0, 0, 0, 0, 0, 0],  # top
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0, 0],
            [0, 1, 2, 1, 0, 0, 0],
            [1, 2, 1, 2, 0, 0, 0],  # bottom
        ]
        move = self.get_best_move(self.board, self.player)
        self.get_logger().info(f"Optimal move for {self.color}: column {move}")

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

    def get_best_move(self, board, player, depth=5):
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
    rclpy.spin_once(node, timeout_sec=0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()