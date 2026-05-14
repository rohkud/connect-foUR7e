#!/usr/bin/env python3

import rclpy
from planning_interfaces.srv import SolveMove
from rclpy.node import Node


class GameSolver(Node):
    def __init__(self):
        super().__init__("game_solver")

        self.solve_srv = self.create_service(
            SolveMove, "/solve_move", self.solve_callback
        )

        self.get_logger().info("Game solver service started: /solve_move")

    def solve_callback(self, request, response):
        data = list(request.board)

        if len(data) != 42:
            response.success = False
            response.column = -1
            response.message = f"Invalid board size: {len(data)}, expected 42"
            return response

        board = [data[i * 7 : (i + 1) * 7] for i in range(6)]
        player = int(request.player)

        if player not in [1, 2]:
            response.success = False
            response.column = -1
            response.message = f"Invalid player: {player}, expected 1 or 2"
            return response

        self.get_logger().info(f"SolveMove called for player {player}")

        move = self.get_best_move(board, player)

        if move is None:
            response.success = False
            response.column = -1
            response.message = "No valid move found"
            return response

        response.success = True
        response.column = int(move)
        response.message = f"Best move is column {move}"

        self.get_logger().info(f"Optimal move for player {player}: column {move}")
        return response

    def get_row(self, board, col):
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                return r
        return None

    def check_win(self, board, player):
        for r in range(6):
            for c in range(4):
                if all(board[r][c + i] == player for i in range(4)):
                    return True

        for c in range(7):
            for r in range(3):
                if all(board[r + i][c] == player for i in range(4)):
                    return True

        for r in range(3):
            for c in range(4):
                if all(board[r + i][c + i] == player for i in range(4)):
                    return True

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

    def get_best_move(self, board, player, depth=2):
        valid_moves = [c for c in range(7) if board[0][c] == 0]

        if not valid_moves:
            return None

        opponent = 3 - player

        for col in valid_moves:
            if self.is_winning_move(board, col, player):
                return col

        for col in valid_moves:
            if self.is_winning_move(board, col, opponent):
                return col

        def score_position(board):
            score = 0
            center_col = [board[r][3] for r in range(6)]
            score += center_col.count(player) * 3
            return score

        def minimax(board, depth, maximizing):
            valid_moves = [c for c in range(7) if board[0][c] == 0]

            if self.check_win(board, player):
                return None, 100000

            if self.check_win(board, opponent):
                return None, -100000

            if depth == 0 or not valid_moves:
                return None, score_position(board)

            if maximizing:
                best_score = -float("inf")
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

            best_score = float("inf")
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


if __name__ == "__main__":
    main()

