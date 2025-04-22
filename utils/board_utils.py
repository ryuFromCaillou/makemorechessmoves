import chess
import numpy as np

PIECE_TO_IDX = {
    None: 0,
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

def encode_board(board):
    board_array = np.zeros((8, 8), dtype=np.int64)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        row = 7 - (square // 8)
        col = square % 8

        if piece is not None:
            base = PIECE_TO_IDX[piece.piece_type]
            offset = 0 if piece.color == chess.WHITE else 6
            board_array[row][col] = base + offset
        else:
            board_array[row][col] = 0  # empty

    return board_array  # shape: [8,8] of ints in [0,12]
