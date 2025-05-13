from chess import pgn
from utils.board_utils import encode_board
import torch

def load_dataset(raw):
    with open(raw, "r") as file:
        game = pgn.read_game(file)
        board = game.board()
        X, Y = [], []

        for move in game.mainline_moves():
           board_state = encode_board(board)  # BEFORE the move
           try:
               move_index = move_list.index(move.uci()) # move is INDEXED here, move list starts from first move
               X.append(board_state) # first board state is neutral board, it must be appended to have good beginning game
               Y.append(move_index_map[move.uci()])
               board.push(move)  # Move AFTER data capture
           except ValueError:
               print(f"Move {move.uci()} not found in move_list. Skipping this move.")
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

if __name__ == "__main__": 
    pass