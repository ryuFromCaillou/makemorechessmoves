import chess
import json

def generate_uci_move_list():
    '''Creates an index mapping for all possible moves (4672)
    analogously, makemorenames indexes every possible move (letter)
    we perform softmax on this tensor
    '''
    all_moves = set()
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            move = chess.Move(from_sq, to_sq)
            all_moves.add(move.uci())
            # Add promotions
            # Add promotions
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                from_rank = chess.square_rank(from_sq)
                to_rank = chess.square_rank(to_sq)
                from_file = chess.square_file(from_sq)
                to_file = chess.square_file(to_sq)
                # Only allow forward promotion (white or black)
                if (from_rank, to_rank) in [(6, 7), (1, 0)]:  # white/black promotion ranks
                    if abs(from_file - to_file) <= 1:         # straight or diagonal
                        promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                        all_moves.add(promo_move.uci())
    return sorted(all_moves)


def save_move_index_map(path="data/move_index_map.json"):
    moves = generate_uci_move_list()
    uci_to_index = {uci: i for i, uci in enumerate(moves)}
    with open(path, "w") as f:
        json.dump(uci_to_index, f)
