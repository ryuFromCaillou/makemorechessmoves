import chess
import json

def generate_uci_move_list():
    '''Creates an index mapping for all possible moves (4672)
    analogously, makemorenames indexes every possible move (letter)
    we perform softmax on this tensor
    '''
    
    all_moves = set()
    board = chess.Board()
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            move = chess.Move(from_sq, to_sq)
            all_moves.add(move.uci())
            # Add promotions
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                all_moves.add(promo_move.uci())
    return sorted(all_moves)

def save_move_index_map(path="data/move_index_map.json"):
    moves = generate_uci_move_list()
    uci_to_index = {uci: i for i, uci in enumerate(moves)}
    with open(path, "w") as f:
        json.dump(uci_to_index, f)
