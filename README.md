# makemorechessmoves
Predictive text based chess engine, built using Karpathy's 
makemore engine. Instead of using on name context, we get 
the state of a board. 

# DevLog – Chess Policy Network
## Project Overview
Predict strong chess moves from board states using NN.

## Directory Layout
- /data: raw PGNs and processed board-move pairs
- /models: neural network architecture
- /train: training loop and dataset logic
- /util: preprocessed board data

## Key Concepts
- `move_index_map.json`: maps UCI moves ↔ index
- Dataset: (board_state, move_index) pairs

## ToDo / In Progress
- [x] Implement `Dataset` class
- [ ] Normalize board state input
- [ ] Try masking logits to legal moves only

## Design Decisions
- Using `python-chess` to validate moves.
- Training with categorical cross-entropy.
- Skipped augmentation for now—only master games.

## Gotchas
- Some UCI moves like `a1a1` are legal in format but useless.
- PGN files often include annotations—strip before parsing.

## April 21, 2025 – 3hr Session

- Went through the output tensor build
- Hit problem: empty label tensor when illegal moves snuck in
- Solution: filtered with `board.is_legal(move)`
- Next: build context for training
- embedding vector is an 8x8 vector (or 64x1) with integer entries in [0,12]

