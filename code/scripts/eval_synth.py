import torch, numpy as np
from sudoku_rl_patched import make_simple_sudoku, SudokuEnv, DQNAgent, test_model_on_sudoku

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = SudokuEnv(np.zeros((9,9), dtype=int))
agent = DQNAgent(env, device)
agent.q.load_state_dict(torch.load("sudoku_dqn_final_10empty.pth", map_location=device))
agent.q.eval(); agent.eps = 0.0

for k in range(1, 11):
    puzzles = make_simple_sudoku(empty_cells=k, n_variants=300)
    print(f"\n=== k={k} ===")
    test_model_on_sudoku(agent, puzzles, verbose=False)