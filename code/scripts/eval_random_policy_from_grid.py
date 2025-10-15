# eval_random_policy_from_grid.py
import argparse, numpy as np, pandas as pd, random

def compute_legal_actions_from_grid(grid):
    legal = []
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                continue
            row = set(grid[r, :]) - {0}
            col = set(grid[:, c]) - {0}
            br, bc = 3*(r//3), 3*(c//3)
            box = set(grid[br:br+3, bc:bc+3].reshape(-1)) - {0}
            used = row | col | box
            for d in range(1, 10):
                if d not in used:
                    legal.append((r, c, d))
    return legal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sudoku_rl_patched")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    backend = __import__(args.backend, fromlist=['*'])
    np.random.seed(args.seed); random.seed(args.seed)

    df = pd.read_csv(args.csv).sample(args.n, random_state=args.seed)
    solved, tot = 0, 0
    for _, row in df.iterrows():
        q = np.array([int(c) for c in str(row["quizzes"])], dtype=int).reshape(9,9)
        sol = np.array([int(c) for c in str(row["solutions"])], dtype=int).reshape(9,9)
        env = backend.SudokuEnv(q)
        s = env.reset()
        steps = 0
        while steps < 81:
            legal = compute_legal_actions_from_grid(s)
            if not legal:
                break
            a = random.choice(legal)
            ns, r, done, _ = env.step(a)
            s = ns; steps += 1
            if done:
                break
        solved += int((env.grid == sol).all()); tot += 1
    print(f"[RANDOM-from-grid] Solved {solved}/{tot} ({solved/tot:.3f})")

if __name__ == "__main__":
    main()