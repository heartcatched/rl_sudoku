# eval_random_policy.py
import argparse, numpy as np, torch, random, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sudoku_rl_patched")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    backend = __import__(args.backend, fromlist=['*'])
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    df = pd.read_csv(args.csv).sample(args.n, random_state=args.seed)
    solved, tot = 0, 0
    for _, row in df.iterrows():
        q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
        sol = np.array([int(ch) for ch in str(row["solutions"])], dtype=int).reshape(9,9)
        env = backend.SudokuEnv(q)
        s = env.reset()
        steps = 0
        while steps < 200:  
            acts = env.legal_actions()
            if not acts: break
            a = random.choice(acts)
            ns, r, done, _ = env.step(a)
            s = ns; steps += 1
            if done: break
        solved += int((env.grid == sol).all()); tot += 1
    print(f"[RANDOM] Solved {solved}/{tot} ({solved/tot:.3f})")

if __name__ == "__main__":
    main()