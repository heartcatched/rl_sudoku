# eval_buckets_dqn_strict_inline.py
import argparse, sys, re, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def import_backend(name: str):
    try:
        return __import__(name, fromlist=['*'])
    except Exception as e:
        print(f"[ERROR] import backend '{name}': {e}")
        sys.exit(1)

def compute_legal_actions_from_grid(grid: np.ndarray):
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

@torch.no_grad()
def select_action_greedy(backend, agent, state: np.ndarray):
    enc = backend.SudokuDataset
    legal = compute_legal_actions_from_grid(state)  
    if not legal:
        return None
    st = enc._encode_state(state).unsqueeze(0).to(agent.device)
    q = agent.q(st).squeeze(0)
    mask = torch.full_like(q, -float('inf'), device=agent.device)
    idxs = [enc._encode_action(a) for a in legal]
    mask[idxs] = 0.0
    best_idx = (q + mask).argmax().item()
    d = (best_idx % 9) + 1
    rc = best_idx // 9
    r, c = divmod(rc, 9)
    return (r, c, d)

def evaluate_with_gt(backend, agent, data, max_steps=81):
    total = 0; solved = 0; total_reward = 0.0
    for (p, sol) in data:
        env = backend.SudokuEnv(p)
        s = env.reset(); agent.env = env
        done = False; steps = 0; ep_r = 0.0
        while not done and steps < max_steps:
            a = select_action_greedy(backend, agent, s)
            if a is None:
                break
            ns, r, done, _ = env.step(a)
            ep_r += float(r); s = ns; steps += 1
        total_reward += ep_r
        ok = (env.grid == sol).all()
        solved += int(ok); total += 1
    return dict(total=total, solved=solved,
                solved_rate=(solved/total if total else 0.0),
                avg_reward=(total_reward/total if total else 0.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="sudoku_rl_patched")
    ap.add_argument("--test_n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--glob", default="sudoku_test_h*.csv")
    args = ap.parse_args()

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    backend = import_backend(args.backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_env = backend.SudokuEnv(np.zeros((9,9), dtype=int))
    agent = backend.DQNAgent(dummy_env, device)
    agent.q.eval()
    try:
        sd = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(args.model, map_location=device)
    agent.q.load_state_dict(sd)

    root = Path(args.root)
    files = sorted(root.glob(args.glob))
    if not files:
        print(f"[ERROR] Не нашли {args.glob} в {root}")
        sys.exit(2)

    rows = []
    for f in files:
        m = re.search(r"h(\d+)", f.stem)
        holes = int(m.group(1)) if m else -1
        df = pd.read_csv(f)
        if ("quizzes" not in df.columns) or ("solutions" not in df.columns):
            print(f"[SKIP] {f.name}: нет quizzes/solutions")
            continue
        sub = df[df["solutions"].astype(str).str.len().eq(81)]
        if sub.empty:
            print(f"[SKIP] {f.name}: нет валидных solutions")
            continue
        sub = sub.sample(min(args.test_n, len(sub)), random_state=args.seed)

        data = []
        for _, row in sub.iterrows():
            q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
            sol = np.array([int(ch) for ch in str(row["solutions"])], dtype=int).reshape(9,9)
            data.append((q, sol))

        res = evaluate_with_gt(backend, agent, data)
        print(f"[holes={holes:>2}] Tested on {res['total']} | Solved: {res['solved']} ({res['solved_rate']:.3f}) | AvgR: {res['avg_reward']:.2f}")
        rows.append({"holes": holes, **res})

    if rows:
        out = pd.DataFrame(rows).sort_values("holes")
        out_path = root / "dqn_eval_summary_strict.csv"
        out.to_csv(out_path, index=False)
        print("\n=== DQN STRICT summary ===")
        print(out.to_string(index=False))
        print(f"\n[OK] Сводка сохранена: {out_path}")
    else:
        print("[ERROR] Не удалось собрать ни одной метрики.")
        sys.exit(3)

if __name__ == "__main__":
    main()