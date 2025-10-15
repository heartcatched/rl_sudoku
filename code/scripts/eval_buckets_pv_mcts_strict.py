# eval_buckets_pv_mcts_strict.py
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def import_backend(name):
    try:
        return __import__(name, fromlist=['*'])
    except Exception as e:
        print(f"[ERROR] backend import failed: {e}"); sys.exit(1)

def encode_action_idx(a):
    r,c,d=a; return (r*9+c)*9+(d-1)

def decode_action_idx(idx):
    d = (idx%9)+1; rc=idx//9; r,c=divmod(rc,9); return (r,c,d)

def legal_action_indices(env, enc):
    acts = env.legal_actions()
    return [enc(a) for a in acts] if acts else []

def run_csv_strict(backend, net, device, csv_path, n=300):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["quizzes","solutions"])
    df = df[(df["quizzes"].astype(str).str.len()==81) & (df["solutions"].astype(str).str.len()==81)]
    if len(df)>n: df = df.sample(n, random_state=42)

    enc = backend.SudokuDataset
    total=0; solved=0; total_reward=0.0
    net.eval()
    for _,row in df.iterrows():
        q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
        sol= np.array([int(ch) for ch in str(row["solutions"])], dtype=int).reshape(9,9)
        env = backend.SudokuEnv(q); s = env.reset()
        steps=0; done=False
        while not done and steps<81:
            with torch.no_grad():
                st = enc._encode_state(s).unsqueeze(0).to(device)
                p_logits, _ = net(st)
                p = torch.softmax(p_logits, dim=1).squeeze(0).cpu().numpy()
            legal_idx = legal_action_indices(env, encode_action_idx)
            if not legal_idx: break
            mask = np.full_like(p, -np.inf); mask[legal_idx]=0.0
            idx = int(np.argmax(p+mask))
            a = decode_action_idx(idx)
            ns, r, done, _ = env.step(a)
            total_reward += float(r)
            s = ns; steps+=1
        if (env.grid == sol).all(): solved+=1
        total+=1
    avg_r = total_reward/max(total,1)
    return total, solved, solved/max(total,1), avg_r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Папка с sudoku_test_h*.csv")
    ap.add_argument("--model", required=True, help="pv_mcts.pth")
    ap.add_argument("--backend", default="sudoku_rl_patched")
    ap.add_argument("--holes", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30")
    ap.add_argument("--per-bucket", type=int, default=300)
    args = ap.parse_args()

    backend = import_backend(args.backend)

    from sudoku_pv_mcts import PVNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = backend.SudokuDataset._encode_state(np.zeros((9,9), dtype=int))
    net = PVNet(in_ch=dummy.shape[0], hid=128, widen=2).to(device)
    try:
        sd = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(args.model, map_location=device)
    net.load_state_dict(sd); net.eval()

    root = Path(args.root)
    holes = [int(x) for x in args.holes.split(",")]

    rows=[]
    for h in holes:
        fname = root / f"sudoku_test_h{h}.csv"
        if not fname.exists():
            print(f"[SKIP] no file: {fname}"); continue
        total, solved, rate, avg_r = run_csv_strict(backend, net, device, str(fname), n=args.per_bucket)
        print(f"[holes={h:2d}] Tested on {total} | Solved: {solved} ({rate:.3f}) | AvgR: {avg_r:.2f}")
        rows.append({"holes":h,"total":total,"solved":solved,"solved_rate":rate,"avg_reward":avg_r})

    if rows:
        out = root / "pv_mcts_eval_summary_strict.csv"
        pd.DataFrame(rows).sort_values("holes").to_csv(out, index=False)
        print(f"[OK] Сводка сохранена: {out}")

if __name__ == "__main__":
    main()
