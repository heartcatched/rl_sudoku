# dump_dqfd_dataset.py
import argparse, json, random, numpy as np, torch
import pandas as pd
import random, os     

def import_backend(name):
    return __import__(name, fromlist=['*'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='sudoku_rl_patched')
    ap.add_argument('--model', required=True)
    ap.add_argument('--samples', type=int, default=80000)
    ap.add_argument('--synthetic', type=int, default=10)   
    ap.add_argument('--n-variants', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mode', choices=['policy_cell','policy_digit','q_reg'], default='policy_digit')
    ap.add_argument('--out', default='dqfd_dump')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    backend = import_backend(args.backend)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy = backend.SudokuEnv(np.zeros((9,9), dtype=int))
    agent = backend.DQNAgent(dummy, device)
    sd = torch.load(args.model, map_location=device, weights_only=False)
    agent.q.load_state_dict(sd); agent.q.eval()

    from human_features_sudoku import extract_features_for_cell_digit
    enc = backend.SudokuDataset

    def legal_actions(env):
        return env.legal_actions()  

    def q_values_for_state(state):
        with torch.no_grad():
            st = backend.SudokuDataset._encode_state(state).unsqueeze(0).to(device)
            q = agent.q(st).squeeze(0).cpu().numpy()  
        return q

    rows = []
    total = 0
    puzzles = backend.make_simple_sudoku(args.synthetic, args.n_variants)
    for pz in puzzles:
        env = backend.SudokuEnv(pz)
        s = env.reset()
        step = 0
        while (s==0).any() and step < 81 and total < args.samples:
            acts = legal_actions(env)
            if not acts: break
            q = q_values_for_state(s)  
            idx_legal = [enc._encode_action(a) for a in acts]
            best_idx = max(idx_legal, key=lambda i: q[i])
            best_a = acts[idx_legal.index(best_idx)]
            for a in acts:
                r,c,d = a
                feat = extract_features_for_cell_digit(s, r, c, d)  
                aa = enc._encode_action(a)
                if args.mode == 'q_reg':
                    target = float(q[aa])
                    rows.append({'y': target, 'feat': feat.tolist()})
                elif args.mode == 'policy_digit':
                    y = d  
                    if a == best_a: rows.append({'y': y, 'feat': feat.tolist(), 'label':1})
                elif args.mode == 'policy_cell':
                    y = r*9 + c 
                    if a == best_a: rows.append({'y': y, 'feat': feat.tolist(), 'label':1})
                total += 1
                if total >= args.samples: break
            ns, r, done, _ = env.step(best_a)
            s = ns; step += 1
            if done: break


    y = np.array([row['y'] for row in rows], dtype=np.float32)
    X = np.array([row['feat'] for row in rows], dtype=np.float32)
    np.savez_compressed(f"{args.out}_{args.mode}.npz", X=X, y=y)
    print(f"[OK] Saved: {args.out}_{args.mode}.npz | X shape={X.shape}, y shape={y.shape}")

if __name__ == '__main__':
    main()