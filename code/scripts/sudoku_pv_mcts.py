# sudoku_pv_mcts.py
r"""
Policy+Value + MCTS (AlphaZero-style) for Sudoku
-----------------------------------------------

Backend (default: sudoku_rl_patched) must provide:
 - SudokuEnv(grid: np.ndarray)
 - SudokuDataset with:
     _encode_state(np.ndarray[9x9 int]) -> torch.Tensor [C,H,W]
     _encode_action((r,c,d)) -> int in [0..728] where index = (r*9+c)*9 + (d-1)
 - make_simple_sudoku(empty_cells:int, n_variants:int) -> List[np.ndarray]

Two-stage training:
1) Behavior Cloning from CSV (quizzes, solutions) [OPTIONAL]
   - Policy: supervised CE toward solution move sequence
   - Value : +1 for all states along the expert trace (light bootstrap)

2) Self-play with MCTS
   - Generate (state, pi, z) where pi is visit-count distribution from MCTS, z is game outcome (+1 solved, -1 fail)
   - Loss: CE(pi, policy_logits) + lambda_v * MSE(z, value) + weight_decay

Strict evaluation:
 - if CSV with valid solutions: "Solved" only if env.grid == solution
 - else: structural sudoku validity (rows/cols/boxes are perms of {1..9})

Example (Windows PowerShell):

# BC pretraining (5 epochs), then self-play with curriculum 10→30
python sudoku_pv_mcts.py `
  --backend sudoku_rl_patched `
  --csv C:\Users\ilya\Downloads\data\sudoku.csv `
  --limit 8000 `
  --bc-epochs 5 --bc-steps 3000 `
  --episodes 2000 `
  --mcts-sims 80 `
  --train-per-move 2 `
  --curriculum `
  --holes-stages "10,12,14,16,18,20,22,24,26,28,30" `
  --episodes-per-stage 200

# Strict eval on a prepared bucket
python sudoku_pv_mcts.py `
  --backend sudoku_rl_patched `
  --test-model pv_mcts.pth `
  --csv C:\Users\ilya\Downloads\data\sudoku_test_h20.csv `
  --test-n 300
"""

import argparse, sys, time, math, random, copy
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def import_backend(backend_name: str):
    try:
        mod = __import__(backend_name, fromlist=['*'])
        return mod
    except Exception as e:
        print(f"[ERROR] Could not import backend '{backend_name}': {e}")
        sys.exit(1)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def encode_action_idx(a: Tuple[int,int,int]) -> int:
    r,c,d = a
    return (r*9 + c)*9 + (d-1)

def decode_action_idx(idx: int) -> Tuple[int,int,int]:
    d = (idx % 9) + 1
    rc = idx // 9
    r, c = divmod(rc, 9)
    return (r,c,d)

def legal_action_indices(env, encode_action_fn) -> List[int]:
    acts = env.legal_actions()
    if not acts: return []
    return [encode_action_fn(a) for a in acts]

def is_valid_sudoku_grid(grid: np.ndarray) -> bool:
    S = set(range(1, 10))
    for r in range(9):
        if set(grid[r, :]) != S: return False
    for c in range(9):
        if set(grid[:, c]) != S: return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(grid[br:br+3, bc:bc+3].reshape(-1)) != S: return False
    return True

class PVNet(nn.Module):
    """
    Policy head: logits over 81*9 = 729 actions
    Value  head: scalar in [-1,1] via tanh
    """
    def __init__(self, in_ch=3, hid=128, widen=2):
        super().__init__()
        ch = hid
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch*widen, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head_policy = nn.Sequential(
            nn.Conv2d(ch*widen, ch, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(ch*9*9, 729)
        )
        self.head_value = nn.Sequential(
            nn.Conv2d(ch*widen, ch, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(ch*9*9, ch),
            nn.ReLU(inplace=True),
            nn.Linear(ch, 1),
            nn.Tanh()
        )

    def forward(self, x):  
        h = self.trunk(x)
        p = self.head_policy(h)  
        v = self.head_value(h)   
        return p, v.squeeze(1)

class MCTSNode:
    __slots__ = ("P", "N", "W", "Q", "children", "is_expanded")
    def __init__(self, prior: float):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded = False

class MCTS:
    def __init__(self, net: PVNet, device, env_ctor, state_encoder, encode_action_fn,
                 c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25, sims=80):
        self.net = net
        self.device = device
        self.env_ctor = env_ctor
        self.enc_state = state_encoder
        self.enc_action = encode_action_fn
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_eps = dirichlet_eps
        self.sims = sims

    def policy_value(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            st = self.enc_state(state).unsqueeze(0).to(self.device)
            p_logits, v = self.net(st)
            p = torch.softmax(p_logits, dim=1).squeeze(0).cpu().numpy() 
            return p, float(v.item())

    def run(self, root_env, game_max_len=81, temperature=1.0) -> Tuple[np.ndarray, int]:
        """
        Run MCTS from root_env; return (pi over 729, chosen action index)
        """
        root_state = root_env.grid.copy()
        legal = legal_action_indices(root_env, self.enc_action)
        if not legal:
            pi = np.zeros(729, dtype=np.float32)
            return pi, -1

        p_root, v_root = self.policy_value(root_state)
        mask = np.full_like(p_root, -np.inf)
        mask[legal] = 0.0
        p_masked = softmax_np(p_root + mask)
        root = MCTSNode(1.0)
        root.is_expanded = True
        for a_idx in legal:
            root.children[a_idx] = MCTSNode(p_masked[a_idx])

        if self.dir_eps > 0 and len(legal) > 2:
            noise = np.random.dirichlet([self.dir_alpha]*len(legal))
            for i, a_idx in enumerate(legal):
                root.children[a_idx].P = (1 - self.dir_eps)*root.children[a_idx].P + self.dir_eps*noise[i]

        for _ in range(self.sims):
            node = root
            env = self._clone_env(root_env)

            path = []
            while node.is_expanded and node.children:
                a_best = self._select_child(node)
                path.append((node, a_best))
                a = decode_action_idx(a_best)
                ns, r, done, _ = env.step(a)
                if done:
                    z = 1.0 if (ns != 0).all() else -1.0
                    self._backprop(path, z)
                    node = None
                    break
                node = node.children.get(a_best, None)
                if node is None:
                    break

            if node is not None:
                st = env.grid.copy()
                legal2 = legal_action_indices(env, self.enc_action)
                if not legal2:
                    z = -1.0
                    self._backprop(path, z); continue
                p, v = self.policy_value(st)
                mask2 = np.full_like(p, -np.inf)
                mask2[legal2] = 0.0
                p2 = softmax_np(p + mask2)

                node.is_expanded = True
                for a_idx in legal2:
                    node.children[a_idx] = MCTSNode(p2[a_idx])
                self._backprop(path, v)

        visits = np.zeros(729, dtype=np.float32)
        for a_idx, child in root.children.items():
            visits[a_idx] = child.N
        if temperature <= 1e-8:
            best = np.argmax(visits)
            pi = np.zeros_like(visits); pi[best] = 1.0
            return pi, int(best)
        else:
            pi = visits ** (1.0 / max(temperature, 1e-6))
            if pi.sum() > 0:
                pi = pi / pi.sum()
            else:
                pi = softmax_np(visits)
            a_choice = int(np.random.choice(729, p=pi))
            return pi, a_choice

    def _select_child(self, node: MCTSNode) -> int:
        total_N = max(1, sum(child.N for child in node.children.values()))
        best, best_score = None, -1e9
        for a_idx, ch in node.children.items():
            U = self.c_puct * ch.P * (math.sqrt(total_N) / (1 + ch.N))
            score = ch.Q + U
            if score > best_score:
                best_score = score; best = a_idx
        return best

    def _backprop(self, path, z: float):
        for node, a_idx in reversed(path):
            child = node.children[a_idx]
            child.N += 1
            child.W += z
            child.Q = child.W / child.N

    def _clone_env(self, env):
        new_env = self.env_ctor(env.grid.copy())
        return new_env

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e/s if s>0 else np.ones_like(x)/len(x)

class ReplayAZ:
    """Stores (state_tensor, pi (729,), z) for self-play."""
    def __init__(self, cap=300_000):
        from collections import deque
        self.buf = deque(maxlen=cap)
    def push(self, s_t: torch.Tensor, pi: np.ndarray, z: float):
        self.buf.append((s_t.cpu(), pi.astype(np.float32), float(z)))
    def sample(self, bs: int):
        bs = min(bs, len(self.buf))
        idxs = np.random.choice(len(self.buf), size=bs, replace=False)
        items = [self.buf[i] for i in idxs]
        S = torch.stack([x[0] for x in items], dim=0)
        Pi = torch.tensor([x[1] for x in items], dtype=torch.float32)
        Z  = torch.tensor([x[2] for x in items], dtype=torch.float32)
        return S, Pi, Z
    def __len__(self): return len(self.buf)

def build_bc_batches_from_csv(backend, df, limit=5000, seed=42, device="cpu"):
    """
    Build (state_t, action_idx) sequences from (quiz, solution). Teacher-forcing along GT path.
    Value target for states: +1
    """
    if limit and limit < len(df):
        df = df.sample(limit, random_state=seed)
    enc = backend.SudokuDataset
    pairs = []
    for _, row in df.iterrows():
        q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
        s = np.array([int(ch) for ch in str(row["solutions"])], dtype=int).reshape(9,9)
        env = backend.SudokuEnv(q)
        state = env.reset()
        steps = 0
        while (state == 0).any() and steps < 81:
            empties = list(zip(*np.where(state == 0)))
            r, c = min(empties)
            d = int(s[r,c])
            a = (r,c,d)
            state_t = enc._encode_state(state).to(device)  
            pairs.append( (state_t, encode_action_idx(a)) )
            ns, rwd, done, _ = env.step(a)
            state = ns
            steps += 1
    return pairs

def az_loss(net: PVNet, states, pi_targets, z_targets, weight_decay=1e-4):
    p_logits, v = net(states)
    ce = torch.sum(-pi_targets * F.log_softmax(p_logits, dim=1), dim=1).mean()  
    mse = F.mse_loss(v, z_targets)
    l2 = sum((p**2).sum() for p in net.parameters())
    return ce + mse + weight_decay * l2 * 1e-6, {'ce': ce.item(), 'mse': mse.item()}

def evaluate_strict(backend, net, device, df_or_list, test_n=300, from_csv=True):
    """
    If from_csv=True -> df (with quizzes,solutions)
      strict success only if final grid == solution
    Else -> list of puzzles (np arrays), structural check
    """
    enc = backend.SudokuDataset
    total=0; solved=0; total_reward=0.0
    if from_csv:
        df = df_or_list
        if test_n and len(df)>test_n:
            df = df.sample(test_n, random_state=42)
        for _, row in df.iterrows():
            q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
            sol = np.array([int(ch) for ch in str(row["solutions"])], dtype=int).reshape(9,9)
            env = backend.SudokuEnv(q); s = env.reset()
            done=False; steps=0
            while not done and steps<81:
                with torch.no_grad():
                    st = enc._encode_state(s).unsqueeze(0).to(device)
                    p_logits, _ = net(st)
                    p = torch.softmax(p_logits, dim=1).squeeze(0).cpu().numpy()
                legal_idx = legal_action_indices(env, encode_action_idx)
                if not legal_idx: break
                mask = np.full_like(p, -np.inf); mask[legal_idx]=0.0
                idx = int(np.argmax(p + mask))
                a = decode_action_idx(idx)
                ns, r, done, _ = env.step(a)
                total_reward += float(r)
                s = ns; steps+=1
            if (env.grid == sol).all(): solved += 1
            total += 1
    else:
        puzzles = df_or_list
        for p in puzzles[:test_n]:
            env = backend.SudokuEnv(p); s = env.reset()
            done=False; steps=0
            while not done and steps<81:
                with torch.no_grad():
                    st = enc._encode_state(s).unsqueeze(0).to(device)
                    p_logits, _ = net(st)
                    p = torch.softmax(p_logits, dim=1).squeeze(0).cpu().numpy()
                legal_idx = legal_action_indices(env, encode_action_idx)
                if not legal_idx: break
                mask = np.full_like(p, -np.inf); mask[legal_idx]=0.0
                idx = int(np.argmax(p + mask))
                a = decode_action_idx(idx)
                ns, r, done, _ = env.step(a)
                total_reward += float(r)
                s = ns; steps+=1
            if is_valid_sudoku_grid(env.grid): solved += 1
            total += 1
    avg_r = total_reward / max(total,1)
    return {'total': total, 'solved': solved, 'solved_rate': solved/max(total,1), 'avg_reward': avg_r}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sudoku_rl_patched")

    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--bc-steps", type=int, default=3000)
    ap.add_argument("--bc-batch", type=int, default=128)

    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--train-per-move", type=int, default=2, help="optimizer steps per environment move")
    ap.add_argument("--replay-cap", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--mcts-sims", type=int, default=80)
    ap.add_argument("--puct", type=float, default=1.5)
    ap.add_argument("--dir-alpha", type=float, default=0.3)
    ap.add_argument("--dir-eps", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--holes-stages", type=str, default="10,12,14,16,18,20,22,24,26,28,30")
    ap.add_argument("--episodes-per-stage", type=int, default=200)

    ap.add_argument("--test-model", type=str, default=None)
    ap.add_argument("--test-synth", type=int, default=None)
    ap.add_argument("--test-n", type=int, default=300)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    backend = import_backend(args.backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    enc = backend.SudokuDataset

    dummy = enc._encode_state(np.zeros((9,9), dtype=int))
    in_ch = dummy.shape[0]
    net = PVNet(in_ch=in_ch, hid=128, widen=2).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.test_model is not None:
        print(f"Loading model from {args.test_model}")
        try:
            sd = torch.load(args.test_model, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(args.test_model, map_location=device)
        net.load_state_dict(sd); net.eval()
        if args.test_synth is not None:
            puzzles = backend.make_simple_sudoku(empty_cells=args.test_synth, n_variants=args.test_n)
            res = evaluate_strict(backend, net, device, puzzles, test_n=args.test_n, from_csv=False)
        else:
            import pandas as pd
            assert args.csv, "--csv required for strict eval with GT"
            df = pd.read_csv(args.csv)
            df = df.dropna(subset=["quizzes","solutions"])
            df = df[(df["quizzes"].astype(str).str.len()==81) & (df["solutions"].astype(str).str.len()==81)]
            res = evaluate_strict(backend, net, device, df, test_n=args.test_n, from_csv=True)
        print(f"Tested on {res['total']} | Solved: {res['solved']} ({res['solved_rate']:.3f}) | AvgR: {res['avg_reward']:.2f}")
        return

    if args.csv and args.bc_epochs>0 and args.bc_steps>0:
        import pandas as pd
        df = pd.read_csv(args.csv)
        df = df.dropna(subset=["quizzes","solutions"])
        df = df[(df["quizzes"].astype(str).str.len()==81) & (df["solutions"].astype(str).str.len()==81)]
        print(f"[BC] Building teacher-forcing sequences from CSV (limit={args.limit}) ...")
        bc_pairs = build_bc_batches_from_csv(backend, df, limit=args.limit, seed=args.seed, device=device)
        print(f"[BC] Collected pairs: {len(bc_pairs)}")
        for ep in range(1, args.bc_epochs+1):
            random.shuffle(bc_pairs)
            total = 0; ce_sum=0.0; mse_sum=0.0
            for i in range(0, min(args.bc_steps, len(bc_pairs)), args.bc_batch):
                batch = bc_pairs[i:i+args.bc_batch]
                S = torch.stack([p[0] for p in batch], dim=0).to(device)
                A = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
                pi_t = torch.zeros((S.shape[0], 729), dtype=torch.float32, device=device)
                pi_t[torch.arange(S.shape[0], device=device), A] = 1.0
                z_t = torch.ones((S.shape[0],), dtype=torch.float32, device=device)  
                loss, parts = az_loss(net, S, pi_t, z_t, weight_decay=args.weight_decay)
                optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optim.step()
                total += 1; ce_sum += parts['ce']; mse_sum += parts['mse']
            print(f"[BC] Epoch {ep}/{args.bc_epochs} | loss≈{(ce_sum+mse_sum)/max(total,1):.3f} | ce={ce_sum/max(total,1):.3f} | mse={mse_sum/max(total,1):.3f}")
        torch.save(net.state_dict(), "pv_mcts_bc.pth"); print("[SAVE] pv_mcts_bc.pth")

    if args.episodes <= 0:
        torch.save(net.state_dict(), "pv_mcts.pth"); print("[SAVE] pv_mcts.pth (after BC only)")
        return

    mcts = MCTS(net, device, backend.SudokuEnv, enc._encode_state, encode_action_idx,
                c_puct=args.puct, dirichlet_alpha=args.dir_alpha, dirichlet_eps=args.dir_eps, sims=args.mcts_sims)
    replay = ReplayAZ(cap=args.replay_cap)

    holes_schedule = [int(x) for x in args.holes_stages.split(",")] if args.curriculum else None
    stage_idx = 0; ep_in_stage = 0
    start = time.time()

    for ep in range(1, args.episodes+1):
        holes = holes_schedule[stage_idx] if holes_schedule is not None else random.randint(10,30)
        grid = backend.make_simple_sudoku(empty_cells=holes, n_variants=1)[0]
        env = backend.SudokuEnv(grid)
        s = env.reset()
        done=False; steps=0
        episode_states = []
        episode_pis = []

        while not done and steps < 81:
            pi, a_idx = mcts.run(env, game_max_len=81, temperature=args.temperature)
            if a_idx < 0: break
            episode_states.append(enc._encode_state(env.grid.copy()).to(device))
            episode_pis.append(pi.astype(np.float32))
            a = decode_action_idx(a_idx)
            ns, r, done, _ = env.step(a)
            s = ns; steps += 1

        z = 1.0 if (env.grid != 0).all() else -1.0
        for st, pi in zip(episode_states, episode_pis):
            replay.push(st, pi, z)

        for _ in range(args.train_per_move if hasattr(args,'train-per-move') else args.train_per_move):
            if len(replay) < args.batch_size: break
            S, Pi, Z = replay.sample(args.batch_size)
            S = S.to(device); Pi = Pi.to(device); Z = Z.to(device)
            loss, parts = az_loss(net, S, Pi, Z, weight_decay=args.weight_decay)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()

        if holes_schedule is not None:
            ep_in_stage += 1
            if ep_in_stage >= args.episodes_per_stage and (stage_idx+1) < len(holes_schedule):
                stage_idx += 1; ep_in_stage = 0
                print(f"[Curriculum] -> Next stage: holes={holes_schedule[stage_idx]}")

        if (ep % 20)==0 or ep==args.episodes:
            elapsed = (time.time()-start)/60.0
            print(f"[SelfPlay] Ep {ep}/{args.episodes} | steps={steps} | z={z:+.0f} | replay={len(replay)} | {elapsed:.1f} min")

        if (ep % 200)==0:
            net.eval()
            msg = []
            for h in [10,20,30]:
                puzzles = backend.make_simple_sudoku(empty_cells=h, n_variants=100)
                res = evaluate_strict(backend, net, device, puzzles, test_n=100, from_csv=False)
                msg.append(f"h{h}:{res['solved']:3d}/100({res['solved_rate']:.2f})")
            print(f"[Eval@{ep}] " + " | ".join(msg))
            net.train()

    torch.save(net.state_dict(), "pv_mcts.pth")
    print("[SAVE] pv_mcts.pth")

if __name__ == "__main__":
    main()
