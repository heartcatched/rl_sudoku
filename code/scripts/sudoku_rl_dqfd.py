# sudoku_rl_dqfd.py
r"""
DQfD for Sudoku (Behavior Cloning -> fine-tune with large-margin loss)
with curriculum by hole count and expert dataset caching to NPZ.

Backend module (default: sudoku_rl_patched) must define:
 - SudokuEnv
 - DQNAgent  (fields: .q, .q_target, .device)
 - SudokuDataset with:
     _encode_state(state: np.ndarray) -> torch.Tensor [C,H,W]
     _encode_action(a: Tuple[int,int,int]) -> int in [0..728]
 - make_simple_sudoku(empty_cells:int, n_variants:int) -> List[np.ndarray]

CSV format (for expert data):
 - columns: 'quizzes', 'solutions' (each is 81-char 0..9 string)

Outputs:
 - dqn_dqfd.pth  : final Q-network weights (state_dict)

Examples (Windows):
  # BC + DQfD with curriculum, expert cached to NPZ:
  python sudoku_rl_dqfd.py ^
    --backend sudoku_rl_patched ^
    --csv "C:\Users\ilya\Downloads\data\sudoku.csv" --limit 8000 ^
    --expert-limit 150000 --save-expert-npz "C:\Users\ilya\Downloads\expert_150k.npz" ^
    --bc-epochs 6 --bc-steps-per-epoch 4000 ^
    --episodes 3000 --train-steps-per-episode 16 ^
    --expert-ratio 0.7 --dqfd-margin 0.8 --nstep 3 ^
    --eps-start 0.2 --eps-final 0.05 --eps-decay 0.9997 ^
    --curriculum --holes-stages "5,7,9,12,15,18,21,24,27,30" --episodes-per-stage 200

  # Train only from cached expert NPZ (fast):
  python sudoku_rl_dqfd.py ^
    --backend sudoku_rl_patched ^
    --load-expert-npz "C:\Users\ilya\Downloads\expert_150k.npz" ^
    --bc-epochs 8 --bc-steps-per-epoch 4000 ^
    --episodes 1500 --train-steps-per-episode 16 ^
    --expert-ratio 0.7

  # Evaluate trained model:
  python sudoku_rl_dqfd.py ^
    --backend sudoku_rl_patched ^
    --test-model dqn_dqfd.pth ^
    --test-synth 10 --test-n 300
"""

import argparse, sys, math, random, time
from typing import List, Tuple
import numpy as np
import torch
import copy



def import_backend(backend_name: str):
    try:
        mod = __import__(backend_name, fromlist=['*'])
        return mod
    except Exception as e:
        print(f"[ERROR] Could not import backend '{backend_name}': {e}")
        sys.exit(1)



class DQfDReplay:
    """Two pools (expert/online) with uniform sampling."""
    def __init__(self, cap_expert=200_000, cap_online=200_000, nstep=3, gamma=0.99):
        from collections import deque
        self.exp = deque(maxlen=cap_expert)
        self.onl = deque(maxlen=cap_online)
        self.nstep = int(nstep)
        self.gamma = float(gamma)

    def push_expert(self, transitions: List[Tuple[np.ndarray, Tuple[int,int,int], float, np.ndarray, bool]]):
        self.exp.extend(transitions)

    def push_online(self, transition):
        self.onl.append(transition)

    def sample(self, batch_size: int, expert_ratio: float = 0.5):
        ne = min(int(batch_size * expert_ratio), len(self.exp))
        no = min(batch_size - ne, len(self.onl))
        exp_batch = random.sample(self.exp, ne) if ne > 0 else []
        onl_batch = random.sample(self.onl, no) if no > 0 else []
        return exp_batch + onl_batch

    def __len__(self):
        return len(self.exp) + len(self.onl)



def build_expert_transitions_from_csv(backend, df, limit=3000, seed=42) -> List[Tuple[np.ndarray, Tuple[int,int,int], float, np.ndarray, bool]]:
    """Build expert (s,a,r,s',done) using ground-truth solutions (teacher forcing)."""
    import pandas as pd
    assert {'quizzes','solutions'} <= set(df.columns), "CSV must have 'quizzes' and 'solutions'"
    df = df.sample(limit, random_state=seed) if limit else df
    transitions = []

    for q, s in zip(df['quizzes'], df['solutions']):
        grid0 = np.array([int(ch) for ch in q], dtype=int).reshape(9,9)
        sol   = np.array([int(ch) for ch in s], dtype=int).reshape(9,9)
        env = backend.SudokuEnv(grid0)
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 81 and (state == 0).any():
            empties = list(zip(*np.where(state == 0)))
            cand_counts = []
            for (r,c) in empties:
                row = set(state[r, :]) - {0}
                col = set(state[:, c]) - {0}
                br, bc = 3*(r//3), 3*(c//3)
                box = set(state[br:br+3, bc:bc+3].flatten()) - {0}
                used = row | col | box
                cand_counts.append(9 - len(used))
            r, c = empties[int(np.argmin(cand_counts))]
            d = int(sol[r, c])

            a = (r, c, d)
            ns, rwd, done, _ = env.step(a)
            transitions.append((state.copy(), a, float(rwd), ns.copy(), bool(done)))
            state = ns
            steps += 1
    return transitions


def pack_expert_npz(path, transitions):
    """Save list of (s, (r,c,d), r, s', done) to compressed NPZ."""
    N = len(transitions)
    states = np.zeros((N, 9, 9), dtype=np.int8)
    actions = np.zeros((N, 3), dtype=np.int8)
    rewards = np.zeros((N,), dtype=np.float32)
    next_states = np.zeros((N, 9, 9), dtype=np.int8)
    dones = np.zeros((N,), dtype=bool)
    for i, (s, a, r, s2, d) in enumerate(transitions):
        states[i] = s
        actions[i] = np.array(a, dtype=np.int8)
        rewards[i] = r
        next_states[i] = s2
        dones[i] = d
    np.savez_compressed(path, states=states, actions=actions,
                        rewards=rewards, next_states=next_states, dones=dones)


def load_expert_npz(path):
    """Load compressed NPZ to list of transitions."""
    z = np.load(path, allow_pickle=False)
    states = z["states"]; actions = z["actions"]; rewards = z["rewards"]
    next_states = z["next_states"]; dones = z["dones"]
    transitions = []
    for i in range(states.shape[0]):
        s = states[i].astype(np.int16)
        r, c, d = actions[i].tolist()
        s2 = next_states[i].astype(np.int16)
        transitions.append((s, (int(r),int(c),int(d)), float(rewards[i]), s2, bool(dones[i])))
    return transitions



def select_action_greedy(backend, agent, state: np.ndarray, eps: float = 0.0):
    """Argmax Q among legal actions with epsilon-greedy."""
    legal = agent.env.legal_actions()
    if (not legal) or (len(legal) == 0):
        return None
    if random.random() < eps:
        return random.choice(legal)

    with torch.no_grad():
        st = backend.SudokuDataset._encode_state(state).unsqueeze(0).to(agent.device)
        q = agent.q(st).squeeze(0)  
        mask = torch.full_like(q, -float('inf'), device=agent.device)
        idxs = [backend.SudokuDataset._encode_action(a) for a in legal]
        mask[idxs] = 0.0
        best_idx = (q + mask).argmax().item()
    d = (best_idx % 9) + 1
    rc = best_idx // 9
    r, c = divmod(rc, 9)
    return (r, c, d)


def soft_update(target, online, tau=0.005):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


def hard_update(target, online):
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.copy_(p.data)


def make_batches_tensors(backend, agent, batch):
    """Convert list of (s,a,r,s',done) to tensors on device."""
    enc = backend.SudokuDataset
    states = torch.stack([enc._encode_state(s) for (s,_,_,_,_) in batch]).to(agent.device)
    next_states = torch.stack([enc._encode_state(sn) for (_,_,_,sn,_) in batch]).to(agent.device)
    actions = torch.tensor([enc._encode_action(a) for (_,a,_,_,_) in batch], dtype=torch.long, device=agent.device)
    rewards = torch.tensor([float(r) for (_,_,r,_,_) in batch], dtype=torch.float32, device=agent.device)
    dones = torch.tensor([float(d) for (_,_,_,_,d) in batch], dtype=torch.float32, device=agent.device)
    return states, actions, rewards, next_states, dones


def dqfd_loss(backend, agent, batch, margin=0.8, gamma=0.99):
    """TD loss + Large-Margin classification loss."""
    states, actions, rewards, next_states, dones = make_batches_tensors(backend, agent, batch)

    q = agent.q(states)                       
    q_a = q.gather(1, actions.view(-1,1)).squeeze(1)

    with torch.no_grad():
        q_next = agent.q_target(next_states).max(1)[0]
        target = rewards + (1.0 - dones) * gamma * q_next

    td_loss = torch.nn.functional.smooth_l1_loss(q_a, target)

    margin_mat = margin * torch.ones_like(q)
    margin_mat.scatter_(1, actions.view(-1,1), 0.0)
    viol = (q + margin_mat).max(dim=1)[0] - q_a
    margin_loss = torch.relu(viol).mean()

    return td_loss + margin_loss, {'td': td_loss.item(), 'margin': margin_loss.item()}



def evaluate(backend, agent, puzzles: List[np.ndarray], max_steps=81):
    """Runs greedy policy on given puzzles; returns dict."""
    total, solved, total_reward = 0, 0, 0.0
    for p in puzzles:
        env = backend.SudokuEnv(p)
        s = env.reset()
        agent.env = env
        done = False
        steps = 0
        ep_r = 0.0
        while not done and steps < max_steps:
            a = select_action_greedy(backend, agent, s, eps=0.0)
            if a is None:
                break
            ns, r, done, _ = env.step(a)
            ep_r += float(r)
            s = ns
            steps += 1
        total_reward += ep_r
        if (env.grid != 0).all():
            solved += 1
        total += 1
    return {'total': total, 'solved': solved, 'solved_rate': (solved/total if total else 0.0),
            'avg_reward': (total_reward/total if total else 0.0)}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sudoku_rl_patched")

    ap.add_argument("--csv", type=str, default=None, help="CSV with quizzes,solutions to build expert data")
    ap.add_argument("--limit", type=int, default=5000, help="How many rows to read from CSV")
    ap.add_argument("--expert-limit", type=int, default=80_000, help="Max expert transitions to prefill")

    ap.add_argument("--save-expert-npz", type=str, default=None,
                    help="Path to save expert transitions as NPZ")
    ap.add_argument("--load-expert-npz", type=str, default=None,
                    help="Path to load expert transitions NPZ instead of rebuilding")

    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--bc-steps-per-epoch", type=int, default=3000)

    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--train-steps-per-episode", type=int, default=16)
    ap.add_argument("--expert-ratio", type=float, default=0.7)
    ap.add_argument("--dqfd-margin", type=float, default=0.8)
    ap.add_argument("--nstep", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--target-tau", type=float, default=0.01)
    ap.add_argument("--target-update-every", type=int, default=1)

    ap.add_argument("--eps-start", type=float, default=0.2)
    ap.add_argument("--eps-final", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=float, default=0.9997)

    ap.add_argument("--cap-expert", type=int, default=200_000)
    ap.add_argument("--cap-online", type=int, default=200_000)

    ap.add_argument("--test-model", type=str, default=None)
    ap.add_argument("--test-synth", type=int, default=None, help="Evaluate on synthetic puzzles with N holes")
    ap.add_argument("--test-n", type=int, default=300)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--curriculum", action="store_true",
                    help="Enable curriculum: increase holes by stages during fine-tune")
    ap.add_argument("--holes-stages", type=str, default="5,7,9,12,15,18,21,24,27,30",
                    help="Comma-separated hole counts per stage (used if --curriculum)")
    ap.add_argument("--episodes-per-stage", type=int, default=200,
                    help="Episodes per curriculum stage")

    args = ap.parse_args()

    backend = import_backend(args.backend)
    import pandas as pd  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    holes_schedule = [int(x) for x in args.holes_stages.split(",")] if args.curriculum else None
    stage_idx = 0
    ep_in_stage = 0

    dummy_env = backend.SudokuEnv(np.zeros((9,9), dtype=int))
    agent = backend.DQNAgent(dummy_env, device)
    agent.q.train()

    if not hasattr(agent, "q_target") or agent.q_target is None:
        agent.q_target = copy.deepcopy(agent.q).to(agent.device)
        for p in agent.q_target.parameters():
            p.requires_grad = False
    agent.q_target.load_state_dict(agent.q.state_dict())
    agent.q_target.eval()

    agent.q_target.eval()

    hard_update(agent.q_target, agent.q)

    optim = torch.optim.Adam(agent.q.parameters(), lr=args.lr)

    if args.test_model is not None:
        print(f"Loading model from {args.test_model}")
        try:
            sd = torch.load(args.test_model, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(args.test_model, map_location=device)
        agent.q.load_state_dict(sd)
        agent.q.eval()

        if args.test_synth is not None:
            puzzles = backend.make_simple_sudoku(empty_cells=args.test_synth, n_variants=args.test_n)
        else:
            if args.csv:
                df = pd.read_csv(args.csv)
                puzzles = []
                for q in df.sample(args.test_n, random_state=args.seed)["quizzes"]:
                    puzzles.append(np.array([int(ch) for ch in q], dtype=int).reshape(9,9))
            else:
                puzzles = backend.make_simple_sudoku(empty_cells=10, n_variants=args.test_n)
        res = evaluate(backend, agent, puzzles)
        print(f"Tested on {res['total']} | Solved: {res['solved']} ({res['solved_rate']:.3f}) | AvgR: {res['avg_reward']:.2f}")
        return

    replay = DQfDReplay(cap_expert=args.cap_expert, cap_online=args.cap_online, nstep=args.nstep, gamma=args.gamma)
    expert_transitions = None

    if args.load_expert_npz:
        print(f"[Expert] Loading NPZ: {args.load_expert_npz}")
        expert_transitions = load_expert_npz(args.load_expert_npz)
    elif args.csv:
        df = pd.read_csv(args.csv)
        print(f"[Expert] Building expert transitions from CSV (limit={args.limit}) ...")
        expert_transitions = build_expert_transitions_from_csv(backend, df, limit=args.limit, seed=args.seed)
        if args.expert_limit:
            expert_transitions = expert_transitions[:args.expert_limit]
        if args.save_expert_npz:
            print(f"[Expert] Saving NPZ: {args.save_expert_npz}")
            pack_expert_npz(args.save_expert_npz, expert_transitions)

    total_expert = 0
    if expert_transitions:
        replay.push_expert(expert_transitions)
        total_expert = len(expert_transitions)
        print(f"[Expert] Prefilled {total_expert} transitions.")

    if total_expert > 0 and args.bc_epochs > 0:
        print(f"[BC] Starting Behavior Cloning for {args.bc_epochs} epochs ...")
        for ep in range(1, args.bc_epochs + 1):
            losses, td_losses, margin_losses = [], [], []
            steps = 0
            while steps < args.bc_steps_per_epoch:
                batch = replay.sample(args.batch_size, expert_ratio=1.0)
                if not batch:
                    break
                loss, parts = dqfd_loss(backend, agent, batch, margin=args.dqfd_margin, gamma=args.gamma)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.q.parameters(), 1.0)
                optim.step()
                if (steps % args.target_update_every) == 0:
                    soft_update(agent.q_target, agent.q, tau=args.target_tau)
                losses.append(loss.item()); td_losses.append(parts['td']); margin_losses.append(parts['margin'])
                steps += 1
            print(f"[BC] Epoch {ep}/{args.bc_epochs} | steps={steps} | loss={np.mean(losses):.4f} "
                  f"(td={np.mean(td_losses):.4f}, margin={np.mean(margin_losses):.4f})")

    if args.episodes <= 0:
        torch.save(agent.q.state_dict(), "dqn_dqfd.pth")
        print("Saved: dqn_dqfd.pth (after BC only)")
        return

    print("[DQfD] Starting fine-tune ...")
    global_step = 0
    eps = args.eps_start
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        if args.curriculum:
            holes = holes_schedule[stage_idx]
        else:
            holes = random.randint(5, 30)
        puzzles = backend.make_simple_sudoku(empty_cells=holes, n_variants=1)
        env = backend.SudokuEnv(puzzles[0])
        s = env.reset(); agent.env = env
        done = False; steps = 0; ep_r = 0.0

        while not done and steps < 81:
            a = select_action_greedy(backend, agent, s, eps=eps)
            if a is None:
                break
            ns, r, done, _ = env.step(a)
            replay.push_online((s, a, float(r), ns, bool(done)))
            s = ns
            steps += 1; ep_r += float(r)

            for _ in range(args.train_steps_per_episode):
                batch = replay.sample(args.batch_size, expert_ratio=args.expert_ratio)
                if not batch:
                    break
                loss, parts = dqfd_loss(backend, agent, batch, margin=args.dqfd_margin, gamma=args.gamma)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.q.parameters(), 1.0)
                optim.step()
                global_step += 1
                if (global_step % args.target_update_every) == 0:
                    soft_update(agent.q_target, agent.q, tau=args.target_tau)

            if done or steps >= 81:
                break

        eps = max(args.eps_final, eps * args.eps_decay)

        if args.curriculum:
            ep_in_stage += 1
            if ep_in_stage >= args.episodes_per_stage and (stage_idx + 1) < len(holes_schedule):
                stage_idx += 1
                ep_in_stage = 0
                print(f"[Curriculum] -> Next stage: holes={holes_schedule[stage_idx]}")

        if (ep % 20) == 0 or ep == args.episodes:
            print(f"[DQfD] Ep {ep}/{args.episodes} | epR={ep_r:.1f} | eps={eps:.3f} | replay={len(replay)}")

        if (ep % 200) == 0:
            eval_pz = backend.make_simple_sudoku(empty_cells=10, n_variants=100)
            agent.q.eval()
            res = evaluate(backend, agent, eval_pz)
            agent.q.train()
            elapsed = time.time() - start_time
            print(f"[Eval@{ep}] 10-holes | Solved {res['solved']}/{res['total']} ({res['solved_rate']:.3f}) "
                  f"| AvgR {res['avg_reward']:.1f} | {elapsed/60:.1f} min")

    torch.save(agent.q.state_dict(), "dqn_dqfd.pth")
    print("Saved: dqn_dqfd.pth")


if __name__ == "__main__":
    main()