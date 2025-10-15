"""
Sudoku RL Framework — v5.1 Corrected & Stable

Main fixes:
- Resolved the DataLoader deadlock by completely separating experience gathering and training logic.
- The agent now collects experience in one method and trains from the buffer in another,
  preventing hangs and ensuring stable training.
- The main training loop is updated to orchestrate this new, robust workflow.

Usage:
  python sudoku_rl.py --dataset sudoku.csv --episodes 50000 --batch-size 512
"""
from __future__ import annotations
import math, random, argparse, os
from collections import deque
from typing import List, Tuple
import numpy as np
import time

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None



def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_kaggle_sudoku(path='sudoku.csv', n_samples=None):
    df = pd.read_csv(path)
    df['num_zeros'] = df['quizzes'].apply(lambda s: s.count('0'))
    df = df.sort_values('num_zeros')
    if n_samples:
        df = df.head(n_samples)
    quizzes = [np.array([int(ch) for ch in s]).reshape(9, 9) for s in df['quizzes']]
    solutions = [np.array([int(ch) for ch in s]).reshape(9, 9) for s in df['solutions']]
    return quizzes, solutions

def make_curriculum_from_solutions(solutions, n_variants=10, max_empty=10):
    curriculum = [[] for _ in range(max_empty)]
    for sol in solutions:
        for n_empty in range(1, max_empty+1):
            for _ in range(n_variants):
                puzzle = sol.copy()
                idxs = np.arange(81)
                np.random.shuffle(idxs)
                puzzle.flat[idxs[:n_empty]] = 0
                curriculum[n_empty-1].append(puzzle.copy())
    return curriculum

class SudokuEnv:
    def __init__(self, puzzle: np.ndarray, block: int = 3):
        self.init_grid = puzzle.astype(int)
        self.n = 9
        self.block = block
        self.rows = [set() for _ in range(self.n)]
        self.cols = [set() for _ in range(self.n)]
        self.blocks = [set() for _ in range(self.n)]
        self.reset()

    def reset(self):
        self.grid = self.init_grid.copy()
        for s in self.rows: s.clear()
        for s in self.cols: s.clear()
        for s in self.blocks: s.clear()
        for r in range(self.n):
            for c in range(self.n):
                digit = self.grid[r, c]
                if digit != 0: self._add_to_sets(r, c, digit)
        return self.grid.copy()

    def _get_block_idx(self, r, c): return (r // self.block) * self.block + (c // self.block)
    def _add_to_sets(self, r, c, d): self.rows[r].add(d); self.cols[c].add(d); self.blocks[self._get_block_idx(r, c)].add(d)
    def is_valid(self, r, c, d): return d not in self.rows[r] and d not in self.cols[c] and d not in self.blocks[self._get_block_idx(r, c)]

    def step(self, a: Tuple[int, int, int]):
        r, c, d = a; done = False
        if self.grid[r, c] != 0:
            return self.grid.copy(), -1, done, {}
        if not self.is_valid(r, c, d):
            return self.grid.copy(), -2, done, {}
        self.grid[r, c] = d; self._add_to_sets(r, c, d); reward = 10
        if (self.grid != 0).all():
            reward += 100; done = True
        return self.grid.copy(), reward, done, {}

    def legal_actions(self):
        acts = []
        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r, c] == 0:
                    for d in range(1, self.n + 1):
                        if self.is_valid(r, c, d): acts.append((r, c, d))
        return acts

class ReplayBuffer(deque):
    def __init__(self, cap=100000):
        super().__init__(maxlen=cap)

class SudokuDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer): self.buffer = buffer
    @staticmethod
    def _encode_state(s: np.ndarray):
        one_hot = np.zeros((10, 9, 9), dtype=np.float32);
        for i in range(10): one_hot[i, :, :] = (s == i)
        return torch.from_numpy(one_hot)
    @staticmethod
    def _encode_action(a: Tuple[int, int, int]): r, c, d = a; return (r * 9 + c) * 9 + (d - 1)
    def __getitem__(self, i):
        s, a, r, sn, d = self.buffer[i]
        return self._encode_state(s), self._encode_action(a), r, self._encode_state(sn), d
    def __len__(self): return len(self.buffer)

if torch:
    class QNetCNN(nn.Module):
        def __init__(self, n=9, action_size=729):
            super().__init__()
            self.conv_stack = nn.Sequential(nn.Conv2d(n + 1, 64, kernel_size=3, padding=1), nn.ReLU(),
                                          nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
            self.fc_stack = nn.Sequential(nn.Flatten(), nn.Linear(128 * n * n, 512), nn.ReLU(), nn.Linear(512, action_size))
        def forward(self, x): return self.fc_stack(self.conv_stack(x))

    class DQNAgent:
        def __init__(self, env, device, **kwargs):
            self.env, self.device = env, device
            self.gamma = kwargs.get('gamma', 0.99)
            self.batch_size = kwargs.get('batch_size', 32)
            self.eps_decay = kwargs.get('eps_decay', 0.9999)
            self.min_buffer_size = kwargs.get('min_buffer_size', 128)
            self.num_workers = kwargs.get('num_workers', 0)
            self.train_steps_per_episode = kwargs.get('train_steps_per_episode', 4)

            self.q = QNetCNN().to(self.device); self.t = QNetCNN().to(self.device)
            self.t.load_state_dict(self.q.state_dict()); self.t.eval()
            self.opt = optim.Adam(self.q.parameters(), lr=kwargs.get('lr', 2e-4))
            
            self.replay_cap = 200000
            self.buf = ReplayBuffer(cap=self.replay_cap)
            self.eps = 1.0; self.eps_end = 0.05

        def is_ready(self):
            return len(self.buf) >= self.min_buffer_size

        def add_to_buffer(self, s, a, r, sn, d):
            self.buf.append((s, a, r, sn, d))
            if self.eps > self.eps_end: self.eps *= self.eps_decay

        @staticmethod
        def _decode_action(i: int): d = (i % 9) + 1; i //= 9; r, c = divmod(i, 9); return r, c, d
        
        def act(self, s: np.ndarray, env: SudokuEnv = None):
            env = env or self.env
            legal_actions = env.legal_actions()
            if random.random() < self.eps: return random.choice(legal_actions)
            with torch.no_grad():
                state_t = SudokuDataset._encode_state(s).unsqueeze(0).to(self.device)
                q_values = self.q(state_t).squeeze()
                mask = torch.full_like(q_values, -float('inf'), device=self.device)
                legal_indices = [SudokuDataset._encode_action(a) for a in legal_actions]
                mask[legal_indices] = 0
                return self._decode_action((q_values + mask).argmax().item())

        def train_step(self):
            dataset = SudokuDataset(self.buf)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=(self.device.type == 'cuda'))
            try:
                s_t, a_t, r_t, sn_t, d_t = next(iter(loader))
            except StopIteration:
                return
            s_t, sn_t = s_t.to(self.device), sn_t.to(self.device)
            a_t = a_t.to(self.device)
            r_t = r_t.to(self.device, dtype=torch.float32)
            d_t = d_t.to(self.device, dtype=torch.float32)
            q_vals = self.q(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q_online = self.q(sn_t)
                next_actions = next_q_online.argmax(1)
                next_q_target = self.t(sn_t).gather(1, next_actions.unsqueeze(1)).squeeze()
                tgt = r_t + self.gamma * next_q_target * (1 - d_t)
            loss = nn.functional.smooth_l1_loss(q_vals, tgt)
            self.opt.zero_grad(); loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
            except Exception:
                pass
            self.opt.step()

        def update_target_net(self):
            self.t.load_state_dict(self.q.state_dict())

def load_dataset(path: str, limit: int|None=None):
    if not pd: raise ImportError("Pandas must be installed.")
    df = pd.read_csv(path)
    if limit: df = df.sample(limit, random_state=42)
    def str_to_grid(s): return np.array([int(ch) for ch in s]).reshape(9, 9)
    return [str_to_grid(q) for q in df['quizzes']]

def run_experience_gathering_episode(env: SudokuEnv, agent: DQNAgent):
    state = env.reset(); total_reward = 0.0; max_steps = 81
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        agent.add_to_buffer(state, action, reward, next_state, done)
        state = next_state
        if done: return True, total_reward
    return False, total_reward

def make_simple_sudoku(empty_cells=1, n_variants=10):
    full_grid = np.array([
        [5,3,4,6,7,8,9,1,2],
        [6,7,2,1,9,5,3,4,8],
        [1,9,8,3,4,2,5,6,7],
        [8,5,9,7,6,1,4,2,3],
        [4,2,6,8,5,3,7,9,1],
        [7,1,3,9,2,4,8,5,6],
        [9,6,1,5,3,7,2,8,4],
        [2,8,7,4,1,9,6,3,5],
        [3,4,5,2,8,6,1,7,9],
    ])
    puzzles = []
    for _ in range(n_variants):
        grid = full_grid.copy()
        idxs = np.arange(81)
        np.random.shuffle(idxs)
        grid.flat[idxs[:empty_cells]] = 0
        puzzles.append(grid.copy())
    return puzzles

def test_model_on_sudoku(agent, puzzles, max_steps=81, verbose=False):
    solved_count = 0
    total_reward = 0
    for i, puzzle in enumerate(puzzles):
        env = SudokuEnv(puzzle)
        agent.env = env              
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        while not done and steps < max_steps:
            action = agent.act(state)  
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
        total_reward += episode_reward
        if (env.grid != 0).all():
            solved_count += 1
        if verbose:
            print(f"Test #{i+1}: {'Solved' if (env.grid != 0).all() else 'Failed'}, Reward: {episode_reward}")
    avg_reward = total_reward / len(puzzles)
    solved_rate = solved_count / len(puzzles)
    print(f"Tested on {len(puzzles)} puzzles | Solved: {solved_count} ({solved_rate:.3f}) | Avg Reward: {avg_reward:.2f}")

def split_sudoku_by_holes(input_csv='sudoku.csv', output_dir='sudoku_by_holes'):
    """
    Сохраняет для каждого количества дырок (нулей) отдельный файл с задачами.
    Например: sudoku_holes_40.csv — все задачи с 40 дырками.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    df['num_zeros'] = df['quizzes'].apply(lambda s: s.count('0'))
    for num_zeros, group in df.groupby('num_zeros'):
        out_path = os.path.join(output_dir, f'sudoku_holes_{num_zeros}.csv')
        group[['quizzes', 'solutions']].to_csv(out_path, index=False)
        print(f"Saved {len(group)} puzzles with {num_zeros} holes to {out_path}")

def main():
    
    ap = argparse.ArgumentParser(description="Sudoku RL Framework (v5.1 Stable)")
    ap.add_argument('--test-model', type=str, default=None, help="Путь к сохранённой модели для теста")
    ap.add_argument('--test-data', type=str, default=None, help="Путь к csv с тестовыми задачами (quizzes)")
    ap.add_argument('--test-limit', type=int, default=100, help="Сколько тестовых задач использовать")
    ap.add_argument('--dataset-curriculum', type=str, default=None, help="Путь к датасету для curriculum learning")
    ap.add_argument('--n-samples', type=int, default=1000, help="Сколько задач брать из датасета")
    ap.add_argument('--dataset', required=False)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=50000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--eps-decay', type=float, default=0.9999)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--train-steps-per-episode', type=int, default=4)
    ap.add_argument('--curriculum', action='store_true', help="Enable curriculum learning (auto-increase difficulty)")
    
    args = ap.parse_args()

    try:
        set_seed(42)
    except Exception:
        pass

    if not torch:
        raise ImportError("PyTorch must be installed.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.test_model and args.test_data:
        print(f"Загружаем модель из {args.test_model}")
        dummy_puzzle = np.zeros((9, 9), dtype=int)
        env = SudokuEnv(dummy_puzzle)
        agent = DQNAgent(env, device, **vars(args))
        agent.q.load_state_dict(torch.load(args.test_model, map_location=device))
        agent.q.eval()
        agent.eps = 0.0  
        df = pd.read_csv(args.test_data)
        test_puzzles = [np.array([int(ch) for ch in s]).reshape(9, 9) for s in df['quizzes'][:args.test_limit]]
        print(f"Тестируем на {len(test_puzzles)} задачах...")
        test_model_on_sudoku(agent, test_puzzles, verbose=True)
        return

    if args.dataset_curriculum:
        print(f"Загружаем датасет из {args.dataset_curriculum} ...")
        quizzes, solutions = load_kaggle_sudoku(args.dataset_curriculum, n_samples=args.n_samples)
        curriculum = make_curriculum_from_solutions(solutions, n_variants=10, max_empty=50)
        max_empty = 50
        solved_threshold = 0.95

        episodes_per_stage_list = [50000]*50

        for empty_cells in range(1, max_empty+1):
            episodes_per_stage = episodes_per_stage_list[empty_cells-1]

        max_empty = 50



        min_buffer_sizes = []
        for i in range(50):
            if i < 15:
                buffer_size = 512 * (2 ** (i // 3))
            elif i < 35:
                buffer_size = 4096 * (2 ** ((i - 15) // 5))
            else:
                buffer_size = 16384 * (2 ** ((i - 35) // 3))
            
            buffer_size = min(buffer_size, 131072)
            min_buffer_sizes.append(buffer_size)

        min_buffer_sizes = [min(200000, x) for x in min_buffer_sizes]

        batch_sizes = [max(32, min(1024, buffer_size // 16)) for buffer_size in min_buffer_sizes]
        total_start = time.time()
        last_agent = None
        for empty_cells in range(1, max_empty+1):
            print(f"\n=== CURRICULUM DATASET: {empty_cells} пустых клеток ===")
            stage_start = time.time()
            all_puzzles = curriculum[empty_cells-1]
            env = SudokuEnv(all_puzzles[0])
            agent = DQNAgent(
                env, device,
                batch_size=batch_sizes[empty_cells-1],
                min_buffer_size=min_buffer_sizes[empty_cells-1],
                num_workers=args.num_workers,
                lr=args.lr,
                eps_decay=args.eps_decay,
                train_steps_per_episode=args.train_steps_per_episode
            )
            last_agent = agent
            print(f"--- Warming up replay buffer to {agent.min_buffer_size} experiences ---")
            while not agent.is_ready():
                puzzle = random.choice(all_puzzles)
                env.init_grid = puzzle
                run_experience_gathering_episode(env, agent)
            print("--- Buffer ready. Starting main training loop. ---")
            solved_count = 0; rewards_q = deque(maxlen=100)
            for ep in range(episodes_per_stage):
                puzzle = random.choice(all_puzzles)
                env.init_grid = puzzle
                solved, total_reward = run_experience_gathering_episode(env, agent)
                for _ in range(agent.train_steps_per_episode):
                    agent.train_step()
                if (ep + 1) % 5 == 0:
                    agent.update_target_net()
                solved_count += solved; rewards_q.append(total_reward)
                if (ep + 1) % 10 == 0:
                    avg_r = sum(rewards_q) / len(rewards_q)
                    print(f'Ep {ep+1}/{episodes_per_stage} | Solved Rate (total): {solved_count/(ep+1):.3f} | '
                          f'Avg Reward (last 10): {avg_r:.2f} | Epsilon: {agent.eps:.3f}')
            solved_rate = solved_count / episodes_per_stage
            stage_time = time.time() - stage_start
            print(f"Время на уровень с {empty_cells} пустыми клетками: {stage_time:.1f} сек.")
            torch.save(agent.q.state_dict(), f'sudoku_dqn_stage_{empty_cells}empty.pth')
            print(f"Модель сохранена в sudoku_dqn_stage_{empty_cells}empty.pth")
        total_time = time.time() - total_start
        print(f"Curriculum learning на датасете завершён. Общее время: {total_time:.1f} сек.")
        if last_agent is not None:
            torch.save(last_agent.q.state_dict(), f'sudoku_dqn_final_dataset_{empty_cells}empty.pth')
            print(f"Модель сохранена в sudoku_dqn_final_dataset_{empty_cells}empty.pth")
        return 

    if args.curriculum:
        max_empty = 10
        solved_threshold = 0.95
        episodes_per_stage = 10000
        min_buffer_sizes = [min(16384, 128 * (2 ** (i // 5))) for i in range(max_empty)]
        batch_sizes = [min(512, 32 * (2 ** (i // 5))) for i in range(max_empty)]
        total_start = time.time()
        last_agent = None
        for empty_cells in range(1, max_empty+1):
            print(f"\n=== CURRICULUM: {empty_cells} пустых клеток ===")
            stage_start = time.time()
            all_puzzles = make_simple_sudoku(empty_cells, n_variants=100)
            env = SudokuEnv(all_puzzles[0])
            agent = DQNAgent(
                env, device,
                batch_size=batch_sizes[empty_cells-1],
                min_buffer_size=min_buffer_sizes[empty_cells-1],
                num_workers=args.num_workers,
                lr=args.lr,
                eps_decay=args.eps_decay,
                train_steps_per_episode=args.train_steps_per_episode
            )
            last_agent = agent
            print(f"--- Warming up replay buffer to {agent.min_buffer_size} experiences ---")
            while not agent.is_ready():
                puzzle = random.choice(all_puzzles)
                env.init_grid = puzzle
                run_experience_gathering_episode(env, agent)
            print("--- Buffer ready. Starting main training loop. ---")
            solved_count = 0; rewards_q = deque(maxlen=100)
            for ep in range(episodes_per_stage):
                puzzle = random.choice(all_puzzles)
                env.init_grid = puzzle
                solved, total_reward = run_experience_gathering_episode(env, agent)
                for _ in range(agent.train_steps_per_episode):
                    agent.train_step()
                if (ep + 1) % 5 == 0:
                    agent.update_target_net()
                solved_count += solved; rewards_q.append(total_reward)
                if (ep + 1) % 10 == 0:
                    avg_r = sum(rewards_q) / len(rewards_q)
                    print(f'Ep {ep+1}/{episodes_per_stage} | Solved Rate (total): {solved_count/(ep+1):.3f} | '
                          f'Avg Reward (last 10): {avg_r:.2f} | Epsilon: {agent.eps:.3f}')
            solved_rate = solved_count / episodes_per_stage
            stage_time = time.time() - stage_start
            print(f"Время на уровень с {empty_cells} пустыми клетками: {stage_time:.1f} сек.")
            if solved_rate < solved_threshold:
                print(f"Не удалось достичь порога {solved_threshold*100:.0f}% на {empty_cells} пустых клетках. Останов.")
                break
        total_time = time.time() - total_start
        print(f"Curriculum learning завершён. Общее время: {total_time:.1f} сек.")
        if last_agent is not None:
            torch.save(last_agent.q.state_dict(), f'sudoku_dqn_final_{empty_cells}empty.pth')
            print(f"Модель сохранена в sudoku_dqn_final_{empty_cells}empty.pth")
        return

    if args.dataset:
        start_time = time.time()
        all_puzzles = load_dataset(args.dataset, limit=args.limit)
        env = SudokuEnv(all_puzzles[0])
        agent = DQNAgent(env, device, **vars(args))
        print(f"--- Warming up replay buffer to {agent.min_buffer_size} experiences ---")
        while not agent.is_ready():
            puzzle = random.choice(all_puzzles)
            env.init_grid = puzzle
            run_experience_gathering_episode(env, agent)
        print("--- Buffer ready. Starting main training loop. ---")
        solved_count = 0; rewards_q = deque(maxlen=100)
        for ep in range(args.episodes):
            puzzle = random.choice(all_puzzles)
            env.init_grid = puzzle
            solved, total_reward = run_experience_gathering_episode(env, agent)
            for _ in range(agent.train_steps_per_episode):
                agent.train_step()
            if (ep + 1) % 5 == 0:
                agent.update_target_net()
            solved_count += solved; rewards_q.append(total_reward)
            if (ep + 1) % 10 == 0:
                avg_r = sum(rewards_q) / len(rewards_q)
                print(f'Ep {ep+1}/{args.episodes} | Solved Rate (total): {solved_count/(ep+1):.3f} | '
                      f'Avg Reward (last 10): {avg_r:.2f} | Epsilon: {agent.eps:.3f}')
        total_time = time.time() - start_time
        print(f"Обучение завершено. Общее время: {total_time:.1f} сек.")
        torch.save(agent.q.state_dict(), 'sudoku_dqn_final.pth')
        print("Модель сохранена в sudoku_dqn_final.pth")
        return

    start_time = time.time()
    all_puzzles = make_simple_sudoku(1)
    env = SudokuEnv(all_puzzles[0])
    agent = DQNAgent(env, device, **vars(args))
    print(f"--- Warming up replay buffer to {agent.min_buffer_size} experiences ---")
    while not agent.is_ready():
        puzzle = random.choice(all_puzzles)
        env.init_grid = puzzle
        run_experience_gathering_episode(env, agent)
    print("--- Buffer ready. Starting main training loop. ---")
    solved_count = 0; rewards_q = deque(maxlen=100)
    for ep in range(args.episodes):
        puzzle = random.choice(all_puzzles)
        env.init_grid = puzzle
        solved, total_reward = run_experience_gathering_episode(env, agent)
        for _ in range(agent.train_steps_per_episode):
            agent.train_step()
        if (ep + 1) % 5 == 0:
            agent.update_target_net()
        solved_count += solved; rewards_q.append(total_reward)
        if (ep + 1) % 10 == 0:
            avg_r = sum(rewards_q) / len(rewards_q)
            print(f'Ep {ep+1}/{args.episodes} | Solved Rate (total): {solved_count/(ep+1):.3f} | '
                  f'Avg Reward (last 10): {avg_r:.2f} | Epsilon: {agent.eps:.3f}')
    total_time = time.time() - start_time
    print(f"Обучение завершено. Общее время: {total_time:.1f} сек.")
    torch.save(agent.q.state_dict(), 'sudoku_dqn_final.pth')
    print("Модель сохранена в sudoku_dqn_final.pth")

if __name__ == '__main__':
    main()