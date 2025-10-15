# symbolic_policy_humanfeat.py
"""
Two-stage symbolic policy for Sudoku with human-understandable features.

Stage A (CELL selector, 81 classes):
  Features (flattened across all cells, index r*9+c):
   - empty_mask[81]
   - cand_count[81]
   - single_mask[81]
  + Global context:
   - row_empty[9], col_empty[9], box_empty[9]

Stage B (DIGIT selector, 9 classes), conditioned on chosen cell (r,c):
  Features:
   - cand[1..9], row_has[1..9], col_has[1..9], box_has[1..9]
   - row_pos[1..9], col_pos[1..9], box_pos[1..9]
   - only_row[1..9], only_col[1..9], only_box[1..9]
   - empties_in_row, empties_in_col, empties_in_box, cell_cand_count

Backends:
  - gplearn SymbolicClassifier (по умолчанию) — pip install gplearn joblib
  - PySR (--use-pysr) при наличии Julia
"""

import argparse, json, sys, random
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import joblib

def try_import(name):
    try:
        return __import__(name, fromlist=['*'])
    except Exception:
        return None

gplearn = try_import("gplearn")
pysr = try_import("pysr")

def import_backend(backend_name: str):
    try:
        mod = __import__(backend_name, fromlist=['*'])
        return mod
    except Exception as e:
        print(f"[ERROR] Could not import backend '{backend_name}': {e}")
        sys.exit(1)


def box_id(r, c): return (r // 3) * 3 + (c // 3)

def compute_candidates(grid: np.ndarray):
    """Return cand[9,9,9] bool: cand[r,c,d-1] = True if d legal at (r,c)."""
    cand = np.zeros((9,9,9), dtype=bool)
    row_has = np.zeros((9,9), dtype=bool)
    col_has = np.zeros((9,9), dtype=bool)
    box_has = np.zeros((9,9), dtype=bool)
    for r in range(9):
        for c in range(9):
            v = grid[r,c]
            if v != 0:
                row_has[r, v-1] = True
                col_has[c, v-1] = True
                box_has[box_id(r,c), v-1] = True
    for r in range(9):
        for c in range(9):
            if grid[r,c] != 0:
                continue
            b = box_id(r,c)
            for d in range(1,10):
                if (not row_has[r, d-1]) and (not col_has[c, d-1]) and (not box_has[b, d-1]):
                    cand[r,c,d-1] = True
    return cand, row_has, col_has, box_has

def engineered_features_cell(grid: np.ndarray):
    """
    Return 261-dim vector:
      81 empty + 81 cand_count + 81 single + 9 row_empty + 9 col_empty + 9 box_empty
    """
    cand, _, _, _ = compute_candidates(grid)
    empty_mask = (grid.flatten() == 0).astype(np.float32)                 
    cand_count = cand.sum(axis=2).astype(np.float32).flatten()            
    single_mask = (cand_count == 1).astype(np.float32)                    

    empties_per_row = (grid == 0).sum(axis=1).astype(np.float32)          
    empties_per_col = (grid == 0).sum(axis=0).astype(np.float32)          
    empties_per_box = np.zeros((9,), dtype=np.float32)                    
    for r in range(9):
        for c in range(9):
            if grid[r,c] == 0:
                empties_per_box[box_id(r,c)] += 1.0

    return np.concatenate([empty_mask, cand_count, single_mask,
                           empties_per_row, empties_per_col, empties_per_box], axis=0)

def engineered_features_digit(grid: np.ndarray, r: int, c: int):
    """
    Return 94-dim vector for chosen cell (r,c):
      9 cand + 9 row_has + 9 col_has + 9 box_has +
      9 row_pos + 9 col_pos + 9 box_pos +
      9 only_row + 9 only_col + 9 only_box +
      4 (empties_in_row, empties_in_col, empties_in_box, cell_cand_count)
    """
    cand, row_has, col_has, box_has = compute_candidates(grid)
    cand_digit = cand[r,c].astype(np.float32)                    
    row_has_d = row_has[r].astype(np.float32)                    
    col_has_d = col_has[c].astype(np.float32)                    
    box_has_d = box_has[box_id(r,c)].astype(np.float32)          

    row_possible = cand[r].sum(axis=0).astype(np.float32)        
    col_possible = cand[:,c,:].sum(axis=0).astype(np.float32)   
    b = box_id(r,c); br, bc = 3*(b//3), 3*(b%3)
    box_possible = cand[br:br+3, bc:bc+3, :].reshape(-1,9).sum(axis=0).astype(np.float32)  

    only_row = (row_possible == cand_digit).astype(np.float32) * (cand_digit > 0)
    only_col = (col_possible == cand_digit).astype(np.float32) * (cand_digit > 0)
    only_box = (box_possible == cand_digit).astype(np.float32) * (cand_digit > 0)

    empties_in_row = float((grid[r]==0).sum())
    empties_in_col = float((grid[:,c]==0).sum())
    empties_in_box = float((grid[br:br+3, bc:bc+3]==0).sum())
    cell_cand_count = float(cand_digit.sum())

    return np.concatenate([
        cand_digit, row_has_d, col_has_d, box_has_d,
        row_possible, col_possible, box_possible,
        only_row, only_col, only_box,
        np.array([empties_in_row, empties_in_col, empties_in_box, cell_cand_count], dtype=np.float32)
    ], axis=0)

def decode_action(i: int):
    d = (i % 9) + 1; i //= 9; r, c = divmod(i, 9); return r, c, d


def collect_with_features(backend, agent, puzzles: List[np.ndarray], max_steps=81, samples=60000, seed=42):
    """
    Return:
      - X_cell (N, 261), y_cell (N,)
      - X_digit (N, 94), y_digit (N,)
    """
    Xc, yc, Xd, yd = [], [], [], []
    Env = backend.SudokuEnv
    for p in puzzles:
        env = Env(p); agent.env = env
        s = env.reset()
        steps, done = 0, False
        while not done and steps < max_steps:
            legal = env.legal_actions()
            if not legal: break
            with backend.torch.no_grad():
                s_t = backend.SudokuDataset._encode_state(s).unsqueeze(0).to(agent.device)
                q = agent.q(s_t).squeeze()
                mask = backend.torch.full_like(q, -float('inf'), device=agent.device)
                legal_idx = [backend.SudokuDataset._encode_action(a) for a in legal]
                mask[legal_idx] = 0
                best = (q + mask).argmax().item()
            r,c,d = decode_action(best)

            Xc.append(engineered_features_cell(s))
            Xd.append(engineered_features_digit(s, r, c))
            yc.append(r*9 + c)
            yd.append(d-1)

            ns, rew, done, _ = env.step((r,c,d)); s = ns; steps += 1
            if len(Xc) >= samples: break
        if len(Xc) >= samples: break

    return (np.asarray(Xc, np.float32), np.asarray(yc, np.int64),
            np.asarray(Xd, np.float32), np.asarray(yd, np.int64))


def dump_rules_ovr(ovr_model) -> str:
    """
    Возвращает формулы по классам из OvR-обёртки (для gplearn SymbolicClassifier).
    Каждый estimator_ — бинарный символический классификатор «класс vs остальные».
    """
    lines = []
    try:
        for cls_idx, est in enumerate(ovr_model.estimators_):
            prog = None
            if hasattr(est, "_program") and est._program is not None:
                prog = est._program
            elif hasattr(est, "_best_programs") and est._best_programs:
                for p in reversed(est._best_programs):
                    if p is not None:
                        prog = p
                        break
            lines.append(f"[Class {ovr_model.classes_[cls_idx]}] {prog}")
    except Exception as e:
        lines.append(f"[WARN] Could not extract OvR programs: {e}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained DQN .pth")
    ap.add_argument("--backend", default="sudoku_rl_patched", help="Backend module name (Env & Agent)")
    ap.add_argument("--synthetic", type=int, default=None, help="Use synthetic puzzles with this number of holes")
    ap.add_argument("--n-variants", type=int, default=2000, help="How many synthetic variants to generate")
    ap.add_argument("--csv", type=str, default=None, help="Path to sudoku.csv (columns: quizzes, solutions)")
    ap.add_argument("--limit", type=int, default=3000, help="How many CSV puzzles to sample")
    ap.add_argument("--samples", type=int, default=60000, help="(state,action) pairs to collect from DQN")
    ap.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for fidelity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-pysr", action="store_true", help="Try PySR (if installed); otherwise use gplearn")
    ap.add_argument("--gp-generations", type=int, default=20)
    ap.add_argument("--gp-popsize", type=int, default=2000)
    ap.add_argument("--gp-tournament", type=int, default=20)
    ap.add_argument("--gp-parsimony", type=float, default=0.003)
    args = ap.parse_args()

    backend = import_backend(args.backend)
    device = backend.torch.device("cuda" if backend.torch.cuda.is_available() else "cpu")
    dummy = backend.SudokuEnv(np.zeros((9,9), dtype=int))
    agent = backend.DQNAgent(dummy, device)

    try:
        sd = backend.torch.load(args.model, map_location=device, weights_only=True) 
    except TypeError:
        sd = backend.torch.load(args.model, map_location=device) 
    agent.q.load_state_dict(sd)
    agent.q.eval(); agent.eps = 0.0

    puzzles = []
    if args.synthetic is not None:
        puzzles = backend.make_simple_sudoku(empty_cells=args.synthetic, n_variants=args.n_variants)
    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if "quizzes" not in df.columns:
            print("[ERROR] CSV must have 'quizzes' column"); sys.exit(2)
        df = df.sample(args.limit, random_state=args.seed) if args.limit else df
        for s in df["quizzes"]:
            puzzles.append(np.array([int(ch) for ch in s], dtype=int).reshape(9,9))
    else:
        print("[ERROR] Provide either --synthetic or --csv"); sys.exit(2)

    print(f"[Collect] Using {len(puzzles)} puzzles; target samples: {args.samples}")
    Xc, yc, Xd, yd = collect_with_features(backend, agent, puzzles, samples=args.samples, seed=args.seed)
    print(f"[Collect] cell={len(Xc)} samples, digit={len(Xd)} samples")

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=args.test_size, random_state=args.seed, stratify=yc)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=args.test_size, random_state=args.seed, stratify=yd)

    rules_cell, rules_digit = "", ""

    if args.use_pysr and pysr is not None:
        from pysr import PySRClassifier
        print("[Train] Using PySRClassifier (equations)...")
        clf_cell = PySRClassifier(
            niterations=200, model_selection="best", maxsize=40, population_size=2000,
            binary_operators=["+", "-", "*"], unary_operators=["abs", "square"],
            loss="logloss", random_state=args.seed, procs=0
        )
        clf_cell.fit(Xc_tr, yc_tr)

        clf_digit = PySRClassifier(
            niterations=200, model_selection="best", maxsize=30, population_size=2000,
            binary_operators=["+", "-", "*"], unary_operators=["abs", "square"],
            loss="logloss", random_state=args.seed, procs=0
        )
        clf_digit.fit(Xd_tr, yd_tr)

        model_cell, model_digit = clf_cell, clf_digit
        acc_cell = float(accuracy_score(yc_te, clf_cell.predict(Xc_te)))
        acc_digit = float(accuracy_score(yd_te, clf_digit.predict(Xd_te)))
        print(f"[Fidelity] Cell={acc_cell:.3f} Digit={acc_digit:.3f}")

        rules_cell, rules_digit = str(clf_cell), str(clf_digit)

    else:
        if gplearn is None:
            print("[ERROR] gplearn is not installed. Install: pip install gplearn joblib")
            sys.exit(2)
        from gplearn.genetic import SymbolicClassifier
        function_set = ('add','sub','mul','div','sqrt','log','abs','neg','max','min')
        print("[Train] Using gplearn SymbolicClassifier with OneVsRest...")

        feat_names_cell = (
            [f"empty[{i}]" for i in range(81)] +
            [f"cand_count[{i}]" for i in range(81)] +
            [f"single[{i}]" for i in range(81)] +
            [f"row_empty[{i}]" for i in range(9)] +
            [f"col_empty[{i}]" for i in range(9)] +
            [f"box_empty[{i}]" for i in range(9)]
        )
        feat_names_digit = (
            [f"cand[{d}]" for d in range(1,10)] +
            [f"row_has[{d}]" for d in range(1,10)] +
            [f"col_has[{d}]" for d in range(1,10)] +
            [f"box_has[{d}]" for d in range(1,10)] +
            [f"row_pos[{d}]" for d in range(1,10)] +
            [f"col_pos[{d}]" for d in range(1,10)] +
            [f"box_pos[{d}]" for d in range(1,10)] +
            [f"only_row[{d}]" for d in range(1,10)] +
            [f"only_col[{d}]" for d in range(1,10)] +
            [f"only_box[{d}]" for d in range(1,10)] +
            ["empties_in_row","empties_in_col","empties_in_box","cell_cand_count"]
        )

        base_cell = SymbolicClassifier(
            function_set=function_set,
            population_size=args.gp_popsize,
            generations=args.gp_generations,
            tournament_size=args.gp_tournament,
            const_range=(-2.0, 2.0),
            init_depth=(2, 4),
            init_method='half and half',
            p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1,
            max_samples=0.9,
            parsimony_coefficient=args.gp_parsimony,
            feature_names=feat_names_cell,
            verbose=1,
            n_jobs=1,
            random_state=args.seed,
        )
        clf_cell = OneVsRestClassifier(base_cell, n_jobs=-1)   

        base_digit = SymbolicClassifier(
            function_set=function_set,
            population_size=args.gp_popsize,
            generations=args.gp_generations,
            tournament_size=args.gp_tournament,
            const_range=(-2.0, 2.0),
            init_depth=(2, 4),
            init_method='half and half',
            p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1,
            max_samples=0.9,
            parsimony_coefficient=args.gp_parsimony,
            feature_names=feat_names_digit,
            verbose=1,
            n_jobs=1,
            random_state=args.seed,
        )
        clf_digit = OneVsRestClassifier(base_digit, n_jobs=-1) 

        print("[Train] Fitting CELL OvR...")
        clf_cell.fit(Xc_tr, yc_tr)
        print("[Train] Fitting DIGIT OvR...")
        clf_digit.fit(Xd_tr, yd_tr)

        model_cell, model_digit = clf_cell, clf_digit

        acc_cell = float(accuracy_score(yc_te, clf_cell.predict(Xc_te)))
        acc_digit = float(accuracy_score(yd_te, clf_digit.predict(Xd_te)))
        print(f"[Fidelity] Cell={acc_cell:.3f} Digit={acc_digit:.3f}")

        rules_cell = dump_rules_ovr(clf_cell)
        rules_digit = dump_rules_ovr(clf_digit)

    fid_cell = float(accuracy_score(yc_te, model_cell.predict(Xc_te)))
    fid_digit = float(accuracy_score(yd_te, model_digit.predict(Xd_te)))
    print(f"[Fidelity Summary] Cell acc={fid_cell:.3f} | Digit acc={fid_digit:.3f}")

    def play_env(backend, model_cell, model_digit, puzzles, max_steps=81):
        solved, total, total_reward = 0, 0, 0.0
        for p in puzzles:
            env = backend.SudokuEnv(p); s = env.reset()
            done=False; steps=0; ep_r=0.0
            while not done and steps < max_steps:
                x_cell = engineered_features_cell(s).reshape(1,-1)
                cell_cls = int(model_cell.predict(x_cell)[0]); r, c = divmod(cell_cls, 9)
                x_digit = engineered_features_digit(s, r, c).reshape(1,-1)

                if hasattr(model_digit, "predict_proba"):
                    try:
                        probs = model_digit.predict_proba(x_digit)[0]
                    except Exception:
                        probs = np.zeros((9,), dtype=np.float32)
                        probs[int(model_digit.predict(x_digit)[0])] = 1.0
                else:
                    probs = np.zeros((9,), dtype=np.float32)
                    probs[int(model_digit.predict(x_digit)[0])] = 1.0

                legal = env.legal_actions()
                if not legal: break
                legal_digits = [d for (rr,cc,d) in legal if rr==r and cc==c]
                if legal_digits:
                    d = max(legal_digits, key=lambda dd: probs[dd-1] if 0<=dd-1<len(probs) else -1.0)
                    a = (r,c,d)
                else:
                    best, bestp = None, -1.0
                    for (rr,cc,d) in legal:
                        p_ = probs[d-1] if 0<=d-1<len(probs) else 0.0
                        if p_>bestp: bestp, best = p_, (rr,cc,d)
                    a = best

                ns, rwd, done, _ = env.step(a); ep_r += rwd; s = ns; steps += 1
            total_reward += ep_r
            if (env.grid != 0).all(): solved += 1
            total += 1
        return {"solved": int(solved), "total": int(total),
                "solved_rate": float(solved/total if total else 0.0),
                "avg_reward": float(total_reward/total if total else 0.0)}

    eval_puzzles = puzzles[: min(1000, len(puzzles))]
    env_metrics = play_env(backend, model_cell, model_digit, eval_puzzles)

    joblib.dump(model_cell, "sym_cell_humanfeat.joblib")
    joblib.dump(model_digit, "sym_digit_humanfeat.joblib")
    with open("sym_rules_cell_humanfeat.txt", "w", encoding="utf-8") as f:
        f.write(rules_cell)
    with open("sym_rules_digit_humanfeat.txt", "w", encoding="utf-8") as f:
        f.write(rules_digit)
    with open("sym_report_humanfeat.json", "w", encoding="utf-8") as f:
        json.dump({"fidelity": {"cell": fid_cell, "digit": fid_digit}, "env": env_metrics}, f, indent=2)

    print("[Saved] sym_cell_humanfeat.joblib, sym_digit_humanfeat.joblib, "
          "sym_rules_cell_humanfeat.txt, sym_rules_digit_humanfeat.txt, sym_report_humanfeat.json")

if __name__ == "__main__":
    main()