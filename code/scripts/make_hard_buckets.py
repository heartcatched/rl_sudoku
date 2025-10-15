# make_hard_buckets.py  (версия с прогресс-логами)
import argparse
from pathlib import Path
import sys, time
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

def str81_to_grid(s: str) -> np.ndarray:
    a = [int(ch) for ch in s.strip()]
    if len(a) != 81:
        raise ValueError("string length != 81")
    return np.array(a, dtype=int).reshape(9, 9)

def grid_to_str81(grid: np.ndarray) -> str:
    return "".join(str(int(x)) for x in grid.reshape(-1))

def singles_stats(grid: np.ndarray) -> Tuple[int,int,float]:
    empty = 0; singles = 0
    for r in range(9):
        for c in range(9):
            if grid[r,c] != 0: continue
            empty += 1
            row=set(grid[r,:])-{0}; col=set(grid[:,c])-{0}
            br,bc=3*(r//3),3*(c//3)
            box=set(grid[br:br+3,bc:bc+3].ravel())-{0}
            used=row|col|box
            k=9-len(used)
            singles += int(k==1)
    frac = (singles/empty) if empty>0 else 1.0
    return empty, singles, frac

def check_solution_consistent(quiz: np.ndarray, sol: np.ndarray) -> bool:
    if not ((sol>=1)&(sol<=9)).all(): return False
    mask = (quiz!=0)
    return (sol[mask]==quiz[mask]).all()

def _candidates(grid: np.ndarray, r: int, c: int) -> List[int]:
    row=set(grid[r,:])-{0}; col=set(grid[:,c])-{0}
    br,bc=3*(r//3),3*(c//3)
    box=set(grid[br:br+3,bc:bc+3].ravel())-{0}
    used=row|col|box
    return [d for d in range(1,10) if d not in used]

def _find_empty_cell(grid: np.ndarray) -> Optional[Tuple[int,int]]:
    best=None; best_k=10
    for r in range(9):
        for c in range(9):
            if grid[r,c]==0:
                k=len(_candidates(grid,r,c))
                if k<best_k:
                    best=(r,c); best_k=k
                    if best_k<=1: return best
    return best

def solve_count_limited(quiz: np.ndarray, limit_solutions: int = 2) -> Tuple[int, Optional[np.ndarray]]:
    grid = quiz.copy(); count=0; first_sol=None
    def backtrack():
        nonlocal count, first_sol
        if count>=limit_solutions: return
        pos=_find_empty_cell(grid)
        if pos is None:
            count+=1
            if first_sol is None: first_sol=grid.copy()
            return
        r,c=pos
        cand=_candidates(grid,r,c); rng.shuffle(cand)
        for d in cand:
            grid[r,c]=d
            backtrack()
            if count>=limit_solutions: return
            grid[r,c]=0
    backtrack()
    return count, first_sol

def has_unique_solution(quiz: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    n, sol = solve_count_limited(quiz, limit_solutions=2)
    return (n==1), sol

rng = np.random.default_rng(42)

def generate_full_solution() -> np.ndarray:
    grid=np.zeros((9,9),dtype=int)
    def backtrack():
        pos=_find_empty_cell(grid)
        if pos is None: return True
        r,c=pos
        cand=_candidates(grid,r,c); rng.shuffle(cand)
        for d in cand:
            grid[r,c]=d
            if backtrack(): return True
            grid[r,c]=0
        return False
    ok=backtrack()
    if not ok: return generate_full_solution()
    return grid

def knock_out_cells_preserving_uniqueness(full: np.ndarray, holes: int,
                                          max_singles_frac: float,
                                          ensure_unique: bool,
                                          max_attempts: int = 200) -> Optional[np.ndarray]:
    H=holes
    for _ in range(max_attempts):
        quiz=full.copy()
        cells=[(r,c) for r in range(9) for c in range(9)]
        rng.shuffle(cells)
        removed=0
        for (r,c) in cells:
            if removed>=H: break
            tmp=quiz[r,c]
            if tmp==0: continue
            quiz[r,c]=0
            if ensure_unique:
                nsol,_=solve_count_limited(quiz, limit_solutions=2)
                if nsol!=1:
                    quiz[r,c]=tmp
                    continue
            removed+=1
        if removed<H: continue
        _,_,frac=singles_stats(quiz)
        if frac<=max_singles_frac:
            return quiz
    return None

def run_mode_csv(args):
    in_path=Path(args.input); out_dir=Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(in_path)
    print(f"[Info] CSV mode | rows={len(df)}")
    for holes in range(args.min_holes, args.max_holes+1):
        t0=time.time()
        sub=df.dropna(subset=["quizzes","solutions"]).copy()
        sub["q"]=sub["quizzes"].astype(str); sub["s"]=sub["solutions"].astype(str)
        sub=sub[(sub["q"].str.len()==81)&(sub["s"].str.len()==81)]
        sub=sub[sub["q"].str.count("0")==holes].sample(frac=1.0, random_state=args.seed)
        picked=[]; max_frac=args.max_singles_frac
        def try_pick(th):
            nonlocal picked
            for _,row in sub.iterrows():
                if len(picked)>=args.per_bucket: break
                try:
                    q=str81_to_grid(row["q"]); s=str81_to_grid(row["s"])
                except: continue
                if not check_solution_consistent(q,s): continue
                empty,_,frac=singles_stats(q)
                if empty!=holes or frac>th: continue
                if args.ensure_unique:
                    uniq, solf = has_unique_solution(q)
                    if not uniq: continue
                    if not (solf==s).all(): continue
                picked.append({"quizzes":row["q"],"solutions":row["s"]})
        try_pick(max_frac)
        step=0
        while len(picked)<args.per_bucket and step<args.relax_steps:
            step+=1; max_frac=min(1.0, max_frac+args.relax_delta)
            try_pick(max_frac)
        out=out_dir/f"sudoku_hard_h{holes}.csv"
        pd.DataFrame(picked).to_csv(out, index=False)
        avg_frac=np.nan
        if picked:
            fr=[singles_stats(str81_to_grid(r["quizzes"]))[2] for r in picked]
            avg_frac=float(np.mean(fr))
        print(f"[OK][csv] h={holes:>2} | picked={len(picked):>3}/{args.per_bucket} "
              f"| thr≈{max_frac:.2f} | avg_singles_frac={avg_frac if picked else 'n/a'} "
              f"| {time.time()-t0:.1f}s -> {out}")
    print("[DONE] CSV mode")

def run_mode_generate(args):
    out_dir=Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] GENERATE mode | outdir={out_dir}")
    for holes in range(args.min_holes, args.max_holes+1):
        t0=time.time()
        picked=[]; max_frac=args.max_singles_frac
        attempts=0; last_log=0
        def try_fill(th):
            nonlocal picked, attempts, last_log
            while len(picked)<args.per_bucket:
                attempts+=1
                if attempts-last_log>=args.status_every:
                    print(f"[gen] h={holes:>2} | picked={len(picked):>3}/{args.per_bucket} "
                          f"| thr={th:.2f} | attempts={attempts}", flush=True)
                    last_log=attempts
                full=generate_full_solution()
                quiz=knock_out_cells_preserving_uniqueness(
                    full, holes, max_singles_frac=th,
                    ensure_unique=args.ensure_unique,
                    max_attempts=args.max_attempts_per_quiz
                )
                if quiz is None: continue
                if args.ensure_unique:
                    uniq, sol = has_unique_solution(quiz)
                    if not uniq: continue
                else:
                    n, sol = solve_count_limited(quiz, limit_solutions=1)
                    if n<1: continue
                if sol is None: continue
                if not check_solution_consistent(quiz, sol): continue
                picked.append({"quizzes":grid_to_str81(quiz), "solutions":grid_to_str81(sol)})
        try_fill(max_frac)
        step=0
        while len(picked)<args.per_bucket and step<args.relax_steps:
            step+=1; max_frac=min(1.0, max_frac+args.relax_delta)
            print(f"[relax] h={holes} -> thr={max_frac:.2f}", flush=True)
            try_fill(max_frac)
        out=out_dir/f"sudoku_hard_h{holes}.csv"
        pd.DataFrame(picked).to_csv(out, index=False)
        avg_frac=np.nan
        if picked:
            fr=[singles_stats(str81_to_grid(r["quizzes"]))[2] for r in picked]
            avg_frac=float(np.mean(fr))
        print(f"[OK][gen] h={holes:>2} | picked={len(picked):>3}/{args.per_bucket} "
              f"| final_thr≈{max_frac:.2f} | avg_singles_frac={avg_frac if picked else 'n/a'} "
              f"| attempts={attempts} | {time.time()-t0:.1f}s -> {out}", flush=True)
    print("[DONE] GENERATE mode")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["csv","generate"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-holes", type=int, default=20)
    ap.add_argument("--max-holes", type=int, default=30)
    ap.add_argument("--per-bucket", type=int, default=300)
    ap.add_argument("--max-singles-frac", type=float, default=0.20)
    ap.add_argument("--ensure-unique", action="store_true")
    ap.add_argument("--relax-steps", type=int, default=0)
    ap.add_argument("--relax-delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", help="CSV для --mode csv")
    ap.add_argument("--gen-tries-per-puzzle", type=int, default=200)
    ap.add_argument("--max-attempts-per-quiz", type=int, default=200,
                    help="макс. попыток выбивания цифр внутри одной заготовки")
    ap.add_argument("--status-every", type=int, default=50,
                    help="лог каждые N попыток в generate-режиме")
    args=ap.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    if args.mode=="csv":
        if not args.input:
            print("[ERROR] --input обязателен для --mode csv", file=sys.stderr); sys.exit(1)
        run_mode_csv(args)
    else:
        run_mode_generate(args)

if __name__=="__main__":
    main()
