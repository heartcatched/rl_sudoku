# fix_tests_add_solutions.py
import argparse
from pathlib import Path
import sys
import pandas as pd


def str81_to_grid(s: str):
    import numpy as np
    a = [int(ch) for ch in s.strip()]
    return np.array(a, dtype=int).reshape(9, 9)

def grid_to_str81(grid) -> str:
    return "".join(str(int(x)) for x in grid.reshape(-1))

def find_empty(grid):
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                return r, c
    return None

def valid(grid, r, c, d):
    br, bc = 3*(r//3), 3*(c//3)
    if d in grid[r, :]: return False
    if d in grid[:, c]: return False
    if d in grid[br:br+3, bc:bc+3]: return False
    return True

def solve_sudoku(grid) -> bool:
    pos = find_empty(grid)
    if pos is None:
        return True
    r, c = pos
    for d in range(1, 10):
        if valid(grid, r, c, d):
            grid[r, c] = d
            if solve_sudoku(grid):
                return True
            grid[r, c] = 0
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Папка с data/sudoku.csv и sudoku_test_h*.csv")
    ap.add_argument("--master", default="sudoku.csv", help="Имя мастер-CSV (по умолчанию data/sudoku.csv в корне)")
    ap.add_argument("--pattern", default="sudoku_test_h*.csv", help="Шаблон тестовых файлов")
    args = ap.parse_args()

    root = Path(args.root)
    master_path = root / args.master
    if not master_path.exists():
        alt = root / "sudoku.csv"
        data_alt = root / "data" / "sudoku.csv"
        if data_alt.exists():
            master_path = data_alt
        elif alt.exists():
            master_path = alt
        else:
            print(f"[ERROR] Не найден мастер CSV: {master_path} (и {data_alt} тоже нет).", file=sys.stderr)
            sys.exit(2)

    print(f"[Info] Master CSV: {master_path}")
    mdf = pd.read_csv(master_path)
    if "quizzes" not in mdf.columns or "solutions" not in mdf.columns:
        print("[ERROR] В мастер CSV должны быть столбцы 'quizzes' и 'solutions'.", file=sys.stderr)
        sys.exit(3)

    master_map = dict(zip(mdf["quizzes"].astype(str), mdf["solutions"].astype(str)))

    test_files = sorted(root.glob(args.pattern))
    if not test_files:
        print(f"[ERROR] Не найдено файлов {args.pattern} в {root}", file=sys.stderr)
        sys.exit(4)

    total_fixed = 0
    for f in test_files:
        df = pd.read_csv(f)
        if "quizzes" not in df.columns:
            print(f"[WARN] {f.name}: нет колонки 'quizzes' — пропускаю.")
            continue
        if "solutions" not in df.columns:
            df["solutions"] = ""

        need_solve_idx = []

        quizzes = df["quizzes"].astype(str)
        sols = df["solutions"].astype(str)

        mapped = quizzes.map(master_map).fillna("")
        mask_bad = ~sols.str.len().eq(81)
        sols = sols.where(~mask_bad, mapped)

        mask_still = ~sols.str.len().eq(81)
        indices = df.index[mask_still].tolist()

        print(f"[Proc] {f.name}: rows={len(df)}, from_master={mask_bad.sum() - mask_still.sum()}, to_solve={len(indices)}")

        if indices:
            import numpy as np
            for idx in indices:
                q = str(df.at[idx, "quizzes"])
                try:
                    grid = str81_to_grid(q)
                except Exception:
                    continue
                grid = grid.copy()
                if solve_sudoku(grid):
                    sols_i = grid_to_str81(grid)
                    if len(sols_i) == 81 and sols_i.isdigit():
                        sols.iloc[idx] = sols_i
                        total_fixed += 1

        df["solutions"] = sols
        bad_after = (~df["solutions"].astype(str).str.len().eq(81)).sum()
        if bad_after > 0:
            print(f"[WARN] {f.name}: {bad_after} строк всё ещё без валидного solutions (81). Они останутся как есть.")

        df.to_csv(f, index=False, encoding="utf-8")
        print(f"[OK] {f.name} — сохранён.")

    print(f"[DONE] Дополнено решений: {total_fixed}")

if __name__ == "__main__":
    main()