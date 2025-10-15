# make_hole_tests_and_fill.py
r"""
Создаёт по N задач на каждую корзину по числу дырок.
1) Берёт задачи из CSV (quizzes[, solutions]) по корзине.
2) Если не хватает — дозаполняет генератором backend.make_simple_sudoku(...).
3) Если в CSV нет задач для корзины, создаёт всё из генератора.
4) Если после фильтра вообще пусто — всё берём из генератора по всему диапазону дырок.

Выход (рядом с входным CSV):
  - sudoku_test_h{H}.csv     — ровно N задач с H дырками
  - sudoku_test_all.csv      — объединение всех корзин
  - sudoku_test_summary.csv  — сводка

Примеры:
  python make_hole_tests_and_fill.py --input data/sudoku.csv --per-bucket 300 --min-holes 1 --max-holes 30
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def count_holes(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return s.count("0")


def grid_to_quiz_str(grid: np.ndarray) -> str:
    return "".join(str(int(x)) for x in grid.reshape(-1))


def maybe_import_backend(name: str):
    if not name:
        return None
    try:
        mod = __import__(name, fromlist=['*'])
        return mod
    except Exception as e:
        print(f"[WARN] Не удалось импортировать backend '{name}': {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Путь к data/sudoku.csv")
    ap.add_argument("--per-bucket", type=int, default=300, help="Сколько задач на каждую корзину дырок")
    ap.add_argument("--seed", type=int, default=42, help="random_state для детерминированного сэмплинга")
    ap.add_argument("--min-holes", type=int, default=None, help="Минимум дырок (вкл.)")
    ap.add_argument("--max-holes", type=int, default=None, help="Максимум дырок (вкл.)")
    ap.add_argument("--backend", type=str, default="sudoku_rl_patched",
                    help="Модуль бэкенда (make_simple_sudoku)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    src = Path(args.input)
    if not src.exists():
        print(f"[ERROR] Не найден файл: {src}", file=sys.stderr)
        sys.exit(2)

    out_dir = src.parent
    print(f"[Info] Читаем: {src}")
    df = pd.read_csv(src)

    has_quizzes = "quizzes" in df.columns
    if has_quizzes:
        df = df.copy()
        df["holes"] = df["quizzes"].astype(str).apply(count_holes)
        if args.min_holes is not None:
            df = df[df["holes"] >= args.min_holes]
        if args.max_holes is not None:
            df = df[df["holes"] <= args.max_holes]
        hole_vals_csv = sorted(df["holes"].unique().tolist()) if len(df) else []
        cols_out = [c for c in df.columns if c != "holes"]
    else:
        print("[WARN] В CSV нет столбца 'quizzes' — всё будет сгенерировано.", file=sys.stderr)
        df = pd.DataFrame(columns=["quizzes", "holes"])
        hole_vals_csv = []
        cols_out = ["quizzes"] + (["solutions"] if "solutions" in df.columns else [])

    min_h = args.min_holes if args.min_holes is not None else (min(hole_vals_csv) if hole_vals_csv else 1)
    max_h = args.max_holes if args.max_holes is not None else (max(hole_vals_csv) if hole_vals_csv else 30)
    min_h = int(max(0, min_h))
    max_h = int(min(81, max_h))
    hole_vals_full = list(range(min_h, max_h + 1))

    print(f"[Info] Диапазон дырок: {min_h}..{max_h}")
    if not hole_vals_csv:
        print("[Info] В CSV нет задач в заданном диапазоне — всё будет сгенерировано.")


    backend = maybe_import_backend(args.backend)
    if backend is None:
        print("[ERROR] Бэкенд не импортирован, а CSV не покрывает корзины. Укажи --backend корректно.", file=sys.stderr)

    summary_rows = []
    test_all_parts = []

    for h in hole_vals_full:
        if hole_vals_csv and h in hole_vals_csv:
            bucket = df[df["holes"] == h]
            from_csv = min(args.per_bucket, len(bucket))
            sample_csv = bucket.sample(from_csv, random_state=args.seed).copy()
        else:
            sample_csv = pd.DataFrame(columns=cols_out + (["holes"] if "holes" not in cols_out else []))

        need = args.per_bucket - len(sample_csv)
        gen_df = None
        if need > 0:
            if backend is None:
                print(f"[WARN] holes={h}: нужно сгенерировать {need}, но backend недоступен — сохраню только из CSV ({len(sample_csv)})", file=sys.stderr)
            else:
                try:
                    print(f"[Fill] holes={h}: генерация {need} задач через {args.backend}...")
                    puzzles = backend.make_simple_sudoku(empty_cells=int(h), n_variants=int(need))
                    quizzes_new = [grid_to_quiz_str(p) for p in puzzles]
                    gen_df = pd.DataFrame({"quizzes": quizzes_new})
                    if "solutions" in df.columns:
                        gen_df["solutions"] = ""  
                    gen_df["holes"] = h
                except Exception as e:
                    print(f"[WARN] holes={h}: генерация не удалась: {e}", file=sys.stderr)

        parts = [sample_csv]
        if gen_df is not None and not gen_df.empty:
            need_cols = [c for c in ["quizzes", "solutions"] if c in gen_df.columns]
            gen_export = gen_df[need_cols]
            if "solutions" in df.columns and "solutions" not in gen_export.columns:
                gen_export["solutions"] = ""
            parts.append(gen_export)

        merged = pd.concat(parts, axis=0, ignore_index=True) if len(parts) > 1 else sample_csv

        if len(merged) > args.per_bucket:
            merged = merged.sample(args.per_bucket, random_state=args.seed)
        if len(merged) < args.per_bucket:
            print(f"[WARN] holes={h}: удалось собрать {len(merged)} из {args.per_bucket}", file=sys.stderr)

        out_name = f"sudoku_test_h{h}.csv"
        out_path = out_dir / out_name

        save_cols = ["quizzes"] + (["solutions"] if "solutions" in merged.columns else [])
        for c in save_cols:
            if c not in merged.columns:
                merged[c] = ""
        merged[save_cols].to_csv(out_path, index=False, encoding="utf-8")

        csv_used = len(sample_csv)
        generated = 0 if gen_df is None else len(gen_df)
        print(f"[OK] holes={h:>2}: saved {len(merged):>4} -> {out_path.name} (из CSV: {csv_used}, сгенерировано: {generated})")

        summary_rows.append({
            "holes": h,
            "target": args.per_bucket,
            "csv_used": int(csv_used),
            "generated": int(generated),
            "saved": int(len(merged)),
            "file": out_name
        })
        test_all_parts.append(merged[save_cols])

    if test_all_parts:
        test_all = pd.concat(test_all_parts, axis=0, ignore_index=True)
        all_path = out_dir / "sudoku_test_all.csv"
        test_all.to_csv(all_path, index=False, encoding="utf-8")
        print(f"[ALL] {len(test_all)} задач сохранено в {all_path.name}")

    summary_df = pd.DataFrame(summary_rows, columns=["holes", "target", "csv_used", "generated", "saved", "file"])
    summary_path = out_dir / "sudoku_test_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[Summary] {summary_path.name} создан. Готово.")


if __name__ == "__main__":
    main()