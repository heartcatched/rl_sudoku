# eval_buckets_dqn.py
import argparse, subprocess, sys, re
from pathlib import Path
import pandas as pd

LINE_PATTERNS = [
    re.compile(r"Tested on\s+(\d+)\s+puzzles\s+\|\s+Solved:\s+(\d+)\s+\(([\d.]+)\)\s+\|\s+Avg Reward:\s+([-\d.]+)"),
    re.compile(r"Tested on\s+(\d+)\s+\|\s+Solved:\s+(\d+)\s+\(([\d.]+)\)\s+\|\s+AvgR:\s+([-\d.]+)"),
]

def parse_metrics(text: str):
    """Пытается вытащить метрики из вывода теста."""
    for line in text.splitlines():
        for pat in LINE_PATTERNS:
            m = pat.search(line)
            if m:
                return {
                    "total": int(m.group(1)),
                    "solved": int(m.group(2)),
                    "solved_rate": float(m.group(3)),
                    "avg_reward": float(m.group(4)),
                }
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Папка с sudoku_test_h*.csv (обычно рядом с data/sudoku.csv)")
    ap.add_argument("--model", type=str, required=True,
                    help="Путь к весам DQN, например dqn_final.pth или sudoku_dqn_final_10empty.pth")
    ap.add_argument("--script", type=str, default="sudoku_rl_patched.py",
                    help="Скрипт тестирования DQN (по умолчанию sudoku_rl_patched.py)")
    ap.add_argument("--test_n", type=int, default=300,
                    help="Сколько задач брать из файла (обычно 300)")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("sudoku_test_h*.csv"))
    if not files:
        print(f"[ERROR] Не нашли sudoku_test_h*.csv в {root}")
        sys.exit(2)

    rows = []
    for f in files:
        m = re.search(r"h(\d+)", f.stem)
        holes = int(m.group(1)) if m else -1
        print(f"[RUN] holes={holes:>2} -> {f.name}")

        cmd = [
            sys.executable, args.script,
            "--test-model", args.model,
            "--test-data", str(f),
            "--test-limit", str(args.test_n),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = (res.stdout or "") + "\n" + (res.stderr or "")
        metrics = parse_metrics(out)

        if not metrics:
            print(out)
            print(f"[WARN] Не удалось распарсить метрики для {f.name}")
            continue

        rows.append({"holes": holes, **metrics})

    if not rows:
        print("[ERROR] Ничего не распознано — проверь вывод твоего скрипта/путь к модели.")
        sys.exit(3)

    df = pd.DataFrame(rows).sort_values("holes")
    out_csv = root / "dqn_eval_summary.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== DQN summary ===")
    print(df.to_string(index=False))
    print(f"\n[OK] Сводка сохранена: {out_csv}")

if __name__ == "__main__":
    main()