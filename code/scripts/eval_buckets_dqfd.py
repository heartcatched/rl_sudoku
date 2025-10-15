# eval_buckets_dqfd.py
import argparse, subprocess, sys, re
from pathlib import Path
import pandas as pd

LINE_PAT = re.compile(
    r"Tested on\s+(\d+)\s+\|\s+Solved:\s+(\d+)\s+\(([\d.]+)\)\s+\|\s+AvgR:\s+([-\d.]+)"
)

def parse_metrics(text: str):
    """Вытащить метрики из stdout строгого скрипта."""
    for line in text.splitlines():
        m = LINE_PAT.search(line)
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
    ap.add_argument("--model", type=str, default="dqn_dqfd.pth",
                    help="Путь к весам DQfD модели")
    ap.add_argument("--backend", type=str, default="sudoku_rl_patched",
                    help="Имя backend модуля")
    ap.add_argument("--script", type=str, default="sudoku_rl_dqfd_eval.py",
                    help="Скрипт, выполняющий строгий тест")
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
            "--backend", args.backend,
            "--test-model", args.model,
            "--csv", str(f),
            "--test-n", str(args.test_n),
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
        print("[ERROR] Не удалось собрать ни одной метрики — проверь путь к скрипту/модели.")
        sys.exit(3)

    df = pd.DataFrame(rows).sort_values("holes")
    out_csv = root / "dqfd_eval_summary.csv"
    df.to_csv(out_csv, index=False)
    print("\n=== DQfD STRICT summary ===")
    print(df.to_string(index=False))
    print(f"\n[OK] Сводка сохранена: {out_csv}")

if __name__ == "__main__":
    main()