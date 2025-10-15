# train_pysr.py
import argparse, os, json
import numpy as np

from pysr import PySRRegressor

def load_npz(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.float32) 
    names = [str(s) for s in z["feature_names"].tolist()] if "feature_names" in z.files else [f"x{i}" for i in range(X.shape[1])]
    w = z["w"].astype(np.float32) if "w" in z.files else None
    return X, y, names, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Путь к .npz датасету (X,y[,w,feature_names])")
    ap.add_argument("--mode", choices=["policy_digit","policy_cell", "q_reg"], required=True)
    ap.add_argument("--out-prefix", required=True, help="Префикс для сохраняемых файлов")
    ap.add_argument("--max-iter", type=int, default=1500)
    ap.add_argument("--population", type=int, default=2000)
    ap.add_argument("--ncycles-per-iteration", type=int, default=300)
    ap.add_argument("--nworkers", type=int, default=0, help=">=1 включает multiprocessing (Julia procs)")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--timeout", type=int, default=0, help="Ограничение по времени (сек). 0=без лимита")
    args = ap.parse_args()

    X, y, names, w = load_npz(args.npz)

    parallelism = "multiprocessing" if (args.nworkers and args.nworkers > 0) else "serial"
    procs = int(max(1, args.nworkers)) if parallelism == "multiprocessing" else 0

    timeout = int(args.timeout) if args.timeout else 0

    binary_ops = [
        "+",
        "-",
        "*",
        "protected_division(x, y) = x / (y + one(x)*1f-9)",
    ]

    unary_ops = [
        "neg(x) = -x",
        "safe_abs(x) = ifelse(x >= zero(x), x, -x)",
    ]

    sympy_map = {
        "protected_division": lambda x, y: x / (y + 1e-9),
        "neg": lambda x: -x,
        "safe_abs": lambda x: abs(x),
    }


    elementwise_loss = "loss(x, y) = y*log1p(exp(-x)) + (1 - y)*log1p(exp(x))"

    model = PySRRegressor(
    niterations=args.max_iter,
    population_size=args.population,
    ncycles_per_iteration=args.ncycles_per_iteration,
    tournament_selection_n=max(3, args.population // 4),

    maxsize=20,

    parallelism=parallelism,            
    procs=procs,                        
    timeout_in_seconds=timeout,

    binary_operators=[
        "+", "-", "*",
        "protected_division(x, y) = x / (y + one(x)*1f-9)"
    ],
    unary_operators=[
        "neg(x) = -x",
        "safe_abs(x) = ifelse(x >= zero(x), x, -x)"
    ],
    extra_sympy_mappings={
        "protected_division": lambda x, y: x / (y + 1e-9),
        "neg": lambda x: -x,
        "safe_abs": lambda x: abs(x),
    },

    loss_scale="linear",

    batching=True,

    progress=bool(args.progress),
)

    model.fit(X, y, weights=w)

    eq_csv = args.out_prefix + "_equations.csv"
    mdl_pkl = args.out_prefix + "_model.pkl"
    model.equations_.to_csv(eq_csv, index=False)
    model.save(mdl_pkl)

    report = {
        "mode": args.mode,
        "npz": args.npz,
        "X_shape": list(X.shape),
        "y_stats": {"min": float(np.min(y)), "max": float(np.max(y)), "mean": float(np.mean(y))},
        "feature_names": names,
        "operators": {"binary": binary_ops, "unary": unary_ops},
        "hyperparams": {
            "niterations": args.max_iter,
            "population_size": args.population,
            "ncycles_per_iteration": args.ncycles_per_iteration,
        },
        "parallel": {"parallelism": parallelism, "procs": procs, "timeout_in_seconds": timeout},
        "out_files": {"equations_csv": eq_csv, "model_pkl": mdl_pkl},
        "best_equation": str(model.get_best()["equation"]) if model.get_best() is not None else None,
        "score": float(model.get_best()["score"]) if model.get_best() is not None else None,
    }
    with open(args.out_prefix + "_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {eq_csv}")
    print(f"[OK] Saved: {mdl_pkl}")
    if report["best_equation"] is not None:
        print(f"[Best] {report['best_equation']} | score={report['score']:.6f}")

if __name__ == "__main__":
    main()
