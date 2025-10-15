# neuro_symbolic_sudoku.py
r"""
Neuro-Symbolic Sudoku (human-action features)
---------------------------------------------
- Использует human_features_sudoku.py: build_action_feature_matrix(grid) -> (idxs, X, names)
- Интерпретируемая политика: LogisticRegression (L1/elastic) или DecisionTree
- Режимы:
  * TRAIN на CSV с solutions (teacher forcing, pairwise)
  * EVAL strict (совпадение с solutions) и struct (валидность без solutions)
  * EVAL buckets (строгий/структурный)
  * EVAL synth (генерация б/о CSV)
  * КЭШ фичей в NPZ (быстрое переобучение)
  * Диагностика: one-step accuracy, inspect example, dump action probs

Примеры:
# Обучить логрег + сохранить важности
python neuro_symbolic_sudoku.py ^
  --backend sudoku_rl_patched ^
  --train-csv C:\Users\ilya\Downloads\data\sudoku.csv ^
  --limit 8000 ^
  --model-out C:\Users\ilya\Downloads\ns_logreg.joblib ^
  --model-type logreg --penalty l1 --C 0.5 --max-neg-per-state 8 ^
  --export-weights C:\Users\ilya\Downloads\ns_logreg_weights.csv

# Строгая проверка на всех бакетах (solutions присутствуют)
python neuro_symbolic_sudoku.py ^
  --backend sudoku_rl_patched ^
  --eval-root C:\Users\ilya\Downloads\data ^
  --model C:\Users\ilya\Downloads\ns_logreg.joblib ^
  --holes 1-30 --per-bucket 300

# Структурная проверка (если solutions нет)
python neuro_symbolic_sudoku.py ^
  --backend sudoku_rl_patched ^
  --eval-root C:\Users\ilya\Downloads\no_solutions_data ^
  --model C:\Users\ilya\Downloads\ns_logreg.joblib ^
  --holes 10-20 --per-bucket 300 --struct

# Кэш фичей в NPZ (быстрое переобучение)
python neuro_symbolic_sudoku.py ^
  --backend sudoku_rl_patched ^
  --cache-features C:\Users\ilya\Downloads\data\sudoku.csv ^
  --cache-out C:\Users\ilya\Downloads\data\sudoku_features.npz ^
  --limit 8000 --max-neg-per-state 8

# Обучение из NPZ кэша
python neuro_symbolic_sudoku.py ^
  --train-npz C:\Users\ilya\Downloads\data\sudoku_features.npz ^
  --model-out C:\Users\ilya\Downloads\ns_logreg.joblib ^
  --model-type logreg --penalty l1 --C 0.5

# One-step accuracy (GT-выбор в одном состоянии)
python neuro_symbolic_sudoku.py ^
  --backend sudoku_rl_patched ^
  --model C:\Users\ilya\Downloads\ns_logreg.joblib ^
  --one-step-acc C:\Users\ilya\Downloads\data\sudoku_test_h20.csv ^
  --test-n 300
"""

import argparse, sys, random, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

def import_backend(name: str):
    try:
        return __import__(name, fromlist=['*'])
    except Exception as e:
        print(f"[ERROR] backend import failed: {e}")
        sys.exit(1)

def encode_action_idx(a: Tuple[int,int,int]) -> int:
    r,c,d=a; return (r*9+c)*9+(d-1)

def decode_action_idx(idx: int) -> Tuple[int,int,int]:
    d=(idx%9)+1; rc=idx//9; r,c=divmod(rc,9); return (r,c,d)

def get_features_for_grid(grid: np.ndarray):
    """
    Ожидается в human_features_sudoku.py:
    build_action_feature_matrix(grid) -> (idxs, X, names)
    """
    try:
        from human_features_sudoku import build_action_feature_matrix
    except Exception as e:
        raise RuntimeError(
            "Импорт human_features_sudoku.build_action_feature_matrix не удался. "
            "Проверь файл/функцию."
        ) from e
    idxs, X, names = build_action_feature_matrix(grid)
    return list(map(int, idxs)), np.asarray(X, dtype=np.float32), list(map(str, names))

def is_valid_sudoku_grid(grid: np.ndarray) -> bool:
    S = set(range(1,10))
    for r in range(9):
        if set(grid[r,:]) != S: return False
    for c in range(9):
        if set(grid[:,c]) != S: return False
    for br in range(0,9,3):
        for bc in range(0,9,3):
            if set(grid[br:br+3, bc:bc+3].reshape(-1)) != S: return False
    return True

def build_training_pairs_from_csv(backend, df: pd.DataFrame, limit: Optional[int],
                                  neg_cap_per_state: int = 8, seed: int = 42):
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=seed)
    X_list, y_list = [], []
    feat_names = None

    for _, row in df.iterrows():
        q = str(row["quizzes"]); s = str(row.get("solutions",""))
        if len(q) != 81 or len(s) != 81: continue
        grid0 = np.array([int(ch) for ch in q], dtype=int).reshape(9,9)
        sol   = np.array([int(ch) for ch in s], dtype=int).reshape(9,9)
        env = backend.SudokuEnv(grid0.copy()); state = env.reset()
        steps=0
        while (state==0).any() and steps<81:
            empties = list(zip(*np.where(state==0)))
            if not empties: break
            r,c = min(empties)
            d = int(sol[r,c]); gt_idx = encode_action_idx((r,c,d))
            idxs, X, names = get_features_for_grid(state)
            if feat_names is None: feat_names = names
            try:
                pos = idxs.index(gt_idx)
            except ValueError:
                ns, _, _, _ = env.step((r,c,d)); state = ns; steps+=1; continue
            y = np.zeros((len(idxs),), dtype=np.int8); y[pos]=1
            if neg_cap_per_state>0 and (len(idxs)-1)>neg_cap_per_state:
                negs = [i for i in range(len(idxs)) if i!=pos]
                keep = [pos] + sorted(random.sample(negs, neg_cap_per_state))
                X = X[keep]; y = y[keep]
            X_list.append(X); y_list.append(y)
            ns, _, _, _ = env.step((r,c,d)); state = ns; steps+=1

    if not X_list: raise RuntimeError("Не удалось собрать пары — проверь CSV/совместимость фич с env.")
    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int8)
    return X_all, y_all, feat_names

def cache_features_from_csv(backend, csv_path: str, out_npz: str,
                            limit: Optional[int], neg_cap_per_state: int, seed: int=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["quizzes","solutions"])
    df = df[(df["quizzes"].astype(str).str.len()==81) & (df["solutions"].astype(str).str.len()==81)]
    X, y, names = build_training_pairs_from_csv(backend, df, limit, neg_cap_per_state, seed)
    np.savez_compressed(out_npz, X=X, y=y, names=np.array(names, dtype=object))
    print(f"[CACHE] saved: {out_npz} | X={X.shape} y={y.shape} F={len(names)}")

def load_feature_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32); y = z["y"].astype(np.int8)
    names = list(z["names"].tolist())
    return X, y, names

def train_model(model_type: str, X: np.ndarray, y: np.ndarray,
                penalty: str = "l1", C: float = 0.5,
                max_depth: int = 6, min_samples_leaf: int = 10,
                random_state: int = 42):
    if model_type == "logreg":
        clf = LogisticRegression(
            penalty=penalty, C=C, solver="saga",
            max_iter=2000, class_weight="balanced",
            n_jobs=-1, random_state=random_state
        )
    elif model_type == "tree":
        clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight="balanced", random_state=random_state
        )
    else:
        raise ValueError("model_type ∈ {logreg, tree}")
    clf.fit(X, y)
    return clf

def greedy_play_with_model(backend, model, grid: np.ndarray, max_steps: int = 81):
    env = backend.SudokuEnv(grid.copy()); s = env.reset()
    total_r = 0.0; steps=0; done=False
    while not done and steps<max_steps:
        idxs, X, _ = get_features_for_grid(s)
        if len(idxs)==0: break
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[:,1]
        else:
            z = model.decision_function(X); p = 1.0/(1.0+np.exp(-z))
        a_idx = idxs[int(np.argmax(p))]
        a = decode_action_idx(a_idx)
        ns, r, done, _ = env.step(a)
        total_r += float(r); s = ns; steps+=1
    return env.grid, total_r, steps

def strict_eval_csv(backend, model, csv_path: str, test_n: int = 300, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["quizzes","solutions"])
    df = df[(df["quizzes"].astype(str).str.len()==81) & (df["solutions"].astype(str).str.len()==81)]
    if len(df)>test_n: df = df.sample(test_n, random_state=seed)
    total=0; solved=0; total_r=0.0
    for _, row in df.iterrows():
        q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
        s = np.array([int(ch) for ch in str(row["solutions"])], dtype=int).reshape(9,9)
        fin, ep_r, _ = greedy_play_with_model(backend, model, q, max_steps=81)
        total_r += ep_r; solved += int((fin==s).all()); total+=1
    return {'total':total, 'solved':solved, 'solved_rate':solved/max(total,1), 'avg_reward': total_r/max(total,1)}

def struct_eval_csv(backend, model, csv_path: str, test_n: int = 300, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["quizzes"])
    df = df[(df["quizzes"].astype(str).str.len()==81)]
    if len(df)>test_n: df = df.sample(test_n, random_state=seed)
    total=0; solved=0; total_r=0.0
    for _, row in df.iterrows():
        q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
        fin, ep_r, _ = greedy_play_with_model(backend, model, q, max_steps=81)
        total_r += ep_r; solved += int(is_valid_sudoku_grid(fin)); total+=1
    return {'total':total, 'solved':solved, 'solved_rate':solved/max(total,1), 'avg_reward': total_r/max(total,1)}

def parse_holes_arg(s: str) -> List[int]:
    out=[]
    for part in s.split(","):
        part=part.strip()
        if "-" in part:
            a,b=part.split("-"); out+=list(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return sorted(set(out))

def eval_buckets_strict(backend, model, root: str, holes: List[int], per_bucket: int, seed: int):
    rows=[]; rootp=Path(root)
    for h in holes:
        fp = rootp / f"sudoku_test_h{h}.csv"
        if not fp.exists(): print(f"[SKIP] {fp}"); continue
        res = strict_eval_csv(backend, model, str(fp), test_n=per_bucket, seed=seed)
        print(f"[holes={h:2d}] Tested on {res['total']} | Solved: {res['solved']} ({res['solved_rate']:.3f}) | AvgR: {res['avg_reward']:.2f}")
        rows.append({'holes':h, **res})
    if rows:
        out = rootp / "neurosym_eval_summary_strict.csv"
        pd.DataFrame(rows).sort_values("holes").to_csv(out, index=False)
        print(f"[OK] saved: {out}")

def eval_buckets_struct(backend, model, root: str, holes: List[int], per_bucket: int, seed: int):
    rows=[]; rootp=Path(root)
    for h in holes:
        fp = rootp / f"sudoku_test_h{h}.csv"
        if not fp.exists(): print(f"[SKIP] {fp}"); continue
        res = struct_eval_csv(backend, model, str(fp), test_n=per_bucket, seed=seed)
        print(f"[holes={h:2d}] Tested(struct) on {res['total']} | Solved: {res['solved']} ({res['solved_rate']:.3f}) | AvgR: {res['avg_reward']:.2f}")
        rows.append({'holes':h, **res})
    if rows:
        out = rootp / "neurosym_eval_summary_struct.csv"
        pd.DataFrame(rows).sort_values("holes").to_csv(out, index=False)
        print(f"[OK] saved: {out}")

def eval_synth(backend, model, holes: int, n: int=300, seed: int=42):
    random.seed(seed); np.random.seed(seed)
    puzzles = backend.make_simple_sudoku(empty_cells=holes, n_variants=n)
    total=0; solved=0; total_r=0.0
    for q in puzzles:
        fin, ep_r, _ = greedy_play_with_model(backend, model, q, max_steps=81)
        total_r += ep_r; solved += int(is_valid_sudoku_grid(fin)); total+=1
    print(f"[SYNTH h={holes}] Tested {total} | Solved: {solved} ({solved/max(total,1):.3f}) | AvgR: {total_r/max(total,1):.2f}")

def one_step_accuracy_csv(backend, model, csv_path: str, test_n: int=300, seed: int=42):
    """
    На первом GT-состоянии проверяет, выбирает ли модель тот же ход (top-1).
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["quizzes","solutions"])
    df = df[(df["quizzes"].astype(str).str.len()==81) & (df["solutions"].astype(str).str.len()==81)]
    if len(df)>test_n: df = df.sample(test_n, random_state=seed)
    total=0; correct=0
    for _, row in df.iterrows():
        q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
        s = np.array([int(ch) for ch in str(row["solutions"])], dtype=int).reshape(9,9)
        empties = list(zip(*np.where(q==0)))
        if not empties: continue
        r,c = min(empties); d=int(s[r,c]); gt_idx=encode_action_idx((r,c,d))
        idxs, X, _ = get_features_for_grid(q)
        if len(idxs)==0: continue
        if hasattr(model,"predict_proba"): p=model.predict_proba(X)[:,1]
        else: z=model.decision_function(X); p=1.0/(1.0+np.exp(-z))
        pred_idx = idxs[int(np.argmax(p))]
        correct += int(pred_idx==gt_idx); total+=1
    print(f"[OneStep] {correct}/{total} ({(correct/max(total,1)):.3f}) top-1 GT move")

def inspect_example(backend, model, csv_path: str, row_id: int=0, topk: int=10):
    """
    Печать топ-фичей для выбора модели на первой позиции.
    """
    df = pd.read_csv(csv_path)
    row = df.iloc[row_id]
    q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
    idxs, X, names = get_features_for_grid(q)
    if hasattr(model,"predict_proba"):
        p = model.predict_proba(X)[:,1]
        w = getattr(model, "coef_", None)
    else:
        z = model.decision_function(X); p=1.0/(1.0+np.exp(-z)); w=None
    best = int(np.argmax(p)); a = decode_action_idx(idxs[best])
    print(f"[Inspect] best action={a} prob={p[best]:.3f}")
    if w is not None and w.ndim==2 and w.shape[0]==1:
        wv = w.reshape(-1)
        order = np.argsort(-np.abs(wv))[:topk]
        print("[Top global weights]")
        for i in order:
            nm = names[i] if i<len(names) else f"f{i}"
            print(f"{nm:30s}  w={wv[i]:+.3f}")

def dump_action_probs_for_grid(backend, model, csv_path: str, out_json: str, row_id: int=0):
    df = pd.read_csv(csv_path)
    row = df.iloc[row_id]
    q = np.array([int(ch) for ch in str(row["quizzes"])], dtype=int).reshape(9,9)
    idxs, X, names = get_features_for_grid(q)
    if hasattr(model,"predict_proba"): p=model.predict_proba(X)[:,1].tolist()
    else:
        z=model.decision_function(X); p=(1.0/(1.0+np.exp(-z))).tolist()
    actions = [decode_action_idx(i) for i in idxs]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"actions": actions, "probs": p, "feature_names": names}, f, ensure_ascii=False, indent=2)
    print(f"[DUMP] {out_json}")

def export_feature_importance_csv(model, names: List[str], out_csv: str):
    if not hasattr(model, "coef_"):
        print("[WARN] export_feature_importance_csv: модель не поддерживает coef_. Пропуск.")
        return
    w = model.coef_.reshape(-1)
    df = pd.DataFrame({"feature": names, "weight": w, "abs_weight": np.abs(w)})
    df = df.sort_values("abs_weight", ascending=False)
    df.to_csv(out_csv, index=False)
    print(f"[EXPORT] weights -> {out_csv}")

def export_tree_rules_txt(model, names: List[str], out_txt: str, depth: int = 6):
    if not isinstance(model, DecisionTreeClassifier):
        print("[WARN] export_tree_rules_txt: это не DecisionTreeClassifier. Пропуск.")
        return
    txt = export_text(model, feature_names=names, max_depth=depth)
    Path(out_txt).write_text(txt, encoding="utf-8")
    print(f"[EXPORT] tree rules -> {out_txt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sudoku_rl_patched")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train-csv", type=str, default=None)
    ap.add_argument("--train-npz", type=str, default=None)          
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--max-neg-per-state", type=int, default=8)
    ap.add_argument("--model-out", type=str, default="ns_logreg.joblib")
    ap.add_argument("--model-type", choices=["logreg","tree"], default="logreg")
    ap.add_argument("--penalty", choices=["l1","l2","elasticnet"], default="l1")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-leaf", type=int, default=10)

    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--eval-csv", type=str, default=None)
    ap.add_argument("--struct", action="store_true") 
    ap.add_argument("--test-n", type=int, default=300)

    ap.add_argument("--eval-root", type=str, default=None)
    ap.add_argument("--holes", type=str, default="1-30")
    ap.add_argument("--per-bucket", type=int, default=300)

    ap.add_argument("--eval-synth", type=int, default=None)  
    ap.add_argument("--eval-synth-n", type=int, default=300)

    ap.add_argument("--cache-features", type=str, default=None)
    ap.add_argument("--cache-out", type=str, default=None)

    ap.add_argument("--one-step-acc", type=str, default=None)   
    ap.add_argument("--inspect", type=str, default=None)        
    ap.add_argument("--inspect-row", type=int, default=0)
    ap.add_argument("--dump-probs", type=str, default=None)     
    ap.add_argument("--export-weights", type=str, default=None) 
    ap.add_argument("--export-tree", type=str, default=None)    

    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    backend = import_backend(args.backend)

    if args.cache_features and args.cache_out:
        cache_features_from_csv(backend, args.cache_features, args.cache_out, args.limit, args.max_neg_per_state, args.seed)
        return

    if args.train_npz:
        X, y, names = load_feature_npz(args.train_npz)
        print(f"[TRAIN/NPZ] X={X.shape} y={y.shape} F={len(names)}")
        clf = train_model(args.model_type, X, y, args.penalty, args.C, args.max_depth, args.min_leaf, args.seed)
        dump({'model':clf,'feat_names':names}, args.model_out)
        print(f"[SAVE] {args.model_out}")
        if args.export_weights: export_feature_importance_csv(clf, names, args.export_weights)
        if args.export_tree: export_tree_rules_txt(clf, names, args.export_tree)
        return

    if args.train_csv:
        df = pd.read_csv(args.train_csv)
        df = df.dropna(subset=["quizzes","solutions"])
        df = df[(df["quizzes"].astype(str).str.len()==81) & (df["solutions"].astype(str).str.len()==81)]
        print(f"[TRAIN/CSV] build pairs: limit={args.limit} neg_cap={args.max_neg_per_state}")
        X, y, names = build_training_pairs_from_csv(backend, df, args.limit, args.max_neg_per_state, args.seed)
        print(f"[TRAIN] samples={X.shape[0]} features={X.shape[1]}")
        clf = train_model(args.model_type, X, y, args.penalty, args.C, args.max_depth, args.min_leaf, args.seed)
        dump({'model':clf,'feat_names':names}, args.model_out)
        print(f"[SAVE] {args.model_out}")
        if args.export_weights: export_feature_importance_csv(clf, names, args.export_weights)
        if args.export_tree: export_tree_rules_txt(clf, names, args.export_tree)

    pack=None; model=None; names=None
    if args.model:
        pack = load(args.model)
        model = pack['model']; names = pack.get('feat_names', None)

    if args.eval_csv and model is not None:
        if args.struct:
            res = struct_eval_csv(backend, model, args.eval_csv, test_n=args.test_n, seed=args.seed)
            print(f"[EVAL/STRUCT] Tested on {res['total']} | Solved: {res['solved']} ({res['solved_rate']:.3f}) | AvgR: {res['avg_reward']:.2f}")
        else:
            res = strict_eval_csv(backend, model, args.eval_csv, test_n=args.test_n, seed=args.seed)
            print(f"[EVAL/STRICT] Tested on {res['total']} | Solved: {res['solved']} ({res['solved_rate']:.3f}) | AvgR: {res['avg_reward']:.2f}")

    if args.eval_root and model is not None:
        holes = parse_holes_arg(args.holes)
        if args.struct:
            eval_buckets_struct(backend, model, args.eval_root, holes, args.per_bucket, args.seed)
        else:
            eval_buckets_strict(backend, model, args.eval_root, holes, args.per_bucket, args.seed)

    if args.eval_synth is not None and model is not None:
        eval_synth(backend, model, args.eval_synth, n=args.eval_synth_n, seed=args.seed)

    if args.one_step_acc and model is not None:
        one_step_accuracy_csv(backend, model, args.one_step_acc, test_n=args.test_n, seed=args.seed)
    if args.inspect and model is not None:
        inspect_example(backend, model, args.inspect, row_id=args.inspect_row)
    if args.dump_probs and args.inspect and model is not None:
        dump_action_probs_for_grid(backend, model, args.inspect, args.dump_probs, row_id=args.inspect_row)

if __name__ == "__main__":
    main()
