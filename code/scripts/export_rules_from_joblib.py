import argparse
import os
import sys
import joblib

def extract_rule_from_estimator(est):
    """Return a string representation of the best symbolic program inside a gplearn estimator."""
    prog = None
    try:
        if hasattr(est, "_program") and est._program is not None:
            prog = est._program
        elif hasattr(est, "_best_programs") and est._best_programs:
            for p in reversed(est._best_programs):
                if p is not None:
                    prog = p
                    break
        elif hasattr(est, "program_") and est.program_ is not None:
            prog = est.program_
    except Exception as e:
        return f"[WARN] cannot extract program: {e}"
    return str(prog) if prog is not None else "None"

def export_rules_from_joblib(model_path, out_path=None):
    model = joblib.load(model_path)
    lines = []
    header = [f"# Model: {os.path.basename(model_path)}",
              f"# Type: {type(model)}",
              ""]
    lines.extend(header)

    if hasattr(model, "estimators_") and hasattr(model, "classes_"):
        try:
            for cls_idx, est in enumerate(model.estimators_):
                cls_label = model.classes_[cls_idx] if hasattr(model, "classes_") else cls_idx
                rule = extract_rule_from_estimator(est)
                lines.append(f"[Class {cls_label}] {rule}")
        except Exception as e:
            lines.append(f"[ERROR] Failed to iterate estimators_: {e}")
    else:
        lines.append(str(model))

    text = "\n".join(lines)

    if out_path is None:
        base = os.path.splitext(os.path.basename(model_path))[0]
        out_path = f"{base}_rules.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help="Path(s) to .joblib file(s)")
    ap.add_argument("--out", default=None, help="Optional output file (used if one model)")
    args = ap.parse_args()

    if len(args.model) > 1 and args.out:
        print("[WARN] --out is ignored when multiple models are provided; using auto names per model.", file=sys.stderr)

    for m in args.model:
        out = args.out if (args.out and len(args.model) == 1) else None
        out_path = export_rules_from_joblib(m, out)
        print(f"[OK] Wrote: {out_path}")

if __name__ == "__main__":
    main()
