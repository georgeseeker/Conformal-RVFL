"""Conformal prediction sets for a three-head edRVFL.

Three methods, one model, two fits per fold:
  J3            aggregate score, 10-fold CV+ calibration   (proposed)
  PH-UNION-J3   per-head calibration, union of the sets    (ablation)
  SPLIT-J3      aggregate score, 25% held-out calibration  (split baseline)

Output: results/conformal.csv (or --out), one row per (task, seed, fold, method, alpha).
Failures are recorded as error rows, never dropped.
Each run deletes everything under results/ first, so old CSVs and analysis files do not mix in.
"""
import os
import sys
import csv
import time
import shutil
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))

from data_loader import (load_uci_dataset, fold_preprocess, should_standardize,  # noqa: E402
                         MULTICLASS_TASKS, GROUP)
from models.RVFLC import HeadedDeepRVFL3, CVPlusConformal  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30]
# Five seeds drawn once from 1..10000 and frozen for reproducibility.
SEEDS = [1618, 3979, 6187, 7333, 8443]
N_FOLDS = 5
CAL_FRAC = 0.25
CV_FOLDS = 10
N_HIDDEN, LAMDA = 50, 0.1
DEVICE = "cuda"
RESULTS_DIR = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS_DIR, "conformal.csv")
FIELDS = ["group", "uci_id", "task", "seed", "fold", "method", "n", "alpha", "K", "n_te", "n_tr",
          "n_cal", "n_unseen", "acc", "cover", "size", "size_q90", "frac_single", "frac_empty",
          "t_fit", "t_pv", "error"]
METHODS = ["J3", "PH-UNION-J3", "SPLIT-J3"]


def task_list(names=None):
    """The 25 multiclass tasks of tasks.MULTICLASS_TASKS (optionally filtered by name)."""
    core = [(u, n, GROUP) for u, n in MULTICLASS_TASKS]
    if names:
        return [t for t in core if t[1] in names]
    return core


def acc_of(classes, P, y_te):
    return float(np.mean(classes[np.argmax(P, axis=1)] == y_te))


def std_incl(P):
    return {a: P > a for a in ALPHAS}


def records_for(meta, r, y_te):
    cl = r["classes"]
    K, n = len(cl), len(y_te)
    idx = np.clip(np.searchsorted(cl, y_te), 0, K - 1)
    valid = cl[idx] == y_te
    out = []
    for a in ALPHAS:
        inc = r["incl"][a]
        size = inc.sum(axis=1)
        cover = valid & inc[np.arange(n), idx]
        rec = dict(meta)
        rec.update(alpha=a, K=int(K), n_te=int(n), n_tr=int(r["n_tr"]), n_cal=int(r["n_cal"]),
                   n_unseen=int((~valid).sum()), acc=float(r["acc"]), cover=float(cover.mean()),
                   size=float(size.mean()), size_q90=float(np.quantile(size, 0.9)),
                   frac_single=float((size == 1).mean()), frac_empty=float((size == 0).mean()),
                   t_fit=float(r["t_fit"]), t_pv=float(r["t_pv"]), error=None)
        out.append(rec)
    return out


def family_cross(X, y, tr, te, std, model_seed):
    """CV+ calibration on the training fold; aggregate set and per-head union."""
    Xtr, Xte = fold_preprocess(X[tr], X[te], std)
    m = CVPlusConformal(n_folds=CV_FOLDS, n_layers=3, n_hidden=N_HIDDEN, lamda=LAMDA,
                        device=DEVICE, seed=model_seed, head_mode="edrvfl")
    t0 = time.time()
    m.fit(Xtr, y[tr])                       # K fits + held-out scores
    t_fit = time.time() - t0
    t0 = time.time()
    P_h, P_a = m.pvalues(Xte, per_head=True)
    t_pv = time.time() - t0
    Q_h, Q_a = m.predict_proba(Xte)
    cl, yt = m.classes_, y[te]
    acc = acc_of(cl, Q_a, yt)
    base = dict(n_tr=len(tr), n_cal=len(tr), t_fit=t_fit, t_pv=t_pv, acc=acc)
    return {
        "J3": dict(classes=cl, incl=std_incl(P_a), **base),
        "PH-UNION-J3": dict(classes=cl, incl={a: (P_h > a).any(axis=0) for a in ALPHAS}, **base),
    }


def family_split(X, y, tr, te, std, model_seed):
    """Split conformal: 75/25 within the training fold, aggregate score."""
    try:
        pr, ca = train_test_split(tr, test_size=CAL_FRAC, stratify=y[tr], random_state=model_seed)
    except ValueError:
        pr, ca = train_test_split(tr, test_size=CAL_FRAC, random_state=model_seed)
    classes = np.unique(y[pr])
    ca = ca[np.isin(y[ca], classes)]                  # calibration labels must be seen in training
    Xpr, Xcte = fold_preprocess(X[pr], np.vstack([X[ca], X[te]]), std)
    Xca, Xte = Xcte[:len(ca)], Xcte[len(ca):]
    m = HeadedDeepRVFL3(n_layers=3, n_hidden=N_HIDDEN, lamda=LAMDA, device=DEVICE,
                        seed=model_seed, dtype="float64")
    t0 = time.time()
    m.fit(Xpr, y[pr])
    t_fit = time.time() - t0
    t0 = time.time()
    Pc_h, Pc_a = m.predict_proba(Xca)
    Pt_h, Pt_a = m.predict_proba(Xte)
    t_pv = time.time() - t0
    cl = m.classes_
    yi = np.searchsorted(cl, y[ca])
    idx = np.arange(len(ca))
    s_cal = 1.0 - Pc_a[idx, yi]
    s_sorted = np.sort(s_cal)
    n = len(s_cal)
    n_ge = n - np.searchsorted(s_sorted, 1.0 - Pt_a, side="left")
    P = (n_ge + 1.0) / (n + 1.0)
    return {"SPLIT-J3": dict(classes=cl, incl=std_incl(P), acc=acc_of(cl, Pt_a, y[te]),
                             n_tr=len(pr), n_cal=len(ca), t_fit=t_fit, t_pv=t_pv)}


FAMILIES = [
    (["J3", "PH-UNION-J3"], family_cross),
    (["SPLIT-J3"], family_split),
]


def run_fold(X, y, tr, te, std, seed, fold):
    model_seed = 1000 * seed + fold
    res = {}
    for mlist, fn in FAMILIES:
        try:
            res.update(fn(X, y, tr, te, std, model_seed))
        except Exception as e:              # failures are recorded, never silently dropped
            for m in mlist:
                res[m] = dict(error=f"{type(e).__name__}: {e}")
    return res


def load_done(path):
    done = set()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["group"], int(r["uci_id"]), r["task"], int(r["seed"]), int(r["fold"]),
                          r["method"]))
    return done


def run(out, seeds, names=None):
    """Run the selected tasks. Wipes results/ first so a new run never appends to old rows."""
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
        print(f"cleared {RESULTS_DIR}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tasks = task_list(names)
    if not tasks:
        raise SystemExit("没有匹配的任务。检查任务名，或论文主实验。")
    done = load_done(out)
    print(f"tasks={len(tasks)} seeds={seeds} alphas={ALPHAS} methods={METHODS} "
          f"done={len(done)} out={out}", flush=True)
    new_file = not os.path.exists(out) or os.path.getsize(out) == 0
    with open(out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="raise")
        if new_file:
            writer.writeheader()
        for seed in seeds:
            for uci_id, name, group in tasks:
                X, y, _, _ = load_uci_dataset(uci_id, name)
                y = np.asarray(y).ravel()
                std = should_standardize(uci_id)
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                for fold, (tr, te) in enumerate(skf.split(X, y), 1):
                    todo = [m for m in METHODS if (group, uci_id, name, seed, fold, m) not in done]
                    if not todo:
                        continue
                    res = run_fold(X, y, tr, te, std, seed, fold)
                    for m in todo:
                        meta = dict(group=group, uci_id=uci_id, task=name, seed=seed, fold=fold,
                                    method=m, n=int(len(y)))
                        r = res[m]
                        if "error" in r:
                            recs = []
                            for a in ALPHAS:
                                rec = dict(meta)
                                rec.update(alpha=a, error=r["error"], acc=None, cover=None, size=None)
                                recs.append(rec)
                        else:
                            recs = records_for(meta, r, y[te])
                        for rec in recs:
                            writer.writerow(rec)
                    f.flush()
            print(f"done seed={seed}", flush=True)
    print("ALL DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Conformal edRVFL experiments")
    ap.add_argument("--mode", choices=["paper"], default="paper",
                    help="paper: 25 multiclass UCI")
    ap.add_argument("--out", default="", help="CSV path; default results/conformal.csv")
    ap.add_argument("--seeds", default="", help="comma list, default the five frozen seeds")
    ap.add_argument("--tasks", default="", help="comma list of task names; overrides --mode")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    if args.tasks:
        names = set(s.strip() for s in args.tasks.split(",") if s.strip())
        out = args.out or OUT
    else:
        names = None
        out = args.out or OUT
    run(out, seeds, names)


if __name__ == "__main__":
    main()
