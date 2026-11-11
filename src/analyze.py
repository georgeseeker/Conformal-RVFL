"""Analysis of the conformal edRVFL experiment.

Two contrasts:
  P1  J3 - PH-UNION-J3   aggregate-then-calibrate vs per-head calibration + union
  S1  J3 - SPLIT-J3      CV+ calibration vs standard split conformal
Reads results/conformal.csv.
"""
import collections
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "conformal.csv")

TARGET = 0.90
ALPHA_REF = 0.10
ACC_TOL = 0.01
N_BOOT = 10000
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30]
METHODS = ["J3", "PH-UNION-J3", "SPLIT-J3"]
# Must match experiment.SEEDS (five draws from 1..10000). Old rows with seeds 30-39 are ignored.
SEEDS = [1618, 3979, 6187, 7333, 8443]
N_EXPECT = len(SEEDS) * 5
# Scope: exactly the 25 multiclass tasks of tasks.MULTICLASS_TASKS, matched on (group, uci_id, task).
# Any other rows that may exist in a CSV (old binary/Flags runs) are listed as out of scope, not dropped.
sys.path.insert(0, os.path.join(ROOT, "src"))
from tasks import TASK_KEYS  # noqa: E402

CONTRASTS = [
    ("P1 J3 - PH-UNION-J3  (PRIMARY: aggregate then calibrate vs per-head union)", "J3", "PH-UNION-J3"),
    ("S1 J3 - SPLIT-J3     (CV+ vs standard split conformal)", "J3", "SPLIT-J3"),
]

NUMERIC = {"uci_id": int, "seed": int, "fold": int, "n": int, "alpha": float, "K": int, "n_te": int,
           "n_tr": int, "n_cal": int, "n_unseen": int, "acc": float, "cover": float, "size": float,
           "size_q90": float, "frac_single": float, "frac_empty": float, "t_fit": float, "t_pv": float}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def wsr_p(d):
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, alternative="two-sided").pvalue)


def boot_ci(d, rng):
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(N_BOOT)])
    return np.percentile(boots, [2.5, 97.5])


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in NUMERIC:
                    row[k] = NUMERIC[k](v) if v != "" else None
                else:
                    row[k] = v if v != "" else None
            rows.append(row)
    return rows


def main():
    rows = load_csv(PATH)
    n_err = sum(1 for r in rows if r.get("error"))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("error"):
            continue
        if r.get("seed") not in SEEDS:
            continue
        key = ((r["group"], r["uci_id"], r["task"]), r["method"], r["alpha"])
        for f in ("cover", "size", "acc", "t_fit", "t_pv", "frac_single", "frac_empty"):
            agg[key][f].append(r[f])

    all_tasks = sorted({(r["group"], r["uci_id"], r["task"]) for r in rows})
    in_scope = [t for t in all_tasks if t in TASK_KEYS]
    out_of_scope = [t for t in all_tasks if t not in TASK_KEYS]
    print(f"scope: the {len(TASK_KEYS)} multiclass tasks of src/tasks.py "
          f"-> {len(in_scope)} of {len(all_tasks)} tasks in CSV "
          f"({len(out_of_scope)} out of scope, not by failure{': ' + str(out_of_scope) if out_of_scope else ''})")
    missing = sorted(TASK_KEYS - set(all_tasks))
    if missing:
        print(f"WARNING: kept tasks absent from CSV: {missing}")
    complete = [t for t in in_scope
                if all(len(agg[(t, m, a)]["cover"]) == N_EXPECT for m in METHODS for a in ALPHAS)]
    print(f"records={len(rows)} errors={n_err} tasks={len(all_tasks)} complete={len(complete)}")
    for t in in_scope:
        if t not in complete:
            print(f"  excluded {t[2]}: "
                  f"{ {m: sum(1 for a in ALPHAS if len(agg[(t, m, a)]['cover']) != N_EXPECT) for m in METHODS if any(len(agg[(t, m, a)]['cover']) != N_EXPECT for a in ALPHAS)} }")

    def cov(t, m, a):
        return float(np.mean(agg[(t, m, a)]["cover"]))

    def size(t, m, a):
        return float(np.mean(agg[(t, m, a)]["size"]))

    def size_at_target(t, m):
        cs = np.array([cov(t, m, a) for a in ALPHAS])
        ss = np.array([size(t, m, a) for a in ALPHAS])
        order = np.argsort(cs)
        cs, ss = cs[order], ss[order]
        if not (cs.min() <= TARGET <= cs.max()):
            return None
        return float(np.interp(TARGET, cs, ss))

    star = {(t, m): size_at_target(t, m) for t in complete for m in METHODS}
    print(f"\n== tasks where {TARGET:.2f} is outside a method's coverage range ==")
    for m in METHODS:
        out = [t[2] for t in complete if star[(t, m)] is None]
        print(f"  {m:13s}: {len(out)} {out if out else ''}")

    rng = np.random.default_rng(0)
    res = {}
    print(f"\n== matched-coverage set size at {TARGET:.2f} (negative = first is smaller) ==")
    for label, a, b in CONTRASTS:
        tasks = [t for t in complete if star[(t, a)] is not None and star[(t, b)] is not None]
        d = np.array([star[(t, a)] - star[(t, b)] for t in tasks])
        rel = np.array([(star[(t, a)] - star[(t, b)]) / star[(t, b)] for t in tasks]) * 100
        ci = boot_ci(d, rng)
        p = wsr_p(d)
        res[label] = dict(tasks=tasks, d=d, mean=float(d.mean()), ci=ci, p=p, rel=float(rel.mean()),
                          wins=int((d < 0).sum()), ties=int((d == 0).sum()), losses=int((d > 0).sum()))
        print(f"{label}: n={len(tasks)} mean diff={d.mean():+.3f}, rel={rel.mean():+.1f}%, "
              f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], Wilcoxon p={p:.4g}, "
              f"smaller/tie/larger={res[label]['wins']}/{res[label]['ties']}/{res[label]['losses']}")

    labels = [c[0] for c in CONTRASTS]
    adj = holm([res[k]["p"] for k in labels])
    print("\n== Holm across the two pre-specified contrasts ==")
    for k, a_ in zip(labels, adj):
        res[k]["holm"] = a_
        print(f"  {k.split(' ')[0]}: raw p={res[k]['p']:.4g}, Holm p={a_:.4g}")

    p1 = res[labels[0]]
    t1 = p1["tasks"]
    acc = {(t, m): float(np.mean(agg[(t, m, ALPHA_REF)]["acc"])) for t in complete for m in METHODS}
    dacc = {t: acc[(t, "J3")] - acc[(t, "PH-UNION-J3")] for t in t1}
    print("\n== accuracy check for P1 (same model, so accuracy is identical by construction) ==")
    print(f"  max |acc(J3) - acc(PH-UNION-J3)| over tasks: {max(abs(v) for v in dacc.values()):.2e}")

    print(f"\n== coverage at alpha={ALPHA_REF:.2f} (target {1 - ALPHA_REF:.2f}) ==")
    for m in METHODS:
        cs = [cov(t, m, ALPHA_REF) for t in complete]
        print(f"  {m:13s}: mean={np.mean(cs):.3f} (dev {np.mean(cs) - (1 - ALPHA_REF):+.3f}), "
              f"tasks with mean < 0.88: {sum(c < 0.88 for c in cs)}/{len(cs)}")

    print(f"\n== set structure at alpha={ALPHA_REF:.2f} (mean over complete tasks) ==")
    for m in METHODS:
        fs = np.mean([np.mean(agg[(t, m, ALPHA_REF)]["frac_single"]) for t in complete])
        fe = np.mean([np.mean(agg[(t, m, ALPHA_REF)]["frac_empty"]) for t in complete])
        print(f"  {m:13s}: singleton={fs:.3f}, empty={fe:.3f}")

    print("\n== compute time (mean seconds per fold) ==")
    for m in METHODS:
        tf = [np.mean(agg[(t, m, ALPHA_REF)]["t_fit"]) for t in complete]
        tp = [np.mean(agg[(t, m, ALPHA_REF)]["t_pv"]) for t in complete]
        print(f"  {m:13s}: fit={np.mean(tf):.4f}s, scoring={np.mean(tp):.4f}s")

    s1 = res[labels[1]]
    print("\n== decision (plan rules) ==")
    crit = lambda r: r["mean"] < 0 and r["ci"][1] < 0 and r["holm"] < 0.05
    print(f"  H1 aggregate-then-calibrate beats per-head union (P1): {crit(p1)}")
    print(f"  S1 CV+ beats split conformal (descriptive):            {crit(s1)}")
    print("  Note: S1 also differs in training data (90% vs 75%); it is not a clean "
          "comparison of calibration methods alone.")


if __name__ == "__main__":
    main()
