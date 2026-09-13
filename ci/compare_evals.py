#!/usr/bin/env python3
"""Compare a fresh eval run against the baseline committed in results/.

    python ci/compare_evals.py pick_eval  /tmp/eval/pick_eval.json --mode exact
    python ci/compare_evals.py task_eval  /tmp/eval/task_eval.json --mode outcome

--mode exact    every trial field identical except host wall time (and the
                .md report byte-identical) -- for the platform the baselines
                were made on.
--mode outcome  the same trials pass and fail, and no trial is missing;
                sim_time and grasp detail may drift across platforms.

Exit 1 on any difference, with the differing trials listed. A difference is a
question for the team (re-baseline, or a regression?), never a thing to
paper over by editing the baseline here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "Hand_and_Wrists" / "results"
KEYS = {"pick_eval": ("object", "seed"), "pick_eval_vision": ("object", "seed"),
        "task_eval": ("mode", "item", "seed"), "task_eval_truth": ("mode", "item", "seed")}
IGNORED = {"wall_time"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", choices=sorted(KEYS), help="which committed results/<name>.json")
    p.add_argument("new", type=Path, help="the fresh run's .json")
    p.add_argument("--mode", choices=["exact", "outcome"], default="outcome")
    args = p.parse_args(argv)

    key = KEYS[args.baseline]
    base_json = RESULTS / f"{args.baseline}.json"
    base = {tuple(r.get(k) for k in key): r for r in json.loads(base_json.read_text())}
    new = {tuple(r.get(k) for k in key): r for r in json.loads(args.new.read_text())}

    diffs = []
    for k in sorted(set(base) | set(new), key=str):
        b, n = base.get(k), new.get(k)
        if b is None or n is None:
            diffs.append(f"{k}: missing from {'baseline' if b is None else 'new run'}")
        elif args.mode == "outcome":
            if bool(b.get("success")) != bool(n.get("success")):
                diffs.append(f"{k}: success {b.get('success')} -> {n.get('success')} ({n.get('failure')})")
        else:
            changed = {f: (b.get(f), n.get(f)) for f in set(b) | set(n)
                       if f not in IGNORED and b.get(f) != n.get(f)}
            if changed:
                diffs.append(f"{k}: {changed}")

    if args.mode == "exact":
        new_md = args.new.with_suffix(".md")
        if new_md.exists() and new_md.read_bytes().replace(b"\r\n", b"\n") != \
                base_json.with_suffix(".md").read_bytes().replace(b"\r\n", b"\n"):
            diffs.append("report .md differs")

    passes = lambda rows: sum(bool(r.get("success")) for r in rows.values())  # noqa: E731
    print(f"{args.baseline} ({args.mode}): {len(new)}/{len(base)} trials, {passes(new)} pass "
          f"(baseline {passes(base)}), {len(diffs)} difference(s)")
    for d in diffs[:20]:
        print("  ", d)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
