#!/usr/bin/env python3
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

if len(sys.argv) != 4:
    raise SystemExit("usage: bench_arena.py BASELINE_BIN ARENA_BIN OUT_DIR")

baseline_bin = Path(sys.argv[1]).resolve()
arena_bin = Path(sys.argv[2]).resolve()
out_dir = Path(sys.argv[3]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

budgets = [128, 512]
policies = ["uct", "exact-voc"]
seeds = [1, 2, 3, 4]

env = os.environ.copy()
env["RAYON_NUM_THREADS"] = "1"

def run_one(binary: Path, impl: str, budget: int, policy: str, seed: int):
    run_dir = out_dir / "runs" / f"{impl}-{policy}-b{budget}-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--games", "1",
        "--budgets", str(budget),
        "--policies", policy,
        "--seed", str(seed),
        "--out", str(run_dir),
        "--log-every", "1",
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    with (run_dir / "games.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    row["_stderr"] = proc.stderr
    return row

# Warm both binaries before timed comparisons. Internal wall_ms excludes process
# startup, but this also stabilizes page cache / CPU state a little.
for impl, binary in [("baseline", baseline_bin), ("arena", arena_bin)]:
    run_one(binary, f"warm-{impl}", 64, "uct", 999)

pairs = []
all_match = True
for budget in budgets:
    for policy_i, policy in enumerate(policies):
        for seed in seeds:
            # Alternate AB/BA order to reduce drift/thermal bias.
            order = ["baseline", "arena"] if (seed + policy_i + budget) % 2 == 0 else ["arena", "baseline"]
            got = {}
            for impl in order:
                binary = baseline_bin if impl == "baseline" else arena_bin
                got[impl] = run_one(binary, impl, budget, policy, seed)

            b = got["baseline"]
            a = got["arena"]
            semantic_fields = ["final_score", "max_tile", "moves"]
            match = all(b[k] == a[k] for k in semantic_fields)
            all_match &= match
            b_ms = float(b["wall_ms"])
            a_ms = float(a["wall_ms"])
            pairs.append({
                "budget": budget,
                "policy": policy,
                "seed": seed,
                "baseline_wall_ms": b_ms,
                "arena_wall_ms": a_ms,
                "speedup": b_ms / a_ms,
                "baseline_sims_per_second": float(b["sims_per_second"]),
                "arena_sims_per_second": float(a["sims_per_second"]),
                "final_score": int(a["final_score"]),
                "max_tile": int(a["max_tile"]),
                "moves": int(a["moves"]),
                "semantic_match": match,
            })
            print(
                f"B={budget:>3} {policy:>9} seed={seed}: "
                f"{b_ms/1000:7.3f}s -> {a_ms/1000:7.3f}s  "
                f"speedup={b_ms/a_ms:6.3f}x  match={match}",
                flush=True,
            )

with (out_dir / "pairs.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
    writer.writeheader()
    writer.writerows(pairs)

summary = []
for budget in budgets:
    for policy in policies:
        xs = [r for r in pairs if r["budget"] == budget and r["policy"] == policy]
        speedups = [r["speedup"] for r in xs]
        b_total = sum(r["baseline_wall_ms"] for r in xs)
        a_total = sum(r["arena_wall_ms"] for r in xs)
        summary.append({
            "budget": budget,
            "policy": policy,
            "n": len(xs),
            "aggregate_speedup": b_total / a_total,
            "mean_pair_speedup": statistics.fmean(speedups),
            "median_pair_speedup": statistics.median(speedups),
            "baseline_total_s": b_total / 1000.0,
            "arena_total_s": a_total / 1000.0,
            "all_semantic_match": all(r["semantic_match"] for r in xs),
        })

with (out_dir / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

(out_dir / "summary.json").write_text(json.dumps({
    "all_semantic_match": all_match,
    "rows": summary,
}, indent=2))

print("\nSUMMARY")
for r in summary:
    print(
        f"B={r['budget']:>3} {r['policy']:>9}: aggregate {r['aggregate_speedup']:.3f}x, "
        f"median {r['median_pair_speedup']:.3f}x, semantic_match={r['all_semantic_match']}"
    )

if not all_match:
    raise SystemExit("arena implementation changed deterministic game outcomes")
