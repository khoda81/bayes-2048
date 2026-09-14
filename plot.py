#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import io

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
args = ap.parse_args()

csv = Path(args.csv)
out = Path(args.out) if args.out else csv.parent
out.mkdir(parents=True, exist_ok=True)

# The benchmark appends progressively. If we catch the file in the middle of
# its final row, ignore that incomplete line and plot every completed game.
try:
    df = pd.read_csv(csv)
except Exception:
    data = csv.read_bytes()
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        raise
    df = pd.read_csv(io.BytesIO(data[:last_newline + 1]))

agg = (df.groupby(["simulations","policy"])
       .agg(mean_score=("final_score","mean"),
            median_score=("final_score","median"),
            mean_tile=("max_tile","mean"),
            mean_wall_ms=("wall_ms","mean"),
            mean_sps=("sims_per_second","mean"),
            games=("final_score","size"))
       .reset_index())
agg.to_csv(out/"summary.csv", index=False)

plt.figure(figsize=(8,5))
for p, d in agg.groupby("policy"):
    plt.plot(d.simulations, d.mean_score, marker="o", label=p)
plt.xscale("log", base=2)
plt.xlabel("MCTS simulations per move")
plt.ylabel("Mean final 2048 score")
plt.title("2048 strength vs search budget")
plt.grid(alpha=.25)
plt.legend()
plt.tight_layout()
plt.savefig(out/"score_vs_budget.png", dpi=180)

# Log-log Pareto plot. Both coordinates are minimized, so better is closer
# to the origin. Hollow points are dominated; filled points lie on the
# staircase frontier.
per_game = df.assign(ms_per_move=df.wall_ms / df.moves)
pareto = (per_game.groupby(["simulations", "policy"])
          .agg(mean_score=("final_score", "mean"),
               mean_ms_per_move=("ms_per_move", "mean"))
          .reset_index())
best_score = pareto.mean_score.max()
pareto["inverse_relative_score"] = best_score / pareto.mean_score

def dominated(row):
    others = pareto.drop(index=row.name)
    no_worse = ((others.mean_ms_per_move <= row.mean_ms_per_move) &
                (others.inverse_relative_score <= row.inverse_relative_score))
    strictly_better = ((others.mean_ms_per_move < row.mean_ms_per_move) |
                       (others.inverse_relative_score < row.inverse_relative_score))
    return bool((no_worse & strictly_better).any())

pareto["on_frontier"] = ~pareto.apply(dominated, axis=1)
pareto.to_csv(out/"pareto.csv", index=False)
front = pareto[pareto.on_frontier].sort_values("mean_ms_per_move")

plt.figure(figsize=(9,5.8))
for _, row in pareto.iterrows():
    kwargs = dict(marker="o", markersize=8, linestyle="None")
    if not row.on_frontier:
        kwargs.update(markerfacecolor="none")
    plt.plot(row.mean_ms_per_move, row.inverse_relative_score, **kwargs)
    plt.annotate(f"{row.policy} {int(row.simulations)}",
                 (row.mean_ms_per_move, row.inverse_relative_score),
                 textcoords="offset points", xytext=(4,4), fontsize=8)

if len(front):
    x = front.mean_ms_per_move.to_numpy()
    y = front.inverse_relative_score.to_numpy()
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        xs.extend([x[i], x[i]])
        ys.extend([y[i-1], y[i]])
    plt.plot(xs, ys, linewidth=1.5)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mean search time per move (ms) — lower is better")
plt.ylabel("Best mean score / configuration mean score — lower is better")
plt.title("Pareto frontier: performance vs compute")
plt.grid(alpha=.25)
plt.tight_layout()
plt.savefig(out/"pareto_score_vs_compute.png", dpi=180)

# Paired score-vs-UCT win rate by shared seed.
if "uct" in set(df.policy):
    rows=[]
    for B, db in df.groupby("simulations"):
        u = db[db.policy=="uct"][["seed","final_score"]].rename(columns={"final_score":"uct"})
        for p in sorted(set(db.policy)-{"uct"}):
            x = db[db.policy==p][["seed","final_score"]].rename(columns={"final_score":"other"})
            m = u.merge(x,on="seed")
            if len(m):
                score = ((m.other > m.uct).sum() + .5*(m.other == m.uct).sum()) / len(m)
                rows.append({"simulations":B,"policy":p,"paired_score_vs_uct":score,"pairs":len(m)})
    h = pd.DataFrame(rows)
    h.to_csv(out/"paired_vs_uct.csv",index=False)
    plt.figure(figsize=(8,5))
    for p,d in h.groupby("policy"):
        plt.plot(d.simulations,d.paired_score_vs_uct,marker="o",label=p)
    plt.axhline(.5,ls="--",lw=1)
    plt.xscale("log",base=2)
    plt.xlabel("MCTS simulations per move")
    plt.ylabel("Paired score vs UCT")
    plt.title("How often each allocator beats UCT on matched seeds")
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/"paired_vs_uct.png",dpi=180)

print(agg.to_string(index=False))
print(f"\nwrote charts to {out}")
