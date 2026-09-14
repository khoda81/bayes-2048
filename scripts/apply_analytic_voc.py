#!/usr/bin/env python3
"""Apply the analytic one-step VOC experiment to the benchmark source.

This is intentionally deterministic and idempotent so the self-hosted benchmark
runner can materialize the experiment, test it, and commit the real Rust edits
back to the feature branch before running the sweep.
"""
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"could not find patch anchor for {label}")
    return text.replace(old, new, 1)


root = Path(__file__).resolve().parents[1]

# Cargo dependency for stable Student-t pdf/cdf/survival evaluation.
cargo_path = root / "Cargo.toml"
cargo = cargo_path.read_text()
if 'statrs = ' not in cargo:
    cargo = cargo.rstrip() + '\nstatrs = "0.18"\n'
cargo_path.write_text(cargo)

main_path = root / "src/main.rs"
s = main_path.read_text()
s = replace_once(
    s,
    "use serde::{Deserialize, Serialize};\n",
    "use serde::{Deserialize, Serialize};\nuse statrs::distribution::{Continuous, ContinuousCDF, StudentsT};\n",
    "statrs import",
)
s = replace_once(
    s,
    "enum Policy {\n    Uct,\n    Thompson,\n    McVoc(usize),\n}",
    "enum Policy {\n    Uct,\n    Thompson,\n    ExactVoc,\n    McVoc(usize),\n}",
    "policy enum",
)
s = replace_once(
    s,
    '        if s == "thompson" {\n            return Ok(Self::Thompson);\n        }\n        if let Some(rest) = s.strip_prefix("mc-voc-") {',
    '        if s == "thompson" {\n            return Ok(Self::Thompson);\n        }\n        if s == "exact-voc" {\n            return Ok(Self::ExactVoc);\n        }\n        if let Some(rest) = s.strip_prefix("mc-voc-") {',
    "policy parser",
)
s = replace_once(
    s,
    'Err(format!("unknown policy {s}; use uct, thompson, mc-voc-N"))',
    'Err(format!("unknown policy {s}; use uct, thompson, exact-voc, mc-voc-N"))',
    "parser error",
)
s = replace_once(
    s,
    '            Self::Uct => "uct".into(),\n            Self::Thompson => "thompson".into(),\n            Self::McVoc(n) => format!("mc-voc-{n}"),',
    '            Self::Uct => "uct".into(),\n            Self::Thompson => "thompson".into(),\n            Self::ExactVoc => "exact-voc".into(),\n            Self::McVoc(n) => format!("mc-voc-{n}"),',
    "policy name",
)

exact_fn = r'''
fn exact_voc(stats: Stats, other_best: f64) -> f64 {
    // Under Jeffreys' p(mu,sigma) ∝ 1/sigma, after n observations the next
    // posterior mean is mu' = mu + tau*T_nu with nu=n-1 and
    // tau=sd/sqrt(n(n+1)). For c=best competing posterior mean,
    // VOC = E[(mu'-c)_+] - (mu-c)_+.
    if stats.n < 3 {
        return 0.0;
    }
    let sd = stats.sd();
    if !(sd > 0.0) || !sd.is_finite() {
        return 0.0;
    }

    let n = stats.n as f64;
    let nu = n - 1.0;
    let tau = sd / (n * (n + 1.0)).sqrt();
    if !(tau > 0.0) || !tau.is_finite() {
        return 0.0;
    }

    let z = (other_best - stats.mean) / tau;
    let t = StudentsT::new(0.0, 1.0, nu).expect("valid Student-t parameters");
    let pdf = t.pdf(z);
    let leading = ((nu + z * z) / (nu - 1.0)) * pdf;

    // Equivalent stable forms of
    // tau * [((nu+z^2)/(nu-1))*f(z) - z*(1-F(z))] - (mu-c)_+.
    let voc = if z >= 0.0 {
        tau * (leading - z * t.sf(z))
    } else {
        tau * (leading + z * t.cdf(z))
    };
    voc.max(0.0)
}
'''
if "fn exact_voc(" not in s:
    s = s.replace("fn least_sampled(edges: &[Edge]) -> usize {", exact_fn + "\nfn least_sampled(edges: &[Edge]) -> usize {", 1)

exact_arm = r'''        Policy::ExactVoc => {
            if node.edges.iter().any(|e| e.stats.n < 3) {
                return least_sampled(&node.edges);
            }

            let means: Vec<f64> = node.edges.iter().map(|e| e.stats.mean).collect();
            let mut scores = vec![0.0; node.edges.len()];
            for (i, e) in node.edges.iter().enumerate() {
                let other_best = means
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &x)| (j != i).then_some(x))
                    .fold(f64::NEG_INFINITY, f64::max);
                scores[i] = exact_voc(e.stats, other_best);
            }

            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let eps = 1e-12 * (1.0 + max_score.abs());
            let ties: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| ((x - max_score).abs() <= eps).then_some(i))
                .collect();
            ties[rng.random_range(0..ties.len())]
        }
'''
if "Policy::ExactVoc =>" not in s:
    s = s.replace("        Policy::McVoc(mc) => {", exact_arm + "        Policy::McVoc(mc) => {", 1)

s = s.replace(
    "/// Comma-separated policies: uct, thompson, mc-voc-N.",
    "/// Comma-separated policies: uct, thompson, exact-voc, mc-voc-N.",
)

tests = r'''
    #[test]
    fn exact_voc_is_positive_at_decision_boundary() {
        let mut s = Stats::default();
        for x in [1.0, 2.0, 3.0, 4.0] {
            s.observe(x);
        }
        assert!(exact_voc(s, s.mean) > 0.0);
    }

    #[test]
    fn exact_voc_is_affine_equivariant() {
        let xs = [100.0, 300.0, 200.0, 500.0, 250.0];
        let mut a = Stats::default();
        let mut b = Stats::default();
        for x in xs {
            a.observe(x);
            b.observe(1000.0 * x + 37.0);
        }
        let va = exact_voc(a, 280.0);
        let vb = exact_voc(b, 1000.0 * 280.0 + 37.0);
        let rel = (vb - 1000.0 * va).abs() / (1.0 + vb.abs());
        assert!(rel < 1e-10, "va={va} vb={vb} rel={rel}");
    }
'''
if "exact_voc_is_affine_equivariant" not in s:
    s = s.replace("    #[test]\n    fn affine_stats_preserve_order() {", tests + "\n    #[test]\n    fn affine_stats_preserve_order() {", 1)

main_path.write_text(s)

# Document the policy and experiment.
readme_path = root / "README.md"
r = readme_path.read_text()
if "- `exact-voc`:" not in r:
    r = r.replace(
        "- `mc-voc-N`: Monte-Carlo estimate of **one-step expected reduction in Bayes simple regret** using `N` posterior-predictive samples.\n",
        "- `exact-voc`: analytic **one-step expected reduction in Bayes simple regret** under the Student-t posterior predictive; no MC sample-count parameter.\n"
        "- `mc-voc-N`: Monte-Carlo estimate of **one-step expected reduction in Bayes simple regret** using `N` posterior-predictive samples.\n",
    )
if "## Analytic VOC experiment" not in r:
    r += r'''

## Analytic VOC experiment

For edge statistics `(n, mu, s)`, one additional observation makes the updated
posterior mean

```text
mu' = mu + tau * T_nu
nu  = n - 1
tau = s / sqrt(n(n+1))
```

so for best competing posterior mean `c`,

```text
VOC = E[(mu' - c)+] - (mu - c)+.
```

The Student-t positive-part expectation has a closed form, so `exact-voc`
requires no inner Monte-Carlo estimator and remains positive-affine reward
invariant.

```bash
cargo run --release -- \
  --games 100 \
  --budgets 64,128,256,512 \
  --policies exact-voc \
  --out results-exact
```
'''
readme_path.write_text(r)

# Replace the old raw wall-time plot with the requested log-log Pareto plot.
plot_path = root / "plot.py"
p = plot_path.read_text()
old = '''plt.figure(figsize=(8,5))
for p, d in agg.groupby("policy"):
    plt.plot(d.mean_wall_ms, d.mean_score, marker="o", label=p)
plt.xlabel("Mean wall time per game (ms)")
plt.ylabel("Mean final 2048 score")
plt.title("2048 strength vs actual compute")
plt.grid(alpha=.25)
plt.legend()
plt.tight_layout()
plt.savefig(out/"score_vs_walltime.png", dpi=180)
'''
new = '''# Log-log Pareto plot. Both coordinates are minimized, so better is closer
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
'''
if "pareto_score_vs_compute.png" not in p:
    if old not in p:
        raise RuntimeError("could not find old wall-time plotting block")
    p = p.replace(old, new, 1)
plot_path.write_text(p)

print("analytic VOC patch applied")
