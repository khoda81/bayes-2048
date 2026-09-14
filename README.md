# bayes-2048

A self-contained 2048 benchmark for comparing compute-allocation rules inside MCTS.

Policies:

- `uct`: scale-free UCT. Returns are empirically normalized at each node, then the canonical UCB1 bonus is used.
- `thompson`: Student-t Thompson sampling under the Jeffreys normal model `p(mu,sigma) ∝ 1/sigma`.
- `exact-voc`: analytic **one-step expected reduction in Bayes simple regret** under the Student-t posterior predictive; no MC sample-count parameter.
- `mc-voc-N`: Monte-Carlo estimate of **one-step expected reduction in Bayes simple regret** using `N` posterior-predictive samples.

The game engine uses standard 2048 rules: tile-2 probability 0.9, tile-4 probability 0.1, and the true merge score as reward.

## Run

```bash
cargo test
cargo run --release -- \
  --games 20 \
  --budgets 8,16,32,64,128 \
  --policies uct,thompson,mc-voc-8,mc-voc-32,mc-voc-128 \
  --out results
```

Then:

```bash
uv run --with pandas --with matplotlib plot.py results/games.csv
```

For a larger run:

```bash
cargo run --release -- \
  --games 100 \
  --budgets 8,16,32,64,128,256,512 \
  --policies uct,thompson,mc-voc-8,mc-voc-32,mc-voc-128 \
  --out results-big
```

## Reward-scale invariance check

The allocator can observe an affine transform of every rollout return while the actual game and reported score stay unchanged:

```bash
cargo run --release -- --games 20 --budgets 32,128 \
  --policies mc-voc-32 --reward-scale 1000 --reward-shift 37 \
  --out transformed
```

With identical seeds, MC-VOC should be invariant up to floating-point / tie-breaking effects.

## Important modeling choice

At each decision edge, rollout returns are modeled as Normal with unknown mean and scale under the Jeffreys prior

`p(mu, sigma) ∝ 1/sigma`.

After three observations, this gives Student-t posterior and posterior-predictive distributions without a reward-scale hyperparameter. The first three samples per legal action are forced because the finite-mean posterior needs enough data.

`mc-voc-N` scores action `i` by

`E_y[max_j E[mu_j | D, y_i]] - max_j E[mu_j | D]`.

This is the myopic value of computation for terminal simple regret. The MC noise is intentionally retained after clamping negative estimates to zero, matching the X-O experiment where estimator noise appeared to rescue some zero-VOC dead zones.


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
