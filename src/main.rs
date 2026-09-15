use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StudentT};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, StudentsT};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::fs::{self, OpenOptions};
use std::io::{self, Write as IoWrite};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DIRS: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Dir::Up => "up",
            Dir::Down => "down",
            Dir::Left => "left",
            Dir::Right => "right",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Board([u8; 16]); // exponent: 0=empty, 1=2, 2=4, ...

impl Board {
    fn empty() -> Self {
        Self([0; 16])
    }

    fn empty_count(self) -> usize {
        self.0.iter().filter(|&&x| x == 0).count()
    }

    fn max_tile(self) -> u64 {
        let e = self.0.iter().copied().max().unwrap_or(0);
        if e == 0 { 0 } else { 1u64 << e }
    }

    fn legal_moves(self) -> Vec<(Dir, Board, u64)> {
        DIRS.into_iter()
            .filter_map(|d| self.moved(d).map(|(b, score)| (d, b, score)))
            .collect()
    }

    fn moved(self, dir: Dir) -> Option<(Board, u64)> {
        let mut out = self;
        let mut reward = 0u64;
        let mut changed = false;

        for line_idx in 0..4 {
            let idxs = match dir {
                Dir::Left => [
                    line_idx * 4,
                    line_idx * 4 + 1,
                    line_idx * 4 + 2,
                    line_idx * 4 + 3,
                ],
                Dir::Right => [
                    line_idx * 4 + 3,
                    line_idx * 4 + 2,
                    line_idx * 4 + 1,
                    line_idx * 4,
                ],
                Dir::Up => [line_idx, line_idx + 4, line_idx + 8, line_idx + 12],
                Dir::Down => [line_idx + 12, line_idx + 8, line_idx + 4, line_idx],
            };
            let input = [
                self.0[idxs[0]],
                self.0[idxs[1]],
                self.0[idxs[2]],
                self.0[idxs[3]],
            ];
            let (line, r) = merge_line(input);
            reward += r;
            if line != input {
                changed = true;
            }
            for k in 0..4 {
                out.0[idxs[k]] = line[k];
            }
        }

        changed.then_some((out, reward))
    }

    fn spawn<R: Rng + ?Sized>(mut self, rng: &mut R) -> Self {
        let empties: Vec<usize> = self
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| (x == 0).then_some(i))
            .collect();
        if empties.is_empty() {
            return self;
        }
        let idx = empties[rng.random_range(0..empties.len())];
        // Standard 2048: 90% tile 2, 10% tile 4.
        self.0[idx] = if rng.random::<f64>() < 0.9 { 1 } else { 2 };
        self
    }

    fn initial<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::empty().spawn(rng).spawn(rng)
    }
}

fn merge_line(input: [u8; 4]) -> ([u8; 4], u64) {
    let mut nz = [0u8; 4];
    let mut n = 0usize;
    for x in input {
        if x != 0 {
            nz[n] = x;
            n += 1;
        }
    }

    let mut out = [0u8; 4];
    let mut oi = 0usize;
    let mut i = 0usize;
    let mut reward = 0u64;
    while i < n {
        if i + 1 < n && nz[i] == nz[i + 1] {
            let e = nz[i] + 1;
            out[oi] = e;
            reward += 1u64 << e;
            i += 2;
        } else {
            out[oi] = nz[i];
            i += 1;
        }
        oi += 1;
    }
    (out, reward)
}

#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    n: u32,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn observe(&mut self, x: f64) {
        if self.n == 0 {
            self.n = 1;
            self.mean = x;
            self.m2 = 0.0;
            self.min = x;
            self.max = x;
            return;
        }
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }

    fn sd(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            (self.m2 / (self.n - 1) as f64).max(0.0).sqrt()
        }
    }
}

struct Edge {
    dir: Dir,
    moved: Board,
    immediate_reward: u64,
    stats: Stats,
    children: HashMap<Board, Box<Node>>,
}

impl Edge {
    fn new(dir: Dir, moved: Board, immediate_reward: u64) -> Self {
        Self {
            dir,
            moved,
            immediate_reward,
            stats: Stats::default(),
            children: HashMap::new(),
        }
    }
}

struct Node {
    // Retained in the tree representation to keep the benchmark node layout
    // unchanged; live root rendering receives its board explicitly.
    _board: Board,
    visits: u32,
    edges: Vec<Edge>,
}

impl Node {
    fn new(board: Board) -> Self {
        let edges = board
            .legal_moves()
            .into_iter()
            .map(|(d, b, r)| Edge::new(d, b, r))
            .collect();
        Self {
            _board: board,
            visits: 0,
            edges,
        }
    }

    fn terminal(&self) -> bool {
        self.edges.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
enum Policy {
    Uct,
    Thompson,
    ExactVoc,
    McVoc(usize),
}

impl Policy {
    fn parse(s: &str) -> Result<Self, String> {
        if s == "uct" {
            return Ok(Self::Uct);
        }
        if s == "thompson" {
            return Ok(Self::Thompson);
        }
        if s == "exact-voc" {
            return Ok(Self::ExactVoc);
        }
        if let Some(rest) = s.strip_prefix("mc-voc-") {
            let n: usize = rest
                .parse()
                .map_err(|_| format!("bad MC sample count in {s}"))?;
            if n == 0 {
                return Err(format!("MC sample count must be >0 in {s}"));
            }
            return Ok(Self::McVoc(n));
        }
        Err(format!(
            "unknown policy {s}; use uct, thompson, exact-voc, mc-voc-N"
        ))
    }

    fn name(self) -> String {
        match self {
            Self::Uct => "uct".into(),
            Self::Thompson => "thompson".into(),
            Self::ExactVoc => "exact-voc".into(),
            Self::McVoc(n) => format!("mc-voc-{n}"),
        }
    }
}

#[derive(Clone, Copy)]
struct RewardTransform {
    scale: f64,
    shift: f64,
}

impl RewardTransform {
    fn apply(self, raw: f64) -> f64 {
        self.scale * raw + self.shift
    }
}

struct SearchCfg {
    policy: Policy,
    rollout_cap: usize,
    reward_transform: RewardTransform,
}

fn sample_student_t<R: Rng + ?Sized>(rng: &mut R, df: f64) -> f64 {
    StudentT::new(df)
        .expect("positive degrees of freedom")
        .sample(rng)
}

fn posterior_mean_sample<R: Rng + ?Sized>(s: Stats, rng: &mut R) -> f64 {
    // Jeffreys normal model p(mu,sigma) ∝ 1/sigma.
    // With n>=3, mu | data is Student-t(df=n-1, loc=xbar, scale=sd/sqrt(n)).
    if s.n < 3 || s.sd() == 0.0 {
        return s.mean;
    }
    let t = sample_student_t(rng, (s.n - 1) as f64);
    s.mean + s.sd() / (s.n as f64).sqrt() * t
}

fn predictive_sample<R: Rng + ?Sized>(s: Stats, rng: &mut R) -> f64 {
    // y_next | data is Student-t(df=n-1, loc=xbar, scale=sd*sqrt(1+1/n)).
    if s.n < 3 || s.sd() == 0.0 {
        return s.mean;
    }
    let t = sample_student_t(rng, (s.n - 1) as f64);
    s.mean + s.sd() * (1.0 + 1.0 / s.n as f64).sqrt() * t
}

struct OneStepMeanDistribution {
    tau: f64,
    z: f64,
    student_t: StudentsT,
    currently_best: bool,
}

impl OneStepMeanDistribution {
    fn voc(&self) -> f64 {
        let nu = self.student_t.freedom();
        let pdf = self.student_t.pdf(self.z);
        let leading = ((nu + self.z * self.z) / (nu - 1.0)) * pdf;

        // Equivalent stable forms of
        // tau * [((nu+z^2)/(nu-1))*f(z) - z*(1-F(z))] - (mu-c)_+.
        let voc = if 0.0 <= self.z {
            self.tau * (leading - self.z * self.student_t.sf(self.z))
        } else {
            self.tau * (leading + self.z * self.student_t.cdf(self.z))
        };
        voc.max(0.0)
    }

    fn switch_prob(&self) -> f64 {
        let probability = if self.currently_best {
            self.student_t.cdf(self.z)
        } else {
            self.student_t.sf(self.z)
        };
        probability.clamp(0.0, 1.0)
    }
}

fn one_step_mean_distribution(stats: Stats, other_best: f64) -> Option<OneStepMeanDistribution> {
    // Under Jeffreys' p(mu,sigma) ∝ 1/sigma, after n observations the next
    // posterior mean is mu' = mu + tau*T_nu with nu=n-1 and
    // tau=sd/sqrt(n(n+1)). For c=best competing posterior mean,
    // VOC = E[(mu'-c)_+] - (mu-c)_+.
    if stats.n < 3 {
        return None;
    }
    let sd = stats.sd();
    if sd <= 0.0 || !sd.is_finite() {
        return None;
    }

    let n = stats.n as f64;
    let nu = n - 1.0;
    let tau = sd / (n * (n + 1.0)).sqrt();
    if tau <= 0.0 || !tau.is_finite() {
        return None;
    }

    let z = (other_best - stats.mean) / tau;
    Some(OneStepMeanDistribution {
        tau,
        z,
        student_t: StudentsT::new(0.0, 1.0, nu).expect("valid Student-t parameters"),
        currently_best: other_best <= stats.mean,
    })
}

fn exact_voc(stats: Stats, other_best: f64) -> f64 {
    one_step_mean_distribution(stats, other_best).map_or(0.0, |distribution| distribution.voc())
}

fn one_step_voc_and_switch(stats: Stats, other_best: f64) -> (f64, f64) {
    // A dashboard snapshot needs both values, so reuse its Student-t object.
    // The exact-VOC selection hot path above deliberately computes VOC alone.
    one_step_mean_distribution(stats, other_best).map_or((0.0, 0.0), |distribution| {
        (distribution.voc(), distribution.switch_prob())
    })
}

fn least_sampled(edges: &[Edge]) -> usize {
    let min_n = edges.iter().map(|e| e.stats.n).min().unwrap_or(0);
    edges.iter().position(|e| e.stats.n == min_n).unwrap()
}

fn select_edge<R: Rng + ?Sized>(node: &Node, cfg: &SearchCfg, rng: &mut R) -> usize {
    match cfg.policy {
        Policy::Uct => {
            // UCT only needs every arm visited once.
            if node.edges.iter().any(|e| e.stats.n == 0) {
                return least_sampled(&node.edges);
            }
            // Scale-free UCT: normalize each empirical mean using the observed
            // return range at this node, then use the canonical UCB1 bonus.
            let lo = node
                .edges
                .iter()
                .map(|e| e.stats.min)
                .fold(f64::INFINITY, f64::min);
            let hi = node
                .edges
                .iter()
                .map(|e| e.stats.max)
                .fold(f64::NEG_INFINITY, f64::max);
            let range = hi - lo;
            let log_n = (node.visits.max(1) as f64).ln();

            let mut best_i = 0;
            let mut best = f64::NEG_INFINITY;
            for (i, e) in node.edges.iter().enumerate() {
                let q = if 0.0 < range {
                    (e.stats.mean - lo) / range
                } else {
                    0.5
                };
                let bonus = (2.0 * log_n / e.stats.n as f64).sqrt();
                let score = q + bonus;
                if best < score {
                    best = score;
                    best_i = i;
                }
            }
            best_i
        }
        Policy::Thompson => {
            // The Jeffreys normal posterior has a finite posterior mean once n>=3.
            if node.edges.iter().any(|e| e.stats.n < 3) {
                return least_sampled(&node.edges);
            }
            let mut best_i = 0;
            let mut best = f64::NEG_INFINITY;
            for (i, e) in node.edges.iter().enumerate() {
                let x = posterior_mean_sample(e.stats, rng);
                if best < x {
                    best = x;
                    best_i = i;
                }
            }
            best_i
        }
        Policy::ExactVoc => {
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
        Policy::McVoc(mc) => {
            // Same finite-mean requirement as Thompson.
            if node.edges.iter().any(|e| e.stats.n < 3) {
                return least_sampled(&node.edges);
            }
            let means: Vec<f64> = node.edges.iter().map(|e| e.stats.mean).collect();
            let current_best = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut scores = vec![0.0; node.edges.len()];

            for (i, e) in node.edges.iter().enumerate() {
                let other_best = means
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &x)| (j != i).then_some(x))
                    .fold(f64::NEG_INFINITY, f64::max);

                let mut sum_best_after = 0.0;
                for _ in 0..mc {
                    let y = predictive_sample(e.stats, rng);
                    let updated_mean =
                        (e.stats.n as f64 * e.stats.mean + y) / (e.stats.n as f64 + 1.0);
                    sum_best_after += other_best.max(updated_mean);
                }
                // True one-step VOC is nonnegative; retaining MC noise after the
                // max(0) is intentional and mirrors the X-O experiment.
                scores[i] = (sum_best_after / mc as f64 - current_best).max(0.0);
            }

            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ties: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| ((x - max_score).abs() <= 1e-12).then_some(i))
                .collect();
            ties[rng.random_range(0..ties.len())]
        }
    }
}

fn random_rollout<R: Rng + ?Sized>(mut board: Board, rng: &mut R, cap: usize) -> u64 {
    let mut score = 0u64;
    for _ in 0..cap {
        let legal = board.legal_moves();
        if legal.is_empty() {
            break;
        }
        let (_, moved, reward) = legal[rng.random_range(0..legal.len())];
        score += reward;
        board = moved.spawn(rng);
    }
    score
}

fn simulate<R: Rng + ?Sized>(node: &mut Node, cfg: &SearchCfg, rng: &mut R) -> u64 {
    if node.terminal() {
        return 0;
    }
    node.visits += 1;
    let edge_i = select_edge(node, cfg, rng);

    let (moved, immediate) = {
        let e = &node.edges[edge_i];
        (e.moved, e.immediate_reward)
    };
    let spawned = moved.spawn(rng);

    let downstream = {
        let edge = &mut node.edges[edge_i];
        if let Some(child) = edge.children.get_mut(&spawned) {
            simulate(child, cfg, rng)
        } else {
            let rollout = random_rollout(spawned, rng, cfg.rollout_cap);
            edge.children.insert(spawned, Box::new(Node::new(spawned)));
            rollout
        }
    };

    let total = immediate + downstream;
    let observed = cfg.reward_transform.apply(total as f64);
    node.edges[edge_i].stats.observe(observed);
    total
}

fn choose_move<R: Rng + ?Sized>(
    board: Board,
    simulations: usize,
    cfg: &SearchCfg,
    rng: &mut R,
) -> Option<Dir> {
    let mut root = Node::new(board);
    if root.terminal() {
        return None;
    }
    for _ in 0..simulations {
        simulate(&mut root, cfg, rng);
    }

    // Terminal Bayes action: maximize posterior expected return.
    root.edges
        .iter()
        .max_by(|a, b| a.stats.mean.total_cmp(&b.stats.mean))
        .map(|e| e.dir)
}

#[derive(Debug, Clone, Serialize)]
struct ActionTrace {
    dir: String,
    samples: u32,
    share: f64,
    mean: f64,
    sd: f64,
    gap_to_best: f64,
    voc: f64,
    switch_prob: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceStep {
    move_index: usize,
    score_before: u64,
    score_after: u64,
    board_before: [u64; 16],
    chosen: String,
    reward: u64,
    /// Search time with live diagnostic/rendering work excluded.
    search_ms: f64,
    search_wall_ms: f64,
    search_overhead_ms: f64,
    search_simulations: u32,
    spawn_index: usize,
    spawn_tile: u64,
    board_after: [u64; 16],
    actions: Vec<ActionTrace>,
}

#[derive(Debug, Clone, Serialize)]
struct GameTrace {
    seed: u64,
    policy: String,
    simulations: Option<usize>,
    time_ms: Option<f64>,
    final_score: u64,
    max_tile: u64,
    moves: Vec<TraceStep>,
}

fn board_values(board: Board) -> [u64; 16] {
    std::array::from_fn(|i| {
        let e = board.0[i];
        if e == 0 { 0 } else { 1u64 << e }
    })
}

#[derive(Clone, Copy, Debug)]
struct RootActionSnapshot {
    dir: Dir,
    moved: Board,
    samples: u32,
    share: f64,
    mean: f64,
    sd: f64,
    gap_to_best: f64,
    voc: f64,
    switch_prob: f64,
}

impl RootActionSnapshot {
    fn trace(self) -> ActionTrace {
        ActionTrace {
            dir: self.dir.to_string(),
            samples: self.samples,
            share: self.share,
            mean: self.mean,
            sd: self.sd,
            gap_to_best: self.gap_to_best,
            voc: self.voc,
            switch_prob: self.switch_prob,
        }
    }
}

#[derive(Clone, Debug)]
struct RootSnapshot {
    actions: Vec<RootActionSnapshot>,
    total_samples: u32,
    best_dir: Dir,
    top_mean_gap: f64,
    allocation_entropy: f64,
}

impl RootSnapshot {
    fn action(&self, dir: Dir) -> Option<&RootActionSnapshot> {
        self.actions.iter().find(|action| action.dir == dir)
    }

    fn best_action(&self) -> &RootActionSnapshot {
        self.action(self.best_dir)
            .expect("snapshot best action must be legal")
    }
}

fn snapshot_root(root: &Node) -> RootSnapshot {
    debug_assert!(!root.edges.is_empty());
    let means: Vec<f64> = root.edges.iter().map(|e| e.stats.mean).collect();
    let best_mean = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: u32 = root.edges.iter().map(|e| e.stats.n).sum();

    let actions: Vec<RootActionSnapshot> = root
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let (voc, switch_prob) = if 1 < root.edges.len() {
                let other_best = means
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &x)| (j != i).then_some(x))
                    .fold(f64::NEG_INFINITY, f64::max);
                one_step_voc_and_switch(e.stats, other_best)
            } else {
                // With no competing action there is no decision to switch and
                // information cannot reduce root simple regret.
                (0.0, 0.0)
            };
            RootActionSnapshot {
                dir: e.dir,
                moved: e.moved,
                samples: e.stats.n,
                share: if 0 < total {
                    e.stats.n as f64 / total as f64
                } else {
                    0.0
                },
                mean: e.stats.mean,
                sd: e.stats.sd(),
                gap_to_best: e.stats.mean - best_mean,
                voc,
                switch_prob,
            }
        })
        .collect();

    let best_dir = root
        .edges
        .iter()
        .max_by(|a, b| a.stats.mean.total_cmp(&b.stats.mean))
        .expect("non-terminal root has a legal action")
        .dir;
    let mut sorted_means = means;
    sorted_means.sort_by(|a, b| b.total_cmp(a));
    let top_mean_gap = sorted_means
        .get(1)
        .map_or(0.0, |runner_up| sorted_means[0] - runner_up);
    let allocation_entropy = if 0 < total && 1 < actions.len() {
        let entropy = actions
            .iter()
            .filter_map(|action| (0.0 < action.share).then_some(-action.share * action.share.ln()))
            .sum::<f64>();
        entropy / (actions.len() as f64).ln()
    } else {
        0.0
    };

    RootSnapshot {
        actions,
        total_samples: total,
        best_dir,
        top_mean_gap,
        allocation_entropy,
    }
}

fn tile_symbol(exp: u8) -> char {
    match exp {
        0 => '.',
        1..=9 => (b'0' + exp) as char,
        10..=35 => (b'a' + exp - 10) as char,
        _ => '?',
    }
}

fn write_compact_board(output: &mut String, board: Board) -> fmt::Result {
    for row in board.0.as_chunks::<4>().0 {
        writeln!(
            output,
            "{} {} {} {}",
            tile_symbol(row[0]),
            tile_symbol(row[1]),
            tile_symbol(row[2]),
            tile_symbol(row[3]),
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum SearchBudget {
    Simulations(usize),
    Time(Duration),
}

#[derive(Clone, Copy, Debug)]
struct LiveSearch {
    frame_interval: Duration,
    move_index: usize,
    score: u64,
}

struct SearchOutcome {
    dir: Dir,
    root: RootSnapshot,
    simulations: u32,
    compute_elapsed: Duration,
    wall_elapsed: Duration,
    excluded_overhead: Duration,
}

fn progress_bar(fraction: f64) -> String {
    const WIDTH: usize = 24;
    let filled = (fraction.clamp(0.0, 1.0) * WIDTH as f64).round() as usize;
    format!("{}{}", "=".repeat(filled), "-".repeat(WIDTH - filled))
}

fn render_search_frame(
    board: Board,
    snapshot: &RootSnapshot,
    policy: Policy,
    live: LiveSearch,
    budget: SearchBudget,
    compute_elapsed: Duration,
) -> io::Result<()> {
    let compute_seconds = compute_elapsed.as_secs_f64();
    let sims_per_second = if 0.0 < compute_seconds {
        snapshot.total_samples as f64 / compute_seconds
    } else {
        0.0
    };
    let best = snapshot.best_action();
    let max_voc = snapshot
        .actions
        .iter()
        .map(|action| action.voc)
        .fold(0.0, f64::max);

    let mut output = String::with_capacity(2048);
    writeln!(
        output,
        "{} search | move {} | score {} | max {} | empty {} | legal {}",
        policy.name(),
        live.move_index + 1,
        live.score,
        tile_symbol(board.0.iter().copied().max().unwrap_or(0)),
        board.empty_count(),
        snapshot.actions.len(),
    )
    .expect("writing to String cannot fail");

    match budget {
        SearchBudget::Time(duration) => {
            let fraction = compute_seconds / duration.as_secs_f64();
            writeln!(
                output,
                "think {:>6.1}/{:<6.1} ms [{}] {:>3.0}% | {:>7} sims | {:>8.0} sims/s",
                compute_seconds * 1000.0,
                duration.as_secs_f64() * 1000.0,
                progress_bar(fraction),
                100.0 * fraction.clamp(0.0, 1.0),
                snapshot.total_samples,
                sims_per_second,
            )
            .expect("writing to String cannot fail");
        }
        SearchBudget::Simulations(simulations) => {
            let fraction = snapshot.total_samples as f64 / simulations.max(1) as f64;
            writeln!(
                output,
                "think {:>7}/{:<7} sims [{}] {:>3.0}% | {:>8.1} ms | {:>8.0} sims/s",
                snapshot.total_samples,
                simulations,
                progress_bar(fraction),
                100.0 * fraction.clamp(0.0, 1.0),
                compute_seconds * 1000.0,
                sims_per_second,
            )
            .expect("writing to String cannot fail");
        }
    }

    writeln!(
        output,
        "best {:>5} | top gap {:>9.1} | best alloc {:>5.1}% | entropy {:.3} | max VOC {:.4}",
        snapshot.best_dir,
        snapshot.top_mean_gap,
        100.0 * best.share,
        snapshot.allocation_entropy,
        max_voc,
    )
    .expect("writing to String cannot fail");

    writeln!(output, "\nbefore:").expect("writing to String cannot fail");
    write_compact_board(&mut output, board).expect("writing to String cannot fail");

    writeln!(output, "\nafter (current best, before spawn):")
        .expect("writing to String cannot fail");
    write_compact_board(&mut output, best.moved).expect("writing to String cannot fail");

    writeln!(output, "\nroot posterior:").expect("writing to String cannot fail");
    writeln!(
        output,
        " dir |      n | alloc |       mean |       sd |       gap |       VOC | switch"
    )
    .expect("writing to String cannot fail");
    writeln!(
        output,
        "-----+--------+-------+------------+----------+-----------+-----------+-------"
    )
    .expect("writing to String cannot fail");
    for dir in DIRS {
        if let Some(action) = snapshot.action(dir) {
            let mark = if snapshot.best_dir == dir { '*' } else { ' ' };
            writeln!(
                output,
                "{}{:>4} | {:>6} | {:>5.1}% | {:>10.1} | {:>8.1} | {:>9.1} | {:>9.4} | {:>5.1}%",
                mark,
                action.dir,
                action.samples,
                100.0 * action.share,
                action.mean,
                action.sd,
                action.gap_to_best,
                action.voc,
                100.0 * action.switch_prob,
            )
            .expect("writing to String cannot fail");
        } else {
            writeln!(
                output,
                " {:>4} | {:>6} | {:>6} | {:>10} | {:>8} | {:>9} | {:>9} | {:>6}",
                dir, "-", "-", "illegal", "-", "-", "-", "-",
            )
            .expect("writing to String cannot fail");
        }
    }
    writeln!(
        output,
        "\n* posterior-mean best | switch = P(one sample flips this arm vs its competitor)"
    )
    .expect("writing to String cannot fail");

    let mut stdout = io::stdout().lock();
    stdout.write_all(b"\x1b[2J\x1b[H")?;
    stdout.write_all(output.as_bytes())?;
    stdout.flush()
}

fn search_move_with_diagnostics<R: Rng + ?Sized>(
    board: Board,
    budget: SearchBudget,
    cfg: &SearchCfg,
    rng: &mut R,
    live: Option<LiveSearch>,
) -> io::Result<Option<SearchOutcome>> {
    let mut root = Node::new(board);
    if root.terminal() {
        return Ok(None);
    }

    let wall_start = Instant::now();
    let mut excluded_overhead = Duration::ZERO;
    let mut sims_done = 0u32;
    let mut next_frame = live.map(|display| display.frame_interval);
    let mut last_rendered_simulations = None;
    let min_time_budget_sims = match cfg.policy {
        Policy::Uct => root.edges.len(),
        Policy::Thompson | Policy::ExactVoc | Policy::McVoc(_) => 3 * root.edges.len(),
    } as u32;

    loop {
        let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
        let budget_exhausted = match budget {
            SearchBudget::Simulations(limit) => limit <= sims_done as usize,
            SearchBudget::Time(limit) => {
                limit <= compute_elapsed && min_time_budget_sims <= sims_done
            }
        };
        if budget_exhausted {
            break;
        }

        simulate(&mut root, cfg, rng);
        sims_done += 1;

        let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
        if let (Some(display), Some(frame_due)) = (live, next_frame)
            && frame_due <= compute_elapsed
        {
            let render_start = Instant::now();
            let snapshot = snapshot_root(&root);
            render_search_frame(
                board,
                &snapshot,
                cfg.policy,
                display,
                budget,
                compute_elapsed,
            )?;
            excluded_overhead += render_start.elapsed();
            last_rendered_simulations = Some(sims_done);

            // Schedule frames by search-compute time. If one simulation spans
            // several intervals, skip missed frames instead of emitting a burst.
            let mut following = frame_due + display.frame_interval;
            while following <= compute_elapsed {
                following += display.frame_interval;
            }
            next_frame = Some(following);
        }
    }

    let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
    let final_frame_start = Instant::now();
    let snapshot = snapshot_root(&root);
    if let Some(display) = live
        && last_rendered_simulations != Some(sims_done)
    {
        render_search_frame(
            board,
            &snapshot,
            cfg.policy,
            display,
            budget,
            compute_elapsed,
        )?;
        excluded_overhead += final_frame_start.elapsed();
    }

    Ok(Some(SearchOutcome {
        dir: snapshot.best_dir,
        root: snapshot,
        simulations: sims_done,
        compute_elapsed,
        wall_elapsed: wall_start.elapsed(),
        excluded_overhead,
    }))
}

fn play_trace(
    seed: u64,
    policy: Policy,
    budget: SearchBudget,
    rollout_cap: usize,
    transform: RewardTransform,
    live_frame_interval: Option<Duration>,
) -> io::Result<GameTrace> {
    let mut env_rng = SmallRng::seed_from_u64(seed);
    let mut board = Board::initial(&mut env_rng);
    let mut score = 0u64;
    let mut moves = Vec::new();
    let cfg = SearchCfg {
        policy,
        rollout_cap,
        reward_transform: transform,
    };

    loop {
        let move_index = moves.len();
        let search_seed =
            seed ^ (move_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        let mut search_rng = SmallRng::seed_from_u64(search_seed);
        let live = live_frame_interval.map(|frame_interval| LiveSearch {
            frame_interval,
            move_index,
            score,
        });
        let Some(search) =
            search_move_with_diagnostics(board, budget, &cfg, &mut search_rng, live)?
        else {
            break;
        };
        let Some((moved, reward)) = board.moved(search.dir) else {
            break;
        };

        let before = board;
        let score_before = score;
        score += reward;
        let spawned = moved.spawn(&mut env_rng);
        let spawn_index = moved
            .0
            .iter()
            .zip(spawned.0.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        let spawn_tile = {
            let e = spawned.0[spawn_index];
            if e == 0 { 0 } else { 1u64 << e }
        };

        moves.push(TraceStep {
            move_index,
            score_before,
            score_after: score,
            board_before: board_values(before),
            chosen: search.dir.to_string(),
            reward,
            search_ms: search.compute_elapsed.as_secs_f64() * 1000.0,
            search_wall_ms: search.wall_elapsed.as_secs_f64() * 1000.0,
            search_overhead_ms: search.excluded_overhead.as_secs_f64() * 1000.0,
            search_simulations: search.simulations,
            spawn_index,
            spawn_tile,
            board_after: board_values(spawned),
            actions: search
                .root
                .actions
                .into_iter()
                .map(RootActionSnapshot::trace)
                .collect(),
        });
        board = spawned;
    }

    let (simulations, time_ms) = match budget {
        SearchBudget::Simulations(count) => (Some(count), None),
        SearchBudget::Time(duration) => (None, Some(duration.as_secs_f64() * 1000.0)),
    };
    Ok(GameTrace {
        seed,
        policy: policy.name(),
        simulations,
        time_ms,
        final_score: score,
        max_tile: board.max_tile(),
        moves,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GameRow {
    seed: u64,
    policy: String,
    simulations: usize,
    mc_samples: usize,
    final_score: u64,
    max_tile: u64,
    moves: usize,
    wall_ms: f64,
    sims_per_second: f64,
    reward_scale: f64,
    reward_shift: f64,
}

fn play_game(
    seed: u64,
    policy: Policy,
    simulations: usize,
    rollout_cap: usize,
    transform: RewardTransform,
) -> GameRow {
    let start = Instant::now();
    // Keep environment randomness independent from search randomness.
    // Otherwise a more expensive allocator literally changes the future tile
    // stream just by consuming more RNG draws.
    let mut env_rng = SmallRng::seed_from_u64(seed);
    let mut board = Board::initial(&mut env_rng);
    let mut score = 0u64;
    let mut moves = 0usize;

    let cfg = SearchCfg {
        policy,
        rollout_cap,
        reward_transform: transform,
    };

    loop {
        // Deterministic per-(game, move) search seed. This also makes the
        // affine-reward invariance test meaningful.
        let search_seed =
            seed ^ (moves as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        let mut search_rng = SmallRng::seed_from_u64(search_seed);

        let Some(dir) = choose_move(board, simulations, &cfg, &mut search_rng) else {
            break;
        };
        let Some((moved, reward)) = board.moved(dir) else {
            // Should be impossible because choose_move only exposes legal moves.
            break;
        };
        score += reward;
        moves += 1;
        board = moved.spawn(&mut env_rng);
    }

    let secs = start.elapsed().as_secs_f64();
    let total_sims = simulations as f64 * moves as f64;
    GameRow {
        seed,
        policy: policy.name(),
        simulations,
        mc_samples: match policy {
            Policy::McVoc(n) => n,
            _ => 0,
        },
        final_score: score,
        max_tile: board.max_tile(),
        moves,
        wall_ms: secs * 1000.0,
        sims_per_second: if 0.0 < secs { total_sims / secs } else { 0.0 },
        reward_scale: transform.scale,
        reward_shift: transform.shift,
    }
}

#[derive(Parser, Debug)]
#[command(about = "Scale-free Bayesian compute allocation benchmark for 2048")]
struct Args {
    /// Number of complete games per policy/budget.
    #[arg(long, default_value_t = 20)]
    games: usize,

    /// Comma-separated simulations per move.
    #[arg(long, default_value = "8,16,32,64,128")]
    budgets: String,

    /// Comma-separated policies: uct, thompson, exact-voc, mc-voc-N.
    #[arg(long, default_value = "uct,thompson,mc-voc-8,mc-voc-32,mc-voc-128")]
    policies: String,

    /// Base RNG seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Maximum random-rollout moves before truncation.
    #[arg(long, default_value_t = 1000)]
    rollout_cap: usize,

    /// Affine transform applied ONLY to returns seen by the search allocator.
    /// Actual game score remains unchanged. Useful for scale-invariance tests.
    #[arg(long, default_value_t = 1.0)]
    reward_scale: f64,

    /// Affine shift applied ONLY to returns seen by the search allocator.
    #[arg(long, default_value_t = 0.0)]
    reward_shift: f64,

    /// Play one game live in the terminal instead of running the benchmark.
    #[arg(long, default_value_t = false)]
    play: bool,

    /// Wall-clock thinking budget per move in milliseconds for --play/--trace-json.
    /// When set, --budgets is ignored for the single-game trace/play path.
    #[arg(long)]
    time_ms: Option<f64>,

    /// Live terminal refresh interval in milliseconds while searching.
    #[arg(long, default_value_t = 50.0)]
    frame_ms: f64,

    /// Optionally write the single played/traced game as JSON.
    #[arg(long)]
    trace_json: Option<PathBuf>,

    /// Output directory.
    #[arg(long, default_value = "results")]
    out: PathBuf,

    /// Resume from an existing games.csv, skipping already completed
    /// (seed, policy, simulations) tuples.
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Print a one-line aggregate checkpoint every N completed games.
    #[arg(long, default_value_t = 50)]
    log_every: usize,
}

fn parse_usizes(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|x| {
            x.trim()
                .parse::<usize>()
                .map_err(|_| format!("bad integer: {x}"))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let single_game = args.play || args.trace_json.is_some();
    // A wall-clock single game does not consult the fixed-simulation budget at
    // all. In particular, even an otherwise invalid --budgets value is ignored.
    let budgets = if single_game && args.time_ms.is_some() {
        Vec::new()
    } else {
        parse_usizes(&args.budgets)?
    };
    let policies: Vec<Policy> = args
        .policies
        .split(',')
        .map(|s| Policy::parse(s.trim()))
        .collect::<Result<_, _>>()?;

    if !args.reward_scale.is_finite() || args.reward_scale <= 0.0 {
        return Err("--reward-scale must be finite and > 0".into());
    }
    if !args.reward_shift.is_finite() {
        return Err("--reward-shift must be finite".into());
    }

    fs::create_dir_all(&args.out)?;
    let transform = RewardTransform {
        scale: args.reward_scale,
        shift: args.reward_shift,
    };

    if args.time_ms.is_some() && !args.play && args.trace_json.is_none() {
        return Err("--time-ms currently requires --play or --trace-json".into());
    }

    if single_game {
        if policies.len() != 1 {
            return Err("--play/--trace-json requires exactly one --policies entry".into());
        }
        if !args.frame_ms.is_finite() || args.frame_ms <= 0.0 {
            return Err("--frame-ms must be finite and > 0".into());
        }
        if let Some(ms) = args.time_ms {
            if !ms.is_finite() || ms <= 0.0 {
                return Err("--time-ms must be finite and > 0".into());
            }
        } else if budgets.len() != 1 {
            return Err(
                "without --time-ms, --play/--trace-json requires exactly one --budgets entry"
                    .into(),
            );
        }

        let budget = if let Some(ms) = args.time_ms {
            SearchBudget::Time(Duration::from_secs_f64(ms / 1000.0))
        } else {
            SearchBudget::Simulations(budgets[0])
        };
        let trace = play_trace(
            args.seed,
            policies[0],
            budget,
            args.rollout_cap,
            transform,
            args.play
                .then(|| Duration::from_secs_f64(args.frame_ms / 1000.0)),
        )?;

        if let Some(trace_path) = &args.trace_json {
            if let Some(parent) = trace_path.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent)?;
            }
            fs::write(trace_path, serde_json::to_string_pretty(&trace)?)?;
        }

        let budget = match (trace.time_ms, trace.simulations) {
            (Some(ms), _) => format!("{ms:.1} ms/move"),
            (_, Some(n)) => format!("{n} sims/move"),
            _ => "unknown budget".into(),
        };
        eprintln!(
            "done: seed={} policy={} budget={} score={} tile={} moves={}",
            trace.seed,
            trace.policy,
            budget,
            trace.final_score,
            trace.max_tile,
            trace.moves.len(),
        );
        return Ok(());
    }

    let path = args.out.join("games.csv");

    // Resume support: read completed keys before constructing the pending queue.
    let mut completed_keys: HashSet<(u64, String, usize)> = HashSet::new();
    if args.resume && path.exists() {
        let mut rdr = csv::Reader::from_path(&path)?;
        for row in rdr.deserialize::<GameRow>() {
            let row = row?;
            completed_keys.insert((row.seed, row.policy, row.simulations));
        }
        eprintln!(
            "resume: found {} completed games in {}",
            completed_keys.len(),
            path.display()
        );
    }

    let games = args.games;
    let base_seed = args.seed;
    let all_jobs: Vec<(u64, Policy, usize)> = budgets
        .iter()
        .flat_map(|&budget| {
            policies.iter().flat_map(move |&policy| {
                (0..games).map(move |g| {
                    let seed = base_seed.wrapping_add(g as u64);
                    (seed, policy, budget)
                })
            })
        })
        .collect();

    let mut jobs: Vec<(u64, Policy, usize)> = all_jobs
        .into_iter()
        .filter(|(seed, policy, budget)| !completed_keys.contains(&(*seed, policy.name(), *budget)))
        .collect();

    // Mix cheap/expensive budgets and policies so the measured throughput and
    // ETA become representative early instead of looking great until the slow
    // B=256/512 tail arrives.
    let mut job_rng = SmallRng::seed_from_u64(args.seed ^ 0xA076_1D64_78BD_642F);
    jobs.shuffle(&mut job_rng);

    let total_requested = policies.len() * budgets.len() * args.games;
    let already_done = total_requested - jobs.len();

    // Progressive CSV writer. In resume mode append to the existing file;
    // otherwise replace it and write a fresh header.
    let writer = if args.resume && path.exists() {
        let f = OpenOptions::new().append(true).open(&path)?;
        csv::WriterBuilder::new().has_headers(false).from_writer(f)
    } else {
        csv::Writer::from_path(&path)?
    };
    let writer = Arc::new(Mutex::new(writer));

    let pb = ProgressBar::new(total_requested as u64);
    pb.set_position(already_done as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {pos}/{len} ({percent}%) ETA {eta_precise} | {per_sec} | {msg}",
        )?
        .progress_chars("=>-"),
    );
    pb.set_message("starting…");

    eprintln!(
        "running {} pending games / {} total ({} policies × {} budgets × {} seeds) on {} threads",
        jobs.len(),
        total_requested,
        policies.len(),
        budgets.len(),
        args.games,
        rayon::current_num_threads()
    );

    let done = Arc::new(AtomicUsize::new(already_done));
    let run_start = Instant::now();
    let log_every = args.log_every.max(1);

    // Lightweight progressive aggregates keyed by (budget, policy).
    type AggregateMap = HashMap<(usize, String), (usize, u128, u128)>;
    let aggregates: Arc<Mutex<AggregateMap>> = Arc::new(Mutex::new(HashMap::new()));

    jobs.par_iter().for_each(|&(seed, policy, budget)| {
        let row = play_game(seed, policy, budget, args.rollout_cap, transform);

        {
            let mut w = writer.lock().expect("CSV writer mutex poisoned");
            w.serialize(&row).expect("failed to serialize result row");
            // Flush every row so a crash/interrupt loses at most the currently
            // running games, and plot.py can read partial results immediately.
            w.flush().expect("failed to flush result CSV");
        }

        {
            let mut a = aggregates.lock().expect("aggregate mutex poisoned");
            let e = a.entry((row.simulations, row.policy.clone())).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += row.final_score as u128;
            e.2 += row.max_tile as u128;
        }

        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        pb.inc(1);
        pb.set_message(format!(
            "last: {} B={} score={} tile={} ({:.1}s)",
            row.policy, row.simulations, row.final_score, row.max_tile, row.wall_ms / 1000.0
        ));

        if n.is_multiple_of(log_every) || n == total_requested {
            let elapsed = run_start.elapsed().as_secs_f64();
            let completed_this_run = n.saturating_sub(already_done);
            let rate = if 0.0 < elapsed {
                completed_this_run as f64 / elapsed
            } else {
                0.0
            };
            let remaining = total_requested.saturating_sub(n);
            let eta_s = if 0.0 < rate {
                remaining as f64 / rate
            } else {
                f64::INFINITY
            };

            let a = aggregates.lock().expect("aggregate mutex poisoned");
            let mut parts: Vec<String> = a.iter()
                .map(|((b, p), (count, score_sum, _tile_sum))| {
                    format!("{p}@{b}: {:.0}", *score_sum as f64 / *count as f64)
                })
                .collect();
            parts.sort();
            // Don't spam an enormous summary: show the most recently relevant
            // aggregates only by truncating the joined line.
            let mut summary = parts.join(" | ");
            if 260 < summary.len() {
                summary.truncate(257);
                summary.push_str("...");
            }

            pb.println(format!(
                "checkpoint {n}/{total_requested} ({:.1}%) | elapsed {:.1}m | ETA {:.1}m | {:.2} games/s | {}",
                100.0 * n as f64 / total_requested as f64,
                elapsed / 60.0,
                eta_s / 60.0,
                rate,
                summary
            ));
        }
    });

    pb.finish_with_message(format!("done; wrote {}", path.display()));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_around(mean: f64) -> Stats {
        let mut stats = Stats::default();
        for value in [mean - 1.0, mean, mean + 1.0] {
            stats.observe(value);
        }
        stats
    }

    #[test]
    fn merge_once_per_tile() {
        assert_eq!(merge_line([1, 1, 1, 1]), ([2, 2, 0, 0], 8));
        assert_eq!(merge_line([1, 1, 2, 0]), ([2, 2, 0, 0], 4));
        assert_eq!(merge_line([2, 2, 2, 0]), ([3, 2, 0, 0], 8));
        assert_eq!(merge_line([1, 0, 1, 1]), ([2, 1, 0, 0], 4));
    }

    #[test]
    fn left_move_score() {
        let b = Board([1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 1, 0]);
        let (m, r) = b.moved(Dir::Left).unwrap();
        assert_eq!(r, 4 + 8 + 8 + 4);
        assert_eq!(m.0, [2, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,]);
    }

    #[test]
    fn no_move_is_none() {
        let b = Board([1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7]);
        assert!(b.legal_moves().is_empty());
    }

    #[test]
    fn exact_voc_is_positive_at_decision_boundary() {
        let mut s = Stats::default();
        for x in [1.0, 2.0, 3.0, 4.0] {
            s.observe(x);
        }
        assert!(0.0 < exact_voc(s, s.mean));
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

    #[test]
    fn affine_stats_preserve_order() {
        let xs = [100.0, 300.0, 200.0, 500.0];
        let mut a = Stats::default();
        let mut b = Stats::default();
        for x in xs {
            a.observe(x);
            b.observe(1000.0 * x + 37.0);
        }
        assert!((b.mean - (1000.0 * a.mean + 37.0)).abs() < 1e-8);
        assert!((b.sd() - 1000.0 * a.sd()).abs() < 1e-8);
    }

    #[test]
    fn best_arm_switch_probability_uses_runner_up_with_four_actions() {
        let means = [10.0, 9.0, 0.0, -10.0];
        let edges = DIRS
            .into_iter()
            .zip(means)
            .map(|(dir, mean)| {
                let mut edge = Edge::new(dir, Board::empty(), 0);
                edge.stats = stats_around(mean);
                edge
            })
            .collect();
        let root = Node {
            _board: Board::empty(),
            visits: 12,
            edges,
        };

        let snapshot = snapshot_root(&root);
        let best = snapshot.action(Dir::Up).unwrap();
        let expected = one_step_voc_and_switch(stats_around(10.0), 9.0).1;
        let wrong_distant_competitor = one_step_voc_and_switch(stats_around(10.0), 0.0).1;

        assert_eq!(snapshot.best_dir, Dir::Up);
        assert!((best.switch_prob - expected).abs() < 1e-12);
        assert!(wrong_distant_competitor < best.switch_prob);
        assert!((snapshot.allocation_entropy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn switch_probability_is_symmetric_for_equal_uncertainty() {
        let best = one_step_voc_and_switch(stats_around(10.0), 9.0).1;
        let challenger = one_step_voc_and_switch(stats_around(9.0), 10.0).1;
        assert!((best - challenger).abs() < 1e-12);
    }

    #[test]
    fn compact_tiles_use_exponents() {
        let board = Board([0, 1, 9, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0]);
        let mut output = String::new();
        write_compact_board(&mut output, board).unwrap();
        assert_eq!(output, ". 1 9 a\nb c 2 3\n4 5 6 7\n8 . . .\n");
    }

    #[test]
    fn fixed_budget_diagnostic_search_matches_benchmark_search() {
        let mut env_rng = SmallRng::seed_from_u64(46);
        let board = Board::initial(&mut env_rng);
        let cfg = SearchCfg {
            policy: Policy::ExactVoc,
            rollout_cap: 100,
            reward_transform: RewardTransform {
                scale: 1.0,
                shift: 0.0,
            },
        };
        let mut benchmark_rng = SmallRng::seed_from_u64(1234);
        let mut diagnostic_rng = SmallRng::seed_from_u64(1234);

        let benchmark_choice = choose_move(board, 64, &cfg, &mut benchmark_rng);
        let diagnostic = search_move_with_diagnostics(
            board,
            SearchBudget::Simulations(64),
            &cfg,
            &mut diagnostic_rng,
            None,
        )
        .unwrap()
        .unwrap();

        assert_eq!(benchmark_choice, Some(diagnostic.dir));
        assert_eq!(diagnostic.simulations, 64);
        assert_eq!(diagnostic.root.total_samples, 64);
    }
}
