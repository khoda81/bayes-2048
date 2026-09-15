use clap::Parser;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    Clear, ClearType, DisableLineWrap, EnableLineWrap, EnterAlternateScreen, LeaveAlternateScreen,
    disable_raw_mode, enable_raw_mode,
};
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
use std::io::{self, IsTerminal, Write as IoWrite};
use std::num::NonZeroUsize;
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
            Self::Up => "↑",
            Self::Down => "↓",
            Self::Left => "←",
            Self::Right => "→",
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
                return Err(format!("MC sample count must be positive in {s}"));
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
    // With 3 <= n, mu | data is Student-t(df=n-1, loc=xbar, scale=sd/sqrt(n)).
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

/// Maximum one-step exact VOC among root actions, in the search allocator's
/// reward units. A single legal action has zero decision value: there is
/// nothing another simulation can change about which move will be played.
fn max_exact_voc(node: &Node) -> f64 {
    if node.edges.len() <= 1 {
        return 0.0;
    }
    if node.edges.iter().any(|edge| edge.stats.n < 3) {
        return f64::INFINITY;
    }

    let means: Vec<f64> = node.edges.iter().map(|edge| edge.stats.mean).collect();
    node.edges
        .iter()
        .enumerate()
        .map(|(i, edge)| {
            let other_best = means
                .iter()
                .enumerate()
                .filter_map(|(j, &mean)| (i != j).then_some(mean))
                .fold(f64::NEG_INFINITY, f64::max);
            exact_voc(edge.stats, other_best)
        })
        .fold(0.0, f64::max)
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
            // The Jeffreys normal posterior has a finite posterior mean once 3 <= n.
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
    search_budget_ms: Option<f64>,
    search_budget_simulations: Option<usize>,
    search_unlimited: bool,
    search_stopped_by_voc: bool,
    search_max_voc: f64,
    search_voc_cost: Option<f64>,
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
    voc_cost: Option<f64>,
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SearchConstraints {
    simulation_limit: Option<NonZeroUsize>,
    time_limit: Option<Duration>,
    min_voc: Option<f64>,
}

impl SearchConstraints {
    fn has_stopping_condition(self) -> bool {
        self.simulation_limit.is_some() || self.time_limit.is_some() || self.min_voc.is_some()
    }

    fn simulation_exhausted(self, simulations: u32) -> bool {
        self.simulation_limit
            .is_some_and(|limit| limit.get() <= simulations as usize)
    }

    fn time_exhausted(self, elapsed: Duration) -> bool {
        self.time_limit.is_some_and(|limit| limit <= elapsed)
    }

    fn hard_limits_disabled(self) -> bool {
        self.simulation_limit.is_none() && self.time_limit.is_none()
    }

    fn describe(self) -> String {
        let mut parts = Vec::new();
        if let Some(limit) = self.simulation_limit {
            parts.push(format!("{} sims", limit.get()));
        }
        if let Some(limit) = self.time_limit {
            parts.push(format!("{:.0} ms", limit.as_secs_f64() * 1000.0));
        }
        if let Some(min_voc) = self.min_voc {
            parts.push(format!("VOC >= {min_voc:.4}"));
        }
        if parts.is_empty() {
            "no automatic stop".into()
        } else {
            parts.join(" + ")
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LiveSearch {
    move_index: usize,
    score: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveCommand {
    Continue,
    ActBest,
    Force(Dir),
    Quit,
}

struct LiveTerminal {
    alternate_screen: bool,
}

impl LiveTerminal {
    fn enter() -> io::Result<Self> {
        let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
        if !interactive {
            return Ok(Self {
                alternate_screen: false,
            });
        }

        enable_raw_mode()?;
        let mut stdout = io::stdout().lock();
        if let Err(error) = execute!(stdout, EnterAlternateScreen, Hide, DisableLineWrap) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        Ok(Self {
            alternate_screen: true,
        })
    }

    fn interactive(&self) -> bool {
        self.alternate_screen
    }

    fn draw(&self, frame: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        if self.alternate_screen {
            execute!(stdout, MoveTo(0, 0), Clear(ClearType::All))?;
            // Raw mode disables newline translation, so emit CRLF explicitly.
            stdout.write_all(frame.replace('\n', "\r\n").as_bytes())?;
        } else {
            stdout.write_all(b"\x1b[2J\x1b[H")?;
            stdout.write_all(frame.as_bytes())?;
        }
        stdout.flush()
    }
}

impl Drop for LiveTerminal {
    fn drop(&mut self) {
        if !self.alternate_screen {
            return;
        }
        let _ = disable_raw_mode();
        let mut stdout = io::stdout().lock();
        let _ = execute!(stdout, LeaveAlternateScreen, Show, EnableLineWrap);
        let _ = stdout.flush();
    }
}

struct LiveSession {
    terminal: LiveTerminal,
    constraints: SearchConstraints,
    saved_simulation_limit: Option<NonZeroUsize>,
    saved_time_limit: Option<Duration>,
    frame_interval: Duration,
    notice: String,
    revision: u64,
}

impl LiveSession {
    const TIME_STEP: Duration = Duration::from_millis(50);
    const MIN_TIME: Duration = Duration::from_millis(10);
    const SIMULATION_STEP: usize = 128;
    const FRAME_STEP: Duration = Duration::from_millis(10);
    const MIN_FRAME: Duration = Duration::from_millis(10);

    fn new(constraints: SearchConstraints, frame_interval: Duration) -> io::Result<Self> {
        Ok(Self {
            terminal: LiveTerminal::enter()?,
            constraints,
            saved_simulation_limit: constraints.simulation_limit,
            saved_time_limit: constraints.time_limit,
            frame_interval,
            notice: "searching".into(),
            revision: 0,
        })
    }

    fn interactive(&self) -> bool {
        self.terminal.interactive()
    }

    fn set_notice(&mut self, notice: impl Into<String>) {
        self.notice = notice.into();
        self.revision = self.revision.wrapping_add(1);
    }

    fn toggle_resource_limits(&mut self) {
        if self.constraints.hard_limits_disabled() {
            self.constraints.simulation_limit = self
                .saved_simulation_limit
                .or_else(|| NonZeroUsize::new(512));
            self.constraints.time_limit = self.saved_time_limit;
            self.set_notice(format!("restored {}", self.constraints.describe()));
        } else {
            self.saved_simulation_limit = self.constraints.simulation_limit;
            self.saved_time_limit = self.constraints.time_limit;
            self.constraints.simulation_limit = None;
            self.constraints.time_limit = None;
            self.set_notice("resource caps disabled; other stop constraints remain active");
        }
    }

    fn adjust_resource_limit(&mut self, increase: bool) {
        if let Some(duration) = self.constraints.time_limit {
            let adjusted = if increase {
                duration.saturating_add(Self::TIME_STEP)
            } else {
                duration.saturating_sub(Self::TIME_STEP).max(Self::MIN_TIME)
            };
            self.constraints.time_limit = Some(adjusted);
            self.saved_time_limit = Some(adjusted);
        } else {
            let current = self
                .constraints
                .simulation_limit
                .map_or(512, NonZeroUsize::get);
            let adjusted = if increase {
                current.saturating_add(Self::SIMULATION_STEP)
            } else {
                current.saturating_sub(Self::SIMULATION_STEP).max(1)
            };
            self.constraints.simulation_limit = NonZeroUsize::new(adjusted);
            self.saved_simulation_limit = self.constraints.simulation_limit;
        }
        self.set_notice(format!("constraints {}", self.constraints.describe()));
    }

    fn adjust_frame_interval(&mut self, increase: bool) {
        self.frame_interval = if increase {
            self.frame_interval.saturating_add(Self::FRAME_STEP)
        } else {
            self.frame_interval
                .saturating_sub(Self::FRAME_STEP)
                .max(Self::MIN_FRAME)
        };
        self.set_notice(format!(
            "refresh interval {:.0} ms",
            self.frame_interval.as_secs_f64() * 1000.0
        ));
    }

    fn handle_key(&mut self, key: KeyEvent) -> LiveCommand {
        if matches!(key.kind, KeyEventKind::Release) {
            return LiveCommand::Continue;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return LiveCommand::Quit;
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => LiveCommand::Quit,
            KeyCode::Char(' ') | KeyCode::Enter => {
                self.set_notice("acting now with current best");
                LiveCommand::ActBest
            }
            KeyCode::Up => LiveCommand::Force(Dir::Up),
            KeyCode::Down => LiveCommand::Force(Dir::Down),
            KeyCode::Left => LiveCommand::Force(Dir::Left),
            KeyCode::Right => LiveCommand::Force(Dir::Right),
            KeyCode::Char('i') => {
                self.toggle_resource_limits();
                LiveCommand::Continue
            }
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::PageUp => {
                self.adjust_resource_limit(true);
                LiveCommand::Continue
            }
            KeyCode::Char('-') | KeyCode::PageDown => {
                self.adjust_resource_limit(false);
                LiveCommand::Continue
            }
            KeyCode::Char('[') => {
                self.adjust_frame_interval(false);
                LiveCommand::Continue
            }
            KeyCode::Char(']') => {
                self.adjust_frame_interval(true);
                LiveCommand::Continue
            }
            _ => LiveCommand::Continue,
        }
    }

    fn poll_command(&mut self) -> io::Result<LiveCommand> {
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()? {
                let command = self.handle_key(key);
                if command != LiveCommand::Continue {
                    return Ok(command);
                }
            }
        }
        Ok(LiveCommand::Continue)
    }
}

struct SearchOutcome {
    dir: Dir,
    root: RootSnapshot,
    constraints: SearchConstraints,
    simulations: u32,
    compute_elapsed: Duration,
    wall_elapsed: Duration,
    excluded_overhead: Duration,
    stopped_by_voc: bool,
    max_voc: f64,
}

struct LiveSearchRuntime<'a> {
    session: &'a mut LiveSession,
    context: LiveSearch,
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
    session: &LiveSession,
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

    let hard_fraction = [
        session
            .constraints
            .simulation_limit
            .map(|limit| snapshot.total_samples as f64 / limit.get() as f64),
        session
            .constraints
            .time_limit
            .map(|limit| compute_elapsed.as_secs_f64() / limit.as_secs_f64()),
    ]
    .into_iter()
    .flatten()
    .fold(None::<f64>, |acc, x| Some(acc.map_or(x, |a| a.max(x))));

    if let Some(fraction) = hard_fraction {
        writeln!(
            output,
            "think {:>8.1} ms | {:>7} sims [{}] {:>3.0}% | {:>8.0} sims/s",
            compute_seconds * 1000.0,
            snapshot.total_samples,
            progress_bar(fraction),
            100.0 * fraction.clamp(0.0, 1.0),
            sims_per_second,
        )
        .expect("writing to String cannot fail");
    } else {
        writeln!(
            output,
            "think {:>8.1} ms | {:>7} sims [   no resource cap    ] | {:>8.0} sims/s",
            compute_seconds * 1000.0,
            snapshot.total_samples,
            sims_per_second,
        )
        .expect("writing to String cannot fail");
    }

    let sim_limit = session
        .constraints
        .simulation_limit
        .map_or_else(|| "off".into(), |n| n.get().to_string());
    let time_limit = session.constraints.time_limit.map_or_else(
        || "off".into(),
        |d| format!("{:.1} ms", d.as_secs_f64() * 1000.0),
    );
    let voc_limit = session
        .constraints
        .min_voc
        .map_or_else(|| "off".into(), |v| format!("{v:.4}"));
    writeln!(
        output,
        "limits sims {sim_limit} | time {time_limit} | min VOC {voc_limit}"
    )
    .expect("writing to String cannot fail");

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
    writeln!(output, "status {}", session.notice).expect("writing to String cannot fail");

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
        let mark = if snapshot.best_dir == dir { '*' } else { ' ' };
        write!(output, " {mark} {dir:<2}|").expect("writing to String cannot fail");
        if let Some(action) = snapshot.action(dir) {
            writeln!(
                output,
                " {:>6} | {:>5.1}% | {:>10.1} | {:>8.1} | {:>9.1} | {:>9.4} | {:>5.1}%",
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
                " {:>6} | {:>6} | {:>10} | {:>8} | {:>9} | {:>9} | {:>6}",
                "-", "-", "illegal", "-", "-", "-", "-",
            )
            .expect("writing to String cannot fail");
        }
    }
    writeln!(
        output,
        "\n* posterior-mean best | switch = P(one sample flips this arm vs its competitor)"
    )
    .expect("writing to String cannot fail");
    writeln!(
        output,
        "keys: space/enter act | arrows force | +/- resource cap | i toggle resource caps | [/] refresh | q quit"
    )
    .expect("writing to String cannot fail");

    session.terminal.draw(&output)
}

fn search_move_with_diagnostics<R: Rng + ?Sized>(
    board: Board,
    constraints: SearchConstraints,
    cfg: &SearchCfg,
    rng: &mut R,
    mut live: Option<LiveSearchRuntime<'_>>,
) -> io::Result<Option<SearchOutcome>> {
    const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(5);

    let mut root = Node::new(board);
    if root.terminal() {
        return Ok(None);
    }

    if let Some(runtime) = live.as_mut() {
        runtime.session.set_notice("searching");
    }

    let wall_start = Instant::now();
    let mut excluded_overhead = Duration::ZERO;
    let mut sims_done = 0u32;
    let mut active_constraints = live
        .as_ref()
        .map_or(constraints, |runtime| runtime.session.constraints);
    let mut next_frame = live.as_ref().map(|runtime| runtime.session.frame_interval);
    let mut next_input_poll = live
        .as_ref()
        .filter(|runtime| runtime.session.interactive())
        .map(|_| Duration::ZERO);
    let mut last_rendered_state = None;
    let mut selected_dir = None;
    let mut stopped_by_voc = false;

    loop {
        let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);

        if let Some(input_due) = next_input_poll
            && input_due <= compute_elapsed
        {
            let input_start = Instant::now();
            let runtime = live.as_mut().expect("input polling requires live mode");
            let old_frame_interval = runtime.session.frame_interval;
            let command = runtime.session.poll_command()?;
            active_constraints = runtime.session.constraints;
            if old_frame_interval != runtime.session.frame_interval {
                next_frame = Some(compute_elapsed + runtime.session.frame_interval);
            }
            excluded_overhead += input_start.elapsed();

            let mut following = input_due + INPUT_POLL_INTERVAL;
            while following <= compute_elapsed {
                following += INPUT_POLL_INTERVAL;
            }
            next_input_poll = Some(following);

            match command {
                LiveCommand::Continue => {}
                LiveCommand::ActBest => break,
                LiveCommand::Force(dir) => {
                    if root.edges.iter().any(|edge| edge.dir == dir) {
                        runtime.session.set_notice(format!("forcing {dir}"));
                        selected_dir = Some(dir);
                        break;
                    }
                    runtime
                        .session
                        .set_notice(format!("ignored: {dir} is illegal"));
                }
                LiveCommand::Quit => return Ok(None),
            }
        }

        // Every enabled field is an independent stopping constraint. Hard
        // resource limits are always enforced. The VOC constraint becomes
        // evaluable once each competing root action has the n=3 bootstrap
        // required by the Jeffreys-Normal model.
        let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
        if active_constraints.simulation_exhausted(sims_done)
            || active_constraints.time_exhausted(compute_elapsed)
        {
            break;
        }

        if let Some(min_voc) = active_constraints.min_voc {
            let bootstrap_complete =
                root.edges.len() <= 1 || root.edges.iter().all(|edge| 3 <= edge.stats.n);
            if bootstrap_complete {
                let max_voc = max_exact_voc(&root);
                let effective_min_voc = min_voc * cfg.reward_transform.scale;
                if max_voc <= effective_min_voc {
                    stopped_by_voc = true;
                    if let Some(runtime) = live.as_mut() {
                        runtime.session.set_notice(format!(
                            "VOC stop: max {:.4} <= min {:.4}",
                            max_voc / cfg.reward_transform.scale,
                            min_voc,
                        ));
                    }
                    break;
                }
            }
        }

        simulate(&mut root, cfg, rng);
        sims_done += 1;

        let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
        if let Some(frame_due) = next_frame
            && frame_due <= compute_elapsed
        {
            let render_start = Instant::now();
            let snapshot = snapshot_root(&root);
            let runtime = live.as_ref().expect("frame rendering requires live mode");
            render_search_frame(
                board,
                &snapshot,
                cfg.policy,
                runtime.context,
                runtime.session,
                compute_elapsed,
            )?;
            excluded_overhead += render_start.elapsed();
            last_rendered_state = Some((sims_done, runtime.session.revision));

            // Schedule frames by search-compute time. If one simulation spans
            // several intervals, skip missed frames instead of emitting a burst.
            let mut following = frame_due + runtime.session.frame_interval;
            while following <= compute_elapsed {
                following += runtime.session.frame_interval;
            }
            next_frame = Some(following);
        }
    }

    let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
    let final_frame_start = Instant::now();
    let snapshot = snapshot_root(&root);
    if let Some(runtime) = live.as_ref()
        && last_rendered_state != Some((sims_done, runtime.session.revision))
    {
        render_search_frame(
            board,
            &snapshot,
            cfg.policy,
            runtime.context,
            runtime.session,
            compute_elapsed,
        )?;
        excluded_overhead += final_frame_start.elapsed();
    }

    Ok(Some(SearchOutcome {
        dir: selected_dir.unwrap_or(snapshot.best_dir),
        root: snapshot,
        constraints: active_constraints,
        simulations: sims_done,
        compute_elapsed,
        wall_elapsed: wall_start.elapsed(),
        excluded_overhead,
        stopped_by_voc,
        max_voc: max_exact_voc(&root) / cfg.reward_transform.scale,
    }))
}

fn play_trace(
    seed: u64,
    policy: Policy,
    constraints: SearchConstraints,
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
    let mut live_session = live_frame_interval
        .map(|frame_interval| LiveSession::new(constraints, frame_interval))
        .transpose()?;

    loop {
        let move_index = moves.len();
        let search_seed =
            seed ^ (move_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        let mut search_rng = SmallRng::seed_from_u64(search_seed);
        let live = live_session.as_mut().map(|session| LiveSearchRuntime {
            session,
            context: LiveSearch { move_index, score },
        });
        let Some(search) =
            search_move_with_diagnostics(board, constraints, &cfg, &mut search_rng, live)?
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
        let search_budget_ms = search
            .constraints
            .time_limit
            .map(|duration| duration.as_secs_f64() * 1000.0);
        let search_budget_simulations = search.constraints.simulation_limit.map(NonZeroUsize::get);
        let search_unlimited = search.constraints.hard_limits_disabled();

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
            search_budget_ms,
            search_budget_simulations,
            search_unlimited,
            search_stopped_by_voc: search.stopped_by_voc,
            search_max_voc: search.max_voc,
            search_voc_cost: search.constraints.min_voc,
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

    let simulations = constraints.simulation_limit.map(NonZeroUsize::get);
    let time_ms = constraints
        .time_limit
        .map(|duration| duration.as_secs_f64() * 1000.0);
    Ok(GameTrace {
        seed,
        policy: policy.name(),
        simulations,
        time_ms,
        voc_cost: constraints.min_voc,
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

    /// Comma-separated fixed simulation budgets for benchmark mode. In
    /// single-game mode, one value is accepted as a legacy simulation limit.
    #[arg(long)]
    budgets: Option<String>,

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

    /// Maximum simulations per move in --play/--trace-json mode.
    #[arg(long)]
    simulation_limit: Option<NonZeroUsize>,

    /// Maximum search-compute time per move in milliseconds in
    /// --play/--trace-json mode.
    #[arg(long)]
    time_ms: Option<f64>,

    /// Minimum root one-step VOC required to buy another simulation,
    /// expressed in actual 2048 score units. This is an independent stopping
    /// constraint alongside the simulation and time limits.
    #[arg(long)]
    voc_cost: Option<f64>,

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
    const DEFAULT_BUDGETS: &str = "8,16,32,64,128";
    let budgets = if single_game {
        Vec::new()
    } else {
        parse_usizes(args.budgets.as_deref().unwrap_or(DEFAULT_BUDGETS))?
    };
    let policies: Vec<Policy> = args
        .policies
        .split(',')
        .map(|s| Policy::parse(s.trim()))
        .collect::<Result<_, _>>()?;

    if !args.reward_scale.is_finite() || args.reward_scale <= 0.0 {
        return Err("--reward-scale must be finite and positive".into());
    }
    if !args.reward_shift.is_finite() {
        return Err("--reward-shift must be finite".into());
    }

    fs::create_dir_all(&args.out)?;
    let transform = RewardTransform {
        scale: args.reward_scale,
        shift: args.reward_shift,
    };

    if args.simulation_limit.is_some() && !single_game {
        return Err("--simulation-limit requires --play or --trace-json".into());
    }
    if args.time_ms.is_some() && !single_game {
        return Err("--time-ms requires --play or --trace-json".into());
    }
    if let Some(cost) = args.voc_cost {
        if !cost.is_finite() || cost < 0.0 {
            return Err("--voc-cost must be finite and nonnegative".into());
        }
        if !single_game {
            return Err("--voc-cost currently requires --play or --trace-json".into());
        }
        if policies.len() != 1 || !matches!(policies[0], Policy::ExactVoc) {
            return Err("--voc-cost currently requires --policies exact-voc".into());
        }
    }

    if single_game {
        if policies.len() != 1 {
            return Err("--play/--trace-json requires exactly one --policies entry".into());
        }
        if !args.frame_ms.is_finite() || args.frame_ms <= 0.0 {
            return Err("--frame-ms must be finite and positive".into());
        }

        let legacy_simulation_limit = if let Some(raw) = args.budgets.as_deref() {
            if args.simulation_limit.is_some() {
                return Err(
                    "use either --simulation-limit or --budgets in single-game mode, not both"
                        .into(),
                );
            }
            let parsed = parse_usizes(raw)?;
            if parsed.len() != 1 {
                return Err(
                    "single-game --budgets accepts exactly one value; prefer --simulation-limit"
                        .into(),
                );
            }
            NonZeroUsize::new(parsed[0])
                .ok_or("simulation limit must be positive")?
                .into()
        } else {
            None
        };

        let time_limit = if let Some(ms) = args.time_ms {
            if !ms.is_finite() || ms <= 0.0 {
                return Err("--time-ms must be finite and positive".into());
            }
            Some(Duration::from_secs_f64(ms / 1000.0))
        } else {
            None
        };

        let constraints = SearchConstraints {
            simulation_limit: args.simulation_limit.or(legacy_simulation_limit),
            time_limit,
            min_voc: args.voc_cost,
        };
        if !constraints.has_stopping_condition() && !args.play {
            return Err(
                "non-interactive trace search needs at least one stopping constraint".into(),
            );
        }
        if matches!(constraints.min_voc, Some(0.0))
            && constraints.hard_limits_disabled()
            && !args.play
        {
            return Err("--voc-cost 0 without a resource limit may not terminate".into());
        }

        let trace = play_trace(
            args.seed,
            policies[0],
            constraints,
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

        let constraints = SearchConstraints {
            simulation_limit: trace.simulations.and_then(NonZeroUsize::new),
            time_limit: trace.time_ms.map(|ms| Duration::from_secs_f64(ms / 1000.0)),
            min_voc: trace.voc_cost,
        };
        eprintln!(
            "done: seed={} policy={} constraints={} score={} tile={} moves={}",
            trace.seed,
            trace.policy,
            constraints.describe(),
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
    fn direction_symbols_have_fixed_width_labels() {
        let labels: Vec<String> = DIRS.into_iter().map(|dir| format!("  {dir:<2}|")).collect();

        assert_eq!(labels, ["  ↑ |", "  ↓ |", "  ← |", "  → |"]);
        assert!(labels.iter().all(|label| label.chars().count() == 5));
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
            SearchConstraints {
                simulation_limit: NonZeroUsize::new(64),
                ..SearchConstraints::default()
            },
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

    #[test]
    fn live_resource_constraints_toggle_and_adjust() {
        let initial = SearchConstraints {
            simulation_limit: NonZeroUsize::new(512),
            time_limit: Some(Duration::from_millis(300)),
            min_voc: Some(1.0),
        };
        let mut session = LiveSession {
            terminal: LiveTerminal {
                alternate_screen: false,
            },
            constraints: initial,
            saved_simulation_limit: initial.simulation_limit,
            saved_time_limit: initial.time_limit,
            frame_interval: Duration::from_millis(50),
            notice: String::new(),
            revision: 0,
        };

        assert_eq!(
            session.handle_key(KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE)),
            LiveCommand::Continue
        );
        assert!(session.constraints.hard_limits_disabled());
        assert_eq!(session.constraints.min_voc, Some(1.0));

        session.handle_key(KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE));
        assert_eq!(session.constraints, initial);
        session.handle_key(KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE));
        assert_eq!(
            session.constraints.time_limit,
            Some(Duration::from_millis(350))
        );
        session.handle_key(KeyEvent::new(KeyCode::Char('-'), KeyModifiers::NONE));
        assert_eq!(
            session.constraints.time_limit,
            Some(Duration::from_millis(300))
        );

        session.handle_key(KeyEvent::new(KeyCode::Char('['), KeyModifiers::NONE));
        assert_eq!(session.frame_interval, Duration::from_millis(40));
        session.handle_key(KeyEvent::new(KeyCode::Char(']'), KeyModifiers::NONE));
        assert_eq!(session.frame_interval, Duration::from_millis(50));
    }

    #[test]
    fn live_action_and_quit_keys_map_to_commands() {
        let constraints = SearchConstraints {
            simulation_limit: NonZeroUsize::new(512),
            ..SearchConstraints::default()
        };
        let mut session = LiveSession {
            terminal: LiveTerminal {
                alternate_screen: false,
            },
            constraints,
            saved_simulation_limit: constraints.simulation_limit,
            saved_time_limit: constraints.time_limit,
            frame_interval: Duration::from_millis(50),
            notice: String::new(),
            revision: 0,
        };

        assert_eq!(
            session.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            LiveCommand::ActBest
        );
        assert_eq!(
            session.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
            LiveCommand::Force(Dir::Left)
        );
        assert_eq!(
            session.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            LiveCommand::Quit
        );
    }
}
