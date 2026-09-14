use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StudentT};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Continuous, ContinuousCDF, StudentsT};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    board: Board,
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
            board,
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
                let q = if range > 0.0 {
                    (e.stats.mean - lo) / range
                } else {
                    0.5
                };
                let bonus = (2.0 * log_n / e.stats.n as f64).sqrt();
                let score = q + bonus;
                if score > best {
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
                if x > best {
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
        sims_per_second: if secs > 0.0 { total_sims / secs } else { 0.0 },
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
    let budgets = parse_usizes(&args.budgets)?;
    let policies: Vec<Policy> = args
        .policies
        .split(',')
        .map(|s| Policy::parse(s.trim()))
        .collect::<Result<_, _>>()?;

    if !(args.reward_scale.is_finite() && args.reward_scale > 0.0) {
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
    let aggregates: Arc<Mutex<HashMap<(usize, String), (usize, u128, u128)>>> =
        Arc::new(Mutex::new(HashMap::new()));

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

        if n % log_every == 0 || n == total_requested {
            let elapsed = run_start.elapsed().as_secs_f64();
            let completed_this_run = n.saturating_sub(already_done);
            let rate = if elapsed > 0.0 { completed_this_run as f64 / elapsed } else { 0.0 };
            let remaining = total_requested.saturating_sub(n);
            let eta_s = if rate > 0.0 { remaining as f64 / rate } else { f64::INFINITY };

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
            if summary.len() > 260 {
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
}
