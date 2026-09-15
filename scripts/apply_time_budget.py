#!/usr/bin/env python3
from pathlib import Path

main = Path("src/main.rs")
s = main.read_text()

s = s.replace("use std::time::Instant;", "use std::time::{Duration, Instant};")

# Extend trace records with the actual search budget/result per move.
s = s.replace(
    "    reward: u64,\n    spawn_index: usize,",
    "    reward: u64,\n    search_ms: f64,\n    search_simulations: u32,\n    spawn_index: usize,",
)
s = s.replace(
    "    simulations: usize,\n    final_score: u64,",
    "    simulations: Option<usize>,\n    time_ms: Option<f64>,\n    final_score: u64,",
    1,
)

choose_start = s.index("fn choose_move_with_trace")
choose_end = s.index("\nfn play_trace(", choose_start)
new_choose = r'''fn choose_move_with_trace<R: Rng + ?Sized>(
    board: Board,
    simulations: Option<usize>,
    time_ms: Option<f64>,
    cfg: &SearchCfg,
    rng: &mut R,
) -> Option<(Dir, Vec<ActionTrace>, u32, f64)> {
    let mut root = Node::new(board);
    if root.terminal() {
        return None;
    }

    let start = Instant::now();
    let mut sims_done = 0u32;

    if let Some(ms) = time_ms {
        let duration = Duration::from_secs_f64(ms / 1000.0);
        // Exact VOC / Thompson need three observations per arm before their
        // Jeffreys-normal posterior has the moments we use. UCT needs one.
        // For normal interactive budgets (e.g. 300 ms) this warm-up is tiny;
        // for absurdly small budgets we prefer a valid posterior to choosing
        // from uninitialized means.
        let min_sims = match cfg.policy {
            Policy::Uct => root.edges.len(),
            _ => 3 * root.edges.len(),
        } as u32;

        while sims_done < min_sims || start.elapsed() < duration {
            simulate(&mut root, cfg, rng);
            sims_done += 1;
        }
    } else {
        let n = simulations.expect("simulation budget when time_ms is absent");
        for _ in 0..n {
            simulate(&mut root, cfg, rng);
            sims_done += 1;
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let means: Vec<f64> = root.edges.iter().map(|e| e.stats.mean).collect();
    let actions = root
        .edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let other_best = means
                .iter()
                .enumerate()
                .filter_map(|(j, &x)| (j != i).then_some(x))
                .fold(f64::NEG_INFINITY, f64::max);
            let voc = match cfg.policy {
                Policy::ExactVoc => exact_voc(e.stats, other_best),
                _ => 0.0,
            };
            ActionTrace {
                dir: e.dir.to_string(),
                samples: e.stats.n,
                mean: e.stats.mean,
                voc,
            }
        })
        .collect();

    // The action itself is still the Bayes action: maximize posterior expected
    // return. The wall-clock limit only decides how much evidence we gather.
    let dir = root
        .edges
        .iter()
        .max_by(|a, b| a.stats.mean.total_cmp(&b.stats.mean))?
        .dir;
    Some((dir, actions, sims_done, elapsed_ms))
}
'''
s = s[:choose_start] + new_choose + s[choose_end:]

play_start = s.index("fn play_trace(")
play_end = s.index("\n#[derive(Debug, Clone, Serialize, Deserialize)]\nstruct GameRow", play_start)
new_play = r'''fn print_live_trace(
    step: usize,
    score: u64,
    before: Board,
    after: Board,
    dir: Dir,
    reward: u64,
    search_ms: f64,
    search_simulations: u32,
    actions: &[ActionTrace],
) {
    print!("\x1b[2J\x1b[H");
    println!(
        "exact-VOC live | move {} | score {} | chose {} (+{}) | thought {:.1} ms | {} sims\n",
        step + 1,
        score,
        dir,
        reward,
        search_ms,
        search_simulations
    );

    println!("before:");
    let vals = board_values(before);
    for r in 0..4 {
        for c in 0..4 {
            let v = vals[4 * r + c];
            if v == 0 {
                print!("{:>6}", ".");
            } else {
                print!("{:>6}", v);
            }
        }
        println!();
    }

    println!("\nroot posterior:");
    for a in actions {
        let mark = if a.dir == dir.to_string() { "  <" } else { "" };
        println!(
            "  {:>5}  n={:>5}  E[R]={:>10.1}  VOC={:>10.4}{}",
            a.dir, a.samples, a.mean, a.voc, mark
        );
    }

    println!("\nafter spawn:");
    let vals = board_values(after);
    for r in 0..4 {
        for c in 0..4 {
            let v = vals[4 * r + c];
            if v == 0 {
                print!("{:>6}", ".");
            } else {
                print!("{:>6}", v);
            }
        }
        println!();
    }
}

fn play_trace(
    seed: u64,
    policy: Policy,
    simulations: Option<usize>,
    time_ms: Option<f64>,
    rollout_cap: usize,
    transform: RewardTransform,
    live: bool,
) -> GameTrace {
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
        let search_seed = seed
            ^ (move_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ 0xD1B5_4A32_D192_ED03;
        let mut search_rng = SmallRng::seed_from_u64(search_seed);
        let Some((dir, actions, search_simulations, search_ms)) = choose_move_with_trace(
            board,
            simulations,
            time_ms,
            &cfg,
            &mut search_rng,
        ) else {
            break;
        };
        let Some((moved, reward)) = board.moved(dir) else {
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

        if live {
            print_live_trace(
                move_index,
                score,
                before,
                spawned,
                dir,
                reward,
                search_ms,
                search_simulations,
                &actions,
            );
        }

        moves.push(TraceStep {
            move_index,
            score_before,
            score_after: score,
            board_before: board_values(before),
            chosen: dir.to_string(),
            reward,
            search_ms,
            search_simulations,
            spawn_index,
            spawn_tile,
            board_after: board_values(spawned),
            actions,
        });
        board = spawned;
    }

    GameTrace {
        seed,
        policy: policy.name(),
        simulations,
        time_ms,
        final_score: score,
        max_tile: board.max_tile(),
        moves,
    }
}
'''
s = s[:play_start] + new_play + s[play_end:]

# CLI: --play gives a live single game; --time-ms bounds each move by wall time.
args_anchor = '''    /// Write one traced game as JSON instead of running the benchmark.\n    /// Requires exactly one policy and one budget.\n    #[arg(long)]\n    trace_json: Option<PathBuf>,\n'''
args_replacement = '''    /// Play one game live in the terminal instead of running the benchmark.\n    #[arg(long, default_value_t = false)]\n    play: bool,\n\n    /// Wall-clock thinking budget per move in milliseconds for --play/--trace-json.\n    /// When set, --budgets is ignored for the single-game trace/play path.\n    #[arg(long)]\n    time_ms: Option<f64>,\n\n    /// Optionally write the single played/traced game as JSON.\n    #[arg(long)]\n    trace_json: Option<PathBuf>,\n'''
if "    play: bool," not in s:
    s = s.replace(args_anchor, args_replacement)

trace_start = s.index("    if let Some(trace_path) = &args.trace_json {")
trace_end = s.index("\n    let path = args.out.join(\"games.csv\");", trace_start)
new_main_trace = r'''    if args.time_ms.is_some() && !args.play && args.trace_json.is_none() {
        return Err("--time-ms currently requires --play or --trace-json".into());
    }

    if args.play || args.trace_json.is_some() {
        if policies.len() != 1 {
            return Err("--play/--trace-json requires exactly one --policies entry".into());
        }
        if let Some(ms) = args.time_ms {
            if !(ms.is_finite() && ms > 0.0) {
                return Err("--time-ms must be finite and > 0".into());
            }
        } else if budgets.len() != 1 {
            return Err(
                "without --time-ms, --play/--trace-json requires exactly one --budgets entry".into(),
            );
        }

        let simulations = args.time_ms.is_none().then_some(budgets[0]);
        let trace = play_trace(
            args.seed,
            policies[0],
            simulations,
            args.time_ms,
            args.rollout_cap,
            transform,
            args.play,
        );

        if let Some(trace_path) = &args.trace_json {
            if let Some(parent) = trace_path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
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
'''
s = s[:trace_start] + new_main_trace + s[trace_end:]

main.write_text(s)
print("patched src/main.rs with --play and --time-ms")
