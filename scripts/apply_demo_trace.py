#!/usr/bin/env python3
from pathlib import Path

cargo = Path("Cargo.toml")
text = cargo.read_text()
if 'serde_json = "1.0"' not in text:
    text = text.rstrip() + '\nserde_json = "1.0"\n'
cargo.write_text(text)

main = Path("src/main.rs")
s = main.read_text()

marker = "#[derive(Debug, Clone, Serialize, Deserialize)]\nstruct GameRow {"
insert = r'''#[derive(Debug, Clone, Serialize)]
struct ActionTrace {
    dir: String,
    samples: u32,
    mean: f64,
    voc: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceStep {
    move_index: usize,
    score_before: u64,
    score_after: u64,
    board_before: [u64; 16],
    chosen: String,
    reward: u64,
    spawn_index: usize,
    spawn_tile: u64,
    board_after: [u64; 16],
    actions: Vec<ActionTrace>,
}

#[derive(Debug, Clone, Serialize)]
struct GameTrace {
    seed: u64,
    policy: String,
    simulations: usize,
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

fn choose_move_with_trace<R: Rng + ?Sized>(
    board: Board,
    simulations: usize,
    cfg: &SearchCfg,
    rng: &mut R,
) -> Option<(Dir, Vec<ActionTrace>)> {
    let mut root = Node::new(board);
    if root.terminal() {
        return None;
    }
    for _ in 0..simulations {
        simulate(&mut root, cfg, rng);
    }

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

    let dir = root
        .edges
        .iter()
        .max_by(|a, b| a.stats.mean.total_cmp(&b.stats.mean))?
        .dir;
    Some((dir, actions))
}

fn play_trace(
    seed: u64,
    policy: Policy,
    simulations: usize,
    rollout_cap: usize,
    transform: RewardTransform,
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
        let Some((dir, actions)) =
            choose_move_with_trace(board, simulations, &cfg, &mut search_rng)
        else {
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

        moves.push(TraceStep {
            move_index,
            score_before,
            score_after: score,
            board_before: board_values(before),
            chosen: dir.to_string(),
            reward,
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
        final_score: score,
        max_tile: board.max_tile(),
        moves,
    }
}

'''
if "struct ActionTrace" not in s:
    s = s.replace(marker, insert + marker)

args_marker = '''    /// Output directory.\n    #[arg(long, default_value = "results")]\n    out: PathBuf,\n'''
args_insert = '''    /// Write one traced game as JSON instead of running the benchmark.\n    /// Requires exactly one policy and one budget.\n    #[arg(long)]\n    trace_json: Option<PathBuf>,\n\n'''
if "trace_json: Option<PathBuf>" not in s:
    s = s.replace(args_marker, args_insert + args_marker)

main_marker = '''    let path = args.out.join("games.csv");\n\n    // Resume support: read completed keys before constructing the pending queue.\n'''
main_insert = '''    if let Some(trace_path) = &args.trace_json {\n        if policies.len() != 1 || budgets.len() != 1 {\n            return Err("--trace-json requires exactly one --policies entry and one --budgets entry".into());\n        }\n        if let Some(parent) = trace_path.parent() {\n            if !parent.as_os_str().is_empty() {\n                fs::create_dir_all(parent)?;\n            }\n        }\n        let trace = play_trace(\n            args.seed,\n            policies[0],\n            budgets[0],\n            args.rollout_cap,\n            transform,\n        );\n        fs::write(trace_path, serde_json::to_string_pretty(&trace)?)?;\n        eprintln!(\n            "trace: seed={} policy={} B={} score={} tile={} moves={} -> {}",\n            trace.seed,\n            trace.policy,\n            trace.simulations,\n            trace.final_score,\n            trace.max_tile,\n            trace.moves.len(),\n            trace_path.display()\n        );\n        return Ok(());\n    }\n\n    let path = args.out.join("games.csv");\n\n    // Resume support: read completed keys before constructing the pending queue.\n'''
if "--trace-json requires exactly one" not in s:
    s = s.replace(main_marker, main_insert)

main.write_text(s)
