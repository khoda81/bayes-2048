#!/usr/bin/env python3
from pathlib import Path

main = Path("src/main.rs")
s = main.read_text()

if "fn render_search_frame(" in s:
    print("live dashboard already applied")
    raise SystemExit(0)

if "fn choose_move_with_trace" not in s or "time_ms: Option<f64>" not in s:
    raise SystemExit("apply scripts/apply_time_budget.py first")

# Enrich per-action trace statistics.
s = s.replace(
    "    samples: u32,\n    mean: f64,\n    voc: f64,\n}",
    "    samples: u32,\n    share: f64,\n    mean: f64,\n    sd: f64,\n    gap_to_best: f64,\n    voc: f64,\n    switch_prob: f64,\n}",
    1,
)

# Add the probability that one additional observation would flip the current
# best-vs-competitor ordering. This is a useful companion to VOC: probability
# says how plausible a flip is, VOC says how consequential it is.
anchor = "fn least_sampled(edges: &[Edge]) -> usize {"
helper = r'''fn one_step_switch_prob(stats: Stats, other_best: f64) -> f64 {
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
    if stats.mean >= other_best {
        t.cdf(z)
    } else {
        t.sf(z)
    }
}

'''
if "fn one_step_switch_prob(" not in s:
    s = s.replace(anchor, helper + anchor)

choose_start = s.index("fn choose_move_with_trace")
choose_end = s.index("\nfn print_live_trace(", choose_start)

new_choose = r'''fn root_action_traces(root: &Node, cfg: &SearchCfg) -> Vec<ActionTrace> {
    let means: Vec<f64> = root.edges.iter().map(|e| e.stats.mean).collect();
    let best_mean = means
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let total: u32 = root.edges.iter().map(|e| e.stats.n).sum();

    root.edges
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
                share: if total > 0 {
                    e.stats.n as f64 / total as f64
                } else {
                    0.0
                },
                mean: e.stats.mean,
                sd: e.stats.sd(),
                gap_to_best: e.stats.mean - best_mean,
                voc,
                switch_prob: one_step_switch_prob(e.stats, other_best),
            }
        })
        .collect()
}

fn render_search_frame(
    board: Board,
    root: &Node,
    cfg: &SearchCfg,
    move_index: usize,
    score: u64,
    search_elapsed: Duration,
    budget: Duration,
    sims_done: u32,
) {
    let actions = root_action_traces(root, cfg);
    let best_edge = root
        .edges
        .iter()
        .max_by(|a, b| a.stats.mean.total_cmp(&b.stats.mean));

    let elapsed_ms = search_elapsed.as_secs_f64() * 1000.0;
    let budget_ms = budget.as_secs_f64() * 1000.0;
    let progress = (elapsed_ms / budget_ms).clamp(0.0, 1.0);
    let bar_n = 24usize;
    let filled = (progress * bar_n as f64).round() as usize;
    let bar = format!("{}{}", "=".repeat(filled.min(bar_n)), "-".repeat(bar_n - filled.min(bar_n)));
    let sps = if search_elapsed.as_secs_f64() > 0.0 {
        sims_done as f64 / search_elapsed.as_secs_f64()
    } else {
        0.0
    };

    let mut sorted_means: Vec<f64> = root.edges.iter().map(|e| e.stats.mean).collect();
    sorted_means.sort_by(|a, b| b.total_cmp(a));
    let mean_gap = if sorted_means.len() >= 2 {
        sorted_means[0] - sorted_means[1]
    } else {
        0.0
    };

    let total_n: f64 = root.edges.iter().map(|e| e.stats.n as f64).sum();
    let alloc_entropy = if total_n > 0.0 && root.edges.len() > 1 {
        let h = root
            .edges
            .iter()
            .filter_map(|e| {
                let p = e.stats.n as f64 / total_n;
                (p > 0.0).then_some(-p * p.ln())
            })
            .sum::<f64>();
        h / (root.edges.len() as f64).ln()
    } else {
        0.0
    };

    print!("\x1b[2J\x1b[H");
    println!(
        "exact-VOC search | move {} | score {} | max {} | empty {}",
        move_index + 1,
        score,
        tile_symbol(board.0.iter().copied().max().unwrap_or(0)),
        board.empty_count(),
    );
    println!(
        "think {:>6.1}/{:<6.1} ms [{}] {:>3.0}% | {:>7} sims | {:>8.0} sims/s",
        elapsed_ms,
        budget_ms,
        bar,
        100.0 * progress,
        sims_done,
        sps,
    );
    println!(
        "best {:>5} | mean gap {:>9.1} | allocation entropy {:.3}",
        best_edge.map(|e| e.dir.to_string()).unwrap_or_else(|| "-".into()),
        mean_gap,
        alloc_entropy,
    );

    println!("\nbefore:");
    print_compact_board(board);

    println!("\nafter (current best, before spawn):");
    if let Some(edge) = best_edge {
        print_compact_board(edge.moved);
    } else {
        print_compact_board(board);
    }

    println!("\nroot posterior:");
    println!(" dir |      n | alloc |       mean |       sd |     gap |       VOC | switch");
    println!("-----+--------+-------+------------+----------+---------+-----------+-------");
    for dir in DIRS {
        if let Some(a) = actions.iter().find(|a| a.dir == dir.to_string()) {
            let mark = if best_edge.map(|e| e.dir) == Some(dir) { '*' } else { ' ' };
            println!(
                "{}{:>4} | {:>6} | {:>4.0}% | {:>10.1} | {:>8.1} | {:>7.1} | {:>9.4} | {:>5.1}%",
                mark,
                a.dir,
                a.samples,
                100.0 * a.share,
                a.mean,
                a.sd,
                a.gap_to_best,
                a.voc,
                100.0 * a.switch_prob,
            );
        } else {
            println!(" {:>4} | {:>6} | {:>5} | {:>10} | {:>8} | {:>7} | {:>9} | {:>6}", dir, "-", "-", "illegal", "-", "-", "-", "-");
        }
    }
    println!("\n* current Bayes action | switch = P(one more sample flips best-vs-competitor ordering)");
}

fn choose_move_with_trace<R: Rng + ?Sized>(
    board: Board,
    simulations: Option<usize>,
    time_ms: Option<f64>,
    cfg: &SearchCfg,
    rng: &mut R,
    live: bool,
    frame_ms: f64,
    move_index: usize,
    score: u64,
) -> Option<(Dir, Vec<ActionTrace>, u32, f64)> {
    let mut root = Node::new(board);
    if root.terminal() {
        return None;
    }

    let start = Instant::now();
    let mut render_overhead = Duration::ZERO;
    let mut sims_done = 0u32;

    if let Some(ms) = time_ms {
        let duration = Duration::from_secs_f64(ms / 1000.0);
        let frame_every = Duration::from_secs_f64(frame_ms / 1000.0);
        let mut next_frame_wall = frame_every;
        let min_sims = match cfg.policy {
            Policy::Uct => root.edges.len(),
            _ => 3 * root.edges.len(),
        } as u32;

        loop {
            let compute_elapsed = start.elapsed().saturating_sub(render_overhead);
            if sims_done >= min_sims && compute_elapsed >= duration {
                break;
            }

            simulate(&mut root, cfg, rng);
            sims_done += 1;

            if live && start.elapsed() >= next_frame_wall {
                let render_start = Instant::now();
                let compute_elapsed = start.elapsed().saturating_sub(render_overhead);
                render_search_frame(
                    board,
                    &root,
                    cfg,
                    move_index,
                    score,
                    compute_elapsed.min(duration),
                    duration,
                    sims_done,
                );
                render_overhead += render_start.elapsed();
                next_frame_wall += frame_every;
            }
        }

        if live {
            let render_start = Instant::now();
            let compute_elapsed = start.elapsed().saturating_sub(render_overhead);
            render_search_frame(
                board,
                &root,
                cfg,
                move_index,
                score,
                compute_elapsed.min(duration),
                duration,
                sims_done,
            );
            render_overhead += render_start.elapsed();
        }
    } else {
        let n = simulations.expect("simulation budget when time_ms is absent");
        for _ in 0..n {
            simulate(&mut root, cfg, rng);
            sims_done += 1;
        }
    }

    let search_elapsed = start.elapsed().saturating_sub(render_overhead);
    let elapsed_ms = search_elapsed.as_secs_f64() * 1000.0;
    let actions = root_action_traces(&root, cfg);
    let dir = root
        .edges
        .iter()
        .max_by(|a, b| a.stats.mean.total_cmp(&b.stats.mean))?
        .dir;
    Some((dir, actions, sims_done, elapsed_ms))
}
'''

s = s[:choose_start] + new_choose + s[choose_end:]

# Remove the old post-move renderer entirely: frames now happen DURING search.
print_start = s.index("fn print_live_trace(")
print_end = s.index("\nfn play_trace(", print_start)
s = s[:print_start] + s[print_end:]

# Add frame_ms to play_trace and route it into the live search.
s = s.replace(
    "    transform: RewardTransform,\n    live: bool,\n) -> GameTrace {",
    "    transform: RewardTransform,\n    live: bool,\n    frame_ms: f64,\n) -> GameTrace {",
    1,
)
s = s.replace(
    "            &cfg,\n            &mut search_rng,\n        ) else {",
    "            &cfg,\n            &mut search_rng,\n            live,\n            frame_ms,\n            move_index,\n            score,\n        ) else {",
    1,
)

# Delete the old render-after-spawn call if apply_time_budget.py inserted it.
old_render = '''        if live {\n            print_live_trace(\n                move_index,\n                score,\n                before,\n                spawned,\n                dir,\n                reward,\n                search_ms,\n                search_simulations,\n                &actions,\n            );\n        }\n\n'''
s = s.replace(old_render, "")

# CLI frame interval.
arg_anchor = '''    /// Optionally write the single played/traced game as JSON.\n    #[arg(long)]\n    trace_json: Option<PathBuf>,\n'''
arg_insert = '''    /// Live terminal refresh interval in milliseconds while searching.\n    #[arg(long, default_value_t = 50.0)]\n    frame_ms: f64,\n\n'''
if "    frame_ms: f64," not in s:
    s = s.replace(arg_anchor, arg_insert + arg_anchor)

# Validate frame interval and pass it to play_trace.
validation_anchor = '''        if let Some(ms) = args.time_ms {\n            if !(ms.is_finite() && ms > 0.0) {\n                return Err("--time-ms must be finite and > 0".into());\n            }\n        } else if budgets.len() != 1 {\n'''
validation_replacement = '''        if !(args.frame_ms.is_finite() && args.frame_ms > 0.0) {\n            return Err("--frame-ms must be finite and > 0".into());\n        }\n        if let Some(ms) = args.time_ms {\n            if !(ms.is_finite() && ms > 0.0) {\n                return Err("--time-ms must be finite and > 0".into());\n            }\n        } else if budgets.len() != 1 {\n'''
s = s.replace(validation_anchor, validation_replacement, 1)

s = s.replace(
    "            transform,\n            args.play,\n        );",
    "            transform,\n            args.play,\n            args.frame_ms,\n        );",
    1,
)

main.write_text(s)
print("patched live fixed-frame search dashboard")
