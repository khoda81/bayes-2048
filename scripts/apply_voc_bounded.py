from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()


def replace_once(old: str, new: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f"expected snippet not found:\n{old[:240]}")
    s = s.replace(old, new, 1)


replace_once(
'''fn exact_voc(stats: Stats, other_best: f64) -> f64 {
    one_step_mean_distribution(stats, other_best).map_or(0.0, |distribution| distribution.voc())
}
''',
'''fn exact_voc(stats: Stats, other_best: f64) -> f64 {
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
''')

replace_once(
'''    search_unlimited: bool,
    spawn_index: usize,
''',
'''    search_unlimited: bool,
    search_stopped_by_voc: bool,
    search_max_voc: f64,
    search_voc_cost: Option<f64>,
    spawn_index: usize,
''')

replace_once(
'''    simulations: Option<usize>,
    time_ms: Option<f64>,
    final_score: u64,
''',
'''    simulations: Option<usize>,
    time_ms: Option<f64>,
    voc_cost: Option<f64>,
    final_score: u64,
''')

replace_once(
'''    excluded_overhead: Duration,
}
''',
'''    excluded_overhead: Duration,
    stopped_by_voc: bool,
    max_voc: f64,
}
''')

replace_once(
'''    cfg: &SearchCfg,
    rng: &mut R,
    mut live: Option<LiveSearchRuntime<'_>>,
) -> io::Result<Option<SearchOutcome>> {
''',
'''    cfg: &SearchCfg,
    rng: &mut R,
    mut live: Option<LiveSearchRuntime<'_>>,
    voc_cost: Option<f64>,
) -> io::Result<Option<SearchOutcome>> {
''')

replace_once(
'''    let mut last_rendered_state = None;
    let mut selected_dir = None;
    let min_time_budget_sims = match cfg.policy {
''',
'''    let mut last_rendered_state = None;
    let mut selected_dir = None;
    let mut stopped_by_voc = false;
    let min_time_budget_sims = match cfg.policy {
''')

replace_once(
'''        let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
        let budget_exhausted = match active_budget {
''',
'''        // A VOC price turns the outer budget into a ceiling rather than a
        // quota. Bootstrap every competing root action to n=3 first because
        // the Jeffreys-Normal posterior mean/VOC is not defined before then.
        // With only one legal action there is no decision to improve, so no
        // bootstrap is necessary at all.
        if let Some(raw_cost) = voc_cost {
            let bootstrap_complete = root.edges.len() <= 1
                || root.edges.iter().all(|edge| 3 <= edge.stats.n);
            if bootstrap_complete {
                let max_voc = max_exact_voc(&root);
                // --voc-cost is expressed in actual game-score units. The
                // allocator may see an affine-scaled reward, so scale the
                // price too; reward shifts cancel from VOC.
                let effective_cost = raw_cost * cfg.reward_transform.scale;
                if max_voc <= effective_cost {
                    stopped_by_voc = true;
                    if let Some(runtime) = live.as_mut() {
                        runtime.session.set_notice(format!(
                            "VOC stop: max {:.4} <= cost {:.4}",
                            max_voc / cfg.reward_transform.scale,
                            raw_cost,
                        ));
                    }
                    break;
                }
            }
        }

        let compute_elapsed = wall_start.elapsed().saturating_sub(excluded_overhead);
        let budget_exhausted = match active_budget {
''')

replace_once(
'''        excluded_overhead,
    }))
}

fn play_trace(
''',
'''        excluded_overhead,
        stopped_by_voc,
        max_voc: max_exact_voc(&root) / cfg.reward_transform.scale,
    }))
}

fn play_trace(
''')

replace_once(
'''    transform: RewardTransform,
    live_frame_interval: Option<Duration>,
) -> io::Result<GameTrace> {
''',
'''    transform: RewardTransform,
    live_frame_interval: Option<Duration>,
    voc_cost: Option<f64>,
) -> io::Result<GameTrace> {
''')

replace_once(
'''            search_move_with_diagnostics(board, budget, &cfg, &mut search_rng, live)?
''',
'''            search_move_with_diagnostics(board, budget, &cfg, &mut search_rng, live, voc_cost)?
''')

replace_once(
'''            search_unlimited,
            spawn_index,
''',
'''            search_unlimited,
            search_stopped_by_voc: search.stopped_by_voc,
            search_max_voc: search.max_voc,
            search_voc_cost: voc_cost,
            spawn_index,
''')

replace_once(
'''        simulations,
        time_ms,
        final_score: score,
''',
'''        simulations,
        time_ms,
        voc_cost,
        final_score: score,
''')

replace_once(
'''    /// Live terminal refresh interval in milliseconds while searching.
    #[arg(long, default_value_t = 50.0)]
    frame_ms: f64,
''',
'''    /// Stop buying root simulations once max one-step exact VOC is at or
    /// below this price, expressed in actual 2048 score units per simulation.
    /// The fixed simulation/time budget remains a hard safety ceiling.
    #[arg(long)]
    voc_cost: Option<f64>,

    /// Live terminal refresh interval in milliseconds while searching.
    #[arg(long, default_value_t = 50.0)]
    frame_ms: f64,
''')

replace_once(
'''    if args.time_ms.is_some() && !args.play && args.trace_json.is_none() {
        return Err("--time-ms currently requires --play or --trace-json".into());
    }

    if single_game {
''',
'''    if args.time_ms.is_some() && !args.play && args.trace_json.is_none() {
        return Err("--time-ms currently requires --play or --trace-json".into());
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
''')

replace_once(
'''            args.play
                .then(|| Duration::from_secs_f64(args.frame_ms / 1000.0)),
        )?;
''',
'''            args.play
                .then(|| Duration::from_secs_f64(args.frame_ms / 1000.0)),
            args.voc_cost,
        )?;
''')

replace_once(
'''        let diagnostic = search_move_with_diagnostics(
            board,
            SearchBudget::Simulations(64),
            &cfg,
            &mut diagnostic_rng,
            None,
        )
''',
'''        let diagnostic = search_move_with_diagnostics(
            board,
            SearchBudget::Simulations(64),
            &cfg,
            &mut diagnostic_rng,
            None,
            None,
        )
''')

path.write_text(s)
