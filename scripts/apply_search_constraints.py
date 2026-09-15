from pathlib import Path
import re

path = Path("src/main.rs")
s = path.read_text()


def replace_once(old: str, new: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f"expected snippet not found:\n{old[:320]}")
    s = s.replace(old, new, 1)


def sub_once(pattern: str, repl: str) -> None:
    global s
    s2, n = re.subn(pattern, repl, s, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"expected one regex match, got {n}: {pattern[:180]}")
    s = s2


replace_once(
    "use std::io::{self, IsTerminal, Write as IoWrite};\n",
    "use std::io::{self, IsTerminal, Write as IoWrite};\nuse std::num::NonZeroUsize;\n",
)

replace_once(
'''#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchBudget {
    Simulations(usize),
    Time(Duration),
    Unlimited,
}
''',
'''#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
''')

sub_once(
    r'''struct LiveSession \{.*?\n\}\n\nimpl LiveSession \{.*?\n\}\n\nfn describe_budget\(budget: SearchBudget\) -> String \{.*?\n\}\n''',
'''struct LiveSession {
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
''')

replace_once(
'''    budget: SearchBudget,
    simulations: u32,
''',
'''    constraints: SearchConstraints,
    simulations: u32,
''')

sub_once(
    r'''    match session\.budget \{.*?\n    \}\n\n    writeln!\(\n        output,\n        "best''',
'''    let hard_fraction = [
        session.constraints.simulation_limit.map(|limit| {
            snapshot.total_samples as f64 / limit.get() as f64
        }),
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
        "best''')

replace_once(
'''        "keys: space/enter act | arrows force | +/- budget | i infinite | [/] refresh | q quit"
''',
'''        "keys: space/enter act | arrows force | +/- resource cap | i toggle resource caps | [/] refresh | q quit"
''')

replace_once(
'''fn search_move_with_diagnostics<R: Rng + ?Sized>(
    board: Board,
    budget: SearchBudget,
    cfg: &SearchCfg,
    rng: &mut R,
    mut live: Option<LiveSearchRuntime<'_>>,
    voc_cost: Option<f64>,
) -> io::Result<Option<SearchOutcome>> {
''',
'''fn search_move_with_diagnostics<R: Rng + ?Sized>(
    board: Board,
    constraints: SearchConstraints,
    cfg: &SearchCfg,
    rng: &mut R,
    mut live: Option<LiveSearchRuntime<'_>>,
) -> io::Result<Option<SearchOutcome>> {
''')

replace_once(
'''    let mut active_budget = live
        .as_ref()
        .map_or(budget, |runtime| runtime.session.budget);
''',
'''    let mut active_constraints = live
        .as_ref()
        .map_or(constraints, |runtime| runtime.session.constraints);
''')

sub_once(
    r'''    let mut stopped_by_voc = false;\n    let min_time_budget_sims = match cfg\.policy \{.*?\n    \} as u32;\n''',
'''    let mut stopped_by_voc = false;
''')

replace_once(
'''            active_budget = runtime.session.budget;
''',
'''            active_constraints = runtime.session.constraints;
''')

sub_once(
    r'''        // A VOC price turns the outer budget into a ceiling rather than a\n.*?        if budget_exhausted \{\n            break;\n        \}\n''',
'''        // Every enabled field is an independent stopping constraint. Hard
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
''')

replace_once(
'''        budget: active_budget,
        simulations: sims_done,
''',
'''        constraints: active_constraints,
        simulations: sims_done,
''')

replace_once(
'''fn play_trace(
    seed: u64,
    policy: Policy,
    budget: SearchBudget,
    rollout_cap: usize,
    transform: RewardTransform,
    live_frame_interval: Option<Duration>,
    voc_cost: Option<f64>,
) -> io::Result<GameTrace> {
''',
'''fn play_trace(
    seed: u64,
    policy: Policy,
    constraints: SearchConstraints,
    rollout_cap: usize,
    transform: RewardTransform,
    live_frame_interval: Option<Duration>,
) -> io::Result<GameTrace> {
''')

replace_once(
'''        .map(|frame_interval| LiveSession::new(budget, frame_interval))
''',
'''        .map(|frame_interval| LiveSession::new(constraints, frame_interval))
''')

replace_once(
'''            search_move_with_diagnostics(board, budget, &cfg, &mut search_rng, live, voc_cost)?
''',
'''            search_move_with_diagnostics(board, constraints, &cfg, &mut search_rng, live)?
''')

sub_once(
    r'''        let \(search_budget_ms, search_budget_simulations, search_unlimited\) = match search\.budget \{.*?\n        \};\n''',
'''        let search_budget_ms = search
            .constraints
            .time_limit
            .map(|duration| duration.as_secs_f64() * 1000.0);
        let search_budget_simulations = search
            .constraints
            .simulation_limit
            .map(NonZeroUsize::get);
        let search_unlimited = search.constraints.hard_limits_disabled();
''')

replace_once(
'''            search_voc_cost: voc_cost,
''',
'''            search_voc_cost: search.constraints.min_voc,
''')

sub_once(
    r'''    let \(simulations, time_ms\) = match budget \{.*?\n    \};\n    Ok\(GameTrace \{\n        seed,\n        policy: policy\.name\(\),\n        simulations,\n        time_ms,\n        voc_cost,\n''',
'''    let simulations = constraints.simulation_limit.map(NonZeroUsize::get);
    let time_ms = constraints
        .time_limit
        .map(|duration| duration.as_secs_f64() * 1000.0);
    Ok(GameTrace {
        seed,
        policy: policy.name(),
        simulations,
        time_ms,
        voc_cost: constraints.min_voc,
''')

replace_once(
'''    /// Comma-separated simulations per move.
    #[arg(long, default_value = "8,16,32,64,128")]
    budgets: String,
''',
'''    /// Comma-separated fixed simulation budgets for benchmark mode. In
    /// single-game mode, one value is accepted as a legacy simulation limit.
    #[arg(long)]
    budgets: Option<String>,
''')

replace_once(
'''    /// Play one game live in the terminal instead of running the benchmark.
    #[arg(long, default_value_t = false)]
    play: bool,

    /// Wall-clock thinking budget per move in milliseconds for --play/--trace-json.
    /// When set, --budgets is ignored for the single-game trace/play path.
    #[arg(long)]
    time_ms: Option<f64>,
''',
'''    /// Play one game live in the terminal instead of running the benchmark.
    #[arg(long, default_value_t = false)]
    play: bool,

    /// Maximum simulations per move in --play/--trace-json mode.
    #[arg(long)]
    simulation_limit: Option<NonZeroUsize>,

    /// Maximum search-compute time per move in milliseconds in
    /// --play/--trace-json mode.
    #[arg(long)]
    time_ms: Option<f64>,
''')

replace_once(
'''    /// Stop buying root simulations once max one-step exact VOC is at or
    /// below this price, expressed in actual 2048 score units per simulation.
    /// The fixed simulation/time budget remains a hard safety ceiling.
''',
'''    /// Minimum root one-step VOC required to buy another simulation,
    /// expressed in actual 2048 score units. This is an independent stopping
    /// constraint alongside the simulation and time limits.
''')

sub_once(
    r'''fn main\(\) -> Result<\(\), Box<dyn std::error::Error>> \{\n    let args = Args::parse\(\);\n    let single_game = args\.play \|\| args\.trace_json\.is_some\(\);\n.*?    let policies: Vec<Policy> = args\n        \.policies''',
'''fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let single_game = args.play || args.trace_json.is_some();
    const DEFAULT_BUDGETS: &str = "8,16,32,64,128";
    let budgets = if single_game {
        Vec::new()
    } else {
        parse_usizes(args.budgets.as_deref().unwrap_or(DEFAULT_BUDGETS))?
    };
    let policies: Vec<Policy> = args
        .policies''')

replace_once(
'''    if args.time_ms.is_some() && !args.play && args.trace_json.is_none() {
        return Err("--time-ms currently requires --play or --trace-json".into());
    }
''',
'''    if args.simulation_limit.is_some() && !single_game {
        return Err("--simulation-limit requires --play or --trace-json".into());
    }
    if args.time_ms.is_some() && !single_game {
        return Err("--time-ms requires --play or --trace-json".into());
    }
''')

sub_once(
    r'''    if single_game \{\n        if policies\.len\(\) != 1 \{.*?        let trace = play_trace\(\n            args\.seed,\n            policies\[0\],\n            budget,\n            args\.rollout_cap,\n            transform,\n            args\.play\n                \.then\(\|\| Duration::from_secs_f64\(args\.frame_ms / 1000\.0\)\),\n            args\.voc_cost,\n        \)\?;\n''',
'''    if single_game {
        if policies.len() != 1 {
            return Err("--play/--trace-json requires exactly one --policies entry".into());
        }
        if !args.frame_ms.is_finite() || args.frame_ms <= 0.0 {
            return Err("--frame-ms must be finite and positive".into());
        }

        let legacy_simulation_limit = if let Some(raw) = args.budgets.as_deref() {
            if args.simulation_limit.is_some() {
                return Err("use either --simulation-limit or --budgets in single-game mode, not both".into());
            }
            let parsed = parse_usizes(raw)?;
            if parsed.len() != 1 {
                return Err("single-game --budgets accepts exactly one value; prefer --simulation-limit".into());
            }
            NonZeroUsize::new(parsed[0]).ok_or("simulation limit must be positive")?
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
            return Err("non-interactive trace search needs at least one stopping constraint".into());
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
''')

sub_once(
    r'''        let budget = match \(trace\.time_ms, trace\.simulations\) \{.*?\n        \};\n        eprintln!\(\n            "done: seed=\{\} policy=\{\} budget=\{\} score=\{\} tile=\{\} moves=\{\}",\n''',
'''        let constraints = SearchConstraints {
            simulation_limit: trace.simulations.and_then(NonZeroUsize::new),
            time_limit: trace.time_ms.map(|ms| Duration::from_secs_f64(ms / 1000.0)),
            min_voc: trace.voc_cost,
        };
        eprintln!(
            "done: seed={} policy={} constraints={} score={} tile={} moves={}",
''')

replace_once(
'''            budget,
            trace.final_score,
''',
'''            constraints.describe(),
            trace.final_score,
''')

# Tests: fixed diagnostic search uses one simulation constraint.
replace_once(
'''            SearchBudget::Simulations(64),
            &cfg,
            &mut diagnostic_rng,
            None,
            None,
''',
'''            SearchConstraints {
                simulation_limit: NonZeroUsize::new(64),
                ..SearchConstraints::default()
            },
            &cfg,
            &mut diagnostic_rng,
            None,
''')

sub_once(
    r'''    #\[test\]\n    fn live_budget_controls_toggle_and_adjust\(\) \{.*?\n    \}\n\n    #\[test\]\n    fn live_action_and_quit_keys_map_to_commands\(\) \{.*?\n        assert_eq!\(\n            session\.handle_key\(KeyEvent::new\(KeyCode::Char\('c'\), KeyModifiers::CONTROL\)\),\n            LiveCommand::Quit\n        \);\n    \}\n''',
'''    #[test]
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
        assert_eq!(session.constraints.time_limit, Some(Duration::from_millis(350)));
        session.handle_key(KeyEvent::new(KeyCode::Char('-'), KeyModifiers::NONE));
        assert_eq!(session.constraints.time_limit, Some(Duration::from_millis(300)));

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
''')

path.write_text(s)
print("refactored search budgets into composable search constraints")
