from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()

if "enum ConstraintKind" in s:
    print("constraint TUI already applied")
    raise SystemExit(0)


def replace_once(old: str, new: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f"expected snippet not found:\n{old[:500]}")
    s = s.replace(old, new, 1)


replace_once(
'''#[derive(Clone, Copy, Debug)]
struct LiveSearch {
    move_index: usize,
    score: u64,
}
''',
'''#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstraintKind {
    Simulations,
    Time,
    Voc,
}

impl ConstraintKind {
    fn next(self) -> Self {
        match self {
            Self::Simulations => Self::Time,
            Self::Time => Self::Voc,
            Self::Voc => Self::Simulations,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Simulations => "simulations",
            Self::Time => "time",
            Self::Voc => "VOC",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LiveSearch {
    move_index: usize,
    score: u64,
}
''')

replace_once(
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
''',
'''struct LiveSession {
    terminal: LiveTerminal,
    constraints: SearchConstraints,
    selected_constraint: ConstraintKind,
    saved_simulation_limit: NonZeroUsize,
    saved_time_limit: Duration,
    saved_min_voc: f64,
    frame_interval: Duration,
    notice: String,
    revision: u64,
}

impl LiveSession {
    const TIME_STEP: Duration = Duration::from_millis(50);
    const MIN_TIME: Duration = Duration::from_millis(10);
    const SIMULATION_STEP: usize = 128;
    const DEFAULT_SIMULATION_LIMIT: NonZeroUsize = NonZeroUsize::new(512).unwrap();
    const DEFAULT_TIME_LIMIT: Duration = Duration::from_millis(300);
    const DEFAULT_MIN_VOC: f64 = 1.0;
    const VOC_ZERO_NUDGE: f64 = 0.001;
    const VOC_FACTOR: f64 = 10.0;
    const FRAME_STEP: Duration = Duration::from_millis(10);
    const MIN_FRAME: Duration = Duration::from_millis(10);

    fn new(constraints: SearchConstraints, frame_interval: Duration) -> io::Result<Self> {
        Ok(Self {
            terminal: LiveTerminal::enter()?,
            constraints,
            selected_constraint: ConstraintKind::Simulations,
            saved_simulation_limit: constraints
                .simulation_limit
                .unwrap_or(Self::DEFAULT_SIMULATION_LIMIT),
            saved_time_limit: constraints.time_limit.unwrap_or(Self::DEFAULT_TIME_LIMIT),
            saved_min_voc: constraints.min_voc.unwrap_or(Self::DEFAULT_MIN_VOC),
            frame_interval,
            notice: "searching".into(),
            revision: 0,
        })
    }
''')

start = s.index("    fn toggle_resource_limits(&mut self) {")
end = s.index("    fn adjust_frame_interval(&mut self, increase: bool) {", start)
s = s[:start] + '''    fn select_constraint(&mut self, selected: ConstraintKind) {
        self.selected_constraint = selected;
        self.set_notice(format!("selected {} constraint", selected.name()));
    }

    fn toggle_selected_constraint(&mut self) {
        match self.selected_constraint {
            ConstraintKind::Simulations => {
                if let Some(value) = self.constraints.simulation_limit {
                    self.saved_simulation_limit = value;
                    self.constraints.simulation_limit = None;
                    self.set_notice("simulation constraint disabled");
                } else {
                    self.constraints.simulation_limit = Some(self.saved_simulation_limit);
                    self.set_notice(format!(
                        "simulation constraint restored to {}",
                        self.saved_simulation_limit
                    ));
                }
            }
            ConstraintKind::Time => {
                if let Some(value) = self.constraints.time_limit {
                    self.saved_time_limit = value;
                    self.constraints.time_limit = None;
                    self.set_notice("time constraint disabled");
                } else {
                    self.constraints.time_limit = Some(self.saved_time_limit);
                    self.set_notice(format!(
                        "time constraint restored to {:.0} ms",
                        self.saved_time_limit.as_secs_f64() * 1000.0
                    ));
                }
            }
            ConstraintKind::Voc => {
                if let Some(value) = self.constraints.min_voc {
                    self.saved_min_voc = value;
                    self.constraints.min_voc = None;
                    self.set_notice("VOC constraint disabled");
                } else {
                    self.constraints.min_voc = Some(self.saved_min_voc);
                    self.set_notice(format!(
                        "VOC constraint restored to {:.4}",
                        self.saved_min_voc
                    ));
                }
            }
        }
    }

    fn adjust_selected_constraint(&mut self, increase: bool) {
        match self.selected_constraint {
            ConstraintKind::Simulations => {
                let current = self
                    .constraints
                    .simulation_limit
                    .unwrap_or(self.saved_simulation_limit)
                    .get();
                let adjusted = if increase {
                    current.saturating_add(Self::SIMULATION_STEP)
                } else {
                    current.saturating_sub(Self::SIMULATION_STEP).max(1)
                };
                self.saved_simulation_limit = NonZeroUsize::new(adjusted).unwrap();
                if self.constraints.simulation_limit.is_some() {
                    self.constraints.simulation_limit = Some(self.saved_simulation_limit);
                }
                self.set_notice(format!(
                    "simulation {} = {}{}",
                    if self.constraints.simulation_limit.is_some() { "limit" } else { "saved value" },
                    adjusted,
                    if self.constraints.simulation_limit.is_some() { "" } else { " (off)" },
                ));
            }
            ConstraintKind::Time => {
                let current = self.constraints.time_limit.unwrap_or(self.saved_time_limit);
                let adjusted = if increase {
                    current.saturating_add(Self::TIME_STEP)
                } else {
                    current.saturating_sub(Self::TIME_STEP).max(Self::MIN_TIME)
                };
                self.saved_time_limit = adjusted;
                if self.constraints.time_limit.is_some() {
                    self.constraints.time_limit = Some(adjusted);
                }
                self.set_notice(format!(
                    "time {} = {:.0} ms{}",
                    if self.constraints.time_limit.is_some() { "limit" } else { "saved value" },
                    adjusted.as_secs_f64() * 1000.0,
                    if self.constraints.time_limit.is_some() { "" } else { " (off)" },
                ));
            }
            ConstraintKind::Voc => {
                let current = self.constraints.min_voc.unwrap_or(self.saved_min_voc);
                let adjusted = if increase {
                    if current == 0.0 {
                        Self::VOC_ZERO_NUDGE
                    } else {
                        current * Self::VOC_FACTOR
                    }
                } else if current <= Self::VOC_ZERO_NUDGE {
                    0.0
                } else {
                    current / Self::VOC_FACTOR
                };
                self.saved_min_voc = adjusted;
                if self.constraints.min_voc.is_some() {
                    self.constraints.min_voc = Some(adjusted);
                }
                self.set_notice(format!(
                    "VOC {} = {:.4}{}",
                    if self.constraints.min_voc.is_some() { "threshold" } else { "saved value" },
                    adjusted,
                    if self.constraints.min_voc.is_some() { "" } else { " (off)" },
                ));
            }
        }
    }

''' + s[end:]

replace_once(
'''            KeyCode::Char('i') => {
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
''',
'''            KeyCode::Char('1') | KeyCode::Char('s') => {
                self.select_constraint(ConstraintKind::Simulations);
                LiveCommand::Continue
            }
            KeyCode::Char('2') | KeyCode::Char('t') => {
                self.select_constraint(ConstraintKind::Time);
                LiveCommand::Continue
            }
            KeyCode::Char('3') | KeyCode::Char('v') => {
                self.select_constraint(ConstraintKind::Voc);
                LiveCommand::Continue
            }
            KeyCode::Tab => {
                self.select_constraint(self.selected_constraint.next());
                LiveCommand::Continue
            }
            KeyCode::Char('x') => {
                self.toggle_selected_constraint();
                LiveCommand::Continue
            }
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::PageUp => {
                self.adjust_selected_constraint(true);
                LiveCommand::Continue
            }
            KeyCode::Char('-') | KeyCode::PageDown => {
                self.adjust_selected_constraint(false);
                LiveCommand::Continue
            }
''')

# Replace the aggregate hard-cap bar and one-line limit summary with one row per constraint.
old_start = s.index("    let hard_fraction = [", s.index("fn render_search_frame("))
old_end = s.index("    writeln!(\n        output,\n        \"best {:>5}", old_start)
new_panel = '''    writeln!(
        output,
        "think {:>8.1} ms | {:>7} sims | {:>8.0} sims/s",
        compute_seconds * 1000.0,
        snapshot.total_samples,
        sims_per_second,
    )
    .expect("writing to String cannot fail");
    writeln!(output, "constraints (first enabled condition to trip stops search):")
        .expect("writing to String cannot fail");

    let sim_mark = if session.selected_constraint == ConstraintKind::Simulations { '>' } else { ' ' };
    if let Some(limit) = session.constraints.simulation_limit {
        let fraction = snapshot.total_samples as f64 / limit.get() as f64;
        writeln!(
            output,
            "{sim_mark} 1 sims  {:>7} / {:<7} [{}] {:>3.0}%",
            snapshot.total_samples,
            limit.get(),
            progress_bar(fraction),
            100.0 * fraction.clamp(0.0, 1.0),
        )
        .expect("writing to String cannot fail");
    } else {
        writeln!(
            output,
            "{sim_mark} 1 sims  off | current {:>7} | saved {}",
            snapshot.total_samples,
            session.saved_simulation_limit,
        )
        .expect("writing to String cannot fail");
    }

    let time_mark = if session.selected_constraint == ConstraintKind::Time { '>' } else { ' ' };
    if let Some(limit) = session.constraints.time_limit {
        let fraction = compute_elapsed.as_secs_f64() / limit.as_secs_f64();
        writeln!(
            output,
            "{time_mark} 2 time  {:>7.1} / {:<7.1} ms [{}] {:>3.0}%",
            compute_seconds * 1000.0,
            limit.as_secs_f64() * 1000.0,
            progress_bar(fraction),
            100.0 * fraction.clamp(0.0, 1.0),
        )
        .expect("writing to String cannot fail");
    } else {
        writeln!(
            output,
            "{time_mark} 2 time  off | current {:>7.1} ms | saved {:.1} ms",
            compute_seconds * 1000.0,
            session.saved_time_limit.as_secs_f64() * 1000.0,
        )
        .expect("writing to String cannot fail");
    }

    let voc_mark = if session.selected_constraint == ConstraintKind::Voc { '>' } else { ' ' };
    if let Some(min_voc) = session.constraints.min_voc {
        let relation = if max_voc <= min_voc { "STOP" } else { "keep" };
        writeln!(
            output,
            "{voc_mark} 3 VOC   {:>9.4} > {:<9.4} [{relation}]",
            max_voc,
            min_voc,
        )
        .expect("writing to String cannot fail");
    } else {
        writeln!(
            output,
            "{voc_mark} 3 VOC   off | current {:>9.4} | saved {:.4}",
            max_voc,
            session.saved_min_voc,
        )
        .expect("writing to String cannot fail");
    }

'''
s = s[:old_start] + new_panel + s[old_end:]

replace_once(
'''        "keys: space/enter act | arrows force | +/- resource cap | i toggle resource caps | [/] refresh | q quit"
''',
'''        "keys: 1/s sims  2/t time  3/v VOC  tab cycle | x toggle | +/- adjust | space/enter act | arrows force | [/] refresh | q quit"
''')

# Update the constraint-control test to exercise each constraint independently.
start = s.index("    #[test]\n    fn live_resource_constraints_toggle_and_adjust() {")
end = s.index("    #[test]\n    fn live_action_and_quit_keys_map_to_commands() {", start)
s = s[:start] + '''    #[test]
    fn live_constraints_select_toggle_and_adjust_independently() {
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
            selected_constraint: ConstraintKind::Simulations,
            saved_simulation_limit: NonZeroUsize::new(512).unwrap(),
            saved_time_limit: Duration::from_millis(300),
            saved_min_voc: 1.0,
            frame_interval: Duration::from_millis(50),
            notice: String::new(),
            revision: 0,
        };

        // Time can be cleared without touching simulations or VOC.
        session.handle_key(KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE));
        assert_eq!(session.selected_constraint, ConstraintKind::Time);
        session.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(session.constraints.time_limit, None);
        assert_eq!(session.constraints.simulation_limit, initial.simulation_limit);
        assert_eq!(session.constraints.min_voc, initial.min_voc);

        // Adjusting a disabled constraint edits its remembered value but does
        // not silently enable it; toggling restores the edited value.
        session.handle_key(KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE));
        assert_eq!(session.constraints.time_limit, None);
        assert_eq!(session.saved_time_limit, Duration::from_millis(350));
        session.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(
            session.constraints.time_limit,
            Some(Duration::from_millis(350))
        );

        // VOC uses logarithmic decade steps: 1 -> 0.1 -> 1.
        session.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE));
        session.handle_key(KeyEvent::new(KeyCode::Char('-'), KeyModifiers::NONE));
        assert_eq!(session.constraints.min_voc, Some(0.1));
        session.handle_key(KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE));
        assert_eq!(session.constraints.min_voc, Some(1.0));
        session.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(session.constraints.min_voc, None);
        assert_eq!(session.constraints.simulation_limit, initial.simulation_limit);
        assert_eq!(
            session.constraints.time_limit,
            Some(Duration::from_millis(350))
        );

        // Simulation limit is independently selectable and restorable too.
        session.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE));
        session.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(session.constraints.simulation_limit, None);
        session.handle_key(KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE));
        assert_eq!(session.saved_simulation_limit.get(), 640);
        assert_eq!(session.constraints.simulation_limit, None);
        session.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(session.constraints.simulation_limit.unwrap().get(), 640);

        session.handle_key(KeyEvent::new(KeyCode::Char('['), KeyModifiers::NONE));
        assert_eq!(session.frame_interval, Duration::from_millis(40));
        session.handle_key(KeyEvent::new(KeyCode::Char(']'), KeyModifiers::NONE));
        assert_eq!(session.frame_interval, Duration::from_millis(50));
    }

''' + s[end:]

replace_once(
'''        let mut session = LiveSession {
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
''',
'''        let mut session = LiveSession {
            terminal: LiveTerminal {
                alternate_screen: false,
            },
            constraints,
            selected_constraint: ConstraintKind::Simulations,
            saved_simulation_limit: NonZeroUsize::new(512).unwrap(),
            saved_time_limit: Duration::from_millis(300),
            saved_min_voc: 1.0,
            frame_interval: Duration::from_millis(50),
            notice: String::new(),
            revision: 0,
        };
''')

path.write_text(s)
print("patched independent constraint controls into TUI")
