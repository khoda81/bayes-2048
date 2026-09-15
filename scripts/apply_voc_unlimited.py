from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()

if "let implicit_unlimited_voc =" in s:
    print("VOC unlimited-budget semantics already applied")
    raise SystemExit(0)


def replace_once(old: str, new: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f"expected snippet not found:\n{old[:320]}")
    s = s.replace(old, new, 1)


replace_once(
'''    fn new(budget: SearchBudget, frame_interval: Duration) -> io::Result<Self> {
        debug_assert!(!matches!(budget, SearchBudget::Unlimited));
        Ok(Self {
            terminal: LiveTerminal::enter()?,
            budget,
            last_limited_budget: budget,
''',
'''    fn new(budget: SearchBudget, frame_interval: Duration) -> io::Result<Self> {
        // A positive VOC price can make Unlimited the natural initial outer
        // budget. Keep a finite fallback so the interactive `i`/`+`/`-`
        // controls can still restore or edit a safety cap.
        let last_limited_budget = match budget {
            SearchBudget::Unlimited => SearchBudget::Simulations(512),
            limited => limited,
        };
        Ok(Self {
            terminal: LiveTerminal::enter()?,
            budget,
            last_limited_budget,
''')

replace_once(
'''    let (simulations, time_ms) = match budget {
        SearchBudget::Simulations(count) => (Some(count), None),
        SearchBudget::Time(duration) => (None, Some(duration.as_secs_f64() * 1000.0)),
        SearchBudget::Unlimited => unreachable!("CLI budget is always limited"),
    };
''',
'''    let (simulations, time_ms) = match budget {
        SearchBudget::Simulations(count) => (Some(count), None),
        SearchBudget::Time(duration) => (None, Some(duration.as_secs_f64() * 1000.0)),
        SearchBudget::Unlimited => (None, None),
    };
''')

replace_once(
'''    let args = Args::parse();
    let single_game = args.play || args.trace_json.is_some();
    // A wall-clock single game does not consult the fixed-simulation budget at
    // all. In particular, even an otherwise invalid --budgets value is ignored.
    let budgets = if single_game && args.time_ms.is_some() {
        Vec::new()
    } else {
        parse_usizes(&args.budgets)?
    };
''',
'''    let args = Args::parse();
    let single_game = args.play || args.trace_json.is_some();
    const DEFAULT_BUDGETS: &str = "8,16,32,64,128";

    // With a positive computation price, omitting both explicit outer caps
    // means exactly what it sounds like: keep searching until VOC says to stop.
    // `budgets` still has its benchmark default, so use that untouched default
    // as the signal that no single-game simulation cap was supplied.
    let implicit_unlimited_voc = single_game
        && args.time_ms.is_none()
        && matches!(args.voc_cost, Some(cost) if 0.0 < cost)
        && args.budgets == DEFAULT_BUDGETS;

    // Wall-clock and implicitly-unbounded VOC searches do not consult the
    // fixed-simulation budget. An explicitly supplied single --budgets value
    // remains a hard safety ceiling.
    let budgets = if single_game && (args.time_ms.is_some() || implicit_unlimited_voc) {
        Vec::new()
    } else {
        parse_usizes(&args.budgets)?
    };
''')

replace_once(
'''        if policies.len() != 1 || !matches!(policies[0], Policy::ExactVoc) {
            return Err("--voc-cost currently requires --policies exact-voc".into());
        }
    }

    if single_game {
''',
'''        if policies.len() != 1 || !matches!(policies[0], Policy::ExactVoc) {
            return Err("--voc-cost currently requires --policies exact-voc".into());
        }
        if cost == 0.0 && args.time_ms.is_none() && args.budgets == DEFAULT_BUDGETS {
            return Err(
                "--voc-cost 0 requires an explicit --budgets or --time-ms safety cap"
                    .into(),
            );
        }
    }

    if single_game {
''')

replace_once(
'''        } else if budgets.len() != 1 {
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
''',
'''        } else if !implicit_unlimited_voc && budgets.len() != 1 {
            return Err(
                "without --time-ms or a positive --voc-cost, --play/--trace-json requires exactly one --budgets entry"
                    .into(),
            );
        }

        let budget = if let Some(ms) = args.time_ms {
            SearchBudget::Time(Duration::from_secs_f64(ms / 1000.0))
        } else if implicit_unlimited_voc {
            SearchBudget::Unlimited
        } else {
            SearchBudget::Simulations(budgets[0])
        };
''')

replace_once(
'''        let budget = match (trace.time_ms, trace.simulations) {
            (Some(ms), _) => format!("{ms:.1} ms/move"),
            (_, Some(n)) => format!("{n} sims/move"),
            _ => "unknown budget".into(),
        };
''',
'''        let budget = match (trace.time_ms, trace.simulations, trace.voc_cost) {
            (Some(ms), _, Some(cost)) => {
                format!("{ms:.1} ms cap; VOC cost {cost:.4}/sim")
            }
            (_, Some(n), Some(cost)) => {
                format!("{n} sims cap; VOC cost {cost:.4}/sim")
            }
            (None, None, Some(cost)) => {
                format!("VOC cost {cost:.4}/sim; no outer cap")
            }
            (Some(ms), _, None) => format!("{ms:.1} ms/move"),
            (_, Some(n), None) => format!("{n} sims/move"),
            _ => "unknown budget".into(),
        };
''')

path.write_text(s)
print("patched src/main.rs with implicit unlimited VOC-budget semantics")
