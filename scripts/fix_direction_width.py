from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()

if "fn dir_symbol(dir: Dir) -> char" not in s:
    display_impl = '''impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Up => "↑",
            Self::Down => "↓",
            Self::Left => "←",
            Self::Right => "→",
        })
    }
}
'''
    helper = display_impl + '''
fn dir_symbol(dir: Dir) -> char {
    match dir {
        Dir::Up => '↑',
        Dir::Down => '↓',
        Dir::Left => '←',
        Dir::Right => '→',
    }
}
'''
    if display_impl not in s:
        raise SystemExit("Dir Display impl not found")
    s = s.replace(display_impl, helper, 1)

replacements = [
    (
        '        write!(output, " {mark} {dir:<2}|").expect("writing to String cannot fail");\n',
        '        write!(output, " {mark} {:<2}|", dir_symbol(dir))\n            .expect("writing to String cannot fail");\n',
    ),
    (
        '        let labels: Vec<String> = DIRS.into_iter().map(|dir| format!("  {dir:<2}|")).collect();\n',
        '        let labels: Vec<String> = DIRS\n            .into_iter()\n            .map(|dir| format!("  {:<2}|", dir_symbol(dir)))\n            .collect();\n',
    ),
    (
        '        write!(output, " {mark} {:<2}|", dir.symbol()).expect("writing to String cannot fail");\n',
        '        write!(output, " {mark} {:<2}|", dir_symbol(dir))\n            .expect("writing to String cannot fail");\n',
    ),
    (
        '            .map(|dir| format!("  {:<2}|", dir.symbol()))\n',
        '            .map(|dir| format!("  {:<2}|", dir_symbol(dir)))\n',
    ),
]

changed = False
for old, new in replacements:
    if old in s:
        s = s.replace(old, new, 1)
        changed = True

path.write_text(s)
print("fixed Unicode direction label width")
