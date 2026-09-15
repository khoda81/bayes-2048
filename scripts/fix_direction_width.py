from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()

replacements = [
    (
        '        write!(output, " {mark} {dir:<2}|").expect("writing to String cannot fail");\n',
        '        write!(output, " {mark} {:<2}|", dir.symbol()).expect("writing to String cannot fail");\n',
    ),
    (
        '        let labels: Vec<String> = DIRS.into_iter().map(|dir| format!("  {dir:<2}|")).collect();\n',
        '        let labels: Vec<String> = DIRS\n            .into_iter()\n            .map(|dir| format!("  {:<2}|", dir.symbol()))\n            .collect();\n',
    ),
]

changed = False
for old, new in replacements:
    if old in s:
        s = s.replace(old, new, 1)
        changed = True

if changed:
    path.write_text(s)
    print("fixed Unicode direction label width")
else:
    print("direction label width already fixed")
