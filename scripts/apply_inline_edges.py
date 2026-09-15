from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()

old = "use serde::{Deserialize, Serialize};\nuse statrs::distribution::{Continuous, ContinuousCDF, StudentsT};"
new = "use serde::{Deserialize, Serialize};\nuse smallvec::SmallVec;\nuse statrs::distribution::{Continuous, ContinuousCDF, StudentsT};"
assert old in s
s = s.replace(old, new, 1)

old = "    edges: Vec<Edge>,"
new = "    edges: SmallVec<[Edge; 4]>,"
assert old in s
s = s.replace(old, new, 1)

# Make the intended inline representation explicit instead of relying on
# assignment-context inference at collect().
old = "            .map(|(d, b, r)| Edge::new(d, b, r))\n            .collect();"
new = "            .map(|(d, b, r)| Edge::new(d, b, r))\n            .collect::<SmallVec<[Edge; 4]>>();"
assert old in s
s = s.replace(old, new, 1)

path.write_text(s)
