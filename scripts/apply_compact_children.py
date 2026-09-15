from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()

old = "    children: HashMap<Board, NodeId>,"
new = "    children: SmallVec<[(u8, NodeId); 4]>,"
assert old in s
s = s.replace(old, new, 1)

old = "            children: HashMap::new(),"
new = "            children: SmallVec::new(),"
assert old in s
s = s.replace(old, new, 1)

marker = "fn merge_line(input: [u8; 4]) -> ([u8; 4], u64) {"
assert marker in s
helper = r'''/// Compact identity of a stochastic 2048 spawn relative to the board
/// immediately after the chosen move: 2 * cell + tile_kind, where tile_kind
/// is 0 for a 2 tile and 1 for a 4 tile.
///
/// A spawn changes exactly one cell, so this is a complete key for a chance
/// child while avoiding storing/hashing another 16-byte Board per entry.
fn spawn_key(before: Board, after: Board) -> u8 {
    for i in 0..16 {
        if before.0[i] != after.0[i] {
            debug_assert_eq!(before.0[i], 0);
            let kind = match after.0[i] {
                1 => 0,
                2 => 1,
                exponent => panic!("unexpected spawned exponent {exponent}"),
            };
            return (2 * i + kind) as u8;
        }
    }
    panic!("spawned board did not differ from moved board")
}

'''
s = s.replace(marker, helper + marker, 1)

old = """    let child_id = tree.node(node_id).edges[edge_i]
        .children
        .get(&spawned)
        .copied();"""
new = """    let child_key = spawn_key(moved, spawned);
    let child_id = tree.node(node_id).edges[edge_i]
        .children
        .iter()
        .find_map(|&(key, id)| (key == child_key).then_some(id));"""
assert old in s
s = s.replace(old, new, 1)

old = """        tree.node_mut(node_id).edges[edge_i]
            .children
            .insert(spawned, child_id);"""
new = """        tree.node_mut(node_id).edges[edge_i]
            .children
            .push((child_key, child_id));"""
assert old in s
s = s.replace(old, new, 1)

path.write_text(s)
