#!/usr/bin/env python3
from pathlib import Path

main = Path("src/main.rs")
s = main.read_text()

if "struct Tree {" in s and "children: HashMap<Board, NodeId>" in s:
    print("arena tree already applied")
    raise SystemExit(0)

# Replace recursive ownership (HashMap<Board, Box<Node>>) with an append-only
# arena and compact integer child handles. Keep Vec<Edge> unchanged in this
# experiment so the benchmark isolates the arena allocation change.
start = s.index("struct Edge {")
end = s.index("\n#[derive(Clone, Copy, Debug)]\nenum Policy", start)
new_tree = r'''type NodeId = u32;

struct Edge {
    dir: Dir,
    moved: Board,
    immediate_reward: u64,
    stats: Stats,
    children: HashMap<Board, NodeId>,
}

impl Edge {
    fn new(dir: Dir, moved: Board, immediate_reward: u64) -> Self {
        Self {
            dir,
            moved,
            immediate_reward,
            stats: Stats::default(),
            children: HashMap::new(),
        }
    }
}

struct Node {
    // Retained in the tree representation to keep the benchmark node layout
    // comparable; live root rendering receives its board explicitly.
    _board: Board,
    visits: u32,
    edges: Vec<Edge>,
}

impl Node {
    fn new(board: Board) -> Self {
        let edges = board
            .legal_moves()
            .into_iter()
            .map(|(d, b, r)| Edge::new(d, b, r))
            .collect();
        Self {
            _board: board,
            visits: 0,
            edges,
        }
    }

    fn terminal(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Append-only per-move MCTS arena.
///
/// Node IDs are Vec indices, so Vec reallocation can move Node values without
/// invalidating tree links. A simulation creates at most one node, making a
/// fixed-simulation search easy to reserve exactly up front.
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    const ROOT: NodeId = 0;

    fn with_capacity(root: Board, capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity.max(1));
        nodes.push(Node::new(root));
        Self { nodes }
    }

    fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    fn alloc(&mut self, node: Node) -> NodeId {
        let index = self.nodes.len();
        assert!(index <= NodeId::MAX as usize, "MCTS arena exhausted u32 node IDs");
        self.nodes.push(node);
        index as NodeId
    }
}
'''
s = s[:start] + new_tree + s[end:]

sim_start = s.index("fn simulate<R: Rng + ?Sized>(")
choose_start = s.index("fn choose_move<R: Rng + ?Sized>(", sim_start)
new_sim = r'''fn simulate<R: Rng + ?Sized>(
    tree: &mut Tree,
    node_id: NodeId,
    cfg: &SearchCfg,
    rng: &mut R,
) -> u64 {
    if tree.node(node_id).terminal() {
        return 0;
    }

    tree.node_mut(node_id).visits += 1;
    let edge_i = {
        let node = tree.node(node_id);
        select_edge(node, cfg, rng)
    };

    let (moved, immediate) = {
        let edge = &tree.node(node_id).edges[edge_i];
        (edge.moved, edge.immediate_reward)
    };
    let spawned = moved.spawn(rng);

    // Do not retain any references into the arena across recursion: pushing a
    // newly expanded node may reallocate the Vec. Integer IDs remain stable.
    let child_id = tree.node(node_id).edges[edge_i]
        .children
        .get(&spawned)
        .copied();
    let downstream = if let Some(child_id) = child_id {
        simulate(tree, child_id, cfg, rng)
    } else {
        // Preserve the old RNG/expansion order exactly: rollout first, then
        // materialize the newly encountered child.
        let rollout = random_rollout(spawned, rng, cfg.rollout_cap);
        let child_id = tree.alloc(Node::new(spawned));
        tree.node_mut(node_id).edges[edge_i]
            .children
            .insert(spawned, child_id);
        rollout
    };

    let total = immediate + downstream;
    let observed = cfg.reward_transform.apply(total as f64);
    tree.node_mut(node_id).edges[edge_i].stats.observe(observed);
    total
}

'''
s = s[:sim_start] + new_sim + s[choose_start:]

choose_start = s.index("fn choose_move<R: Rng + ?Sized>(")
action_start = s.index("\n#[derive(Debug, Clone, Serialize)]\nstruct ActionTrace", choose_start)
new_choose = r'''fn choose_move<R: Rng + ?Sized>(
    board: Board,
    simulations: usize,
    cfg: &SearchCfg,
    rng: &mut R,
) -> Option<Dir> {
    // One simulation can expand at most one node, so this avoids arena growth
    // for the fixed-budget benchmark path.
    let mut tree = Tree::with_capacity(board, simulations.saturating_add(1));
    let root_id = Tree::ROOT;
    if tree.node(root_id).terminal() {
        return None;
    }
    for _ in 0..simulations {
        simulate(&mut tree, root_id, cfg, rng);
    }

    // Terminal Bayes action: maximize posterior expected return.
    tree.node(root_id)
        .edges
        .iter()
        .max_by(|a, b| a.stats.mean.total_cmp(&b.stats.mean))
        .map(|e| e.dir)
}
'''
s = s[:choose_start] + new_choose + s[action_start:]

# The interactive/trace search owns the same arena, but its budget may be time
# based or changed while live, so only fixed-simulation mode can reserve the
# exact upper bound.
search_start = s.index("fn search_move_with_diagnostics<R: Rng + ?Sized>(")
search_end = s.index("\nfn play_trace(", search_start)
block = s[search_start:search_end]

old = '''    let mut root = Node::new(board);\n    if root.terminal() {\n        return Ok(None);\n    }\n'''
new = '''    let arena_capacity = match budget {\n        SearchBudget::Simulations(limit) => limit.saturating_add(1),\n        SearchBudget::Time(_) | SearchBudget::Unlimited => 1024,\n    };\n    let mut tree = Tree::with_capacity(board, arena_capacity);\n    let root_id = Tree::ROOT;\n    if tree.node(root_id).terminal() {\n        return Ok(None);\n    }\n'''
if old not in block:
    raise SystemExit("could not find diagnostic root construction")
block = block.replace(old, new, 1)

block = block.replace(
    '''        Policy::Uct => root.edges.len(),\n        Policy::Thompson | Policy::ExactVoc | Policy::McVoc(_) => 3 * root.edges.len(),\n''',
    '''        Policy::Uct => tree.node(root_id).edges.len(),\n        Policy::Thompson | Policy::ExactVoc | Policy::McVoc(_) => {\n            3 * tree.node(root_id).edges.len()\n        }\n''',
    1,
)
block = block.replace(
    "if root.edges.iter().any(|edge| edge.dir == dir) {",
    "if tree.node(root_id).edges.iter().any(|edge| edge.dir == dir) {",
    1,
)
block = block.replace(
    "        simulate(&mut root, cfg, rng);",
    "        simulate(&mut tree, root_id, cfg, rng);",
    1,
)
block = block.replace("snapshot_root(&root)", "snapshot_root(tree.node(root_id))")

if "root.edges" in block or "&mut root" in block or "snapshot_root(&root)" in block:
    raise SystemExit("unconverted root ownership remains in diagnostic search")

s = s[:search_start] + block + s[search_end:]

# A small structural test catches accidental ID invalidation if Tree internals
# are changed later.
test_anchor = '''    #[test]\n    fn merge_once_per_tile() {\n'''
test = r'''    #[test]
    fn arena_node_ids_survive_vec_growth() {
        let root_board = Board::empty();
        let mut tree = Tree::with_capacity(root_board, 1);
        let root = Tree::ROOT;
        for _ in 0..1024 {
            tree.alloc(Node::new(root_board));
        }
        assert_eq!(tree.node(root)._board, root_board);
    }

'''
if test_anchor in s and "fn arena_node_ids_survive_vec_growth" not in s:
    s = s.replace(test_anchor, test + test_anchor, 1)

main.write_text(s)
print("patched MCTS tree to append-only Vec<Node> arena")
