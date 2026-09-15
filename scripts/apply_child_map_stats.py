from pathlib import Path

path = Path("src/main.rs")
s = path.read_text()

old = "struct Tree {\n    nodes: Vec<Node>,\n}"
new = """struct Tree {
    nodes: Vec<Node>,
    // Instrumentation used only when BAYES2048_CHILD_STATS is set. Bucket 30
    // includes all child-map sizes >= 30.
    child_lookup_hist: [u64; 31],
}"""
assert old in s
s = s.replace(old, new, 1)

old = """        nodes.push(Node::new(root));
        Self { nodes }
    }"""
new = """        nodes.push(Node::new(root));
        Self {
            nodes,
            child_lookup_hist: [0; 31],
        }
    }"""
assert old in s
s = s.replace(old, new, 1)

old = """    let child_id = tree.node(node_id).edges[edge_i]
        .children
        .get(&spawned)
        .copied();"""
new = """    let child_count = tree.node(node_id).edges[edge_i].children.len().min(30);
    tree.child_lookup_hist[child_count] += 1;
    let child_id = tree.node(node_id).edges[edge_i]
        .children
        .get(&spawned)
        .copied();"""
assert old in s
s = s.replace(old, new, 1)

marker = "fn choose_move<R: Rng + ?Sized>("
assert marker in s
helper = r'''fn maybe_record_child_stats(tree: &Tree, policy: Policy, simulations: usize) {
    let Ok(path) = std::env::var("BAYES2048_CHILD_STATS") else {
        return;
    };

    let mut map_hist = [0u64; 31];
    let mut edges = 0u64;
    let mut total_children = 0u64;
    let mut max_children = 0usize;
    for node in &tree.nodes {
        for edge in &node.edges {
            let n = edge.children.len();
            edges += 1;
            total_children += n as u64;
            max_children = max_children.max(n);
            map_hist[n.min(30)] += 1;
        }
    }

    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open child stats output");
    write!(
        out,
        "{},{},{},{},{},{},{}",
        policy.name(),
        simulations,
        tree.nodes.len(),
        edges,
        total_children,
        max_children,
        tree.child_lookup_hist.iter().sum::<u64>(),
    )
    .expect("write child stats prefix");
    for value in tree.child_lookup_hist {
        write!(out, ",{value}").expect("write lookup histogram");
    }
    for value in map_hist {
        write!(out, ",{value}").expect("write map histogram");
    }
    writeln!(out).expect("finish child stats row");
}

'''
s = s.replace(marker, helper + marker, 1)

old = """    for _ in 0..simulations {
        simulate(&mut tree, root_id, cfg, rng);
    }

    // Terminal Bayes action: maximize posterior expected return."""
new = """    for _ in 0..simulations {
        simulate(&mut tree, root_id, cfg, rng);
    }

    maybe_record_child_stats(&tree, cfg.policy, simulations);

    // Terminal Bayes action: maximize posterior expected return."""
assert old in s
s = s.replace(old, new, 1)

path.write_text(s)
