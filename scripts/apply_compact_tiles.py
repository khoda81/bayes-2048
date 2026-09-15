#!/usr/bin/env python3
from pathlib import Path

main = Path("src/main.rs")
s = main.read_text()

marker = '''fn print_live_trace(\n'''
helper = r'''fn tile_symbol(exp: u8) -> char {
    match exp {
        0 => '.',
        1..=9 => (b'0' + exp) as char,
        10..=35 => (b'a' + (exp - 10)) as char,
        _ => '?',
    }
}

fn print_compact_board(board: Board) {
    for r in 0..4 {
        for c in 0..4 {
            if c != 0 {
                print!(" ");
            }
            print!("{}", tile_symbol(board.0[4 * r + c]));
        }
        println!();
    }
}

'''
if "fn tile_symbol(exp: u8)" not in s:
    s = s.replace(marker, helper + marker)

old_before = '''    println!("before:");
    let vals = board_values(before);
    for r in 0..4 {
        for c in 0..4 {
            let v = vals[4 * r + c];
            if v == 0 {
                print!("{:>6}", ".");
            } else {
                print!("{:>6}", v);
            }
        }
        println!();
    }
'''
new_before = '''    println!("before:");
    print_compact_board(before);
'''
s = s.replace(old_before, new_before)

old_after = '''    println!("\\nafter spawn:");
    let vals = board_values(after);
    for r in 0..4 {
        for c in 0..4 {
            let v = vals[4 * r + c];
            if v == 0 {
                print!("{:>6}", ".");
            } else {
                print!("{:>6}", v);
            }
        }
        println!();
    }
'''
new_after = '''    println!("\\nafter spawn:");
    print_compact_board(after);
'''
s = s.replace(old_after, new_after)

main.write_text(s)
print("patched live renderer to compact exponent tiles")
