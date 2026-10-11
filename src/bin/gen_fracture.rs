//! Offline tool: generates assets/fracture.bin (3-level hierarchical Voronoi).
//! `cargo run --release --bin gen_fracture -- [seed]`
use oneill_colony::bh::fracture;

fn main() {
    let seed = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(fracture::DEFAULT_SEED);
    let a = fracture::generate(12, 3, seed);
    std::fs::create_dir_all("assets").unwrap();
    let b = fracture::save(&a);
    std::fs::write("assets/fracture.bin", &b).unwrap();
    println!("assets/fracture.bin: {} pieces, {} bytes", a.pieces.len(), b.len());
}
