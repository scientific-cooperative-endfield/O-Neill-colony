//! Offline tool: bakes the HDR star / Milky Way cubemap into assets/sky_<N>.bin.
//! `cargo run --release --bin gen_sky -- [size] [seed]` (default 1024, fixed seed).
use oneill_colony::bh::sky;

fn main() {
    let mut a = std::env::args().skip(1);
    let size: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(sky::DEFAULT_SEED);
    let t = std::time::Instant::now();
    let s = sky::generate(size, seed);
    std::fs::create_dir_all("assets").expect("assets dir");
    let b = sky::save(&s);
    let path = sky::path_for(size);
    std::fs::write(&path, &b).expect("write sky");
    println!("{path}: {size}² × 6 faces, {} mips, {} MB, {:.1} s", s.mips.len(), b.len() / 1_000_000, t.elapsed().as_secs_f32());
}
