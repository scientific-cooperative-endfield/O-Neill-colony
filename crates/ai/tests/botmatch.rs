//! Bots-vs-bots in the prototype district: deterministic and lively.

use gameplay::sim::Sim;
use gameplay::{Arena, GameConfig};

fn run(seed: u64, ticks: u32, bh: bool) -> (u64, u32) {
    let mut sim = Sim::skirmish(GameConfig::builtin(), Arena::prototype(seed), seed, false, 9);
    if bh {
        sim.attach_black_hole(4.3e6, 40.0);
    }
    let mut bots = ai::Bots::new(&sim, &ai::config::builtin());
    for _ in 0..ticks {
        let c = bots.think(&sim);
        sim.step(&c);
    }
    let shots = sim.ecs.stats.values().iter().map(|s| s.shots).sum();
    (sim.state_hash(), shots)
}

#[test]
fn bot_match_is_deterministic() {
    let a = run(5, 2400, false);
    let b = run(5, 2400, false);
    assert_eq!(a.0, b.0, "same seed → same state hash");
    let c = run(6, 2400, false);
    assert_ne!(a.0, c.0);
}

#[test]
fn bots_fight_near_a_black_hole_too() {
    let (h1, _) = run(7, 1800, true);
    let (h2, _) = run(7, 1800, true);
    assert_eq!(h1, h2);
}
