//! Acceptance: identical command streams → identical state hash after N ticks;
//! rollback (restore a snapshot and re-simulate) reproduces the same state.

use gameplay::det::Rng;
use gameplay::{buttons, Arena, Command, GameConfig, Sim};

fn scripted(tick: u32, player: u8, salt: u64) -> Command {
    let mut r = Rng::keyed(&[salt, tick as u64 / 20, player as u64]);
    let mut c = Command::new(tick, player);
    c.set_move(r.range(-1.0, 1.0), r.range(-1.0, 1.0), 0.0);
    let mut j = Rng::keyed(&[salt, tick as u64, player as u64]);
    c.set_look(j.range(-0.02, 0.02), j.range(-0.01, 0.01), 0.0);
    let mut b = 0;
    if r.f64() < 0.5 {
        b |= buttons::FIRE | buttons::ADS;
    }
    if r.f64() < 0.3 {
        b |= buttons::RUN;
    }
    if j.f64() < 0.02 {
        b |= buttons::JUMP;
    }
    if tick % 600 == 300 {
        b |= buttons::DEVICE;
    }
    if tick % 900 == 450 {
        b |= buttons::DEVICE_NEXT;
    }
    c.buttons = b;
    if tick.is_multiple_of(400) {
        c.weapon = 1 + (r.index(3) as u8);
    }
    c
}

fn new_sim() -> Sim {
    let mut s = Sim::skirmish(GameConfig::builtin(), Arena::prototype(11), 42, true, 5);
    // pack everyone into one street so they really fight
    let (u, v) = s.arena.st.spawns[0];
    for (i, &e) in s.players.clone().iter().enumerate() {
        let b = s.ecs.bodies.get_mut(e).unwrap();
        b.pos = gameplay::geom::surf_to_world(u + 3.0 * i as f64, v + 1.0, 0.02);
    }
    s
}

fn run(s: &mut Sim, from: u32, to: u32, salt: u64) {
    for t in from..to {
        let cmds: Vec<Command> = (0..s.players.len() as u8).map(|p| scripted(t, p, salt)).collect();
        s.step(&cmds);
    }
}

#[test]
fn same_commands_same_hash() {
    let n = 1800;
    let mut a = new_sim();
    let mut b = new_sim();
    run(&mut a, 0, n, 1);
    run(&mut b, 0, n, 1);
    assert_eq!(a.state_hash(), b.state_hash());
    let shots: u32 = a.ecs.stats.values().iter().map(|s| s.shots).sum();
    assert!(shots > 50, "the scripted match should actually shoot ({shots})");
    let mut c = new_sim();
    run(&mut c, 0, n, 2);
    assert_ne!(a.state_hash(), c.state_hash());
}

#[test]
fn rollback_resimulation_is_exact() {
    let mut a = new_sim();
    run(&mut a, 0, 700, 3);
    let snap = a.clone();
    let h_snap = snap.state_hash();
    run(&mut a, 700, 1400, 3);
    let h_end = a.state_hash();
    // roll back, mispredict, roll back again and replay the true inputs
    let mut b = snap.clone();
    run(&mut b, 700, 1000, 99);
    assert_ne!(b.state_hash(), h_snap);
    let mut c = snap;
    assert_eq!(c.state_hash(), h_snap);
    run(&mut c, 700, 1400, 3);
    assert_eq!(c.state_hash(), h_end);
}

#[test]
fn black_hole_mode_is_deterministic_too() {
    let mk = || {
        let mut s = new_sim();
        s.attach_black_hole(4.3e6, 30.0);
        s
    };
    let mut a = mk();
    let mut b = mk();
    run(&mut a, 0, 300, 5);
    run(&mut b, 0, 300, 5);
    assert_eq!(a.state_hash(), b.state_hash());
    assert!(a.frame.bh.is_some());
}
