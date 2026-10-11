//! Headless bots-vs-bots soak test ("Схватка" without a human).
//!
//! cargo run --release -p ai --bin botmatch -- --minutes 10 --bots 9 --seed 1 [--bh] [--verify 3600] [--log out/botmatch.log]
//!
//! Runs the deterministic 60 Hz simulation as fast as possible, the bots
//! sending ordinary Commands; logs per-minute statistics and a final table.

use ai::config;
use gameplay::sim::Sim;
use gameplay::{Arena, Event, GameConfig, TICK_HZ};
use std::fs::{self, File};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

struct FileLogger {
    file: Mutex<File>,
}

impl log::Log for FileLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            let line = format!("[{}] {}: {}\n", r.level(), r.target(), r.args());
            if let Ok(mut f) = self.file.lock() {
                let _ = f.write_all(line.as_bytes());
            }
            if r.level() <= log::Level::Warn {
                eprint!("{line}");
            }
        }
    }
    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
    }
}

struct Args {
    minutes: f64,
    bots: usize,
    seed: u64,
    bh: bool,
    verify: u32,
    log: String,
    levels: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args { minutes: 10.0, bots: 9, seed: 1, bh: false, verify: 0, log: "out/botmatch.log".into(), levels: None };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let next = |i: usize| v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--minutes" => a.minutes = next(i).parse().expect("--minutes <f64>"),
            "--bots" => a.bots = next(i).parse().expect("--bots <n>"),
            "--seed" => a.seed = next(i).parse().expect("--seed <u64>"),
            "--verify" => a.verify = next(i).parse().expect("--verify <ticks>"),
            "--log" => a.log = next(i),
            "--levels" => a.levels = Some(next(i)),
            "--bh" => {
                a.bh = true;
                i += 1;
                continue;
            }
            "-h" | "--help" => {
                println!("botmatch [--minutes 10] [--bots 9] [--seed 1] [--bh] [--verify ticks] [--log out/botmatch.log] [--levels novice,regular,veteran]");
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    a
}

fn make(a: &Args, minutes: f64) -> (Sim, ai::Bots) {
    let mut cfg = GameConfig::builtin();
    // soak test: one long match
    cfg.rules.time_limit_s = minutes * 60.0 + 10.0;
    cfg.rules.frag_limit = u32::MAX;
    let mut sim = Sim::skirmish(cfg, Arena::prototype(a.seed), a.seed, false, a.bots);
    if a.bh {
        sim.attach_black_hole(4.3e6, 40.0);
    }
    let all = config::builtin();
    let levels: Vec<config::Difficulty> = match &a.levels {
        Some(s) => s.split(',').map(|n| config::by_name(&all, n.trim()).unwrap_or_else(|| panic!("unknown level {n}"))).collect(),
        None => all,
    };
    let bots = ai::Bots::new(&sim, &levels);
    (sim, bots)
}

fn main() {
    let a = parse_args();
    if let Some(dir) = std::path::Path::new(&a.log).parent() {
        let _ = fs::create_dir_all(dir);
    }
    let file = File::create(&a.log).expect("log file");
    log::set_logger(Box::leak(Box::new(FileLogger { file: Mutex::new(file) }))).expect("logger");
    log::set_max_level(log::LevelFilter::Info);

    let t_build = Instant::now();
    let (mut sim, mut bots) = make(&a, a.minutes);
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    let ticks = (a.minutes * 60.0 * TICK_HZ as f64) as u32;
    log::info!(
        "botmatch: {} bots, {} min ({} ticks), seed {}, BH {}, arena '{}', nav open {:.1} %, setup {:.0} ms",
        a.bots,
        a.minutes,
        ticks,
        a.seed,
        a.bh,
        sim.arena.st.name,
        100.0 * bots.nav.open_fraction(),
        build_ms
    );
    println!("botmatch: {} bots × {} min, seed {} — log: {}", a.bots, a.minutes, a.seed, a.log);

    let mut t_sim = 0.0f64;
    let mut t_ai = 0.0f64;
    let mut worst_tick = 0.0f64;
    let mut minute_acc = (0.0f64, 0.0f64);
    let mut kills_by_cause = std::collections::BTreeMap::<String, u32>::new();
    let mut device_failures = std::collections::BTreeMap::<&'static str, u32>::new();
    let (mut panes, mut boxes) = (0u32, 0u32);
    let wall = Instant::now();
    for _ in 0..ticks {
        let t0 = Instant::now();
        let cmds = bots.think(&sim);
        let t1 = Instant::now();
        sim.step(&cmds);
        let t2 = Instant::now();
        let (ai_s, sim_s) = ((t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64());
        t_ai += ai_s;
        t_sim += sim_s;
        minute_acc.0 += ai_s;
        minute_acc.1 += sim_s;
        worst_tick = worst_tick.max(ai_s + sim_s);
        for ev in &sim.events {
            match ev {
                Event::Kill { killer, victim, cause } => {
                    *kills_by_cause.entry(format!("{cause:?}")).or_default() += 1;
                    let name = |e: gameplay::Entity| sim.ecs.pilots.get(e).map_or("?".to_string(), |p| p.name.clone());
                    log::info!("t={:7.2}s kill: {} → {} ({:?})", sim.tick as f64 / TICK_HZ as f64, killer.map_or("—".into(), name), name(*victim), cause);
                }
                Event::DeviceFailure { owner, mode, what } => {
                    *device_failures.entry(what).or_default() += 1;
                    log::info!("t={:7.2}s device failure: {:?} {:?}: {}", sim.tick as f64 / TICK_HZ as f64, owner, mode, what);
                }
                Event::ContainmentLost { owner } => log::warn!("t={:7.2}s micro-hole containment lost ({owner:?})", sim.tick as f64 / TICK_HZ as f64),
                Event::PaneBroken { .. } => panes += 1,
                Event::BoxDestroyed { .. } => boxes += 1,
                _ => {}
            }
        }
        if sim.tick % (60 * TICK_HZ) == 0 {
            let n = (60 * TICK_HZ) as f64;
            let (k, sh, hi): (u32, u32, u32) = sim.ecs.stats.values().iter().fold((0, 0, 0), |a, s| (a.0 + s.kills, a.1 + s.shots, a.2 + s.hits));
            let line = format!(
                "minute {:2}: kills {:4} shots {:6} hits {:5} | projectiles {:3} debris {:3} fields {} | air {:.4} atm, panes broken {} boxes destroyed {} | ai {:.3} ms/tick, sim {:.3} ms/tick",
                sim.tick / (60 * TICK_HZ),
                k,
                sh,
                hi,
                sim.ecs.projectiles.len(),
                sim.ecs.debris.len(),
                sim.ecs.fields.len(),
                sim.atmo.frac,
                panes,
                boxes,
                minute_acc.0 / n * 1e3,
                minute_acc.1 / n * 1e3
            );
            log::info!("{line}");
            println!("{line}");
            minute_acc = (0.0, 0.0);
        }
    }
    let wall_s = wall.elapsed().as_secs_f64();
    let hash = sim.state_hash();

    // ---- final table ----
    let mut table = String::new();
    table += &format!(
        "\n{:<10} {:<8} {:>5} {:>6} {:>5} {:>6} {:>6} {:>6} {:>4} {:>7} {:>5} {:>5} {:>4}\n",
        "bot", "level", "kills", "deaths", "suic", "shots", "hits", "acc%", "hs", "dmg", "dev", "fail", "env"
    );
    let mut by_level = std::collections::BTreeMap::<String, (u32, u32, u32, u32)>::new();
    for b in &bots.bots {
        let p = sim.ecs.pilots.get(b.e).unwrap();
        let s = sim.ecs.stats.get(b.e).unwrap();
        let acc = if s.shots > 0 { 100.0 * s.hits as f64 / s.shots as f64 } else { 0.0 };
        table += &format!(
            "{:<10} {:<8} {:>5} {:>6} {:>5} {:>6} {:>6} {:>6.1} {:>4} {:>7.0} {:>5} {:>5} {:>4}\n",
            p.name,
            b.diff.name,
            s.kills,
            s.deaths,
            s.suicides,
            s.shots,
            s.hits,
            acc,
            s.headshots,
            s.damage_dealt,
            s.device_uses,
            s.device_failures,
            s.env_deaths
        );
        let e = by_level.entry(b.diff.name.clone()).or_default();
        e.0 += s.kills;
        e.1 += s.deaths;
        e.2 += s.shots;
        e.3 += s.hits;
    }
    table += "\nby level:\n";
    for (l, (k, d, sh, hi)) in &by_level {
        table += &format!("  {:<8} kills {:4} deaths {:4} accuracy {:5.1} %\n", l, k, d, if *sh > 0 { 100.0 * *hi as f64 / *sh as f64 } else { 0.0 });
    }
    table += &format!("\ndeaths by cause: {kills_by_cause:?}\ndevice failures: {device_failures:?}\n");
    let mut goals = [0u32; 6];
    let (mut replans, mut fails, mut stuck, mut jets, mut vaults) = (0, 0, 0, 0, 0);
    for b in &bots.bots {
        for (g, n) in b.stats.decisions.iter().enumerate() {
            goals[g] += n;
        }
        replans += b.stats.replans;
        fails += b.stats.path_fail;
        stuck += b.stats.stuck;
        jets += b.stats.jet_recoveries;
        vaults += b.stats.vaults;
    }
    table += "goal switches: ";
    for g in ai::brain::Goal::ALL {
        table += &format!("{} {}  ", g.label(), goals[g as usize]);
    }
    table += &format!("\nnav: replans {replans}, failed {fails}, stuck events {stuck}, vaults {vaults}, jetpack recoveries {jets}\n");
    table += &format!(
        "\nsimulated {:.1} min in {:.1} s wall ({:.0}× real time); per tick: sim {:.3} ms, ai {:.3} ms, worst {:.2} ms\nfinal tick {} state hash {:016x}\n",
        ticks as f64 / TICK_HZ as f64 / 60.0,
        wall_s,
        ticks as f64 / TICK_HZ as f64 / wall_s,
        t_sim / ticks as f64 * 1e3,
        t_ai / ticks as f64 * 1e3,
        worst_tick * 1e3,
        sim.tick,
        hash
    );
    if a.verify > 0 {
        // determinism: rerun the first `verify` ticks from scratch (bots included) and compare
        let (mut s1, mut b1) = make(&a, a.minutes);
        let (mut s2, mut b2) = make(&a, a.minutes);
        for _ in 0..a.verify {
            let c1 = b1.think(&s1);
            s1.step(&c1);
            let c2 = b2.think(&s2);
            s2.step(&c2);
        }
        let (h1, h2) = (s1.state_hash(), s2.state_hash());
        table += &format!("determinism check over {} ticks: {:016x} vs {:016x} → {}\n", a.verify, h1, h2, if h1 == h2 { "OK" } else { "MISMATCH" });
        if h1 != h2 {
            log::error!("determinism mismatch");
        }
    }
    log::info!("{table}");
    println!("{table}");
    log::logger().flush();
}
