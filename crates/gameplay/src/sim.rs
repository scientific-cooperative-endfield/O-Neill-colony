//! The deterministic fixed-tick simulation (60 Hz).
//!
//! `Sim::step(&commands)` advances exactly one tick. The state is a plain
//! `Clone` value (ECS storages + destructible arena state + optional black-hole
//! structure), so rollback is `let snap = sim.clone(); … *sim = snap;` and the
//! state hash (`state_hash`) folds every bit of gameplay state.

use crate::ballistics::{self, Env, Round};
use crate::components::*;
use crate::config::GameConfig;
use crate::damage;
use crate::det::{self, Rng, StateHasher};
use crate::ecs::{Entities, Entity, Storage};
use crate::fields::{self, shell_response, Crossing, Field, FieldKind, FieldSet};
use crate::frame::{Atmosphere, BhField, StationFrame};
use crate::geom::{self, C_LIGHT};
use crate::input::{buttons, Command};
use crate::map::{Arena, HitTarget, Material};
use crate::player::{self, Ctx};
use glam::{DMat3, DVec3};
use std::sync::Arc;

pub const TICK_HZ: u32 = 60;
pub const DT: f64 = 1.0 / TICK_HZ as f64;
pub const MAX_DEBRIS: usize = 384;
pub const MOA: f64 = std::f64::consts::PI / (180.0 * 60.0);

#[derive(Clone, Default)]
pub struct Ecs {
    pub entities: Entities,
    pub pilots: Storage<Pilot>,
    pub bodies: Storage<Body>,
    pub health: Storage<Health>,
    pub suits: Storage<Suit>,
    pub clocks: Storage<ProperTime>,
    pub arsenals: Storage<Arsenal>,
    pub devices: Storage<Device>,
    pub stats: Storage<Stats>,
    pub projectiles: Storage<Projectile>,
    pub fields: Storage<Field>,
    pub debris: Storage<Debris>,
}

impl Ecs {
    pub fn despawn(&mut self, e: Entity) {
        self.pilots.remove(e);
        self.bodies.remove(e);
        self.health.remove(e);
        self.suits.remove(e);
        self.clocks.remove(e);
        self.arsenals.remove(e);
        self.devices.remove(e);
        self.stats.remove(e);
        self.projectiles.remove(e);
        self.fields.remove(e);
        self.debris.remove(e);
        self.entities.despawn(e);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Shot { shooter: Entity, weapon: u8 },
    Hit { shooter: Entity, victim: Entity, damage: f64, head: bool, distance: f64 },
    Kill { killer: Option<Entity>, victim: Entity, cause: DeathCause },
    Respawn { who: Entity },
    PaneBroken { key: (i32, i32), by: Entity },
    BoxDestroyed { index: u32, material: Material },
    DeviceUsed { owner: Entity, mode: DeviceMode },
    DeviceFailure { owner: Entity, mode: DeviceMode, what: &'static str },
    ContainmentLost { owner: Entity },
    FieldEnded { owner: Entity, kind: &'static str },
    MatchOver { winner: Option<Entity> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoundKind {
    Shot,
    Step,
    Impact,
    Break,
    Device,
}

/// A sound emitted at `tick`; listeners hear it when the wavefront (343 m/s)
/// reaches them — and only if there is air at both ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sound {
    pub pos: DVec3,
    pub tick: u32,
    /// level at 1 m (dB SPL)
    pub db: f64,
    pub source: Entity,
    pub kind: SoundKind,
    pub in_air: bool,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MatchState {
    pub time: f64,
    pub over: bool,
    pub winner: Option<Entity>,
}

/// Optional coupling to the black-hole structure simulation: the station
/// free-falls on its PG geodesic, its fragments define our rotating frame,
/// shot-out windows puncture its panels and its pressure is our air.
#[derive(Clone)]
pub struct BhLink {
    /// GM/c² (m)
    pub m: f64,
    pub station: structure::Station,
    pub frag: usize,
}

/// Structure panel under a colony-frame point (f64 port of `bhlink::node_of`).
pub fn node_of(p: DVec3) -> usize {
    use structure::{N_CAP, N_THETA, N_Z};
    let s = geom::world_to_surf(p);
    let th = s.u / geom::R;
    if (p.z < 2.0 || p.z > geom::L - 2.0) && geom::r_perp(p) < geom::R - 2.0 {
        let k = ((th / (std::f64::consts::TAU / N_CAP as f64)) as usize).min(N_CAP - 1);
        return N_THETA * N_Z + if p.z > 0.5 * geom::L { N_CAP } else { 0 } + k;
    }
    let iz = (p.z / (geom::L / N_Z as f64)).floor().clamp(0.0, (N_Z - 1) as f64) as usize;
    let j = ((th / (std::f64::consts::TAU / N_THETA as f64)) as usize).min(N_THETA - 1);
    iz * N_THETA + j
}

#[derive(Clone)]
pub struct Sim {
    pub tick: u32,
    pub seed: u64,
    pub cfg: Arc<GameConfig>,
    pub rounds: Vec<Round>,
    pub ecs: Ecs,
    pub arena: Arena,
    pub frame: StationFrame,
    pub atmo: Atmosphere,
    /// snapshot of the active spacetime fields for this tick
    pub fields: FieldSet,
    pub game: MatchState,
    pub events: Vec<Event>,
    pub sounds: Vec<Sound>,
    /// player id → entity
    pub players: Vec<Entity>,
    pub bh: Option<BhLink>,
    /// Voronoi debris template (centroid in the unit cell, volume fraction)
    fracture: Arc<Vec<(DVec3, f64)>>,
    scratch: Vec<u32>,
}

impl Sim {
    pub fn new(cfg: GameConfig, arena: Arena, seed: u64) -> Sim {
        let rounds = cfg.weapons.iter().map(Round::from_cfg).collect();
        let asset = structure::fracture::generate(12, 1, structure::fracture::DEFAULT_SEED);
        let pieces = asset
            .pieces
            .iter()
            .filter(|p| p.level == 1)
            .map(|p| (DVec3::new(p.centroid[0] as f64, p.centroid[1] as f64, p.centroid[2] as f64), p.volume as f64))
            .collect();
        Sim {
            tick: 0,
            seed,
            cfg: Arc::new(cfg),
            rounds,
            ecs: Ecs::default(),
            arena,
            frame: StationFrame::colony(),
            atmo: Atmosphere::default(),
            fields: FieldSet::default(),
            game: MatchState::default(),
            events: Vec::new(),
            sounds: Vec::new(),
            players: Vec::new(),
            bh: None,
            fracture: Arc::new(pieces),
            scratch: Vec::new(),
        }
    }

    /// Skirmish ("Схватка"): `n_bots` bots plus, optionally, one human (player 0).
    pub fn skirmish(cfg: GameConfig, arena: Arena, seed: u64, human: bool, n_bots: usize) -> Sim {
        let mut s = Sim::new(cfg, arena, seed);
        if human {
            s.add_player("Игрок", 0, false);
        }
        for i in 0..n_bots {
            let id = s.players.len() as u8;
            s.add_player(&format!("Бот-{}", i + 1), id, true);
        }
        s
    }

    /// Couples the match to a free-falling station near a black hole of
    /// `solar` masses, starting at `r_m` (units of M) on a "rain" geodesic.
    pub fn attach_black_hole(&mut self, solar: f64, r_m: f64) {
        let m = relativity::m_geo(solar);
        let body = relativity::Geo::rain(DVec3::new(r_m, 0.0, 0.0));
        let station = structure::Station::new(structure::Spec::default(), body, DVec3::new(0.0, 0.0, 1.0));
        self.bh = Some(BhLink { m, station, frag: 0 });
        self.sync_bh();
    }

    pub fn add_player(&mut self, name: &str, team: u8, bot: bool) -> u8 {
        let cfg = self.cfg.clone();
        let e = self.ecs.entities.spawn();
        let id = self.players.len() as u8;
        self.players.push(e);
        let p = &cfg.player;
        self.ecs.pilots.insert(e, Pilot { id, name: name.to_string(), team, bot, prev_buttons: 0, last_cmd: Command::new(0, id), neuro: false });
        self.ecs.bodies.insert(e, player::new_body(p, DVec3::ZERO, 0.0));
        self.ecs.health.insert(
            e,
            Health {
                hp: p.max_hp,
                max: p.max_hp,
                alive: true,
                respawn_in: 0.0,
                protect: 0.0,
                last_attacker: None,
                last_attack_tick: 0,
                dose_gy: 0.0,
                cause: None,
                recent_damage: 0.0,
            },
        );
        self.ecs.suits.insert(e, Suit { o2_s: p.suit_o2_s, integrity: 1.0, sealed: false, propellant_kg: p.jetpack.propellant_kg, p_prev: self.atmo.p_floor0 });
        self.ecs.clocks.insert(e, ProperTime::default());
        self.ecs.arsenals.insert(e, self.fresh_arsenal());
        self.ecs.devices.insert(
            e,
            Device { mode: DeviceMode::Bubble, charge_j: cfg.devices.capacity_j, instability: 0.0, cooldown: 0.0, active: None, uses: 0, failures: 0 },
        );
        self.ecs.stats.insert(e, Stats::default());
        self.respawn(e);
        id
    }

    fn fresh_arsenal(&self) -> Arsenal {
        Arsenal {
            slots: self.cfg.weapons.iter().map(|w| WeaponState { mag: w.magazine, reserve: w.reserve, cooldown: 0.0, reload: 0.0, shots: 0 }).collect(),
            current: 1.min(self.cfg.weapons.len() - 1),
            recoil: 0.0,
            switching: 0.0,
        }
    }

    pub fn entity(&self, id: u8) -> Option<Entity> {
        self.players.get(id as usize).copied()
    }
    pub fn is_alive(&self, e: Entity) -> bool {
        self.ecs.health.get(e).is_some_and(|h| h.alive)
    }
    pub fn team(&self, e: Entity) -> u8 {
        self.ecs.pilots.get(e).map_or(255, |p| p.team)
    }
    pub fn eye(&self, e: Entity) -> DVec3 {
        let b = self.ecs.bodies.get(e).expect("body");
        player::eye(b, &self.cfg.player)
    }
    /// Current view direction including muzzle climb.
    pub fn aim_dir(&self, e: Entity) -> DVec3 {
        let b = self.ecs.bodies.get(e).expect("body");
        let climb = self.ecs.arsenals.get(e).map_or(0.0, |a| a.recoil);
        player::forward(b, climb)
    }
    pub fn env(&self) -> Env<'_> {
        Env::new(&self.frame, &self.atmo, &self.fields).with_arena(&self.arena)
    }
    pub fn round(&self, weapon: usize) -> &Round {
        &self.rounds[weapon]
    }

    // ------------------------------------------------------------- tick --

    pub fn step(&mut self, cmds: &[Command]) {
        self.events.clear();
        self.rebuild_fields();
        let n = self.players.len();
        let mut by_player: Vec<Option<Command>> = vec![None; n];
        for c in cmds {
            if c.tick == self.tick && (c.player as usize) < n {
                by_player[c.player as usize] = Some(*c);
            }
        }
        for (id, cmd) in by_player.into_iter().enumerate() {
            let e = self.players[id];
            let cmd = cmd.unwrap_or_else(|| {
                // prediction for a missing command: keep buttons and movement, no look change
                let mut c = self.ecs.pilots.get(e).map(|p| p.last_cmd).unwrap_or_default();
                c.tick = self.tick;
                c.look = [0; 3];
                c.weapon = 0;
                c
            });
            self.player_tick(e, cmd);
        }
        self.rebuild_fields();
        self.step_projectiles();
        self.step_fields();
        self.step_debris();
        self.step_environment();
        self.step_air_and_structure();
        self.step_rules();
        let t = self.tick;
        self.sounds.retain(|s| t.saturating_sub(s.tick) < 6 * TICK_HZ);
        self.tick += 1;
    }

    fn rebuild_fields(&mut self) {
        self.fields.items.clear();
        self.fields.items.extend(self.ecs.fields.values().iter().copied());
    }

    fn sound(&mut self, pos: DVec3, db: f64, source: Entity, kind: SoundKind) {
        let in_air = self.atmo.density(&self.frame, pos) > 0.01;
        self.sounds.push(Sound { pos, tick: self.tick, db, source, kind, in_air });
    }

    fn player_tick(&mut self, e: Entity, cmd: Command) {
        let cfg = self.cfg.clone();
        let prev = self.ecs.pilots.get(e).map_or(0, |p| p.prev_buttons);
        let pressed = |b: u16| cmd.held(b) && prev & b == 0;
        let alive = self.is_alive(e);
        if let Some(p) = self.ecs.pilots.get_mut(e) {
            p.prev_buttons = cmd.buttons;
            p.last_cmd = cmd;
            if pressed(buttons::NEURO) {
                p.neuro = !p.neuro;
            }
        }
        if !alive {
            let h = self.ecs.health.get_mut(e).unwrap();
            h.respawn_in -= DT;
            if h.respawn_in <= 0.0 && !self.game.over {
                self.respawn(e);
            }
            return;
        }
        let rate = self.ecs.clocks.get(e).map_or(1.0, |c| c.dyn_rate);
        let dt = DT * rate;
        if pressed(buttons::DEVICE_NEXT) {
            if let Some(d) = self.ecs.devices.get_mut(e) {
                d.mode = d.mode.next();
            }
        }
        if cmd.weapon > 0 && (cmd.weapon as usize) <= cfg.weapons.len() {
            let a = self.ecs.arsenals.get_mut(e).unwrap();
            let w = cmd.weapon as usize - 1;
            if w != a.current {
                a.current = w;
                a.switching = 0.6;
                a.slots[w].reload = 0.0;
            }
        }
        // movement
        let mut body = self.ecs.bodies.get(e).unwrap().clone();
        let mut suit = self.ecs.suits.get(e).unwrap().clone();
        let out = {
            let ctx = Ctx { cfg: &cfg.player, frame: &self.frame, atmo: &self.atmo, arena: &self.arena, fields: &self.fields };
            player::step(&mut body, &mut suit, &cmd, prev, &ctx, dt, &mut self.scratch)
        };
        if out.footstep {
            let db = if body.crouched {
                40.0
            } else if cmd.held(buttons::RUN) {
                72.0
            } else {
                58.0
            };
            self.sound(body.pos, db, e, SoundKind::Step);
        }
        let pos = body.pos;
        self.ecs.bodies.insert(e, body);
        self.ecs.suits.insert(e, suit);
        if out.impact > 0.0 {
            let d = damage::impact(out.impact, &cfg.damage);
            if d > 0.0 {
                self.sound(pos, 80.0 + out.impact, e, SoundKind::Impact);
                self.hurt(e, d, None, DeathCause::Impact, false);
            }
        }
        if let Some(stab) = out.crossed_fold {
            // throat tide during the traversal
            let d = (25.0 / stab.max(0.05) - 20.0).max(0.0);
            self.hurt(e, d, None, DeathCause::Tidal, false);
            self.destabilise_folds(cfg.player.mass_kg);
        }
        if !self.is_alive(e) {
            return;
        }
        self.weapons_tick(e, &cmd, prev, dt);
        let d = self.ecs.devices.get_mut(e).unwrap();
        d.cooldown = (d.cooldown - dt).max(0.0);
        d.charge_j = (d.charge_j + cfg.devices.recharge_w * dt).min(cfg.devices.capacity_j);
        d.instability *= det::exp(-cfg.devices.instability_decay * dt);
        if let Some(f) = d.active {
            if !self.ecs.fields.contains(f) {
                d.active = None;
            }
        }
        if pressed(buttons::DEVICE) {
            self.activate_device(e);
        }
    }

    // ---------------------------------------------------------- weapons --

    fn weapons_tick(&mut self, e: Entity, cmd: &Command, prev: u16, dt: f64) {
        let cfg = self.cfg.clone();
        let mut a = self.ecs.arsenals.get(e).unwrap().clone();
        let w = a.current;
        let wc = &cfg.weapons[w];
        a.recoil *= det::exp(-wc.recoil_recovery * dt);
        a.switching = (a.switching - dt).max(0.0);
        let ws = &mut a.slots[w];
        ws.cooldown = (ws.cooldown - dt).max(0.0);
        let pressed = |b: u16| cmd.held(b) && prev & b == 0;
        if pressed(buttons::RELOAD) && ws.mag < wc.magazine && ws.reserve > 0 && ws.reload <= 0.0 {
            ws.reload = wc.reload_s;
        }
        if ws.reload > 0.0 {
            ws.reload -= dt;
            if ws.reload <= 0.0 {
                let take = (wc.magazine - ws.mag).min(ws.reserve);
                ws.mag += take;
                ws.reserve -= take;
                ws.reload = 0.0;
            }
        }
        let trigger = if wc.automatic { cmd.held(buttons::FIRE) } else { pressed(buttons::FIRE) };
        let mut fire = None;
        if trigger && ws.cooldown <= 0.0 && ws.reload <= 0.0 && a.switching <= 0.0 {
            if ws.mag == 0 {
                if ws.reserve > 0 {
                    ws.reload = wc.reload_s;
                }
            } else {
                ws.mag -= 1;
                ws.cooldown += 60.0 / wc.rpm;
                ws.shots += 1;
                fire = Some(ws.shots);
            }
        }
        let climb = a.recoil;
        if fire.is_some() {
            a.recoil += wc.recoil_mrad * 1e-3;
        }
        self.ecs.arsenals.insert(e, a);
        let Some(shot) = fire else { return };
        let body = self.ecs.bodies.get(e).unwrap().clone();
        let dir = player::forward(&body, climb);
        let moving = body.vel.length();
        let ads = cmd.held(buttons::ADS);
        let sigma = if ads { wc.spread_moa_ads * (1.0 + moving / 4.0) } else { wc.spread_moa_hip * (1.0 + moving / 6.0) } * MOA;
        let mut rng = Rng::keyed(&[self.seed, self.tick as u64, e.bits(), shot as u64, 0x5807]);
        let a1 = dir.any_orthonormal_vector();
        let a2 = dir.cross(a1);
        let d = (dir + a1 * (sigma * rng.normal()) + a2 * (sigma * rng.normal())).normalize();
        let muzzle = player::eye(&body, &self.cfg.player) + d * 0.4;
        let vel = body.vel + d * wc.muzzle_velocity;
        let r = self.rounds[w];
        let p = self.ecs.entities.spawn();
        self.ecs
            .projectiles
            .insert(p, Projectile { pos: muzzle, vel, owner: e, weapon: w as u8, age: 0.0, e0: 0.5 * r.mass * wc.muzzle_velocity * wc.muzzle_velocity });
        // momentum conservation: the shooter recoils (matters in zero g)
        let m_tot = body.mass + self.ecs.suits.get(e).map_or(0.0, |s| s.propellant_kg);
        if let Some(b) = self.ecs.bodies.get_mut(e) {
            b.vel -= d * (r.mass * wc.muzzle_velocity / m_tot);
        }
        if let Some(s) = self.ecs.stats.get_mut(e) {
            s.shots += 1;
        }
        self.sound(muzzle, wc.loudness_db, e, SoundKind::Shot);
        self.events.push(Event::Shot { shooter: e, weapon: w as u8 });
    }

    /// First body capsule hit by p0 → p1 (fraction, entity, point, head).
    fn raycast_bodies(&self, p0: DVec3, p1: DVec3, skip: Option<Entity>) -> Option<(f64, Entity, DVec3, bool)> {
        let mut best: Option<(f64, Entity, DVec3, bool)> = None;
        for (e, b) in self.ecs.bodies.iter() {
            if Some(e) == skip || !self.ecs.health.get(e).is_some_and(|h| h.alive) {
                continue;
            }
            let ax = b.axis();
            let a = b.pos + ax * b.radius;
            let c = b.pos + ax * (b.height - b.radius);
            // quick reject
            let mid = 0.5 * (a + c);
            let reach = 0.5 * b.height + b.radius + (p1 - p0).length();
            if (mid - p0).length_squared() > reach * reach {
                continue;
            }
            let (s, _) = seg_seg(p0, p1, a, c);
            let dist = |t: f64| point_seg_dist(p0 + (p1 - p0) * t, a, c) - b.radius;
            if dist(s) > 0.0 {
                continue;
            }
            let t = if dist(0.0) <= 0.0 {
                0.0
            } else {
                let (mut lo, mut hi) = (0.0, s);
                for _ in 0..30 {
                    let m = 0.5 * (lo + hi);
                    if dist(m) > 0.0 {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                hi
            };
            if best.is_none_or(|x| t < x.0) {
                let q = p0 + (p1 - p0) * t;
                let along = (q - b.pos).dot(ax);
                best = Some((t, e, q, along > b.height - 0.28));
            }
        }
        best
    }

    fn step_projectiles(&mut self) {
        let cfg = self.cfg.clone();
        let list: Vec<Entity> = self.ecs.projectiles.entities().to_vec();
        for pe in list {
            let mut p = self.ecs.projectiles.get(pe).unwrap().clone();
            let wc = &cfg.weapons[p.weapon as usize];
            let round = self.rounds[p.weapon as usize];
            let rate = self.fields.time_rate(p.pos);
            let dt = DT * rate;
            let n = ballistics::substeps(p.vel.length(), dt);
            let h = dt / n as f64;
            let mut dead = false;
            for _ in 0..n {
                let (p1, v1) = {
                    let env = self.env();
                    env.rk4(&round, p.pos, p.vel, h)
                };
                let skip = if p.age < 0.05 { Some(p.owner) } else { None };
                let st = self.arena.raycast(p.pos, p1, &mut self.scratch);
                let bd = self.raycast_bodies(p.pos, p1, skip);
                let fx = self.fields.crossing(p.pos, p1);
                let ts = st.map_or(2.0, |x| x.t);
                let tb = bd.map_or(2.0, |x| x.0);
                let tf = fx.map_or(2.0, |x| x.0);
                let pv = p.vel;
                let vt = |t: f64| pv + (v1 - pv) * t;
                if tb <= ts && tb <= tf && tb <= 1.0 {
                    let (t, victim, q, head) = bd.unwrap();
                    let tv = self.ecs.bodies.get(victim).map_or(DVec3::ZERO, |b| b.vel);
                    let rel = vt(t) - tv;
                    let e_imp = 0.5 * round.mass * rel.length_squared();
                    let dmg = wc.damage * (e_imp / p.e0).sqrt() * if head { wc.head_mult } else { 1.0 };
                    let dist = (q - self.ecs.bodies.get(p.owner).map_or(q, |b| b.pos)).length();
                    if let Some(s) = self.ecs.suits.get_mut(victim) {
                        s.integrity = (s.integrity - 0.12).max(0.0);
                    }
                    // the body absorbs the bullet's momentum
                    if let Some(b) = self.ecs.bodies.get_mut(victim) {
                        b.vel += rel * (round.mass / b.mass);
                    }
                    if let Some(s) = self.ecs.stats.get_mut(p.owner) {
                        s.hits += 1;
                        s.headshots += head as u32;
                        s.damage_dealt += dmg;
                    }
                    self.events.push(Event::Hit { shooter: p.owner, victim, damage: dmg, head, distance: dist });
                    self.hurt(victim, dmg, Some(p.owner), DeathCause::Bullet(p.weapon), true);
                    dead = true;
                } else if ts <= tf && ts <= 1.0 {
                    let hit = st.unwrap();
                    let v = vt(hit.t);
                    let e_k = 0.5 * round.mass * v.length_squared();
                    let (loss, _) = hit.material.props();
                    match hit.target {
                        HitTarget::Box(i) if hit.material.destructible() => {
                            let dep = e_k.min(loss);
                            if self.arena.damage_box(i, dep) {
                                self.box_destroyed(i, v.normalize_or_zero());
                            }
                            if e_k > loss {
                                p.vel = v * ((e_k - loss) / e_k).sqrt();
                                p.pos = hit.point + p.vel.normalize() * 0.02;
                            } else {
                                dead = true;
                            }
                        }
                        HitTarget::Pane(a, b) => {
                            if self.arena.damage_pane((a, b), e_k, self.tick) {
                                self.pane_broken((a, b), hit.point, p.owner);
                            }
                            dead = true;
                        }
                        _ => {
                            // ricochet at grazing incidence (< 10°), otherwise stop
                            let cosi = -v.normalize_or_zero().dot(hit.normal);
                            if cosi < 0.17 && cosi > 0.0 {
                                p.vel = (v - 2.0 * v.dot(hit.normal) * hit.normal) * 0.45;
                                p.pos = hit.point + hit.normal * 0.02;
                            } else {
                                dead = true;
                            }
                        }
                    }
                } else if tf <= 1.0 {
                    let (t, x) = fx.unwrap();
                    let q = p.pos + (p1 - p.pos) * t;
                    match x {
                        Crossing::Shell { normal, barrier } => {
                            let (v, refl) = shell_response(vt(t), normal, barrier);
                            p.vel = v;
                            p.pos = if refl { q - vt(t).normalize_or_zero() * 0.01 } else { q + v.normalize_or_zero() * 0.01 };
                        }
                        Crossing::Fold { shift, .. } => {
                            p.pos = q + shift;
                            p.vel = vt(t);
                            self.destabilise_folds(round.mass);
                        }
                    }
                } else {
                    p.pos = p1;
                    p.vel = v1;
                }
                if dead {
                    break;
                }
            }
            p.age += dt;
            let h = geom::world_to_surf(p.pos).h;
            if dead || p.age > wc.max_flight_s || p.vel.length() < 25.0 || h < -3000.0 {
                self.ecs.despawn(pe);
            } else {
                self.ecs.projectiles.insert(pe, p);
            }
        }
    }

    fn pane_broken(&mut self, key: (i32, i32), point: DVec3, by: Entity) {
        self.events.push(Event::PaneBroken { key, by });
        self.sound(point, 135.0, by, SoundKind::Break);
        let pane = self.arena.st.pane;
        let (u, v) = self.arena.pane_centre(key);
        let b = crate::map::BoxCol { u, v, hu: 0.5 * pane, hv: 0.5 * pane, h0: -crate::map::WINDOW_THICKNESS, h1: 0.0, mat: Material::Window };
        self.spawn_debris(&b, geom::up_at(point) * -1.0);
        if let Some(l) = self.bh.as_mut() {
            l.station.puncture(node_of(point), pane * pane);
        }
    }

    fn box_destroyed(&mut self, i: u32, dir: DVec3) {
        let b = self.arena.st.boxes[i as usize];
        self.events.push(Event::BoxDestroyed { index: i, material: b.mat });
        self.sound(b.centre(), 115.0, Entity::default(), SoundKind::Break);
        self.spawn_debris(&b, dir);
    }

    /// Voronoi debris of a destroyed box (pieces of the shared fracture template).
    fn spawn_debris(&mut self, b: &crate::map::BoxCol, dir: DVec3) {
        let n = b.mat.debris_pieces().min(self.fracture.len());
        let c = b.centre();
        let (east, up, north) = geom::basis_at(c);
        let size = DVec3::new(2.0 * b.hu, b.h1 - b.h0, 2.0 * b.hv);
        let mass = b.volume() * b.mat.density();
        let mut rng = Rng::keyed(&[self.seed, self.tick as u64, b.u.to_bits(), b.v.to_bits()]);
        for k in 0..n {
            let (cen, vol) = self.fracture[k];
            let off = east * (cen.x * size.x) + up * (cen.y * size.y) + north * (cen.z * size.z);
            let vel = dir * rng.range(1.0, 6.0) + off.normalize_or_zero() * rng.range(0.5, 3.0);
            if self.ecs.debris.len() >= MAX_DEBRIS {
                let oldest = self.ecs.debris.entities()[0];
                self.ecs.despawn(oldest);
            }
            let e = self.ecs.entities.spawn();
            self.ecs.debris.insert(
                e,
                Debris { pos: c + off, vel, mass: mass * vol, size: (vol * size.x * size.y * size.z).max(1e-6).powf(1.0 / 3.0), life: 12.0, material: b.mat },
            );
        }
    }

    fn step_debris(&mut self) {
        let list: Vec<Entity> = self.ecs.debris.entities().to_vec();
        for e in list {
            let mut d = self.ecs.debris.get(e).unwrap().clone();
            d.vel += (self.frame.accel(d.pos, d.vel) + self.fields.accel(d.pos)) * DT;
            d.pos += d.vel * DT;
            let s = self.arena.surf(d.pos);
            if let Some((h, _, _)) = self.arena.floor(s.u, s.v) {
                if s.h < h && s.h > h - 1.0 {
                    d.pos += geom::up_at(d.pos) * (h - s.h);
                    d.vel *= 0.3;
                }
            }
            d.life -= DT;
            if d.life <= 0.0 || s.h < -500.0 {
                self.ecs.despawn(e);
            } else {
                self.ecs.debris.insert(e, d);
            }
        }
    }

    // ------------------------------------------------------- damage/life --

    /// Applies damage; `protected` damage (bullets) respects spawn protection.
    pub fn hurt(&mut self, e: Entity, dmg: f64, by: Option<Entity>, cause: DeathCause, protected: bool) {
        if dmg <= 0.0 {
            return;
        }
        let tick = self.tick;
        let Some(h) = self.ecs.health.get_mut(e) else { return };
        if !h.alive || (protected && h.protect > 0.0) {
            return;
        }
        h.hp -= dmg;
        h.recent_damage += dmg;
        if by.is_some() {
            h.last_attacker = by;
            h.last_attack_tick = tick;
        }
        if h.hp <= 0.0 {
            // credit environmental kills to whoever hit the victim in the last 5 s
            let killer = by.or_else(|| (tick.saturating_sub(h.last_attack_tick) < 5 * TICK_HZ).then_some(h.last_attacker).flatten());
            h.alive = false;
            h.hp = 0.0;
            h.respawn_in = self.cfg.rules.respawn_s;
            h.cause = Some(cause);
            if let Some(s) = self.ecs.stats.get_mut(e) {
                s.deaths += 1;
                if !matches!(cause, DeathCause::Bullet(_)) {
                    s.env_deaths += 1;
                }
            }
            match killer {
                Some(k) if k != e => {
                    if let Some(s) = self.ecs.stats.get_mut(k) {
                        s.kills += 1;
                    }
                }
                _ => {
                    if let Some(s) = self.ecs.stats.get_mut(e) {
                        s.suicides += 1;
                    }
                }
            }
            self.events.push(Event::Kill { killer, victim: e, cause });
        }
    }

    fn respawn(&mut self, e: Entity) {
        let team = self.team(e);
        let enemies: Vec<DVec3> = self
            .players
            .iter()
            .filter(|&&o| o != e && self.team(o) != team && self.is_alive(o))
            .filter_map(|&o| self.ecs.bodies.get(o).map(|b| b.pos))
            .collect();
        let spawns = self.arena.st.spawns.clone();
        let mut best = (f64::MIN, 0usize);
        for (i, &(u, v)) in spawns.iter().enumerate() {
            let p = geom::surf_to_world(u, v, 0.0);
            let d = enemies.iter().map(|q| (*q - p).length()).fold(f64::MAX, f64::min);
            // deterministic tie-break that still varies between respawns
            let jitter = Rng::keyed(&[self.seed, self.tick as u64, e.bits(), i as u64]).f64() * 30.0;
            if d + jitter > best.0 {
                best = (d + jitter, i);
            }
        }
        let (u, v) = spawns[best.1];
        let h = self.arena.floor(u, v).map_or(0.0, |f| f.0) + 0.02;
        let feet = geom::surf_to_world(u, v, h);
        let yaw = Rng::keyed(&[self.seed, self.tick as u64, e.bits(), 77]).f64() * std::f64::consts::TAU;
        let cfg = self.cfg.clone();
        self.ecs.bodies.insert(e, player::new_body(&cfg.player, feet, yaw));
        let hh = self.ecs.health.get_mut(e).unwrap();
        hh.hp = hh.max;
        hh.alive = true;
        hh.protect = cfg.rules.spawn_protection_s;
        hh.dose_gy = 0.0;
        hh.cause = None;
        hh.last_attacker = None;
        hh.recent_damage = 0.0;
        self.ecs.suits.insert(
            e,
            Suit {
                o2_s: cfg.player.suit_o2_s,
                integrity: 1.0,
                sealed: false,
                propellant_kg: cfg.player.jetpack.propellant_kg,
                p_prev: self.atmo.pressure(&self.frame, feet),
            },
        );
        let keep = self.ecs.arsenals.get(e).map_or(1, |a| a.current);
        let mut a = self.fresh_arsenal();
        a.current = keep;
        self.ecs.arsenals.insert(e, a);
        self.events.push(Event::Respawn { who: e });
    }

    fn step_environment(&mut self) {
        let cfg = self.cfg.clone();
        let bh_rate = self.bh.as_ref().map(|l| {
            let st = &l.station;
            (st, l.frag)
        });
        let mut rates = Vec::with_capacity(self.players.len());
        for &e in &self.players {
            let b = self.ecs.bodies.get(e).unwrap();
            // proper time: SR factor of the inertial speed × field lapse × black-hole panel clock
            let c = b.centre();
            let v_in = self.frame.frame_velocity(c) + b.vel;
            let beta2 = v_in.length_squared() / (C_LIGHT * C_LIGHT);
            let lapse = self.fields.time_rate(c);
            let panel = bh_rate.map_or(1.0, |(st, _)| {
                let n = &st.nodes[node_of(c)];
                n.rate * st.frags[n.frag].body.dt_dtau(1.0)
            });
            rates.push((e, (1.0 - beta2).sqrt() * lapse * panel, lapse));
        }
        for (e, rate, lapse) in rates {
            if let Some(cl) = self.ecs.clocks.get_mut(e) {
                cl.rate = rate;
                cl.dyn_rate = lapse;
                cl.tau += rate * DT;
            }
        }
        for i in 0..self.players.len() {
            let e = self.players[i];
            if !self.is_alive(e) {
                continue;
            }
            let b = self.ecs.bodies.get(e).unwrap().clone();
            let mut suit = self.ecs.suits.get(e).unwrap().clone();
            let mut h = self.ecs.health.get(e).unwrap().clone();
            let dt = DT * self.ecs.clocks.get(e).map_or(1.0, |c| c.dyn_rate);
            h.protect = (h.protect - dt).max(0.0);
            h.recent_damage *= det::exp(-dt);
            let env = damage::environment(&b, &mut suit, &mut h, &cfg.damage, &self.frame, &self.atmo, &self.fields, dt);
            self.ecs.suits.insert(e, suit);
            self.ecs.health.insert(e, h);
            if let Some((d, cause)) = env {
                self.hurt(e, d, None, cause, false);
            }
            // micro holes shred what touches them
            let c = b.centre();
            if self.fields.items.iter().any(|f| matches!(f.kind, FieldKind::MicroHole { .. }) && (f.pos - c).length() < 0.25) {
                self.hurt(e, 1e9, None, DeathCause::Spaghettified, false);
            }
            if geom::world_to_surf(b.pos).h < -300.0 {
                self.hurt(e, 1e9, None, DeathCause::LostInSpace, false);
            }
        }
    }

    fn step_air_and_structure(&mut self) {
        if self.bh.is_some() {
            self.step_bh();
        } else {
            let a = self.arena.breach_area();
            self.atmo.vent(a, DT);
        }
    }

    fn step_bh(&mut self) {
        let Some(l) = self.bh.as_mut() else { return };
        let m_s = l.m / C_LIGHT;
        let d_t = DT / m_s * l.station.reference.dt_dtau(1.0);
        l.station.step(l.m, d_t);
        self.sync_bh();
    }

    /// Rotating frame, black-hole field and air from the station fragment
    /// that carries the arena.
    fn sync_bh(&mut self) {
        let Some(l) = self.bh.as_mut() else { return };
        let (u, v) = self.arena.centre();
        let node = node_of(geom::surf_to_world(u, v, 0.0));
        let fi = l.station.nodes[node].frag;
        l.frag = fi;
        let f = &l.station.frags[fi];
        let qi = f.q.inverse();
        let c0 = DVec3::new(0.0, 0.0, 0.5 * geom::L);
        self.frame.spin = qi * f.omega;
        self.frame.com = f.com + c0;
        self.frame.bh = Some(BhField { m: l.m, com_bh: f.body.x * l.m, to_bh: DMat3::from_quat(f.q) });
        self.atmo.frac = (l.station.pressure / l.station.spec.pressure).clamp(0.0, 1.0);
    }

    fn step_rules(&mut self) {
        if self.game.over {
            return;
        }
        self.game.time += DT;
        let r = &self.cfg.rules;
        let mut lead: Option<(u32, Entity)> = None;
        for &e in &self.players {
            let k = self.ecs.stats.get(e).map_or(0, |s| s.kills);
            if lead.is_none_or(|(lk, _)| k > lk) {
                lead = Some((k, e));
            }
        }
        if self.game.time >= r.time_limit_s || lead.is_some_and(|(k, _)| k >= r.frag_limit) {
            self.game.over = true;
            self.game.winner = lead.map(|x| x.1);
            self.events.push(Event::MatchOver { winner: self.game.winner });
        }
    }

    // ---------------------------------------------------------- devices --

    fn activate_device(&mut self, e: Entity) {
        let cfg = self.cfg.clone();
        let dc = &cfg.devices;
        let mut d = self.ecs.devices.get(e).unwrap().clone();
        if d.cooldown > 0.0 || d.active.is_some() {
            return;
        }
        let (cost, instab) = match d.mode {
            DeviceMode::Bubble => (dc.bubble.cost_j, dc.bubble.instability),
            DeviceMode::MicroHole => (dc.micro_hole.cost_j, dc.micro_hole.instability),
            DeviceMode::Fold => (dc.fold.cost_j, dc.fold.instability),
        };
        if d.charge_j < cost {
            return;
        }
        d.charge_j -= cost;
        d.uses += 1;
        d.cooldown = dc.cooldown_s;
        let p_fail = 1.0 - det::exp(-d.instability / dc.failure_scale);
        let mut rng = Rng::keyed(&[self.seed, self.tick as u64, e.bits(), d.uses as u64, 0xDE71CE]);
        let fail = rng.f64() < p_fail;
        d.instability += instab;
        if fail {
            d.failures += 1;
        }
        if let Some(s) = self.ecs.stats.get_mut(e) {
            s.device_uses += 1;
            s.device_failures += fail as u32;
        }
        let mode = d.mode;
        let body = self.ecs.bodies.get(e).unwrap().clone();
        let eye = player::eye(&body, &cfg.player);
        let fwd = player::forward(&body, 0.0);
        let mk = |kind: FieldKind, pos: DVec3, vel: DVec3, ttl: f64| Field { kind, pos, vel, owner: e, age: 0.0, ttl, heat: 0.0 };
        let mut spawned = None;
        match (mode, fail) {
            (DeviceMode::Bubble, false) => {
                let b = &dc.bubble;
                spawned = Some(mk(
                    FieldKind::Bubble { radius: b.radius, barrier: b.barrier_j_per_kg, time_rate: b.time_rate },
                    body.centre(),
                    DVec3::ZERO,
                    b.duration_s,
                ));
            }
            (DeviceMode::Bubble, true) => {
                // inverted collapse: a shear wave through everything within 3 radii
                self.shear_pulse(body.centre(), 3.0 * dc.bubble.radius, 120.0, Some(e));
                self.events
                    .push(Event::DeviceFailure { owner: e, mode, what: "пузырь схлопнулся внутрь — сдвиговая волна" });
            }
            (DeviceMode::MicroHole, f) => {
                let h = &dc.micro_hole;
                let vel = body.vel + if f { DVec3::ZERO } else { fwd * h.launch_speed };
                spawned = Some(mk(FieldKind::MicroHole { mass: h.mass_kg, contained: !f }, eye + fwd * 1.2, vel, if f { f64::INFINITY } else { h.duration_s }));
                if f {
                    self.events.push(Event::DeviceFailure {
                        owner: e, mode, what: "удержание не включилось — свободная ЧД, излучение Хокинга"
                    });
                }
            }
            (DeviceMode::Fold, f) => {
                let fc = &dc.fold;
                let entry = eye + fwd * 2.0;
                let (exit, stab) = if f {
                    // the throat opens somewhere it should not — possibly through the floor
                    let dir = DVec3::new(rng.normal(), rng.normal(), rng.normal()).normalize_or(DVec3::Z);
                    (entry + dir * rng.range(5.0, fc.max_range), 0.3)
                } else {
                    let far = entry + fwd * fc.max_range;
                    let hit = self.arena.raycast(entry, far, &mut self.scratch);
                    (hit.map_or(far, |h| h.point - fwd * 2.0 * fc.mouth_radius), 1.0)
                };
                if f {
                    let s = self.arena.surf(exit);
                    if s.h < 0.0 && !crate::map::is_land(s.u) {
                        let k = self.arena.pane_key(s.u, s.v);
                        if self.arena.damage_pane(k, 1e9, self.tick) {
                            self.pane_broken(k, exit, e);
                        }
                    }
                    self.events.push(Event::DeviceFailure { owner: e, mode, what: "складка открылась не туда" });
                }
                spawned = Some(mk(FieldKind::Fold { exit, radius: fc.mouth_radius, stability: stab }, entry, DVec3::ZERO, fc.duration_s));
            }
        }
        if let Some(f) = spawned {
            let fe = self.ecs.entities.spawn();
            self.ecs.fields.insert(fe, f);
            d.active = Some(fe);
            self.sound(eye, 130.0, e, SoundKind::Device);
            self.events.push(Event::DeviceUsed { owner: e, mode });
        }
        self.ecs.devices.insert(e, d);
    }

    fn destabilise_folds(&mut self, mass: f64) {
        for (_, f) in self.ecs.fields.iter_mut() {
            if let FieldKind::Fold { stability, .. } = &mut f.kind {
                *stability -= mass / 400.0 + 0.002;
            }
        }
    }

    /// Gravitational shear wave: tidal damage falling off linearly to `radius`.
    fn shear_pulse(&mut self, c: DVec3, radius: f64, peak: f64, by: Option<Entity>) {
        let victims: Vec<(Entity, f64)> = self
            .ecs
            .bodies
            .iter()
            .filter_map(|(e, b)| {
                let d = (b.centre() - c).length();
                (d < radius).then_some((e, peak * (1.0 - d / radius)))
            })
            .collect();
        for (e, d) in victims {
            self.hurt(e, d, by, DeathCause::Tidal, false);
        }
        // glass and light structures nearby fail as well
        let s = self.arena.surf(c);
        let mut near = Vec::new();
        self.arena.boxes_near(s.u, s.v, radius, &mut near);
        for i in near {
            if self.arena.damage_box(i, 1e6) {
                let dir = (self.arena.st.boxes[i as usize].centre() - c).normalize_or_zero();
                self.box_destroyed(i, dir);
            }
        }
    }

    fn step_fields(&mut self) {
        let cfg = self.cfg.clone();
        let list: Vec<Entity> = self.ecs.fields.entities().to_vec();
        for fe in list {
            let mut f = *self.ecs.fields.get(fe).unwrap();
            f.age += DT;
            let mut end: Option<&'static str> = None;
            match &mut f.kind {
                FieldKind::Bubble { .. } => {
                    if f.age >= f.ttl {
                        end = Some("пузырь");
                    }
                }
                FieldKind::MicroHole { mass, contained } => {
                    // a free massive body in the rotating frame: it falls "down" and through the floor
                    let old = f.pos;
                    f.vel += self.frame.accel(f.pos, f.vel) * DT;
                    f.pos += f.vel * DT;
                    let (s0, s1) = (self.arena.surf(old), self.arena.surf(f.pos));
                    if s0.h >= 0.0 && s1.h < 0.0 && !crate::map::is_land(s1.u) {
                        // tidal shredding of the glass around the passage
                        let k = self.arena.pane_key(s1.u, s1.v);
                        if self.arena.damage_pane(k, 1e9, self.tick) {
                            self.pane_broken(k, f.pos, f.owner);
                        }
                    }
                    if *contained {
                        f.heat += fields::hawking_power(*mass) * DT;
                        if f.heat > cfg.devices.micro_hole.containment_capacity_j {
                            *contained = false;
                            f.ttl = f64::INFINITY;
                            self.events.push(Event::ContainmentLost { owner: f.owner });
                        } else if f.age >= f.ttl {
                            end = Some("микро-ЧД отозвана");
                        }
                    }
                    if s1.h < -400.0 {
                        end = Some("микро-ЧД ушла в космос");
                    }
                }
                FieldKind::Fold { stability, exit, .. } => {
                    if *stability < 0.15 {
                        let (a, b) = (f.pos, *exit);
                        self.shear_pulse(a, 6.0, 90.0, Some(f.owner));
                        self.shear_pulse(b, 6.0, 90.0, Some(f.owner));
                        end = Some("складка схлопнулась");
                    } else if f.age >= f.ttl {
                        end = Some("складка");
                    }
                }
            }
            if let Some(kind) = end {
                self.events.push(Event::FieldEnded { owner: f.owner, kind });
                self.ecs.despawn(fe);
            } else {
                self.ecs.fields.insert(fe, f);
            }
        }
    }

    // ----------------------------------------------------------- state --

    /// FNV-1a hash of the complete gameplay state.
    pub fn state_hash(&self) -> u64 {
        let mut h = StateHasher::default();
        h.u64(self.tick as u64);
        fn eat<T: HashState>(h: &mut StateHasher, s: &Storage<T>) {
            h.u64(s.len() as u64);
            for (e, c) in s.iter() {
                h.u64(e.bits());
                c.hash(h);
            }
        }
        eat(&mut h, &self.ecs.pilots);
        eat(&mut h, &self.ecs.bodies);
        eat(&mut h, &self.ecs.health);
        eat(&mut h, &self.ecs.suits);
        eat(&mut h, &self.ecs.clocks);
        eat(&mut h, &self.ecs.arsenals);
        eat(&mut h, &self.ecs.devices);
        eat(&mut h, &self.ecs.stats);
        eat(&mut h, &self.ecs.projectiles);
        eat(&mut h, &self.ecs.fields);
        eat(&mut h, &self.ecs.debris);
        let d = &self.arena.dynamic;
        for (k, v) in &d.box_damage {
            h.u64(*k as u64);
            h.f64(*v);
        }
        for (i, x) in d.destroyed.iter().enumerate() {
            if *x {
                h.u64(i as u64);
            }
        }
        for (k, v) in &d.pane_damage {
            h.u64((k.0 as u32 as u64) << 32 | k.1 as u32 as u64);
            h.f64(*v);
        }
        for (k, t) in &d.broken {
            h.u64((k.0 as u32 as u64) << 32 | k.1 as u32 as u64);
            h.u64(*t as u64);
        }
        h.f64(self.atmo.frac);
        h.f64(self.game.time);
        h.bool(self.game.over);
        if let Some(l) = &self.bh {
            l.station.hash_into(&mut |x| h.f64(x));
        }
        h.0
    }
}

/// Closest parameters (s on p, t on q) between segments p0p1 and q0q1.
fn seg_seg(p0: DVec3, p1: DVec3, q0: DVec3, q1: DVec3) -> (f64, f64) {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);
    if a < 1e-18 {
        return (0.0, (f / e).clamp(0.0, 1.0));
    }
    let c = d1.dot(r);
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    let mut s = if denom > 1e-18 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (s, t)
}

fn point_seg_dist(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-18)).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}
