//! "Схватка" inside the rendered colony: drives the deterministic
//! `gameplay::Sim` at a fixed 60 Hz from the keyboard/mouse (as `Command`s,
//! exactly what a network client would send), lets `ai::Bots` play the
//! others, and turns the simulation state into things the renderer already
//! knows how to draw: instanced ellipsoids (bodies, debris, micro holes),
//! glowing ribbons (tracers, fields, the neuro-interface) and an egui HUD.

use std::collections::VecDeque;

use egui::{Align2, Color32, FontId, Pos2, Stroke};
use gameplay::components::{DeathCause, DeviceMode, MoveMode};
use gameplay::fields::FieldKind;
use gameplay::hud::{neuro_projection, NeuroHud};
use gameplay::sim::{Sim, SoundKind};
use gameplay::{buttons, geom, player, Arena, Body, Command, Entity, Event, GameConfig, DT};
use glam::{DQuat, DVec3, Quat, Vec3};

use crate::avatar::{self, PoseIn, ViewIn};
use crate::fx::Fx;
use crate::render::props::{ParticleGpu, PropInst};
use crate::render::{BallGpu, LineVertex};

/// Default battlefield: the "Мегаполис" valley next to its window strip.
pub fn default_origin() -> (f64, f64) {
    (std::f64::consts::TAU / 3.0 * geom::R + 300.0, 15_000.0)
}

pub const ARENA_SIZE: f64 = 2000.0;

#[derive(Clone, Debug)]
pub struct Setup {
    pub bots: usize,
    /// "novice", "regular", "veteran" or "mixed"
    pub level: String,
    pub spectate: bool,
    pub seed: u64,
}

impl Default for Setup {
    fn default() -> Self {
        Setup { bots: 8, level: "mixed".into(), spectate: false, seed: 1 }
    }
}

/// Held controls this frame (sampled into a Command every tick).
#[derive(Clone, Copy, Debug, Default)]
pub struct Controls {
    pub fwd: f32,
    pub right: f32,
    pub up: f32,
    pub roll: f32,
    pub run: bool,
    pub crouch: bool,
    pub jump: bool,
    pub fire: bool,
    pub ads: bool,
    pub boots: bool,
}

struct Feed {
    text: String,
    color: Color32,
    t: f64,
}

pub struct Combat {
    pub sim: Sim,
    pub bots: ai::Bots,
    pub me: Option<u8>,
    pub follow: usize,
    pub setup: Setup,
    acc: f64,
    look: [f64; 3],
    latched: u16,
    weapon: u8,
    prev_pos: Vec<DVec3>,
    feed: VecDeque<Feed>,
    hit_marker: (f32, bool),
    hurt: Vec<(DVec3, f32)>,
    neuro: Option<NeuroHud>,
    pub scoreboard: bool,
    pub time: f64,
    pub tick_ms: f32,
    fx: Fx,
    gait: Vec<f32>,
    crouch: Vec<f32>,
    hit_flash: Vec<f32>,
    propellant: Vec<f64>,
    muzzles: Vec<(Vec3, Vec3)>,
    vm: Vm,
    props: Vec<(u16, PropInst)>,
    particles: Vec<ParticleGpu>,
    shake: f32,
    kill_confirm: (f32, String, bool),
    pub fx_quality: f32,
}

/// First-person weapon state (purely visual).
#[derive(Default)]
struct Vm {
    ads: f32,
    kick: f32,
    sway: [f32; 2],
    last_fwd: Vec3,
    sprint: f32,
    muzzle: Vec3,
    eject: Vec3,
    fwd: Vec3,
}

fn hue_of(team: u8) -> f32 {
    (team as f32 * 0.618_034).fract()
}

/// Saturated team colour (linear RGB) for armour panels and suit lights.
fn team_rgb(team: u8) -> [f32; 3] {
    let h = hue_of(team);
    let f = |k: f32| ((((h + k).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0)).powf(2.2) * 0.85 + 0.06;
    [f(0.0), f(2.0 / 3.0), f(1.0 / 3.0)]
}

fn v3(p: DVec3) -> Vec3 {
    p.as_vec3()
}

fn cause_ru(c: DeathCause, sim: &Sim) -> String {
    match c {
        DeathCause::Bullet(w) => match sim.cfg.weapons.get(w as usize).map(|w| w.id.as_str()) {
            Some("pistol") => "пистолет".into(),
            Some("rifle") => "автомат".into(),
            Some("sniper") => "снайперка".into(),
            Some(other) => other.into(),
            None => "пуля".into(),
        },
        DeathCause::Impact => "удар".into(),
        DeathCause::Decompression => "декомпрессия".into(),
        DeathCause::Tidal => "приливные силы".into(),
        DeathCause::Radiation => "радиация".into(),
        DeathCause::Spaghettified => "спагеттификация".into(),
        DeathCause::LostInSpace => "унесло в космос".into(),
        DeathCause::Debris => "обломки".into(),
    }
}

impl Combat {
    /// Builds the arena out of the rendered world around (u0, v0) (corner).
    pub fn new(world: &oneill_world::world::World, origin: (f64, f64), setup: Setup) -> Combat {
        let t0 = std::time::Instant::now();
        let v0 = origin.1.clamp(200.0, geom::L - ARENA_SIZE - 200.0);
        let arena = Arena::from_world(world, origin.0, v0, ARENA_SIZE);
        let human = !setup.spectate;
        let sim = Sim::skirmish(GameConfig::builtin(), arena, setup.seed, human, setup.bots);
        let all = ai::config::builtin();
        let levels: Vec<ai::Difficulty> = match setup.level.as_str() {
            "mixed" => all,
            n => ai::config::by_name(&all, n).map(|d| vec![d]).unwrap_or(all),
        };
        let bots = ai::Bots::new(&sim, &levels);
        log::info!(
            "skirmish: arena {} boxes, nav {:.0} % open, set up in {:.0} ms",
            sim.arena.st.boxes.len(),
            bots.nav.open_fraction() * 100.0,
            t0.elapsed().as_secs_f64() * 1e3
        );
        let n = sim.players.len();
        let prev_pos = sim.players.iter().map(|&e| sim.ecs.bodies.get(e).map_or(DVec3::ZERO, |b| b.pos)).collect();
        Combat {
            me: human.then_some(0),
            follow: 0,
            bots,
            sim,
            setup,
            acc: 0.0,
            look: [0.0; 3],
            latched: 0,
            weapon: 0,
            prev_pos,
            feed: VecDeque::new(),
            hit_marker: (0.0, false),
            hurt: Vec::new(),
            neuro: None,
            scoreboard: false,
            time: 0.0,
            tick_ms: 0.0,
            fx: Fx::default(),
            gait: vec![0.0; n],
            crouch: vec![0.0; n],
            hit_flash: vec![0.0; n],
            propellant: vec![0.0; n],
            muzzles: vec![(Vec3::ZERO, Vec3::ZERO); n],
            vm: Vm::default(),
            props: Vec::new(),
            particles: Vec::new(),
            shake: 0.0,
            kill_confirm: (0.0, String::new(), false),
            fx_quality: 1.0,
        }
    }

    pub fn me_entity(&self) -> Option<Entity> {
        self.me.and_then(|i| self.sim.entity(i))
    }

    /// Whose eyes / back the camera follows.
    pub fn viewed(&self) -> Entity {
        self.me_entity().unwrap_or_else(|| self.sim.players[self.follow % self.sim.players.len()])
    }

    pub fn mouse(&mut self, dx: f64, dy: f64, sens: f64) {
        self.look[0] -= dx * sens;
        self.look[1] -= dy * sens;
    }

    /// Edge-triggered buttons (reload, device, …) survive until the next tick.
    pub fn press(&mut self, b: u16) {
        self.latched |= b;
    }

    pub fn select_weapon(&mut self, slot: u8) {
        self.weapon = slot;
    }

    pub fn cycle_follow(&mut self, d: i32) {
        let n = self.sim.players.len() as i32;
        self.follow = (self.follow as i32 + d).rem_euclid(n) as usize;
    }

    fn command(&mut self, c: &Controls, tick: u32, id: u8) -> Command {
        let mut cmd = Command::new(tick, id);
        cmd.set_move(c.fwd as f64, c.right as f64, c.up as f64);
        // keep the quantisation remainder so slow mouse motion is not lost
        let (y, p, r) = (self.look[0], self.look[1], self.look[2] + c.roll as f64 * 1.5 * DT);
        cmd.set_look(y, p, r);
        let q = cmd.look_delta();
        self.look = [y - q[0], p - q[1], r - q[2]];
        let mut b = self.latched;
        self.latched = 0;
        let mut hold = |on: bool, bit: u16| {
            if on {
                b |= bit;
            }
        };
        hold(c.fire, buttons::FIRE);
        hold(c.ads, buttons::ADS);
        hold(c.jump, buttons::JUMP);
        hold(c.crouch, buttons::CROUCH);
        hold(c.run, buttons::RUN);
        hold(c.boots, buttons::MAG_BOOTS);
        cmd.buttons = b;
        cmd.weapon = std::mem::take(&mut self.weapon);
        cmd
    }

    /// Advances the fixed-step simulation by the real frame time.
    pub fn update(&mut self, dt: f64, controls: &Controls) {
        self.time += dt;
        self.acc = (self.acc + dt).min(0.25);
        let t0 = std::time::Instant::now();
        let mut n = 0;
        while self.acc >= DT {
            self.acc -= DT;
            n += 1;
            self.prev_pos.clear();
            self.prev_pos.extend(self.sim.players.iter().map(|&e| self.sim.ecs.bodies.get(e).map_or(DVec3::ZERO, |b| b.pos)));
            let mut cmds = self.bots.think(&self.sim);
            if let Some(id) = self.me {
                let c = self.command(controls, self.sim.tick, id);
                cmds.push(c);
            }
            self.sim.step(&cmds);
            self.events();
            if self.sim.tick.is_multiple_of(4) || self.neuro.is_some() && self.sim.tick.is_multiple_of(2) {
                self.neuro = neuro_projection(&self.sim, self.viewed());
            }
        }
        if n > 0 {
            self.tick_ms = 0.8 * self.tick_ms + 0.2 * (t0.elapsed().as_secs_f32() * 1e3 / n as f32);
        }
        let fdt = dt as f32;
        self.hit_marker.0 = (self.hit_marker.0 - fdt).max(0.0);
        self.hurt.retain_mut(|h| {
            h.1 -= fdt;
            h.1 > 0.0
        });
        self.kill_confirm.0 = (self.kill_confirm.0 - fdt).max(0.0);
        self.shake *= (-fdt * 6.0).exp();
        for h in &mut self.hit_flash {
            *h = (*h - fdt * 6.0).max(0.0);
        }
        self.visuals(fdt, controls);
        while self.feed.front().is_some_and(|f| self.time - f.t > 7.0) {
            self.feed.pop_front();
        }
    }

    fn name(&self, e: Entity) -> String {
        self.sim.ecs.pilots.get(e).map_or("?".into(), |p| p.name.clone())
    }

    fn push_feed(&mut self, text: String, color: Color32) {
        self.feed.push_back(Feed { text, color, t: self.time });
        if self.feed.len() > 6 {
            self.feed.pop_front();
        }
    }

    fn events(&mut self) {
        let me = self.viewed();
        let evs = self.sim.events.clone();
        for ev in evs {
            match ev {
                Event::Kill { killer, victim, cause } => {
                    if let Some(b) = self.sim.ecs.bodies.get(victim) {
                        let col = team_rgb(self.sim.team(victim));
                        self.fx.death(v3(b.centre()), v3(geom::up_at(b.pos)), col);
                    }
                    if killer == Some(me) && victim != me {
                        let head = self.sim.ecs.health.get(victim).is_some_and(|h| matches!(h.cause, Some(DeathCause::Bullet(_)))) && self.hit_marker.1;
                        self.kill_confirm = (2.2, self.name(victim), head);
                    }
                    let c = cause_ru(cause, &self.sim);
                    let t = match killer {
                        Some(k) if k != victim => format!("{} ✖ {}  ({c})", self.name(k), self.name(victim)),
                        _ => format!("{} — {c}", self.name(victim)),
                    };
                    let col = if victim == me {
                        Color32::from_rgb(255, 90, 80)
                    } else if killer == Some(me) {
                        Color32::from_rgb(120, 255, 140)
                    } else {
                        Color32::from_gray(225)
                    };
                    self.push_feed(t, col);
                }
                Event::Hit { shooter, victim, head, damage, .. } => {
                    if let (Some(a), Some(b)) = (self.sim.ecs.bodies.get(shooter), self.sim.ecs.bodies.get(victim)) {
                        let c = if head { self.sim.eye(victim) } else { b.centre() };
                        let dir = v3(c - a.centre()).normalize_or(Vec3::Z);
                        self.fx.breach(v3(c) - dir * 0.2, -dir, head);
                    }
                    if let Some(i) = self.sim.players.iter().position(|&x| x == victim) {
                        self.hit_flash[i] = 1.0;
                    }
                    if victim == me {
                        self.shake = (self.shake + damage as f32 / 35.0).min(1.5);
                    }
                    if shooter == me {
                        self.hit_marker = (0.25, head);
                    }
                    if victim == me {
                        if let (Some(a), Some(b)) = (self.sim.ecs.bodies.get(shooter), self.sim.ecs.bodies.get(me)) {
                            self.hurt.push((a.pos - b.pos, 1.5));
                        }
                    }
                }
                Event::DeviceFailure { owner, mode, what } => {
                    let t = format!("⚠ {}: {} — {}", self.name(owner), mode.label(), what);
                    self.push_feed(t, Color32::from_rgb(255, 190, 60));
                }
                Event::ContainmentLost { owner } => {
                    let t = format!("☢ {}: удержание микро-ЧД потеряно!", self.name(owner));
                    self.push_feed(t, Color32::from_rgb(255, 120, 40));
                }
                Event::PaneBroken { .. } => {}
                Event::MatchOver { winner } => {
                    let t = format!("Матч окончен — победитель: {}", winner.map_or("ничья".into(), |w| self.name(w)));
                    self.push_feed(t, Color32::from_rgb(255, 230, 120));
                }
                _ => {}
            }
        }
        // effects of this tick's sounds: shots, impacts, breaking glass, steps, devices
        let first_person = self.first_person();
        let cam = self.cam_pos();
        let sounds: Vec<_> = self.sim.sounds.iter().filter(|s| s.tick + 1 == self.sim.tick).copied().collect();
        for s in sounds {
            let pos = v3(s.pos);
            let near = pos.distance(cam) < 120.0;
            match s.kind {
                SoundKind::Shot => {
                    let Some(b) = self.sim.ecs.bodies.get(s.source) else { continue };
                    let w = self.sim.ecs.arsenals.get(s.source).map_or(1, |a| a.current);
                    let aim = v3(self.sim.aim_dir(s.source));
                    let up = v3(geom::up_at(b.pos));
                    let own = s.source == me && first_person;
                    let i = self.sim.players.iter().position(|&x| x == s.source);
                    let (muzzle, eject) = if own {
                        (self.vm.muzzle, self.vm.eject)
                    } else {
                        i.map(|i| self.muzzles[i]).filter(|m| m.0 != Vec3::ZERO && m.0.distance(pos) < 3.0).unwrap_or((pos + aim * 0.6, pos))
                    };
                    let dir = if own { self.vm.fwd } else { aim };
                    if near || own {
                        self.fx.muzzle(muzzle, dir, w, own);
                        let right = dir.cross(up).normalize_or(Vec3::X);
                        let floor = (geom::R - geom::r_perp(b.pos)) as f32;
                        if w != 2 || self.sim.ecs.arsenals.get(s.source).is_some_and(|a| a.slots[a.current].cooldown < 0.5) {
                            self.fx.eject(eject, right, up, dir, floor);
                        }
                    } else {
                        self.fx.light(muzzle, -6.0, 0.05);
                    }
                    if own {
                        self.vm.kick = (self.vm.kick + [0.6, 0.45, 1.0][w.min(2)]).min(1.4);
                        self.shake = (self.shake + [0.05, 0.04, 0.25][w.min(2)]).min(1.5);
                    }
                }
                SoundKind::Impact if near => {
                    let n = (cam - pos).normalize_or(Vec3::Y);
                    self.fx.impact(pos, n, s.db > 120.0);
                }
                SoundKind::Break if pos.distance(cam) < 300.0 => self.fx.glass(pos),
                SoundKind::Step if pos.distance(cam) < 25.0 => self.fx.dust_step(pos, v3(geom::up_at(s.pos))),
                SoundKind::Device => self.fx.shockwave(pos, 2.0, [0.4, 1.0, 2.5]),
                _ => {}
            }
        }
    }

    fn alpha(&self) -> f64 {
        (self.acc / DT).clamp(0.0, 1.0)
    }

    fn lerp_pos(&self, i: usize, b: &Body) -> DVec3 {
        match self.prev_pos.get(i) {
            Some(&p) if (p - b.pos).length() < 5.0 => p + (b.pos - p) * self.alpha(),
            _ => b.pos,
        }
    }

    /// Camera (eye, forward, up). First person for the player; a chase camera when spectating or dead.
    pub fn camera(&self) -> (Vec3, Vec3, Vec3) {
        let e = self.viewed();
        let i = self.sim.players.iter().position(|&x| x == e).unwrap_or(0);
        let Some(b) = self.sim.ecs.bodies.get(e) else { return (Vec3::ZERO, Vec3::Z, Vec3::Y) };
        let feet = self.lerp_pos(i, b);
        let eye = feet + (self.sim.eye(e) - b.pos);
        let fwd = self.sim.aim_dir(e);
        let up = if b.mode == MoveMode::Jetpack { b.orient * DVec3::Y } else { geom::up_at(b.pos) };
        let alive = self.sim.is_alive(e);
        if self.me.is_some() && alive && chase_override().is_none() {
            let f = v3(fwd);
            let u = v3(up);
            let l = u.cross(f).normalize_or(Vec3::X);
            let t = self.time as f32;
            let k = self.shake * 0.02;
            let f = (f + l * ((t * 41.0).sin() * k) + u * ((t * 33.0 + 1.3).sin() * k + self.vm.kick * 0.004)).normalize();
            return (v3(eye), f, u);
        }
        // chase camera: behind and above, not inside walls
        let back = if alive { fwd } else { (fwd - up * fwd.dot(up)).normalize_or(DVec3::Z) };
        let mut want = eye - back * 4.5 + up * 1.2;
        // dev: ONEILL_CHASE="azimuth°,distance,height" orbits the chase camera (0 = behind)
        if let Some((az, d, h)) = chase_override() {
            let flat = (back - up * back.dot(up)).normalize_or(DVec3::Z);
            let dir = DQuat::from_axis_angle(up, az.to_radians()) * flat;
            want = eye - dir * d + up * h;
            let mut scratch = Vec::new();
            let cam = match self.sim.arena.raycast(eye, want, &mut scratch) {
                Some(hh) => eye + (want - eye) * (hh.t * 0.85),
                None => want,
            };
            let look = (eye - up * 0.55 - cam).normalize_or(fwd);
            return (v3(cam), v3(look), v3(up));
        }
        let mut scratch = Vec::new();
        let cam = match self.sim.arena.raycast(eye, want, &mut scratch) {
            Some(h) => eye + (want - eye) * (h.t * 0.85),
            None => want,
        };
        let look = (eye + fwd * 6.0 - cam).normalize_or(fwd);
        (v3(cam), v3(look), v3(up))
    }

    // ------------------------------------------------------------ drawing --

    /// Bodies, debris, holes → instanced ellipsoids; tracers, fields, breaches, neuro path → ribbons.
    pub fn draw(&self, cam: Vec3, balls: &mut Vec<BallGpu>, lines: &mut Vec<LineVertex>) {
        let sim = &self.sim;
        for d in sim.ecs.debris.values() {
            let style = if matches!(d.material, gameplay::map::Material::Glass | gameplay::map::Material::Window) { 11 } else { 12 };
            let r = (0.5 * d.size) as f32;
            let rot = Quat::from_rotation_arc(Vec3::Y, v3(d.vel).normalize_or(Vec3::Y));
            balls.push(BallGpu { pos: v3(d.pos).to_array(), radius: r.max(0.03), rot: rot.to_array(), style, emissive: 0.0, ext: [0.35, 0.1] });
        }
        // tracers: every round is drawn as a short streak along its velocity
        for p in sim.ecs.projectiles.values() {
            let a = v3(p.pos - p.vel * 0.012);
            let b = v3(p.pos);
            let hot = (1.0 - p.age as f32 / 2.0).clamp(0.2, 1.0);
            crate::ribbon(lines, &[a, b], cam, [3.0 * hot, 1.5 * hot, 0.4 * hot], 1.0);
        }
        for f in &sim.fields.items {
            let c = v3(f.pos);
            match f.kind {
                FieldKind::Bubble { radius, .. } => {
                    let r = radius as f32;
                    let t = self.time as f32;
                    for (k, axis) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
                        let q = Quat::from_rotation_z(t * 0.3 + k as f32);
                        ring(lines, c, q * *axis, r, cam, [0.3, 0.8, 1.6]);
                    }
                }
                FieldKind::MicroHole { contained, .. } => {
                    balls.push(BallGpu {
                        pos: c.to_array(),
                        radius: 0.12,
                        rot: [0.0, 0.0, 0.0, 1.0],
                        style: 13,
                        emissive: if contained { 2.0 } else { 12.0 },
                        ext: [0.0; 2],
                    });
                    let up = geom::up_at(f.pos).as_vec3();
                    ring(lines, c, up, 0.35, cam, if contained { [0.5, 1.2, 2.5] } else { [4.0, 1.4, 0.3] });
                }
                FieldKind::Fold { exit, radius, stability } => {
                    let r = radius as f32;
                    let col = [1.2 * stability as f32 + 0.3, 0.3, 1.8];
                    let dir = v3(exit - f.pos).normalize_or(Vec3::Z);
                    ring(lines, c, dir, r, cam, col);
                    ring(lines, v3(exit), dir, r, cam, col);
                }
            }
        }
        // window breaches: outlined holes in the glass (air rushes out)
        let st = &sim.arena.st;
        for &(ku, kv) in sim.arena.dynamic.broken.keys() {
            let (u, v) = sim.arena.pane_centre((ku, kv));
            let h = 0.5 * st.pane;
            let corners: Vec<Vec3> = [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)].iter().map(|&(a, b)| v3(geom::surf_to_world(u + a, v + b, 0.05))).collect();
            if corners[0].distance(cam) < 1200.0 {
                crate::ribbon(lines, &corners, cam, [2.5, 0.5, 0.3], 1.0);
            }
        }
        // neuro-interface: the real path vs the barrel line
        if let Some(n) = &self.neuro {
            let pts: Vec<Vec3> = n.trajectory.points.iter().map(|&p| v3(p)).collect();
            crate::ribbon(lines, &pts, cam, [0.25, 1.0, 0.45], 1.4);
            if let (Some(&a), Some(&b0)) = (pts.last(), pts.first()) {
                crate::marker(lines, a, cam, [0.25, 1.0, 0.45]);
                let naive = v3(n.naive_point);
                crate::ribbon(lines, &[b0, naive], cam, [1.0, 0.25, 0.15], 0.6);
                crate::marker(lines, naive, cam, [1.0, 0.25, 0.15]);
            }
        }
    }

    /// Looking through the player's own eyes (alive, not spectating, no dev chase camera).
    fn first_person(&self) -> bool {
        self.me.is_some() && self.sim.is_alive(self.viewed()) && chase_override().is_none()
    }

    fn cam_pos(&self) -> Vec3 {
        self.camera().0
    }

    /// Vertical field of view: aiming down sights zooms (×4 through the sniper scope).
    pub fn fov_deg(&self) -> f32 {
        let w = self.me_entity().and_then(|e| self.sim.ecs.arsenals.get(e)).map_or(1, |a| a.current);
        let zoom = [1.12, 1.3, 4.0][w.min(2)];
        let a = self.vm.ads * self.vm.ads * (3.0 - 2.0 * self.vm.ads);
        70.0 / (1.0 + (zoom - 1.0) * a)
    }

    fn scoped(&self) -> bool {
        self.me_entity().and_then(|e| self.sim.ecs.arsenals.get(e)).is_some_and(|a| a.current == 2) && self.vm.ads > 0.85
    }

    /// This frame's posed parts and particles.
    pub fn frame(&self) -> (&[(u16, PropInst)], &[ParticleGpu]) {
        (&self.props, &self.particles)
    }

    /// Short-lived point lights (muzzle flashes, device pulses).
    pub fn lights(&self, n: usize) -> Vec<[f32; 4]> {
        self.fx.lights(n)
    }

    pub fn particle_count(&self) -> usize {
        self.fx.len()
    }

    /// Poses every operative, the first-person arms and weapon, and advances the effects.
    fn visuals(&mut self, dt: f32, controls: &Controls) {
        self.props.clear();
        self.particles.clear();
        self.fx.quality = self.fx_quality;
        let (eye, fwd, up) = self.camera();
        let viewer = self.viewed();
        let first_person = self.first_person();
        let rules = self.sim.cfg.rules.respawn_s;
        for i in 0..self.sim.players.len() {
            let e = self.sim.players[i];
            let Some(b) = self.sim.ecs.bodies.get(e) else { continue };
            let feet = v3(self.lerp_pos(i, b));
            let bup = v3(geom::up_at(b.pos));
            let hv = v3(b.vel) - bup * v3(b.vel).dot(bup);
            let speed = hv.length();
            if b.grounded {
                self.gait[i] = (self.gait[i] + speed * dt / (0.7 + 0.06 * speed.min(7.0)) * std::f32::consts::PI) % (2.0 * std::f32::consts::TAU);
            }
            let ck = if b.crouched { 1.0 } else { 0.0 };
            self.crouch[i] += (ck - self.crouch[i]) * (dt * 10.0).min(1.0);
            let own = e == viewer && first_person;
            let dist = feet.distance(eye);
            if dist > 600.0 && !own {
                continue;
            }
            let jet = b.mode == MoveMode::Jetpack;
            let (pup, pfwd) = if jet {
                (v3(b.orient * DVec3::Y), v3(b.orient * DVec3::Z))
            } else {
                let aim = v3(self.sim.aim_dir(e));
                (bup, (aim - bup * aim.dot(bup)).normalize_or(v3(geom::basis_at(b.pos).2)))
            };
            let h = self.sim.ecs.health.get(e);
            let alive = self.sim.is_alive(e);
            let a = self.sim.ecs.arsenals.get(e);
            let w = a.map_or(1, |a| a.current);
            let reload = a.map_or(0.0, |a| {
                let r = a.slots[a.current].reload;
                if r > 0.0 {
                    (1.0 - r / self.sim.cfg.weapons[a.current].reload_s) as f32
                } else {
                    0.0
                }
            });
            let team = team_rgb(self.sim.team(e));
            let protect = h.is_some_and(|h| h.protect > 0.0 && h.alive);
            let pin = PoseIn {
                feet,
                up: pup,
                fwd: pfwd,
                aim: v3(self.sim.aim_dir(e)),
                vel: v3(b.vel),
                gait: self.gait[i],
                grounded: b.grounded,
                crouch: self.crouch[i],
                jet,
                dead_t: if alive { None } else { Some(h.map_or(9.0, |h| (rules - h.respawn_in) as f32)) },
                weapon: w,
                reload,
                recoil: a.map_or(0.0, |a| (a.recoil * 20.0).min(1.0) as f32),
                ads: self.sim.ecs.pilots.get(e).map_or(0.0, |p| if p.last_cmd.held(buttons::ADS) { 1.0 } else { 0.0 }),
                switching: a.map_or(0.0, |a| a.switching as f32),
                tint: [team[0], team[1], team[2], if alive { 1.0 } else { 0.0 }],
                ext: [if protect { 0.6 } else { 0.0 }, if own { 2.0 } else { 0.0 }, self.hit_flash[i], 0.0],
            };
            let posed = avatar::pose_body(&pin, &mut self.props);
            self.muzzles[i] = (posed.muzzle, posed.eject);
            // jetpack plume while propellant is being burnt
            let prop = self.sim.ecs.suits.get(e).map_or(0.0, |s| s.propellant_kg);
            if jet && prop < self.propellant[i] - 1e-6 && dist < 200.0 {
                let down = -pup;
                for n in posed.nozzles {
                    self.fx.jet(n, down, dt);
                }
            }
            self.propellant[i] = prop;
        }
        if first_person {
            let e = viewer;
            let ads_held = controls.ads;
            let target = if ads_held { 1.0 } else { 0.0 };
            self.vm.ads += (target - self.vm.ads) * (dt * 9.0).min(1.0);
            self.vm.kick *= (-dt * 12.0).exp();
            let left = up.cross(fwd);
            let d = fwd - if self.vm.last_fwd == Vec3::ZERO { fwd } else { self.vm.last_fwd };
            self.vm.last_fwd = fwd;
            let ts = [(d.dot(left) * 0.6).clamp(-0.04, 0.04), (-d.dot(up) * 0.6).clamp(-0.04, 0.04)];
            for (sw, t) in self.vm.sway.iter_mut().zip(ts) {
                *sw += (t - *sw) * (dt * 10.0).min(1.0);
            }
            let b = self.sim.ecs.bodies.get(e);
            let speed = b.map_or(0.0, |b| v3(b.vel).length());
            let sprint = if controls.run && speed > 3.5 && !ads_held { 1.0 } else { 0.0 };
            self.vm.sprint += (sprint - self.vm.sprint) * (dt * 8.0).min(1.0);
            let a = self.sim.ecs.arsenals.get(e);
            let w = a.map_or(1, |a| a.current);
            let reload = a.map_or(0.0, |a| {
                let r = a.slots[a.current].reload;
                if r > 0.0 {
                    (1.0 - r / self.sim.cfg.weapons[a.current].reload_s) as f32
                } else {
                    0.0
                }
            });
            let i = self.sim.players.iter().position(|&x| x == e).unwrap_or(0);
            let team = team_rgb(self.sim.team(e));
            let vin = ViewIn {
                eye,
                fwd,
                up,
                weapon: w,
                ads: self.vm.ads,
                reload,
                kick: self.vm.kick,
                switching: a.map_or(0.0, |a| a.switching as f32),
                sway: self.vm.sway,
                gait: self.gait[i],
                moving: (speed / 3.0).min(1.0) * if b.is_some_and(|b| b.grounded) { 1.0 } else { 0.0 },
                sprint: self.vm.sprint,
                tint: [team[0], team[1], team[2], 1.0],
            };
            let start = self.props.len();
            let posed = avatar::pose_viewmodel(&vin, &mut self.props);
            self.vm.muzzle = posed.muzzle;
            self.vm.eject = posed.eject;
            self.vm.fwd = posed.weapon.matrix3.z_axis.into();
            if self.scoped() {
                self.props.truncate(start); // looking through the scope
            }
        } else {
            self.vm.ads = 0.0;
        }
        self.fx.update(dt);
        self.fx.casings(avatar::ids().casing, &mut self.props);
        self.fx.write(eye, &mut self.particles);
    }

    // ---------------------------------------------------------------- HUD --

    pub fn hud(&self, ctx: &egui::Context, view_proj: glam::Mat4) {
        let sim = &self.sim;
        let e = self.viewed();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("combat_hud")));
        let rect = ctx.screen_rect();
        let c = rect.center();
        let white = Color32::from_rgba_unmultiplied(235, 240, 245, 230);
        let dim = Color32::from_rgba_unmultiplied(0, 0, 0, 140);
        let f = |s: f32| FontId::proportional(s);
        let alive = sim.is_alive(e);
        let Some(b) = sim.ecs.bodies.get(e) else { return };
        let me_name = self.name(e);
        self.overlays(&painter, rect, e);

        // crosshair: gap grows with the weapon's current dispersion (hidden when sighting)
        if alive && self.vm.ads < 0.6 {
            let a = sim.ecs.arsenals.get(e);
            let w = a.map_or(1, |a| a.current);
            let wc = &sim.cfg.weapons[w];
            let ads = sim.ecs.pilots.get(e).is_some_and(|p| p.last_cmd.held(buttons::ADS));
            let moa = if ads { wc.spread_moa_ads } else { wc.spread_moa_hip } * (1.0 + b.vel.length() / 5.0);
            let gap = (moa as f32 * 0.29e-3 * rect.height() / (2.0 * (35f32).to_radians().tan()) * 2.0).clamp(3.0, 60.0);
            let st = Stroke::new(1.5_f32, white);
            for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                painter.line_segment([c + egui::vec2(dx * gap, dy * gap), c + egui::vec2(dx * (gap + 8.0), dy * (gap + 8.0))], st);
            }
            if self.hit_marker.0 > 0.0 {
                let col = if self.hit_marker.1 { Color32::from_rgb(255, 70, 60) } else { white };
                let k = 6.0 + 10.0 * self.hit_marker.0;
                for (dx, dy) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                    painter.line_segment([c + egui::vec2(dx * 4.0, dy * 4.0), c + egui::vec2(dx * k, dy * k)], Stroke::new(2.0_f32, col));
                }
            }
        }
        // damage direction arcs
        let (_, fwd, up) = self.camera();
        let right = fwd.cross(up).normalize_or_zero();
        for &(d, life) in &self.hurt {
            let d = v3(d);
            let ang = d.dot(right).atan2(d.dot(fwd));
            let r = rect.height() * 0.22;
            let p = c + egui::vec2(ang.sin() * r, -ang.cos() * r);
            painter.circle_filled(p, 9.0, Color32::from_rgba_unmultiplied(255, 40, 30, (life / 1.5 * 200.0) as u8));
        }

        // neuro readout
        if let Some(n) = &self.neuro {
            let tof = n.trajectory.time_of_flight;
            let dist = n.trajectory.points.first().zip(n.trajectory.points.last()).map_or(0.0, |(a, b)| (*b - *a).length());
            let txt = format!("НЕЙРО  {:.0} м · {:.2} с · снос {:.1} м", dist, tof, n.deviation);
            stext(&painter, c + egui::vec2(0.0, 46.0), Align2::CENTER_TOP, txt, f(14.0), Color32::from_rgb(110, 255, 150));
            if let Some(&p) = n.trajectory.points.last() {
                if let Some(s) = project(view_proj, v3(p), rect) {
                    painter.circle_stroke(s, 7.0, Stroke::new(1.5_f32, Color32::from_rgb(110, 255, 150)));
                }
            }
        }

        // vitals (bottom left)
        let h = sim.ecs.health.get(e);
        let suit = sim.ecs.suits.get(e);
        let hp = h.map_or(0.0, |h| (h.hp / h.max).clamp(0.0, 1.0)) as f32;
        let bl = Pos2::new(rect.left() + 24.0, rect.bottom() - 24.0);
        let bar = |y: f32, frac: f32, col: Color32, label: String| {
            let r = egui::Rect::from_min_size(Pos2::new(bl.x, bl.y - y), egui::vec2(220.0, 14.0));
            painter.rect_filled(r, 3.0, dim);
            painter.rect_filled(egui::Rect::from_min_size(r.min, egui::vec2(220.0 * frac.clamp(0.0, 1.0), 14.0)), 3.0, col);
            stext(&painter, Pos2::new(r.left() + 6.0, r.center().y), Align2::LEFT_CENTER, label, f(11.0), white);
        };
        bar(18.0, hp, Color32::from_rgb(200, 50, 50), format!("ЗДОРОВЬЕ {:.0}", hp * 100.0));
        if let Some(s) = suit {
            bar(
                38.0,
                (s.o2_s / sim.cfg.player.suit_o2_s) as f32,
                Color32::from_rgb(60, 140, 220),
                format!("O₂ {:.0} с{}", s.o2_s, if s.sealed { " · шлем закрыт" } else { "" }),
            );
            bar(
                58.0,
                (s.propellant_kg / sim.cfg.player.jetpack.propellant_kg) as f32,
                Color32::from_rgb(220, 160, 50),
                format!("РАНЕЦ {:.1} кг", s.propellant_kg),
            );
        }
        let p_kpa = sim.atmo.pressure(&sim.frame, sim.eye(e)) / 1000.0;
        let g = player::felt_g(b, &sim.frame);
        let tau = sim.ecs.clocks.get(e).map_or(0.0, |c| c.tau);
        let t_station = sim.tick as f64 * DT;
        let mode = match b.mode {
            MoveMode::Walk => "шаг",
            MoveMode::Float => "свободный полёт",
            MoveMode::Jetpack => "РАНЕЦ 6DOF",
        };
        let info = format!("{me_name} · {mode}\n{:.2} g · {:.1} кПа · v {:.1} м/с\nτ − t = {:+.3} мс", g, p_kpa, b.vel.length(), (tau - t_station) * 1e3);
        stext(&painter, Pos2::new(bl.x, bl.y - 78.0), Align2::LEFT_BOTTOM, info, f(13.0), white);

        // weapon + device (bottom right)
        let br = Pos2::new(rect.right() - 24.0, rect.bottom() - 44.0);
        if let Some(a) = sim.ecs.arsenals.get(e) {
            let wc = &sim.cfg.weapons[a.current];
            let ws = &a.slots[a.current];
            let state = if ws.reload > 0.0 { format!("перезарядка {:.1} с", ws.reload) } else { format!("{} / {}", ws.mag, ws.reserve) };
            stext(&painter, br, Align2::RIGHT_BOTTOM, format!("{}\n{}", wc.name, state), f(16.0), white);
        }
        if let Some(d) = sim.ecs.devices.get(e) {
            let col = if d.instability > 0.5 { Color32::from_rgb(255, 120, 60) } else { Color32::from_rgb(170, 140, 255) };
            let txt = format!(
                "{}{}\nзаряд {:.0} % · нестабильность {:.0} %",
                d.mode.label(),
                if d.active.is_some() {
                    " · АКТИВНО"
                } else if d.cooldown > 0.0 {
                    " · остывает"
                } else {
                    ""
                },
                100.0 * d.charge_j / sim.cfg.devices.capacity_j,
                100.0 * d.instability
            );
            stext(&painter, br - egui::vec2(0.0, 48.0), Align2::RIGHT_BOTTOM, txt, f(13.0), col);
        }

        // kill feed (top right)
        let mut y = rect.top() + 52.0;
        for it in &self.feed {
            stext(&painter, Pos2::new(rect.right() - 16.0, y), Align2::RIGHT_TOP, &it.text, f(14.0), it.color);
            y += 20.0;
        }

        // spectator / death banner
        if self.me.is_none() {
            let bot = self.bots.bots.iter().find(|x| x.e == e);
            let t = format!(
                "наблюдение: {}  [{}]  — клавиши [ и ] — другой бот",
                me_name,
                bot.map_or(String::new(), |b| format!("{} · {}", b.diff.name, b.goal.label()))
            );
            let p = Pos2::new(c.x, rect.top() + 128.0);
            let g = painter.layout_no_wrap(t, f(15.0), white);
            let r = egui::Rect::from_center_size(p + egui::vec2(0.0, g.size().y * 0.5), g.size() + egui::vec2(20.0, 8.0));
            painter.rect_filled(r, 6.0, Color32::from_black_alpha(150));
            painter.galley(r.min + egui::vec2(10.0, 4.0), g, white);
        } else if !alive {
            let (by, cause) = h.map_or((None, None), |h| (h.last_attacker, h.cause));
            let t = format!(
                "ВЫ ПОГИБЛИ{}{}\nвозрождение через {:.1} с",
                cause.map_or(String::new(), |c| format!(" — {}", cause_ru(c, sim))),
                by.map_or(String::new(), |k| format!(" ({})", self.name(k))),
                h.map_or(0.0, |h| h.respawn_in.max(0.0))
            );
            stext(&painter, c - egui::vec2(0.0, 80.0), Align2::CENTER_CENTER, t, f(24.0), Color32::from_rgb(255, 90, 80));
        }
        if sim.game.over || self.scoreboard {
            self.draw_scoreboard(&painter, rect);
        }
    }

    /// Screen-space effects: scope, damage vignette, low-O₂ tint, kill confirmation, acoustic radar.
    fn overlays(&self, painter: &egui::Painter, rect: egui::Rect, e: Entity) {
        let c = rect.center();
        let sim = &self.sim;
        if self.scoped() {
            // scope: dark outside a circle, fine crosshair with mil dots
            let r = rect.height() * 0.46;
            let ring = 2000.0;
            painter.circle_stroke(c, r + ring * 0.5, Stroke::new(ring, Color32::BLACK));
            painter.circle_stroke(c, r, Stroke::new(3.0_f32, Color32::from_gray(20)));
            let st = Stroke::new(1.2_f32, Color32::from_rgba_unmultiplied(10, 10, 10, 230));
            painter.line_segment([c - egui::vec2(r, 0.0), c + egui::vec2(r, 0.0)], st);
            painter.line_segment([c - egui::vec2(0.0, r), c + egui::vec2(0.0, r)], st);
            for k in 1..=8 {
                let d = k as f32 * r / 9.0;
                for p in [c + egui::vec2(d, 0.0), c - egui::vec2(d, 0.0), c + egui::vec2(0.0, d), c - egui::vec2(0.0, d)] {
                    painter.circle_filled(p, 1.8, Color32::from_gray(15));
                }
            }
            painter.circle_filled(c, 1.5, Color32::from_rgb(255, 40, 30));
        } else if self.vm.ads > 0.6 && sim.is_alive(e) {
            painter.circle_filled(c, 1.3, Color32::from_rgba_unmultiplied(255, 255, 255, 120));
        }
        // red vignette: recent damage + low health pulse; blue when oxygen runs out
        let hp = sim.ecs.health.get(e).map_or(1.0, |h| (h.hp / h.max) as f32);
        let recent = self.hurt.iter().map(|h| h.1 / 1.5).fold(0.0f32, f32::max);
        let pulse = if hp < 0.35 && sim.is_alive(e) { (0.35 - hp) * 2.0 * (0.6 + 0.4 * (self.time as f32 * 6.0).sin()) } else { 0.0 };
        let red = (recent * 0.7 + pulse).min(1.0);
        let o2 = sim.ecs.suits.get(e).map_or(1.0, |s| (s.o2_s / 60.0) as f32).min(1.0);
        let blue = (1.0 - o2).max(0.0);
        for (amt, col) in [(red, [200u8, 10, 10]), (blue, [10, 40, 140])] {
            if amt > 0.01 {
                vignette(painter, rect, Color32::from_rgba_unmultiplied(col[0], col[1], col[2], (amt * 170.0) as u8));
            }
        }
        // kill confirmation under the crosshair
        if self.kill_confirm.0 > 0.0 {
            let a = (self.kill_confirm.0 / 0.4).min(1.0);
            let col = if self.kill_confirm.2 {
                Color32::from_rgba_unmultiplied(255, 210, 80, (a * 255.0) as u8)
            } else {
                Color32::from_rgba_unmultiplied(255, 90, 70, (a * 255.0) as u8)
            };
            let t = format!("✖  {}{}", self.kill_confirm.1, if self.kill_confirm.2 { "  · в голову" } else { "" });
            let pop = 1.0 + 0.25 * ((self.kill_confirm.0 - 2.0).max(0.0) / 0.2);
            stext(painter, c + egui::vec2(0.0, 46.0), Align2::CENTER_TOP, t, FontId::proportional(18.0 * pop), col);
        }
        // acoustic radar: what this suit's microphones heard (wavefront delay, no sound in vacuum)
        let Some(b) = sim.ecs.bodies.get(e) else { return };
        let rc = Pos2::new(rect.left() + 84.0, rect.bottom() - 210.0);
        let rr = 58.0;
        painter.circle_filled(rc, rr, Color32::from_rgba_unmultiplied(5, 12, 18, 150));
        painter.circle_stroke(rc, rr, Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(120, 200, 255, 90)));
        painter.circle_stroke(rc, rr * 0.5, Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(120, 200, 255, 50)));
        let (_, fwd, up) = self.camera();
        let left = up.cross(fwd).normalize_or(Vec3::X);
        let f = (fwd - up * fwd.dot(up)).normalize_or(Vec3::Z);
        let range = 80.0;
        let now = sim.tick;
        for snd in &sim.sounds {
            if snd.source == e || !snd.in_air || !matches!(snd.kind, SoundKind::Shot | SoundKind::Step | SoundKind::Device | SoundKind::Break) {
                continue;
            }
            let d = v3(snd.pos - b.pos);
            let dist = d.length();
            let arrive = snd.tick as f32 + dist / 343.0 * 60.0;
            let age = (now as f32 - arrive) / 60.0;
            if dist > range * 1.8 || !(0.0..2.5).contains(&age) {
                continue;
            }
            let x = -d.dot(left) / range;
            let y = -d.dot(f) / range;
            let v = egui::vec2(x, y);
            let v = if v.length() > 1.0 { v / v.length() } else { v };
            let a = ((1.0 - age / 2.5) * 255.0) as u8;
            let (col, rad) = match snd.kind {
                SoundKind::Shot => (Color32::from_rgba_unmultiplied(255, 80, 60, a), 3.5),
                SoundKind::Step => (Color32::from_rgba_unmultiplied(230, 230, 230, a / 2), 2.0),
                _ => (Color32::from_rgba_unmultiplied(120, 180, 255, a), 3.0),
            };
            painter.circle_filled(rc + v * rr, rad, col);
        }
        painter.circle_filled(rc, 2.5, Color32::from_rgb(120, 255, 160));
        stext(painter, rc + egui::vec2(0.0, -rr - 4.0), Align2::CENTER_BOTTOM, "акустика 80 м", FontId::proportional(11.0), Color32::from_gray(190));
    }

    fn draw_scoreboard(&self, painter: &egui::Painter, rect: egui::Rect) {
        let sim = &self.sim;
        let mut rows: Vec<(u32, u32, String, f64, String)> = sim
            .players
            .iter()
            .filter_map(|&e| {
                let s = sim.ecs.stats.get(e)?;
                let p = sim.ecs.pilots.get(e)?;
                let lvl = self.bots.bots.iter().find(|b| b.e == e).map_or("человек".to_string(), |b| b.diff.name.clone());
                let acc = if s.shots > 0 { 100.0 * s.hits as f64 / s.shots as f64 } else { 0.0 };
                Some((s.kills, s.deaths, p.name.clone(), acc, lvl))
            })
            .collect();
        rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let w = 520.0;
        let h = 60.0 + 22.0 * rows.len() as f32;
        let r = egui::Rect::from_center_size(rect.center(), egui::vec2(w, h));
        painter.rect_filled(r, 6.0, Color32::from_rgba_unmultiplied(10, 12, 18, 210));
        let font = FontId::monospace(14.0);
        let mut y = r.top() + 14.0;
        let t = sim.tick as f64 * DT;
        let header = format!("СХВАТКА  {:02}:{:02} / {:.0} мин", (t / 60.0) as u32, (t % 60.0) as u32, sim.cfg.rules.time_limit_s / 60.0);
        painter.text(Pos2::new(r.center().x, y), Align2::CENTER_TOP, header, FontId::proportional(16.0), Color32::WHITE);
        y += 30.0;
        for (k, d, n, acc, lvl) in rows {
            let line = format!("{:<12} {:<9} {:>4} {:>4} {:>6.1}%", n, lvl, k, d, acc);
            painter.text(Pos2::new(r.left() + 20.0, y), Align2::LEFT_TOP, line, font.clone(), Color32::from_gray(230));
            y += 22.0;
        }
    }

    /// One-line status for the main HUD.
    pub fn status(&self) -> String {
        format!("тик {} · sim+AI {:.2} мс", self.sim.tick, self.tick_ms)
    }

    pub fn device_mode(&self) -> Option<DeviceMode> {
        self.sim.ecs.devices.get(self.viewed()).map(|d| d.mode)
    }
}

/// Spacesuit figure made of ellipsoids: legs swing with the stride, arms hold
/// the weapon along the aim; the dead lie on their back.
/// Camera-facing ring of ribbons.
fn ring(out: &mut Vec<LineVertex>, c: Vec3, normal: Vec3, r: f32, cam: Vec3, col: [f32; 3]) {
    let n = normal.normalize_or(Vec3::Z);
    let a = n.any_orthonormal_vector();
    let b = n.cross(a);
    let pts: Vec<Vec3> = (0..=40).map(|i| i as f32 / 40.0 * std::f32::consts::TAU).map(|t| c + (a * t.cos() + b * t.sin()) * r).collect();
    crate::ribbon(out, &pts, cam, col, 1.0);
}

fn project(vp: glam::Mat4, p: Vec3, rect: egui::Rect) -> Option<Pos2> {
    let c = vp * p.extend(1.0);
    if c.w <= 0.0 {
        return None;
    }
    let n = c.truncate() / c.w;
    Some(Pos2::new(rect.left() + (n.x * 0.5 + 0.5) * rect.width(), rect.top() + (0.5 - n.y * 0.5) * rect.height()))
}

#[allow(dead_code)]
fn quat_d(q: DQuat) -> Quat {
    q.as_quat()
}

/// HUD text with a 1-px dark drop shadow so it stays readable over bright sky / walls.
fn stext(p: &egui::Painter, pos: egui::Pos2, align: egui::Align2, text: impl ToString, font: egui::FontId, color: egui::Color32) {
    let t = text.to_string();
    p.text(pos + egui::vec2(1.0, 1.0), align, &t, font.clone(), egui::Color32::from_black_alpha(190));
    p.text(pos, align, t, font, color);
}

/// Darkened screen edges in `col` (alpha = strength).
fn vignette(p: &egui::Painter, rect: egui::Rect, col: egui::Color32) {
    let clear = egui::Color32::TRANSPARENT;
    let inner = rect.shrink2(egui::vec2(rect.width() * 0.22, rect.height() * 0.22));
    let mut m = egui::Mesh::default();
    let o = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom()];
    let i = [inner.left_top(), inner.right_top(), inner.right_bottom(), inner.left_bottom()];
    for k in 0..4 {
        m.colored_vertex(o[k], col);
        m.colored_vertex(i[k], clear);
    }
    for k in 0..4u32 {
        let n = (k + 1) % 4;
        m.add_triangle(2 * k, 2 * n, 2 * k + 1);
        m.add_triangle(2 * n, 2 * n + 1, 2 * k + 1);
    }
    p.add(egui::Shape::mesh(m));
}

fn chase_override() -> Option<(f64, f64, f64)> {
    static V: std::sync::OnceLock<Option<(f64, f64, f64)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let s = std::env::var("ONEILL_CHASE").ok()?;
        let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (v.len() == 3).then(|| (v[0], v[1], v[2]))
    })
}
