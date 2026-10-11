//! Bot controller: perception → utility brain → path / steering / fire control
//! → a `Command`, exactly like a human client would send. Bots never touch
//! the simulation state directly, so they work unchanged over a network.

use crate::aim::{self, Solution};
use crate::brain::{self, Blackboard, Goal};
use crate::config::Difficulty;
use crate::nav::{NavGrid, PathScratch, Waypoint};
use crate::nav3d;
use crate::perception::Perception;
use gameplay::det::{self, Rng};
use gameplay::sim::Sim;
use gameplay::{buttons, geom, player, Command, DeviceMode, Entity, Event, MoveMode, DT, TICK_HZ};
use glam::DVec3;

const SLOT_PISTOL: usize = 0;
const SLOT_RIFLE: usize = 1;
const SLOT_SNIPER: usize = 2;
/// A* node budget for one re-plan (~1–2 ms worst case)
const PATH_BUDGET: usize = 60_000;

#[derive(Clone, Debug, Default)]
pub struct BotStats {
    pub decisions: [u32; 6],
    pub replans: u32,
    pub path_fail: u32,
    pub stuck: u32,
    pub jet_recoveries: u32,
    pub vaults: u32,
    pub solutions: u32,
}

#[derive(Clone)]
pub struct Bot {
    pub id: u8,
    pub e: Entity,
    pub diff: Difficulty,
    pub per: Perception,
    pub goal: Goal,
    goal_since: u32,
    pub target: Option<Entity>,
    path: Vec<Waypoint>,
    path_i: usize,
    path_goal: Option<(f64, f64)>,
    next_replan: u32,
    patrol_i: usize,
    cover_spot: Option<(f64, f64)>,
    flank_side: f64,
    strafe: f64,
    strafe_until: u32,
    aim_noise: (f64, f64),
    aim_noise_next: (f64, f64),
    noise_tick: u32,
    solution: Option<(Solution, u32, Entity)>,
    burst: u32,
    burst_pause_until: u32,
    prev_buttons: u16,
    stuck_ref: (DVec3, u32),
    unstick_until: u32,
    device_plan: Option<DeviceMode>,
    airborne_since: Option<u32>,
    /// learned aim-point offset from watching own tracers ("walking the fire in")
    aim_bias: DVec3,
    bias_ref: Option<(Entity, f64)>,
    pending_tracers: Vec<(u32, Entity, DVec3)>,
    pub stats: BotStats,
    rng: Rng,
}

pub struct Bots {
    pub bots: Vec<Bot>,
    pub nav: NavGrid,
    scratch: PathScratch,
    ray: Vec<u32>,
    cover: Vec<(f64, f64)>,
    mantle: f64,
    step: f64,
    /// one expensive A* per tick across all bots
    planned_this_tick: bool,
}

fn surface_dir(p: DVec3, du: f64, dv: f64) -> DVec3 {
    let (e, _, n) = geom::basis_at(p);
    (e * du + n * dv).normalize_or_zero()
}

impl Bots {
    /// Controls every pilot flagged as a bot; `levels` are assigned round-robin.
    pub fn new(sim: &Sim, levels: &[Difficulty]) -> Bots {
        let nav = NavGrid::build(&sim.arena, 4.0, sim.cfg.player.mantle_height, sim.cfg.player.step_height);
        let mut bots = Vec::new();
        for (k, &e) in sim.players.iter().enumerate() {
            let Some(p) = sim.ecs.pilots.get(e) else { continue };
            if !p.bot {
                continue;
            }
            let diff = levels[bots.len() % levels.len()].clone();
            bots.push(Bot::new(k as u8, e, diff, sim.seed));
        }
        Bots {
            bots,
            nav,
            scratch: PathScratch::default(),
            ray: Vec::new(),
            cover: Vec::new(),
            mantle: sim.cfg.player.mantle_height,
            step: sim.cfg.player.step_height,
            planned_this_tick: false,
        }
    }

    /// Commands for the coming tick (`sim.tick`), given the state after the last step.
    pub fn think(&mut self, sim: &Sim) -> Vec<Command> {
        for ev in &sim.events {
            match *ev {
                Event::BoxDestroyed { index, .. } => self.nav.on_box_destroyed(&sim.arena, index, self.mantle, self.step),
                Event::PaneBroken { key, .. } => self.nav.on_pane_broken(&sim.arena, key, self.mantle, self.step),
                Event::Respawn { who } | Event::Kill { victim: who, .. } => {
                    if let Some(b) = self.bots.iter_mut().find(|b| b.e == who) {
                        b.reset(sim.tick);
                    }
                }
                Event::Shot { shooter, weapon } => {
                    if let Some(b) = self.bots.iter_mut().find(|b| b.e == shooter) {
                        b.watch_tracer(sim, weapon as usize);
                    }
                }
                _ => {}
            }
        }
        self.planned_this_tick = false;
        let mut out = Vec::with_capacity(self.bots.len());
        let n = self.bots.len();
        // rotate who gets the first shot at path planning
        let first = (sim.tick as usize) % n.max(1);
        for k in 0..n {
            let i = (first + k) % n;
            let mut bot = std::mem::replace(&mut self.bots[i], Bot::placeholder());
            let cmd = bot.think(sim, self);
            self.bots[i] = bot;
            out.push(cmd);
        }
        out.sort_by_key(|c| c.player);
        out
    }
}

impl Bot {
    fn placeholder() -> Bot {
        let d = crate::config::builtin().remove(0);
        Bot::new(255, Entity::default(), d, 0)
    }

    pub fn new(id: u8, e: Entity, diff: Difficulty, seed: u64) -> Bot {
        let mut rng = Rng::keyed(&[seed, id as u64, 0xB07]);
        let patrol_i = rng.next_u64() as usize;
        Bot {
            id,
            e,
            diff,
            per: Perception::default(),
            goal: Goal::Patrol,
            goal_since: 0,
            target: None,
            path: Vec::new(),
            path_i: 0,
            path_goal: None,
            next_replan: 0,
            patrol_i,
            cover_spot: None,
            flank_side: 1.0,
            strafe: 0.0,
            strafe_until: 0,
            aim_noise: (0.0, 0.0),
            aim_noise_next: (0.0, 0.0),
            noise_tick: 0,
            solution: None,
            burst: 0,
            burst_pause_until: 0,
            prev_buttons: 0,
            stuck_ref: (DVec3::ZERO, 0),
            unstick_until: 0,
            device_plan: None,
            airborne_since: None,
            aim_bias: DVec3::ZERO,
            bias_ref: None,
            pending_tracers: Vec::new(),
            stats: BotStats::default(),
            rng,
        }
    }

    fn reset(&mut self, tick: u32) {
        self.per.clear();
        self.goal = Goal::Patrol;
        self.goal_since = tick;
        self.target = None;
        self.path.clear();
        self.path_goal = None;
        self.cover_spot = None;
        self.solution = None;
        self.device_plan = None;
        self.airborne_since = None;
        self.aim_bias = DVec3::ZERO;
        self.bias_ref = None;
        self.pending_tracers.clear();
        self.stuck_ref = (DVec3::ZERO, tick);
        let n = self.rng.next_u64() as usize;
        self.patrol_i = self.patrol_i.wrapping_add(n % 7 + 1);
    }

    fn think(&mut self, sim: &Sim, ctx: &mut Bots) -> Command {
        let tick = sim.tick;
        let mut cmd = Command::new(tick, self.id);
        if !sim.is_alive(self.e) {
            self.prev_buttons = 0;
            return cmd;
        }
        let body = sim.ecs.bodies.get(self.e).expect("bot body").clone();
        let hp = sim.ecs.health.get(self.e).map_or(1.0, |h| h.hp / h.max);
        let arsenal = sim.ecs.arsenals.get(self.e).expect("arsenal").clone();
        let me_s = sim.arena.surf(body.pos);
        let eye = sim.eye(self.e);

        self.learn_from_tracers(tick);
        // ---- perception ----
        self.per.listen(sim, self.e, &self.diff, &mut self.rng);
        if (tick + self.id as u32).is_multiple_of(6) {
            self.per.look(sim, self.e, &self.diff, &mut ctx.ray);
        }

        // pick the target: nearest visible, else freshest known
        let mut vis: Vec<(f64, Entity)> =
            self.per.contacts.iter().filter(|c| c.visible && sim.is_alive(c.e)).map(|c| ((c.pos - body.pos).length(), c.e)).collect();
        vis.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.index.cmp(&b.1.index)));
        let known = self
            .per
            .contacts
            .iter()
            .filter(|c| !c.visible && sim.is_alive(c.e))
            .max_by_key(|c| (c.last_seen.max(c.last_heard), c.e.index))
            .map(|c| (c.e, c.pos, tick.saturating_sub(c.last_seen.max(c.last_heard)) as f64 * DT));
        self.target = vis.first().map(|v| v.1).or(known.map(|k| k.0));
        let threat_pos = self.target.and_then(|t| self.per.contact(t)).map(|c| c.pos);

        // ---- weapons state ----
        let w = arsenal.current;
        let ws = &arsenal.slots[w];
        let reloading = ws.reload > 0.0;
        let total = |s: usize| arsenal.slots.get(s).map_or(0, |x| x.mag + x.reserve);
        let out_of_ammo = (0..arsenal.slots.len()).all(|s| total(s) == 0);

        // ---- decide (10 Hz) ----
        if (tick + self.id as u32).is_multiple_of(6) {
            let in_cover = self.cover_spot.is_some_and(|c| (c.0 - me_s.u).hypot(c.1 - me_s.v) < 2.5);
            let cover_available = match threat_pos {
                Some(tp) if self.goal != Goal::SeekCover => {
                    let ts = sim.arena.surf(tp);
                    ctx.nav.cover_near(me_s.u, me_s.v, 36.0, (ts.u, ts.v), &mut ctx.cover);
                    !ctx.cover.is_empty()
                }
                Some(_) => true,
                None => false,
            };
            let bb = Blackboard {
                hp,
                visible: vis.first().map(|v| v.0),
                known_age: known.map(|k| k.2),
                hit_age: self.per.under_fire.map(|(t, _)| tick.saturating_sub(t) as f64 * DT),
                noise_age: self.per.noise.map(|(_, t)| tick.saturating_sub(t) as f64 * DT),
                reloading,
                out_of_ammo,
                cover_available,
                in_cover,
                n_visible: vis.len(),
            };
            let age = tick.saturating_sub(self.goal_since) as f64 * DT;
            let g = brain::choose(self.goal, age, &bb, &self.diff);
            if g != self.goal {
                self.goal = g;
                self.goal_since = tick;
                self.path.clear();
                self.path_goal = None;
                if g != Goal::SeekCover {
                    self.cover_spot = None;
                }
                if g == Goal::Flank {
                    self.flank_side = if self.rng.f64() < 0.5 { -1.0 } else { 1.0 };
                }
                self.stats.decisions[g as usize] += 1;
            }
            self.plan_device(sim, hp, vis.first().map(|v| v.0));
        }

        // ---- where to go ----
        let mut run = false;
        let mut crouch = false;
        let mut dest: Option<(f64, f64)> = None;
        match self.goal {
            Goal::Patrol => {
                let pts = &sim.arena.st.patrol;
                if !pts.is_empty() {
                    let p = pts[self.patrol_i % pts.len()];
                    if (p.0 - me_s.u).hypot(p.1 - me_s.v) < 6.0 {
                        self.patrol_i = self.patrol_i.wrapping_add(1 + self.rng.index(3));
                    }
                    dest = Some(p);
                }
            }
            Goal::Investigate => {
                let p = known.map(|k| k.1).or(self.per.noise.map(|n| n.0));
                if let Some(p) = p {
                    let s = sim.arena.surf(p);
                    dest = Some((s.u, s.v));
                    run = true;
                }
                if dest.is_some_and(|d| (d.0 - me_s.u).hypot(d.1 - me_s.v) < 5.0) {
                    self.per.noise = None;
                }
            }
            Goal::Attack => {}
            Goal::SeekCover => {
                if self.cover_spot.is_none() {
                    if let Some(tp) = threat_pos {
                        self.cover_spot = self.find_cover(sim, ctx, me_s.u, me_s.v, tp);
                    }
                }
                if let Some(c) = self.cover_spot {
                    if (c.0 - me_s.u).hypot(c.1 - me_s.v) < 2.0 {
                        crouch = true;
                    } else {
                        dest = Some(c);
                        run = true;
                    }
                }
            }
            Goal::Retreat => {
                if let Some(tp) = threat_pos {
                    let ts = sim.arena.surf(tp);
                    let (du, dv) = (me_s.u - ts.u, me_s.v - ts.v);
                    let l = du.hypot(dv).max(1.0);
                    let d = self.path_goal.unwrap_or((me_s.u + du / l * 80.0, me_s.v + dv / l * 80.0));
                    dest = Some(clamp_to_arena(sim, d));
                    run = true;
                }
            }
            Goal::Flank => {
                if let Some((_, kp, _)) = known {
                    let ks = sim.arena.surf(kp);
                    let d = self.path_goal.unwrap_or_else(|| {
                        let (du, dv) = (ks.u - me_s.u, ks.v - me_s.v);
                        let l = du.hypot(dv).max(1.0);
                        let (pu, pv) = (-dv / l * self.flank_side, du / l * self.flank_side);
                        let r = 45.0f64.min(0.7 * l);
                        (ks.u + pu * r - du / l * 15.0, ks.v + pv * r - dv / l * 15.0)
                    });
                    dest = Some(clamp_to_arena(sim, d));
                    run = true;
                }
            }
        }

        // ---- path following ----
        let mut wish = DVec3::ZERO;
        let mut vault_now = false;
        if let Some(d) = dest {
            let arrived = (d.0 - me_s.u).hypot(d.1 - me_s.v) < 2.5;
            let moved_goal = self.path_goal.is_none_or(|g| (g.0 - d.0).hypot(g.1 - d.1) > 8.0);
            let finished = self.path_i >= self.path.len();
            let since = tick + TICK_HZ * 4 - self.next_replan.min(tick + TICK_HZ * 4);
            let need =
                !arrived && ((moved_goal && since > TICK_HZ / 2) || (finished && since > TICK_HZ) || tick >= self.next_replan || self.path_goal.is_none());
            if need && !ctx.planned_this_tick {
                ctx.planned_this_tick = true;
                self.stats.replans += 1;
                match ctx.nav.find_path(&mut ctx.scratch, (me_s.u, me_s.v), d, PATH_BUDGET) {
                    Some(p) => {
                        self.path = p;
                        self.path_i = 0;
                    }
                    None => {
                        self.stats.path_fail += 1;
                        self.path.clear();
                    }
                }
                self.path_goal = Some(d);
                self.next_replan = tick + TICK_HZ * 4 + self.rng.index(TICK_HZ as usize) as u32;
            }
            while self.path_i < self.path.len() {
                let wp = self.path[self.path_i];
                let dd = (wp.u - me_s.u).hypot(wp.v - me_s.v);
                let last = self.path_i + 1 == self.path.len();
                if dd < if last { 1.2 } else { 2.0 } {
                    self.path_i += 1;
                } else {
                    break;
                }
            }
            if let Some(wp) = self.path.get(self.path_i) {
                wish = surface_dir(body.pos, wp.u - me_s.u, wp.v - me_s.v);
                if wp.vault && (wp.u - me_s.u).hypot(wp.v - me_s.v) < 2.4 {
                    vault_now = true;
                }
            }
        }
        if self.goal == Goal::Attack {
            wish = self.combat_move(sim, ctx, &body, me_s.u, me_s.v, threat_pos, w);
        }

        // ---- stuck detection ----
        if tick.saturating_sub(self.stuck_ref.1) > TICK_HZ * 2 {
            let moved = (body.pos - self.stuck_ref.0).length();
            if wish != DVec3::ZERO && moved < 0.6 && body.mode == MoveMode::Walk {
                self.stats.stuck += 1;
                self.unstick_until = tick + TICK_HZ / 2;
                self.strafe = if self.rng.f64() < 0.5 { -1.0 } else { 1.0 };
                self.next_replan = tick;
            }
            self.stuck_ref = (body.pos, tick);
        }
        let unsticking = tick < self.unstick_until;
        if unsticking {
            let (_, up, _) = geom::basis_at(body.pos);
            wish = (wish.cross(up) * self.strafe - wish * 0.3).normalize_or_zero();
        }

        // ---- look ----
        let mut look_dir = if wish != DVec3::ZERO { wish } else { sim.aim_dir(self.e) };
        if let (false, Some(tp)) = (vault_now || unsticking, threat_pos) {
            look_dir = (tp + body.up * 1.2 - eye).normalize_or(look_dir);
        }
        let mut fire = false;
        let mut ads = false;
        let mut want_slot = None;
        if let Some(&(dist, t)) = vis.first() {
            want_slot = Some(self.weapon_for(sim, dist, &arsenal));
            if let Some(dir) = self.fire_solution(sim, t, eye, &body, w) {
                look_dir = self.noisy(dir, tick, body.vel.length());
                // target ≈ 0.3 m disc; a looser trigger only at short range
                let tol = 0.3 / dist.max(1.0) + self.diff.fire_tolerance_mrad * 1e-3 * (40.0 / dist.max(1.0)).clamp(0.15, 1.0);
                let err = sim.aim_dir(self.e).angle_between(look_dir);
                let first_seen = self.per.contact(t).map_or(tick, |c| c.first_seen);
                let ready = tick.saturating_sub(first_seen) as f64 * DT >= self.diff.reaction_s;
                // the less careful the bot, the further beyond its real ability it shoots
                let in_range = dist < self.effective_range(sim, w) * (1.25 + 1.5 * (1.0 - self.diff.caution));
                fire = ready && in_range && err < tol && !vault_now && self.goal != Goal::Retreat;
                ads = dist > 40.0;
            }
        } else if !reloading && ws.mag * 2 < sim.cfg.weapons[w].magazine && ws.reserve > 0 {
            cmd.buttons |= buttons::RELOAD;
        }
        if want_slot.is_none() && w != SLOT_RIFLE && total(SLOT_RIFLE) > 0 && vis.is_empty() {
            want_slot = Some(SLOT_RIFLE);
        }
        if let Some(s) = want_slot {
            if s != w {
                cmd.weapon = s as u8 + 1;
            }
        }

        // ---- trigger discipline ----
        let auto = sim.cfg.weapons[w].automatic;
        if fire && ws.mag > 0 && tick >= self.burst_pause_until {
            if auto {
                cmd.buttons |= buttons::FIRE;
                if self.prev_buttons & buttons::FIRE != 0 && ws.cooldown > 0.0 && ws.cooldown + DT >= 60.0 / sim.cfg.weapons[w].rpm {
                    self.burst += 1;
                }
                if self.burst >= self.diff.burst {
                    self.burst = 0;
                    self.burst_pause_until = tick + (TICK_HZ as f64 * 0.35) as u32;
                }
            } else if self.prev_buttons & buttons::FIRE == 0 {
                cmd.buttons |= buttons::FIRE;
            }
        } else if !fire {
            self.burst = 0;
        }
        if fire && ws.mag == 0 && !reloading {
            cmd.buttons |= buttons::RELOAD;
        }
        if ads {
            cmd.buttons |= buttons::ADS;
        }

        // ---- device ----
        if let Some(mode) = self.device_plan {
            if let Some(dev) = sim.ecs.devices.get(self.e) {
                if dev.mode != mode {
                    if self.prev_buttons & buttons::DEVICE_NEXT == 0 {
                        cmd.buttons |= buttons::DEVICE_NEXT;
                    }
                } else if self.prev_buttons & buttons::DEVICE == 0 {
                    if mode == DeviceMode::Fold {
                        // throw the fold away from the threat, slightly upwards
                        if let Some(tp) = threat_pos {
                            look_dir = ((body.pos - tp).normalize_or_zero() + body.up * 0.15).normalize_or(look_dir);
                        }
                    }
                    cmd.buttons |= buttons::DEVICE;
                    self.device_plan = None;
                }
            }
        }

        // ---- locomotion ----
        let felt = player::felt_g(&body, &sim.frame);
        let floor = sim.arena.floor(me_s.u, me_s.v).map_or(0.0, |f| f.0);
        let height = me_s.h - floor;
        let airborne = body.mode != MoveMode::Walk && !body.grounded;
        if airborne {
            self.airborne_since.get_or_insert(tick);
        } else {
            self.airborne_since = None;
        }
        let long_air = self.airborne_since.is_some_and(|t| tick - t > TICK_HZ / 2) && (height > 4.0 || felt < sim.cfg.player.walk_min_g);
        let propellant = sim.ecs.suits.get(self.e).map_or(0.0, |s| s.propellant_kg);
        if body.mode == MoveMode::Jetpack {
            // fly down to the floor below and land softly, then cut the jet
            let mass = body.mass + propellant;
            let max_acc = sim.cfg.player.jetpack.thrust_n / mass;
            let goal = geom::surf_to_world(me_s.u, me_s.v, floor + 1.0) + wish * 10.0;
            let a = nav3d::steer(&sim.frame, body.pos, body.vel, goal, DVec3::ZERO, max_acc);
            let (f, r, u) = nav3d::to_axes(&body, a, max_acc);
            cmd.set_move(f, r, u);
            if height < 1.6 && body.vel.length() < 3.0 && self.prev_buttons & buttons::JETPACK == 0 {
                cmd.buttons |= buttons::JETPACK;
            }
            if threat_pos.is_none() {
                look_dir = (goal - eye).normalize_or(look_dir);
            }
        } else if long_air && propellant > 0.5 && self.prev_buttons & buttons::JETPACK == 0 && (height > 3.0 || body.vel.length() > 6.0) {
            cmd.buttons |= buttons::JETPACK;
            self.stats.jet_recoveries += 1;
        } else {
            let (e, up, n) = geom::basis_at(body.pos);
            let (sy, cy) = det::sin_cos(body.yaw);
            let ff = e * sy + n * cy;
            let fr = ff.cross(up);
            cmd.set_move(wish.dot(ff), wish.dot(fr), 0.0);
            if vault_now && body.grounded && self.prev_buttons & buttons::JUMP == 0 {
                cmd.buttons |= buttons::JUMP;
                self.stats.vaults += 1;
            }
            if run && !crouch && !ads {
                cmd.buttons |= buttons::RUN;
            }
            if crouch {
                cmd.buttons |= buttons::CROUCH;
            }
        }

        // ---- turn the head (rate-limited) ----
        let (dy, dp) = player::look_delta_to(&body, look_dir);
        let lim = self.diff.turn_rate * DT;
        let m = dy.hypot(dp);
        let k = if m > lim { lim / m } else { 1.0 };
        cmd.set_look(dy * k, dp * k, 0.0);

        self.prev_buttons = cmd.buttons;
        cmd
    }

    /// Range (m) at which this bot still puts ~half its aimed shots into a
    /// 0.45 m circle: own aim error and the weapon's dispersion combined.
    pub fn effective_range(&self, sim: &Sim, w: usize) -> f64 {
        let wc = &sim.cfg.weapons[w];
        let spread = wc.spread_moa_ads * gameplay::sim::MOA;
        let aim = self.diff.aim_error_mrad * 1e-3;
        let r = 0.45 / (aim * aim + spread * spread).sqrt();
        // the round must also get there with useful energy within its flight time
        r.min(wc.muzzle_velocity * wc.max_flight_s * 0.3)
    }

    fn weapon_for(&self, sim: &Sim, dist: f64, a: &gameplay::Arsenal) -> usize {
        let has = |s: usize| a.slots.get(s).is_some_and(|x| x.mag + x.reserve > 0);
        if has(SLOT_SNIPER)
            && dist > 0.8 * self.effective_range(sim, SLOT_RIFLE)
            && self.effective_range(sim, SLOT_SNIPER) > self.effective_range(sim, SLOT_RIFLE)
        {
            SLOT_SNIPER
        } else if has(SLOT_RIFLE) {
            SLOT_RIFLE
        } else if has(SLOT_SNIPER) {
            SLOT_SNIPER
        } else {
            SLOT_PISTOL
        }
    }

    /// Fire-control solution against `t`'s chest with the bot's own ballistic model (cached ~0.1 s).
    fn fire_solution(&mut self, sim: &Sim, t: Entity, eye: DVec3, body: &gameplay::Body, w: usize) -> Option<DVec3> {
        let tick = sim.tick;
        if let Some((s, at, who)) = self.solution {
            if who == t && tick.saturating_sub(at) < 6 {
                return Some(s.dir);
            }
        }
        let c = self.per.contact(t)?;
        let tb = sim.ecs.bodies.get(t)?;
        let dist = (c.pos - eye).length();
        if self.bias_ref.is_none_or(|(e, d)| e != t || (dist - d).abs() > 0.3 * d + 10.0) {
            self.aim_bias = DVec3::ZERO;
            self.bias_ref = Some((t, dist));
        }
        let target = c.pos + tb.up * (0.62 * tb.height) + self.aim_bias;
        let wc = &sim.cfg.weapons[w];
        let env = aim::model_env(sim.env(), &self.diff);
        let sol = aim::solve(&env, sim.round(w), wc.muzzle_velocity, eye, body.vel, target, c.vel * self.diff.lead, wc.max_flight_s.min(3.0))?;
        self.stats.solutions += 1;
        self.solution = Some((sol, tick, t));
        Some(sol.dir)
    }

    /// After a shot: the bot watches its tracer / impact (available ≈ one flight
    /// time later) and shifts its aim point against the observed miss. This is
    /// how a novice who knows nothing about Coriolis still walks fire onto a
    /// distant target after a few rounds.
    fn watch_tracer(&mut self, sim: &Sim, w: usize) {
        let Some(t) = self.target else { return };
        let Some(c) = self.per.contact(t) else { return };
        if !c.visible {
            return;
        }
        let Some(tb) = sim.ecs.bodies.get(t) else { return };
        let Some(p) = sim.ecs.projectiles.values().iter().filter(|p| p.owner == self.e).min_by(|a, b| a.age.total_cmp(&b.age)) else { return };
        let aim_point = c.pos + tb.up * (0.62 * tb.height);
        let (miss, tof) = gameplay::ballistics::closest_approach(&sim.env(), sim.round(w), p.pos, p.vel, aim_point, c.vel, 3.0, DT);
        let seen_at = sim.tick + (tof * TICK_HZ as f64) as u32 + (self.diff.reaction_s * TICK_HZ as f64) as u32;
        if self.pending_tracers.len() < 16 {
            self.pending_tracers.push((seen_at, t, miss));
        }
    }

    fn learn_from_tracers(&mut self, tick: u32) {
        let gain = 0.25 + 0.4 * self.diff.caution;
        let mut i = 0;
        while i < self.pending_tracers.len() {
            let (at, t, miss) = self.pending_tracers[i];
            if tick >= at {
                if self.bias_ref.is_some_and(|(e, _)| e == t) {
                    self.aim_bias -= miss * gain;
                    let l = self.aim_bias.length();
                    if l > 40.0 {
                        self.aim_bias *= 40.0 / l;
                    }
                    self.solution = None;
                }
                self.pending_tracers.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Adds the bot's hand tremor (smoothly varying, re-drawn every 0.4 s).
    fn noisy(&mut self, dir: DVec3, tick: u32, speed: f64) -> DVec3 {
        let period = (TICK_HZ as f64 * 0.4) as u32;
        if tick >= self.noise_tick + period {
            self.noise_tick = tick;
            self.aim_noise = self.aim_noise_next;
            let s = self.diff.aim_error_mrad * 1e-3 * (1.0 + speed / 4.0);
            self.aim_noise_next = (self.rng.normal() * s, self.rng.normal() * s);
        }
        let f = (tick - self.noise_tick) as f64 / period as f64;
        let ny = self.aim_noise.0 + (self.aim_noise_next.0 - self.aim_noise.0) * f;
        let np = self.aim_noise.1 + (self.aim_noise_next.1 - self.aim_noise.1) * f;
        let a1 = dir.any_orthonormal_vector();
        let a2 = dir.cross(a1);
        (dir + a1 * ny + a2 * np).normalize()
    }

    /// Strafing / range keeping while engaging.
    #[allow(clippy::too_many_arguments)]
    fn combat_move(&mut self, sim: &Sim, ctx: &Bots, body: &gameplay::Body, u: f64, v: f64, threat: Option<DVec3>, w: usize) -> DVec3 {
        let Some(tp) = threat else { return DVec3::ZERO };
        let ts = sim.arena.surf(tp);
        let (du, dv) = (ts.u - u, ts.v - v);
        let dist = du.hypot(dv).max(1e-3);
        let eff = self.effective_range(sim, w);
        let (pref_min, pref_max) = if w == SLOT_SNIPER { (150.0f64.min(0.5 * eff), 0.8 * eff) } else { (12.0, (0.7 * eff).clamp(25.0, 200.0)) };
        let tick = sim.tick;
        if tick >= self.strafe_until {
            let r = self.rng.f64();
            self.strafe = if r < 0.4 {
                -1.0
            } else if r < 0.8 {
                1.0
            } else {
                0.0
            };
            self.strafe_until = tick + (TICK_HZ as f64 * self.rng.range(0.5, 1.6)) as u32;
        }
        // novices do not dodge much and the sniper stays still to shoot
        let strafe = if w == SLOT_SNIPER { 0.0 } else { self.strafe * (0.4 + 0.6 * self.diff.caution) };
        let radial = if dist > pref_max {
            1.0
        } else if dist < pref_min {
            -1.0
        } else {
            0.0
        };
        let (fu, fv) = (du / dist, dv / dist);
        let (mu, mv) = (fu * radial - fv * strafe, fv * radial + fu * strafe);
        if mu == 0.0 && mv == 0.0 {
            return DVec3::ZERO;
        }
        let ahead = (u + mu * 3.0, v + mv * 3.0);
        if !ctx.nav.walkable_line((u, v), ahead) {
            self.strafe = -self.strafe;
            return DVec3::ZERO;
        }
        surface_dir(body.pos, mu, mv)
    }

    fn find_cover(&mut self, sim: &Sim, ctx: &mut Bots, u: f64, v: f64, threat: DVec3) -> Option<(f64, f64)> {
        let ts = sim.arena.surf(threat);
        ctx.nav.cover_near(u, v, 36.0, (ts.u, ts.v), &mut ctx.cover);
        let mut cands: Vec<(f64, (f64, f64))> =
            ctx.cover.iter().map(|&(cu, cv)| ((cu - u).hypot(cv - v) + 0.1 * (cu - ts.u).hypot(cv - ts.v).min(50.0), (cu, cv))).collect();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let threat_eye = threat + geom::up_at(threat) * 1.6;
        for (_, (cu, cv)) in cands.into_iter().take(12) {
            let floor = sim.arena.floor(cu, cv).map_or(0.0, |f| f.0);
            let head = geom::surf_to_world(cu, cv, floor + 1.0);
            if !sim.arena.visible(threat_eye, head, &mut ctx.ray) {
                return Some((cu, cv));
            }
        }
        None
    }

    fn plan_device(&mut self, sim: &Sim, hp: f64, vis_dist: Option<f64>) {
        if self.device_plan.is_some() {
            return;
        }
        let Some(dev) = sim.ecs.devices.get(self.e) else { return };
        let dc = &sim.cfg.devices;
        if dev.active.is_some() || dev.cooldown > 0.0 {
            return;
        }
        let hit_recent = self.per.under_fire.is_some_and(|(t, _)| sim.tick.saturating_sub(t) < TICK_HZ);
        // a careful bot does not gamble on an unstable device
        let max_instab = 0.6 - 0.45 * self.diff.caution;
        if dev.instability > max_instab {
            return;
        }
        let p = self.diff.device_use * 0.08;
        if hit_recent && hp < 0.7 && dev.charge_j >= dc.bubble.cost_j && self.rng.f64() < p {
            self.device_plan = Some(DeviceMode::Bubble);
        } else if self.goal == Goal::Retreat && hp < 0.35 && dev.charge_j >= dc.fold.cost_j && self.rng.f64() < p {
            self.device_plan = Some(DeviceMode::Fold);
        } else if let Some(d) = vis_dist {
            // throwing a micro black hole is a reckless move: only the incautious try it
            if (20.0..60.0).contains(&d) && dev.charge_j >= dc.micro_hole.cost_j && self.rng.f64() < p * 0.1 * (1.0 - self.diff.caution) {
                self.device_plan = Some(DeviceMode::MicroHole);
            }
        }
    }
}

fn clamp_to_arena(sim: &Sim, d: (f64, f64)) -> (f64, f64) {
    let s = &*sim.arena.st;
    (d.0.clamp(s.u0 + 10.0, s.u0 + s.size_u - 10.0), d.1.clamp(s.v0 + 10.0, s.v0 + s.size_v - 10.0))
}
