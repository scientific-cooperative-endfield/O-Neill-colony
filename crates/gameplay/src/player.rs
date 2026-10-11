//! First-person controller in the rotating station frame.
//!
//! The body is a capsule (feet point + height). Every tick:
//!
//! * the frame acceleration a = −Ω×(Ω×d) − 2Ω×v + Δa_PW + a_fields acts on it
//!   (so a runner heading spinward is 2ωv heavier, one heading antispinward
//!   lighter; near the axis everything floats);
//! * on a support, traction is limited by friction μ·N where N comes from the
//!   *actual* felt acceleration — in low gravity you cannot accelerate, stop or
//!   turn quickly, exactly as on the Moon;
//! * jumping adds a fixed take-off Δv (legs, not gravity, set it);
//! * aerodynamic drag (C_D·A ≈ 0.7 m²) uses the local air density and the
//!   outflow towards breaches — an open window drags people to it;
//! * with the jetpack, thrust acts along the body axes (6 DOF) and the mass
//!   drops as propellant is used (Tsiolkovsky).
//!
//! Landing / wall-impact speeds are reported for fall damage.

use crate::components::{Body, MoveMode, Suit};
use crate::config::PlayerCfg;
use crate::det;
use crate::fields::{shell_response, Crossing, FieldSet};
use crate::frame::{Atmosphere, StationFrame};
use crate::geom::{self, G0};
use crate::input::{buttons, Command};
use crate::map::{Arena, WINDOW_THICKNESS};
use glam::{DQuat, DVec3};

/// Human C_D·A (m²), standing / tucked.
pub const BODY_CDA: f64 = 0.7;

pub struct Ctx<'a> {
    pub cfg: &'a PlayerCfg,
    pub frame: &'a StationFrame,
    pub atmo: &'a Atmosphere,
    pub arena: &'a Arena,
    pub fields: &'a FieldSet,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepOut {
    /// normal impact speed against the ground or a wall this tick (m/s)
    pub impact: f64,
    pub footstep: bool,
    /// went through a fold / bounced on a shell
    pub crossed_fold: Option<f64>,
    pub bounced: bool,
    /// fell out of the hull this tick
    pub left_hull: bool,
}

pub fn new_body(cfg: &PlayerCfg, feet: DVec3, yaw: f64) -> Body {
    let up = geom::up_at(feet);
    Body {
        pos: feet,
        vel: DVec3::ZERO,
        mass: cfg.mass_kg,
        radius: cfg.radius,
        height: cfg.height,
        crouched: false,
        grounded: false,
        ground: None,
        ground_pane: None,
        mode: MoveMode::Walk,
        yaw,
        pitch: 0.0,
        orient: DQuat::IDENTITY,
        stride: 0.0,
        up,
    }
}

/// Eye position.
pub fn eye(b: &Body, cfg: &PlayerCfg) -> DVec3 {
    b.pos + b.axis() * (b.height - cfg.eye_below_top)
}

/// View direction, with extra muzzle climb `climb` (rad).
pub fn forward(b: &Body, climb: f64) -> DVec3 {
    match b.mode {
        MoveMode::Jetpack => {
            let q = b.orient * det::quat_axis_angle(DVec3::X, -climb);
            q * DVec3::Z
        }
        _ => {
            let (e, up, n) = geom::basis_at(b.pos);
            let (sy, cy) = det::sin_cos(b.yaw);
            let (sp, cp) = det::sin_cos((b.pitch + climb).clamp(-1.55, 1.55));
            ((e * sy + n * cy) * cp + up * sp).normalize()
        }
    }
}

/// Body orientation as a quaternion (z forward, y up) from walking yaw/pitch.
fn orient_from_view(b: &Body) -> DQuat {
    let f = forward(b, 0.0);
    let up0 = geom::up_at(b.pos);
    let r = up0.cross(f).normalize_or(DVec3::X); // body x
    let u = f.cross(r);
    DQuat::from_mat3(&glam::DMat3::from_cols(r, u, f)).normalize()
}

/// Walking yaw / pitch from a body orientation.
fn view_from_orient(b: &mut Body) {
    let f = b.orient * DVec3::Z;
    let (e, up, n) = geom::basis_at(b.pos);
    b.pitch = det::asin(f.dot(up)).clamp(-1.5, 1.5);
    b.yaw = det::atan2(f.dot(e), f.dot(n));
}

fn arena_u(a: &Arena, p: DVec3) -> f64 {
    a.surf(p).u
}

fn wish_speed(b: &Body, cfg: &PlayerCfg, cmd: &Command) -> f64 {
    if b.crouched {
        cfg.crouch_speed
    } else if cmd.held(buttons::RUN) {
        cfg.run_speed
    } else {
        cfg.walk_speed
    }
}

/// Advances one body by `dt` (already scaled by the local time rate).
pub fn step(b: &mut Body, suit: &mut Suit, cmd: &Command, prev_buttons: u16, ctx: &Ctx, dt: f64, scratch: &mut Vec<u32>) -> StepOut {
    let cfg = ctx.cfg;
    let mut out = StepOut::default();
    let pressed = |bt: u16| cmd.held(bt) && prev_buttons & bt == 0;
    let [ax_f, ax_r, ax_u] = cmd.axes();
    let [dyaw, dpitch, droll] = cmd.look_delta();
    let up = geom::up_at(b.pos);
    b.up = up;

    // ---- mode changes ----
    if pressed(buttons::JETPACK) {
        if b.mode == MoveMode::Jetpack {
            view_from_orient(b);
            b.mode = MoveMode::Float;
        } else if suit.propellant_kg > 0.0 {
            b.orient = orient_from_view(b);
            b.mode = MoveMode::Jetpack;
            b.grounded = false;
        }
    }
    let want_crouch = cmd.held(buttons::CROUCH);
    if want_crouch != b.crouched && (want_crouch || head_room(b, cfg, ctx.arena, scratch)) {
        b.crouched = want_crouch;
        b.height = if want_crouch { cfg.crouch_height } else { cfg.height };
    }

    // ---- look ----
    if b.mode == MoveMode::Jetpack {
        let lim = cfg.jetpack.max_turn_rate * dt;
        let q = det::quat_axis_angle(DVec3::Y, dyaw.clamp(-lim, lim))
            * det::quat_axis_angle(DVec3::X, -dpitch.clamp(-lim, lim))
            * det::quat_axis_angle(DVec3::Z, droll.clamp(-lim, lim));
        b.orient = (b.orient * q).normalize();
    } else {
        b.yaw = (b.yaw + dyaw).rem_euclid(std::f64::consts::TAU);
        b.pitch = (b.pitch + dpitch).clamp(-1.5, 1.5);
    }

    // ---- forces ----
    let centre = b.pos + up * (0.5 * b.height);
    let rho = ctx.atmo.density(ctx.frame, centre);
    let wind = if rho > 0.0 { ctx.arena.breach_wind(centre, ctx.atmo.frac, ctx.atmo.speed_of_sound()) } else { DVec3::ZERO };
    let drag = |v: DVec3, m: f64| -> DVec3 {
        let va = v - wind;
        -va * (0.5 * rho * BODY_CDA * va.length() / m)
    };
    let a_frame = ctx.frame.accel(b.pos + up * (0.5 * b.height), b.vel) + ctx.fields.accel(centre);
    let felt_down = -a_frame.dot(up); // > 0: pressed to the floor

    let mass = b.mass + suit.propellant_kg;
    match b.mode {
        MoveMode::Jetpack => {
            let (r, u, f) = (b.orient * DVec3::X, b.orient * DVec3::Y, b.orient * DVec3::Z);
            let mut dir = f * ax_f - r * ax_r + u * ax_u;
            if dir.length_squared() > 1.0 {
                dir = dir.normalize();
            }
            let thrust = if suit.propellant_kg > 0.0 { cfg.jetpack.thrust_n * dir.length() } else { 0.0 };
            let mdot = thrust / (cfg.jetpack.isp_s * G0);
            suit.propellant_kg = (suit.propellant_kg - mdot * dt).max(0.0);
            let a = a_frame + dir.normalize_or_zero() * (thrust / mass) + drag(b.vel, mass);
            b.vel += a * dt;
            if suit.propellant_kg <= 0.0 {
                view_from_orient(b);
                b.mode = MoveMode::Float;
            }
        }
        MoveMode::Walk | MoveMode::Float => {
            let (e, _, n) = geom::basis_at(b.pos);
            let (sy, cy) = det::sin_cos(b.yaw);
            let flat_f = e * sy + n * cy;
            let flat_r = flat_f.cross(up);
            let mut wish = flat_f * ax_f + flat_r * ax_r;
            if wish.length_squared() > 1.0 {
                wish = wish.normalize();
            }
            // wading: water resists the legs (~speed falls with depth)
            let depth = ctx.arena.water_depth(arena_u(ctx.arena, b.pos), b.pos.z);
            let wade = if depth > 0.2 { (1.0 - depth / 1.6).clamp(0.25, 1.0) } else { 1.0 };
            let wish = wish * wish_speed(b, cfg, cmd) * wade;
            if b.grounded && felt_down > cfg.walk_min_g {
                b.mode = MoveMode::Walk;
                let vn = b.vel.dot(up);
                let vt = b.vel - up * vn;
                let dv = wish - vt;
                // traction limited by friction μ·N
                let max_dv = cfg.friction * felt_down * dt;
                let dvl = dv.length();
                let dv = if dvl > max_dv { dv * (max_dv / dvl) } else { dv };
                b.vel += dv;
                if pressed(buttons::JUMP) {
                    // vault / mantle: arms push the body over a waist-high obstacle in
                    // front (a 0.37 m jump alone would never clear a 0.9 m wall)
                    let feet = ctx.arena.surf(b.pos).h;
                    let probe = ctx.arena.surf(b.pos + flat_f * (cfg.radius + 0.6));
                    let lift = ctx
                        .arena
                        .ground(probe.u, probe.v, feet + cfg.mantle_height, 0.0, scratch)
                        .map(|g| g.h - feet)
                        .filter(|&dh| dh > cfg.step_height && felt_down > 0.0);
                    let v_up = match lift {
                        Some(dh) => (2.0 * felt_down * (dh + 0.15)).sqrt().max(cfg.jump_speed),
                        None => cfg.jump_speed,
                    };
                    b.vel += up * v_up;
                    if lift.is_some() {
                        b.vel += flat_f * 1.0;
                    }
                    b.grounded = false;
                }
                // stride → footstep sounds
                b.stride += vt.length() * dt;
                let step_len = if b.crouched { 0.6 } else { 0.8 + 0.12 * vt.length() };
                if b.stride > step_len {
                    b.stride = 0.0;
                    out.footstep = true;
                }
            } else {
                b.mode = MoveMode::Float;
                b.vel += wish.normalize_or_zero() * (cfg.air_control * dt);
                if b.grounded && pressed(buttons::JUMP) {
                    // push off in low gravity
                    b.vel += up * cfg.jump_speed;
                    b.grounded = false;
                }
            }
            b.vel += (a_frame + drag(b.vel, mass)) * dt;
        }
    }

    // ---- move with field crossings ----
    let old = b.pos;
    let c0 = old + up * (0.5 * b.height);
    b.pos += b.vel * dt;
    if !ctx.fields.is_empty() {
        let c1 = b.pos + up * (0.5 * b.height);
        if let Some((_, x)) = ctx.fields.crossing(c0, c1) {
            match x {
                Crossing::Shell { normal, barrier } => {
                    let (v, refl) = shell_response(b.vel, normal, barrier);
                    if refl {
                        b.vel = v;
                        b.pos = old;
                        out.bounced = true;
                    }
                }
                Crossing::Fold { shift, stability, .. } => {
                    b.pos += shift;
                    out.crossed_fold = Some(stability);
                }
            }
        }
    }
    let was_inside = geom::world_to_surf(old).h > -WINDOW_THICKNESS;
    collide(b, cfg, ctx, &mut out, scratch);
    let inside = geom::world_to_surf(b.pos).h > -WINDOW_THICKNESS;
    if was_inside && !inside {
        out.left_hull = true;
    }
    out
}

fn head_room(b: &Body, cfg: &PlayerCfg, arena: &Arena, scratch: &mut Vec<u32>) -> bool {
    let s = arena.surf(b.pos);
    arena.boxes_near(s.u, s.v, b.radius + 0.5, scratch);
    let top = s.h + cfg.height;
    !scratch.iter().any(|&i| {
        let x = &arena.st.boxes[i as usize];
        (s.u - x.u).abs() < x.hu + b.radius && (s.v - x.v).abs() < x.hv + b.radius && x.h0 > s.h + 0.1 && x.h0 < top
    })
}

fn collide(b: &mut Body, cfg: &PlayerCfg, ctx: &Ctx, out: &mut StepOut, scratch: &mut Vec<u32>) {
    let arena = ctx.arena;
    let s = arena.surf(b.pos);
    let (east, up, north) = geom::basis_at(b.pos);
    // spindle and end caps
    let rp = geom::r_perp(b.pos);
    if rp < geom::SPINDLE_R + b.radius && rp > 1e-6 {
        let n = -up;
        b.pos += n * (geom::SPINDLE_R + b.radius - rp);
        let vn = b.vel.dot(n);
        if vn < 0.0 {
            out.impact = out.impact.max(-vn);
            b.vel -= n * vn;
        }
    }
    for (lim, nz) in [(b.radius, 1.0), (geom::L - b.radius, -1.0)] {
        if (b.pos.z - lim) * nz < 0.0 && rp < geom::R {
            b.pos.z = lim;
            let vn = b.vel.z * nz;
            if vn < 0.0 {
                out.impact = out.impact.max(-vn);
                b.vel.z = 0.0;
            }
        }
    }
    // outside the hull: the outer skin is a wall unless there is a hole
    // under land the hull lies below the soil (terrain may dip below h = 0)
    let skin = if crate::map::is_land(s.u) { arena.terrain(s.u, s.v).min(0.0) - 2.0 } else { -WINDOW_THICKNESS };
    if s.h < skin {
        if s.h > skin - 0.6 && arena.floor(s.u, s.v).is_some() {
            // pressed against the outer skin (magnetic boots / climbing back)
            let n = -up;
            b.pos += n * (s.h - skin + 0.6);
            let vn = b.vel.dot(n);
            if vn < 0.0 {
                b.vel -= n * vn;
            }
        }
        b.grounded = false;
        b.ground = None;
        return;
    }
    // walls of boxes: push out horizontally where the box overlaps the body
    let feet = s.h;
    let head = s.h + b.height;
    arena.boxes_near(s.u, s.v, b.radius + 1.0, scratch);
    let mut push = DVec3::ZERO;
    for &i in scratch.iter() {
        let x = &arena.st.boxes[i as usize];
        if feet > x.h1 - cfg.step_height || head < x.h0 {
            continue;
        }
        let du = s.u - x.u;
        let dv = s.v - x.v;
        let ou = x.hu + b.radius - du.abs();
        let ov = x.hv + b.radius - dv.abs();
        if ou > 0.0 && ov > 0.0 {
            if x.h0 > feet + 0.3 && head > x.h0 && feet < x.h0 {
                // bumped the head on an overhang
                let d = head - x.h0;
                push -= up * d;
                continue;
            }
            if ou < ov {
                push += east * (ou * du.signum());
            } else {
                push += north * (ov * dv.signum());
            }
        }
    }
    if push != DVec3::ZERO {
        b.pos += push;
        let pn = push.normalize();
        let vn = b.vel.dot(pn);
        if vn < 0.0 {
            out.impact = out.impact.max(-vn);
            b.vel -= pn * vn;
        }
    }
    // ground
    let s = arena.surf(b.pos);
    let ground = arena.ground(s.u, s.v, s.h + cfg.step_height, b.radius, scratch);
    match ground {
        Some(g)
            if s.h < g.h
                || (s.h <= g.h + 0.02 && b.vel.dot(up) <= 0.5)
                || (b.grounded && b.vel.dot(up) < 0.5 && s.h < g.h + cfg.step_height && b.mode != MoveMode::Jetpack) =>
        {
            let lift = g.h - s.h;
            b.pos += up * lift;
            let vn = b.vel.dot(g.normal);
            if vn < 0.0 {
                if !b.grounded {
                    out.impact = out.impact.max(-vn);
                }
                b.vel -= g.normal * vn;
            }
            // too steep: slide
            let slope = det::acos(g.normal.dot(up));
            b.grounded = slope < cfg.max_slope_deg.to_radians();
            b.ground = Some(g.material);
            b.ground_pane = g.pane;
            if b.mode == MoveMode::Jetpack && b.vel.dot(up) <= 0.1 {
                // touch-down ends powered flight
                view_from_orient(b);
                b.mode = MoveMode::Walk;
            }
        }
        _ => {
            b.grounded = false;
            b.ground = None;
            b.ground_pane = None;
        }
    }
}

/// Felt gravity in g at the feet (for the HUD).
pub fn felt_g(b: &Body, frame: &StationFrame) -> f64 {
    frame.accel(b.pos, b.vel).length() / G0
}

/// Look deltas (Δyaw, Δpitch) that turn the view of `b` towards `dir`
/// (works in both walking and 6-DOF modes; used by bots and tests).
pub fn look_delta_to(b: &Body, dir: DVec3) -> (f64, f64) {
    let d = dir.normalize_or(DVec3::Z);
    match b.mode {
        MoveMode::Jetpack => {
            let db = b.orient.inverse() * d;
            (det::atan2(db.x, db.z), det::atan2(db.y, (db.x * db.x + db.z * db.z).sqrt()))
        }
        _ => {
            let (e, up, n) = geom::basis_at(b.pos);
            let yaw = det::atan2(d.dot(e), d.dot(n));
            let pitch = det::asin(d.dot(up));
            let mut dy = (yaw - b.yaw).rem_euclid(std::f64::consts::TAU);
            if dy > std::f64::consts::PI {
                dy -= std::f64::consts::TAU;
            }
            (dy, pitch - b.pitch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GameConfig;
    use crate::input::Command;

    fn setup() -> (GameConfig, StationFrame, Atmosphere, Arena, FieldSet) {
        (GameConfig::builtin(), StationFrame::colony(), Atmosphere::default(), Arena::prototype(3), FieldSet::default())
    }

    fn suit() -> Suit {
        Suit { o2_s: 900.0, integrity: 1.0, sealed: false, propellant_kg: 12.0, p_prev: 101_325.0 }
    }

    fn open_spot(a: &Arena) -> (f64, f64) {
        // a street intersection is free of boxes
        a.st.spawns[0]
    }

    #[test]
    fn stands_still_on_the_floor_and_jumps_to_the_right_height() {
        let (cfg, frame, atmo, arena, fs) = setup();
        let ctx = Ctx { cfg: &cfg.player, frame: &frame, atmo: &atmo, arena: &arena, fields: &fs };
        let (u, v) = open_spot(&arena);
        let mut b = new_body(&cfg.player, geom::surf_to_world(u, v, 0.5), 0.0);
        let mut s = suit();
        let mut sc = Vec::new();
        let dt = 1.0 / 240.0;
        let idle = Command::default();
        for _ in 0..480 {
            step(&mut b, &mut s, &idle, 0, &ctx, dt, &mut sc);
        }
        assert!(b.grounded);
        let h0 = arena.surf(b.pos).h;
        let p0 = b.pos;
        for _ in 0..240 {
            step(&mut b, &mut s, &idle, 0, &ctx, dt, &mut sc);
        }
        assert!((b.pos - p0).length() < 1e-6, "drift {}", (b.pos - p0).length());
        // jump: apex = v²/2g (Coriolis on a vertical jump only shifts it sideways by mm)
        let jump = Command { buttons: buttons::JUMP, ..Default::default() };
        step(&mut b, &mut s, &jump, 0, &ctx, dt, &mut sc);
        let mut apex: f64 = 0.0;
        for _ in 0..400 {
            step(&mut b, &mut s, &jump, buttons::JUMP, &ctx, dt, &mut sc);
            apex = apex.max(arena.surf(b.pos).h - h0);
        }
        let expect = cfg.player.jump_speed * cfg.player.jump_speed / (2.0 * geom::G0);
        assert!((apex - expect).abs() / expect < 0.04, "apex {apex} vs {expect}");
    }

    #[test]
    fn running_spinward_is_heavier_than_antispinward() {
        let (_, frame, _, _, _) = setup();
        let p = geom::surf_to_world(10_000.0, 16_000.0, 0.0);
        let (e, up, _) = geom::basis_at(p);
        let heavy = -frame.accel(p, e * 7.5).dot(up);
        let light = -frame.accel(p, -e * 7.5).dot(up);
        assert!((heavy - light - 4.0 * frame.omega() * 7.5).abs() < 1e-9);
        assert!(heavy > light);
    }

    #[test]
    fn jetpack_rocket_equation() {
        // In a weightless frame (no spin) full thrust for 10 s gives Δv = Isp g₀ ln(m0/m1).
        let (cfg, _, _, arena, fs) = setup();
        let mut frame = StationFrame::colony();
        frame.spin = DVec3::ZERO;
        let atmo = Atmosphere { frac: 0.0, ..Default::default() };
        let ctx = Ctx { cfg: &cfg.player, frame: &frame, atmo: &atmo, arena: &arena, fields: &fs };
        let mut b = new_body(&cfg.player, geom::surf_to_world(arena.st.u0 + 500.0, arena.st.v0 + 500.0, 300.0), 0.0);
        let mut s = suit();
        let mut sc = Vec::new();
        let mut c = Command { buttons: buttons::JETPACK, ..Default::default() };
        step(&mut b, &mut s, &c, 0, &ctx, 1.0 / 60.0, &mut sc);
        assert_eq!(b.mode, MoveMode::Jetpack);
        b.vel = DVec3::ZERO;
        let m0 = b.mass + s.propellant_kg;
        c.buttons = 0;
        c.set_move(1.0, 0.0, 0.0);
        let dt = 1.0 / 600.0;
        for _ in 0..6000 {
            step(&mut b, &mut s, &c, 0, &ctx, dt, &mut sc);
        }
        let m1 = b.mass + s.propellant_kg;
        let dv = cfg.player.jetpack.isp_s * G0 * (m0 / m1).ln();
        assert!((b.vel.length() - dv).abs() / dv < 1e-3, "{} vs {}", b.vel.length(), dv);
    }
}
