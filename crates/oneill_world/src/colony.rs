//! O'Neill cylinder geometry and the rotating reference frame.
//!
//! Axis = +Z (0..L). Floor at radius R, "up" points to the axis.
//! Surface coordinates: u = θ·R (wraps at 2πR), v = z, h = height above floor.
//! Valleys are centred at θ = 0°, 120°, 240°; windows in between.

use std::f32::consts::{PI, TAU};

use glam::{Vec3, Vec4};

pub const R: f32 = 4000.0; // canonical Island Three: 8 km diameter
pub const L: f32 = 32000.0; // 32 km long
pub const CIRC: f32 = TAU * R;
pub const STRIP_ANGLE: f32 = TAU / 3.0;
pub const HALF_LAND: f32 = PI / 6.0;
pub const G0: f32 = 9.81;

pub const STRIP_NAMES: [&str; 3] = ["Пастораль", "Мегаполис", "Дикие земли"];

pub fn half_land_width() -> f32 {
    HALF_LAND * R
}

/// Angular velocity giving `g_floor` (in g) of artificial gravity at the floor.
pub fn omega_for(g_floor: f32) -> f32 {
    (g_floor.max(0.0) * G0 / R).sqrt()
}

pub fn surf_to_world(u: f32, v: f32, h: f32) -> Vec3 {
    let th = u / R;
    let r = R - h;
    Vec3::new(r * th.cos(), r * th.sin(), v)
}

pub fn angle_of(p: Vec3) -> f32 {
    let mut th = p.y.atan2(p.x);
    if th < 0.0 {
        th += TAU;
    }
    th
}

/// (u, v, h) with u in [0, CIRC).
pub fn world_to_surf(p: Vec3) -> Vec3 {
    Vec3::new(angle_of(p) * R, p.z, R - p.truncate().length())
}

pub fn east_of(th: f32) -> Vec3 {
    Vec3::new(-th.sin(), th.cos(), 0.0)
}

pub fn up_of(th: f32) -> Vec3 {
    Vec3::new(-th.cos(), -th.sin(), 0.0)
}

/// (east, up, north) basis at a point. Returns None too close to the axis.
pub fn basis_at(p: Vec3) -> Option<(Vec3, Vec3, Vec3)> {
    if p.truncate().length() < 1.0 {
        return None;
    }
    let th = angle_of(p);
    Some((east_of(th), up_of(th), Vec3::Z))
}

/// Nearest valley index and across-valley offset s (m).
pub fn strip_of(th: f32) -> (usize, f32) {
    let mut k = (th / STRIP_ANGLE).round();
    let d = th - k * STRIP_ANGLE;
    if k > 2.5 {
        k = 0.0;
    }
    (k as usize, d * R)
}

pub fn is_land(th: f32) -> bool {
    strip_of(th).1.abs() < half_land_width()
}

/// Bring `u` within ±CIRC/2 of `reference` (for local computations).
pub fn unwrap_u(u: f32, reference: f32) -> f32 {
    let mut d = (u - reference) % CIRC;
    if d > CIRC * 0.5 {
        d -= CIRC;
    }
    if d < -CIRC * 0.5 {
        d += CIRC;
    }
    reference + d
}

/// Light arriving at each valley through the opposite window, after bouncing
/// off the external mirror. `beta` tilts the beam along the axis (the mirror
/// angle sweeps during the day), `strength` is 0 when the mirrors are closed.
pub fn light_dirs(beta: f32, strength: f32) -> [Vec4; 3] {
    let mut out = [Vec4::ZERO; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let up = up_of(k as f32 * STRIP_ANGLE);
        let d = (up * beta.cos() + Vec3::Z * beta.sin()).normalize();
        *o = d.extend(strength);
    }
    out
}

/// Acceleration felt in the rotating frame (no real gravity in the colony):
/// centrifugal ω²·r_perp plus Coriolis −2 ω×v.
pub fn frame_accel(pos: Vec3, vel: Vec3, omega: f32, coriolis: bool) -> Vec3 {
    let cf = Vec3::new(pos.x, pos.y, 0.0) * (omega * omega);
    if coriolis {
        cf + Vec3::new(2.0 * omega * vel.y, -2.0 * omega * vel.x, 0.0)
    } else {
        cf
    }
}
