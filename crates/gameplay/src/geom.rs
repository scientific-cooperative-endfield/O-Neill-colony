//! Cylinder geometry in double precision (the f32 helpers live in
//! `oneill_world::colony`; gameplay needs f64 for bit-reproducible ballistics
//! at 4 km from the axis).
//!
//! Colony frame: axis = +Z (0 … L), floor at radius R, "up" points to the axis.
//! Surface ("flat") coordinates: u = θ·R (arc length round the hull, wraps at
//! 2πR), v = z (along the axis), h = height above the floor = R − r⊥.

use crate::det;
use glam::DVec3;
use std::f64::consts::TAU;

pub const R: f64 = oneill_world::colony::R as f64;
pub const L: f64 = oneill_world::colony::L as f64;
pub const CIRC: f64 = TAU * R;
pub const SPINDLE_R: f64 = oneill_world::physics::SPINDLE_R as f64;
/// Standard gravity used for the design spin (the floor feels exactly 1 g).
pub const G0: f64 = 9.806_65;
pub const C_LIGHT: f64 = relativity::C;
pub const G_NEWTON: f64 = relativity::G;

/// Surface point (u, v, h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surf {
    pub u: f64,
    pub v: f64,
    pub h: f64,
}

pub fn surf_to_world(u: f64, v: f64, h: f64) -> DVec3 {
    let (s, c) = det::sin_cos(u / R);
    let r = R - h;
    DVec3::new(r * c, r * s, v)
}

/// u in [0, CIRC).
pub fn world_to_surf(p: DVec3) -> Surf {
    let mut th = det::atan2(p.y, p.x);
    if th < 0.0 {
        th += TAU;
    }
    Surf { u: th * R, v: p.z, h: R - (p.x * p.x + p.y * p.y).sqrt() }
}

/// Bring `u` within ±CIRC/2 of `reference`.
pub fn unwrap_u(u: f64, reference: f64) -> f64 {
    let mut d = (u - reference).rem_euclid(CIRC);
    if d > 0.5 * CIRC {
        d -= CIRC;
    }
    reference + d
}

pub fn r_perp(p: DVec3) -> f64 {
    (p.x * p.x + p.y * p.y).sqrt()
}

/// Local (east, up, north) basis. Near the axis an arbitrary but continuous
/// fallback is used.
pub fn basis_at(p: DVec3) -> (DVec3, DVec3, DVec3) {
    let r = r_perp(p);
    if r < 1e-6 {
        return (DVec3::Y, DVec3::NEG_X, DVec3::Z);
    }
    let (c, s) = (p.x / r, p.y / r);
    (DVec3::new(-s, c, 0.0), DVec3::new(-c, -s, 0.0), DVec3::Z)
}

pub fn up_at(p: DVec3) -> DVec3 {
    basis_at(p).1
}

/// Is the point inside the pressurised volume of the cylinder?
pub fn inside_hull(p: DVec3) -> bool {
    r_perp(p) <= R + 1e-6 && p.z > 0.0 && p.z < L
}
