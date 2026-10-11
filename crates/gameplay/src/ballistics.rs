//! Exterior ballistics in the rotating station frame.
//!
//! Bullets are projectiles, never hitscan. Each one is integrated with RK4 in
//! sub-steps of ≤ 4 m:
//!
//!   a = −Ω×(Ω×d) − 2Ω×v + Δa_PW(d) + a_fields(p) − (ρ(p) π / 8 BC) · C_D,G7(M) · |v_air| v_air
//!
//! * centrifugal, Coriolis and the black hole's differential Paczyński–Wiita
//!   gravity come from `StationFrame`;
//! * the micro-hole gravity comes from `FieldSet`;
//! * drag uses the published G7 ballistic coefficient of the round and the
//!   standard G7 drag curve; the air density follows the hydrostatic profile
//!   of the rotating atmosphere and drops to zero outside the hull. The wind
//!   is the outflow towards breached windows (the air otherwise co-rotates).

use crate::fields::FieldSet;
use crate::frame::{Atmosphere, StationFrame};
use crate::map::Arena;
use glam::DVec3;

/// lb/in² → kg/m²
pub const LB_IN2: f64 = 703.069_6;

/// Standard G7 drag function (Mach → C_D), the BRL / JBM table (84 points).
#[allow(clippy::approx_constant)] // tabulated G7 data, not log10(2)
const G7: [(f64, f64); 84] = [
    (0.0, 0.1198),
    (0.05, 0.1197),
    (0.1, 0.1196),
    (0.15, 0.1194),
    (0.2, 0.1193),
    (0.25, 0.1194),
    (0.3, 0.1194),
    (0.35, 0.1194),
    (0.4, 0.1193),
    (0.45, 0.1193),
    (0.5, 0.1194),
    (0.55, 0.1193),
    (0.6, 0.1194),
    (0.65, 0.1197),
    (0.7, 0.1202),
    (0.725, 0.1207),
    (0.75, 0.1215),
    (0.775, 0.1226),
    (0.8, 0.1242),
    (0.825, 0.1266),
    (0.85, 0.1306),
    (0.875, 0.1368),
    (0.9, 0.1464),
    (0.925, 0.1660),
    (0.95, 0.2054),
    (0.975, 0.2993),
    (1.0, 0.3803),
    (1.025, 0.4015),
    (1.05, 0.4043),
    (1.075, 0.4034),
    (1.1, 0.4014),
    (1.125, 0.3987),
    (1.15, 0.3955),
    (1.2, 0.3884),
    (1.25, 0.3810),
    (1.3, 0.3732),
    (1.35, 0.3657),
    (1.4, 0.3580),
    (1.5, 0.3440),
    (1.55, 0.3376),
    (1.6, 0.3315),
    (1.65, 0.3260),
    (1.7, 0.3209),
    (1.75, 0.3160),
    (1.8, 0.3117),
    (1.85, 0.3078),
    (1.9, 0.3042),
    (1.95, 0.3010),
    (2.0, 0.2980),
    (2.05, 0.2951),
    (2.1, 0.2922),
    (2.15, 0.2892),
    (2.2, 0.2864),
    (2.25, 0.2835),
    (2.3, 0.2807),
    (2.35, 0.2779),
    (2.4, 0.2752),
    (2.45, 0.2725),
    (2.5, 0.2697),
    (2.55, 0.2670),
    (2.6, 0.2643),
    (2.65, 0.2615),
    (2.7, 0.2588),
    (2.75, 0.2561),
    (2.8, 0.2533),
    (2.85, 0.2506),
    (2.9, 0.2479),
    (2.95, 0.2451),
    (3.0, 0.2424),
    (3.1, 0.2368),
    (3.2, 0.2313),
    (3.3, 0.2258),
    (3.4, 0.2205),
    (3.5, 0.2154),
    (3.6, 0.2106),
    (3.7, 0.2060),
    (3.8, 0.2017),
    (3.9, 0.1975),
    (4.0, 0.1935),
    (4.2, 0.1861),
    (4.4, 0.1793),
    (4.6, 0.1730),
    (4.8, 0.1672),
    (5.0, 0.1618),
];

pub fn cd_g7(mach: f64) -> f64 {
    if mach >= G7[G7.len() - 1].0 {
        return G7[G7.len() - 1].1;
    }
    let i = G7.partition_point(|e| e.0 <= mach).max(1);
    let (m0, c0) = G7[i - 1];
    let (m1, c1) = G7[i];
    c0 + (c1 - c0) * (mach - m0) / (m1 - m0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Round {
    pub mass: f64,
    /// G7 ballistic coefficient in kg/m²
    pub bc: f64,
    pub caliber: f64,
}

impl Round {
    pub fn from_cfg(w: &crate::config::WeaponCfg) -> Round {
        Round { mass: w.bullet_mass_g * 1e-3, bc: w.bc_g7 * LB_IN2, caliber: w.caliber_mm * 1e-3 }
    }
}

/// Everything the integrator needs to know about the world.
#[derive(Clone, Copy)]
pub struct Env<'a> {
    pub frame: &'a StationFrame,
    pub atmo: &'a Atmosphere,
    pub arena: Option<&'a Arena>,
    pub fields: &'a FieldSet,
    /// how much of the Coriolis term to include (1 = physics; AI models use < 1)
    pub coriolis: f64,
    /// include the centrifugal / tidal "gravity" (AI models may switch it off)
    pub gravity: f64,
    pub drag: bool,
}

impl<'a> Env<'a> {
    pub fn new(frame: &'a StationFrame, atmo: &'a Atmosphere, fields: &'a FieldSet) -> Env<'a> {
        Env { frame, atmo, arena: None, fields, coriolis: if frame.coriolis { 1.0 } else { 0.0 }, gravity: 1.0, drag: true }
    }

    pub fn with_arena(mut self, a: &'a Arena) -> Self {
        self.arena = Some(a);
        self
    }

    /// Acceleration of a round at (p, v).
    #[inline]
    pub fn accel(&self, r: &Round, p: DVec3, v: DVec3) -> DVec3 {
        let mut a = self.frame.accel_static(p) * self.gravity - 2.0 * self.coriolis * self.frame.spin.cross(v);
        if !self.fields.is_empty() {
            a += self.fields.accel(p);
        }
        if self.drag {
            let rho = self.atmo.density(self.frame, p);
            if rho > 0.0 {
                let wind = match self.arena {
                    Some(ar) => ar.breach_wind(p, self.atmo.frac, self.atmo.speed_of_sound()),
                    None => DVec3::ZERO,
                };
                let va = v - wind;
                let s = va.length();
                let mach = s / self.atmo.speed_of_sound();
                a -= va * (rho * std::f64::consts::PI / (8.0 * r.bc) * cd_g7(mach) * s);
            }
        }
        a
    }

    /// One classical RK4 step.
    #[inline]
    pub fn rk4(&self, r: &Round, p: DVec3, v: DVec3, h: f64) -> (DVec3, DVec3) {
        let a1 = self.accel(r, p, v);
        let (p2, v2) = (p + v * (0.5 * h), v + a1 * (0.5 * h));
        let a2 = self.accel(r, p2, v2);
        let (p3, v3) = (p + v2 * (0.5 * h), v + a2 * (0.5 * h));
        let a3 = self.accel(r, p3, v3);
        let (p4, v4) = (p + v3 * h, v + a3 * h);
        let a4 = self.accel(r, p4, v4);
        (p + (v + 2.0 * v2 + 2.0 * v3 + v4) * (h / 6.0), v + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (h / 6.0))
    }
}

/// Number of sub-steps for one tick so that a sub-step is ≤ 4 m.
#[inline]
pub fn substeps(speed: f64, dt: f64) -> usize {
    ((speed * dt / 4.0).ceil() as usize).clamp(1, 32)
}

/// First-order analytic estimate of the vertical split between a shot fired
/// spinward (east) and one fired antispinward over horizontal distance `d` at
/// speed `v` in vacuum: the Coriolis term −2Ω×v adds ±2ωv to the felt gravity,
/// so the drops differ by ½·(4ωv)·(d/v)² = 2ωd²/v.
pub fn coriolis_split_estimate(omega: f64, d: f64, v: f64) -> f64 {
    2.0 * omega * d * d / v
}

/// Lateral (axial) deflection does not exist for east/west shots; for a shot
/// along the axis the Coriolis force vanishes (Ω ∥ v). For a shot across the
/// floor towards the axis ("up") the deflection is ω v t² sideways.
pub fn coriolis_lateral_estimate_vertical_shot(omega: f64, v: f64, t: f64) -> f64 {
    omega * v * t * t
}

/// Sampled trajectory (for the neuro-HUD and AI).
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    pub points: Vec<DVec3>,
    pub hit: Option<crate::map::RayHit>,
    pub time_of_flight: f64,
    pub impact_speed: f64,
}

/// Integrates a free round until it hits static geometry (if an arena is
/// given) or `max_t` elapses. Records a point every `every` sub-steps.
pub fn predict(env: &Env, r: &Round, p0: DVec3, v0: DVec3, max_t: f64, dt: f64, scratch: &mut Vec<u32>) -> Trajectory {
    let mut tr = Trajectory { points: vec![p0], ..Default::default() };
    let (mut p, mut v) = (p0, v0);
    let mut t = 0.0;
    while t < max_t {
        let n = substeps(v.length(), dt);
        let h = dt / n as f64;
        for _ in 0..n {
            let (p1, v1) = env.rk4(r, p, v, h);
            if let Some(a) = env.arena {
                if let Some(hit) = a.raycast(p, p1, scratch) {
                    tr.points.push(hit.point);
                    tr.time_of_flight = t + h * hit.t;
                    tr.impact_speed = (v + (v1 - v) * hit.t).length();
                    tr.hit = Some(hit);
                    return tr;
                }
            }
            p = p1;
            v = v1;
            t += h;
        }
        tr.points.push(p);
    }
    tr.time_of_flight = t;
    tr.impact_speed = v.length();
    tr
}

/// Closest approach of a free round to a moving target point:
/// returns (miss vector target → round at closest approach, time).
pub fn closest_approach(env: &Env, r: &Round, p0: DVec3, v0: DVec3, target: DVec3, target_v: DVec3, max_t: f64, dt: f64) -> (DVec3, f64) {
    let (mut p, mut v) = (p0, v0);
    let mut t = 0.0;
    let mut best = (p - target, 0.0);
    let mut best_d = f64::MAX;
    while t < max_t {
        let n = substeps(v.length(), dt);
        let h = dt / n as f64;
        for _ in 0..n {
            let (p1, v1) = env.rk4(r, p, v, h);
            // exact closest point on the segment relative to the moving target
            let q0 = p - (target + target_v * t);
            let q1 = p1 - (target + target_v * (t + h));
            let dq = q1 - q0;
            let s = if dq.length_squared() > 1e-18 { (-q0.dot(dq) / dq.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
            let q = q0 + dq * s;
            let d = q.length_squared();
            if d < best_d {
                best_d = d;
                best = (q, t + h * s);
            } else if s < 1.0 && d > best_d * 1.0001 {
                return best;
            }
            p = p1;
            v = v1;
            t += h;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom;

    fn env_vacuum<'a>(f: &'a StationFrame, a: &'a Atmosphere, fs: &'a FieldSet) -> Env<'a> {
        let mut e = Env::new(f, a, fs);
        e.drag = false;
        e
    }

    /// Fires a round horizontally along ±east from 1.7 m above the floor and
    /// returns its height above the floor after `d` metres of arc.
    fn height_after(env: &Env, r: &Round, v0: f64, east: bool, d: f64) -> f64 {
        let u0 = 10_000.0;
        let p0 = geom::surf_to_world(u0, 16_000.0, 1.7);
        let (e, _, _) = geom::basis_at(p0);
        let mut p = p0;
        let mut v = e * if east { v0 } else { -v0 };
        let dt = 1.0 / 60.0;
        let arc = |p: DVec3| (geom::unwrap_u(geom::world_to_surf(p).u, u0) - u0).abs();
        loop {
            let n = substeps(v.length(), dt);
            let h = dt / n as f64;
            for _ in 0..n {
                let (p1, v1) = env.rk4(r, p, v, h);
                if arc(p1) >= d {
                    // interpolate to the exact arc distance
                    let (a0, a1) = (arc(p), arc(p1));
                    let s = (d - a0) / (a1 - a0);
                    let q = p + (p1 - p) * s;
                    return geom::world_to_surf(q).h;
                }
                p = p1;
                v = v1;
            }
        }
    }

    #[test]
    fn coriolis_split_east_vs_west_matches_analytic_500m() {
        let f = StationFrame::colony();
        let atmo = Atmosphere::default();
        let fs = FieldSet::default();
        let env = env_vacuum(&f, &atmo, &fs);
        let r = Round { mass: 0.01944, bc: 0.381 * LB_IN2, caliber: 0.0086 };
        for v0 in [820.0, 940.0, 360.0] {
            let he = height_after(&env, &r, v0, true, 500.0);
            let hw = height_after(&env, &r, v0, false, 500.0);
            let split = hw - he;
            let est = coriolis_split_estimate(f.omega(), 500.0, v0);
            let rel = (split - est) / est;
            assert!(rel.abs() < 0.05, "v0 = {v0}: numeric split {split:.3} m vs analytic {est:.3} m ({:.2} %)", rel * 100.0);
        }
    }

    #[test]
    fn rk4_matches_exact_inertial_straight_line() {
        // In vacuum with no hole a round moves on a straight line in the inertial
        // frame; transform that exact solution into the rotating frame.
        let f = StationFrame::colony();
        let atmo = Atmosphere::default();
        let fs = FieldSet::default();
        let env = env_vacuum(&f, &atmo, &fs);
        let r = Round { mass: 0.004, bc: 0.151 * LB_IN2, caliber: 0.0057 };
        let p0 = geom::surf_to_world(3000.0, 16_000.0, 1.7);
        let (e, up, n) = geom::basis_at(p0);
        let v0 = (e * 0.6 + up * 0.1 + n * 0.79).normalize() * 940.0;
        let w = f.spin;
        let d0 = p0 - f.com;
        let vin = v0 + w.cross(d0);
        let dt = 1.0 / 60.0;
        let (mut p, mut v) = (p0, v0);
        let mut t = 0.0;
        for _ in 0..60 {
            let k = substeps(v.length(), dt);
            for _ in 0..k {
                let s = env.rk4(&r, p, v, dt / k as f64);
                p = s.0;
                v = s.1;
            }
            t += dt;
        }
        let xin = d0 + vin * t; // inertial position relative to the COM
        let q = crate::det::quat_axis_angle(DVec3::Z, -f.omega() * t);
        let expect = f.com + q * xin;
        assert!((p - expect).length() < 1e-4, "error {} m", (p - expect).length());
    }

    #[test]
    fn subsonic_drag_gives_exact_exponential_decay() {
        // At M ≈ 0.3–0.6 the G7 C_D is constant (0.1194 ± 0.0001), so along the
        // axis (no Coriolis, centrifugal ⟂ v) v(x) = v₀·exp(−k x), k = ρ C_D π / (8 BC).
        let f = StationFrame::colony();
        let atmo = Atmosphere::default();
        let fs = FieldSet::default();
        let env = Env::new(&f, &atmo, &fs);
        let r = Round { mass: 0.008, bc: 0.072 * LB_IN2, caliber: 0.009 };
        let p0 = geom::surf_to_world(3000.0, 16_000.0, 40.0); // high enough not to fall through the floor
        let (_, _, n) = geom::basis_at(p0);
        let (mut p, mut v) = (p0, n * 200.0);
        while (p.z - p0.z) < 150.0 {
            let s = env.rk4(&r, p, v, 1e-3);
            p = s.0;
            v = s.1;
        }
        let x = p.z - p0.z;
        let rho = 0.5 * (atmo.density(&f, p0) + atmo.density(&f, p));
        let k = rho * 0.1194 * std::f64::consts::PI / (8.0 * r.bc);
        let expect = 200.0 * (-k * x).exp();
        assert!((v.length() - expect).abs() / expect < 2e-3, "{} vs {}", v.length(), expect);
    }

    #[test]
    fn g7_table_is_monotonic_where_expected() {
        assert!(cd_g7(0.5) < cd_g7(1.0));
        assert!(cd_g7(1.05) > cd_g7(2.0));
        assert!(cd_g7(10.0) > 0.0);
    }
}
