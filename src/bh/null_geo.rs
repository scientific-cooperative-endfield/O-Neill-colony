//! CPU reference of the per-pixel null-geodesic tracer in `bh_trace.wgsl`.
//!
//! A light ray stays in the plane spanned by the camera position and the ray
//! direction. With u = 1/r (M = 1) the Schwarzschild orbit equation is
//!   d²u/dφ² + u = 3u²,
//! integrated with RK4 in φ. The step is adaptive: a base step in φ, reduced
//! near the photon sphere (u ≈ 1/3) and capped so that u changes by at most
//! 30 % per step (radial rays approach/escape geometrically). Rays end when
//! captured (r < 2M·1.001, moving inwards), when they escape (r > r_escape,
//! moving outwards; the asymptotic direction is then extrapolated along the
//! remaining straight line), or after `max_steps`.
//!
//! The WGSL code is a line-by-line port of `step_size`, `rk4` and `trace`.
#![allow(dead_code)]

pub const BASE_STEP: f64 = 0.07;
pub const U_CAPTURE: f64 = 1.0 / (2.0 * 1.001);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fate {
    Captured,
    /// asymptotic azimuth (in the ray plane) of the escape direction
    Escaped(f64),
}

pub fn step_size(u: f64, w: f64, scale: f64) -> f64 {
    let ps = (3.0 * u - 1.0).abs();
    let mut h = scale * BASE_STEP * (0.25 + 0.75 * (ps * 3.0).clamp(0.0, 1.0));
    if w.abs() > 1e-30 {
        h = h.min(0.3 * u.abs() / w.abs());
    }
    h.max(1e-7)
}

#[inline]
fn acc(u: f64) -> f64 {
    3.0 * u * u - u
}

pub fn rk4(u: f64, w: f64, h: f64) -> (f64, f64) {
    let (k1u, k1w) = (w, acc(u));
    let (k2u, k2w) = (w + 0.5 * h * k1w, acc(u + 0.5 * h * k1u));
    let (k3u, k3w) = (w + 0.5 * h * k2w, acc(u + 0.5 * h * k2u));
    let (k4u, k4w) = (w + h * k3w, acc(u + h * k3u));
    (u + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u), w + h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w))
}

/// Initial (u, du/dφ) of a backward ray at radius r0 whose direction, measured
/// in the local free-fall ("rain") frame, makes cos α with the outward radial.
/// dr/dφ = r (cos α + √(2M/r)) / sin α; valid on both sides of the horizon.
pub fn initial_uw(r0: f64, cos_a: f64) -> (f64, f64) {
    let sin_a = (1.0 - cos_a * cos_a).max(1e-14).sqrt();
    let u = 1.0 / r0;
    let s = (2.0 / r0).sqrt();
    (u, -u * (cos_a + s) / sin_a)
}

/// Integrates a ray in its plane. Returns (fate, steps used, final φ).
pub fn trace(u0: f64, w0: f64, r_escape: f64, max_steps: u32, scale: f64) -> (Fate, u32, f64) {
    let u_esc = 1.0 / r_escape.max(1.01 / u0);
    let (mut u, mut w, mut phi) = (u0, w0, 0.0);
    if u <= u_esc && w < 0.0 {
        return (Fate::Escaped(phi + u.atan2(-w)), 0, phi);
    }
    for i in 0..max_steps {
        let h = step_size(u, w, scale);
        let (u1, w1) = rk4(u, w, h);
        u = u1;
        w = w1;
        phi += h;
        if !u.is_finite() || u > 1e6 || (u > U_CAPTURE && w > 0.0) {
            return (Fate::Captured, i + 1, phi);
        }
        if u < u_esc && w < 0.0 {
            return (Fate::Escaped(phi + u.atan2(-w)), i + 1, phi);
        }
    }
    if w > 0.0 || u > 0.3 {
        (Fate::Captured, max_steps, phi)
    } else {
        (Fate::Escaped(phi + u.atan2(-w)), max_steps, phi)
    }
}

/// Ray arriving from r = 1/u0 with impact parameter b, moving inwards.
pub fn trace_impact(b: f64, u0: f64, max_steps: u32) -> (Fate, u32) {
    let w0 = (1.0 / (b * b) - u0 * u0 * (1.0 - 2.0 * u0)).max(0.0).sqrt();
    let (f, n, _) = trace(u0, w0, 1.0 / u0, max_steps, 1.0);
    (f, n)
}

/// Deflection angle of a ray with impact parameter b (from r = 1/u0 back to r = 1/u0).
pub fn deflection(b: f64, u0: f64) -> f64 {
    match trace_impact(b, u0, 4096).0 {
        Fate::Escaped(phi_inf) => {
            // straight line u = sin(φ + φ0)/b reaches infinity at φ = π − φ0
            let phi0 = (u0 * b).asin();
            phi_inf - (std::f64::consts::PI - phi0)
        }
        Fate::Captured => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BC: f64 = 5.196_152_422_706_632; // √27

    #[test]
    fn capture_boundary_is_sqrt27() {
        let u0 = 1e-4;
        assert_eq!(trace_impact(0.999 * BC, u0, 512).0, Fate::Captured);
        assert!(matches!(trace_impact(1.001 * BC, u0, 512).0, Fate::Escaped(_)));
        let (mut lo, mut hi) = (5.0, 5.4);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if trace_impact(mid, u0, 512).0 == Fate::Captured {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let b = 0.5 * (lo + hi);
        assert!(((b - BC) / BC).abs() < 1e-3, "boundary at b = {b}");
    }

    #[test]
    fn weak_field_deflection() {
        for b in [500.0, 1000.0, 3000.0] {
            let d = deflection(b, 1e-6);
            let weak = 4.0 / b;
            assert!(((d - weak) / weak).abs() < 0.01, "b={b}: {d} vs {weak}");
        }
    }

    #[test]
    fn photon_sphere_is_unstable_circular_orbit() {
        // exact circular orbit at u = 1/3 stays put
        let (mut u, mut w) = (1.0 / 3.0, 0.0);
        for _ in 0..2000 {
            (u, w) = rk4(u, w, 0.01);
        }
        assert!((u - 1.0 / 3.0).abs() < 1e-9 && w.abs() < 1e-9);
        // tiny perturbations run away to either side (instability)
        for (eps, inward) in [(1e-6, true), (-1e-6, false)] {
            let (mut u, mut w): (f64, f64) = (1.0 / 3.0 + eps, 0.0);
            let mut n = 0;
            while (u - 1.0 / 3.0).abs() < 0.05 && n < 100_000 {
                (u, w) = rk4(u, w, 0.01);
                n += 1;
            }
            assert!(n < 100_000);
            assert_eq!(u > 1.0 / 3.0, inward);
        }
        // and no circular photon orbit exists elsewhere: u'' = 0 only at u = 1/3
        for u in [0.2, 0.3, 0.36, 0.45] {
            assert!(acc(u).abs() > 1e-3);
        }
    }

    #[test]
    fn camera_rays_inside_and_outside_horizon() {
        // outside: looking straight out escapes, straight in is captured
        let (u, w) = initial_uw(10.0, 1.0);
        assert!(matches!(trace(u, w, 1000.0, 512, 1.0).0, Fate::Escaped(_)));
        let (u, w) = initial_uw(10.0, -1.0);
        assert_eq!(trace(u, w, 1000.0, 512, 1.0).0, Fate::Captured);
        // just inside the horizon, rays looking outwards still come from outside
        let (u, w) = initial_uw(1.9, 0.9);
        assert!(w < 0.0, "backward ray must move outwards");
        assert!(matches!(trace(u, w, 1000.0, 512, 1.0).0, Fate::Escaped(_)));
    }

    #[test]
    fn step_budget_is_bounded() {
        // the typical escaping ray uses far fewer than the 512-step budget
        let mut total = 0;
        for i in 0..64 {
            let b = 6.0 + i as f64 * 0.5;
            total += trace_impact(b, 1.0 / 30.0, 512).1;
        }
        assert!(total / 64 < 200, "mean steps {}", total / 64);
    }
}
