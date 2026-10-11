//! Kerr black hole (spin a = J/M, units M = 1) in Cartesian Kerr–Schild
//! coordinates — the rotating analogue of the Painlevé–Gullstrand chart:
//! regular across the future horizon, so rays and bodies can cross it.
//!
//!   g_μν = η_μν + f l_μ l_ν,   f = 2r³ / (r⁴ + a²z²),
//!   l_μ = (1, (rx + ay)/(r² + a²), (ry − ax)/(r² + a²), z/r),
//!   r⁴ − (ρ² − a²) r² − a²z² = 0  (ρ² = x² + y² + z²).
//!
//! Photons follow Hamilton's equations for H = ½ g^μν p_μ p_ν with
//! g^μν = η^μν − f l^μ l^ν (l^μ = (−1, l⃗)). E = −p_t and L_z = x p_y − y p_x
//! are conserved (Killing vectors ∂_t, ∂_φ); the tests check them and the
//! analytic shadow edge / ISCO.
use glam::DVec3;

/// Boyer–Lindquist radius of a Kerr–Schild point.
pub fn ks_r(a: f64, x: DVec3) -> f64 {
    let rho2 = x.length_squared();
    let w = rho2 - a * a;
    let r2 = 0.5 * w + (0.25 * w * w + a * a * x.z * x.z).sqrt();
    r2.max(0.0).sqrt()
}

/// f and the spatial part of l (covariant).
pub fn ks_fl(a: f64, x: DVec3) -> (f64, DVec3) {
    let r = ks_r(a, x).max(1e-9);
    let r2 = r * r;
    let f = 2.0 * r2 * r / (r2 * r2 + a * a * x.z * x.z);
    let d = r2 + a * a;
    let l = DVec3::new((r * x.x + a * x.y) / d, (r * x.y - a * x.x) / d, x.z / r);
    (f, l)
}

/// Outer horizon r₊ = 1 + √(1 − a²).
pub fn horizon(a: f64) -> f64 {
    1.0 + (1.0 - a * a).max(0.0).sqrt()
}

/// Innermost stable circular orbit (Bardeen, Press & Teukolsky 1972);
/// prograde for a > 0 when `pro`, retrograde otherwise.
pub fn isco(a: f64, pro: bool) -> f64 {
    let z1 = 1.0 + (1.0 - a * a).cbrt() * ((1.0 + a).cbrt() + (1.0 - a).cbrt());
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    let s = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).max(0.0).sqrt();
    if pro {
        3.0 + z2 - s
    } else {
        3.0 + z2 + s
    }
}

/// Equatorial circular photon orbit radius (prograde / retrograde).
pub fn photon_orbit(a: f64, pro: bool) -> f64 {
    let s = if pro { -a } else { a };
    2.0 * (1.0 + (2.0 / 3.0 * s.acos()).cos())
}

/// Critical impact parameter |L/E| of the equatorial photon orbit: 3√r_ph ∓ a.
pub fn critical_b(a: f64, pro: bool) -> f64 {
    let r = photon_orbit(a, pro);
    if pro {
        3.0 * r.sqrt() - a
    } else {
        3.0 * r.sqrt() + a
    }
}

/// Keplerian angular velocity of a prograde circular equatorial orbit.
pub fn kepler_omega(a: f64, r: f64) -> f64 {
    1.0 / (r.powf(1.5) + a)
}

/// Redshift factor g = E_obs/E_emit for a photon with L_z/E = ξ emitted by
/// the prograde Keplerian disk at r and received at infinity.
pub fn disk_g(a: f64, r: f64, xi: f64) -> Option<f64> {
    let om = kepler_omega(a, r);
    let gtt = -(1.0 - 2.0 / r);
    let gtp = -2.0 * a / r;
    let gpp = r * r + a * a + 2.0 * a * a / r;
    let n = -(gtt + 2.0 * om * gtp + om * om * gpp);
    if n <= 0.0 {
        return None;
    }
    let ut = 1.0 / n.sqrt();
    Some(1.0 / (ut * (1.0 - om * xi)))
}

/// Photon state: position and covariant momentum (p_t fixed = −E).
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub x: DVec3,
    pub p: DVec3,
    pub pt: f64,
    pub lambda: f64,
}

fn lp(a: f64, x: DVec3, p: DVec3, pt: f64) -> (f64, DVec3, f64) {
    let (f, l) = ks_fl(a, x);
    (f, l, -pt + l.dot(p))
}

/// H = ½ g^μν p_μ p_ν.
pub fn hamiltonian(a: f64, r: &Ray) -> f64 {
    let (f, _, s) = lp(a, r.x, r.p, r.pt);
    0.5 * (-r.pt * r.pt + r.p.length_squared() - f * s * s)
}

/// Analytic gradient of q = f (l·p)² (p_t = −1 photon convention generalised by `pt`).
/// Same formulas as `trace_kerr` in trace.wgsl.
pub fn grad_q(a: f64, x: DVec3, p: DVec3, pt: f64) -> DVec3 {
    let r = ks_r(a, x).max(1e-9);
    let (r2, a2) = (r * r, a * a);
    let rho2 = x.length_squared();
    let den_r = r * (2.0 * r2 - rho2 + a2);
    let ez = DVec3::Z;
    let dr = (x * r2 + ez * (a2 * x.z)) / den_r;
    let w = r2 * r2 + a2 * x.z * x.z;
    let f = 2.0 * r2 * r / w;
    let df = (dr * (6.0 * r2 * w) - (dr * (4.0 * r2 * r) + ez * (2.0 * a2 * x.z)) * (2.0 * r2 * r)) / (w * w);
    let d = r2 + a2;
    let lx = (r * x.x + a * x.y) / d;
    let ly = (r * x.y - a * x.x) / d;
    let lz = x.z / r;
    let s = -pt + lx * p.x + ly * p.y + lz * p.z;
    // ∂_i l_x, ∂_i l_y, ∂_i l_z as vectors over i
    let dlx = (dr * x.x + DVec3::new(r, a, 0.0)) / d - dr * ((r * x.x + a * x.y) * 2.0 * r / (d * d));
    let dly = (dr * x.y + DVec3::new(-a, r, 0.0)) / d - dr * ((r * x.y - a * x.x) * 2.0 * r / (d * d));
    let dlz = ez / r - dr * (x.z / r2);
    let ds = dlx * p.x + dly * p.y + dlz * p.z;
    df * (s * s) + ds * (2.0 * f * s)
}

fn deriv(a: f64, x: DVec3, p: DVec3, pt: f64) -> (DVec3, DVec3) {
    let (f, l, s) = lp(a, x, p, pt);
    let dx = p - l * (f * s);
    // dp/dλ = −∂H/∂x = ½ ∂(f (l·p)²)
    (dx, grad_q(a, x, p, pt) * 0.5)
}

impl Ray {
    /// Null ray at x heading along `n` (unit) with energy 1 (p_t = −1).
    pub fn new(a: f64, x: DVec3, n: DVec3) -> Ray {
        let (f, l) = ks_fl(a, x);
        let c = l.dot(n);
        // k²(1 − f c²) − 2 f c k − (1 + f) = 0
        let qa = 1.0 - f * c * c;
        let qb = -2.0 * f * c;
        let qc = -(1.0 + f);
        let k = (-qb + (qb * qb - 4.0 * qa * qc).sqrt()) / (2.0 * qa);
        Ray { x, p: n * k, pt: -1.0, lambda: 0.0 }
    }

    pub fn lz(&self) -> f64 {
        self.x.x * self.p.y - self.x.y * self.p.x
    }

    /// RK4 step in affine parameter.
    pub fn step(&mut self, a: f64, h: f64) {
        let (x, p, pt) = (self.x, self.p, self.pt);
        let (k1x, k1p) = deriv(a, x, p, pt);
        let (k2x, k2p) = deriv(a, x + k1x * (0.5 * h), p + k1p * (0.5 * h), pt);
        let (k3x, k3p) = deriv(a, x + k2x * (0.5 * h), p + k2p * (0.5 * h), pt);
        let (k4x, k4p) = deriv(a, x + k3x * h, p + k3p * h, pt);
        self.x += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) * (h / 6.0);
        self.p += (k1p + 2.0 * k2p + 2.0 * k3p + k4p) * (h / 6.0);
        self.lambda += h;
    }
}

/// Fate of a ray: true = captured by the hole.
pub fn captured(a: f64, mut ray: Ray, r_escape: f64) -> bool {
    let rh = horizon(a);
    for _ in 0..200_000 {
        let r = ks_r(a, ray.x);
        if r < rh * 1.01 {
            return true;
        }
        if r > r_escape && ray.x.dot(ray.p) > 0.0 {
            return false;
        }
        let h = 0.02 * (r - rh * 0.9).max(0.05).min(r * 0.5);
        ray.step(a, h);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isco_limits() {
        assert!((isco(0.0, true) - 6.0).abs() < 1e-12);
        assert!((isco(0.0, false) - 6.0).abs() < 1e-12);
        assert!((isco(1.0, true) - 1.0).abs() < 1e-9);
        assert!((isco(1.0, false) - 9.0).abs() < 1e-9);
        // a = 0.9: 2.3209 M prograde, 8.7173 M retrograde
        assert!((isco(0.9, true) - 2.320_883).abs() < 1e-5, "{}", isco(0.9, true));
        assert!((isco(0.9, false) - 8.717_352).abs() < 1e-5, "{}", isco(0.9, false));
    }

    #[test]
    fn schwarzschild_limit() {
        assert!((photon_orbit(0.0, true) - 3.0).abs() < 1e-12);
        assert!((critical_b(0.0, true) - 27f64.sqrt()).abs() < 1e-12);
        assert!((horizon(0.0) - 2.0).abs() < 1e-12);
        let (f, _) = ks_fl(0.0, DVec3::new(3.0, 4.0, 0.0));
        assert!((f - 0.4).abs() < 1e-12);
    }

    #[test]
    fn conserved_quantities() {
        let a = 0.9;
        let mut r = Ray::new(a, DVec3::new(40.0, 6.5, 1.0), DVec3::new(-1.0, 0.0, 0.0));
        let (lz0, h0) = (r.lz(), hamiltonian(a, &r));
        assert!(h0.abs() < 1e-12);
        for _ in 0..4000 {
            let rr = ks_r(a, r.x);
            if rr < 2.5 {
                break;
            }
            r.step(a, 0.02 * rr.min(20.0));
        }
        assert!((r.lz() - lz0).abs() < 1e-6 * lz0.abs(), "Lz {} vs {}", r.lz(), lz0);
        assert!(hamiltonian(a, &r).abs() < 1e-7, "H = {}", hamiltonian(a, &r));
    }

    /// The capture threshold of equatorial rays from far away reproduces the
    /// analytic shadow edge b_c = 3√r_ph ∓ a on both sides of a spinning hole.
    #[test]
    fn shadow_edge_matches_photon_orbits() {
        let a = 0.9;
        let d = 2000.0;
        for (pro, sign) in [(true, 1.0), (false, -1.0)] {
            let (mut lo, mut hi) = (1.0, 9.0);
            for _ in 0..22 {
                let b = 0.5 * (lo + hi);
                let ray = Ray::new(a, DVec3::new(d, sign * b, 0.0), DVec3::NEG_X);
                if captured(a, ray, d * 1.01) {
                    lo = b;
                } else {
                    hi = b;
                }
            }
            let b = 0.5 * (lo + hi);
            let want = critical_b(a, pro);
            // the ray starts at finite distance: b measured there differs by O(1/d)
            assert!((b - want).abs() < 2e-3 * want, "pro={pro}: b = {b}, analytic {want}");
        }
    }

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let a = 0.7;
        let p = DVec3::new(-0.8, 0.3, 0.2);
        for x in [DVec3::new(5.0, 2.0, 1.0), DVec3::new(-1.5, 2.2, -0.7), DVec3::new(0.3, -3.0, 2.5)] {
            let num = {
                let (f, l) = ks_fl(a, x);
                let _ = (f, l);
                let h = 1e-6;
                let q = |y: DVec3| {
                    let (f, l) = ks_fl(a, y);
                    let s = 1.0 + l.dot(p);
                    f * s * s
                };
                DVec3::new(
                    (q(x + DVec3::X * h) - q(x - DVec3::X * h)) / (2.0 * h),
                    (q(x + DVec3::Y * h) - q(x - DVec3::Y * h)) / (2.0 * h),
                    (q(x + DVec3::Z * h) - q(x - DVec3::Z * h)) / (2.0 * h),
                )
            };
            let an = grad_q(a, x, p, -1.0);
            assert!((an - num).length() < 1e-6 * (1.0 + num.length()), "{x}: {an} vs {num}");
        }
    }

    #[test]
    fn disk_redshift_static_limit() {
        // a = 0, photon with ξ = 0 from r = 6: g = √(1 − 3/r)
        let g = disk_g(0.0, 6.0, 0.0).unwrap();
        assert!((g - 0.5f64.sqrt()).abs() < 1e-12);
    }
}
