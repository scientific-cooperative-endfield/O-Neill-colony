//! Schwarzschild geometry in Painlevé–Gullstrand (PG) coordinates.
//!
//! Geometric units: lengths and times in metres (c = 1), `m` = GM/c² in metres.
//! Metric: ds² = −dT² + |dx − v dT|², v = −√(2m/r) r̂ ("river" velocity).
//! Hamiltonian H = ½(−(p_T + v·p)² + p²). For timelike geodesics with the
//! normalisation H = −½ and A = dT/dτ = −(p_T + v·p) = √(1 + |p|²):
//!   dx/dT = v + p/A,   dp/dT = (s/r)(p − 1.5 r̂ (r̂·p)),   dτ/dT = 1/A,
//! where s = √(2m/r) and p is the spatial covariant momentum per unit mass
//! (= γβ of the body relative to the local free-falling "rain" frame).
//! Every right-hand side is regular at r = 2m; only r → 0 is singular.
#![allow(dead_code)]

use glam::DVec3;

pub const G: f64 = 6.674_30e-11;
pub const C: f64 = 299_792_458.0;
pub const MSUN: f64 = 1.988_47e30;
pub const G_EARTH: f64 = 9.806_65;

#[derive(Clone, Copy, Debug)]
pub struct MassPreset {
    pub name: &'static str,
    pub solar: f64,
}

pub const PRESETS: [MassPreset; 3] = [
    MassPreset { name: "Звёздная (10 M☉)", solar: 10.0 },
    MassPreset { name: "Sgr A* (4,3·10⁶ M☉)", solar: 4.3e6 },
    MassPreset { name: "Сверхмассивная (10⁸ M☉)", solar: 1.0e8 },
];

/// GM/c² in metres.
pub fn m_geo(solar: f64) -> f64 {
    G * solar * MSUN / (C * C)
}

/// Radius where the end-to-end tidal difference over `len` reaches `da` (m/s²).
pub fn tidal_radius(m: f64, len: f64, da: f64) -> f64 {
    (2.0 * m * C * C * len / da).cbrt()
}

pub fn river(m: f64, x: DVec3) -> DVec3 {
    let r = x.length();
    -(2.0 * m / r).sqrt() * x / r
}

/// T = t + F(r); returns None at/inside the horizon (t is undefined there).
pub fn f_shift(m: f64, r: f64) -> Option<f64> {
    if r <= 2.0 * m {
        return None;
    }
    let a = (2.0 * m).sqrt();
    let sr = r.sqrt();
    Some(2.0 * (2.0 * m * r).sqrt() + 2.0 * m * ((sr - a) / (sr + a)).ln())
}

pub fn f_shift_dr(m: f64, r: f64) -> f64 {
    (2.0 * m / r).sqrt() / (1.0 - 2.0 * m / r)
}

#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub x: DVec3,
    /// spatial covariant momentum per unit mass
    pub p: DVec3,
    /// proper time (m)
    pub tau: f64,
    /// PG time T (m)
    pub t_pg: f64,
    pub dead: bool,
}

impl Geo {
    /// Momentarily at rest w.r.t. a static observer.
    pub fn at_rest(m: f64, x: DVec3) -> Self {
        Self::from_coord_velocity(m, x, DVec3::ZERO)
    }
    /// Free fall from rest at infinity.
    pub fn rain(x: DVec3) -> Self {
        Geo { x, p: DVec3::ZERO, tau: 0.0, t_pg: 0.0, dead: false }
    }
    /// From coordinate velocity w = dx/dT (units of c).
    pub fn from_coord_velocity(m: f64, x: DVec3, w: DVec3) -> Self {
        let beta = w - river(m, x);
        Self::from_rain_velocity(x, beta)
    }
    /// From the velocity relative to the local rain frame.
    pub fn from_rain_velocity(x: DVec3, beta: DVec3) -> Self {
        let b2 = beta.length_squared().min(1.0 - 1e-12);
        let g = 1.0 / (1.0 - b2).sqrt();
        Geo { x, p: beta * g, tau: 0.0, t_pg: 0.0, dead: false }
    }
    /// From a velocity `u` measured by the local static observer (r > 2m).
    pub fn from_static_velocity(m: f64, x: DVec3, u: DVec3) -> Self {
        let vs = -river(m, x); // static observer moves outward relative to rain
        Self::from_rain_velocity(x, velocity_add(vs, u))
    }
    pub fn r(&self) -> f64 {
        self.x.length()
    }
    /// dT/dτ
    pub fn dt_dtau(&self, _m: f64) -> f64 {
        (1.0 + self.p.length_squared()).sqrt()
    }
    /// dx/dT
    pub fn coord_velocity(&self, m: f64) -> DVec3 {
        river(m, self.x) + self.p / self.dt_dtau(m)
    }
    /// velocity relative to the rain frame
    pub fn rain_velocity(&self, m: f64) -> DVec3 {
        self.p / self.dt_dtau(m)
    }
    /// contravariant 4-velocity (u^T, u^x, u^y, u^z) in PG coordinates
    pub fn four_velocity(&self, m: f64) -> [f64; 4] {
        let a = self.dt_dtau(m);
        let u = river(m, self.x) * a + self.p;
        [a, u.x, u.y, u.z]
    }
    /// Schwarzschild coordinate time t = T − F(r).
    pub fn t_schw(&self, m: f64) -> Option<f64> {
        f_shift(m, self.r()).map(|f| self.t_pg - f)
    }
    /// dτ/dt (Schwarzschild t); None inside the horizon.
    pub fn dtau_dt(&self, m: f64) -> Option<f64> {
        rates(m, self.x, self.coord_velocity(m)).map(|(a, b)| a / b)
    }
    /// specific energy E = −p_T
    pub fn energy(&self, m: f64) -> f64 {
        self.dt_dtau(m) + river(m, self.x).dot(self.p)
    }

    fn deriv(m: f64, x: DVec3, p: DVec3) -> (DVec3, DVec3, f64) {
        let r = x.length();
        let rh = x / r;
        let s = (2.0 * m / r).sqrt();
        let a = (1.0 + p.length_squared()).sqrt();
        let dx = -s * rh + p / a;
        let dp = (s / r) * (p - 1.5 * rh * rh.dot(p));
        (dx, dp, 1.0 / a)
    }

    /// One adaptive RK4 step (used by the conservation tests); returns the step.
    pub fn step_once(&mut self, m: f64) -> f64 {
        let h = self.h_max(m);
        self.rk4_t(m, h);
        h
    }

    /// Conserved specific angular momentum vector L = x × p.
    pub fn angular_momentum(&self) -> DVec3 {
        self.x.cross(self.p)
    }

    fn rk4_t(&mut self, m: f64, h: f64) {
        let (x0, p0) = (self.x, self.p);
        let (k1x, k1p, k1t) = Self::deriv(m, x0, p0);
        let (k2x, k2p, k2t) = Self::deriv(m, x0 + k1x * (h * 0.5), p0 + k1p * (h * 0.5));
        let (k3x, k3p, k3t) = Self::deriv(m, x0 + k2x * (h * 0.5), p0 + k2p * (h * 0.5));
        let (k4x, k4p, k4t) = Self::deriv(m, x0 + k3x * h, p0 + k3p * h);
        self.x = x0 + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) * (h / 6.0);
        self.p = p0 + (k1p + 2.0 * k2p + 2.0 * k3p + k4p) * (h / 6.0);
        self.tau += (k1t + 2.0 * k2t + 2.0 * k3t + k4t) * (h / 6.0);
        self.t_pg += h;
    }

    fn h_max(&self, m: f64) -> f64 {
        let r = self.r().max(1e-9 * m);
        let a = self.dt_dtau(m);
        // orbital timescale, shortened for ultra-relativistic momenta
        0.004 * (r * r * r / (2.0 * m)).sqrt() / a.sqrt()
    }

    /// Advance by PG time `dt` (m).
    pub fn step_t(&mut self, m: f64, dt: f64) {
        if self.dead {
            return;
        }
        let mut left = dt;
        let mut n = 0;
        while left > 0.0 && n < 20000 {
            if self.r() < 0.05 * m {
                self.dead = true; // singularity guard
                return;
            }
            let h = self.h_max(m).min(left);
            self.rk4_t(m, h);
            left -= h;
            n += 1;
        }
        if left > 0.0 {
            // budget exhausted: finish in one (coarse) step to keep clocks in sync
            self.rk4_t(m, left);
        }
    }

    /// Advance by proper time `dtau` (m).
    pub fn step_tau(&mut self, m: f64, dtau: f64) {
        let target = self.tau + dtau;
        let mut n = 0;
        while self.tau < target && !self.dead && n < 20000 {
            let a = self.dt_dtau(m);
            let h = self.h_max(m).min((target - self.tau) * a);
            if self.r() < 0.05 * m {
                self.dead = true;
                return;
            }
            self.rk4_t(m, h.max(1e-12 * m));
            n += 1;
        }
    }
}

/// (dτ/dT, dt/dT) for a worldline at x with coordinate velocity w = dx/dT.
/// None inside the horizon (t undefined).
pub fn rates(m: f64, x: DVec3, w: DVec3) -> Option<(f64, f64)> {
    let r = x.length();
    if r <= 2.0 * m {
        return None;
    }
    let rel = w - river(m, x);
    let dtau = (1.0 - rel.length_squared()).max(0.0).sqrt();
    let dr = w.dot(x / r);
    let dt = 1.0 - f_shift_dr(m, r) * dr;
    Some((dtau, dt))
}

pub fn dtau_dt_static(m: f64, r: f64) -> f64 {
    (1.0 - 2.0 * m / r).sqrt()
}

/// Relativistic composition: velocity u measured in a frame moving with V.
pub fn velocity_add(v: DVec3, u: DVec3) -> DVec3 {
    let v2 = v.length_squared();
    if v2 < 1e-30 {
        return u;
    }
    let g = 1.0 / (1.0 - v2).sqrt();
    let vu = v.dot(u);
    (v + u / g + v * (g / (1.0 + g)) * vu) / (1.0 + vu)
}

/// Exact Newtonian-form tidal acceleration of separation d (m/s²).
/// In the local free-fall frame this equals the Schwarzschild tidal tensor
/// (radial −2M/r³, transverse +M/r³) for radial and transverse separations.
pub fn tidal_accel(m: f64, x: DVec3, d: DVec3) -> DVec3 {
    let r = x.length();
    let rh = x / r;
    let k = m * C * C / (r * r * r);
    (3.0 * d.dot(rh) * rh - d) * k
}

/// Tidal tensor K (s⁻², a = K·d for a separation d in metres) in the free-fall
/// frame of a geodesic, expressed on the BH-frame axes. Exact for Schwarzschild
/// (Marck's parallel-propagated frame): with x = L²/r²,
///   stretch along r̂: (2 + 3x) M/r³, compression along L̂: (1 + 3x) M/r³,
///   compression along L̂ × r̂: M/r³.
/// `m` is GM/c² in metres, positions of `g` are in units of M.
pub fn tidal_tensor(m: f64, g: &Geo) -> glam::DMat3 {
    use glam::DMat3;
    let r = g.r().max(1e-6);
    let e1 = g.x / r;
    let l = g.x.cross(g.p);
    let x = l.length_squared() / (r * r);
    let e3 = if l.length_squared() > 1e-24 { l.normalize() } else { e1.any_orthonormal_vector() };
    let e2 = e3.cross(e1);
    let k = C * C / (m * m * r * r * r);
    let outer = |a: DVec3| DMat3::from_cols(a * a.x, a * a.y, a * a.z);
    (outer(e1) * (2.0 + 3.0 * x) - outer(e3) * (1.0 + 3.0 * x) - outer(e2)) * k
}

/// Paczyński–Wiita acceleration (m/s²) for x in metres.
pub fn pw_accel(m: f64, x: DVec3) -> DVec3 {
    let r = x.length();
    let d = (r - 2.0 * m).max(1e-6 * m);
    -x / r * (m * C * C / (d * d))
}

/// Specific angular momentum² of PW circular orbits (units of m²c²/m² → m²).
pub fn pw_l2(m: f64, r: f64) -> f64 {
    m * r * r * r / ((r - 2.0 * m) * (r - 2.0 * m))
}

/// Bound geodesic with Schwarzschild turning points `rp` < `ra`, started at
/// apocentre on the +x axis, moving towards +y, in the z = 0 plane.
/// From E² = (1−2M/r)(1+L²/r²) at both turning points.
pub fn elliptic_orbit(m: f64, rp: f64, ra: f64) -> Geo {
    let a = 1.0 - 2.0 * m / ra;
    let b = 1.0 - 2.0 * m / rp;
    let l2 = (a - b) / (b / (rp * rp) - a / (ra * ra));
    let e = (a * (1.0 + l2 / (ra * ra))).sqrt();
    // static observer at apocentre: E = γ√A, L = γ v r
    let gamma = e / a.sqrt();
    let v = l2.sqrt() / (gamma * ra);
    Geo::from_static_velocity(m, DVec3::new(ra, 0.0, 0.0), DVec3::new(0.0, v, 0.0))
}

/// Proper fall time from rest at r0 down to r (cycloid solution).
pub fn fall_time_analytic(m: f64, r0: f64, r: f64) -> f64 {
    let eta = (2.0 * r / r0 - 1.0).clamp(-1.0, 1.0).acos();
    (r0 * r0 * r0 / (8.0 * m)).sqrt() * (eta + eta.sin())
}

/// Energy (in units of its energy at infinity) of an outgoing radial photon,
/// measured by an observer with PG 4-velocity `u` = (u^T, u^x, u^y, u^z) at `x`.
/// PG covariant momentum: p_T = −1, p_r = 1/(1 − √(2M/r)). None at/inside the horizon.
pub fn radial_photon_g(m: f64, x: DVec3, u: [f64; 4]) -> Option<f64> {
    let r = x.length();
    if r <= 2.0 * m * 1.0001 {
        return None;
    }
    let pr = 1.0 / (1.0 - (2.0 * m / r).sqrt());
    let rh = x / r;
    Some(u[0] - pr * (rh.x * u[1] + rh.y * u[2] + rh.z * u[3]))
}

#[cfg(test)]
mod tests {
    use super::*;
    const M: f64 = 1.0;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn static_observer() {
        for r in [2.05, 3.0, 6.0, 20.0, 1000.0] {
            let x = DVec3::new(r, 0.0, 0.0);
            let g = Geo::at_rest(M, x);
            let (a, b) = rates(M, x, g.coord_velocity(M)).unwrap();
            assert!(rel(a / b, dtau_dt_static(M, r)) < 1e-9, "r={r}");
            // at rest: energy equals sqrt(1-2M/r)
            assert!(rel(g.energy(M), dtau_dt_static(M, r)) < 1e-9);
        }
    }

    #[test]
    fn circular_orbit() {
        for r in [4.0, 6.0, 10.0, 30.0] {
            let x = DVec3::new(r, 0.0, 0.0);
            let w = DVec3::new(0.0, (M / r).sqrt(), 0.0);
            let mut g = Geo::from_coord_velocity(M, x, w);
            let period = std::f64::consts::TAU * (r * r * r / M).sqrt();
            let t0 = g.t_schw(M).unwrap();
            let mut rmax: f64 = 0.0;
            for _ in 0..200 {
                g.step_t(M, period / 200.0);
                rmax = rmax.max((g.r() - r).abs() / r);
            }
            let t1 = g.t_schw(M).unwrap();
            let ratio = g.tau / (t1 - t0);
            assert!(rmax < 1e-3, "r={r} drift {rmax}");
            assert!(rel(ratio, (1.0 - 3.0 * M / r).sqrt()) < 1e-3, "r={r} {ratio}");
        }
    }

    #[test]
    fn rain_observer() {
        let mut g = Geo::rain(DVec3::new(0.0, 0.0, 40.0));
        for _ in 0..1000 {
            let (t0, r0, tau0) = (g.t_schw(M), g.r(), g.tau);
            g.step_t(M, 0.01);
            if g.r() < 2.2 {
                break;
            }
            let (t1, r1) = (g.t_schw(M).unwrap(), g.r());
            let rm = 0.5 * (r0 + r1);
            let ratio = (g.tau - tau0) / (t1 - t0.unwrap());
            assert!(rel(ratio, 1.0 - 2.0 * M / rm) < 1e-3, "r={rm} {ratio}");
        }
    }

    #[test]
    fn radial_fall_time() {
        for r0 in [6.0, 10.0, 50.0] {
            let mut g = Geo::at_rest(M, DVec3::new(r0, 0.0, 0.0));
            let mut tau_h = None;
            while !g.dead && g.r() > 0.5 {
                let prev = (g.r(), g.tau);
                g.step_t(M, 0.002 * r0.sqrt());
                if prev.0 > 2.0 && g.r() <= 2.0 {
                    let f = (prev.0 - 2.0) / (prev.0 - g.r());
                    tau_h = Some(prev.1 + f * (g.tau - prev.1));
                }
            }
            let tau_h = tau_h.expect("never crossed the horizon");
            let an = fall_time_analytic(M, r0, 2.0);
            assert!(rel(tau_h, an) < 1e-3, "r0={r0}: {tau_h} vs {an}");
        }
    }

    #[test]
    fn horizon_is_regular() {
        // crossing r = 2M produces finite, smooth momenta and clocks
        let mut g = Geo::at_rest(M, DVec3::new(3.0, 0.4, 0.0));
        while g.r() > 1.0 {
            g.step_t(M, 0.01);
            assert!(g.p.is_finite() && g.tau.is_finite() && g.x.is_finite());
            assert!(g.p.length() < 50.0);
        }
    }

    #[test]
    fn pw_isco_is_6m() {
        let mut best = (0.0, f64::MAX);
        let mut r = 4.0;
        while r < 10.0 {
            let l = pw_l2(M, r);
            if l < best.1 {
                best = (r, l);
            }
            r += 0.001;
        }
        assert!((best.0 - 6.0).abs() < 0.01);
    }

    #[test]
    fn energy_and_momentum_conserved() {
        // eccentric bound orbit (rp = 8M, ra = 30M), 10⁴ adaptive RK4 steps
        let mut g = elliptic_orbit(M, 8.0, 30.0);
        let (e0, l0) = (g.energy(M), g.angular_momentum());
        for _ in 0..10_000 {
            g.step_once(M);
        }
        let de = rel(g.energy(M), e0);
        let dl = (g.angular_momentum() - l0).length() / l0.length();
        assert!(de < 1e-6 && dl < 1e-6, "dE {de:e}, dL {dl:e}");
        assert!(g.r() > 7.9 && g.r() < 30.1);
    }

    #[test]
    fn elliptic_orbit_turning_points() {
        let mut g = elliptic_orbit(M, 8.0, 30.0);
        let (mut lo, mut hi) = (f64::MAX, 0.0f64);
        for _ in 0..4000 {
            g.step_t(M, 0.5);
            lo = lo.min(g.r());
            hi = hi.max(g.r());
        }
        assert!(rel(lo, 8.0) < 2e-3 && rel(hi, 30.0) < 2e-3, "{lo} {hi}");
    }

    #[test]
    fn isco_is_6m_for_geodesics() {
        // analytic: circular orbits exist for L² ≥ 12M², stable root r₊ ≥ 6M
        let mut rmin = f64::MAX;
        let mut l2: f64 = 12.0;
        while l2 < 40.0 {
            let rp = 0.5 * l2 * (1.0 + (1.0 - 12.0 / l2).sqrt());
            rmin = rmin.min(rp);
            l2 += 1e-4;
        }
        assert!((rmin - 6.0).abs() < 0.01, "rmin {rmin}");
        // numerical: slightly perturbed circular orbits survive above 6M and plunge below
        for (r, survives) in [(7.0, true), (5.6, false)] {
            let x = DVec3::new(r, 0.0, 0.0);
            let w = DVec3::new(0.0, (M / r).sqrt() * 0.999, 0.0);
            let mut g = Geo::from_coord_velocity(M, x, w);
            let mut rlo = r;
            for _ in 0..4000 {
                g.step_t(M, 0.5);
                rlo = rlo.min(g.r());
                if g.r() < 2.5 {
                    break;
                }
            }
            assert_eq!(rlo > 4.0, survives, "r = {r}: min radius {rlo}");
        }
    }

    #[test]
    fn newtonian_free_fall_time() {
        // fall from rest at r0 ≫ M: τ ≈ (π/2) √(r0³ / 2GM)
        let r0 = 2.0e4;
        let mut g = Geo::at_rest(M, DVec3::new(r0, 0.0, 0.0));
        let mut prev = (g.r(), g.tau);
        while g.r() > 2.0 {
            prev = (g.r(), g.tau);
            g.step_t(M, 20.0);
        }
        let f = (prev.0 - 2.0) / (prev.0 - g.r());
        let tau = prev.1 + f * (g.tau - prev.1);
        let newton = std::f64::consts::FRAC_PI_2 * (r0 * r0 * r0 / (2.0 * M)).sqrt();
        assert!(rel(tau, newton) < 0.01, "{tau} vs {newton}");
    }

    #[test]
    fn radial_tidal_acceleration() {
        // r = 1000 r_s, L = 32 km, Sgr A*: compare with 2GM·L/r³ and with the
        // difference of Newtonian and Paczyński–Wiita gravity across the body
        let m = m_geo(4.3e6);
        let r = 2000.0 * m;
        let len = 32_000.0;
        let x = DVec3::new(0.0, r, 0.0);
        let a = tidal_accel(m, x, DVec3::new(0.0, len, 0.0)).y;
        let an = 2.0 * G * 4.3e6 * MSUN * len / (r * r * r);
        assert!(rel(a, an) < 0.01, "{a} vs {an}");
        let newton = |rr: f64| -G * 4.3e6 * MSUN / (rr * rr);
        let direct = newton(r + 0.5 * len) - newton(r - 0.5 * len);
        assert!(rel(direct, an) < 0.01);
        let pw = pw_accel(m, DVec3::new(r + 0.5 * len, 0.0, 0.0)).x - pw_accel(m, DVec3::new(r - 0.5 * len, 0.0, 0.0)).x;
        assert!(rel(pw, an) < 0.01, "PW {pw} vs {an}");
    }

    #[test]
    fn tidal_tensor_circular_orbit() {
        // vertical tidal compression on a circular geodesic equals the proper-time
        // vertical epicyclic frequency² = (M/r³)/(1 − 3M/r); the tensor is traceless
        let m = m_geo(4.3e6);
        for r in [4.0f64, 6.0, 10.0, 50.0] {
            let v = (1.0 / (r - 2.0)).sqrt();
            let g = Geo::from_static_velocity(1.0, DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, v, 0.0));
            let k = tidal_tensor(m, &g);
            let unit = C * C / (m * m * r * r * r);
            let kz = k.z_axis.z / unit;
            assert!(rel(-kz, 1.0 / (1.0 - 3.0 / r)) < 1e-9, "r={r} kz={kz}");
            let tr = k.x_axis.x + k.y_axis.y + k.z_axis.z;
            assert!(tr.abs() < 1e-9 * unit);
            // radial infall: reduces to the Newtonian form 3 r̂r̂ − 1
            let f = Geo::rain(DVec3::new(r, 0.0, 0.0));
            let kf = tidal_tensor(m, &f);
            assert!(rel(kf.x_axis.x, 2.0 * unit) < 1e-12 && rel(-kf.y_axis.y, unit) < 1e-12);
        }
    }

    #[test]
    fn presets_match_spec() {
        let m = m_geo(10.0);
        assert!(rel(2.0 * m, 29_530.0) < 0.01, "rs {}", 2.0 * m);
        let rt = tidal_radius(m, 32_000.0, G_EARTH);
        assert!(rt > 1.8e8 && rt < 2.3e8, "rt {rt}");
        let ms = m_geo(4.3e6);
        let da = 2.0 * ms * C * C * 32_000.0 / (2.0 * ms).powi(3) / G_EARTH;
        assert!(da > 1.5 && da < 2.1, "Sgr A* {da}");
    }

    #[test]
    fn radial_photon_energy() {
        // static observer: 1/sqrt(1-2M/r); rain observer: 1/(1-sqrt(2M/r))
        let r = 7.0;
        let x = DVec3::new(r, 0.0, 0.0);
        let st = 1.0 / (1.0 - 2.0 / r).sqrt();
        let u_static = [st, 0.0, 0.0, 0.0];
        let g = radial_photon_g(1.0, x, u_static).unwrap();
        assert!((g - st).abs() < 1e-12, "static {g}");
        let rain = Geo::rain(x).four_velocity(1.0);
        let g = radial_photon_g(1.0, x, rain).unwrap();
        let want = 1.0 / (1.0 - (2.0 / r).sqrt());
        assert!((g / want - 1.0).abs() < 1e-9, "rain {g} vs {want}");
    }
}
