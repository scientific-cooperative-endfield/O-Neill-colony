//! The non-inertial frame the game is played in, and the air in it.
//!
//! Everything is integrated in the rotating frame of the station (or of the
//! player's fragment after a break-up). There is no real gravity inside the
//! colony: the floor pushes you towards the axis because the floor rotates.
//! The acceleration of a free body relative to that frame is exact:
//!
//!   a = −Ω × (Ω × d) − 2 Ω × v + a_grav(com + d) − a_grav(com)
//!
//! with d = p − com. The last two terms are the *differential* gravity of the
//! black hole (the frame itself free-falls, so only the difference acts): we
//! evaluate the Paczyński–Wiita acceleration at both points (no linearisation),
//! or, without a hole, an optional linear tidal tensor (e.g. Marck's tensor
//! from `relativity::tidal_tensor` supplied by the host).

use crate::det;
use crate::geom::{self, R};
use glam::{DMat3, DVec3};

/// Black hole seen from the station frame.
#[derive(Clone, Debug, PartialEq)]
pub struct BhField {
    /// GM/c² (m)
    pub m: f64,
    /// station centre of mass relative to the hole (m, BH frame)
    pub com_bh: DVec3,
    /// rotation colony frame → BH frame
    pub to_bh: DMat3,
}

impl BhField {
    /// Differential PW acceleration at offset `d` (colony frame) from the COM.
    pub fn differential(&self, d: DVec3) -> DVec3 {
        let x = self.com_bh + self.to_bh * d;
        let a = relativity::pw_accel(self.m, x) - relativity::pw_accel(self.m, self.com_bh);
        self.to_bh.transpose() * a
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StationFrame {
    /// angular velocity of the frame (rad/s, colony axes)
    pub spin: DVec3,
    /// centre of rotation / free-fall reference point (colony coordinates)
    pub com: DVec3,
    /// linear tidal tensor (s⁻²) used when `bh` is None
    pub tidal: DMat3,
    pub bh: Option<BhField>,
    /// switches for analysis and tests
    pub coriolis: bool,
    pub centrifugal: bool,
}

impl StationFrame {
    /// Canonical Island Three: ω = √(g₀/R), 1 g on the floor, ≈ 127 s per turn.
    pub fn colony() -> Self {
        Self::with_floor_g(geom::G0)
    }

    pub fn with_floor_g(g: f64) -> Self {
        StationFrame {
            spin: DVec3::new(0.0, 0.0, (g / R).sqrt()),
            com: DVec3::new(0.0, 0.0, 0.5 * geom::L),
            tidal: DMat3::ZERO,
            bh: None,
            coriolis: true,
            centrifugal: true,
        }
    }

    pub fn omega(&self) -> f64 {
        self.spin.length()
    }

    /// Gravity-like part (no velocity dependence).
    pub fn accel_static(&self, p: DVec3) -> DVec3 {
        let d = p - self.com;
        let w = self.spin;
        let mut a = match &self.bh {
            Some(b) => b.differential(d),
            None => self.tidal * d,
        };
        if self.centrifugal {
            a -= w.cross(w.cross(d));
        }
        a
    }

    /// Full acceleration of a free body at `p` moving with `v` relative to the frame.
    #[inline]
    pub fn accel(&self, p: DVec3, v: DVec3) -> DVec3 {
        let mut a = self.accel_static(p);
        if self.coriolis {
            a -= 2.0 * self.spin.cross(v);
        }
        a
    }

    /// Tidal stretch (m/s²) between two points `len` apart along `axis` at `p`:
    /// the part of the field gradient that pulls a body apart (centrifugal
    /// gradient ω² included — it is real tidal stress in the rotating frame).
    pub fn stretch(&self, p: DVec3, axis: DVec3, len: f64) -> f64 {
        let h = 0.5 * len * axis;
        (self.accel_static(p + h) - self.accel_static(p - h)).dot(axis).max(0.0)
    }

    /// Velocity of the frame point `p` relative to the inertial frame of the
    /// COM (m/s): used for proper time.
    pub fn frame_velocity(&self, p: DVec3) -> DVec3 {
        self.spin.cross(p - self.com)
    }
}

/// Air inside the hull: isothermal, hydrostatic in the rotating frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Atmosphere {
    /// pressure at the floor while sealed (Pa)
    pub p_floor0: f64,
    /// current fraction of the sealed pressure (venting lowers it)
    pub frac: f64,
    pub temperature: f64,
    pub molar_mass: f64,
    /// pressurised volume (m³)
    pub volume: f64,
}

pub const R_GAS: f64 = 8.314_462_618;

impl Default for Atmosphere {
    fn default() -> Self {
        Atmosphere { p_floor0: 101_325.0, frac: 1.0, temperature: 288.15, molar_mass: 0.028_964_7, volume: std::f64::consts::PI * R * R * geom::L }
    }
}

impl Atmosphere {
    /// Hydrostatic factor: p(r)/p(R) = exp(−(Mω²/2RT)(R² − r²)) (= e^−0.237 at the axis for 1 g).
    pub fn profile(&self, frame: &StationFrame, p: DVec3) -> f64 {
        let w = frame.omega();
        let r = geom::r_perp(p);
        let k = self.molar_mass * w * w / (2.0 * R_GAS * self.temperature);
        det::exp(-k * (R * R - r * r))
    }
    /// Static pressure (Pa); zero outside the hull.
    pub fn pressure(&self, frame: &StationFrame, p: DVec3) -> f64 {
        if !geom::inside_hull(p) {
            return 0.0;
        }
        self.p_floor0 * self.frac * self.profile(frame, p)
    }
    /// Density (kg/m³).
    pub fn density(&self, frame: &StationFrame, p: DVec3) -> f64 {
        self.pressure(frame, p) * self.molar_mass / (R_GAS * self.temperature)
    }
    pub fn speed_of_sound(&self) -> f64 {
        (1.4 * R_GAS * self.temperature / self.molar_mass).sqrt()
    }
    /// Choked-flow venting through `area` (m²) for `dt` seconds:
    /// dP/dt = −P · C_d A c* / V, c* = c·√(2/(γ+1))^((γ+1)/(γ−1))·… ≈ 0.58 c.
    pub fn vent(&mut self, area: f64, dt: f64) {
        if area <= 0.0 || self.frac <= 0.0 {
            return;
        }
        let rate = 0.58 * self.speed_of_sound() * area / self.volume;
        self.frac *= det::exp(-rate * dt);
        if self.frac < 1e-6 {
            self.frac = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_feels_one_g_and_axis_is_weightless() {
        let f = StationFrame::colony();
        let p = geom::surf_to_world(1234.0, 16000.0, 0.0);
        let a = f.accel_static(p);
        assert!((a.length() - geom::G0).abs() < 1e-9);
        assert!(a.dot(geom::up_at(p)) < 0.0); // pushed to the floor
        assert!(f.accel_static(DVec3::new(0.0, 0.0, 16000.0)).length() < 1e-12);
    }

    #[test]
    fn coriolis_on_a_runner_is_two_omega_v() {
        let f = StationFrame::colony();
        let p = geom::surf_to_world(0.0, 16000.0, 0.0);
        let (east, up, _) = geom::basis_at(p);
        let v = east * 7.0; // running spinward
        let extra = (f.accel(p, v) - f.accel_static(p)).dot(-up);
        assert!((extra - 2.0 * f.omega() * 7.0).abs() < 1e-12);
    }

    #[test]
    fn hydrostatic_air() {
        let f = StationFrame::colony();
        let a = Atmosphere::default();
        let floor = a.density(&f, geom::surf_to_world(0.0, 100.0, 0.0));
        assert!((floor - 1.225).abs() < 0.01);
        let axis = a.pressure(&f, DVec3::new(0.0, 0.0, 100.0)) / a.pressure(&f, geom::surf_to_world(0.0, 100.0, 0.0));
        assert!((axis - (-0.2376f64).exp()).abs() < 2e-3);
        assert_eq!(a.density(&f, geom::surf_to_world(0.0, 100.0, -1.0)), 0.0);
    }

    #[test]
    fn pw_differential_matches_newtonian_tide_far_away() {
        // 10 M☉ hole, station at 1e9 m: tide along r̂ is 2GM d / r³
        let m = relativity::m_geo(10.0);
        let r = 1.0e9;
        let mut f = StationFrame::colony();
        f.centrifugal = false;
        f.bh = Some(BhField { m, com_bh: DVec3::new(r, 0.0, 0.0), to_bh: DMat3::IDENTITY });
        let d = 100.0;
        let a = f.accel_static(f.com + DVec3::new(d, 0.0, 0.0));
        let expect = 2.0 * m * geom::C_LIGHT * geom::C_LIGHT * d / (r * r * r);
        assert!(((a.x - expect) / expect).abs() < 1e-3, "{} vs {}", a.x, expect);
    }
}
