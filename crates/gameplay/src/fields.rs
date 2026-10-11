//! Spacetime fields created by the curvature devices.
//!
//! * **Bubble** — a thin metric shell of radius R_b. Its potential barrier U
//!   (J/kg) reflects anything whose radial speed satisfies ½v_r² < U and lets
//!   faster things through unchanged (conservative thin barrier). Inside, the
//!   lapse differs: proper time runs at `time_rate` × the station clock.
//! * **Micro black hole** — a real Schwarzschild hole of mass M (≈10¹¹ kg,
//!   r_s ≈ 10⁻¹⁶ m). Gravity is Paczyński–Wiita (exact Newtonian far away,
//!   regularised only at 5 cm, where anything is shredded anyway). It radiates
//!   Hawking luminosity P = ħc⁶ / (15360 π G² M²) — gigawatts — which the
//!   device's containment absorbs as heat. Uncontained, it irradiates the area.
//! * **Fold** — a traversable (Morris–Thorne-type) wormhole: a mouth at `pos`
//!   and one at `exit`. Whatever enters one mouth sphere leaves the other at the
//!   antipodal point with the same velocity. The throat has a tidal field that
//!   grows as the fold destabilises.

use crate::ecs::Entity;
use crate::geom::{C_LIGHT, G_NEWTON};
use glam::DVec3;

pub const HBAR: f64 = 1.054_571_817e-34;
pub const K_BOLTZMANN: f64 = 1.380_649e-23;
/// Containment leakage of the Hawking flux (fraction).
pub const CONTAINMENT_LEAK: f64 = 1e-7;
/// Regularisation radius of hole gravity (m).
pub const HOLE_SOFT: f64 = 0.05;

pub fn schwarzschild_radius(mass: f64) -> f64 {
    2.0 * G_NEWTON * mass / (C_LIGHT * C_LIGHT)
}
pub fn hawking_power(mass: f64) -> f64 {
    HBAR * C_LIGHT.powi(6) / (15360.0 * std::f64::consts::PI * G_NEWTON * G_NEWTON * mass * mass)
}
pub fn hawking_lifetime(mass: f64) -> f64 {
    5120.0 * std::f64::consts::PI * G_NEWTON * G_NEWTON * mass * mass * mass / (HBAR * C_LIGHT.powi(4))
}
pub fn hawking_temperature(mass: f64) -> f64 {
    HBAR * C_LIGHT.powi(3) / (8.0 * std::f64::consts::PI * G_NEWTON * mass * K_BOLTZMANN)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    Bubble { radius: f64, barrier: f64, time_rate: f64 },
    MicroHole { mass: f64, contained: bool },
    Fold { exit: DVec3, radius: f64, stability: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Field {
    pub kind: FieldKind,
    pub pos: DVec3,
    pub vel: DVec3,
    pub owner: Entity,
    pub age: f64,
    pub ttl: f64,
    /// heat dumped into the containment (J) — micro holes
    pub heat: f64,
}

/// What happens to something crossing a field boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Crossing {
    /// hit the bubble shell: outward normal at the crossing point
    Shell { normal: DVec3, barrier: f64 },
    /// entered a fold mouth: translate by `shift` (to the antipodal point of the other mouth)
    Fold { shift: DVec3, field: usize, stability: f64 },
}

/// Per-tick snapshot of all active fields (cheap to query, no ECS borrows).
#[derive(Clone, Debug, Default)]
pub struct FieldSet {
    pub items: Vec<Field>,
}

impl FieldSet {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Gravitational acceleration of the micro holes (m/s²).
    pub fn accel(&self, p: DVec3) -> DVec3 {
        let mut a = DVec3::ZERO;
        for f in &self.items {
            if let FieldKind::MicroHole { mass, .. } = f.kind {
                let d = p - f.pos;
                let r = d.length().max(HOLE_SOFT);
                let rs = schwarzschild_radius(mass);
                a -= d / r * (G_NEWTON * mass / ((r - rs) * (r - rs)));
            }
        }
        a
    }

    /// Head-to-foot stretch (m/s²) across `len` along `axis`.
    pub fn stretch(&self, p: DVec3, axis: DVec3, len: f64) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        let h = 0.5 * len * axis;
        let mut s = (self.accel(p + h) - self.accel(p - h)).dot(axis).max(0.0);
        for f in &self.items {
            if let FieldKind::Fold { radius, stability, exit } = f.kind {
                // throat tide felt within two mouth radii of either mouth
                let near = (p - f.pos).length().min((p - exit).length());
                if near < 2.0 * radius {
                    s += (1.0 - near / (2.0 * radius)) * len * 6.0 / stability.max(0.05);
                }
            }
        }
        s
    }

    /// dτ/dt (relative to the station clock) from the fields at `p`.
    pub fn time_rate(&self, p: DVec3) -> f64 {
        let mut k = 1.0;
        for f in &self.items {
            match f.kind {
                FieldKind::Bubble { radius, time_rate, .. } if (p - f.pos).length_squared() < radius * radius => k *= time_rate,
                FieldKind::MicroHole { mass, .. } => {
                    let r = (p - f.pos).length().max(HOLE_SOFT);
                    k *= (1.0 - schwarzschild_radius(mass) / r).max(0.0).sqrt();
                }
                _ => {}
            }
        }
        k
    }

    /// Radiant flux from the holes at `p` (W/m²).
    pub fn radiation(&self, p: DVec3) -> f64 {
        let mut q = 0.0;
        for f in &self.items {
            if let FieldKind::MicroHole { mass, contained } = f.kind {
                let lum = hawking_power(mass) * if contained { CONTAINMENT_LEAK } else { 1.0 };
                let r2 = (p - f.pos).length_squared().max(0.01);
                q += lum / (4.0 * std::f64::consts::PI * r2);
            }
        }
        q
    }

    /// First boundary crossed along p0 → p1 (fraction t, what).
    pub fn crossing(&self, p0: DVec3, p1: DVec3) -> Option<(f64, Crossing)> {
        let mut best: Option<(f64, Crossing)> = None;
        for (i, f) in self.items.iter().enumerate() {
            match f.kind {
                FieldKind::Bubble { radius, barrier, .. } => {
                    if let Some(t) = sphere_cross(p0, p1, f.pos, radius, true) {
                        if best.is_none_or(|b| t < b.0) {
                            let q = p0 + (p1 - p0) * t;
                            best = Some((t, Crossing::Shell { normal: (q - f.pos).normalize_or(DVec3::Z), barrier }));
                        }
                    }
                }
                FieldKind::Fold { exit, radius, stability } => {
                    for (a, b) in [(f.pos, exit), (exit, f.pos)] {
                        if let Some(t) = sphere_cross(p0, p1, a, radius, false) {
                            if best.is_none_or(|bb| t < bb.0) {
                                let q = p0 + (p1 - p0) * t;
                                // antipodal point of the other mouth, nudged out of it
                                let out = b - (q - a) * 1.02;
                                best = Some((t, Crossing::Fold { shift: out - q, field: i, stability }));
                            }
                        }
                    }
                }
                FieldKind::MicroHole { .. } => {}
            }
        }
        best
    }
}

/// Fraction t where the segment crosses the sphere surface. `both` = either
/// direction, otherwise only entering (outside → inside).
fn sphere_cross(p0: DVec3, p1: DVec3, c: DVec3, r: f64, both: bool) -> Option<f64> {
    let d = p1 - p0;
    let m = p0 - c;
    let a = d.length_squared();
    if a < 1e-18 {
        return None;
    }
    let b = m.dot(d);
    let cc = m.length_squared() - r * r;
    let disc = b * b - a * cc;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t_in = (-b - sq) / a;
    let t_out = (-b + sq) / a;
    if cc > 0.0 {
        // outside: entering at t_in
        (t_in > 0.0 && t_in <= 1.0).then_some(t_in)
    } else if both {
        (t_out > 0.0 && t_out <= 1.0).then_some(t_out)
    } else {
        None
    }
}

/// Velocity after meeting a thin barrier: reflection of the radial part if
/// ½v_r² < U, otherwise unchanged (energy conserved either way).
pub fn shell_response(v: DVec3, normal: DVec3, barrier: f64) -> (DVec3, bool) {
    let vr = v.dot(normal);
    if 0.5 * vr * vr < barrier {
        (v - 2.0 * vr * normal, true)
    } else {
        (v, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hawking_numbers() {
        // textbook: M = 1e11 kg → T ≈ 1.2e12 K, P ≈ 3.6e10 W, τ ≈ 8.4e16 s
        let m = 1.0e11;
        assert!((hawking_temperature(m) / 1.227e12 - 1.0).abs() < 0.01);
        assert!((hawking_power(m) / 3.56e10 - 1.0).abs() < 0.01);
        assert!((hawking_lifetime(m) / 8.41e16 - 1.0).abs() < 0.01);
    }

    #[test]
    fn hole_gravity_is_newtonian_at_metres() {
        let fs = FieldSet {
            items: vec![Field {
                kind: FieldKind::MicroHole { mass: 3.0e11, contained: true },
                pos: DVec3::ZERO,
                vel: DVec3::ZERO,
                owner: Entity::default(),
                age: 0.0,
                ttl: 1.0,
                heat: 0.0,
            }],
        };
        let a = fs.accel(DVec3::new(2.0, 0.0, 0.0));
        assert!((a.x + G_NEWTON * 3.0e11 / 4.0).abs() < 1e-9);
        // tidal stretch on a body 1 m from the hole: 2GM/r³·len
        let s = fs.stretch(DVec3::new(0.0, 0.0, 1.5), DVec3::Z, 1.0);
        let expect = G_NEWTON * 3.0e11 * (1.0 / 1.0 - 1.0 / 4.0); // a(1) − a(2)
        assert!((s - expect).abs() / expect < 1e-9);
    }

    #[test]
    fn barrier_reflects_slow_and_passes_fast() {
        let n = DVec3::X;
        let (v, r) = shell_response(DVec3::new(-360.0, 5.0, 0.0), n, 1.8e5);
        assert!(r && v.x > 0.0 && (v.length() - DVec3::new(360.0, 5.0, 0.0).length()).abs() < 1e-9);
        let (v, r) = shell_response(DVec3::new(-940.0, 0.0, 0.0), n, 1.8e5);
        assert!(!r && v.x == -940.0);
    }
}
