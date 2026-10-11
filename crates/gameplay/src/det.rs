//! Deterministic numerics.
//!
//! Gameplay state must evolve bit-identically on every peer (rollback /
//! client-server later). IEEE-754 `+ − × ÷ sqrt` are exactly rounded, but the
//! platform `sin`, `cos`, `exp`, … are not specified bit-exactly, so all
//! transcendental functions used by the simulation go through `libm` (a pure
//! Rust port of musl: same bits on every target). Never call `f64::sin` & co.
//! inside the tick.

use glam::{DMat3, DQuat, DVec3};

#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}
#[inline]
pub fn sin_cos(x: f64) -> (f64, f64) {
    (libm::sin(x), libm::cos(x))
}
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}
#[inline]
pub fn acos(x: f64) -> f64 {
    libm::acos(x.clamp(-1.0, 1.0))
}
#[inline]
pub fn asin(x: f64) -> f64 {
    libm::asin(x.clamp(-1.0, 1.0))
}
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
#[inline]
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// Rotation by `angle` about the unit `axis` (deterministic replacement for `DQuat::from_axis_angle`).
pub fn quat_axis_angle(axis: DVec3, angle: f64) -> DQuat {
    let (s, c) = sin_cos(0.5 * angle);
    let a = axis * s;
    DQuat::from_xyzw(a.x, a.y, a.z, c)
}

/// Rotation by the vector `w` (axis · angle).
pub fn quat_scaled_axis(w: DVec3) -> DQuat {
    let a = w.length();
    if a < 1e-12 {
        return DQuat::IDENTITY;
    }
    quat_axis_angle(w / a, a)
}

/// Shortest rotation taking unit `a` to unit `b`.
pub fn quat_arc(a: DVec3, b: DVec3) -> DQuat {
    let d = a.dot(b);
    if d < -0.999_999 {
        let axis = a.any_orthonormal_vector();
        return DQuat::from_xyzw(axis.x, axis.y, axis.z, 0.0);
    }
    let c = a.cross(b);
    DQuat::from_xyzw(c.x, c.y, c.z, 1.0 + d).normalize()
}

pub fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// SplitMix64 — tiny, fast, statistically good and trivially serialisable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng(pub u64);

impl Rng {
    /// Stream keyed by several integers (seed, tick, entity, salt …): every
    /// random decision gets its own stream, so the order of unrelated draws
    /// never changes a result (important for rollback re-simulation).
    pub fn keyed(keys: &[u64]) -> Rng {
        let mut h = 0x243F_6A88_85A3_08D3u64;
        for &k in keys {
            h = mix(h ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        }
        Rng(h)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }
    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f64()
    }
    /// Approximately standard normal (Irwin–Hall, 4 samples): bounded tails, deterministic.
    pub fn normal(&mut self) -> f64 {
        (self.f64() + self.f64() + self.f64() + self.f64() - 2.0) * 1.732_050_807_568_877_2
    }
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a over the exact bit patterns of the state.
#[derive(Clone, Copy, Debug)]
pub struct StateHasher(pub u64);

impl Default for StateHasher {
    fn default() -> Self {
        StateHasher(0xcbf2_9ce4_8422_2325)
    }
}

impl StateHasher {
    pub fn u64(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    pub fn f64(&mut self, v: f64) {
        // −0 and +0 compare equal; normalise so they hash equal too
        self.u64(if v == 0.0 { 0 } else { v.to_bits() });
    }
    pub fn f32(&mut self, v: f32) {
        self.f64(v as f64);
    }
    pub fn v3(&mut self, v: DVec3) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }
    pub fn quat(&mut self, q: DQuat) {
        self.f64(q.x);
        self.f64(q.y);
        self.f64(q.z);
        self.f64(q.w);
    }
    pub fn bool(&mut self, b: bool) {
        self.u64(b as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_streams_are_reproducible_and_independent() {
        let a: Vec<u64> = (0..4).map(|i| Rng::keyed(&[1, 2, i]).next_u64()).collect();
        let b: Vec<u64> = (0..4).map(|i| Rng::keyed(&[1, 2, i]).next_u64()).collect();
        assert_eq!(a, b);
        assert_ne!(a[0], a[1]);
        let mut r = Rng(5);
        let m: f64 = (0..20000).map(|_| r.normal()).sum::<f64>() / 20000.0;
        assert!(m.abs() < 0.03);
    }

    #[test]
    fn quaternion_helpers() {
        let q = quat_axis_angle(DVec3::Z, std::f64::consts::FRAC_PI_2);
        assert!((q * DVec3::X - DVec3::Y).length() < 1e-12);
        let q = quat_arc(DVec3::X, DVec3::Y);
        assert!((q * DVec3::X - DVec3::Y).length() < 1e-12);
    }
}
