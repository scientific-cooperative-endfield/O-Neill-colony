//! Deterministic hash-based value/gradient noise for the offline bakers (sky, disk).
#![allow(dead_code)]

use glam::DVec3;

pub fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn hash3(i: i32, j: i32, k: i32, seed: u32) -> u32 {
    let mut h = hash_u32(i as u32 ^ seed.wrapping_mul(0x9e37_79b9));
    h = hash_u32(h ^ (j as u32).wrapping_mul(0x85eb_ca6b));
    hash_u32(h ^ (k as u32).wrapping_mul(0xc2b2_ae35))
}

fn grad(h: u32, d: DVec3) -> f64 {
    // 12 edge directions of a cube
    match h % 12 {
        0 => d.x + d.y,
        1 => -d.x + d.y,
        2 => d.x - d.y,
        3 => -d.x - d.y,
        4 => d.x + d.z,
        5 => -d.x + d.z,
        6 => d.x - d.z,
        7 => -d.x - d.z,
        8 => d.y + d.z,
        9 => -d.y + d.z,
        10 => d.y - d.z,
        _ => -d.y - d.z,
    }
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Gradient noise in [-1, 1] (roughly).
pub fn perlin(p: DVec3, seed: u32) -> f64 {
    let i = p.floor();
    let f = p - i;
    let (ix, iy, iz) = (i.x as i32, i.y as i32, i.z as i32);
    let u = DVec3::new(fade(f.x), fade(f.y), fade(f.z));
    let mut v = [0.0; 8];
    for (n, vv) in v.iter_mut().enumerate() {
        let (a, b, c) = ((n & 1) as i32, ((n >> 1) & 1) as i32, ((n >> 2) & 1) as i32);
        let h = hash3(ix + a, iy + b, iz + c, seed);
        *vv = grad(h, f - DVec3::new(a as f64, b as f64, c as f64));
    }
    let lx = |a: f64, b: f64| a + (b - a) * u.x;
    let x00 = lx(v[0], v[1]);
    let x10 = lx(v[2], v[3]);
    let x01 = lx(v[4], v[5]);
    let x11 = lx(v[6], v[7]);
    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}

pub fn fbm(p: DVec3, octaves: u32, seed: u32) -> f64 {
    let (mut s, mut a, mut f, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        s += a * perlin(p * f, seed.wrapping_add(o * 1013));
        norm += a;
        a *= 0.5;
        f *= 2.03;
    }
    s / norm
}

/// Ridged multifractal in [0, 1] — filaments for dust lanes and nebulae.
pub fn ridged(p: DVec3, octaves: u32, seed: u32) -> f64 {
    let (mut s, mut a, mut f, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for o in 0..octaves {
        let n = 1.0 - perlin(p * f, seed.wrapping_add(o * 7919)).abs();
        s += a * n * n;
        norm += a;
        a *= 0.5;
        f *= 2.1;
    }
    s / norm
}

/// xorshift64* PRNG (deterministic, seedable).
#[derive(Clone, Debug)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn unit_vec(&mut self) -> DVec3 {
        let z = 2.0 * self.f64() - 1.0;
        let a = std::f64::consts::TAU * self.f64();
        let r = (1.0 - z * z).max(0.0).sqrt();
        DVec3::new(r * a.cos(), r * a.sin(), z)
    }
}

/// f32 → IEEE half bits (round to nearest, clamps to max finite).
pub fn f16_bits(v: f32) -> u16 {
    let v = v.clamp(-65000.0, 65000.0);
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0 {
        return sign;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7bff;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let h = ((e as u32) << 10) | (man >> 13);
    let round = (man >> 12) & 1;
    sign | (h + round) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_roundtrip() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(0.0), 0);
        assert!(f16_bits(1e9) < 0x7c00, "clamped below infinity");
        assert_eq!(f16_bits(0.5), 0x3800);
    }
    #[test]
    fn noise_is_deterministic() {
        let p = DVec3::new(1.3, -2.7, 0.4);
        assert_eq!(fbm(p, 5, 3).to_bits(), fbm(p, 5, 3).to_bits());
        assert!(fbm(p, 5, 3).abs() <= 1.5);
    }
}
