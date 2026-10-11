//! Procedural HDR sky cubemap: ~200k stars (power-law magnitudes, black-body
//! colours, concentrated towards the galactic plane), a Milky Way band with a
//! bulge, ridged dust lanes and Hα nebulae. Baked offline by `tools/gen_sky.rs`
//! into `assets/sky_<N>.bin`; the runtime loads it (or bakes it on first run).
//!
//! Face order and orientation follow the Vulkan/wgpu cube convention
//! (+X, −X, +Y, −Y, +Z, −Z). Format "SKY1": N, mip count, then RGBA16F texels of
//! every mip level, faces in order, rows top to bottom.
#![allow(dead_code)]

use super::blackbody;
use super::noise::{f16_bits, fbm, hash_u32, ridged, Rng};
use glam::DVec3;

pub struct Sky {
    pub size: u32,
    /// mips[level][face] = RGBA16F bits, row-major
    pub mips: Vec<[Vec<u16>; 6]>,
}

pub fn face_dir(face: usize, s: f64, t: f64) -> DVec3 {
    match face {
        0 => DVec3::new(1.0, -t, -s),
        1 => DVec3::new(-1.0, -t, s),
        2 => DVec3::new(s, 1.0, t),
        3 => DVec3::new(s, -1.0, -t),
        4 => DVec3::new(s, -t, 1.0),
        _ => DVec3::new(-s, -t, -1.0),
    }
    .normalize()
}

/// Inverse of `face_dir`: (face, s, t) with s, t in [-1, 1].
pub fn dir_face(d: DVec3) -> (usize, f64, f64) {
    let a = d.abs();
    if a.x >= a.y && a.x >= a.z {
        if d.x > 0.0 {
            (0, -d.z / a.x, -d.y / a.x)
        } else {
            (1, d.z / a.x, -d.y / a.x)
        }
    } else if a.y >= a.z {
        if d.y > 0.0 {
            (2, d.x / a.y, d.z / a.y)
        } else {
            (3, d.x / a.y, -d.z / a.y)
        }
    } else if d.z > 0.0 {
        (4, d.x / a.z, -d.y / a.z)
    } else {
        (5, -d.x / a.z, -d.y / a.z)
    }
}

/// Galactic frame: the Milky Way plane is tilted 62° against the accretion disk
/// so that the band crosses the frame diagonally.
struct Galaxy {
    n: DVec3,
    e1: DVec3,
    e2: DVec3,
}

impl Galaxy {
    fn new() -> Self {
        let n = DVec3::new(0.35, -0.52, 0.78).normalize();
        let e1 = DVec3::new(0.8, 0.6, 0.0).reject_from(n).normalize();
        let e2 = n.cross(e1);
        Galaxy { n, e1, e2 }
    }
    /// (longitude, latitude) in radians
    fn lb(&self, d: DVec3) -> (f64, f64) {
        (d.dot(self.e2).atan2(d.dot(self.e1)), d.dot(self.n).clamp(-1.0, 1.0).asin())
    }
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Diffuse radiance (linear RGB) of the Milky Way and nebulae in direction d.
fn diffuse(g: &Galaxy, d: DVec3, seed: u32) -> [f64; 3] {
    let (l, b) = g.lb(d);
    let ab = b.abs();
    // faint uniform airglow-free background (zodiacal/extragalactic)
    let mut c = [0.0012, 0.0013, 0.0018];
    if ab > 0.7 {
        return c;
    }
    let p = d * 3.0;
    let thin = (-(b / 0.085).powi(2)).exp();
    let thick = (-(b / 0.26).powi(2)).exp();
    // bulge around l = 0
    let bulge = (-(l * l / 0.30 + b * b / 0.045)).exp();
    let lon = 0.55 + 0.45 * (0.5 + 0.5 * l.cos()).powf(1.5);
    let clump = 0.55 + 0.45 * fbm(p * 2.2, 5, seed);
    let mut band = (thin * 0.9 + thick * 0.28) * lon * clump.max(0.0) + bulge * 1.6;
    // dust lanes: ridged filaments, strongest in the mid-plane
    let lanes = ridged(p * 1.7 + DVec3::splat(11.0), 5, seed ^ 0xd57) * (0.6 + 0.4 * fbm(p * 5.0, 3, seed ^ 3));
    let dust = smooth(0.35, 0.8, lanes) * (-(b / 0.06).powi(2)).exp() * 0.92;
    band *= 1.0 - dust;
    // colour: warm bulge, bluish-white disk, reddened by dust
    let warm = (bulge * 2.0).min(1.0);
    let base = [0.78 + 0.22 * warm, 0.83 + 0.02 * warm, 1.0 - 0.38 * warm];
    let redden = 1.0 - 0.45 * dust;
    let k = 0.030;
    c[0] += k * band * base[0];
    c[1] += k * band * base[1] * redden;
    c[2] += k * band * base[2] * redden * redden;
    // H-alpha emission nebulae near the plane, a few reflection nebulae
    let neb = smooth(0.62, 0.9, ridged(p * 2.6 + DVec3::splat(4.0), 4, seed ^ 0x51)) * smooth(0.15, 0.6, fbm(p * 0.9, 3, seed ^ 0x77)) * thick;
    c[0] += 0.020 * neb;
    c[1] += 0.0035 * neb;
    c[2] += 0.006 * neb;
    let refl = smooth(0.7, 0.95, ridged(p * 3.4 + DVec3::splat(-7.0), 3, seed ^ 0x99)) * thin * (1.0 - dust);
    c[0] += 0.002 * refl;
    c[1] += 0.004 * refl;
    c[2] += 0.009 * refl;
    c
}

pub fn generate(size: u32, seed: u64) -> Sky {
    let n = size as usize;
    let g = Galaxy::new();
    let s32 = (seed as u32) ^ 0x5eed;
    // 1) diffuse, in parallel per face
    let mut faces: Vec<Vec<[f32; 3]>> = Vec::new();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..6)
            .map(|f| {
                let g = &g;
                sc.spawn(move || {
                    let mut px = vec![[0f32; 3]; n * n];
                    for y in 0..n {
                        for x in 0..n {
                            let s = 2.0 * (x as f64 + 0.5) / n as f64 - 1.0;
                            let t = 2.0 * (y as f64 + 0.5) / n as f64 - 1.0;
                            let d = face_dir(f, s, t);
                            let c = diffuse(g, d, s32);
                            // unresolved faint stars: per-texel sparkle, denser in the band
                            let (_, b) = g.lb(d);
                            let h = hash_u32((f * n * n + y * n + x) as u32 ^ s32);
                            let dens = 0.004 + 0.03 * (-(b / 0.2).powi(2)).exp();
                            let mut sp = 0.0;
                            if (h & 0xffff) as f64 / 65536.0 < dens * (1024.0 / n as f64).powi(2) {
                                sp = 0.004 * (1.0 + 6.0 * ((h >> 16) as f64 / 65536.0).powi(3));
                            }
                            px[y * n + x] = [(c[0] + sp) as f32, (c[1] + sp * 0.97) as f32, (c[2] + sp * 1.02) as f32];
                        }
                    }
                    px
                })
            })
            .collect();
        faces = hs.into_iter().map(|h| h.join().expect("sky thread")).collect();
    });
    // 2) resolved stars, splatted with a Gaussian footprint (σ ≈ 0.55 texel)
    let mut rng = Rng(seed | 1);
    let nstars = 220_000;
    let texel = std::f64::consts::FRAC_PI_2 / n as f64;
    for i in 0..nstars {
        let d = if i % 10 < 6 {
            // disk population: concentrated towards the galactic plane
            let l = std::f64::consts::TAU * rng.f64();
            let u1 = rng.f64().max(1e-12);
            let u2 = rng.f64();
            let b = 0.17 * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            (g.e1 * l.cos() * b.cos() + g.e2 * l.sin() * b.cos() + g.n * b.sin()).normalize()
        } else {
            rng.unit_vec()
        };
        // magnitude: N(<m) ∝ 10^(0.42 m) for m in [-1.5, 11]
        let u = rng.f64();
        let a = 0.42 * std::f64::consts::LN_10;
        let (m0, m1) = (-1.5f64, 11.0f64);
        let m = ((a * m0).exp() + u * ((a * m1).exp() - (a * m0).exp())).ln() / a;
        let flux = 6.0 * 10f64.powf(-0.4 * m) * (texel / (std::f64::consts::FRAC_PI_2 / 1024.0)).powi(-2).min(4.0);
        // temperature: mostly cool dwarfs/giants, a few hot blue stars
        let tr = rng.f64();
        let temp = if tr < 0.12 {
            9000.0 + 21000.0 * rng.f64()
        } else if tr < 0.45 {
            5500.0 + 3500.0 * rng.f64()
        } else {
            3000.0 + 2600.0 * rng.f64()
        };
        let col = blackbody::rgb(temp);
        let (f, s, t) = dir_face(d);
        let fx = (s + 1.0) * 0.5 * n as f64 - 0.5;
        let fy = (t + 1.0) * 0.5 * n as f64 - 0.5;
        let (cx, cy) = (fx.round() as i64, fy.round() as i64);
        let mut wsum = 0.0;
        let mut taps = [(0usize, 0.0f64); 9];
        let mut k = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                    continue;
                }
                let r2 = (x as f64 - fx).powi(2) + (y as f64 - fy).powi(2);
                let w = (-r2 / (2.0 * 0.55 * 0.55)).exp();
                taps[k] = (y as usize * n + x as usize, w);
                wsum += w;
                k += 1;
            }
        }
        for &(idx, w) in &taps[..k] {
            let e = flux * w / wsum.max(1e-9);
            let p = &mut faces[f][idx];
            p[0] += (e * col[0]) as f32;
            p[1] += (e * col[1]) as f32;
            p[2] += (e * col[2]) as f32;
        }
    }
    // 3) mip chain (2×2 box) and half-float packing
    let mut mips = Vec::new();
    let mut cur = faces;
    let mut sz = n;
    loop {
        let packed: [Vec<u16>; 6] = std::array::from_fn(|f| {
            let mut v = Vec::with_capacity(sz * sz * 4);
            for p in &cur[f] {
                v.extend_from_slice(&[f16_bits(p[0]), f16_bits(p[1]), f16_bits(p[2]), f16_bits(1.0)]);
            }
            v
        });
        mips.push(packed);
        if sz == 1 {
            break;
        }
        let ns = sz / 2;
        cur = cur
            .iter()
            .map(|src| {
                let mut dst = vec![[0f32; 3]; ns * ns];
                for y in 0..ns {
                    for x in 0..ns {
                        let mut a = [0f32; 3];
                        for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let p = src[(2 * y + oy) * sz + 2 * x + ox];
                            a[0] += p[0];
                            a[1] += p[1];
                            a[2] += p[2];
                        }
                        dst[y * ns + x] = [a[0] * 0.25, a[1] * 0.25, a[2] * 0.25];
                    }
                }
                dst
            })
            .collect();
        sz = ns;
    }
    Sky { size, mips }
}

pub const DEFAULT_SEED: u64 = 0x5_7A85;

pub fn save(s: &Sky) -> Vec<u8> {
    let mut b = b"SKY1".to_vec();
    b.extend_from_slice(&s.size.to_le_bytes());
    b.extend_from_slice(&(s.mips.len() as u32).to_le_bytes());
    for m in &s.mips {
        for f in m.iter() {
            for &h in f {
                b.extend_from_slice(&h.to_le_bytes());
            }
        }
    }
    b
}

pub fn load(b: &[u8]) -> Option<Sky> {
    if b.len() < 12 || &b[0..4] != b"SKY1" {
        return None;
    }
    let size = u32::from_le_bytes(b[4..8].try_into().ok()?);
    let nm = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
    let mut o = 12;
    let mut mips = Vec::new();
    let mut sz = size as usize;
    for _ in 0..nm {
        let len = sz * sz * 4;
        let mut faces: [Vec<u16>; 6] = Default::default();
        for f in faces.iter_mut() {
            let bytes = b.get(o..o + len * 2)?;
            *f = bytes.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
            o += len * 2;
        }
        mips.push(faces);
        sz = (sz / 2).max(1);
    }
    Some(Sky { size, mips })
}

pub fn path_for(size: u32) -> String {
    format!("assets/sky_{size}.bin")
}

pub fn load_or_generate(size: u32) -> Sky {
    let path = path_for(size);
    if let Some(s) = std::fs::read(&path).ok().and_then(|b| load(&b)) {
        if s.size == size {
            return s;
        }
    }
    log::info!("baking sky cubemap {size}² (first run) …");
    let s = generate(size, DEFAULT_SEED);
    let _ = std::fs::create_dir_all("assets");
    let _ = std::fs::write(&path, save(&s));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn face_mapping_roundtrip() {
        for f in 0..6 {
            for (s, t) in [(0.3, -0.7), (-0.9, 0.2), (0.0, 0.0)] {
                let d = face_dir(f, s, t);
                let (f2, s2, t2) = dir_face(d);
                assert_eq!(f, f2);
                assert!((s - s2).abs() < 1e-9 && (t - t2).abs() < 1e-9);
            }
        }
    }
    #[test]
    fn small_sky_roundtrip() {
        let s = generate(16, 3);
        assert_eq!(s.mips.len(), 5);
        let b = load(&save(&s)).unwrap();
        assert_eq!(b.mips[0][3], s.mips[0][3]);
    }
}
