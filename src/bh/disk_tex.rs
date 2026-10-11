//! Baked turbulence texture of the accretion disk in (φ, log r) coordinates.
//! u = φ / 2π (periodic), v = ln(r/r_in) / ln(r_out/r_in). The shader rotates
//! it with the local Keplerian angular velocity Ω = √(M/r³), so differential
//! rotation shears the pattern into trailing spirals over time.
//! R = density (opacity/emission modulation), G = temperature perturbation.

use super::noise::fbm;
use glam::DVec3;

pub const W: usize = 2048;
pub const H: usize = 256;

pub struct DiskTex {
    /// mips[level] = RGBA8 texels (w, h, data)
    pub mips: Vec<(usize, usize, Vec<u8>)>,
}

pub fn generate(seed: u32) -> DiskTex {
    let mut base = vec![0u8; W * H * 4];
    let rows: Vec<Vec<u8>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..2)
            .map(|part| {
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for y in (part * H / 2)..((part + 1) * H / 2) {
                        let v = (y as f64 + 0.5) / H as f64;
                        let mut row = vec![0u8; W * 4];
                        for x in 0..W {
                            let a = std::f64::consts::TAU * (x as f64 + 0.5) / W as f64;
                            let (c, s) = (a.cos(), a.sin());
                            // anisotropic: long streaks along φ, fine structure in r
                            let p = |rad: f64, kv: f64| DVec3::new(c * rad, s * rad, v * kv);
                            let streak = fbm(p(3.0, 46.0), 6, seed);
                            let fine = fbm(p(9.0, 140.0), 4, seed ^ 0x33);
                            let clumps = fbm(p(1.6, 9.0), 4, seed ^ 0x71);
                            // ring structure with noisy phase (gaps and lanes)
                            let ring = (v * 61.0 + 2.2 * fbm(p(1.2, 6.0), 3, seed ^ 0x9)).sin();
                            let mut d = 0.55 + 0.55 * streak + 0.22 * fine + 0.35 * clumps + 0.18 * ring;
                            d = (d * 0.85).clamp(0.0, 1.0);
                            d = d * d * (3.0 - 2.0 * d);
                            let tv = (0.5 + 0.5 * fbm(p(2.2, 20.0), 4, seed ^ 0x5151)).clamp(0.0, 1.0);
                            let i = x * 4;
                            row[i] = (d * 255.0).round() as u8;
                            row[i + 1] = (tv * 255.0).round() as u8;
                            row[i + 2] = 0;
                            row[i + 3] = 255;
                        }
                        out.push(row);
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().expect("disk thread")).collect()
    });
    for (y, r) in rows.iter().enumerate() {
        base[y * W * 4..(y + 1) * W * 4].copy_from_slice(r);
    }
    let mut mips = vec![(W, H, base)];
    loop {
        let (w, h, src) = mips.last().expect("mip");
        if *w == 1 && *h == 1 {
            break;
        }
        let (nw, nh) = ((*w / 2).max(1), (*h / 2).max(1));
        let mut dst = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                for ch in 0..4 {
                    let mut s = 0u32;
                    for (ox, oy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (2 * x + ox).min(*w - 1);
                        let sy = (2 * y + oy).min(*h - 1);
                        s += src[(sy * w + sx) * 4 + ch] as u32;
                    }
                    dst[(y * nw + x) * 4 + ch] = ((s + 2) / 4) as u8;
                }
            }
        }
        mips.push((nw, nh, dst));
    }
    DiskTex { mips }
}
