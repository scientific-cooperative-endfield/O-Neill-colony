//! Black-body colour LUT: Planck spectrum × CIE 1931 colour matching functions
//! (multi-lobe Gaussian fit of Wyman, Sloan & Shirley 2013) → XYZ → linear sRGB,
//! normalised to luminance Y = 1. Intensity is applied separately (∝ T⁴, bolometric).
#![allow(dead_code)]

pub const LUT_SIZE: usize = 256;
pub const T_MIN: f64 = 500.0;
pub const T_MAX: f64 = 60_000.0;

fn lobe(l: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let s = if l < mu { s1 } else { s2 };
    let t = (l - mu) / s;
    (-0.5 * t * t).exp()
}

/// CIE 1931 2° colour matching functions, λ in nm.
pub fn cmf(l: f64) -> [f64; 3] {
    let x = 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2);
    let y = 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
    let z = 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8);
    [x, y, z]
}

/// Planck spectral radiance (arbitrary scale), λ in nm.
fn planck(l: f64, t: f64) -> f64 {
    let lm = l * 1e-9;
    let c2 = 1.438_776_9e-2; // hc/k, m·K
    1.0 / (lm.powi(5) * ((c2 / (lm * t)).exp() - 1.0))
}

/// Linear sRGB colour of a black body at temperature `t` (K), with Y = 1.
pub fn rgb(t: f64) -> [f64; 3] {
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    let mut l = 380.0;
    while l <= 780.0 {
        let p = planck(l, t);
        let c = cmf(l);
        x += p * c[0];
        y += p * c[1];
        z += p * c[2];
        l += 2.0;
    }
    let (x, z) = (x / y, z / y);
    let r = 3.2406 * x - 1.5372 - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 + 1.0570 * z;
    // out-of-gamut (deep red) components are clipped, then Y is restored
    let (r, g, b) = (r.max(0.0), g.max(0.0), b.max(0.0));
    let yy = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    [r / yy, g / yy, b / yy]
}

pub fn lut_temperature(i: usize) -> f64 {
    let f = i as f64 / (LUT_SIZE - 1) as f64;
    (T_MIN.ln() + f * (T_MAX.ln() - T_MIN.ln())).exp()
}

/// RGBA32F texels; alpha = temperature (K) for debugging.
pub fn lut() -> Vec<[f32; 4]> {
    (0..LUT_SIZE)
        .map(|i| {
            let t = lut_temperature(i);
            let c = rgb(t);
            [c[0] as f32, c[1] as f32, c[2] as f32, t as f32]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn white_point_and_trend() {
        // ~6500 K is close to the sRGB white point (D65)
        let c = rgb(6504.0);
        assert!((c[0] - c[2]).abs() < 0.15 && (c[1] - 1.0).abs() < 0.1, "{c:?}");
        // cooler → redder, hotter → bluer
        let lo = rgb(2000.0);
        let hi = rgb(20_000.0);
        assert!(lo[0] > lo[2] * 3.0);
        assert!(hi[2] > hi[0]);
    }
}
