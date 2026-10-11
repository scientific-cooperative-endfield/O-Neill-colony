//! Procedural meshes in unit space (x, z in [-0.5, 0.5], y in [0, 1]); the
//! instance size scales them. Winding is fixed automatically so every
//! triangle faces along its vertex normals (safe back-face culling).

use std::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub part: f32,
}

#[derive(Default)]
pub struct Mesh {
    pub verts: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    fn v(&mut self, p: Vec3, n: Vec3, part: f32) -> u32 {
        self.verts.push(Vertex { pos: p.to_array(), normal: n.normalize_or_zero().to_array(), part });
        (self.verts.len() - 1) as u32
    }

    /// Flat polygon (convex, ordered) whose normal points away from `centre`.
    fn face(&mut self, pts: &[Vec3], centre: Vec3, part: f32) {
        let c = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
        let mut n = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        if n.length_squared() < 1e-12 {
            n = (pts[2] - pts[0]).cross(pts[3 % pts.len()] - pts[0]);
        }
        if n.dot(c - centre) < 0.0 {
            n = -n;
        }
        let base = self.verts.len() as u32;
        for &p in pts {
            self.v(p, n, part);
        }
        for i in 1..pts.len() as u32 - 1 {
            self.indices.extend_from_slice(&[base, base + i, base + i + 1]);
        }
    }

    pub fn add_box(&mut self, mn: Vec3, mx: Vec3, part: f32) {
        let c = (mn + mx) * 0.5;
        let p = |x: f32, y: f32, z: f32| Vec3::new(if x > 0.0 { mx.x } else { mn.x }, if y > 0.0 { mx.y } else { mn.y }, if z > 0.0 { mx.z } else { mn.z });
        for (a, b, cc, d) in [
            ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
            ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
            ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
            ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
            ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
            ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
        ] {
            let q = |t: (i32, i32, i32)| p(t.0 as f32, t.1 as f32, t.2 as f32);
            self.face(&[q(a), q(b), q(cc), q(d)], c, part);
        }
    }

    /// Surface of revolution around +Y. `profile` = (radius, y) from bottom to
    /// top; smooth normals along the profile.
    pub fn lathe(&mut self, profile: &[(f32, f32)], sides: usize, centre: Vec3, part: f32, wobble: f32, seed: u32) {
        let n = profile.len();
        let base = self.verts.len() as u32;
        for (i, &(r, y)) in profile.iter().enumerate() {
            let prev = profile[i.saturating_sub(1)];
            let next = profile[(i + 1).min(n - 1)];
            let (dr, dy) = (next.0 - prev.0, next.1 - prev.1);
            for j in 0..=sides {
                let a = j as f32 / sides as f32 * TAU;
                let jj = (j % sides) as u32;
                let w = if wobble > 0.0 && r > 0.0 {
                    let h = (jj.wrapping_mul(2654435761) ^ (i as u32).wrapping_mul(40503) ^ seed).wrapping_mul(2246822519);
                    1.0 + wobble * (((h >> 8) & 0xffff) as f32 / 65535.0 - 0.5)
                } else {
                    1.0
                };
                let (s, c) = a.sin_cos();
                let p = centre + Vec3::new(r * w * c, y, r * w * s);
                let nrm = Vec3::new(dy * c, -dr, dy * s);
                self.v(p, nrm, part);
            }
        }
        let row = (sides + 1) as u32;
        for i in 0..n as u32 - 1 {
            for j in 0..sides as u32 {
                let a = base + i * row + j;
                let b = a + 1;
                let c = a + row;
                let d = c + 1;
                self.indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sphere(&mut self, centre: Vec3, r: Vec3, sides: usize, rings: usize, part: f32, wobble: f32, seed: u32) {
        let base = self.verts.len();
        let prof: Vec<(f32, f32)> = (0..=rings)
            .map(|i| {
                let phi = i as f32 / rings as f32 * std::f32::consts::PI;
                (phi.sin(), -phi.cos())
            })
            .collect();
        self.lathe(&prof, sides, Vec3::ZERO, part, wobble, seed);
        for v in &mut self.verts[base..] {
            let p = Vec3::from(v.pos);
            let n = if p.length() < 1e-5 { Vec3::Y * p.y.signum() } else { p / r };
            v.normal = n.normalize_or(Vec3::Y).to_array();
            v.pos = (centre + p * r).to_array();
        }
    }

    /// Tapered open tube from `a` to `b`.
    pub fn tube(&mut self, a: Vec3, b: Vec3, r0: f32, r1: f32, sides: usize, part: f32) {
        let axis = (b - a).normalize_or(Vec3::Y);
        let side = if axis.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let u = axis.cross(side).normalize();
        let w = axis.cross(u);
        let base = self.verts.len() as u32;
        for (p, r) in [(a, r0), (b, r1)] {
            for j in 0..=sides {
                let ang = j as f32 / sides as f32 * TAU;
                let d = u * ang.cos() + w * ang.sin();
                self.v(p + d * r, d, part);
            }
        }
        let row = (sides + 1) as u32;
        for j in 0..sides as u32 {
            let (i0, i1) = (base + j, base + j + 1);
            self.indices.extend_from_slice(&[i0, i0 + row, i1, i1, i0 + row, i1 + row]);
        }
    }

    /// Make every triangle wind counter-clockwise when seen from the side
    /// its vertex normals point to.
    pub fn fix_winding(&mut self) {
        for t in self.indices.chunks_mut(3) {
            let p = |i: u32| Vec3::from(self.verts[i as usize].pos);
            let ng = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
            let nv: Vec3 = t.iter().map(|&i| Vec3::from(self.verts[i as usize].normal)).sum();
            if ng.dot(nv) < 0.0 {
                t.swap(1, 2);
            }
        }
    }
}

pub fn unit_box() -> Mesh {
    let mut m = Mesh::default();
    m.add_box(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5), 0.0);
    m
}

pub fn tower() -> Mesh {
    let mut m = Mesh::default();
    m.add_box(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 0.86, 0.5), 0.0);
    m.add_box(Vec3::new(-0.36, 0.86, -0.36), Vec3::new(0.36, 0.93, 0.36), 0.0);
    let tip = Vec3::new(0.0, 1.0, 0.0);
    let b = [Vec3::new(-0.06, 0.93, -0.06), Vec3::new(0.06, 0.93, -0.06), Vec3::new(0.06, 0.93, 0.06), Vec3::new(-0.06, 0.93, 0.06)];
    let c = Vec3::new(0.0, 0.95, 0.0);
    for i in 0..4 {
        m.face(&[b[i], b[(i + 1) % 4], tip], c, 0.0);
    }
    m
}

pub fn dome(sides: usize) -> Mesh {
    let mut m = Mesh::default();
    let rings = sides / 2;
    let prof: Vec<(f32, f32)> = (0..=rings)
        .map(|i| {
            let a = i as f32 / rings as f32 * std::f32::consts::FRAC_PI_2;
            (0.5 * a.cos(), a.sin())
        })
        .collect();
    m.lathe(&prof, sides, Vec3::ZERO, 0.0, 0.0, 0);
    m
}

pub fn cylinder(sides: usize) -> Mesh {
    let mut m = Mesh::default();
    m.lathe(&[(0.5, 0.0), (0.5, 1.0)], sides, Vec3::ZERO, 0.0, 0.0, 0);
    m.lathe(&[(0.5, 1.0), (0.0, 1.0)], sides, Vec3::ZERO, 0.0, 0.0, 0);
    m
}

/// Leaf vertices store canopy "openness" in `part` (0 = deep inside the
/// crown, 0.5 = outer surface) for ambient occlusion; bark uses part = 1.
fn finish_canopy(m: &mut Mesh, from: usize, centre: Vec3, radius: Vec3, bend: f32) {
    for v in &mut m.verts[from..] {
        if v.part > 0.75 {
            continue;
        }
        let p = Vec3::from(v.pos);
        let d = (p - centre) / radius;
        // Blend towards "crown normals" so the canopy shades as one soft mass.
        let crown = d.normalize_or(Vec3::Y);
        let n = Vec3::from(v.normal).lerp(crown, bend).normalize_or(Vec3::Y);
        v.normal = n.to_array();
        v.part = 0.5 * (d.length() * 0.9 + 0.1 * (d.y * 0.5 + 0.5)).clamp(0.0, 1.0);
    }
}

pub fn conifer(lod: usize) -> Mesh {
    let mut m = Mesh::default();
    if lod == 0 {
        m.lathe(&[(0.075, 0.0), (0.05, 0.08), (0.035, 0.6), (0.012, 0.97)], 7, Vec3::ZERO, 1.0, 0.0, 0);
        let start = m.verts.len();
        let tiers = 7;
        for i in 0..tiers {
            let t = i as f32 / (tiers - 1) as f32;
            let yb = 0.1 + t * 0.74;
            let rb = 0.5 * (1.0 - t * 0.82);
            let th = 0.27 * (1.0 - t * 0.35);
            let seed = 11 + i as u32 * 7;
            // underside (droops slightly) and top surface with jagged edge
            m.lathe(&[(0.0, yb + th * 0.28), (rb * 0.7, yb + 0.015), (rb, yb - 0.02)], 12, Vec3::ZERO, 0.0, 0.35, seed);
            m.lathe(&[(rb, yb - 0.02), (rb * 0.78, yb + th * 0.18), (rb * 0.42, yb + th * 0.55), (0.0, yb + th)], 12, Vec3::ZERO, 0.0, 0.35, seed);
        }
        finish_canopy(&mut m, start, Vec3::new(0.0, 0.45, 0.0), Vec3::new(0.5, 0.55, 0.5), 0.25);
    } else {
        m.lathe(&[(0.05, 0.0), (0.04, 0.15)], 4, Vec3::ZERO, 1.0, 0.0, 0);
        let start = m.verts.len();
        m.lathe(&[(0.0, 0.1), (0.48, 0.1), (0.3, 0.42), (0.26, 0.45), (0.0, 1.0)], 6, Vec3::ZERO, 0.0, 0.0, 0);
        finish_canopy(&mut m, start, Vec3::new(0.0, 0.45, 0.0), Vec3::new(0.5, 0.55, 0.5), 0.2);
    }
    m
}

pub fn broadleaf(lod: usize) -> Mesh {
    let mut m = Mesh::default();
    let crown_c = Vec3::new(0.0, 0.66, 0.0);
    let crown_r = Vec3::new(0.5, 0.34, 0.5);
    if lod == 0 {
        // trunk with root flare
        m.lathe(&[(0.07, 0.0), (0.04, 0.05), (0.032, 0.25), (0.026, 0.42)], 8, Vec3::ZERO, 1.0, 0.08, 5);
        // main limbs to the crown clusters
        let lobes = 7;
        let mut tips = Vec::new();
        for i in 0..lobes {
            let a = i as f32 / lobes as f32 * TAU + 0.4 * (i % 2) as f32;
            let rr = 0.25 + 0.04 * ((i * 5) % 3) as f32;
            let y = 0.58 + 0.07 * ((i * 3) % 4) as f32 / 3.0;
            tips.push(Vec3::new(a.cos() * rr, y, a.sin() * rr));
        }
        for (i, t) in tips.iter().enumerate() {
            let base = Vec3::new(0.0, 0.36 + 0.03 * (i % 3) as f32, 0.0);
            let mid = base.lerp(*t, 0.55) + Vec3::Y * 0.04;
            m.tube(base, mid, 0.022, 0.014, 5, 1.0);
            m.tube(mid, *t * Vec3::new(0.85, 1.0, 0.85), 0.014, 0.006, 4, 1.0);
        }
        let start = m.verts.len();
        m.sphere(Vec3::new(0.0, 0.7, 0.0), Vec3::new(0.3, 0.24, 0.3), 12, 7, 0.0, 0.3, 1);
        for (i, t) in tips.iter().enumerate() {
            let r = 0.17 + 0.03 * ((i * 7) % 3) as f32;
            m.sphere(*t + Vec3::Y * 0.02, Vec3::new(r, r * 0.8, r), 10, 6, 0.0, 0.35, 10 + i as u32);
        }
        m.sphere(Vec3::new(0.05, 0.86, -0.03), Vec3::new(0.2, 0.14, 0.2), 9, 6, 0.0, 0.3, 30);
        finish_canopy(&mut m, start, crown_c, crown_r, 0.55);
    } else {
        m.lathe(&[(0.06, 0.0), (0.045, 0.45)], 4, Vec3::ZERO, 1.0, 0.0, 0);
        let start = m.verts.len();
        m.sphere(Vec3::new(0.0, 0.66, 0.0), Vec3::new(0.48, 0.33, 0.48), 7, 5, 0.0, 0.2, 3);
        finish_canopy(&mut m, start, crown_c, crown_r, 0.5);
    }
    m
}

pub fn house() -> Mesh {
    let mut m = Mesh::default();
    m.add_box(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 0.62, 0.5), 0.0);
    let c = Vec3::new(0.0, 0.7, 0.0);
    let (e, y0, r) = (0.56, 0.58, 1.0);
    let z = 0.54;
    m.face(&[Vec3::new(-e, y0, -z), Vec3::new(0.0, r, -z), Vec3::new(0.0, r, z), Vec3::new(-e, y0, z)], c, 0.0);
    m.face(&[Vec3::new(e, y0, -z), Vec3::new(e, y0, z), Vec3::new(0.0, r, z), Vec3::new(0.0, r, -z)], c, 0.0);
    for zz in [-0.5, 0.5] {
        m.face(&[Vec3::new(-0.5, 0.62, zz), Vec3::new(0.5, 0.62, zz), Vec3::new(0.0, r - 0.03, zz)], Vec3::new(0.0, 0.7, 0.0), 0.0);
    }
    m
}

pub fn train_car() -> Mesh {
    let mut m = Mesh::default();
    let sec = [(-0.5, 0.0), (0.5, 0.0), (0.5, 0.62), (0.36, 0.95), (-0.36, 0.95), (-0.5, 0.62)];
    let rings = [(-0.5, 0.82, 0.86), (-0.44, 1.0, 1.0), (0.44, 1.0, 1.0), (0.5, 0.82, 0.86)];
    let pt = |r: usize, k: usize| {
        let (z, sx, sy) = rings[r];
        Vec3::new(sec[k].0 * sx, sec[k].1 * sy, z)
    };
    let c = Vec3::new(0.0, 0.5, 0.0);
    for r in 0..3 {
        for k in 0..6 {
            let k2 = (k + 1) % 6;
            m.face(&[pt(r, k), pt(r, k2), pt(r + 1, k2), pt(r + 1, k)], c, 0.0);
        }
    }
    for r in [0, 3] {
        let pts: Vec<Vec3> = (0..6).map(|k| pt(r, k)).collect();
        m.face(&pts, c, 0.0);
    }
    m
}

pub fn aircar() -> Mesh {
    let mut m = Mesh::default();
    let b = [Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, -0.5, -0.5), Vec3::new(0.42, -0.45, 0.5), Vec3::new(-0.42, -0.45, 0.5)];
    let t = [Vec3::new(-0.36, 0.5, -0.42), Vec3::new(0.36, 0.5, -0.42), Vec3::new(0.3, 0.25, 0.25), Vec3::new(-0.3, 0.25, 0.25)];
    let c = Vec3::ZERO;
    m.face(&b, c, 0.0);
    m.face(&t, c, 0.0);
    m.face(&[b[0], b[1], t[1], t[0]], c, 0.0);
    m.face(&[b[3], b[2], t[2], t[3]], c, 0.0);
    m.face(&[b[0], b[3], t[3], t[0]], c, 0.0);
    m.face(&[b[1], b[2], t[2], t[1]], c, 0.0);
    m
}

pub fn ball_sphere() -> Mesh {
    let mut m = Mesh::default();
    m.sphere(Vec3::ZERO, Vec3::ONE, 28, 18, 0.0, 0.0, 0);
    m
}

/// Terrain chunk grid: (x, y) in [0, 1], z = 1 for skirt vertices.
pub fn terrain_grid(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut v = Vec::new();
    let mut idx = Vec::new();
    let row = n + 1;
    for j in 0..=n {
        for i in 0..=n {
            v.push([i as f32 / n as f32, j as f32 / n as f32, 0.0]);
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = j * row + i;
            idx.extend_from_slice(&[a, a + 1, a + row, a + 1, a + row + 1, a + row]);
        }
    }
    // Skirts along the four edges.
    let edges: [Vec<u32>; 4] =
        [(0..=n).collect(), (0..=n).map(|i| n * row + i).collect(), (0..=n).map(|j| j * row).collect(), (0..=n).map(|j| j * row + n).collect()];
    for e in edges.iter() {
        let base = v.len() as u32;
        for &k in e {
            let p = v[k as usize];
            v.push([p[0], p[1], 1.0]);
        }
        for i in 0..n as usize {
            let (a, b) = (e[i], e[i + 1]);
            let (c, d) = (base + i as u32, base + i as u32 + 1);
            idx.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (v, idx)
}
