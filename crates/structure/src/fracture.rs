//! Hierarchical Voronoi pre-fracture of a unit cell box [-0.5,0.5]^3.
//! Level 0 = whole cell, levels 1..3 = 8..16 pieces each (12 on average).
//! Shared by the runtime and by `tools/gen_fracture.rs` (via #[path]).
#![allow(dead_code)]

use glam::DVec3;

#[derive(Clone, Debug)]
pub struct PieceGeom {
    pub parent: i32,
    pub level: u8,
    pub centroid: [f32; 3],
    pub volume: f32,
    pub extent: [f32; 3],
    pub children: Vec<u32>,
    pub first_index: u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FVert {
    pub pos: [f32; 3],
    pub interior: f32,
    pub nrm: [f32; 3],
    pub _pad: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Asset {
    pub pieces: Vec<PieceGeom>,
    pub vertices: Vec<FVert>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Face {
    pts: Vec<DVec3>,
    interior: bool,
}

#[derive(Clone, Debug)]
struct Poly {
    faces: Vec<Face>,
}

const EPS: f64 = 1e-9;

impl Poly {
    fn unit_box() -> Poly {
        let c = |x: f64, y: f64, z: f64| DVec3::new(x * 0.5, y * 0.5, z * 0.5);
        let faces = vec![
            vec![c(1., -1., -1.), c(1., 1., -1.), c(1., 1., 1.), c(1., -1., 1.)],
            vec![c(-1., -1., -1.), c(-1., -1., 1.), c(-1., 1., 1.), c(-1., 1., -1.)],
            vec![c(-1., 1., -1.), c(-1., 1., 1.), c(1., 1., 1.), c(1., 1., -1.)],
            vec![c(-1., -1., -1.), c(1., -1., -1.), c(1., -1., 1.), c(-1., -1., 1.)],
            vec![c(-1., -1., 1.), c(1., -1., 1.), c(1., 1., 1.), c(-1., 1., 1.)],
            vec![c(-1., -1., -1.), c(-1., 1., -1.), c(1., 1., -1.), c(1., -1., -1.)],
        ];
        Poly { faces: faces.into_iter().map(|pts| Face { pts, interior: false }).collect() }
    }

    /// Keep the half-space n·x <= d; the new cap face is marked interior.
    fn clip(&self, n: DVec3, d: f64) -> Poly {
        let mut out = Vec::new();
        let mut cut: Vec<DVec3> = Vec::new();
        for f in &self.faces {
            let m = f.pts.len();
            let mut np = Vec::new();
            for i in 0..m {
                let a = f.pts[i];
                let b = f.pts[(i + 1) % m];
                let da = n.dot(a) - d;
                let db = n.dot(b) - d;
                if da <= EPS {
                    np.push(a);
                }
                if da.abs() <= EPS {
                    cut.push(a);
                }
                if (da < -EPS && db > EPS) || (da > EPS && db < -EPS) {
                    let p = a + (b - a) * (da / (da - db));
                    np.push(p);
                    cut.push(p);
                }
            }
            if np.len() >= 3 {
                out.push(Face { pts: np, interior: f.interior });
            }
        }
        if cut.len() >= 3 {
            let c = cut.iter().copied().sum::<DVec3>() / cut.len() as f64;
            let u = n.any_orthonormal_vector();
            let v = n.cross(u);
            cut.sort_by(|a, b| {
                let (pa, pb) = (*a - c, *b - c);
                pa.dot(v).atan2(pa.dot(u)).partial_cmp(&pb.dot(v).atan2(pb.dot(u))).unwrap()
            });
            cut.dedup_by(|a, b| a.distance(*b) < 1e-7);
            if cut.len() >= 3 && cut[0].distance(*cut.last().unwrap()) < 1e-7 {
                cut.pop();
            }
            if cut.len() >= 3 {
                out.push(Face { pts: cut, interior: true });
            }
        }
        Poly { faces: out }
    }

    fn centroid_volume(&self) -> (DVec3, f64) {
        let n: usize = self.faces.iter().map(|f| f.pts.len()).sum();
        if n == 0 {
            return (DVec3::ZERO, 0.0);
        }
        let o = self.faces.iter().flat_map(|f| f.pts.iter().copied()).sum::<DVec3>() / n as f64;
        let (mut vol, mut c) = (0.0, DVec3::ZERO);
        for f in &self.faces {
            for i in 1..f.pts.len() - 1 {
                let (a, b, cc) = (f.pts[0], f.pts[i], f.pts[i + 1]);
                let v = (a - o).dot((b - o).cross(cc - o)).abs() / 6.0;
                vol += v;
                c += (o + a + b + cc) * 0.25 * v;
            }
        }
        (if vol > 0.0 { c / vol } else { o }, vol)
    }

    fn bounds(&self) -> (DVec3, DVec3) {
        let (mut lo, mut hi) = (DVec3::splat(1e9), DVec3::splat(-1e9));
        for p in self.faces.iter().flat_map(|f| f.pts.iter()) {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
    }

    fn contains(&self, p: DVec3, c: DVec3) -> bool {
        self.faces.iter().all(|f| face_normal(f, c).dot(p - f.pts[0]) < -1e-6)
    }
}

fn face_normal(f: &Face, inside: DVec3) -> DVec3 {
    let mut n = DVec3::ZERO;
    let m = f.pts.len();
    for i in 0..m {
        let (a, b) = (f.pts[i], f.pts[(i + 1) % m]);
        n += DVec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    let n = n.normalize_or_zero();
    if n.dot(f.pts[0] - inside) < 0.0 {
        -n
    } else {
        n
    }
}

fn rnd(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn voronoi(p: &Poly, k: usize, seed: &mut u64) -> Vec<Poly> {
    let (c, _) = p.centroid_volume();
    let (lo, hi) = p.bounds();
    let mut seeds = Vec::new();
    let mut tries = 0;
    while seeds.len() < k && tries < 20000 {
        tries += 1;
        let q = lo + (hi - lo) * DVec3::new(rnd(seed), rnd(seed), rnd(seed));
        if p.contains(q, c) {
            seeds.push(q);
        }
    }
    let mut out = Vec::new();
    for (i, &a) in seeds.iter().enumerate() {
        let mut cell = p.clone();
        for (j, &b) in seeds.iter().enumerate() {
            if i != j {
                let n = (b - a).normalize();
                cell = cell.clip(n, n.dot((a + b) * 0.5));
                if cell.faces.len() < 4 {
                    break;
                }
            }
        }
        if cell.faces.len() >= 4 && cell.centroid_volume().1 > 1e-8 {
            out.push(cell);
        }
    }
    out
}

/// Builds the hierarchy (≈ 1 + 12 + 144 + 1728 pieces for k = 12, levels = 3).
pub fn generate(k: usize, levels: u8, seed: u64) -> Asset {
    let mut asset = Asset::default();
    let mut s = seed | 1;
    let root = Poly::unit_box();
    push_piece(&mut asset, &root, -1, 0);
    let mut queue: Vec<(usize, Poly)> = vec![(0, root)];
    let mut qi = 0;
    while qi < queue.len() {
        let (pi, poly) = queue[qi].clone();
        qi += 1;
        let lvl = asset.pieces[pi].level;
        if lvl >= levels {
            continue;
        }
        let kk = (k as f64 * (0.75 + 0.5 * rnd(&mut s))).round().clamp(8.0, 16.0) as usize;
        for child in voronoi(&poly, kk, &mut s) {
            let ci = push_piece(&mut asset, &child, pi as i32, lvl + 1);
            asset.pieces[pi].children.push(ci as u32);
            queue.push((ci, child));
        }
    }
    asset
}

fn push_piece(asset: &mut Asset, p: &Poly, parent: i32, level: u8) -> usize {
    let (c, vol) = p.centroid_volume();
    let (lo, hi) = p.bounds();
    let first = asset.indices.len() as u32;
    for f in &p.faces {
        let n = face_normal(f, c);
        let base = asset.vertices.len() as u32;
        for q in &f.pts {
            let rel = *q - c;
            asset.vertices.push(FVert {
                pos: [rel.x as f32, rel.y as f32, rel.z as f32],
                interior: if f.interior { 1.0 } else { 0.0 },
                nrm: [n.x as f32, n.y as f32, n.z as f32],
                _pad: 0.0,
            });
        }
        for i in 1..f.pts.len() as u32 - 1 {
            let (a, b, cc) = (f.pts[0], f.pts[i as usize], f.pts[i as usize + 1]);
            if (b - a).cross(cc - a).dot(n) >= 0.0 {
                asset.indices.extend_from_slice(&[base, base + i, base + i + 1]);
            } else {
                asset.indices.extend_from_slice(&[base, base + i + 1, base + i]);
            }
        }
    }
    asset.pieces.push(PieceGeom {
        parent,
        level,
        centroid: [c.x as f32, c.y as f32, c.z as f32],
        volume: vol as f32,
        extent: [(hi.x - lo.x) as f32, (hi.y - lo.y) as f32, (hi.z - lo.z) as f32],
        children: Vec::new(),
        first_index: first,
        index_count: asset.indices.len() as u32 - first,
    });
    asset.pieces.len() - 1
}

// ---- binary format "VFR1" (little endian) -----------------------------

pub fn save(a: &Asset) -> Vec<u8> {
    let mut b: Vec<u8> = b"VFR1".to_vec();
    let w = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
    let wf = |b: &mut Vec<u8>, v: f32| b.extend_from_slice(&v.to_le_bytes());
    w(&mut b, a.pieces.len() as u32);
    w(&mut b, a.vertices.len() as u32);
    w(&mut b, a.indices.len() as u32);
    for p in &a.pieces {
        w(&mut b, p.parent as u32);
        w(&mut b, p.level as u32);
        p.centroid.iter().for_each(|&v| wf(&mut b, v));
        wf(&mut b, p.volume);
        p.extent.iter().for_each(|&v| wf(&mut b, v));
        w(&mut b, p.first_index);
        w(&mut b, p.index_count);
        w(&mut b, p.children.len() as u32);
        p.children.iter().for_each(|&c| w(&mut b, c));
    }
    for v in &a.vertices {
        v.pos.iter().for_each(|&x| wf(&mut b, x));
        wf(&mut b, v.interior);
        v.nrm.iter().for_each(|&x| wf(&mut b, x));
    }
    a.indices.iter().for_each(|&i| w(&mut b, i));
    b
}

pub fn load(b: &[u8]) -> Option<Asset> {
    if b.len() < 16 || &b[0..4] != b"VFR1" {
        return None;
    }
    let mut o = 4usize;
    let mut r = || -> Option<u32> {
        let v = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?);
        o += 4;
        Some(v)
    };
    let (np, nv, ni) = (r()?, r()?, r()?);
    let mut a = Asset::default();
    for _ in 0..np {
        let parent = r()? as i32;
        let level = r()? as u8;
        let centroid = [f32::from_bits(r()?), f32::from_bits(r()?), f32::from_bits(r()?)];
        let volume = f32::from_bits(r()?);
        let extent = [f32::from_bits(r()?), f32::from_bits(r()?), f32::from_bits(r()?)];
        let first_index = r()?;
        let index_count = r()?;
        let nc = r()?;
        let mut children = Vec::with_capacity(nc as usize);
        for _ in 0..nc {
            children.push(r()?);
        }
        a.pieces.push(PieceGeom { parent, level, centroid, volume, extent, children, first_index, index_count });
    }
    for _ in 0..nv {
        let pos = [f32::from_bits(r()?), f32::from_bits(r()?), f32::from_bits(r()?)];
        let interior = f32::from_bits(r()?);
        let nrm = [f32::from_bits(r()?), f32::from_bits(r()?), f32::from_bits(r()?)];
        a.vertices.push(FVert { pos, interior, nrm, _pad: 0.0 });
    }
    for _ in 0..ni {
        a.indices.push(r()?);
    }
    Some(a)
}

pub const DEFAULT_SEED: u64 = 0xF4AC_7E11;

pub fn load_or_generate(path: &str) -> Asset {
    if let Some(a) = std::fs::read(path).ok().and_then(|b| load(&b)) {
        return a;
    }
    let a = generate(12, 3, DEFAULT_SEED);
    if let Some(dir) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, save(&a));
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn volumes_partition() {
        let a = generate(10, 2, 7);
        assert!((a.pieces[0].volume - 1.0).abs() < 1e-4);
        let s: f32 = a.pieces[0].children.iter().map(|&c| a.pieces[c as usize].volume).sum();
        assert!((s - 1.0).abs() < 2e-3, "sum {s}");
        let n = a.pieces[0].children.len();
        assert!((8..=16).contains(&n), "children {n}");
        let b = load(&save(&a)).unwrap();
        assert_eq!(b.pieces.len(), a.pieces.len());
        assert_eq!(b.indices, a.indices);
    }
}
