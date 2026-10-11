//! Procedural generation of the three valleys.
//!
//!  * Пастораль  (valley 0): rolling hills, a meandering river, patchwork farm
//!    fields, five villages, farmsteads and copses.
//!  * Мегаполис  (valley 1): a 4.6 km long city on an 80 m street grid with a
//!    glass downtown, parks, a central lake and a canal; greenhouses outside.
//!  * Дикие земли (valley 2): steep wooded hills, lakes and an eco-village.
//!
//! Mountains rise near both end caps; promenades run along every window.

use std::f32::consts::{PI, TAU};

use glam::Vec3;

use super::*;
use crate::colony::{half_land_width, strip_of, surf_to_world, world_to_surf};

// ------------------------------------------------------------------ noise --
fn hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^ (h >> 15)
}
fn h01(x: i32, y: i32, seed: u32) -> f32 {
    (hash(x, y, seed) >> 8) as f32 / 16_777_216.0
}
fn vnoise(x: f32, y: f32, seed: u32) -> f32 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx as i32, fy as i32);
    let (tx, ty) = (x - fx, y - fy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let a = h01(ix, iy, seed);
    let b = h01(ix + 1, iy, seed);
    let c = h01(ix, iy + 1, seed);
    let d = h01(ix + 1, iy + 1, seed);
    lerp(lerp(a, b, sx), lerp(c, d, sx), sy)
}
fn fbm(x: f32, y: f32, oct: u32, seed: u32) -> f32 {
    let (mut px, mut py) = (x, y);
    let (mut amp, mut sum, mut norm) = (0.5, 0.0, 0.0);
    for i in 0..oct {
        sum += amp * vnoise(px, py, seed.wrapping_add(i * 101));
        norm += amp;
        let nx = px * 1.6 - py * 1.2 + 5.3;
        py = px * 1.2 + py * 1.6 + 1.7;
        px = nx;
        amp *= 0.5;
    }
    sum / norm
}
fn ridged(x: f32, y: f32, oct: u32, seed: u32) -> f32 {
    let (mut px, mut py) = (x, y);
    let (mut amp, mut sum, mut norm) = (0.5, 0.0, 0.0);
    for i in 0..oct {
        let n = 1.0 - (vnoise(px, py, seed.wrapping_add(i * 37)) * 2.0 - 1.0).abs();
        sum += amp * n * n;
        norm += amp;
        let nx = px * 1.7 - py * 1.1;
        py = px * 1.1 + py * 1.7;
        px = nx;
        amp *= 0.5;
    }
    sum / norm
}
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ------------------------------------------------------------- layout -----
/// Layout scale relative to the original 800 m prototype (canonical R = 4 km).
pub const S: f32 = R / 800.0;
const VILLAGES: [(f32, f32, f32); 12] = [
    (-750.0, 3300.0, 260.0),
    (850.0, 5600.0, 220.0),
    (-300.0, 8200.0, 280.0),
    (600.0, 10500.0, 240.0),
    (-900.0, 12800.0, 260.0),
    (250.0, 15400.0, 300.0),
    (1100.0, 17600.0, 220.0),
    (-500.0, 19800.0, 260.0),
    (700.0, 22300.0, 280.0),
    (-1000.0, 24600.0, 220.0),
    (300.0, 26800.0, 260.0),
    (-700.0, 28700.0, 240.0),
];
const ECO_VILLAGE: (f32, f32, f32) = (300.0, 16000.0, 220.0);
pub const TRACK_S: f32 = 240.0 * S;
pub const STATIONS: [f32; 5] = [3400.0, 8800.0, 16000.0, 23200.0, 28600.0];
const CITY_HALF: f32 = 2300.0 * S;
const CENTER: f32 = L * 0.5;
const FIELD_W: f32 = 56.0;
const FIELD_L: f32 = 112.0;

fn river_s(v: f32) -> f32 {
    110.0 * S * (v / (650.0 * S) + 0.5).sin() + 45.0 * S * (v / (210.0 * S) + 2.0).sin() + 18.0 * (v / 140.0).sin()
}
fn lake_e(s: f32, v: f32) -> f32 {
    ((s / (170.0 * S)).powi(2) + ((v - CENTER) / (320.0 * S)).powi(2)).sqrt()
}
fn block_park(s: f32, v: f32) -> bool {
    let (bs, bv) = ((s / 80.0).floor() as i32, (v / 80.0).floor() as i32);
    h01(bs, bv, 991) < 0.12 || lake_e((bs as f32 + 0.5) * 80.0, (bv as f32 + 0.5) * 80.0) < 1.35
}
fn lake_block(s: f32, v: f32) -> bool {
    let (bs, bv) = ((s / 80.0).floor(), (v / 80.0).floor());
    lake_e((bs + 0.5) * 80.0, (bv + 0.5) * 80.0) < 1.35
}
/// Inside the lake park: every block within 11 m is a lake block, so the
/// streets between lake blocks vanish while the perimeter street stays.
fn lake_park(s: f32, v: f32) -> bool {
    [(-11.0, -11.0), (11.0, -11.0), (-11.0, 11.0), (11.0, 11.0)].iter().all(|(a, c)| lake_block(s + a, v + c))
}
fn mountain_factor(v: f32) -> f32 {
    1.0 - smoothstep(0.0, 520.0 * S, v.min(L - v))
}

#[derive(Clone, Copy)]
pub struct Sample {
    pub h: f32,
    pub z: [f32; 4], // grass, farm, city, forest
}

pub fn sample(u: f32, v: f32) -> Sample {
    let (k, s) = strip_of(u.rem_euclid(CIRC) / R);
    let hw = half_land_width();
    let edge = hw - s.abs();
    let seed = 1000 + k as u32 * 7919;
    let (mut h, mut z);
    match k {
        0 => {
            let hills = (fbm(s * 0.0012, v * 0.0012, 5, seed) - 0.5) * 55.0
                + (fbm(s * 0.0035, v * 0.0035, 3, seed + 7) - 0.5) * 14.0
                + (fbm(s * 0.015, v * 0.015, 2, seed + 1) - 0.5) * 4.0;
            let base = 4.5 + hills;
            let rd = (s - river_s(v)).abs();
            let land = 1.2 + (base.max(0.6) - 0.6) * smoothstep(20.0, 170.0, rd);
            h = lerp(-3.2, land, smoothstep(22.0, 38.0, rd));
            let mut farm = smoothstep(0.44, 0.5, fbm(s * 0.0011 + 7.0, v * 0.0008, 3, seed + 2)) * smoothstep(55.0, 90.0, rd);
            let mut forest = smoothstep(0.6, 0.7, fbm(s * 0.0025 + 31.0, v * 0.002, 4, seed + 3)) * (1.0 - farm) * smoothstep(40.0, 70.0, rd);
            let mut city: f32 = 0.0;
            let mut flat: f32 = 0.0;
            for (vs, vv, r) in VILLAGES {
                let d = ((s - vs).powi(2) + (v - vv).powi(2)).sqrt();
                city = city.max(1.0 - smoothstep(r - 40.0, r, d));
                flat = flat.max(1.0 - smoothstep(r, r + 90.0, d));
            }
            h = lerp(h, 3.0, flat);
            farm *= 1.0 - flat;
            forest *= 1.0 - flat;
            let grass = lerp(0.8, 1.0, fbm(s * 0.02, v * 0.02, 2, seed + 4)) * smoothstep(23.0, 34.0, rd);
            z = [grass, farm, city, forest];
        }
        1 => {
            let base = 1.4 + (fbm(s * 0.004, v * 0.004, 3, seed) - 0.5) * 2.0;
            let in_city = 1.0 - smoothstep(CITY_HALF - 50.0, CITY_HALF + 30.0, (v - CENTER).abs());
            let city = in_city * smoothstep(30.0, 60.0, edge);
            let e = lake_e(s, v);
            h = lerp(-4.5, base, smoothstep(0.85, 1.0, e));
            let cd = (s + 200.0 * S).abs();
            if (v - CENTER).abs() < CITY_HALF + 60.0 {
                h = h.min(lerp(-3.0, base, smoothstep(8.0, 11.0, cd)));
            }
            let park = if city > 0.5 && block_park(s, v) { 1.0 } else { 0.0 };
            let grass = if city > 0.5 { park } else { 0.9 };
            let farm = (1.0 - in_city) * smoothstep(60.0, 90.0, edge);
            let forest = park * 0.25 * smoothstep(0.45, 0.6, fbm(s * 0.03, v * 0.03, 2, seed + 5));
            z = if city > 0.5 && lake_park(s, v) {
                // lawn + scattered trees, below the 0.35 street threshold
                [1.0, 0.0, 0.3, 0.1 + 0.6 * smoothstep(0.5, 0.7, fbm(s * 0.02, v * 0.02, 2, seed + 6))]
            } else {
                [grass, farm, city, forest]
            };
        }
        _ => {
            let n1 = fbm(s * 0.0009, v * 0.0009, 6, seed);
            let r = ridged(s * 0.0025, v * 0.0025, 4, seed + 1);
            h = (n1 - 0.40) * 220.0 + r * 40.0;
            h = h.max(-6.0);
            let (vs, vv, vr) = ECO_VILLAGE;
            let d = ((s - vs).powi(2) + (v - vv).powi(2)).sqrt();
            let city = 1.0 - smoothstep(vr - 35.0, vr, d);
            let flat = 1.0 - smoothstep(vr, vr + 130.0, d);
            h = lerp(h, 12.0, flat);
            let forest = smoothstep(0.36, 0.5, fbm(s * 0.005 + 200.0, v * 0.005, 3, seed + 2))
                * smoothstep(0.6, 2.5, h)
                * (1.0 - smoothstep(120.0, 170.0, h))
                * (1.0 - flat);
            let grass = 1.0 - smoothstep(130.0, 180.0, h) * 0.6;
            z = [grass, 0.0, city, forest];
        }
    }
    // Promenade along the windows.
    let fade_w = [60.0, 45.0, 140.0][k];
    let pe = smoothstep(14.0, 14.0 + fade_w, edge);
    h = lerp(0.8, h, pe);
    z[1] *= pe;
    z[2] *= pe;
    z[3] *= pe;
    z[0] *= smoothstep(10.0, 16.0, edge);
    // Mountains towards the end caps.
    let m = mountain_factor(v);
    if m > 0.0 {
        let dz = v.min(L - v);
        let rr = ridged(s * 0.0028, dz * 0.0028 + k as f32 * 50.0, 5, seed + 9);
        let mountain = (m.powf(1.7) * (700.0 + 450.0 * (rr - 0.4)) + m * rr * 70.0) * smoothstep(0.0, 220.0, edge);
        let t = smoothstep(0.0, 0.3, m);
        h = h * (1.0 - t) + mountain + h.max(0.0) * t * 0.3;
        z[1] *= 1.0 - smoothstep(0.0, 0.08, m);
        z[2] *= 1.0 - smoothstep(0.0, 0.05, m);
        let alpine = smoothstep(0.03, 0.15, m) * (1.0 - smoothstep(0.3, 0.45, m));
        z[3] = z[3].max(alpine * 0.8 * smoothstep(20.0, 60.0, edge));
        z[0] *= 1.0 - smoothstep(0.45, 0.65, m);
    }
    Sample { h, z }
}

fn par_fill<T: Send + Copy + Default>(w: usize, hgt: usize, f: impl Fn(usize, usize) -> T + Sync) -> Vec<T> {
    let mut out = vec![T::default(); w * hgt];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(1);
    let rows_per = hgt.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(rows_per * w).enumerate() {
            let f = &f;
            sc.spawn(move || {
                for (i, val) in chunk.iter_mut().enumerate() {
                    let y = ci * rows_per + i / w;
                    *val = f(i % w, y);
                }
            });
        }
    });
    out
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 16_777_216.0
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        lerp(a, b, self.next())
    }
    fn u(&mut self) -> u32 {
        (self.next() * 65535.0) as u32
    }
}

struct Builder {
    objects: Vec<ObjInstance>,
    colliders: Vec<Collider>,
    n_buildings: usize,
    n_trees: usize,
}

impl Builder {
    #[allow(clippy::too_many_arguments)]
    fn put(&mut self, strip: usize, s: f32, v: f32, h: f32, yaw: f32, size: [f32; 3], kind: u32, mat: u32, seed: u32, collide: bool) {
        let u = strip_u(strip, s);
        let p = surf_to_world(u, v, h);
        self.objects.push(ObjInstance { pos: p.to_array(), yaw, size, style: style(kind, mat, seed) });
        if collide {
            let swap = yaw.sin().abs() > 0.7;
            let (hs, hv) = if swap { (size[2] * 0.5, size[0] * 0.5) } else { (size[0] * 0.5, size[2] * 0.5) };
            let k = if kind == K_CYL || kind == K_DOME { 0.75 } else { 1.0 };
            let top = if kind == K_TOWER {
                0.88
            } else if kind == K_DOME {
                0.8
            } else {
                1.0
            };
            self.colliders.push(Collider { strip: strip as u8, s, v, hs: hs * k, hv: hv * k, h0: h - 1.0, h1: h + size[1] * top });
        }
    }
}

pub fn generate() -> World {
    let du = CIRC / HW as f32;
    let dv = L / (HH - 1) as f32;
    // One pass computes both maps (the zone map shares the heightmap grid).
    let both = par_fill(HW, HH, |i, j| {
        let sm = sample(i as f32 * du, j as f32 * dv);
        (sm.h, sm.z.map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8))
    });
    let height: Vec<f32> = both.iter().map(|b| b.0).collect();
    let zone: Vec<[u8; 4]> = both[..ZW * ZH].iter().map(|b| b.1).collect();
    drop(both);
    let mut world = World {
        height,
        zone,
        objects: Vec::new(),
        colliders: Vec::new(),
        grid: Vec::new(),
        tracks: Vec::new(),
        teleports: Vec::new(),
        n_buildings: 0,
        n_trees: 0,
    };
    let mut b = Builder { objects: Vec::new(), colliders: Vec::new(), n_buildings: 0, n_trees: 0 };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let hw = half_land_width();

    // ---------------------------------------------------------- maglev ----
    for k in 0..3 {
        for side in [-1.0f32, 1.0] {
            let s = TRACK_S * side;
            let u = strip_u(k, s);
            let n = (L / DECK_STEP) as usize + 1;
            let ground: Vec<f32> = (0..n).map(|i| world.height(u, i as f32 * DECK_STEP).max(0.0)).collect();
            let mut deck: Vec<f32> = (0..n)
                .map(|i| {
                    let a = i.saturating_sub(6);
                    let e = (i + 6).min(n - 1);
                    ground[a..=e].iter().cloned().fold(0.0, f32::max) + 11.0
                })
                .collect();
            for _ in 0..4 {
                let src = deck.clone();
                for i in 0..n {
                    let a = i.saturating_sub(8);
                    let e = (i + 8).min(n - 1);
                    let avg = src[a..=e].iter().sum::<f32>() / (e - a + 1) as f32;
                    deck[i] = avg.max(ground[i] + 7.0);
                }
            }
            let tr = Track { strip: k, s, v0: 2300.0, v1: L - 2300.0, deck, stations: STATIONS.to_vec() };
            let mut v = tr.v0;
            let mut seg = 0;
            while v < tr.v1 {
                let vc = v + 10.0;
                let d = tr.deck_at(vc);
                b.put(k, s, vc, d - 1.6, 0.0, [4.6, 1.6, 20.4], K_DECK, M_CONCRETE, seg, true);
                if seg % 2 == 0 {
                    let gr = world.height(u, vc);
                    let hgt = d - 1.6 - gr + 1.0;
                    b.put(k, s, vc, gr - 1.0, 0.0, [2.0, hgt, 2.0], K_CYL, M_CONCRETE, seg, true);
                }
                v += 20.0;
                seg += 1;
            }
            for &sv in &tr.stations {
                let d = tr.deck_at(sv);
                let ps = s + side * 6.9;
                b.put(k, ps, sv, d - 1.4, 0.0, [8.4, 1.4, 72.0], K_DECK, M_STATION, 7, true);
                b.put(k, ps - side * 2.0, sv, d + 5.0, 0.0, [12.0, 0.5, 76.0], K_DECK, M_STATION, 8, false);
                for zz in [-30.0, 0.0, 30.0] {
                    b.put(k, ps + side * 3.6, sv + zz, d, 0.0, [0.5, 5.0, 0.5], K_CYL, M_WHITE, 3, false);
                }
            }
            world.tracks.push(tr);
        }
    }
    let near_track = |s: f32, margin: f32| (s.abs() - TRACK_S).abs() < margin;
    let road_d = |s: f32, v: f32| {
        let f = |x: f32| ((x / 80.0 + 0.5).rem_euclid(1.0) - 0.5).abs() * 80.0;
        f(s).min(f(v))
    };
    let in_village = |k: usize, s: f32, v: f32| {
        (k == 0 && VILLAGES.iter().any(|vl| (s - vl.0).hypot(v - vl.1) < vl.2 + 10.0))
            || (k == 2 && (s - ECO_VILLAGE.0).hypot(v - ECO_VILLAGE.1) < ECO_VILLAGE.2 + 10.0)
    };

    // -------------------------------------------------------- megapolis ----
    let k = 1;
    for bi in -25i32..25 {
        let sc = 40.0 + 80.0 * bi as f32;
        for bj in 0..400 {
            let vc = 40.0 + 80.0 * bj as f32;
            let u = strip_u(k, sc);
            let sm = sample(u, vc);
            if sm.z[2] < 0.8 || sample(strip_u(k, sc + 30.0 * bi.signum() as f32), vc).z[2] < 0.6 {
                continue;
            }
            let wet =
                [(-28.0, -28.0), (28.0, -28.0), (-28.0, 28.0), (28.0, 28.0), (0.0, 0.0)].iter().any(|(a, c)| world.height(strip_u(k, sc + a), vc + c) < 0.4);
            if wet {
                continue;
            }
            if near_track(sc, 45.0) && STATIONS.iter().any(|&sv| (vc - sv).abs() < 80.0) {
                continue; // station plaza under the elevated platform
            }
            if block_park(sc, vc) {
                if h01(bi, bj, 5) < 0.3 {
                    let g = world.height(u, vc);
                    b.put(k, sc, vc, g - 0.3, 0.0, [16.0, 7.0, 16.0], K_DOME, M_GREENHOUSE, rng.u(), true);
                }
                continue;
            }
            let dt = (-((((vc - CENTER).abs() - 2200.0) / 1600.0).powi(2))).exp() * (1.0 - (sc / 1500.0).powi(2)).max(0.0);
            let sub = 0.6
                * ((-((vc - 7600.0) / 1300.0).powi(2)).exp()
                    + (-((vc - 24400.0) / 1300.0).powi(2)).exp()
                    + (-((vc - CENTER).powi(2) + (sc + 1300.0).powi(2)) / 900.0f32.powi(2)).exp());
            let f = dt.max(sub).clamp(0.0, 1.0);
            let r = rng.next();
            if f > 0.3 && r < 0.2 + 0.35 * f {
                let fp = rng.range(32.0, 46.0);
                let ht = rng.range(90.0, 120.0) + 160.0 * f.powf(1.5) * rng.range(0.6, 1.0);
                let ht = ht * (1.0 + 0.9 * f * f * rng.next());
                let mat = [M_GLASS, M_GLASS, M_STONE, M_STONE, M_NEON, M_WHITE][(rng.next() * 5.99) as usize];
                let kind = if ht > 140.0 && mat != M_STONE { K_TOWER } else { K_BOX };
                let g = world.height(u, vc);
                let fz = fp * rng.range(0.8, 1.0);
                if mat == M_STONE {
                    // Art-deco tower: stepped setbacks.
                    let mut y = g - 0.5;
                    let (mut w, mut d, mut hh) = (fp * 1.25, fz * 1.25, ht * 0.42);
                    let seed = rng.u();
                    for _ in 0..4 {
                        b.put(k, sc, vc, y, 0.0, [w, hh, d], K_BOX, M_STONE, seed, true);
                        y += hh;
                        w *= 0.78;
                        d *= 0.78;
                        hh *= 0.62;
                        b.n_buildings += 1;
                    }
                    b.put(k, sc, vc, y, 0.0, [w * 0.5, hh * 1.6, d * 0.5], K_TOWER, M_WHITE, seed, false);
                } else if rng.next() < 0.3 && ht > 150.0 {
                    // Twin towers joined by a sky-bridge.
                    let seed = rng.u();
                    let (w2, h2) = (fp * 0.44, ht * rng.range(0.75, 0.9));
                    b.put(k, sc - fp * 0.28, vc, g - 0.5, 0.0, [w2, ht, fz], kind, mat, seed, true);
                    b.put(k, sc + fp * 0.28, vc, g - 0.5, 0.0, [w2, h2, fz], kind, mat, seed ^ 7, true);
                    b.put(k, sc, vc, g - 0.5 + h2 * 0.55, 0.0, [fp * 0.2, 9.0, fz * 0.4], K_BOX, M_WHITE, seed, false);
                    b.put(k, sc - fp * 0.28, vc, g - 0.5 + ht * 0.99, 0.0, [1.2, 18.0 + ht * 0.08, 1.2], K_CYL, M_WHITE, seed, false);
                    b.n_buildings += 3;
                } else {
                    // Shaft + setback + crown + rooftop plant and mast.
                    let seed = rng.u();
                    let h1 = ht * rng.range(0.58, 0.7);
                    let h2 = ht * rng.range(0.18, 0.26);
                    let h3 = (ht - h1 - h2).max(8.0);
                    let y0 = g - 0.5;
                    b.put(k, sc, vc, y0, 0.0, [fp, h1, fz], K_BOX, mat, seed, true);
                    b.put(k, sc, vc, y0 + h1, 0.0, [fp * 0.8, h2, fz * 0.8], K_BOX, mat, seed, true);
                    b.put(k, sc, vc, y0 + h1 + h2, 0.0, [fp * 0.58, h3, fz * 0.58], kind, mat, seed, true);
                    // Thin vertical fins on the setback corners.
                    for (a, c) in [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)] {
                        b.put(k, sc + a * fp, vc + c * fz, y0 + h1, 0.0, [1.0, h2 + 4.0, 1.0], K_BOX, M_WHITE, seed, false);
                    }
                    let top = y0 + h1 + h2 + h3 * if kind == K_TOWER { 0.85 } else { 1.0 };
                    b.put(
                        k,
                        sc + fp * 0.1,
                        vc - fz * 0.08,
                        y0 + h1 + h2 + h3 * if kind == K_TOWER { 0.0 } else { 1.0 },
                        0.0,
                        [fp * 0.25, 4.5, fz * 0.3],
                        K_BOX,
                        M_WHITE,
                        seed,
                        false,
                    );
                    if ht > 120.0 {
                        b.put(k, sc, vc, top, 0.0, [1.4, 16.0 + ht * 0.1, 1.4], K_CYL, M_WHITE, seed, false);
                    }
                    b.n_buildings += 7;
                }
                b.put(k, sc, vc, g - 0.5, 0.0, [58.0, rng.range(6.0, 12.0), 58.0], K_BOX, [M_WHITE, M_STONE][(rng.next() * 1.99) as usize], rng.u(), true);
                b.n_buildings += 2;
            } else {
                for (ox, oz) in [(-15.25, -15.25), (15.25, -15.25), (-15.25, 15.25), (15.25, 15.25)] {
                    if rng.next() < 0.08 {
                        continue;
                    }
                    let (s, v) = (sc + ox, vc + oz);
                    let fx = rng.range(17.0, 26.0);
                    let fz = rng.range(17.0, 26.0);
                    let ht = 9.0 + rng.next().powf(1.6) * (22.0 + 120.0 * f);
                    let mat = if ht < 35.0 {
                        [M_RESIDENTIAL, M_RESIDENTIAL, M_STONE, M_WHITE][(rng.next() * 3.99) as usize]
                    } else {
                        [M_GLASS, M_WHITE, M_RESIDENTIAL, M_STONE, M_STONE, M_NEON][(rng.next() * 5.99) as usize]
                    };
                    let g = world.height(strip_u(k, s), v);
                    b.put(k, s, v, g - 0.5, 0.0, [fx, ht, fz], K_BOX, mat, rng.u(), true);
                    b.n_buildings += 1;
                    // Rooftop details: plant room, water tank.
                    let r2 = rng.next();
                    if r2 < 0.55 {
                        b.put(
                            k,
                            s + rng.range(-3.0, 3.0),
                            v + rng.range(-3.0, 3.0),
                            g - 0.5 + ht,
                            0.0,
                            [fx * rng.range(0.25, 0.4), rng.range(2.5, 4.5), fz * rng.range(0.25, 0.4)],
                            K_BOX,
                            M_WHITE,
                            rng.u(),
                            false,
                        );
                    }
                    if r2 > 0.75 && mat == M_RESIDENTIAL {
                        b.put(k, s - fx * 0.25, v + fz * 0.2, g - 0.5 + ht, 0.0, [3.0, 4.0, 3.0], K_CYL, M_WHITE, rng.u(), false);
                    }
                }
            }
        }
    }
    // Bridges where cross streets meet the canal.
    {
        let cs = -200.0 * S;
        for j in 0..=(L / 80.0) as i32 {
            let v = j as f32 * 80.0;
            if (v - CENTER).abs() > CITY_HALF + 60.0 {
                continue;
            }
            let u = strip_u(k, cs);
            if world.height(u, v) > 0.3 || sample(u, v).z[2] < 0.35 {
                continue;
            }
            let bank = world.height(strip_u(k, cs - 20.0), v).max(world.height(strip_u(k, cs + 20.0), v));
            b.put(k, cs, v, bank - 0.6, 0.0, [34.0, 0.9, 14.0], K_DECK, M_CONCRETE, j as u32, true);
            for side in [-1.0f32, 1.0] {
                b.put(k, cs, v + side * 6.6, bank + 0.3, 0.0, [34.0, 1.0, 0.4], K_BOX, M_WHITE, j as u32, false);
            }
            b.n_buildings += 1;
        }
    }
    // Greenhouses on the farmland at both ends of the city.
    for bi in -25..26 {
        for bj in 0..533 {
            let s = bi as f32 * 62.0;
            let v = 30.0 + bj as f32 * 60.0;
            if near_track(s, 30.0) {
                continue;
            }
            let u = strip_u(k, s);
            let sm = sample(u, v);
            if sm.z[1] < 0.9
                || mountain_factor(v) > 0.0
                || h01(bi, bj, 77) > 0.4
                || [(-24.0, -24.0), (24.0, 24.0), (-24.0, 24.0), (24.0, -24.0)].iter().any(|(a, c)| sample(strip_u(k, s + a), v + c).z[2] > 0.2)
            {
                continue;
            }
            let g = world.height(u, v);
            b.put(k, s, v, g - 0.3, 0.0, [42.0, 13.0, 42.0], K_DOME, M_GREENHOUSE, rng.u(), true);
            b.n_buildings += 1;
        }
    }

    // ----------------------------------------------------- villages ----------
    let village = |w: &World, b: &mut Builder, rng: &mut Rng, k: usize, (vs, vv, vr): (f32, f32, f32), eco: bool| {
        let grid_d = |x: f32| ((x / 80.0 + 0.5).rem_euclid(1.0) - 0.5).abs() * 80.0;
        let tw = (((vs / 80.0).floor() + 0.5) * 80.0, ((vv / 80.0).floor() + 0.5) * 80.0);
        let mut v = (vv - vr).max(0.0);
        while v < vv + vr {
            let mut s = vs - vr;
            while s < vs + vr {
                let d = ((s - vs).powi(2) + (v - vv).powi(2)).sqrt();
                let rd = grid_d(s).min(grid_d(v));
                let u = strip_u(k, s);
                let g = w.height(u, v);
                // street half-width 7 m + sidewalk 2.5 m + max house half-size 6.5 m + 1.5 m
                let dry = [(-7.0, -7.0), (7.0, -7.0), (-7.0, 7.0), (7.0, 7.0)].iter().all(|(a, c)| w.height(strip_u(k, s + a), v + c) > 0.6);
                if d < vr - 30.0 && rd > 17.5 && !near_track(s, 20.0) && g > 0.5 && dry && (s - tw.0).hypot(v - tw.1) > 16.0 && rng.next() < 0.7 {
                    let yaw = if grid_d(s) < grid_d(v) { PI * 0.5 } else { 0.0 };
                    if eco {
                        let sz = rng.range(9.0, 14.0);
                        b.put(k, s, v, g - 0.3, 0.0, [sz, rng.range(5.0, 7.0), sz], K_DOME, M_GREENHOUSE, rng.u(), true);
                    } else {
                        let size = [rng.range(7.0, 10.0), rng.range(6.0, 9.0), rng.range(9.0, 13.0)];
                        b.put(k, s, v, g - 0.3, yaw, size, K_HOUSE, M_RESIDENTIAL, rng.u(), true);
                    }
                    b.n_buildings += 1;
                }
                s += 16.0;
            }
            v += 16.0;
        }
        let (s, v) = tw;
        let g = w.height(strip_u(k, s), v);
        if eco {
            b.put(k, s, v, g - 0.5, 0.0, [12.0, 46.0, 12.0], K_TOWER, M_WHITE, rng.u(), true);
        } else {
            b.put(k, s, v, g - 0.5, 0.0, [7.0, 30.0, 7.0], K_TOWER, M_WHITE, rng.u(), true);
        }
        b.n_buildings += 1;
    };
    for vl in VILLAGES {
        village(&world, &mut b, &mut rng, 0, vl, false);
    }
    village(&world, &mut b, &mut rng, 2, ECO_VILLAGE, true);

    // Farmsteads: silos and barns at some field corners in the Pastoral.
    for ci in -37..37 {
        for cj in 0..286 {
            let s = ci as f32 * FIELD_W;
            let v = cj as f32 * FIELD_L;
            let u = strip_u(0, s);
            let sm = sample(u, v);
            if sm.z[1] < 0.9 || h01(ci, cj, 31) > 0.07 || near_track(s, 20.0) {
                continue;
            }
            let g = world.height(u, v);
            b.put(0, s + 3.0, v + 3.0, g - 0.5, 0.0, [4.0, 14.0, 4.0], K_CYL, M_WHITE, rng.u(), true);
            b.put(0, s + 8.0, v + 3.0, g - 0.5, 0.0, [4.0, 12.0, 4.0], K_CYL, M_WHITE, rng.u(), true);
            b.put(0, s + 6.0, v + 14.0, g - 0.3, 0.0, [10.0, 8.0, 16.0], K_HOUSE, M_RESIDENTIAL, rng.u() | 1, true);
            b.n_buildings += 3;
        }
    }

    // ------------------------------------------------------------- trees ---
    let step = 7.5;
    for k in 0..3 {
        let mut v = 3.0;
        while v < L - 3.0 {
            let mut s = -hw + 16.0;
            while s < hw - 16.0 {
                let jx = rng.range(-3.3, 3.3);
                let jz = rng.range(-3.3, 3.3);
                let (ss, vv) = (s + jx, v + jz);
                s += step;
                if near_track(ss, 14.0) {
                    continue;
                }
                let u = strip_u(k, ss);
                let [grass, farm, city, forest] = world.zone_at(u, vv);
                // Keep roads, sidewalks and village streets clear (80 m grid,
                // 7 m half-width + sidewalk + crown radius).
                let rd = road_d(ss, vv);
                if (city > 0.15 && rd < 12.5) || (in_village(k, ss, vv) && rd < 15.0) {
                    continue;
                }
                if forest < 0.05 && grass < 0.5 {
                    continue;
                }
                let sm_h = world.height(u, vv);
                let park = city > 0.5 && grass > 0.6;
                let mut p = forest * 0.85;
                if park {
                    p += 0.06;
                } else if city < 0.3 && farm < 0.3 && grass > 0.5 {
                    p += 0.006;
                }
                if city > 0.3 && !park {
                    p = 0.0;
                }
                if farm > 0.4 || sm_h < 0.5 || rng.next() >= p {
                    continue;
                }
                let gl = world.height(u - 2.0, vv);
                let gr = world.height(u + 2.0, vv);
                let gd = world.height(u, vv - 2.0);
                let gu = world.height(u, vv + 2.0);
                let slope = ((gl - gr).powi(2) + (gd - gu).powi(2)).sqrt() / 4.0;
                if slope > 0.9 {
                    continue;
                }
                let m = mountain_factor(vv);
                let g = world.height(u, vv);
                let conifer = m > 0.05 || (k == 2 && rng.next() < 0.65) || (k == 0 && rng.next() < 0.15);
                let yaw = rng.range(0.0, TAU);
                if conifer {
                    let hgt = rng.range(11.0, 25.0) * (1.0 - m * 0.5);
                    let wd = hgt * rng.range(0.3, 0.42);
                    b.put(k, ss, vv, g - 0.4, yaw, [wd, hgt, wd], K_CONIFER, M_CONIFER, rng.u(), false);
                } else {
                    let hgt = rng.range(8.0, 16.0) * if park { 0.85 } else { 1.0 };
                    let wd = hgt * rng.range(0.6, 0.85);
                    b.put(k, ss, vv, g - 0.4, yaw, [wd, hgt, wd], K_BROADLEAF, M_BROADLEAF, rng.u(), false);
                }
                b.colliders.push(Collider { strip: k as u8, s: ss, v: vv, hs: 0.45, hv: 0.45, h0: g - 1.0, h1: g + 4.0 });
                b.n_trees += 1;
            }
            v += step;
        }
    }

    // ----------------------------------------------------------- teleports --
    let ground = |w: &World, k: usize, s: f32, v: f32| {
        let u = strip_u(k, s);
        surf_to_world(u, v, w.height(u, v).max(0.0) + 1.7)
    };
    world.teleports = vec![
        Teleport { name: "Пастораль — луг у реки", pos: ground(&world, 0, river_s(13000.0) + 70.0, 13000.0), fly: false },
        Teleport { name: "Пастораль — поля", pos: ground(&world, 0, -1500.0, 8250.0), fly: false },
        Teleport { name: "Мегаполис — проспект", pos: ground(&world, 1, 160.0, CENTER - 2600.0), fly: false },
        Teleport { name: "Мегаполис — набережная озера", pos: ground(&world, 1, 0.0, CENTER - 1720.0), fly: false },
        Teleport { name: "Мегаполис — над крышами", pos: surf_to_world(strip_u(1, 0.0), CENTER - 4600.0, 520.0), fly: true },
        Teleport { name: "Дикие земли — лес", pos: ground(&world, 2, -550.0, 10000.0), fly: false },
        Teleport { name: "Дикие земли — эко-деревня", pos: ground(&world, 2, ECO_VILLAGE.0 - 20.0, ECO_VILLAGE.1 - 90.0), fly: false },
        Teleport { name: "Горы у торца", pos: ground(&world, 2, 0.0, 2200.0), fly: false },
        Teleport { name: "Променад у окна", pos: ground(&world, 0, hw - 7.0, 15000.0), fly: false },
        Teleport { name: "Ось — невесомость", pos: Vec3::new(crate::physics::SPINDLE_R + 6.0, 4.0, CENTER), fly: false },
        Teleport { name: "Снаружи — вид на колонию", pos: Vec3::new(-26000.0, 17000.0, -21000.0), fly: true },
        Teleport { name: "Торец — вид на всю колонию", pos: Vec3::new(0.0, -1500.0, 700.0), fly: true },
    ];

    // Keep the arrival spots of ground teleports clear of trees.
    let spots: Vec<Vec3> = world.teleports.iter().filter(|t| !t.fly).map(|t| t.pos).collect();
    let is_tree = |st: u32| matches!(st & 0xff, K_CONIFER | K_BROADLEAF);
    b.objects.retain(|o| !is_tree(o.style) || spots.iter().all(|p| p.distance(Vec3::from(o.pos)) > 9.0));
    b.colliders.retain(|c| {
        c.hs != 0.45
            || spots.iter().all(|p| {
                let sf = world_to_surf(*p);
                (sf.x - strip_u(c.strip as usize, c.s)).abs() > 6.0 || (sf.y - c.v).abs() > 6.0
            })
    });
    world.objects = b.objects;
    world.colliders = b.colliders;
    world.n_buildings = b.n_buildings;
    world.n_trees = b.n_trees;
    world
}
