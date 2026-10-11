//! The colony world: heightmap, zone map, static objects, colliders and the
//! maglev tracks. Everything is generated procedurally at start-up.

mod gen;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::colony::{self, CIRC, L, R, STRIP_ANGLE};

pub const HW: usize = 4096; // heightmap texels around the circumference
pub const HH: usize = 4097; // heightmap texels along the axis (inclusive ends)
pub const ZW: usize = 4096; // zone map
pub const ZH: usize = 4096;

// Mesh kinds (index into the mesh table, see render::meshes).
pub const K_BOX: u32 = 0;
pub const K_TOWER: u32 = 1;
pub const K_DOME: u32 = 2;
pub const K_CONIFER: u32 = 3;
pub const K_BROADLEAF: u32 = 4;
pub const K_CYL: u32 = 5;
pub const K_TRAIN: u32 = 6;
pub const K_DECK: u32 = 7;
pub const K_HOUSE: u32 = 8;
pub const NUM_KINDS: usize = 9;

// Materials (must match objects.wgsl).
pub const M_GLASS: u32 = 0;
pub const M_WHITE: u32 = 1;
pub const M_RESIDENTIAL: u32 = 2;
pub const M_CONCRETE: u32 = 3;
pub const M_NEON: u32 = 4;
pub const M_CONIFER: u32 = 5;
pub const M_BROADLEAF: u32 = 6;
pub const M_TRAIN: u32 = 7;
pub const M_GREENHOUSE: u32 = 8;
pub const M_STATION: u32 = 9;
pub const M_STONE: u32 = 10;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct ObjInstance {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub size: [f32; 3],
    pub style: u32,
}

pub fn style(kind: u32, mat: u32, seed: u32) -> u32 {
    kind | (mat << 8) | ((seed & 0xffff) << 16)
}

impl ObjInstance {
    pub fn kind(&self) -> u32 {
        self.style & 0xff
    }
}

/// Axis-aligned box in surface coordinates (s across the valley, v along it,
/// h above the floor). Good enough for balls and the player.
#[derive(Clone, Copy, Debug)]
pub struct Collider {
    pub strip: u8,
    pub s: f32,
    pub v: f32,
    pub hs: f32,
    pub hv: f32,
    pub h0: f32,
    pub h1: f32,
}

pub struct Track {
    pub strip: usize,
    pub s: f32,
    pub v0: f32,
    pub v1: f32,
    /// Deck top height every `DECK_STEP` metres along v.
    pub deck: Vec<f32>,
    pub stations: Vec<f32>,
}

pub const DECK_STEP: f32 = 10.0;

impl Track {
    pub fn deck_at(&self, v: f32) -> f32 {
        let x = (v / DECK_STEP).clamp(0.0, (self.deck.len() - 1) as f32 - 0.001);
        let i = x as usize;
        let f = x - i as f32;
        self.deck[i] * (1.0 - f) + self.deck[i + 1] * f
    }
    pub fn u(&self) -> f32 {
        strip_u(self.strip, self.s)
    }
}

pub fn strip_u(strip: usize, s: f32) -> f32 {
    (strip as f32 * STRIP_ANGLE * R + s).rem_euclid(CIRC)
}

pub struct Teleport {
    pub name: &'static str,
    pub pos: Vec3,
    pub fly: bool,
}

const CELL: f32 = 64.0;
const GU: usize = (CIRC / CELL) as usize + 1;
const GV: usize = (L / CELL) as usize + 1;

pub struct World {
    pub height: Vec<f32>,
    pub zone: Vec<[u8; 4]>,
    pub objects: Vec<ObjInstance>,
    pub colliders: Vec<Collider>,
    grid: Vec<Vec<u32>>,
    pub tracks: Vec<Track>,
    pub teleports: Vec<Teleport>,
    pub n_buildings: usize,
    pub n_trees: usize,
}

impl World {
    pub fn generate() -> World {
        let t0 = std::time::Instant::now();
        let mut w = gen::generate();
        w.build_grid();
        log::info!(
            "world: {} objects ({} buildings, {} trees), {} colliders in {:.2}s",
            w.objects.len(),
            w.n_buildings,
            w.n_trees,
            w.colliders.len(),
            t0.elapsed().as_secs_f32()
        );
        w
    }

    fn build_grid(&mut self) {
        self.grid = vec![Vec::new(); GU * GV];
        for (i, c) in self.colliders.iter().enumerate() {
            let u = strip_u(c.strip as usize, c.s);
            let r = c.hs.max(c.hv);
            let u0 = ((u - r) / CELL).floor() as i32;
            let u1 = ((u + r) / CELL).floor() as i32;
            let v0 = (((c.v - r) / CELL).floor() as i32).max(0);
            let v1 = (((c.v + r) / CELL).floor() as i32).min(GV as i32 - 1);
            for cu in u0..=u1 {
                let cu = cu.rem_euclid(GU as i32) as usize;
                for cv in v0..=v1 {
                    self.grid[cv as usize * GU + cu].push(i as u32);
                }
            }
        }
    }

    /// Colliders overlapping a disc of `radius` around (u, v).
    pub fn colliders_near(&self, u: f32, v: f32, radius: f32, out: &mut Vec<u32>) {
        out.clear();
        let u0 = ((u - radius) / CELL).floor() as i32;
        let u1 = ((u + radius) / CELL).floor() as i32;
        let v0 = (((v - radius) / CELL).floor() as i32).max(0);
        let v1 = (((v + radius) / CELL).floor() as i32).min(GV as i32 - 1);
        for cu in u0..=u1 {
            let cu = cu.rem_euclid(GU as i32) as usize;
            for cv in v0..=v1 {
                for &i in &self.grid[cv as usize * GU + cu] {
                    if !out.contains(&i) {
                        out.push(i);
                    }
                }
            }
        }
    }

    /// Bilinear terrain height, identical to `sample_height` in common.wgsl.
    pub fn height(&self, u: f32, v: f32) -> f32 {
        let du = CIRC / HW as f32;
        let dv = L / (HH - 1) as f32;
        let gx = u / du;
        let gy = (v / dv).clamp(0.0, (HH - 1) as f32 - 0.001);
        let fx = gx.floor();
        let fy = gy.floor();
        let x0 = (fx as i64).rem_euclid(HW as i64) as usize;
        let x1 = (x0 + 1) % HW;
        let y0 = fy as usize;
        let (ax, ay) = (gx - fx, gy - fy);
        let h = |x: usize, y: usize| self.height[y * HW + x];
        let a = h(x0, y0) * (1.0 - ax) + h(x1, y0) * ax;
        let b = h(x0, y0 + 1) * (1.0 - ax) + h(x1, y0 + 1) * ax;
        a * (1.0 - ay) + b * ay
    }

    /// Terrain normal in world space.
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        let e = 1.5;
        let hl = self.height(u - e, v);
        let hr = self.height(u + e, v);
        let hd = self.height(u, v - e);
        let hu = self.height(u, v + e);
        let nl = Vec3::new(hl - hr, 2.0 * e, hd - hu).normalize();
        let th = u / R;
        colony::east_of(th) * nl.x + colony::up_of(th) * nl.y + Vec3::Z * nl.z
    }

    pub fn zone_at(&self, u: f32, v: f32) -> [f32; 4] {
        let x = ((u.rem_euclid(CIRC) / CIRC) * ZW as f32) as usize % ZW;
        let y = (((v / L) * ZH as f32) as usize).min(ZH - 1);
        let z = self.zone[y * ZW + x];
        [z[0] as f32 / 255.0, z[1] as f32 / 255.0, z[2] as f32 / 255.0, z[3] as f32 / 255.0]
    }

    /// Highest solid surface under (u, v) at or below height `h`: terrain or
    /// the top of a collider box (roofs, maglev deck, platforms).
    pub fn floor_below(&self, u: f32, v: f32, h: f32, scratch: &mut Vec<u32>) -> f32 {
        let mut best = self.height(u, v);
        let (k, s) = colony::strip_of(u.rem_euclid(CIRC) / R);
        self.colliders_near(u, v, 1.0, scratch);
        for &i in scratch.iter() {
            let c = &self.colliders[i as usize];
            if c.strip as usize != k {
                continue;
            }
            if (s - c.s).abs() < c.hs && (v - c.v).abs() < c.hv && c.h1 <= h + 0.6 && c.h1 > best {
                best = c.h1;
            }
        }
        best
    }

    pub fn kind_counts(&self) -> [u32; NUM_KINDS] {
        let mut c = [0u32; NUM_KINDS];
        for o in &self.objects {
            c[o.kind() as usize] += 1;
        }
        c
    }
}
