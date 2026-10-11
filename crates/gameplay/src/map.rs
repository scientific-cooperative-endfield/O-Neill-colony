//! The playable district: terrain, buildings, light structures and window
//! panes, in the station's flat surface coordinates (u = arc length, v = axis,
//! h = height) that roll up into the cylinder.
//!
//! Static geometry (`ArenaStatic`, shared through `Arc`) is separated from the
//! destructible state (`ArenaState`: box hit points, pane damage, breaches) so
//! that a rollback snapshot only copies the small dynamic part.
//!
//! Ray casts are done against short segments (one integrator sub-step, a few
//! metres). Over such a segment the curvilinear (u, v, h) coordinates of the
//! box frames are interpolated linearly; the error is ≤ ℓ²/8R (≈ 7 mm for
//! ℓ = 15 m) — well below the size of anything we hit.

use crate::det::Rng;
use crate::geom::{self, Surf, R};
use glam::DVec3;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Material {
    Terrain,
    Concrete,
    Steel,
    Wood,
    /// sheet metal / composite panels (kiosks, fences, car bodies)
    Panel,
    /// shop windows, bus shelters
    Glass,
    Hay,
    /// the station's own window glass (floor of the window strips)
    Window,
}

impl Material {
    /// (energy a bullet loses passing through it, J — `INF` = stops every round;
    ///  energy the structure absorbs before failing, J — `INF` = indestructible)
    pub fn props(self) -> (f64, f64) {
        match self {
            Material::Terrain | Material::Concrete | Material::Steel => (f64::INFINITY, f64::INFINITY),
            Material::Wood => (900.0, 2.5e4),
            Material::Panel => (250.0, 1.2e4),
            Material::Glass => (60.0, 1.5e3),
            Material::Hay => (1400.0, 2.0e4),
            Material::Window => (f64::INFINITY, 4.0e4),
        }
    }
    pub fn destructible(self) -> bool {
        self.props().1.is_finite()
    }
    /// Pieces of Voronoi debris when destroyed.
    pub fn debris_pieces(self) -> usize {
        match self {
            Material::Glass | Material::Window => 12,
            Material::Panel | Material::Wood => 8,
            Material::Hay => 6,
            _ => 0,
        }
    }
    pub fn density(self) -> f64 {
        match self {
            Material::Glass | Material::Window => 2500.0,
            Material::Panel => 300.0, // hollow panels: effective density
            Material::Wood => 550.0,
            Material::Hay => 120.0,
            Material::Steel => 7850.0,
            _ => 2400.0,
        }
    }
}

/// Axis-aligned box in surface coordinates (u absolute arc length).
#[derive(Clone, Copy, Debug)]
pub struct BoxCol {
    pub u: f64,
    pub v: f64,
    pub hu: f64,
    pub hv: f64,
    pub h0: f64,
    pub h1: f64,
    pub mat: Material,
}

impl BoxCol {
    pub fn centre(&self) -> DVec3 {
        geom::surf_to_world(self.u, self.v, 0.5 * (self.h0 + self.h1))
    }
    pub fn volume(&self) -> f64 {
        8.0 * self.hu * self.hv * 0.5 * (self.h1 - self.h0)
    }
}

/// What a ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HitTarget {
    Terrain,
    Box(u32),
    Pane(i32, i32),
    Spindle,
    EndCap,
}

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// fraction along the segment
    pub t: f64,
    pub point: DVec3,
    pub normal: DVec3,
    pub target: HitTarget,
    pub material: Material,
}

/// Support under a body.
#[derive(Clone, Copy, Debug)]
pub struct Ground {
    pub h: f64,
    /// world-space normal of the support surface
    pub normal: DVec3,
    pub material: Material,
    pub pane: Option<(i32, i32)>,
}

pub struct ArenaStatic {
    pub name: String,
    /// region: u ∈ [u0, u0 + size_u), v ∈ [v0, v0 + size_v)
    pub u0: f64,
    pub v0: f64,
    pub size_u: f64,
    pub size_v: f64,
    /// heightfield (nodes every `cell` m), terrain h
    pub cell: f64,
    pub nu: usize,
    pub nv: usize,
    pub heights: Vec<f32>,
    pub boxes: Vec<BoxCol>,
    /// pane size of the window glass (m)
    pub pane: f64,
    grid_cell: f64,
    gnu: usize,
    gnv: usize,
    grid: Vec<Vec<u32>>,
    pub spawns: Vec<(f64, f64)>,
    /// patrol way-points (u, v)
    pub patrol: Vec<(f64, f64)>,
    /// terrain below h = 0 on land is water (lakes / rivers of the generated world)
    pub water: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArenaState {
    /// absorbed energy of damaged boxes (index → J); destroyed boxes are in `destroyed`
    pub box_damage: BTreeMap<u32, f64>,
    pub destroyed: Vec<bool>,
    pub pane_damage: BTreeMap<(i32, i32), f64>,
    /// broken window panes: open holes to vacuum
    pub broken: BTreeMap<(i32, i32), u32>,
}

#[derive(Clone)]
pub struct Arena {
    pub st: Arc<ArenaStatic>,
    pub dynamic: ArenaState,
}

/// Is the arc-length position `u` over land (not a window strip)? (f64 port of `colony::is_land`)
pub fn is_land(u: f64) -> bool {
    let strip = std::f64::consts::TAU / 3.0;
    let th = u.rem_euclid(geom::CIRC) / R;
    let mut k = (th / strip).round();
    if k > 2.5 {
        k = 0.0;
    }
    let d = (th - k * strip) * R;
    d.abs() < std::f64::consts::FRAC_PI_6 * R
}

pub const WINDOW_THICKNESS: f64 = 0.12;

/// Floor surface: (height, material, window pane key).
pub type FloorHit = (f64, Material, Option<(i32, i32)>);

impl Arena {
    fn from_static(st: ArenaStatic) -> Arena {
        let n = st.boxes.len();
        Arena { st: Arc::new(st), dynamic: ArenaState { destroyed: vec![false; n], ..Default::default() } }
    }

    pub fn centre(&self) -> (f64, f64) {
        (self.st.u0 + 0.5 * self.st.size_u, self.st.v0 + 0.5 * self.st.size_v)
    }

    /// u unwrapped next to the arena centre.
    pub fn local_u(&self, u: f64) -> f64 {
        geom::unwrap_u(u, self.st.u0 + 0.5 * self.st.size_u)
    }

    pub fn surf(&self, p: DVec3) -> Surf {
        let mut s = geom::world_to_surf(p);
        s.u = self.local_u(s.u);
        s
    }

    pub fn in_region(&self, u: f64, v: f64) -> bool {
        let u = self.local_u(u);
        u >= self.st.u0 && u < self.st.u0 + self.st.size_u && v >= self.st.v0 && v < self.st.v0 + self.st.size_v
    }

    /// Terrain height (bilinear); 0 outside the region and over windows.
    pub fn terrain(&self, u: f64, v: f64) -> f64 {
        let s = &*self.st;
        let u = self.local_u(u);
        let x = (u - s.u0) / s.cell;
        let y = (v - s.v0) / s.cell;
        if x < 0.0 || y < 0.0 || x >= (s.nu - 1) as f64 || y >= (s.nv - 1) as f64 {
            return 0.0;
        }
        let (fx, fy) = (x.floor(), y.floor());
        let (i, j) = (fx as usize, fy as usize);
        let (ax, ay) = (x - fx, y - fy);
        let h = |i: usize, j: usize| s.heights[j * s.nu + i] as f64;
        let a = h(i, j) * (1.0 - ax) + h(i + 1, j) * ax;
        let b = h(i, j + 1) * (1.0 - ax) + h(i + 1, j + 1) * ax;
        a * (1.0 - ay) + b * ay
    }

    /// Terrain normal (world space).
    pub fn terrain_normal(&self, u: f64, v: f64) -> DVec3 {
        let e = self.st.cell;
        let dhu = (self.terrain(u + e, v) - self.terrain(u - e, v)) / (2.0 * e);
        let dhv = (self.terrain(u, v + e) - self.terrain(u, v - e)) / (2.0 * e);
        let p = geom::surf_to_world(u, v, 0.0);
        let (east, up, north) = geom::basis_at(p);
        (up - east * dhu - north * dhv).normalize()
    }

    pub fn pane_key(&self, u: f64, v: f64) -> (i32, i32) {
        let u = self.local_u(u);
        ((u / self.st.pane).floor() as i32, (v / self.st.pane).floor() as i32)
    }

    pub fn pane_centre(&self, k: (i32, i32)) -> (f64, f64) {
        ((k.0 as f64 + 0.5) * self.st.pane, (k.1 as f64 + 0.5) * self.st.pane)
    }

    pub fn pane_broken(&self, k: (i32, i32)) -> bool {
        self.dynamic.broken.contains_key(&k)
    }

    /// Floor surface (terrain or window glass) at (u, v); None over a hole.
    pub fn floor(&self, u: f64, v: f64) -> Option<FloorHit> {
        if is_land(u) {
            Some((self.terrain(u, v), Material::Terrain, None))
        } else {
            let k = self.pane_key(u, v);
            if self.pane_broken(k) {
                None
            } else {
                Some((0.0, Material::Window, Some(k)))
            }
        }
    }

    /// Water depth (m) at (u, v); 0 on dry land and glass.
    pub fn water_depth(&self, u: f64, v: f64) -> f64 {
        if self.st.water && is_land(u) {
            (-self.terrain(u, v)).max(0.0)
        } else {
            0.0
        }
    }

    pub fn box_alive(&self, i: u32) -> bool {
        !self.dynamic.destroyed[i as usize]
    }

    /// Indices of boxes whose footprint may overlap the disc (u, v, radius).
    pub fn boxes_near(&self, u: f64, v: f64, radius: f64, out: &mut Vec<u32>) {
        out.clear();
        let s = &*self.st;
        let u = self.local_u(u);
        let gx0 = ((u - radius - s.u0) / s.grid_cell).floor() as i64;
        let gx1 = ((u + radius - s.u0) / s.grid_cell).floor() as i64;
        let gy0 = ((v - radius - s.v0) / s.grid_cell).floor() as i64;
        let gy1 = ((v + radius - s.v0) / s.grid_cell).floor() as i64;
        for gy in gy0.max(0)..=gy1.min(s.gnv as i64 - 1) {
            for gx in gx0.max(0)..=gx1.min(s.gnu as i64 - 1) {
                for &i in &s.grid[gy as usize * s.gnu + gx as usize] {
                    if !self.dynamic.destroyed[i as usize] && !out.contains(&i) {
                        out.push(i);
                    }
                }
            }
        }
    }

    /// Highest support under (u, v) whose top is at most `top` (terrain, glass,
    /// box roofs). `radius` widens the box footprints (standing on an edge).
    pub fn ground(&self, u: f64, v: f64, top: f64, radius: f64, scratch: &mut Vec<u32>) -> Option<Ground> {
        let mut best: Option<Ground> = self.floor(u, v).map(|(h, m, pane)| {
            let n = if m == Material::Terrain { self.terrain_normal(u, v) } else { geom::up_at(geom::surf_to_world(u, v, 0.0)) };
            Ground { h, normal: n, material: m, pane }
        });
        let ul = self.local_u(u);
        self.boxes_near(ul, v, radius + 0.5, scratch);
        for &i in scratch.iter() {
            let b = &self.st.boxes[i as usize];
            if (ul - b.u).abs() < b.hu + radius * 0.5 && (v - b.v).abs() < b.hv + radius * 0.5 && b.h1 <= top && best.is_none_or(|g| b.h1 > g.h) {
                best = Some(Ground { h: b.h1, normal: geom::up_at(geom::surf_to_world(u, v, b.h1)), material: b.mat, pane: None });
            }
        }
        best
    }

    /// First hit of the segment p0 → p1 against static geometry.
    /// `skip_inside` ignores boxes containing p0 (a penetrating bullet leaving).
    pub fn raycast(&self, p0: DVec3, p1: DVec3, scratch: &mut Vec<u32>) -> Option<RayHit> {
        let s0 = self.surf(p0);
        let mut s1 = self.surf(p1);
        s1.u = geom::unwrap_u(s1.u, s0.u);
        let mut best: Option<RayHit> = None;
        let len = (p1 - p0).length();
        if len < 1e-9 {
            return None;
        }
        // --- floor (terrain / window glass) ---
        let floor_gap = |t: f64| -> (f64, Option<FloorHit>) {
            let p = p0 + (p1 - p0) * t;
            let s = self.surf(p);
            match self.floor(s.u, s.v) {
                Some(f) => {
                    // glass is a slab: below it is vacuum, so only its top surface counts
                    if f.1 == Material::Window && s.h < f.0 - WINDOW_THICKNESS {
                        (1.0, Some(f))
                    } else {
                        (s.h - f.0, Some(f))
                    }
                }
                None => (1.0, None),
            }
        };
        let n = ((len / 1.5).ceil() as usize).clamp(1, 64);
        let mut prev_t = 0.0;
        let (mut prev_g, _) = floor_gap(0.0);
        for k in 1..=n {
            let t = k as f64 / n as f64;
            let (g, _) = floor_gap(t);
            if prev_g >= 0.0 && g < 0.0 {
                let (mut a, mut b) = (prev_t, t);
                for _ in 0..30 {
                    let m = 0.5 * (a + b);
                    if floor_gap(m).0 >= 0.0 {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                let t = b;
                let p = p0 + (p1 - p0) * t;
                let s = self.surf(p);
                if let (_, Some((_, mat, pane))) = floor_gap(t) {
                    let normal = if mat == Material::Terrain { self.terrain_normal(s.u, s.v) } else { geom::up_at(p) };
                    let target = match pane {
                        Some(k) => HitTarget::Pane(k.0, k.1),
                        None => HitTarget::Terrain,
                    };
                    best = Some(RayHit { t, point: p, normal, target, material: mat });
                }
                break;
            }
            prev_t = t;
            prev_g = g;
        }
        // --- boxes (slab test in interpolated surface coordinates) ---
        let (umin, umax) = (s0.u.min(s1.u), s0.u.max(s1.u));
        let (vmin, vmax) = (s0.v.min(s1.v), s0.v.max(s1.v));
        let reach = 0.5 * ((umax - umin).max(vmax - vmin)) + 1.0;
        self.boxes_near(0.5 * (umin + umax), 0.5 * (vmin + vmax), reach, scratch);
        let o = [s0.u, s0.v, s0.h];
        let d = [s1.u - s0.u, s1.v - s0.v, s1.h - s0.h];
        for &i in scratch.iter() {
            let b = &self.st.boxes[i as usize];
            let bu = geom::unwrap_u(b.u, s0.u);
            let lo = [bu - b.hu, b.v - b.hv, b.h0];
            let hi = [bu + b.hu, b.v + b.hv, b.h1];
            let (mut t0, mut t1) = (0.0f64, best.map_or(1.0, |h| h.t));
            let mut axis = 3usize;
            let mut sign = 0.0;
            let mut inside = true;
            let mut ok = true;
            for a in 0..3 {
                if o[a] < lo[a] || o[a] > hi[a] {
                    inside = false;
                }
                if d[a].abs() < 1e-12 {
                    if o[a] < lo[a] || o[a] > hi[a] {
                        ok = false;
                        break;
                    }
                    continue;
                }
                let inv = 1.0 / d[a];
                let (mut ta, mut tb) = ((lo[a] - o[a]) * inv, (hi[a] - o[a]) * inv);
                let mut sg = -d[a].signum();
                if ta > tb {
                    std::mem::swap(&mut ta, &mut tb);
                }
                if ta > t0 {
                    t0 = ta;
                    axis = a;
                    sg = if d[a] > 0.0 { -1.0 } else { 1.0 };
                    sign = sg;
                }
                let _ = sg;
                t1 = t1.min(tb);
                if t0 > t1 {
                    ok = false;
                    break;
                }
            }
            if !ok || inside || axis == 3 {
                continue;
            }
            let p = p0 + (p1 - p0) * t0;
            let (east, up, north) = geom::basis_at(p);
            let normal = match axis {
                0 => east * sign,
                1 => north * sign,
                _ => up * sign,
            };
            best = Some(RayHit { t: t0, point: p, normal, target: HitTarget::Box(i), material: b.mat });
        }
        // --- spindle and end caps ---
        let rs = |p: DVec3| geom::r_perp(p) - geom::SPINDLE_R;
        if rs(p0) > 0.0 && rs(p1) <= 0.0 {
            let t = bisect(|t| rs(p0 + (p1 - p0) * t), 0.0, 1.0);
            if best.is_none_or(|h| t < h.t) {
                let p = p0 + (p1 - p0) * t;
                best = Some(RayHit { t, point: p, normal: -geom::up_at(p), target: HitTarget::Spindle, material: Material::Steel });
            }
        }
        for (z, nz) in [(0.0, 1.0), (geom::L, -1.0)] {
            let (a, b) = ((p0.z - z) * nz, (p1.z - z) * nz);
            if a > 0.0 && b <= 0.0 && geom::r_perp(p0) < R {
                let t = a / (a - b);
                if best.is_none_or(|h| t < h.t) {
                    best =
                        Some(RayHit { t, point: p0 + (p1 - p0) * t, normal: DVec3::new(0.0, 0.0, nz), target: HitTarget::EndCap, material: Material::Steel });
                }
            }
        }
        best
    }

    /// Line of sight between two points (static geometry only; glass is transparent).
    pub fn visible(&self, a: DVec3, b: DVec3, scratch: &mut Vec<u32>) -> bool {
        let d = b - a;
        let len = d.length();
        let steps = (len / 24.0).ceil().max(1.0) as usize;
        let mut p = a;
        for k in 1..=steps {
            let q = a + d * (k as f64 / steps as f64);
            if let Some(h) = self.raycast(p, q, scratch) {
                if !matches!(h.material, Material::Glass | Material::Window) {
                    return false;
                }
            }
            p = q;
        }
        true
    }

    /// Adds `energy` (J) to a destructible box; returns true when it fails now.
    pub fn damage_box(&mut self, i: u32, energy: f64) -> bool {
        let b = self.st.boxes[i as usize];
        let (_, hp) = b.mat.props();
        if !hp.is_finite() || self.dynamic.destroyed[i as usize] {
            return false;
        }
        let e = self.dynamic.box_damage.entry(i).or_insert(0.0);
        *e += energy;
        if *e >= hp {
            self.dynamic.box_damage.remove(&i);
            self.dynamic.destroyed[i as usize] = true;
            return true;
        }
        false
    }

    /// Adds `energy` (J) to a window pane; returns true when it breaks now.
    pub fn damage_pane(&mut self, k: (i32, i32), energy: f64, tick: u32) -> bool {
        if self.pane_broken(k) {
            return false;
        }
        let e = self.dynamic.pane_damage.entry(k).or_insert(0.0);
        *e += energy;
        if *e >= Material::Window.props().1 {
            self.dynamic.pane_damage.remove(&k);
            self.dynamic.broken.insert(k, tick);
            return true;
        }
        false
    }

    /// Total open area to vacuum (m²).
    pub fn breach_area(&self) -> f64 {
        self.dynamic.broken.len() as f64 * self.st.pane * self.st.pane
    }

    /// Air rushing to the breaches (m/s), a sum of hemispherical sinks with
    /// choked outflow Q = 0.58·c·A·(P/P₀) through each hole; capped at the
    /// choked speed near the hole.
    pub fn breach_wind(&self, p: DVec3, pressure_frac: f64, c_sound: f64) -> DVec3 {
        if self.dynamic.broken.is_empty() || pressure_frac <= 0.0 {
            return DVec3::ZERO;
        }
        let a = self.st.pane * self.st.pane;
        let q = 0.58 * c_sound * a;
        let mut w = DVec3::ZERO;
        for &k in self.dynamic.broken.keys() {
            let (u, v) = self.pane_centre(k);
            let c = geom::surf_to_world(u, v, 0.0);
            let d = c - p;
            let r2 = d.length_squared().max(0.25 * a);
            if r2 > 400.0 * 400.0 {
                continue;
            }
            let speed = (q / (std::f64::consts::TAU * r2)).min(0.58 * c_sound);
            w += d / r2.sqrt() * speed;
        }
        w
    }

    pub fn spawn_points(&self) -> &[(f64, f64)] {
        &self.st.spawns
    }

    // ----------------------------------------------------------- builders --

    /// The prototype district (~2 × 2 km) in the "Мегаполис" valley next to
    /// its window strip: city blocks, a field and ~200 m of window glass.
    pub fn prototype(seed: u64) -> Arena {
        let strip = std::f64::consts::TAU / 3.0 * R; // valley 1 centre (arc length)
        let half_land = std::f64::consts::FRAC_PI_6 * R; // 2094 m
        let (u0, v0, size) = (strip + 300.0, 15_000.0, 2000.0);
        let cell = 2.0;
        let n = (size / cell) as usize + 1;
        let mut rng = Rng::keyed(&[seed, 0xA7E4A]);
        let field_s = 1250.0; // across-valley offset where the city ends
                              // terrain: flat city, rolling field (value-noise fBm), glass is h = 0
        let mut heights = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let u = u0 + i as f64 * cell;
                let v = v0 + j as f64 * cell;
                let s = u - strip;
                let h = if s > field_s + 40.0 && s < half_land - 30.0 {
                    let edge = ((s - field_s - 40.0) / 80.0).min(1.0) * ((half_land - 30.0 - s) / 60.0).clamp(0.0, 1.0);
                    (fbm(u * 0.006, v * 0.006, seed) * 5.0 + fbm(u * 0.03, v * 0.03, seed ^ 7) * 0.6) * edge
                } else {
                    0.0
                };
                heights[j * n + i] = h as f32;
            }
        }
        let mut boxes: Vec<BoxCol> = Vec::new();
        let mut spawns = Vec::new();
        let mut patrol = Vec::new();
        let put = |boxes: &mut Vec<BoxCol>, u: f64, v: f64, hu: f64, hv: f64, h0: f64, h1: f64, mat: Material| {
            boxes.push(BoxCol { u, v, hu, hv, h0, h1, mat });
        };
        // --- city: 90 m blocks, 20 m streets ---
        let block = 90.0;
        let street = 20.0;
        let pitch = block + street;
        let nbu = ((field_s - 300.0) / pitch) as i32;
        let nbv = (size / pitch) as i32;
        for bj in 0..nbv {
            for bi in 0..nbu {
                let cu = u0 + street + bi as f64 * pitch + 0.5 * block;
                let cv = v0 + street + bj as f64 * pitch + 0.5 * block;
                // intersections: spawns / patrol points
                let iu = cu - 0.5 * block - 0.5 * street;
                let iv = cv - 0.5 * block - 0.5 * street;
                if (bi + bj) % 3 == 0 {
                    spawns.push((iu, iv));
                }
                patrol.push((iu, iv));
                let r = rng.f64();
                if r < 0.15 {
                    // plaza: low walls and kiosks
                    for _ in 0..6 {
                        let (du, dv) = (rng.range(-38.0, 38.0), rng.range(-38.0, 38.0));
                        if rng.f64() < 0.5 {
                            put(&mut boxes, cu + du, cv + dv, rng.range(1.5, 4.0), 0.3, 0.0, 1.1, Material::Concrete);
                        } else {
                            put(&mut boxes, cu + du, cv + dv, 1.5, 1.2, 0.0, 2.6, Material::Panel);
                        }
                    }
                    continue;
                }
                // 1–4 buildings per block
                let k = 1 + rng.index(4);
                let sub = if k == 1 { 1 } else { 2 };
                for q in 0..k.min(sub * sub) {
                    let (qx, qy) = ((q % sub) as f64, (q / sub) as f64);
                    let cell_w = block / sub as f64;
                    let bu = cu - 0.5 * block + (qx + 0.5) * cell_w;
                    let bv = cv - 0.5 * block + (qy + 0.5) * cell_w;
                    let hu = 0.5 * cell_w * rng.range(0.55, 0.9);
                    let hv = 0.5 * cell_w * rng.range(0.55, 0.9);
                    let tall = rng.f64();
                    let h = if tall < 0.1 { rng.range(60.0, 140.0) } else { rng.range(9.0, 40.0) };
                    put(&mut boxes, bu, bv, hu, hv, 0.0, h, Material::Concrete);
                    // shop-front glass on the street side
                    if rng.f64() < 0.5 {
                        put(&mut boxes, bu, bv - hv - 0.15, hu * 0.8, 0.1, 0.0, 3.2, Material::Glass);
                    }
                }
                // street furniture: barriers, cars, shelters along the street at the block's south edge
                for _ in 0..3 {
                    let su = cu + rng.range(-40.0, 40.0);
                    let sv = cv - 0.5 * block - rng.range(3.0, 17.0);
                    match rng.index(4) {
                        0 => put(&mut boxes, su, sv, 1.6, 0.3, 0.0, 1.05, Material::Concrete),
                        1 => put(&mut boxes, su, sv, 2.25, 0.9, 0.0, 1.45, Material::Panel),
                        2 => put(&mut boxes, su, sv, 2.0, 0.75, 0.0, 2.5, Material::Glass),
                        _ => put(&mut boxes, su, sv, 0.25, 0.25, 0.0, 4.5, Material::Steel),
                    }
                }
            }
        }
        // --- field: hay bales, tree rows, a barn, stone walls ---
        let field_u0 = strip + field_s + 40.0;
        let field_u1 = strip + half_land - 30.0;
        let terrain_at = |heights: &Vec<f32>, u: f64, v: f64| -> f64 {
            let i = (((u - u0) / cell).round() as usize).min(n - 1);
            let j = (((v - v0) / cell).round() as usize).min(n - 1);
            heights[j * n + i] as f64
        };
        for _ in 0..140 {
            let u = rng.range(field_u0, field_u1);
            let v = rng.range(v0 + 20.0, v0 + size - 20.0);
            let h = terrain_at(&heights, u, v) - 0.2;
            put(&mut boxes, u, v, 0.8, 0.8, h, h + 1.5, Material::Hay);
        }
        let mut v = v0 + 60.0;
        while v < v0 + size - 60.0 {
            let mut u = field_u0 + 30.0;
            while u < field_u1 - 30.0 {
                if rng.f64() < 0.35 {
                    let h = terrain_at(&heights, u, v) - 0.3;
                    put(&mut boxes, u, v, 0.25, 0.25, h, h + rng.range(5.0, 9.0), Material::Wood);
                }
                u += 9.0;
            }
            v += 160.0;
        }
        for b in 0..4 {
            let (u, v) = (field_u0 + 150.0 + b as f64 * 170.0, v0 + 300.0 + b as f64 * 420.0);
            let h = terrain_at(&heights, u, v) - 0.3;
            put(&mut boxes, u, v, 12.0, 7.0, h, h + 8.0, Material::Wood);
            spawns.push((u, v - 20.0));
            for w in 0..3 {
                let (wu, wv) = (u + rng.range(-120.0, 120.0), v + 60.0 + w as f64 * 50.0);
                let h = terrain_at(&heights, wu, wv) - 0.2;
                put(&mut boxes, wu, wv, rng.range(6.0, 20.0), 0.35, h, h + 1.2, Material::Concrete);
            }
        }
        for k in 0..12 {
            patrol.push((rng.range(field_u0, field_u1), v0 + 80.0 + k as f64 * 160.0));
        }
        // --- window strip: glass floor with longitudinal steel mullions every 40 m ---
        let w0 = strip + half_land;
        let mut u = w0 + 20.0;
        while u < u0 + size {
            put(&mut boxes, u, v0 + 0.5 * size, 0.6, 0.5 * size, 0.0, 0.9, Material::Steel);
            u += 40.0;
        }
        for k in 0..6 {
            patrol.push((w0 + 40.0 + (k % 2) as f64 * 80.0, v0 + 200.0 + k as f64 * 320.0));
        }
        spawns.push((w0 + 60.0, v0 + 500.0));
        spawns.push((w0 + 100.0, v0 + 1500.0));
        Arena::from_static(ArenaStatic::build("Мегаполис — квартал у окна".into(), u0, v0, size, size, cell, n, n, heights, boxes, spawns, patrol))
    }

    /// The same district cut out of the procedurally generated colony world
    /// (`oneill_world::world::World`), so the arena matches what is rendered.
    pub fn from_world(world: &oneill_world::world::World, u0: f64, v0: f64, size: f64) -> Arena {
        let cell = 2.0;
        let n = (size / cell) as usize + 1;
        let mut heights = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let u = u0 + i as f64 * cell;
                let v = v0 + j as f64 * cell;
                heights[j * n + i] = if is_land(u) { world.height(u.rem_euclid(geom::CIRC) as f32, v as f32) } else { 0.0 };
            }
        }
        let mut boxes = Vec::new();
        for c in &world.colliders {
            let u = geom::unwrap_u(oneill_world::world::strip_u(c.strip as usize, c.s) as f64, u0 + 0.5 * size);
            let v = c.v as f64;
            if u + (c.hs as f64) < u0 || u - (c.hs as f64) > u0 + size || v + (c.hv as f64) < v0 || v - (c.hv as f64) > v0 + size {
                continue;
            }
            let mat = if c.h1 - c.h0 < 2.0 { Material::Panel } else { Material::Concrete };
            boxes.push(BoxCol { u, v, hu: c.hs as f64, hv: c.hv as f64, h0: c.h0 as f64, h1: c.h1 as f64, mat });
        }
        let mut rng = Rng::keyed(&[0x5EED, boxes.len() as u64]);
        let mut spawns = Vec::new();
        let mut patrol = Vec::new();
        let mut tries = 0;
        while spawns.len() < 16 && tries < 10_000 {
            tries += 1;
            let (u, v) = (rng.range(u0 + 50.0, u0 + size - 50.0), rng.range(v0 + 50.0, v0 + size - 50.0));
            if boxes.iter().any(|b: &BoxCol| (u - b.u).abs() < b.hu + 2.0 && (v - b.v).abs() < b.hv + 2.0) {
                continue;
            }
            if is_land(u) && world.height(u.rem_euclid(geom::CIRC) as f32, v as f32) < 0.0 {
                continue; // no spawning in lakes
            }
            spawns.push((u, v));
            patrol.push((u, v));
        }
        let mut st = ArenaStatic::build("Колония — район".into(), u0, v0, size, size, cell, n, n, heights, boxes, spawns, patrol);
        st.water = true;
        Arena::from_static(st)
    }
}

impl ArenaStatic {
    #[allow(clippy::too_many_arguments)]
    fn build(
        name: String,
        u0: f64,
        v0: f64,
        size_u: f64,
        size_v: f64,
        cell: f64,
        nu: usize,
        nv: usize,
        heights: Vec<f32>,
        boxes: Vec<BoxCol>,
        spawns: Vec<(f64, f64)>,
        patrol: Vec<(f64, f64)>,
    ) -> ArenaStatic {
        let grid_cell = 16.0;
        let gnu = (size_u / grid_cell).ceil() as usize + 1;
        let gnv = (size_v / grid_cell).ceil() as usize + 1;
        let mut grid = vec![Vec::new(); gnu * gnv];
        for (i, b) in boxes.iter().enumerate() {
            let gx0 = (((b.u - b.hu - u0) / grid_cell).floor() as i64).max(0);
            let gx1 = (((b.u + b.hu - u0) / grid_cell).floor() as i64).min(gnu as i64 - 1);
            let gy0 = (((b.v - b.hv - v0) / grid_cell).floor() as i64).max(0);
            let gy1 = (((b.v + b.hv - v0) / grid_cell).floor() as i64).min(gnv as i64 - 1);
            for gy in gy0..=gy1 {
                for gx in gx0..=gx1 {
                    grid[gy as usize * gnu + gx as usize].push(i as u32);
                }
            }
        }
        ArenaStatic { name, u0, v0, size_u, size_v, cell, nu, nv, heights, boxes, pane: 5.0, grid_cell, gnu, gnv, grid, spawns, patrol, water: false }
    }
}

fn bisect(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    for _ in 0..40 {
        let m = 0.5 * (a + b);
        if f(m) > 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    b
}

fn hash2(x: i64, y: i64, seed: u64) -> f64 {
    let mut r = Rng::keyed(&[x as u64, y as u64, seed]);
    r.f64() * 2.0 - 1.0
}

fn vnoise(x: f64, y: f64, seed: u64) -> f64 {
    let (fx, fy) = (x.floor(), y.floor());
    let (ix, iy) = (fx as i64, fy as i64);
    let (tx, ty) = (x - fx, y - fy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let a = hash2(ix, iy, seed) * (1.0 - sx) + hash2(ix + 1, iy, seed) * sx;
    let b = hash2(ix, iy + 1, seed) * (1.0 - sx) + hash2(ix + 1, iy + 1, seed) * sx;
    a * (1.0 - sy) + b * sy
}

fn fbm(x: f64, y: f64, seed: u64) -> f64 {
    let mut s = 0.0;
    let mut a = 0.5;
    let mut f = 1.0;
    for o in 0..4 {
        s += a * vnoise(x * f, y * f, seed.wrapping_add(o));
        a *= 0.5;
        f *= 2.03;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prototype_district_layout() {
        let a = Arena::prototype(1);
        assert!(a.st.boxes.len() > 300, "{} boxes", a.st.boxes.len());
        assert!(a.st.spawns.len() >= 9);
        let (u0, v0) = (a.st.u0, a.st.v0);
        // city at the start of the region is land, the far edge is window glass
        assert!(is_land(u0 + 10.0));
        assert!(!is_land(u0 + 1990.0));
        let mut sc = Vec::new();
        let g = a.ground(u0 + 1990.0, v0 + 1000.3, 0.5, 0.3, &mut sc).unwrap();
        assert_eq!(g.material, Material::Window);
    }

    #[test]
    fn raycast_hits_floor_and_boxes() {
        let a = Arena::prototype(1);
        let mut sc = Vec::new();
        let b = a.st.boxes.iter().position(|b| b.mat == Material::Concrete && b.h1 > 20.0).unwrap();
        let bx = a.st.boxes[b];
        let p0 = geom::surf_to_world(bx.u - bx.hu - 5.0, bx.v, 1.5);
        let p1 = geom::surf_to_world(bx.u - bx.hu + 5.0, bx.v, 1.5);
        let h = a.raycast(p0, p1, &mut sc).unwrap();
        assert!(matches!(h.target, HitTarget::Box(_)));
        assert!((h.t - 0.5).abs() < 0.02, "t = {}", h.t);
        // straight down to the street
        let q0 = geom::surf_to_world(bx.u - bx.hu - 5.0, bx.v, 3.0);
        let q1 = geom::surf_to_world(bx.u - bx.hu - 5.0, bx.v, -3.0);
        let h = a.raycast(q0, q1, &mut sc).unwrap();
        assert_eq!(h.target, HitTarget::Terrain);
        assert!((h.t - 0.5).abs() < 1e-6);
    }

    #[test]
    fn window_pane_breaks_and_opens_a_hole() {
        let mut a = Arena::prototype(1);
        let (u, v) = (a.st.u0 + 1980.0, a.st.v0 + 1001.0);
        let k = a.pane_key(u, v);
        let mut broke = false;
        for _ in 0..100 {
            broke |= a.damage_pane(k, 1800.0, 5);
        }
        assert!(broke);
        assert!(a.floor(u, v).is_none());
        assert!(a.breach_area() > 20.0);
        let w = a.breach_wind(geom::surf_to_world(u + 10.0, v, 1.0), 1.0, 340.0);
        assert!(w.length() > 1.0);
    }
}
