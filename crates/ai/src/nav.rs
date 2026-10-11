//! Navigation grid on the cylinder floor.
//!
//! The floor of the colony is a cylinder; unrolled it is a flat rectangle in
//! (u = arc length, v = axial) coordinates, periodic in u with period 2πR.
//! Distances measured along the floor *are* arc lengths, so A* on a regular
//! grid in (u, v) is exact up to the terrain relief. If the grid spans the
//! whole circumference the u axis wraps around.
//!
//! Cells: blocked (wall, deep hole, steep slope, broken window), open,
//! glass floor (walkable but a single shot can open it to vacuum — costed ×3),
//! vault (waist-high obstacle: passable with JUMP). Cover flags mark open
//! cells next to an obstacle at least 1 m tall (crouch cover).

use gameplay::map::{Arena, Material};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub const BLOCKED: u8 = 0;
pub const F_GLASS: u8 = 1;
pub const F_VAULT: u8 = 2;
/// cover bits: bit k set = obstacle in direction k (0 = +u, then CCW in 45° steps)
pub const DIRS: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

/// Waypoint on the floor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Waypoint {
    pub u: f64,
    pub v: f64,
    /// a vault obstacle lies right before this waypoint: press JUMP
    pub vault: bool,
}

#[derive(Clone)]
pub struct NavGrid {
    pub u0: f64,
    pub v0: f64,
    pub cell: f64,
    pub nu: usize,
    pub nv: usize,
    pub wrap_u: bool,
    /// 0 = blocked, otherwise move cost multiplier
    pub cost: Vec<u8>,
    pub flags: Vec<u8>,
    pub cover: Vec<u8>,
    /// obstacle height above the floor (m) of blocked cells, used for cover
    pub top: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellKind {
    Blocked { top: f32 },
    Open,
    Glass,
    Vault,
}

/// Reusable A* buffers (generation-stamped, never cleared).
#[derive(Default, Clone)]
pub struct PathScratch {
    g: Vec<u32>,
    came: Vec<u32>,
    stamp: Vec<u32>,
    closed: Vec<u32>,
    gen: u32,
    heap: BinaryHeap<Reverse<(u32, u32)>>,
    pub expanded: u64,
}

const BODY_R: f64 = 0.4;
const MAX_SLOPE_COS: f64 = 0.766; // 40°

impl NavGrid {
    /// Grid over a custom area; `classify(u, v)` decides each cell (used by tests).
    pub fn from_fn(u0: f64, v0: f64, cell: f64, nu: usize, nv: usize, wrap_u: bool, mut classify: impl FnMut(f64, f64) -> CellKind) -> NavGrid {
        let n = nu * nv;
        let mut g = NavGrid { u0, v0, cell, nu, nv, wrap_u, cost: vec![0; n], flags: vec![0; n], cover: vec![0; n], top: vec![0.0; n] };
        for j in 0..nv {
            for i in 0..nu {
                let (u, v) = g.centre(i, j);
                g.set(j * nu + i, classify(u, v));
            }
        }
        g.rebuild_cover(0, 0, nu as i32, nv as i32);
        g
    }

    /// Grid over an arena (`cell` ≈ 4 m: half a street lane, 250 k cells for 2 km²).
    pub fn build(arena: &Arena, cell: f64, mantle: f64, step: f64) -> NavGrid {
        let s = &*arena.st;
        let nu = (s.size_u / cell).floor() as usize;
        let nv = (s.size_v / cell).floor() as usize;
        let wrap = s.size_u >= gameplay::geom::CIRC - cell;
        let mut scratch = Vec::new();
        NavGrid::from_fn(s.u0, s.v0, cell, nu, nv, wrap, |u, v| classify(arena, u, v, cell, mantle, step, &mut scratch))
    }

    fn set(&mut self, k: usize, c: CellKind) {
        let (cost, flags, top) = match c {
            CellKind::Blocked { top } => (BLOCKED, 0, top),
            CellKind::Open => (1, 0, 0.0),
            CellKind::Glass => (3, F_GLASS, 0.0),
            CellKind::Vault => (6, F_VAULT, 0.0),
        };
        self.cost[k] = cost;
        self.flags[k] = flags;
        self.top[k] = top;
    }

    pub fn centre(&self, i: usize, j: usize) -> (f64, f64) {
        (self.u0 + (i as f64 + 0.5) * self.cell, self.v0 + (j as f64 + 0.5) * self.cell)
    }

    /// Cell of a floor point (u must be in the arena's local unwrapped range,
    /// or anywhere if the grid wraps).
    pub fn cell_of(&self, u: f64, v: f64) -> Option<(usize, usize)> {
        let mut x = ((u - self.u0) / self.cell).floor() as i64;
        let y = ((v - self.v0) / self.cell).floor() as i64;
        if self.wrap_u {
            x = x.rem_euclid(self.nu as i64);
        }
        if x < 0 || y < 0 || x >= self.nu as i64 || y >= self.nv as i64 {
            return None;
        }
        Some((x as usize, y as usize))
    }

    #[inline]
    fn neighbour(&self, i: usize, j: usize, d: (i32, i32)) -> Option<usize> {
        let mut x = i as i64 + d.0 as i64;
        let y = j as i64 + d.1 as i64;
        if self.wrap_u {
            x = x.rem_euclid(self.nu as i64);
        }
        if x < 0 || y < 0 || x >= self.nu as i64 || y >= self.nv as i64 {
            return None;
        }
        Some(y as usize * self.nu + x as usize)
    }

    pub fn passable(&self, u: f64, v: f64) -> bool {
        self.cell_of(u, v).is_some_and(|(i, j)| self.cost[j * self.nu + i] != BLOCKED)
    }

    fn rebuild_cover(&mut self, i0: i32, j0: i32, i1: i32, j1: i32) {
        for j in j0.max(0)..j1.min(self.nv as i32) {
            for i in i0.max(0)..i1.min(self.nu as i32) {
                let k = j as usize * self.nu + i as usize;
                let mut bits = 0u8;
                if self.cost[k] != BLOCKED {
                    for (b, d) in DIRS.iter().enumerate() {
                        if let Some(n) = self.neighbour(i as usize, j as usize, *d) {
                            if self.cost[n] == BLOCKED && self.top[n] >= 1.0 {
                                bits |= 1 << b;
                            }
                        }
                    }
                }
                self.cover[k] = bits;
            }
        }
    }

    /// Re-classifies the cells overlapping the floor rectangle (centre, half sizes).
    pub fn refresh_area(&mut self, arena: &Arena, u: f64, v: f64, hu: f64, hv: f64, mantle: f64, step: f64) {
        let u = arena.local_u(u);
        let m = self.cell + BODY_R;
        let i0 = ((u - hu - m - self.u0) / self.cell).floor() as i32;
        let i1 = ((u + hu + m - self.u0) / self.cell).ceil() as i32;
        let j0 = ((v - hv - m - self.v0) / self.cell).floor() as i32;
        let j1 = ((v + hv + m - self.v0) / self.cell).ceil() as i32;
        let mut scratch = Vec::new();
        for j in j0.max(0)..j1.min(self.nv as i32) {
            for i in i0.max(0)..i1.min(self.nu as i32) {
                let (cu, cv) = self.centre(i as usize, j as usize);
                let c = classify(arena, cu, cv, self.cell, mantle, step, &mut scratch);
                self.set(j as usize * self.nu + i as usize, c);
            }
        }
        self.rebuild_cover(i0 - 1, j0 - 1, i1 + 1, j1 + 1);
    }

    pub fn on_box_destroyed(&mut self, arena: &Arena, index: u32, mantle: f64, step: f64) {
        let b = arena.st.boxes[index as usize];
        self.refresh_area(arena, b.u, b.v, b.hu, b.hv, mantle, step);
    }

    pub fn on_pane_broken(&mut self, arena: &Arena, key: (i32, i32), mantle: f64, step: f64) {
        let (u, v) = arena.pane_centre(key);
        let h = 0.5 * arena.st.pane;
        self.refresh_area(arena, u, v, h, h, mantle, step);
    }

    /// Nearest passable cell centre to (u, v) within `rings` cells.
    pub fn nearest_open(&self, u: f64, v: f64, rings: i32) -> Option<(usize, usize)> {
        let (ci, cj) = self.cell_of(u, v).or_else(|| {
            let i = (((u - self.u0) / self.cell).floor() as i64).clamp(0, self.nu as i64 - 1);
            let j = (((v - self.v0) / self.cell).floor() as i64).clamp(0, self.nv as i64 - 1);
            Some((i as usize, j as usize))
        })?;
        if self.cost[cj * self.nu + ci] != BLOCKED {
            return Some((ci, cj));
        }
        for r in 1..=rings {
            let mut best: Option<(i64, usize, usize)> = None;
            for dj in -r..=r {
                for di in -r..=r {
                    if di.abs() != r && dj.abs() != r {
                        continue;
                    }
                    if let Some(k) = self.neighbour(ci, cj, (di, dj)) {
                        if self.cost[k] != BLOCKED {
                            let d2 = (di * di + dj * dj) as i64;
                            if best.is_none_or(|b| d2 < b.0) {
                                best = Some((d2, k % self.nu, k / self.nu));
                            }
                        }
                    }
                }
            }
            if let Some((_, i, j)) = best {
                return Some((i, j));
            }
        }
        None
    }

    /// Wrapped cell distance in u.
    #[inline]
    fn du(&self, a: usize, b: usize) -> u32 {
        let d = (a as i64 - b as i64).unsigned_abs() as usize;
        (if self.wrap_u { d.min(self.nu - d) } else { d }) as u32
    }

    /// A* (octile metric, integer costs: 10 straight, 14 diagonal, × cell cost).
    /// `budget` limits node expansions; when exhausted the path to the explored
    /// node closest to the goal is returned (good enough to keep moving).
    pub fn find_path(&self, sc: &mut PathScratch, from: (f64, f64), to: (f64, f64), budget: usize) -> Option<Vec<Waypoint>> {
        let (si, sj) = self.nearest_open(from.0, from.1, 4)?;
        let (gi, gj) = self.nearest_open(to.0, to.1, 8)?;
        let n = self.nu * self.nv;
        if sc.g.len() != n {
            sc.g = vec![0; n];
            sc.came = vec![u32::MAX; n];
            sc.stamp = vec![0; n];
            sc.closed = vec![0; n];
            sc.gen = 0;
        }
        sc.gen = sc.gen.wrapping_add(1);
        if sc.gen == 0 {
            sc.stamp.iter_mut().for_each(|s| *s = 0);
            sc.closed.iter_mut().for_each(|s| *s = 0);
            sc.gen = 1;
        }
        let gen = sc.gen;
        let nu = self.nu;
        let start = sj * nu + si;
        let goal = gj * nu + gi;
        let h = |k: usize| -> u32 {
            let (i, j) = (k % nu, k / nu);
            let dx = self.du(i, gi);
            let dy = (j as i64 - gj as i64).unsigned_abs() as u32;
            10 * dx.max(dy) + 4 * dx.min(dy)
        };
        sc.heap.clear();
        sc.g[start] = 0;
        sc.came[start] = u32::MAX;
        sc.stamp[start] = gen;
        sc.heap.push(Reverse((h(start), start as u32)));
        let mut best = (h(start), start);
        let mut expanded = 0usize;
        let mut reached = false;
        while let Some(Reverse((_, k))) = sc.heap.pop() {
            let k = k as usize;
            if sc.closed[k] == gen {
                continue;
            }
            sc.closed[k] = gen;
            if k == goal {
                reached = true;
                break;
            }
            let hk = h(k);
            if hk < best.0 {
                best = (hk, k);
            }
            expanded += 1;
            if expanded > budget {
                break;
            }
            let (i, j) = (k % nu, k / nu);
            for (di, d) in DIRS.iter().enumerate() {
                let Some(m) = self.neighbour(i, j, *d) else { continue };
                let c = self.cost[m];
                if c == BLOCKED || sc.closed[m] == gen {
                    continue;
                }
                let diag = di % 2 == 1;
                if diag {
                    // no corner cutting
                    let a = self.neighbour(i, j, (d.0, 0));
                    let b = self.neighbour(i, j, (0, d.1));
                    if a.is_none_or(|a| self.cost[a] == BLOCKED) || b.is_none_or(|b| self.cost[b] == BLOCKED) {
                        continue;
                    }
                }
                let step = if diag { 14 } else { 10 } * c as u32;
                let ng = sc.g[k] + step;
                if sc.stamp[m] != gen || ng < sc.g[m] {
                    sc.stamp[m] = gen;
                    sc.g[m] = ng;
                    sc.came[m] = k as u32;
                    sc.heap.push(Reverse((ng + h(m), m as u32)));
                }
            }
        }
        sc.expanded += expanded as u64;
        let end = if reached { goal } else { best.1 };
        let mut cells = vec![end];
        let mut k = end;
        while sc.came[k] != u32::MAX && k != start {
            k = sc.came[k] as usize;
            cells.push(k);
        }
        cells.reverse();
        Some(self.smooth(&cells, if reached { Some(to) } else { None }))
    }

    /// Cells crossed by the floor segment a → b (a body-width corridor) are
    /// open: no wall, no hole, no vault obstacle.
    pub fn walkable_line(&self, a: (f64, f64), b: (f64, f64)) -> bool {
        self.line_ok(a, b, None)
    }

    fn line_ok(&self, a: (f64, f64), b: (f64, f64), skip: Option<usize>) -> bool {
        let (du, dv) = (b.0 - a.0, b.1 - a.1);
        let len = (du * du + dv * dv).sqrt();
        let steps = (len / (0.35 * self.cell)).ceil().max(1.0) as usize;
        let (ou, ov) = if len > 1e-9 { (-dv / len * BODY_R, du / len * BODY_R) } else { (0.0, 0.0) };
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let (u, v) = (a.0 + du * t, a.1 + dv * t);
            for sign in [-1.0, 1.0] {
                match self.cell_of(u + ou * sign, v + ov * sign) {
                    Some((i, j)) => {
                        let k = j * self.nu + i;
                        if self.cost[k] == BLOCKED || (self.flags[k] & F_VAULT != 0 && skip != Some(k)) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
        }
        true
    }

    /// String pulling: keep only the corners needed to stay on walkable lines.
    /// Vault cells always become waypoints (the bot has to jump there).
    fn smooth(&self, cells: &[usize], exact_goal: Option<(f64, f64)>) -> Vec<Waypoint> {
        let period = self.nu as f64 * self.cell;
        let mut pts: Vec<(f64, f64, usize)> = Vec::with_capacity(cells.len());
        for &k in cells {
            let (mut u, v) = self.centre(k % self.nu, k / self.nu);
            if self.wrap_u {
                if let Some(&(pu, _, _)) = pts.last() {
                    u = pu + (u - pu + 0.5 * period).rem_euclid(period) - 0.5 * period;
                }
            }
            pts.push((u, v, k));
        }
        if let (Some(g), Some(last)) = (exact_goal, pts.last_mut()) {
            if self.passable(g.0, g.1) {
                let gu = if self.wrap_u { last.0 + (g.0 - last.0 + 0.5 * period).rem_euclid(period) - 0.5 * period } else { g.0 };
                *last = (gu, g.1, last.2);
            }
        }
        let vault = |k: usize| self.flags[k] & F_VAULT != 0;
        let wp = |p: &(f64, f64, usize)| Waypoint { u: p.0, v: p.1, vault: vault(p.2) };
        let mut out = Vec::new();
        let mut anchor = 0usize;
        let mut i = 1usize;
        while i < pts.len() {
            if vault(pts[i].2) {
                if i - 1 > anchor {
                    out.push(wp(&pts[i - 1]));
                }
                out.push(wp(&pts[i]));
                anchor = i;
                i += 1;
                continue;
            }
            let (a, b) = (&pts[anchor], &pts[i]);
            if self.line_ok((a.0, a.1), (b.0, b.1), Some(a.2)) {
                i += 1;
            } else if i - 1 > anchor {
                out.push(wp(&pts[i - 1]));
                anchor = i - 1;
            } else {
                out.push(wp(&pts[i]));
                anchor = i;
                i += 1;
            }
        }
        if anchor + 1 < pts.len() || out.is_empty() {
            if let Some(p) = pts.last() {
                out.push(wp(p));
            }
        }
        out
    }

    /// Open cells with crouch cover against a threat at (tu, tv), within
    /// `radius` m of (u, v): the obstacle side faces the threat (±45°).
    pub fn cover_near(&self, u: f64, v: f64, radius: f64, threat: (f64, f64), out: &mut Vec<(f64, f64)>) {
        out.clear();
        let Some((ci, cj)) = self.cell_of(u, v) else { return };
        let r = (radius / self.cell).ceil() as i32;
        for dj in -r..=r {
            for di in -r..=r {
                let Some(k) = self.neighbour(ci, cj, (di, dj)) else { continue };
                let bits = self.cover[k];
                if bits == 0 {
                    continue;
                }
                let (mut cu, cv) = self.centre(k % self.nu, k / self.nu);
                if self.wrap_u {
                    let period = self.nu as f64 * self.cell;
                    cu = u + (cu - u + 0.5 * period).rem_euclid(period) - 0.5 * period;
                }
                let (tdu, tdv) = (threat.0 - cu, threat.1 - cv);
                let tl = (tdu * tdu + tdv * tdv).sqrt().max(1e-6);
                for (b, d) in DIRS.iter().enumerate() {
                    if bits & (1 << b) == 0 {
                        continue;
                    }
                    let dl = ((d.0 * d.0 + d.1 * d.1) as f64).sqrt();
                    let c = (d.0 as f64 * tdu + d.1 as f64 * tdv) / (dl * tl);
                    if c > 0.7 {
                        out.push((cu, cv));
                        break;
                    }
                }
            }
        }
    }

    pub fn open_fraction(&self) -> f64 {
        self.cost.iter().filter(|&&c| c != BLOCKED).count() as f64 / self.cost.len() as f64
    }
}

/// Classifies the floor cell centred at (u, v).
pub fn classify(arena: &Arena, u: f64, v: f64, cell: f64, mantle: f64, step: f64, scratch: &mut Vec<u32>) -> CellKind {
    let Some((floor_h, mat, _)) = arena.floor(u, v) else { return CellKind::Blocked { top: 0.0 } };
    // a window pane next to a broken one is still walkable; the hole itself is not
    if mat == Material::Terrain && arena.terrain_normal(u, v).dot(gameplay::geom::up_at(gameplay::geom::surf_to_world(u, v, 0.0))) < MAX_SLOPE_COS {
        return CellKind::Blocked { top: 0.0 };
    }
    if arena.water_depth(u, v) > 0.6 {
        return CellKind::Blocked { top: 0.0 };
    }
    let half = 0.25 * cell + BODY_R;
    arena.boxes_near(u, v, half + 1.0, scratch);
    let ul = arena.local_u(u);
    let mut worst: Option<f64> = None;
    for &i in scratch.iter() {
        let b = &arena.st.boxes[i as usize];
        if (ul - b.u).abs() >= b.hu + half || (v - b.v).abs() >= b.hv + half {
            continue;
        }
        // overlaps the body height range above this floor?
        if b.h1 <= floor_h + step || b.h0 >= floor_h + 1.8 {
            continue;
        }
        let rise = b.h1 - floor_h;
        worst = Some(worst.map_or(rise, |w: f64| w.max(rise)));
    }
    match worst {
        Some(r) if r > mantle => CellKind::Blocked { top: r as f32 },
        Some(_) => CellKind::Vault,
        None if mat == Material::Window => CellKind::Glass,
        None => CellKind::Open,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall_grid(wrap: bool) -> NavGrid {
        // 100 × 60 cells of 2 m; a wall at u ∈ [80, 84) for v < 100 m
        NavGrid::from_fn(
            0.0,
            0.0,
            2.0,
            100,
            60,
            wrap,
            |u, v| if (80.0..84.0).contains(&u) && v < 100.0 { CellKind::Blocked { top: 3.0 } } else { CellKind::Open },
        )
    }

    #[test]
    fn path_goes_around_the_wall() {
        let g = wall_grid(false);
        let mut sc = PathScratch::default();
        let p = g.find_path(&mut sc, (40.0, 20.0), (120.0, 20.0), 100_000).expect("path");
        let last = p.last().unwrap();
        assert!((last.u - 120.0).abs() < 1e-9 && (last.v - 20.0).abs() < 1e-9);
        // must pass north of the wall end (v ≥ 100)
        assert!(p.iter().any(|w| w.v >= 99.0), "{p:?}");
        // every leg is walkable
        let mut a = (40.0, 20.0);
        for w in &p {
            assert!(g.walkable_line(a, (w.u, w.v)) || w.vault, "{a:?} -> {w:?}");
            a = (w.u, w.v);
        }
    }

    #[test]
    fn wrapping_grid_takes_the_short_way_round() {
        // a wall in the middle; start and goal near opposite ends of u:
        // with wrap the path crosses the seam instead of walking around the wall
        let g = wall_grid(true);
        let mut sc = PathScratch::default();
        let p = g.find_path(&mut sc, (6.0, 20.0), (194.0, 20.0), 100_000).expect("path");
        let len: f64 = {
            let mut a = (6.0, 20.0);
            let mut s = 0.0;
            for w in &p {
                s += ((w.u - a.0).powi(2) + (w.v - a.1).powi(2)).sqrt();
                a = (w.u, w.v);
            }
            s
        };
        assert!(len < 20.0, "len {len} path {p:?}");
        assert!(p.iter().all(|w| w.v < 40.0));
        let ng = wall_grid(false);
        let p2 = ng.find_path(&mut sc, (6.0, 20.0), (194.0, 20.0), 100_000).unwrap();
        assert!(p2.iter().any(|w| w.v >= 99.0), "without wrap it must go round");
    }

    #[test]
    fn prototype_district_is_mostly_walkable_and_connected() {
        let a = gameplay::Arena::prototype(3);
        let g = NavGrid::build(&a, 4.0, 1.15, 0.45);
        let f = g.open_fraction();
        assert!(f > 0.5 && f < 0.98, "open fraction {f}");
        let mut sc = PathScratch::default();
        let sp = a.spawn_points();
        let (u0, v0) = sp[0];
        let mut ok = 0;
        for &(u, v) in sp.iter().skip(1).take(6) {
            if let Some(p) = g.find_path(&mut sc, (u0, v0), (u, v), 200_000) {
                let l = p.last().unwrap();
                if ((l.u - u).powi(2) + (l.v - v).powi(2)).sqrt() < 6.0 {
                    ok += 1;
                }
            }
        }
        assert!(ok >= 5, "only {ok} of 6 spawns reachable");
    }

    #[test]
    fn broken_window_blocks_the_cell() {
        let mut a = gameplay::Arena::prototype(3);
        let mut g = NavGrid::build(&a, 4.0, 1.15, 0.45);
        let strip = std::f64::consts::TAU / 3.0 * gameplay::geom::R;
        let (u, v) = (strip + std::f64::consts::FRAC_PI_6 * gameplay::geom::R + 30.0, 15_500.0);
        assert!(g.passable(u, v));
        let k = a.pane_key(u, v);
        while !a.damage_pane(k, 1e9, 1) {}
        g.on_pane_broken(&a, k, 1.15, 0.45);
        assert!(!g.passable(u, v));
    }
}
