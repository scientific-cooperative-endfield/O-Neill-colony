//! Render-side effects for the combat mode: muzzle flashes, gun smoke, sparks,
//! dust, suit-breach vapour, glass glitter, jetpack plumes, shock rings, spent
//! casings bouncing on the floor and short-lived point lights. Purely visual —
//! nothing here feeds back into the deterministic simulation.

use std::cell::Cell;

use glam::{Quat, Vec3};

use crate::render::props::{ParticleGpu, PropInst};

pub const SMOKE: u32 = 0;
pub const GLOW: u32 = 1;
pub const SPARK: u32 = 2;
pub const STAR: u32 = 3;
pub const RING: u32 = 4;

#[derive(Clone, Copy)]
struct P {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    size: [f32; 2],
    color: [f32; 4],
    kind: u32,
    drag: f32,
    /// fraction of local gravity (negative: buoyant)
    grav: f32,
    streak: f32,
}

struct Casing {
    pos: Vec3,
    vel: Vec3,
    rot: Quat,
    spin: Vec3,
    age: f32,
    floor: f32,
    rest: bool,
}

pub struct Fx {
    parts: Vec<P>,
    casings: Vec<Casing>,
    lights: Vec<(Vec3, f32, f32, f32)>,
    rng: Cell<u32>,
    pub quality: f32,
}

fn up_at(p: Vec3) -> Vec3 {
    let r = (p.x * p.x + p.y * p.y).sqrt().max(1e-3);
    Vec3::new(-p.x / r, -p.y / r, 0.0)
}

/// local centrifugal "gravity" magnitude (1 g at the floor, ∝ radius)
fn g_at(p: Vec3) -> f32 {
    9.81 * (p.x * p.x + p.y * p.y).sqrt() / gameplay::geom::R as f32
}

impl Default for Fx {
    fn default() -> Self {
        Fx { parts: Vec::new(), casings: Vec::new(), lights: Vec::new(), rng: Cell::new(0x9e37_79b9), quality: 1.0 }
    }
}

impl Fx {
    fn rnd(&self) -> f32 {
        let mut r = self.rng.get();
        r ^= r << 13;
        r ^= r >> 17;
        r ^= r << 5;
        self.rng.set(r);
        (r >> 8) as f32 / 16_777_216.0
    }
    fn sym(&self) -> f32 {
        self.rnd() * 2.0 - 1.0
    }
    fn dir(&self) -> Vec3 {
        loop {
            let v = Vec3::new(self.sym(), self.sym(), self.sym());
            let l = v.length_squared();
            if l > 1e-4 && l <= 1.0 {
                return v / l.sqrt();
            }
        }
    }
    fn cone(&self, axis: Vec3, spread: f32) -> Vec3 {
        (axis.normalize_or(Vec3::Y) + self.dir() * spread).normalize_or(axis)
    }
    fn count(&self, n: f32) -> usize {
        (n * self.quality).round().max(1.0) as usize
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, kind: u32, pos: Vec3, vel: Vec3, life: f32, size: [f32; 2], color: [f32; 4], drag: f32, grav: f32) {
        if self.parts.len() < 6000 {
            self.parts.push(P { pos, vel, age: 0.0, life, size, color, kind, drag, grav, streak: 0.03 });
        }
    }

    pub fn light(&mut self, pos: Vec3, intensity: f32, life: f32) {
        self.lights.push((pos, intensity, life, life));
    }

    pub fn muzzle(&mut self, pos: Vec3, dir: Vec3, weapon: usize, own: bool) {
        let big = [0.7, 1.0, 1.6][weapon.min(2)];
        let s = if own { 0.75 } else { 1.0 } * big;
        self.emit(STAR, pos + dir * 0.03, Vec3::ZERO, 0.05, [0.13 * s, 0.2 * s], [9.0, 5.5, 2.2, 1.0], 0.0, 0.0);
        self.emit(GLOW, pos + dir * 0.08 * s, dir * 2.0, 0.06, [0.09 * s, 0.16 * s], [6.0, 3.0, 1.0, 1.0], 0.0, 0.0);
        // side jets of the brake
        let side = dir.any_orthonormal_vector();
        for k in [-1.0, 1.0] {
            self.emit(GLOW, pos + side * (0.04 * k), side * (6.0 * k) + dir * 2.0, 0.04, [0.04 * s, 0.09 * s], [5.0, 2.5, 0.8, 1.0], 6.0, 0.0);
        }
        for _ in 0..self.count(3.0 * big) {
            let v = self.cone(dir, 0.35) * (1.0 + 2.5 * self.rnd()) + self.dir() * 0.25;
            let life = 0.9 + 0.9 * self.rnd();
            let g = 0.62 + 0.1 * self.rnd();
            self.emit(SMOKE, pos + dir * 0.05, v, life, [0.04 * s, 0.45 * s], [g, g, g * 1.02, 0.22], 3.0, -0.03);
        }
        for _ in 0..self.count(3.0) {
            let v = self.cone(dir, 0.25) * (12.0 + 18.0 * self.rnd());
            self.emit(SPARK, pos, v, 0.05 + 0.05 * self.rnd(), [0.004, 0.004], [6.0, 3.5, 1.2, 1.0], 2.0, 0.2);
        }
        self.light(pos + dir * 0.2, -9.0 * big, 0.06);
    }

    pub fn eject(&mut self, pos: Vec3, right: Vec3, up: Vec3, fwd: Vec3, floor_h: f32) {
        if self.casings.len() > 160 {
            self.casings.remove(0);
        }
        let vel = right * (1.6 + 1.2 * self.rnd()) + up * (1.4 + 0.8 * self.rnd()) - fwd * (0.3 * self.rnd());
        let spin = self.dir() * 25.0;
        let rot = Quat::from_rotation_arc(Vec3::Y, right);
        self.casings.push(Casing { pos, vel, rot, spin, age: 0.0, floor: floor_h, rest: false });
        // a wisp from the ejection port
        self.emit(SMOKE, pos, up * 0.4 + right * 0.3, 0.6, [0.02, 0.14], [0.7, 0.7, 0.72, 0.12], 2.0, -0.05);
    }

    /// Bullet striking something: dust puff (+ sparks on hard / metal surfaces).
    pub fn impact(&mut self, pos: Vec3, normal: Vec3, metal: bool) {
        let n = normal.normalize_or(up_at(pos));
        let dust = [0.5, 0.47, 0.42];
        for _ in 0..self.count(4.0) {
            let v = self.cone(n, 0.7) * (0.6 + 2.0 * self.rnd());
            let k = 0.85 + 0.3 * self.rnd();
            self.emit(
                SMOKE,
                pos + n * 0.03,
                v,
                0.8 + 1.0 * self.rnd(),
                [0.04, 0.35 + 0.2 * self.rnd()],
                [dust[0] * k, dust[1] * k, dust[2] * k, 0.35],
                4.0,
                0.05,
            );
        }
        for _ in 0..self.count(if metal { 12.0 } else { 6.0 }) {
            let v = self.cone(n, 0.9) * (3.0 + 9.0 * self.rnd());
            let c = if metal { [7.0, 4.0, 1.5, 1.0] } else { [3.0, 2.4, 1.6, 1.0] };
            self.emit(SPARK, pos, v, 0.15 + 0.35 * self.rnd(), [0.006, 0.006], c, 1.0, 1.0);
        }
        // chips falling
        for _ in 0..self.count(4.0) {
            let v = self.cone(n, 0.8) * (1.0 + 3.0 * self.rnd());
            self.emit(GLOW, pos, v, 0.5 + 0.5 * self.rnd(), [0.012, 0.01], [0.25, 0.23, 0.2, 1.0], 0.5, 1.0);
        }
        self.emit(GLOW, pos + n * 0.02, Vec3::ZERO, 0.05, [0.06, 0.1], [4.0, 3.0, 2.0, 1.0], 0.0, 0.0);
    }

    /// A round through a pressure suit: white vapour jet + a few sparks.
    pub fn breach(&mut self, pos: Vec3, dir: Vec3, heavy: bool) {
        let m = if heavy { 2.0 } else { 1.0 };
        for _ in 0..self.count(6.0 * m) {
            let v = self.cone(dir, 0.5) * (1.5 + 4.0 * self.rnd());
            let w = 0.9 + 0.1 * self.rnd();
            self.emit(SMOKE, pos, v, 0.5 + 0.8 * self.rnd(), [0.03, 0.3 * m], [w, w, w, 0.4], 3.5, -0.02);
        }
        for _ in 0..self.count(8.0) {
            let v = self.cone(dir, 0.8) * (3.0 + 6.0 * self.rnd());
            self.emit(SPARK, pos, v, 0.1 + 0.25 * self.rnd(), [0.005, 0.005], [6.0, 2.5, 1.0, 1.0], 1.0, 0.5);
        }
        self.emit(GLOW, pos, Vec3::ZERO, 0.08, [0.08, 0.2], [5.0, 3.0, 2.0, 1.0], 0.0, 0.0);
    }

    /// Explosive decompression of a killed suit.
    pub fn death(&mut self, pos: Vec3, up: Vec3, team: [f32; 3]) {
        for _ in 0..self.count(26.0) {
            let v = self.dir() * (1.0 + 4.0 * self.rnd()) + up * 1.0;
            let w = 0.88 + 0.12 * self.rnd();
            self.emit(SMOKE, pos + self.dir() * 0.2, v, 1.0 + 1.5 * self.rnd(), [0.1, 0.7], [w, w, w, 0.35], 2.5, -0.02);
        }
        for _ in 0..self.count(14.0) {
            let v = self.dir() * (2.0 + 6.0 * self.rnd()) + up * 2.0;
            self.emit(GLOW, pos, v, 0.6 + 0.6 * self.rnd(), [0.02, 0.01], [team[0] * 4.0, team[1] * 4.0, team[2] * 4.0, 1.0], 1.0, 1.0);
        }
        self.emit(RING, pos, Vec3::ZERO, 0.35, [0.2, 2.2], [team[0] * 2.0, team[1] * 2.0, team[2] * 2.0, 1.0], 0.0, 0.0);
    }

    pub fn glass(&mut self, pos: Vec3) {
        for _ in 0..self.count(24.0) {
            let v = self.dir() * (1.0 + 4.0 * self.rnd());
            self.emit(GLOW, pos + self.dir() * 0.5, v, 1.0 + 1.5 * self.rnd(), [0.015, 0.012], [2.5, 3.2, 3.8, 1.0], 0.3, 1.0);
        }
        for _ in 0..self.count(6.0) {
            let v = self.dir() * 0.6;
            self.emit(SMOKE, pos + self.dir() * 0.4, v, 1.5, [0.2, 0.7], [0.8, 0.82, 0.85, 0.2], 1.5, 0.0);
        }
    }

    pub fn dust_step(&mut self, pos: Vec3, up: Vec3) {
        for _ in 0..self.count(2.0) {
            let v = (self.dir() * 0.5 + up * 0.3) * (0.3 + 0.5 * self.rnd());
            self.emit(SMOKE, pos + up * 0.03, v, 0.7 + 0.5 * self.rnd(), [0.05, 0.25], [0.5, 0.47, 0.42, 0.18], 3.0, 0.0);
        }
    }

    pub fn jet(&mut self, pos: Vec3, dir: Vec3, dt: f32) {
        let n = (dt * 90.0 * self.quality).ceil() as usize;
        for _ in 0..n {
            let v = self.cone(dir, 0.12) * (8.0 + 6.0 * self.rnd());
            self.emit(GLOW, pos, v, 0.08 + 0.06 * self.rnd(), [0.05, 0.015], [1.2, 2.2, 6.0, 1.0], 2.0, 0.0);
        }
        if self.rnd() < dt * 20.0 {
            let v = self.cone(dir, 0.3) * 2.0;
            self.emit(SMOKE, pos + dir * 0.3, v, 1.0, [0.05, 0.4], [0.75, 0.78, 0.82, 0.12], 2.0, -0.02);
        }
    }

    pub fn shockwave(&mut self, pos: Vec3, radius: f32, color: [f32; 3]) {
        self.emit(RING, pos, Vec3::ZERO, 0.5, [0.3, radius], [color[0], color[1], color[2], 1.0], 0.0, 0.0);
        self.emit(GLOW, pos, Vec3::ZERO, 0.25, [radius * 0.3, radius * 0.6], [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 1.0], 0.0, 0.0);
        self.light(pos, 14.0, 0.3);
    }

    pub fn update(&mut self, dt: f32) {
        for p in &mut self.parts {
            p.age += dt;
            let up = up_at(p.pos);
            p.vel -= up * (g_at(p.pos) * p.grav * dt);
            p.vel *= 1.0 / (1.0 + p.drag * dt);
            if p.kind == SPARK {
                p.streak = (p.vel.length() * 0.025).clamp(0.01, 0.4);
            }
            p.pos += p.vel * dt;
        }
        self.parts.retain(|p| p.age < p.life);
        for c in &mut self.casings {
            c.age += dt;
            if c.rest {
                continue;
            }
            let up = up_at(c.pos);
            c.vel -= up * (g_at(c.pos) * dt);
            c.pos += c.vel * dt;
            c.rot = (Quat::from_scaled_axis(c.spin * dt) * c.rot).normalize();
            let h = (c.pos.x * c.pos.x + c.pos.y * c.pos.y).sqrt();
            let floor_r = gameplay::geom::R as f32 - c.floor - 0.005;
            if h > floor_r {
                // bounce on the floor (ground level remembered at ejection)
                c.pos -= up * (floor_r - h).abs() * -1.0;
                let vn = c.vel.dot(up);
                if vn < 0.0 {
                    c.vel -= up * vn * 1.45;
                    c.vel *= 0.5;
                    c.spin *= 0.5;
                }
                if c.vel.length() < 0.25 {
                    c.rest = true;
                }
            }
        }
        self.casings.retain(|c| c.age < 6.0);
        for l in &mut self.lights {
            l.2 -= dt;
        }
        self.lights.retain(|l| l.2 > 0.0);
    }

    pub fn write(&self, cam: Vec3, out: &mut Vec<ParticleGpu>) {
        let start = out.len();
        for p in &self.parts {
            let t = (p.age / p.life).clamp(0.0, 1.0);
            let size = p.size[0] + (p.size[1] - p.size[0]) * (1.0 - (1.0 - t) * (1.0 - t));
            let fade = match p.kind {
                SMOKE => (t * 8.0).min(1.0) * (1.0 - t) * (1.0 - t),
                RING => 1.0 - t,
                _ => (1.0 - t).powf(1.5),
            };
            let mut c = p.color;
            c[3] *= fade;
            let axis = if p.kind == SPARK { p.vel.normalize_or(Vec3::Y) * p.streak } else { Vec3::ZERO };
            out.push(ParticleGpu { pos: p.pos.to_array(), size, color: c, axis: axis.to_array(), kind: p.kind });
        }
        // back to front (premultiplied blending of smoke)
        out[start..].sort_by(|a, b| {
            let da = Vec3::from(a.pos).distance_squared(cam);
            let db = Vec3::from(b.pos).distance_squared(cam);
            db.total_cmp(&da)
        });
    }

    pub fn casings(&self, part: u16, out: &mut Vec<(u16, PropInst)>) {
        for c in &self.casings {
            let m = crate::avatar::casing(c.pos, c.rot);
            let hot = (1.0 - c.age / 1.5).max(0.0);
            out.push((part, PropInst::new(m, [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, hot * 0.3, 0.0])));
        }
    }

    /// Up to `n` strongest lights: [x, y, z, ±intensity] (negative = warm flash).
    pub fn lights(&self, n: usize) -> Vec<[f32; 4]> {
        let mut v: Vec<[f32; 4]> = self.lights.iter().map(|l| [l.0.x, l.0.y, l.0.z, l.1 * (l.2 / l.3)]).collect();
        v.sort_by(|a, b| b[3].abs().total_cmp(&a[3].abs()));
        v.truncate(n);
        v
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }
}
