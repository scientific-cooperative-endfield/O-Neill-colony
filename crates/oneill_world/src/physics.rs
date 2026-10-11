//! Physics in the rotating frame of the colony: there is no gravity, only the
//! centrifugal and Coriolis pseudo-forces. Balls of eight kinds (from a
//! helium balloon that drifts up to the axis to a 7-tonne boulder), water
//! buoyancy, rolling, collisions with terrain, buildings and each other.
//! The player walks, jumps and flies under the same laws.

use glam::{Quat, Vec3};

use crate::colony::{self, frame_accel, world_to_surf, CIRC, G0, L, R};
use crate::world::World;

pub const AIR_DENSITY: f32 = 1.2;
pub const WATER_DENSITY: f32 = 1000.0;
pub const MAX_BALLS: usize = 64;
pub const SPINDLE_R: f32 = 30.0;

#[derive(Clone, Copy)]
pub struct BallType {
    pub name: &'static str,
    pub radius: f32,
    pub mass: f32,
    pub restitution: f32,
    pub cd: f32,
    pub friction: f32,
    pub emissive: f32,
}

pub const BALL_TYPES: [BallType; 8] = [
    BallType { name: "Пляжный мяч", radius: 0.35, mass: 0.35, restitution: 0.72, cd: 0.47, friction: 0.5, emissive: 0.0 },
    BallType { name: "Футбольный", radius: 0.11, mass: 0.43, restitution: 0.7, cd: 0.25, friction: 0.6, emissive: 0.0 },
    BallType { name: "Баскетбольный", radius: 0.12, mass: 0.62, restitution: 0.8, cd: 0.47, friction: 0.7, emissive: 0.0 },
    BallType { name: "Шар для боулинга", radius: 0.109, mass: 7.0, restitution: 0.15, cd: 0.47, friction: 0.3, emissive: 0.0 },
    BallType { name: "Светящаяся сфера", radius: 0.25, mass: 1.0, restitution: 0.6, cd: 0.47, friction: 0.5, emissive: 25.0 },
    BallType { name: "Валун", radius: 0.9, mass: 7000.0, restitution: 0.1, cd: 0.8, friction: 0.9, emissive: 0.0 },
    BallType { name: "Хромированная сфера", radius: 0.5, mass: 400.0, restitution: 0.45, cd: 0.47, friction: 0.4, emissive: 0.0 },
    BallType { name: "Гелиевый шар", radius: 0.4, mass: 0.12, restitution: 0.5, cd: 0.6, friction: 0.5, emissive: 0.0 },
];

impl BallType {
    pub fn volume(&self) -> f32 {
        4.0 / 3.0 * std::f32::consts::PI * self.radius.powi(3)
    }
    pub fn density(&self) -> f32 {
        self.mass / self.volume()
    }
}

#[derive(Clone, Copy)]
pub struct Ball {
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: Quat,
    pub ang: Vec3,
    pub ty: usize,
    pub age: f32,
    pub contact: f32, // clearance above the ground (m)
}

#[derive(Clone, Copy)]
pub struct Frame {
    pub omega: f32,
    pub coriolis: bool,
    /// Black-hole mode: the rotating frame of the player's station fragment
    /// with the tidal field of the hole (None = the classic colony).
    pub ext: Option<FrameExt>,
}

/// Rotating frame of a (possibly broken-off) station fragment.
#[derive(Clone, Copy, Debug)]
pub struct FrameExt {
    /// angular velocity in the colony frame (rad/s)
    pub spin: Vec3,
    /// centre of mass of the fragment (colony coordinates, m)
    pub com: Vec3,
    /// tidal tensor in the colony frame (s⁻²): a = K·(p − com)
    pub tidal: glam::Mat3,
}

impl Frame {
    /// Inertial + tidal acceleration felt in the rotating frame.
    pub fn accel(&self, pos: Vec3, vel: Vec3, coriolis: bool) -> Vec3 {
        match &self.ext {
            None => frame_accel(pos, vel, self.omega, coriolis),
            Some(e) => {
                let d = pos - e.com;
                let w = e.spin;
                let mut a = -w.cross(w.cross(d)) + e.tidal * d;
                if coriolis {
                    a -= 2.0 * w.cross(vel);
                }
                a
            }
        }
    }
}

/// Is the point over solid land (not over a window)?
fn floor_height(world: &World, u: f32, v: f32) -> (f32, bool) {
    if colony::is_land(u.rem_euclid(CIRC) / R) {
        let h = world.height(u, v);
        (h, h < 0.0)
    } else {
        (0.0, false) // window glass
    }
}

fn local_basis(p: Vec3) -> (Vec3, Vec3, Vec3) {
    colony::basis_at(p).unwrap_or((Vec3::Y, Vec3::NEG_X, Vec3::Z))
}

/// Acceleration on a ball excluding contacts.
fn ball_accel(b: &Ball, t: &BallType, frame: Frame, water_sub: f32) -> Vec3 {
    let cf = frame.accel(b.pos, Vec3::ZERO, false); // outward "gravity"
    let full = frame.accel(b.pos, b.vel, frame.coriolis);
    let vol = t.volume();
    let area = std::f32::consts::PI * t.radius * t.radius;
    let rho = AIR_DENSITY * (1.0 - water_sub) + WATER_DENSITY * water_sub;
    let buoy = -cf * (rho * vol / t.mass);
    let sp = b.vel.length();
    let drag = -b.vel * (0.5 * rho * t.cd * area * sp / t.mass);
    full + buoy + drag
}

/// Push a sphere out of terrain / boxes / walls. Returns the contact normal.
fn resolve_static(world: &World, pos: &mut Vec3, vel: &mut Vec3, r: f32, e: f32, scratch: &mut Vec<u32>) -> Option<Vec3> {
    let mut contact = None;
    // Hull (window glass and the floor under it), end caps and spindle.
    let rp = pos.truncate().length();
    if rp > R - r {
        let n = -Vec3::new(pos.x, pos.y, 0.0) / rp.max(1e-3);
        *pos += n * (rp - (R - r));
        bounce(vel, n, e);
        contact = Some(n);
    }
    if rp < SPINDLE_R + r && rp > 1e-3 {
        let n = Vec3::new(pos.x, pos.y, 0.0) / rp;
        *pos += n * (SPINDLE_R + r - rp);
        bounce(vel, n, e);
    }
    for (lim, n) in [(r, Vec3::Z), (L - r, Vec3::NEG_Z)] {
        if (n.z > 0.0 && pos.z < lim) || (n.z < 0.0 && pos.z > lim) {
            pos.z = lim;
            bounce(vel, n, e);
        }
    }
    let sf = world_to_surf(*pos);
    let (u, v, h) = (sf.x, sf.y, sf.z);
    let (fh, _) = floor_height(world, u, v);
    if h - r < fh {
        let n = if colony::is_land(u / R) { world.normal(u, v) } else { local_basis(*pos).1 };
        let up = local_basis(*pos).1;
        let pen = (fh - (h - r)) * n.dot(up).max(0.3);
        *pos += n * pen;
        bounce(vel, n, e);
        contact = Some(n);
    }
    // Buildings, trunks, decks.
    let (k, s) = colony::strip_of(u / R);
    world.colliders_near(u, v, r + 1.0, scratch);
    for &i in scratch.iter() {
        let c = &world.colliders[i as usize];
        if c.strip as usize != k {
            continue;
        }
        let q = [(s - c.s).clamp(-c.hs, c.hs) + c.s, (v - c.v).clamp(-c.hv, c.hv) + c.v, h.clamp(c.h0, c.h1)];
        let d = Vec3::new(s - q[0], h - q[2], v - q[1]); // (east, up, north)
        let dist = d.length();
        let nl = if dist > 1e-4 {
            if dist >= r {
                continue;
            }
            d / dist
        } else {
            // Centre inside the box: leave along the shallowest axis.
            let opts = [(c.hs - (s - c.s).abs(), Vec3::X * (s - c.s).signum()), (c.hv - (v - c.v).abs(), Vec3::Z * (v - c.v).signum()), (c.h1 - h, Vec3::Y)];
            opts.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap().1
        };
        let pen = if dist > 1e-4 { r - dist } else { r };
        let (e_, up, nn) = local_basis(*pos);
        let n = e_ * nl.x + up * nl.y + nn * nl.z;
        *pos += n * pen;
        bounce(vel, n, e);
        contact = Some(n);
    }
    contact
}

fn bounce(vel: &mut Vec3, n: Vec3, e: f32) {
    let vn = vel.dot(n);
    if vn < 0.0 {
        let e = if vn > -0.6 { 0.0 } else { e };
        *vel -= n * vn * (1.0 + e);
    }
}

pub fn step_balls(world: &World, balls: &mut [Ball], dt: f32, frame: Frame, scratch: &mut Vec<u32>) {
    let sub = 4;
    let h = dt / sub as f32;
    for _ in 0..sub {
        for b in balls.iter_mut() {
            let t = BALL_TYPES[b.ty];
            let sf = world_to_surf(b.pos);
            let (fh, wet) = floor_height(world, sf.x, sf.y);
            let water_sub = if wet { ((t.radius - sf.z) / (2.0 * t.radius)).clamp(0.0, 1.0) } else { 0.0 };
            let a = ball_accel(b, &t, frame, water_sub);
            b.vel += a * h;
            b.pos += b.vel * h;
            let mut v = b.vel;
            let mut p = b.pos;
            let contact = resolve_static(world, &mut p, &mut v, t.radius, t.restitution, scratch);
            b.pos = p;
            b.vel = v;
            b.contact = (sf.z - t.radius - fh).max(0.0);
            if let Some(n) = contact {
                // Rolling: friction drives the spin towards no-slip and
                // rolling resistance slowly bleeds the speed.
                let vt = b.vel - n * b.vel.dot(n);
                let g = frame.accel(b.pos, Vec3::ZERO, false).length();
                let rr = (0.02 + 0.1 * (1.0 - t.friction)) * g * h;
                let sp = vt.length();
                if sp > 1e-4 {
                    let k = (1.0 - rr / sp).max(0.0);
                    b.vel -= vt * (1.0 - k);
                }
                let target = n.cross(b.vel) / t.radius;
                b.ang = b.ang.lerp(target, (t.friction * 30.0 * h).min(1.0));
            } else {
                b.ang *= 1.0 - 0.05 * h;
            }
            if water_sub > 0.0 {
                b.vel *= 1.0 - (1.5 * water_sub * h).min(0.5);
                b.ang *= 1.0 - (2.0 * h).min(0.5);
            }
            let w = b.ang.length();
            if w > 1e-5 {
                b.rot = (Quat::from_axis_angle(b.ang / w, w * h) * b.rot).normalize();
            }
        }
        // Ball–ball collisions (impulses with masses).
        let n = balls.len();
        for i in 0..n {
            for j in i + 1..n {
                let (ti, tj) = (BALL_TYPES[balls[i].ty], BALL_TYPES[balls[j].ty]);
                let d = balls[j].pos - balls[i].pos;
                let rs = ti.radius + tj.radius;
                let l2 = d.length_squared();
                if l2 >= rs * rs || l2 < 1e-10 {
                    continue;
                }
                let l = l2.sqrt();
                let nrm = d / l;
                let (mi, mj) = (ti.mass, tj.mass);
                let pen = rs - l;
                balls[i].pos -= nrm * pen * mj / (mi + mj);
                balls[j].pos += nrm * pen * mi / (mi + mj);
                let rv = (balls[j].vel - balls[i].vel).dot(nrm);
                if rv < 0.0 {
                    let e = ti.restitution.min(tj.restitution).max(0.1);
                    let jimp = -(1.0 + e) * rv / (1.0 / mi + 1.0 / mj);
                    balls[i].vel -= nrm * (jimp / mi);
                    balls[j].vel += nrm * (jimp / mj);
                }
            }
        }
    }
    for b in balls.iter_mut() {
        b.age += dt;
    }
}

/// Predicted flight of a thrown ball: the real one in the rotating colony
/// and the naive "as on Earth" one (constant gravity, no Coriolis).
pub fn predict(world: &World, start: Vec3, vel: Vec3, ty: usize, frame: Frame, scratch: &mut Vec<u32>) -> (Vec<Vec3>, Vec<Vec3>) {
    let t = BALL_TYPES[ty];
    let dt = 1.0 / 60.0;
    let mut real = vec![start];
    let mut b = Ball { pos: start, vel, rot: Quat::IDENTITY, ang: Vec3::ZERO, ty, age: 0.0, contact: 1.0 };
    for i in 0..60 * 30 {
        let a = ball_accel(&b, &t, frame, 0.0);
        b.vel += a * dt;
        b.pos += b.vel * dt;
        let (mut p, mut v) = (b.pos, b.vel);
        let hit = resolve_static(world, &mut p, &mut v, t.radius, 0.0, scratch).is_some();
        if i % 2 == 0 || hit {
            real.push(p);
        }
        if hit || b.pos.truncate().length() < SPINDLE_R + t.radius {
            break;
        }
    }
    let g = frame.accel(start, Vec3::ZERO, false);
    let mut naive = vec![start];
    let (mut p, mut v) = (start, vel);
    let vol_k = 1.0 - AIR_DENSITY * t.volume() / t.mass;
    let area = std::f32::consts::PI * t.radius * t.radius;
    for i in 0..60 * 30 {
        let drag = -v * (0.5 * AIR_DENSITY * t.cd * area * v.length() / t.mass);
        v += (g * vol_k + drag) * dt;
        p += v * dt;
        let sf = world_to_surf(p);
        let hit = sf.z - t.radius < floor_height(world, sf.x, sf.y).0;
        if i % 2 == 0 || hit {
            naive.push(p);
        }
        if hit || p.truncate().length() < SPINDLE_R || p.z < 0.0 || p.z > L {
            break;
        }
    }
    (real, naive)
}

// ---------------------------------------------------------------- player --
pub struct Player {
    pub pos: Vec3, // eye position
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fly: bool,
    pub fly_speed: f32,
    pub grounded: bool,
    pub ride: Option<usize>,
    basis: (Vec3, Vec3, Vec3),
}

pub const EYE: f32 = 1.7;

#[derive(Default, Clone, Copy)]
pub struct MoveInput {
    pub fwd: f32,
    pub right: f32,
    pub up: f32,
    pub run: bool,
    pub jump: bool,
}

impl Player {
    pub fn new(pos: Vec3) -> Self {
        let mut p = Player {
            pos,
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fly: false,
            fly_speed: 25.0,
            grounded: false,
            ride: None,
            basis: (Vec3::Y, Vec3::NEG_X, Vec3::Z),
        };
        p.update_basis();
        p
    }

    fn update_basis(&mut self) {
        if let Some(b) = colony::basis_at(self.pos) {
            if self.pos.truncate().length() > 3.0 {
                self.basis = b;
            }
        }
    }

    /// (east, up, north) local frame of the player.
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        self.basis
    }

    pub fn forward(&self) -> Vec3 {
        let (e, up, n) = self.basis;
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        ((e * sy + n * cy) * cp + up * sp).normalize()
    }

    pub fn up(&self) -> Vec3 {
        self.basis.1
    }

    pub fn teleport(&mut self, pos: Vec3, fly: bool) {
        self.pos = pos;
        self.vel = Vec3::ZERO;
        self.fly = fly;
        self.ride = None;
        self.update_basis();
    }

    /// Felt artificial gravity in g.
    pub fn g_force(&self, frame: &Frame) -> f32 {
        frame.accel(self.pos, Vec3::ZERO, false).length() / G0
    }

    pub fn step(&mut self, world: &World, input: MoveInput, dt: f32, frame: Frame, scratch: &mut Vec<u32>) {
        self.update_basis();
        let (e, up, n) = self.basis;
        let (sy, cy) = self.yaw.sin_cos();
        let flat_f = e * sy + n * cy;
        let flat_r = n * sy - e * cy; // = flat_f × up (screen right)
        if self.fly {
            let f = self.forward();
            let mut dir = f * input.fwd + flat_r * input.right + up * input.up;
            if dir.length_squared() > 1.0 {
                dir = dir.normalize();
            }
            let sp = self.fly_speed * if input.run { 4.0 } else { 1.0 };
            self.vel = self.vel.lerp(dir * sp, (dt * 6.0).min(1.0));
            self.pos += self.vel * dt;
        } else {
            let walk = if input.run { 11.0 } else { 4.5 };
            let mut wish = flat_f * input.fwd + flat_r * input.right;
            if wish.length_squared() > 1.0 {
                wish = wish.normalize();
            }
            let wish = wish * walk;
            if self.grounded {
                let vert = up * self.vel.dot(up);
                let horiz = self.vel - vert;
                self.vel = horiz.lerp(wish, (dt * 10.0).min(1.0)) + vert;
                if input.jump {
                    self.vel += up * 5.0;
                    self.grounded = false;
                }
            } else {
                self.vel += wish * (dt * 0.6);
            }
            let a = frame.accel(self.pos, self.vel, frame.coriolis);
            self.vel += a * dt;
            self.vel -= self.vel * (0.002 * self.vel.length() * dt);
            self.pos += self.vel * dt;
        }
        self.collide(world, scratch);
    }

    fn collide(&mut self, world: &World, scratch: &mut Vec<u32>) {
        let rp = self.pos.truncate().length();
        // Free flight in vacuum around the colony (exterior views).
        if self.fly && (rp > R + 1.0 || self.pos.z < 0.0 || self.pos.z > L) {
            return;
        }
        if rp > R - 0.2 {
            self.pos *= Vec3::new((R - 0.2) / rp, (R - 0.2) / rp, 1.0);
        }
        if rp < SPINDLE_R + 0.5 && rp > 1e-3 {
            let k = (SPINDLE_R + 0.5) / rp;
            self.pos.x *= k;
            self.pos.y *= k;
        }
        self.pos.z = self.pos.z.clamp(0.5, L - 0.5);
        let sf = world_to_surf(self.pos);
        let (u, v, h) = (sf.x, sf.y, sf.z);
        let up = self.basis.1;
        // Walls: push out of boxes we are inside of (feet below the top).
        let (k, s) = colony::strip_of(u / R);
        world.colliders_near(u, v, 1.5, scratch);
        let feet = h - EYE;
        let mut push = Vec3::ZERO;
        for &i in scratch.iter() {
            let c = &world.colliders[i as usize];
            if c.strip as usize != k || feet > c.h1 - 0.7 || h < c.h0 {
                continue;
            }
            let r = 0.35;
            let ds = s - c.s;
            let dv = v - c.v;
            let ox = c.hs + r - ds.abs();
            let oz = c.hv + r - dv.abs();
            if ox > 0.0 && oz > 0.0 {
                if ox < oz {
                    push += self.basis.0 * ox * ds.signum();
                } else {
                    push += self.basis.2 * oz * dv.signum();
                }
            }
        }
        if push != Vec3::ZERO {
            self.pos += push;
            let pn = push.normalize();
            let vn = self.vel.dot(pn);
            if vn < 0.0 {
                self.vel -= pn * vn;
            }
        }
        let sf = world_to_surf(self.pos);
        let floor = if colony::is_land(sf.x / R) { world.floor_below(sf.x, sf.y, sf.z - EYE, scratch).max(-0.3) } else { 0.0 };
        if sf.z - EYE < floor {
            self.pos += up * (floor - (sf.z - EYE));
            let vn = self.vel.dot(up);
            if vn < 0.0 {
                self.vel -= up * vn;
            }
            self.grounded = true;
        } else if sf.z - EYE > floor + 0.25 {
            self.grounded = false;
        } else if !self.fly && self.vel.dot(up) < 0.5 {
            // Stick to slopes when walking downhill.
            self.pos -= up * (sf.z - EYE - floor);
            self.grounded = true;
        }
    }
}
