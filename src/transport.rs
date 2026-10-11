//! Maglev trains shuttling along the elevated tracks of every valley, with
//! smooth acceleration and stops at the stations. Air-cars are fully
//! procedural on the GPU (see aircars.wgsl).

use crate::colony::surf_to_world;
use crate::world::{style, ObjInstance, Track, World, K_TRAIN, M_TRAIN};

pub const CARS: usize = 6;
pub const CAR_LEN: f32 = 26.0;
pub const CAR_GAP: f32 = 1.2;
pub const CAR_H: f32 = 4.0;
const VMAX: f32 = 70.0; // 252 km/h
const ACCEL: f32 = 1.3;
const DWELL: f32 = 14.0;

pub struct Train {
    pub track: usize,
    pub v: f32, // head position along the valley
    pub speed: f32,
    pub dir: f32,
    pub wait: f32,
    pub target: f32,
}

pub struct Transport {
    pub trains: Vec<Train>,
}

impl Transport {
    pub fn new(world: &World) -> Self {
        let trains = world
            .tracks
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
                let v = t.stations[i % t.stations.len()];
                Train { track: i, v, speed: 0.0, dir, wait: 3.0 + i as f32 * 2.0, target: v }
            })
            .collect();
        Transport { trains }
    }

    fn train_len() -> f32 {
        CARS as f32 * (CAR_LEN + CAR_GAP)
    }

    pub fn update(&mut self, world: &World, dt: f32) {
        for tr in &mut self.trains {
            let track = &world.tracks[tr.track];
            if tr.wait > 0.0 {
                tr.wait -= dt;
                if tr.wait <= 0.0 {
                    // Next stop in the travel direction, or turn around at the end.
                    let next = next_stop(track, tr.v, tr.dir);
                    match next {
                        Some(t) => tr.target = t,
                        None => {
                            tr.dir = -tr.dir;
                            tr.target = next_stop(track, tr.v, tr.dir).unwrap_or(tr.v);
                        }
                    }
                }
                continue;
            }
            let remaining = (tr.target - tr.v) * tr.dir;
            let brake = tr.speed * tr.speed / (2.0 * ACCEL);
            if remaining <= brake + 0.5 {
                tr.speed = (tr.speed - ACCEL * dt).max(1.5_f32.min(remaining.max(0.0)));
            } else {
                tr.speed = (tr.speed + ACCEL * dt).min(VMAX);
            }
            tr.v += tr.speed * tr.dir * dt;
            if (tr.target - tr.v) * tr.dir <= 0.05 {
                tr.v = tr.target;
                tr.speed = 0.0;
                tr.wait = DWELL;
            }
        }
    }

    /// Centre (along v) of car `c` of train `t`.
    pub fn car_v(&self, t: usize, c: usize) -> f32 {
        let tr = &self.trains[t];
        let first = tr.v - tr.dir * (Self::train_len() * 0.5 - CAR_LEN * 0.5);
        let _ = first;
        tr.v - tr.dir * (c as f32 * (CAR_LEN + CAR_GAP) - Self::train_len() * 0.5 + CAR_LEN * 0.5)
    }

    pub fn instances(&self, world: &World, out: &mut Vec<ObjInstance>) {
        out.clear();
        for (ti, tr) in self.trains.iter().enumerate() {
            let track = &world.tracks[tr.track];
            for c in 0..CARS {
                let v = self.car_v(ti, c);
                let h = track.deck_at(v) + 0.35;
                let p = surf_to_world(track.u(), v, h);
                let yaw = if tr.dir > 0.0 { 0.0 } else { std::f32::consts::PI };
                // Only the lead car carries headlights at its front; the shader
                // lights the front white and the back red, which reads right
                // for the leading and trailing car.
                out.push(ObjInstance { pos: p.to_array(), yaw, size: [3.6, CAR_H, CAR_LEN], style: style(K_TRAIN, M_TRAIN, (ti * 8 + c) as u32) });
            }
        }
    }

    /// Roof position for a rider of train `t` (car 1).
    pub fn roof(&self, world: &World, t: usize) -> glam::Vec3 {
        let tr = &self.trains[t];
        let track = &world.tracks[tr.track];
        let v = self.car_v(t, 1);
        surf_to_world(track.u(), v, track.deck_at(v) + 0.35 + CAR_H + crate::physics::EYE)
    }

    pub fn velocity(&self, world: &World, t: usize) -> glam::Vec3 {
        let _ = world;
        let tr = &self.trains[t];
        glam::Vec3::Z * tr.speed * tr.dir
    }

    pub fn nearest(&self, world: &World, pos: glam::Vec3) -> usize {
        let mut best = (f32::MAX, 0);
        for t in 0..self.trains.len() {
            let d = self.roof(world, t).distance(pos);
            if d < best.0 {
                best = (d, t);
            }
        }
        best.1
    }
}

fn next_stop(track: &Track, v: f32, dir: f32) -> Option<f32> {
    let half = Transport::train_len() * 0.5 + 5.0;
    let mut stops: Vec<f32> = track.stations.clone();
    stops.push(track.v0 + half);
    stops.push(track.v1 - half);
    stops.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if dir > 0.0 {
        stops.into_iter().find(|&s| s > v + 1.0)
    } else {
        stops.into_iter().rev().find(|&s| s < v - 1.0)
    }
}
