//! What a bot knows: vision (cone + line of sight, range, target posture) and
//! hearing (sound travels at the local speed of sound, spreads spherically and
//! is absorbed; there is no sound in vacuum). Bots never read hidden state
//! beyond what these channels would reveal (positions of *seen* targets).

use crate::config::Difficulty;
use gameplay::det::Rng;
use gameplay::sim::{Sim, SoundKind};
use gameplay::{Entity, Event, TICK_HZ};
use glam::DVec3;

#[derive(Clone, Debug, PartialEq)]
pub struct Contact {
    pub e: Entity,
    /// last known position (feet) and velocity
    pub pos: DVec3,
    pub vel: DVec3,
    pub visible: bool,
    pub first_seen: u32,
    pub last_seen: u32,
    pub last_heard: u32,
    /// position uncertainty (m)
    pub sigma: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Perception {
    pub contacts: Vec<Contact>,
    last_hear_tick: u32,
    /// tick and rough direction of the last bullet that hit or passed us
    pub under_fire: Option<(u32, DVec3)>,
    /// loud events worth investigating (position, tick)
    pub noise: Option<(DVec3, u32)>,
}

/// Air absorption of a gunshot-like spectrum (dB/m), order of magnitude.
const ABSORB_DB_PER_M: f64 = 0.008;
const P0: f64 = 101_325.0;

/// Sound level (dB SPL) heard at `d` metres in air at pressure `p`.
pub fn heard_level(db_1m: f64, d: f64, p: f64) -> f64 {
    let d = d.max(1.0);
    db_1m - 20.0 * d.log10() - ABSORB_DB_PER_M * d + 20.0 * (p / P0).max(1e-6).log10()
}

impl Perception {
    pub fn contact(&self, e: Entity) -> Option<&Contact> {
        self.contacts.iter().find(|c| c.e == e)
    }

    fn upsert(&mut self, e: Entity, f: impl FnOnce(&mut Contact), tick: u32) {
        if let Some(c) = self.contacts.iter_mut().find(|c| c.e == e) {
            f(c);
        } else {
            let mut c = Contact { e, pos: DVec3::ZERO, vel: DVec3::ZERO, visible: false, first_seen: tick, last_seen: 0, last_heard: 0, sigma: 1e9 };
            f(&mut c);
            self.contacts.push(c);
        }
    }

    /// Vision (call at ~10 Hz).
    pub fn look(&mut self, sim: &Sim, me: Entity, d: &Difficulty, scratch: &mut Vec<u32>) {
        let Some(my_body) = sim.ecs.bodies.get(me) else { return };
        let eye = sim.eye(me);
        let fwd = sim.aim_dir(me);
        let cos_half = (0.5 * d.fov_deg.to_radians()).cos();
        let my_team = sim.team(me);
        let tick = sim.tick;
        let _ = my_body;
        for &o in &sim.players {
            if o == me || sim.team(o) == my_team {
                continue;
            }
            let alive = sim.is_alive(o);
            let Some(b) = sim.ecs.bodies.get(o) else { continue };
            let chest = b.pos + b.up * (0.65 * b.height);
            let to = chest - eye;
            let dist = to.length();
            // smaller silhouette when crouched; motion attracts the eye
            let mut range = d.vision_range * if b.crouched { 0.7 } else { 1.0 };
            if b.vel.length() > 3.0 {
                range *= 1.2;
            }
            let in_cone = dist < 4.0 || to.dot(fwd) > cos_half * dist;
            let seen = alive && dist < range && in_cone && {
                let head = b.pos + b.up * (b.height - 0.1);
                sim.arena.visible(eye, chest, scratch) || sim.arena.visible(eye, head, scratch)
            };
            if seen {
                self.upsert(
                    o,
                    |c| {
                        if !c.visible && tick.saturating_sub(c.last_seen) > TICK_HZ * 3 {
                            c.first_seen = tick;
                        }
                        c.pos = b.pos;
                        c.vel = b.vel;
                        c.visible = true;
                        c.last_seen = tick;
                        c.sigma = 0.0;
                    },
                    tick,
                );
            } else if let Some(c) = self.contacts.iter_mut().find(|c| c.e == o) {
                c.visible = false;
                if !alive {
                    c.last_seen = 0;
                    c.last_heard = 0;
                }
            }
        }
        // forget stale contacts
        self.contacts.retain(|c| tick.saturating_sub(c.last_seen.max(c.last_heard)) < TICK_HZ * 20);
    }

    /// Hearing and damage events (call every tick, after the simulation step).
    pub fn listen(&mut self, sim: &Sim, me: Entity, d: &Difficulty, rng: &mut Rng) {
        let tick = sim.tick;
        let eye = sim.eye(me);
        let my_team = sim.team(me);
        let p_here = sim.atmo.pressure(&sim.frame, eye);
        let c_snd = sim.atmo.speed_of_sound();
        let in_air = sim.atmo.density(&sim.frame, eye) > 0.01;
        let from = self.last_hear_tick;
        self.last_hear_tick = tick;
        if in_air {
            for s in &sim.sounds {
                if s.source == me || !s.in_air || sim.team(s.source) == my_team && s.kind != SoundKind::Impact && s.kind != SoundKind::Break {
                    continue;
                }
                let dist = (s.pos - eye).length();
                let arrive = s.tick + ((dist / c_snd) * TICK_HZ as f64).ceil() as u32;
                if arrive <= from || arrive > tick {
                    continue;
                }
                let level = heard_level(s.db, dist, p_here.min(sim.atmo.pressure(&sim.frame, s.pos)));
                if level < d.hearing_db {
                    continue;
                }
                // localisation: a few degrees for loud sounds, worse when faint
                let sigma = dist * (0.03 + 0.15 * (1.0 - ((level - d.hearing_db) / 40.0).clamp(0.0, 1.0)));
                let guess = s.pos + DVec3::new(rng.normal(), rng.normal(), rng.normal()) * sigma;
                match s.kind {
                    SoundKind::Shot | SoundKind::Step | SoundKind::Device => {
                        if sim.ecs.pilots.contains(s.source) && sim.team(s.source) != my_team {
                            let src = s.source;
                            self.upsert(
                                src,
                                |c| {
                                    if !c.visible && sigma < c.sigma + 5.0 {
                                        c.pos = guess;
                                        c.sigma = sigma;
                                    }
                                    c.last_heard = tick;
                                },
                                tick,
                            );
                        }
                        if s.kind == SoundKind::Shot {
                            self.noise = Some((guess, tick));
                        }
                    }
                    SoundKind::Impact | SoundKind::Break => {
                        if dist < 40.0 {
                            self.noise = Some((guess, tick));
                        }
                    }
                }
            }
        }
        for ev in &sim.events {
            if let Event::Hit { shooter, victim, .. } = *ev {
                if victim == me && shooter != me {
                    // pain tells the direction roughly; the shooter's position is
                    // estimated from it (no sound needed — works in vacuum)
                    let sp = sim.ecs.bodies.get(shooter).map_or(eye, |b| b.pos);
                    let dist = (sp - eye).length();
                    let sigma = 0.1 * dist + 3.0;
                    let guess = sp + DVec3::new(rng.normal(), rng.normal(), rng.normal()) * sigma;
                    self.under_fire = Some((tick, (guess - eye).normalize_or_zero()));
                    self.upsert(
                        shooter,
                        |c| {
                            if !c.visible {
                                c.pos = guess;
                                c.sigma = sigma;
                            }
                            c.last_heard = tick;
                        },
                        tick,
                    );
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.contacts.clear();
        self.under_fire = None;
        self.noise = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sound_levels_are_sane() {
        // a 160 dB rifle shot is still clearly audible 1 km away, a footstep is not
        assert!(heard_level(160.0, 1000.0, P0) > 80.0);
        assert!(heard_level(58.0, 30.0, P0) < 30.0);
        // at a tenth of an atmosphere it is 20 dB quieter
        let a = heard_level(100.0, 10.0, P0);
        let b = heard_level(100.0, 10.0, 0.1 * P0);
        assert!((a - b - 20.0).abs() < 1e-9);
    }
}
