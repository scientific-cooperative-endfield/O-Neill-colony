//! Fire-control solution in the rotating frame.
//!
//! The bot shoots with exactly the same physics as everybody else; what it
//! *knows* about that physics is the `Env` it solves with: `coriolis`,
//! `gravity` and `drag` can be scaled down to model a naive shooter. A novice
//! who aims "flat-Earth style" (no Coriolis) misses a 500 m sniper shot by
//! ≈ ωd²/v ≈ 15 m; a veteran's model is the real one and hits.
//!
//! Solver: shoot, measure the closest-approach miss vector against the
//! (led) target, move the aim point by −miss, repeat. The map from aim point
//! to impact point is a near-identity smooth map, so this fixed-point
//! iteration converges in 2–4 rounds.

use crate::config::Difficulty;
use gameplay::ballistics::{closest_approach, Env, Round};
use gameplay::DT;
use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solution {
    /// launch direction (unit)
    pub dir: DVec3,
    pub time_of_flight: f64,
    /// residual miss of the *model* (m)
    pub residual: f64,
}

/// The bot's mental model of the ballistics.
pub fn model_env<'a>(env: Env<'a>, d: &Difficulty) -> Env<'a> {
    let mut e = env;
    e.coriolis *= d.coriolis_model;
    e.gravity = d.gravity_model;
    e.drag = d.drag_model;
    e
}

/// Muzzle offset along the barrel (same as the simulation).
pub const MUZZLE: f64 = 0.4;

/// Direction to fire from `eye` (moving with `shooter_vel`) so that the round
/// meets `target` moving with `target_vel`. `None` when out of range.
#[allow(clippy::too_many_arguments)]
pub fn solve(env: &Env, round: &Round, muzzle_speed: f64, eye: DVec3, shooter_vel: DVec3, target: DVec3, target_vel: DVec3, max_t: f64) -> Option<Solution> {
    let mut aim = target;
    let mut best: Option<Solution> = None;
    for _ in 0..6 {
        let dir = (aim - eye).normalize_or_zero();
        if dir == DVec3::ZERO {
            return None;
        }
        let p0 = eye + dir * MUZZLE;
        let v0 = shooter_vel + dir * muzzle_speed;
        let (miss, t) = closest_approach(env, round, p0, v0, target, target_vel, max_t, DT);
        let r = miss.length();
        if best.is_none_or(|b| r < b.residual) {
            best = Some(Solution { dir, time_of_flight: t, residual: r });
        }
        if r < 0.02 {
            break;
        }
        aim -= miss;
    }
    best.filter(|b| b.residual < 1.0 && b.time_of_flight < max_t - DT)
}

/// Miss (m) of a round fired along `dir` in the *real* physics.
pub fn true_miss(env: &Env, round: &Round, muzzle_speed: f64, eye: DVec3, shooter_vel: DVec3, dir: DVec3, target: DVec3, target_vel: DVec3, max_t: f64) -> f64 {
    let (miss, _) = closest_approach(env, round, eye + dir * MUZZLE, shooter_vel + dir * muzzle_speed, target, target_vel, max_t, DT);
    miss.length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gameplay::fields::FieldSet;
    use gameplay::frame::{Atmosphere, StationFrame};
    use gameplay::geom;

    fn lvl(name: &str) -> Difficulty {
        crate::config::by_name(&crate::config::builtin(), name).unwrap()
    }

    #[test]
    fn veteran_hits_500m_both_ways_novice_misses_by_coriolis() {
        let frame = StationFrame::colony();
        let atmo = Atmosphere::default();
        let fs = FieldSet::default();
        let real = Env::new(&frame, &atmo, &fs);
        let cfg = gameplay::GameConfig::builtin();
        let w = cfg.weapons.iter().position(|w| w.id == "sniper").unwrap();
        let round = Round::from_cfg(&cfg.weapons[w]);
        let mv = cfg.weapons[w].muzzle_velocity;
        let (u, v, h) = (10_000.0, 16_000.0, 1.6);
        let eye = geom::surf_to_world(u, v, h);
        let vfloor = DVec3::ZERO; // standing still in the rotating frame
        for east in [true, false] {
            let target = geom::surf_to_world(if east { u + 500.0 } else { u - 500.0 }, v, 1.3);
            // veteran
            let vet = model_env(real, &lvl("veteran"));
            let s = solve(&vet, &round, mv, eye, vfloor, target, DVec3::ZERO, 3.0).expect("solution");
            let m = true_miss(&real, &round, mv, eye, vfloor, s.dir, target, DVec3::ZERO, 3.0);
            assert!(m < 0.3, "veteran miss {m} m (east {east})");
            // a shooter who ignores only Coriolis
            let mut naive = lvl("veteran");
            naive.coriolis_model = 0.0;
            let ne = model_env(real, &naive);
            let s2 = solve(&ne, &round, mv, eye, vfloor, target, DVec3::ZERO, 3.0).unwrap();
            let m2 = true_miss(&real, &round, mv, eye, vfloor, s2.dir, target, DVec3::ZERO, 3.0);
            // vertical Coriolis deflection of an east/west shot ≈ ω d² / v (more with drag slowing the round)
            let est = 0.5 * gameplay::ballistics::coriolis_split_estimate(frame.omega(), 500.0, mv);
            assert!(m2 > 0.85 * est && m2 < 1.35 * est, "naive miss {m2} m vs estimate {est} m (east {east})");
            // the real novice (no Coriolis, no drag) misses too
            let nov = model_env(real, &lvl("novice"));
            let s3 = solve(&nov, &round, mv, eye, vfloor, target, DVec3::ZERO, 3.0).unwrap();
            let m3 = true_miss(&real, &round, mv, eye, vfloor, s3.dir, target, DVec3::ZERO, 3.0);
            assert!(m3 > 5.0, "novice miss {m3}");
        }
    }

    #[test]
    fn leads_a_running_target() {
        let frame = StationFrame::colony();
        let atmo = Atmosphere::default();
        let fs = FieldSet::default();
        let real = Env::new(&frame, &atmo, &fs);
        let cfg = gameplay::GameConfig::builtin();
        let round = Round::from_cfg(&cfg.weapons[1]);
        let mv = cfg.weapons[1].muzzle_velocity;
        let eye = geom::surf_to_world(10_000.0, 16_000.0, 1.6);
        let target = geom::surf_to_world(10_000.0, 16_250.0, 1.3);
        let (e, _, _) = geom::basis_at(target);
        let tv = e * 7.5; // sprinting across the line of fire
        let s = solve(&real, &round, mv, eye, DVec3::ZERO, target, tv, 3.0).unwrap();
        assert!(true_miss(&real, &round, mv, eye, DVec3::ZERO, s.dir, target, tv, 3.0) < 0.1);
        // not leading misses by ≈ v·t ≈ 2 m
        let s0 = solve(&real, &round, mv, eye, DVec3::ZERO, target, DVec3::ZERO, 3.0).unwrap();
        assert!(true_miss(&real, &round, mv, eye, DVec3::ZERO, s0.dir, target, tv, 3.0) > 1.5);
    }
}
