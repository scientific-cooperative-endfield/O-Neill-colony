//! The "neuro-interface": the predicted path of the next round drawn into the
//! HUD — the full physics (Coriolis, centrifugal, drag, breach wind, holes),
//! next to the naive straight line, so the player *sees* the station spin.

use crate::ballistics::{self, Trajectory};
use crate::ecs::Entity;
use crate::sim::{Sim, DT};
use glam::DVec3;

#[derive(Clone, Debug, Default)]
pub struct NeuroHud {
    pub trajectory: Trajectory,
    /// where a straight line along the barrel would be at the same time of flight
    pub naive_point: DVec3,
    /// distance between the real impact and the naive point (m)
    pub deviation: f64,
}

/// Trajectory projection for player `e`, if their neuro-interface is on.
pub fn neuro_projection(sim: &Sim, e: Entity) -> Option<NeuroHud> {
    if !sim.ecs.pilots.get(e)?.neuro || !sim.is_alive(e) {
        return None;
    }
    let a = sim.ecs.arsenals.get(e)?;
    let w = a.current;
    let wc = &sim.cfg.weapons[w];
    let b = sim.ecs.bodies.get(e)?;
    let dir = sim.aim_dir(e);
    let p0 = sim.eye(e) + dir * 0.4;
    let v0 = b.vel + dir * wc.muzzle_velocity;
    let env = sim.env();
    let mut scratch = Vec::new();
    let tr = ballistics::predict(&env, sim.round(w), p0, v0, wc.max_flight_s.min(3.0), DT, &mut scratch);
    let end = *tr.points.last().unwrap_or(&p0);
    let naive = p0 + v0 * tr.time_of_flight;
    Some(NeuroHud { naive_point: naive, deviation: (end - naive).length(), trajectory: tr })
}
