//! Zero-g navigation: 6-DOF jetpack steering in the rotating frame.
//!
//! Near the axis there is no "down": a free body drifts on a straight line
//! in the inertial frame, which the rotating frame sees as a spiral outwards
//! (centrifugal + Coriolis). The steering law therefore *cancels the
//! fictitious forces* and then flies a braking-limited velocity profile
//! (v ≤ √(2·a·d), with a margin) to the goal, detouring around the spindle
//! (radius 30 m) whose surface would otherwise be struck at the frame speed.

use gameplay::frame::StationFrame;
use gameplay::geom::{self, SPINDLE_R};
use gameplay::Body;
use glam::DVec3;

/// Stay this far from the spindle surface.
pub const SPINDLE_CLEARANCE: f64 = 12.0;

/// Intermediate goal that keeps the straight segment pos → goal away from the spindle.
pub fn detour(pos: DVec3, goal: DVec3) -> DVec3 {
    let a = pos.truncate();
    let b = goal.truncate();
    let ab = b - a;
    let l2 = ab.length_squared();
    let t = if l2 > 1e-9 { (-a.dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let c = a + ab * t;
    let need = SPINDLE_R + SPINDLE_CLEARANCE;
    if c.length() >= need || t <= 0.0 || t >= 1.0 {
        return goal;
    }
    // push the closest point out to a safe radius (perpendicular to the segment
    // if it passes straight through the axis) and go there first
    let mut n = c.normalize_or_zero();
    if n == glam::DVec2::ZERO {
        n = glam::DVec2::new(-ab.y, ab.x).normalize_or_zero();
    }
    let p = n * (need + 10.0);
    let z = pos.z + (goal.z - pos.z) * t;
    DVec3::new(p.x, p.y, z)
}

/// Desired thrust acceleration (world frame) to reach `goal` and stop there.
pub fn steer(frame: &StationFrame, pos: DVec3, vel: DVec3, goal: DVec3, goal_vel: DVec3, max_acc: f64) -> DVec3 {
    let target = detour(pos, goal);
    let to = target - pos;
    let d = to.length();
    let final_leg = target == goal;
    // braking-limited approach speed, using 60 % of the available thrust
    let v_max = (2.0 * 0.6 * max_acc * d).sqrt().min(25.0);
    let v_des = if d > 1e-6 { to / d * v_max } else { DVec3::ZERO } + if final_leg { goal_vel } else { DVec3::ZERO };
    let k = 1.2; // 1/s velocity loop gain
    let mut a = (v_des - vel) * k - frame.accel(pos, vel);
    // emergency repulsion from the spindle
    let r = geom::r_perp(pos);
    let need = SPINDLE_R + SPINDLE_CLEARANCE * 0.5;
    if r < need && r > 1e-6 {
        let out = DVec3::new(pos.x / r, pos.y / r, 0.0);
        let v_in = -vel.dot(out);
        a += out * (max_acc * (1.0 - r / need) * 4.0 + v_in.max(0.0) * 2.0);
    }
    let l = a.length();
    if l > max_acc {
        a * (max_acc / l)
    } else {
        a
    }
}

/// Converts a world-frame thrust wish into the command axes of a jetpack body
/// (forward, right, up ∈ [−1, 1]); `max_acc` maps to full stick.
pub fn to_axes(b: &Body, a: DVec3, max_acc: f64) -> (f64, f64, f64) {
    let (r, u, f) = (b.orient * DVec3::X, b.orient * DVec3::Y, b.orient * DVec3::Z);
    let s = 1.0 / max_acc.max(1e-9);
    (a.dot(f) * s, -a.dot(r) * s, a.dot(u) * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fly(start: DVec3, goal: DVec3, secs: f64) -> (DVec3, DVec3, f64) {
        let frame = StationFrame::colony();
        let (mut p, mut v) = (start, frame.frame_velocity(start) * 0.0);
        let dt = 1.0 / 60.0;
        let max_acc = 3.2;
        let mut min_r = f64::MAX;
        for _ in 0..(secs / dt) as usize {
            let a = steer(&frame, p, v, goal, DVec3::ZERO, max_acc) + frame.accel(p, v);
            v += a * dt;
            p += v * dt;
            min_r = min_r.min(geom::r_perp(p));
        }
        (p, v, min_r)
    }

    #[test]
    fn reaches_a_point_near_the_axis_and_holds() {
        let start = DVec3::new(300.0, 0.0, 16_000.0);
        let goal = DVec3::new(0.0, 180.0, 16_040.0);
        let (p, v, _) = fly(start, goal, 150.0);
        assert!((p - goal).length() < 1.5, "off by {}", (p - goal).length());
        assert!(v.length() < 0.3, "still moving {}", v.length());
    }

    #[test]
    fn detours_around_the_spindle() {
        let start = DVec3::new(120.0, 0.0, 16_000.0);
        let goal = DVec3::new(-120.0, 0.0, 16_000.0);
        let (p, _, min_r) = fly(start, goal, 200.0);
        assert!(min_r > SPINDLE_R + 3.0, "came within {min_r} m of the axis");
        assert!((p - goal).length() < 2.0, "off by {}", (p - goal).length());
    }
}
