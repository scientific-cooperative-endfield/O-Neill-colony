//! Player input as compact, quantised commands stamped with the tick they
//! apply to. The simulation consumes only these, so feeding the same command
//! stream reproduces the same match bit for bit — the basis for rollback
//! netcode or an authoritative server later. Look input is a *delta*
//! (quantised to 0.1 mrad), so it is valid for both walking (yaw/pitch about
//! the local "up") and 6-DOF jetpack flight (yaw/pitch/roll of the body).

use serde::{Deserialize, Serialize};

pub mod buttons {
    pub const FIRE: u16 = 1;
    pub const ADS: u16 = 1 << 1;
    pub const JUMP: u16 = 1 << 2;
    pub const CROUCH: u16 = 1 << 3;
    pub const RUN: u16 = 1 << 4;
    pub const RELOAD: u16 = 1 << 5;
    /// activate the curvature device in its current mode
    pub const DEVICE: u16 = 1 << 6;
    /// next device mode (edge-triggered)
    pub const DEVICE_NEXT: u16 = 1 << 7;
    /// toggle the neuro-interface trajectory projection (edge)
    pub const NEURO: u16 = 1 << 8;
    /// toggle the jetpack (edge)
    pub const JETPACK: u16 = 1 << 9;
    /// magnetic boots (hold)
    pub const MAG_BOOTS: u16 = 1 << 10;
}

/// Look quantum (rad).
pub const LOOK_Q: f64 = 1e-4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Command {
    pub tick: u32,
    pub player: u8,
    pub buttons: u16,
    /// forward, right, up ∈ [−127, 127]
    pub mv: [i8; 3],
    /// Δyaw, Δpitch, Δroll in units of `LOOK_Q`
    pub look: [i16; 3],
    /// 0 = keep, 1…n = select weapon slot
    pub weapon: u8,
}

impl Command {
    pub fn new(tick: u32, player: u8) -> Self {
        Command { tick, player, ..Default::default() }
    }
    pub fn axes(&self) -> [f64; 3] {
        [self.mv[0] as f64 / 127.0, self.mv[1] as f64 / 127.0, self.mv[2] as f64 / 127.0]
    }
    pub fn look_delta(&self) -> [f64; 3] {
        [self.look[0] as f64 * LOOK_Q, self.look[1] as f64 * LOOK_Q, self.look[2] as f64 * LOOK_Q]
    }
    pub fn held(&self, b: u16) -> bool {
        self.buttons & b != 0
    }
    pub fn set_move(&mut self, fwd: f64, right: f64, up: f64) {
        let q = |x: f64| (x.clamp(-1.0, 1.0) * 127.0).round() as i8;
        self.mv = [q(fwd), q(right), q(up)];
    }
    pub fn set_look(&mut self, d_yaw: f64, d_pitch: f64, d_roll: f64) {
        self.look = [quantize_look(d_yaw), quantize_look(d_pitch), quantize_look(d_roll)];
    }
}

pub fn quantize_look(rad: f64) -> i16 {
    (rad / LOOK_Q).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

/// Commands per tick and player with "repeat last" prediction for gaps.
#[derive(Clone, Debug, Default)]
pub struct CommandLog {
    pub cmds: std::collections::BTreeMap<(u32, u8), Command>,
}

impl CommandLog {
    pub fn push(&mut self, c: Command) {
        self.cmds.insert((c.tick, c.player), c);
    }
    pub fn for_tick(&self, tick: u32) -> Vec<Command> {
        self.cmds.range((tick, 0)..=(tick, u8::MAX)).map(|(_, c)| *c).collect()
    }
}
