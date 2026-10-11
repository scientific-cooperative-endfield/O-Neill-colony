//! Bots for the O'Neill colony shooter.
//!
//! * `nav` — A* grid on the unrolled cylinder floor (arc length × axis, wraps in u)
//! * `nav3d` — 6-DOF jetpack steering in the rotating frame (spindle avoidance)
//! * `perception` — vision cone + line of sight, hearing with sound delay / vacuum
//! * `aim` — fire control with the bot's own (possibly naive) ballistic model
//! * `brain` — utility AI (patrol, investigate, attack, cover, retreat, flank)
//! * `bot` — glue: produces ordinary `Command`s
#![allow(clippy::too_many_arguments)]

pub mod aim;
pub mod bot;
pub mod brain;
pub mod config;
pub mod nav;
pub mod nav3d;
pub mod perception;

pub use bot::{Bot, Bots};
pub use config::Difficulty;
