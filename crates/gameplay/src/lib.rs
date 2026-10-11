//! Gameplay core of the O'Neill colony shooter (solo vs. bots now, netcode later).
//!
//! No wgpu / winit: everything here runs headless and deterministically.
//!
//! * `sim` — fixed 60 Hz tick, command-driven, clonable state (rollback), state hash
//! * `ecs` — minimal sparse-set ECS
//! * `frame`, `ballistics`, `fields` — the physics: rotating frame with
//!   Coriolis / centrifugal / black-hole PW tides, G7 drag, spacetime devices
//! * `player`, `damage`, `map` — controller, health, the district
//! * `hud` — neuro-interface trajectory projection
// physics kernels take many scalar/vector arguments by design (no hidden state)
#![allow(clippy::too_many_arguments)]

pub mod ballistics;
pub mod components;
pub mod config;
pub mod damage;
pub mod det;
pub mod ecs;
pub mod fields;
pub mod frame;
pub mod geom;
pub mod hud;
pub mod input;
pub mod map;
pub mod player;
pub mod sim;

pub use components::*;
pub use config::GameConfig;
pub use ecs::Entity;
pub use input::{buttons, Command};
pub use map::Arena;
pub use sim::{Event, Sim, DT, TICK_HZ};
