//! Station structure: panel graph with bonds, tidal break-up, venting
//! (`graph`, re-exported at the crate root), the hierarchical Voronoi
//! pre-fracture (`fracture`) and deterministic noise / RNG (`noise`).
pub mod fracture;
mod graph;
pub mod noise;

pub use graph::*;
