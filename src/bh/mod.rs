//! Black-hole core shared by the `blackhole` viewer and the colony app:
//! Schwarzschild geometry in Painlevé–Gullstrand coordinates, the time model,
//! the station structure graph and the wgpu ray tracer.
pub mod blackbody;
pub mod disk_tex;
pub use ::structure::fracture;
pub mod frame;
pub mod hull_gpu;
pub mod kerr;
pub use ::structure::noise;
pub mod null_geo;
pub use relativity as physics;
pub mod render;
pub mod scenario;
pub mod sim;
pub mod sky;
pub use ::structure;
