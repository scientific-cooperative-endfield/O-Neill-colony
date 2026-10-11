//! Shared helpers turning a scenario / simulation state into renderer inputs
//! (used by the `blackhole` viewer and by the colony's black-hole mode).
use super::hull_gpu::HullFrame;
use super::render::{self, FrameInput, Settings};
use super::scenario::Scenario;
use super::sim::Sim;
use glam::DVec3;

pub fn settings_from(sc: &Scenario) -> Settings {
    let qi = render::quality_index(&sc.quality);
    Settings {
        quality_idx: qi,
        quality: render::QUALITY[qi],
        r_in: sc.disk.r_in,
        r_out: sc.disk.r_out,
        t_peak: sc.disk.t_peak,
        spin: sc.disk.spin,
        brightness: sc.disk.brightness,
        opacity: sc.disk.opacity,
        halo: sc.disk.halo,
        bloom: sc.post.bloom,
        chromatic: sc.post.chromatic,
        grain: sc.post.grain,
        exposure_ev: sc.post.exposure_ev,
        auto_exposure: sc.post.auto_exposure,
        tonemap: if sc.post.tonemap.eq_ignore_ascii_case("aces") { 1 } else { 0 },
        debug_zones: sc.debug.zones,
        debug_steps: sc.debug.steps,
        debug_g: sc.debug.redshift,
        ..Settings::default()
    }
}

fn v3(v: DVec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

pub fn frame_input(sim: &Sim, fov: f64, basis: (DVec3, DVec3, DVec3), marker: f32, dt: f64, instant: bool, hull: HullFrame) -> FrameInput {
    let (fwd, up, right) = basis;
    let u = sim.station_u();
    let visible = !sim.station.dead && marker > 0.0;
    FrameInput {
        cam_pos: v3(sim.cam.pos),
        cam_fwd: v3(fwd),
        cam_up: v3(up),
        cam_right: v3(right),
        fov_y_deg: fov as f32,
        beta: v3(sim.cam.beta),
        disk_time: sim.disk_time as f32,
        beacon_pos: v3(sim.station.x),
        beacon_radius: if visible { marker } else { 0.0 },
        beacon_u: [u[0] as f32, u[1] as f32, u[2] as f32, u[3] as f32],
        tidal_radius: sim.tidal_radius_m().min(1e6) as f32,
        dt: dt as f32,
        frame: sim.frame as u32,
        instant_exposure: instant,
        hull,
        show_hull: sim.station_cfg.hull,
    }
}
