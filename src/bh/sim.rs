//! Simulation clock, time modes, bodies (station fragments, camera) and the
//! snapshot ring buffer used for the light-travel delay.
//!
//! Everything is integrated in geometric units with M = 1 (lengths and times
//! in GM/c², GM/c³) in Painlevé–Gullstrand coordinates, whose time T is regular
//! at the horizon; Schwarzschild t = T − F(r) is derived only for the HUD and
//! for the distant-observer mode. Every body (station fragment, camera) stores
//! its proper time τ and 4-velocity (via `Geo`).
//! Time modes (a free multiplier ×0.01 … ×1000 applies to all of them):
//!  * Remote  — the clock is the distant observer's t: every body is advanced
//!    until its own t reaches the clock, so near the horizon it slows down,
//!    reddens, dims and freezes;
//!  * Station — the clock is the proper time τ of the station centre: it
//!    crosses the horizon after a finite τ;
//!  * Camera  — the clock is the camera's proper time: a camera hovering near
//!    the hole sees the outside universe speed up.
//!
//! In the Station and Camera modes all bodies share the PG time slice T of the
//! reference body.

use super::physics::{self, Geo, MassPreset, PRESETS};
use super::scenario::{CamKey, Scenario, StationCfg};
use super::structure::{self, Spec, Station};
use bytemuck::{Pod, Zeroable};
use glam::{DQuat, DVec3};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeMode {
    Remote,
    Station,
    Camera,
}

impl TimeMode {
    pub fn parse(s: &str) -> TimeMode {
        match s {
            "station" | "proper" | "comoving" | "global" => TimeMode::Station,
            "camera" => TimeMode::Camera,
            _ => TimeMode::Remote,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            TimeMode::Remote => "Удалённый наблюдатель (t)",
            TimeMode::Station => "Сопровождающий станцию (τ станции)",
            TimeMode::Camera => "Собственное время камеры (τ камеры)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CamKin {
    /// hovering at fixed r (accelerated observer)
    Static,
    /// free fall from rest at infinity
    Rain,
    /// circular geodesic
    Orbit,
    /// rides with the station
    Station,
}

impl CamKin {
    pub fn parse(s: &str) -> CamKin {
        match s {
            "rain" | "fall" => CamKin::Rain,
            "orbit" => CamKin::Orbit,
            "station" => CamKin::Station,
            _ => CamKin::Static,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            CamKin::Static => "неподвижный (висит на r)",
            CamKin::Rain => "свободное падение с бесконечности",
            CamKin::Orbit => "круговая орбита",
            CamKin::Station => "вместе со станцией",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Observer {
    pub pos: DVec3,
    pub tau: f64,
    pub dtau_dt: Option<f64>,
    /// velocity relative to the local rain frame
    pub beta: DVec3,
    /// geodesic of a free-falling / orbiting camera
    pub body: Geo,
    /// PG time of the camera event (M)
    pub t_pg: f64,
    /// distant-observer time at the start (M), for "elapsed outside"
    pub t0: f64,
    pub initialised: bool,
}

/// One hull panel as seen by the tracer (80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct NodeGpu {
    /// offset from the snapshot centre (M), w = kind (0 land, 1 window, 2 cap)
    pub p: [f32; 4],
    /// panel orientation (x tangential, y axial, z outward normal)
    pub q: [f32; 4],
    /// coordinate velocity dx/dT, w = clock rate dτ/dT
    pub w: [f32; 4],
    /// half extents (M), w = dτ/dt (−1 inside the horizon)
    pub h: [f32; 4],
    /// x = clock offset from the station centre (s), y = stress, z = fragment id
    pub e: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub t_pg: f64,
    pub centre: DVec3,
    /// 4-velocity of the reference (u^T, u^i)
    pub u: [f64; 4],
    pub radius: f64,
    pub nodes: Vec<NodeGpu>,
}

#[derive(Clone, Debug)]
pub struct Sim {
    pub mass_idx: usize,
    pub mass_solar: f64,
    pub mode: TimeMode,
    pub speed: f64,
    pub unit_seconds: bool,
    pub paused: bool,
    /// station centre (reference worldline of the structure)
    pub station: Geo,
    pub structure: Station,
    pub station_cfg: StationCfg,
    /// distant-observer clock t (M)
    pub t_remote: f64,
    pub disk_time: f64,
    pub frame: u64,
    pub cam: Observer,
    pub cam_kin: CamKin,
    pub real_time: f64,
    pub snapshots: VecDeque<Snapshot>,
    pub snapshot_dt: f64,
    pub max_snapshots: usize,
    pub light_delay: bool,
    /// station τ and camera τ at the start (for the HUD)
    pub tau0_station: f64,
    pub t0_station: f64,
}

pub fn init_station(c: &StationCfg) -> Geo {
    match c.init.as_str() {
        "radial" => Geo::at_rest(1.0, DVec3::new(c.r0.max(2.1), 0.0, 0.0)),
        "rain" => Geo::rain(DVec3::new(c.r0, 0.0, 0.0)),
        "circular" => circular(c.r0),
        "custom" => {
            let x = DVec3::from_array(c.pos);
            if x.length() > 2.0 {
                Geo::from_static_velocity(1.0, x, DVec3::from_array(c.vel))
            } else {
                Geo::rain(x)
            }
        }
        _ => physics::elliptic_orbit(1.0, c.rp.max(4.1).min(c.ra), c.ra.max(c.rp)),
    }
}

/// Circular geodesic at radius r on the +x axis, moving towards +y (r > 3M).
pub fn circular(r: f64) -> Geo {
    let r = r.max(3.05);
    let v = (1.0 / (r - 2.0)).sqrt();
    Geo::from_static_velocity(1.0, DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, v, 0.0))
}

fn station_spec(c: &StationCfg, seed: u64) -> Spec {
    Spec { safety: c.safety, spin_g: c.spin_g, seed, ..Spec::default() }
}

fn station_axis(c: &StationCfg, g: &Geo) -> DVec3 {
    let rh = g.x.normalize_or(DVec3::X);
    let n = g.angular_momentum().normalize_or(DVec3::Z);
    match c.axis.as_str() {
        "normal" => n,
        "tangent" => n.cross(rh).normalize_or(DVec3::Y),
        _ => rh,
    }
}

/// Newton iteration in PG time: advances `g` until its Schwarzschild t reaches
/// `target`. The step dT needed diverges at the horizon, so the body freezes
/// just outside, as the distant observer sees it. Returns the PG step taken.
fn advance_to_t(g: &Geo, target: f64) -> f64 {
    let mut st = *g;
    let t_start = st.t_pg;
    if st.r() <= 2.0 || st.dead {
        return 0.0;
    }
    for _ in 0..24 {
        let Some(t) = st.t_schw(1.0) else { break };
        let rem = target - t;
        if rem <= 1e-9 * (1.0 + target.abs()) {
            break;
        }
        let Some((_, dt_dtt)) = physics::rates(1.0, st.x, st.coord_velocity(1.0)) else { break };
        let mut dtt = rem / dt_dtt.max(1e-12);
        dtt = dtt.min(rem * 4.0 + 1e-6);
        let before = st;
        st.step_t(1.0, dtt);
        if st.r() <= 2.0 {
            st = before;
            break;
        }
    }
    st.t_pg - t_start
}

impl Sim {
    pub fn new(sc: &Scenario) -> Sim {
        let station = init_station(&sc.station);
        let structure = Station::new(station_spec(&sc.station, sc.seed), station, station_axis(&sc.station, &station));
        let t0 = station.t_schw(1.0).unwrap_or(0.0);
        let mut s = Sim {
            mass_idx: sc.mass_preset.min(PRESETS.len() - 1),
            mass_solar: sc.mass_solar,
            mode: TimeMode::parse(&sc.time.mode),
            speed: sc.time.speed,
            unit_seconds: sc.time.unit == "s",
            paused: false,
            t_remote: t0,
            station,
            structure,
            station_cfg: sc.station.clone(),
            disk_time: 0.0,
            frame: 0,
            cam: Observer {
                pos: DVec3::new(30.0, 0.0, 0.0),
                tau: 0.0,
                dtau_dt: None,
                beta: DVec3::ZERO,
                body: Geo::rain(DVec3::new(30.0, 0.0, 0.0)),
                t_pg: 0.0,
                t0: 0.0,
                initialised: false,
            },
            cam_kin: CamKin::parse(&sc.camera.kinematics),
            real_time: 0.0,
            snapshots: VecDeque::new(),
            snapshot_dt: sc.time.snapshot_dt.max(1e-6),
            max_snapshots: sc.time.snapshots.clamp(2, 64),
            light_delay: sc.time.light_delay,
            tau0_station: 0.0,
            t0_station: t0,
        };
        if let Some(k) = sc.camera.keys.first() {
            s.place_camera(k);
        }
        // start the camera on the same distant-observer time slice as the station
        if let Some(f) = physics::f_shift(1.0, s.cam.pos.length()) {
            s.cam.t_pg = t0 + f;
            s.cam.body.t_pg = s.cam.t_pg;
        }
        if sc.time.start > 0.0 {
            s.advance_clock(sc.time.start);
        }
        s.tau0_station = s.station.tau;
        s.t0_station = s.station.t_schw(1.0).unwrap_or(s.t_remote);
        s.cam.t0 = s.cam_t().unwrap_or(0.0);
        s.capture_snapshot(true);
        s
    }

    pub fn preset(&self) -> MassPreset {
        PRESETS[self.mass_idx]
    }
    pub fn solar_masses(&self) -> f64 {
        if self.mass_solar > 0.0 {
            self.mass_solar
        } else {
            self.preset().solar
        }
    }
    pub fn mass_label(&self) -> String {
        if self.mass_solar > 0.0 {
            format!("{:.3e} M☉", self.mass_solar)
        } else {
            self.preset().name.to_string()
        }
    }
    /// GM/c² in metres
    pub fn m_metres(&self) -> f64 {
        physics::m_geo(self.solar_masses())
    }
    /// GM/c³ in seconds
    pub fn m_seconds(&self) -> f64 {
        self.m_metres() / physics::C
    }

    pub fn reset_station(&mut self) {
        self.station = init_station(&self.station_cfg);
        self.structure = Station::new(station_spec(&self.station_cfg, self.structure.spec.seed), self.station, station_axis(&self.station_cfg, &self.station));
        self.t_remote = self.station.t_schw(1.0).unwrap_or(0.0);
        self.tau0_station = 0.0;
        self.t0_station = self.t_remote;
        self.cam.initialised = false;
        self.snapshots.clear();
        self.capture_snapshot(true);
    }

    /// Schwarzschild t of the camera event (None inside the horizon).
    pub fn cam_t(&self) -> Option<f64> {
        physics::f_shift(1.0, self.cam.pos.length()).map(|f| self.cam.t_pg - f)
    }

    fn cam_static_rate(&self) -> f64 {
        let r = self.cam.pos.length().max(2.0005);
        physics::dtau_dt_static(1.0, r)
    }

    /// PG step of the camera that corresponds to a station PG step (shared slice).
    fn advance_camera_pg(&mut self, dtt: f64) {
        match self.cam_kin {
            CamKin::Static => {
                self.cam.tau += self.cam_static_rate() * dtt;
                self.cam.t_pg += dtt;
            }
            CamKin::Rain | CamKin::Orbit => {
                let t0 = self.cam.body.tau;
                self.cam.body.step_t(1.0, dtt);
                self.cam.tau += self.cam.body.tau - t0;
                self.cam.t_pg = self.cam.body.t_pg;
                self.cam.pos = self.cam.body.x;
            }
            CamKin::Station => {
                self.cam.tau = self.station.tau;
                self.cam.t_pg = self.station.t_pg;
            }
        }
    }

    /// Advances the selected clock by `d` (M).
    pub fn advance_clock(&mut self, d: f64) {
        let m = self.m_metres();
        match self.mode {
            TimeMode::Remote => {
                let target = self.t_remote + d;
                let dtt = advance_to_t(&self.station, target);
                self.structure.step(m, dtt);
                // camera: also on the t slice of the distant observer
                match self.cam_kin {
                    CamKin::Static => {
                        if self.cam.pos.length() > 2.0 {
                            self.cam.tau += self.cam_static_rate() * d;
                            self.cam.t_pg += d;
                        }
                    }
                    CamKin::Rain | CamKin::Orbit => {
                        let dc = advance_to_t(&self.cam.body, target);
                        self.advance_camera_pg(dc);
                    }
                    CamKin::Station => {}
                }
                self.t_remote = target;
            }
            TimeMode::Station => {
                let mut probe = self.station;
                probe.step_tau(1.0, d);
                let dtt = probe.t_pg - self.station.t_pg;
                self.structure.step(m, dtt);
                self.advance_camera_pg(dtt);
            }
            TimeMode::Camera => {
                let dtt = match self.cam_kin {
                    CamKin::Static => d / self.cam_static_rate(),
                    CamKin::Rain | CamKin::Orbit => {
                        let mut probe = self.cam.body;
                        probe.step_tau(1.0, d);
                        probe.t_pg - self.cam.body.t_pg
                    }
                    CamKin::Station => {
                        let mut probe = self.station;
                        probe.step_tau(1.0, d);
                        probe.t_pg - self.station.t_pg
                    }
                };
                self.structure.step(m, dtt);
                self.advance_camera_pg(dtt);
            }
        }
        self.station = self.structure.reference;
        if self.cam_kin == CamKin::Station {
            self.cam.tau = self.station.tau;
            self.cam.t_pg = self.station.t_pg;
        }
        if self.mode != TimeMode::Remote {
            if let Some(t) = self.cam_t().or_else(|| self.station.t_schw(1.0)) {
                self.t_remote = self.t_remote.max(t);
            }
        }
        self.disk_time += d;
        self.capture_snapshot(false);
    }

    /// One fixed simulation step of `dt_real` seconds of wall-clock time.
    pub fn step(&mut self, dt_real: f64) {
        self.real_time += dt_real;
        self.frame += 1;
        if self.paused {
            return;
        }
        let mut d = dt_real * self.speed;
        if self.unit_seconds {
            d /= self.m_seconds();
        }
        self.advance_clock(d);
        self.update_camera_state();
    }

    fn init_camera_body(&mut self, pos: DVec3) {
        self.cam.body = match self.cam_kin {
            CamKin::Orbit if pos.length() > 3.05 => {
                // circular geodesic through pos in the plane containing ẑ
                let r = pos.length();
                let v = (1.0 / (r - 2.0)).sqrt();
                let rh = pos / r;
                let t = DVec3::Z.cross(rh).normalize_or(DVec3::Y);
                Geo::from_static_velocity(1.0, pos, t * v)
            }
            _ => Geo::rain(pos),
        };
        self.cam.body.t_pg = self.cam.t_pg;
        self.cam.body.tau = self.cam.tau;
        self.cam.pos = pos;
        self.cam.initialised = true;
    }

    /// Velocity relative to the rain frame and dτ/dt of the camera.
    pub fn update_camera_state(&mut self) {
        let r = self.cam.pos.length().max(1e-6);
        match self.cam_kin {
            CamKin::Static if r > 2.0005 => {
                // the static observer moves outwards at √(2M/r) through the rain frame
                self.cam.beta = self.cam.pos / r * (2.0 / r).sqrt();
                self.cam.dtau_dt = Some(physics::dtau_dt_static(1.0, r));
            }
            CamKin::Static => {
                self.cam.beta = DVec3::ZERO;
                self.cam.dtau_dt = None;
            }
            CamKin::Rain | CamKin::Orbit => {
                self.cam.beta = self.cam.body.rain_velocity(1.0);
                self.cam.dtau_dt = self.cam.body.dtau_dt(1.0);
            }
            CamKin::Station => {
                self.cam.beta = self.station.rain_velocity(1.0);
                self.cam.dtau_dt = self.station.dtau_dt(1.0);
            }
        }
    }

    /// Places the camera from a key and returns (forward, up, right).
    pub fn place_camera(&mut self, k: &CamKey) -> (DVec3, DVec3, DVec3) {
        let (i, a) = (k.incl.to_radians(), k.azim.to_radians());
        let dir = DVec3::new(i.cos() * a.cos(), i.cos() * a.sin(), i.sin());
        let pos = match self.cam_kin {
            CamKin::Station => {
                let off = if k.offset_km > 0.0 { k.offset_km * 1000.0 / self.m_metres() } else { k.r };
                self.station.x + dir * off
            }
            CamKin::Static => dir * k.r,
            CamKin::Rain | CamKin::Orbit => {
                if !self.cam.initialised {
                    self.init_camera_body(dir * k.r);
                }
                self.cam.body.x
            }
        };
        self.cam.pos = pos;
        self.update_camera_state();
        let target = if k.look_station { self.station.x } else { DVec3::ZERO };
        let mut fwd = (target - pos).normalize_or(DVec3::NEG_X);
        let world_up = if fwd.z.abs() > 0.999 { DVec3::Y } else { DVec3::Z };
        let mut right = fwd.cross(world_up).normalize();
        let mut up = right.cross(fwd);
        let rot = |v: DVec3, axis: DVec3, ang: f64| DQuat::from_axis_angle(axis, ang) * v;
        let yaw = k.yaw.to_radians();
        fwd = rot(fwd, up, -yaw);
        right = rot(right, up, -yaw);
        let pitch = k.pitch.to_radians();
        fwd = rot(fwd, right, pitch);
        up = rot(up, right, pitch);
        let roll = k.roll.to_radians();
        right = rot(right, fwd, roll);
        up = rot(up, fwd, roll);
        (fwd, up, right)
    }

    /// Free-flight camera: hovers wherever the user puts it.
    pub fn set_camera_pos(&mut self, pos: DVec3) {
        if self.cam_kin != CamKin::Static {
            self.cam_kin = CamKin::Static;
        }
        self.cam.pos = pos;
        self.update_camera_state();
    }

    /// Station centre 4-velocity (u^T, u^x, u^y, u^z) in PG coordinates.
    pub fn station_u(&self) -> [f64; 4] {
        self.station.four_velocity(1.0)
    }

    /// Tidal stretch across the station length at its centre (in g).
    pub fn station_tidal_g(&self) -> f64 {
        let m = self.m_metres();
        let r = self.station.r() * m;
        2.0 * m * physics::C * physics::C * self.structure.spec.length / (r * r * r) / physics::G_EARTH
    }

    /// Tidal disruption radius of the station at 1 g end-to-end, in M.
    pub fn tidal_radius_m(&self) -> f64 {
        let m = self.m_metres();
        physics::tidal_radius(m, self.structure.spec.length, physics::G_EARTH) / m
    }

    /// Distant-observer time elapsed since the start at the camera (s or M as `unit_seconds`).
    pub fn elapsed_outside(&self) -> f64 {
        self.cam_t().map(|t| t - self.cam.t0).unwrap_or(self.t_remote - self.cam.t0)
    }

    /// Builds the GPU view of the hull, relative to `centre` (M).
    pub fn hull_nodes(&self, centre: DVec3) -> Vec<NodeGpu> {
        let m = self.m_metres();
        let st = &self.structure;
        let m_s = self.m_seconds();
        st.nodes
            .iter()
            .map(|n| {
                let f = &st.frags[n.frag];
                let q = (f.q * DQuat::from_mat3(&n.axes)).normalize();
                let off = n.x - centre;
                let kind = match n.kind {
                    structure::Kind::Land => 0.0,
                    structure::Kind::Window => 1.0,
                    structure::Kind::Cap => 2.0,
                };
                let dtdt = physics::rates(1.0, n.x, n.w).map(|(a, b)| a / b).unwrap_or(-1.0);
                let desync = (f.body.tau + n.dtau - st.reference.tau) * m_s;
                NodeGpu {
                    p: [off.x as f32, off.y as f32, off.z as f32, kind],
                    q: [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
                    w: [n.w.x as f32, n.w.y as f32, n.w.z as f32, n.rate as f32],
                    h: [(n.half.x / m) as f32, (n.half.y / m) as f32, (n.half.z / m) as f32, dtdt as f32],
                    e: [desync as f32, n.stress as f32, n.frag as f32, if f.alive { 1.0 } else { 0.0 }],
                }
            })
            .collect()
    }

    /// Records the current state; history entries are kept every `snapshot_dt` (M).
    pub fn capture_snapshot(&mut self, force: bool) {
        let centre = self.station.x;
        let radius = self.structure.extent(self.m_metres()) / self.m_metres();
        let snap = Snapshot { t_pg: self.station.t_pg, centre, u: self.station_u(), radius, nodes: self.hull_nodes(centre) };
        let due = self.snapshots.len() < 2 || snap.t_pg - self.snapshots[1].t_pg >= self.snapshot_dt;
        if self.snapshots.is_empty() {
            self.snapshots.push_front(snap);
            return;
        }
        if due || force {
            // the live entry becomes history
            self.snapshots.push_front(snap);
            while self.snapshots.len() > self.max_snapshots {
                self.snapshots.pop_back();
            }
        } else {
            self.snapshots[0] = snap;
        }
    }

    /// FNV-1a hash of the full simulation state (determinism test).
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |v: f64| {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        let s = &self.station;
        for v in [s.x.x, s.x.y, s.x.z, s.p.x, s.p.y, s.p.z, s.tau, s.t_pg, self.t_remote, self.disk_time, self.cam.tau, self.frame as f64] {
            eat(v);
        }
        self.structure.hash_into(&mut eat);
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(mode: &str) -> Scenario {
        let mut s = Scenario::default();
        s.time.mode = mode.into();
        s.time.speed = 40.0;
        s.station.init = "radial".into();
        s.station.r0 = 12.0;
        s
    }

    #[test]
    fn deterministic_state_hash() {
        let run = || {
            let sc = scenario("remote");
            let mut s = Sim::new(&sc);
            for _ in 0..600 {
                s.step(sc.dt);
            }
            s.state_hash()
        };
        let a = run();
        assert_eq!(a, run());
        let mut sc = scenario("remote");
        sc.station.r0 = 12.5;
        let mut s = Sim::new(&sc);
        for _ in 0..600 {
            s.step(sc.dt);
        }
        assert_ne!(a, s.state_hash());
    }

    #[test]
    fn remote_observer_never_sees_horizon_crossing() {
        let sc = scenario("remote");
        let mut s = Sim::new(&sc);
        let mut last_r = s.station.r();
        for _ in 0..20_000 {
            s.step(sc.dt);
            assert!(s.station.r() > 2.0, "crossed in remote mode");
            assert!(s.station.r() <= last_r + 1e-9);
            last_r = s.station.r();
        }
        // frozen just outside, with a vanishing dτ/dt
        assert!(s.station.r() < 2.05, "r = {}", s.station.r());
        assert!(s.station.dtau_dt(1.0).unwrap() < 0.05);
    }

    #[test]
    fn comoving_observer_crosses_horizon() {
        let sc = scenario("station");
        let mut s = Sim::new(&sc);
        for _ in 0..2000 {
            s.step(sc.dt);
        }
        assert!(s.station.r() < 2.0, "r = {}", s.station.r());
        // proper time to the horizon matches the cycloid solution
        let tau_h = physics::fall_time_analytic(1.0, 12.0, 2.0);
        assert!(s.station.tau > tau_h);
    }

    #[test]
    fn hovering_camera_sees_the_universe_speed_up() {
        // camera mode: 1 M of camera proper time at r = 2.05 M lasts
        // 1/√(1 − 2/2.05) M of distant-observer time
        let mut sc = scenario("camera");
        sc.time.speed = 1.0;
        sc.camera.kinematics = "static".into();
        sc.camera.keys = vec![CamKey { r: 2.05, ..CamKey::default() }];
        let mut s = Sim::new(&sc);
        let t0 = s.cam_t().unwrap();
        let tau0 = s.cam.tau;
        for _ in 0..60 {
            s.step(1.0 / 60.0);
        }
        let dtau = s.cam.tau - tau0;
        let dt = s.cam_t().unwrap() - t0;
        let want = 1.0 / (1.0 - 2.0 / 2.05f64).sqrt();
        assert!(((dt / dtau) - want).abs() / want < 1e-3, "{} vs {want}", dt / dtau);
        assert!((s.elapsed_outside() - dt).abs() < 1e-9);
    }

    #[test]
    fn rain_camera_rate_matches_formula() {
        let mut sc = scenario("camera");
        sc.time.speed = 0.2;
        sc.camera.kinematics = "rain".into();
        sc.camera.keys = vec![CamKey { r: 10.0, incl: 0.0, ..CamKey::default() }];
        let mut s = Sim::new(&sc);
        s.step(1.0 / 60.0);
        let r = s.cam.pos.length();
        let want = 1.0 - 2.0 / r;
        let got = s.cam.dtau_dt.unwrap();
        assert!((got - want).abs() / want < 1e-3, "{got} vs {want}");
    }

    #[test]
    fn snapshots_ring_buffer() {
        let mut sc = scenario("station");
        sc.time.snapshot_dt = 0.5;
        sc.time.speed = 60.0;
        let mut s = Sim::new(&sc);
        for _ in 0..200 {
            s.step(1.0 / 60.0);
            if s.station.r() < 4.0 {
                break;
            }
        }
        assert!(s.snapshots.len() > 4 && s.snapshots.len() <= 64);
        // newest first, spaced by ≥ snapshot_dt in T
        for w in s.snapshots.iter().skip(1).collect::<Vec<_>>().windows(2) {
            assert!(w[0].t_pg - w[1].t_pg >= 0.5 - 1e-9);
        }
        assert_eq!(s.snapshots[0].nodes.len(), structure::N_NODES);
    }
}
