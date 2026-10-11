//! Deterministic scenarios (TOML): initial conditions, time mode, camera path
//! and capture frames. A fixed real-time step per frame (`dt`) and a fixed seed
//! make every run bit-identical.

use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Scenario {
    pub name: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    /// real seconds per frame (fixed step)
    pub dt: f64,
    pub quality: String,
    /// 0 = stellar (10 M☉), 1 = Sgr A*, 2 = supermassive (10⁸ M☉)
    pub mass_preset: usize,
    /// custom mass in solar masses (overrides the preset when > 0)
    pub mass_solar: f64,
    pub hud: bool,
    pub disk: DiskCfg,
    pub time: TimeCfg,
    pub station: StationCfg,
    pub camera: CameraCfg,
    pub post: PostCfg,
    pub debug: DebugCfg,
    /// frames to write (empty = all)
    pub capture: Vec<u32>,
    /// optional file names for captured frames (same order as `capture`)
    pub names: Vec<String>,
}

impl Default for Scenario {
    fn default() -> Self {
        Scenario {
            name: "default".into(),
            seed: 1,
            width: 1280,
            height: 720,
            frames: 1,
            dt: 1.0 / 60.0,
            quality: "high".into(),
            mass_preset: 0,
            mass_solar: 0.0,
            hud: true,
            disk: DiskCfg::default(),
            time: TimeCfg::default(),
            station: StationCfg::default(),
            camera: CameraCfg::default(),
            post: PostCfg::default(),
            debug: DebugCfg::default(),
            capture: Vec::new(),
            names: Vec::new(),
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct DiskCfg {
    pub r_in: f32,
    pub r_out: f32,
    pub t_peak: f32,
    pub brightness: f32,
    pub opacity: f32,
    pub halo: f32,
    /// Kerr spin a/M (0 = Schwarzschild; ≠ 0 switches to the Kerr–Schild tracer, disk from the ISCO)
    pub spin: f32,
}
impl Default for DiskCfg {
    fn default() -> Self {
        DiskCfg { r_in: 6.0, r_out: 20.0, t_peak: 6000.0, brightness: 1.0, opacity: 0.85, halo: 0.12, spin: 0.0 }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct TimeCfg {
    /// "remote" (coordinate time t), "station" (station τ) or "camera" (camera τ)
    pub mode: String,
    /// clock units per real second (×0.01 … ×1000)
    pub speed: f64,
    /// "M" (GM/c³) or "s" (physical seconds)
    pub unit: String,
    /// clock value at the start (M) — lets a control frame start mid-orbit
    pub start: f64,
    /// light-travel delay: interval between state snapshots (M) and their count
    pub snapshot_dt: f64,
    pub snapshots: usize,
    pub light_delay: bool,
}
impl Default for TimeCfg {
    fn default() -> Self {
        TimeCfg { mode: "remote".into(), speed: 1.0, unit: "M".into(), start: 0.0, snapshot_dt: 2.0, snapshots: 64, light_delay: true }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct StationCfg {
    /// "radial" (fall from rest at r0), "elliptic" (rp, ra), "custom" (pos, vel)
    pub init: String,
    pub r0: f64,
    pub rp: f64,
    pub ra: f64,
    /// position (M) and velocity measured by the local static observer (c)
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    /// marker sphere radius in M (0 = hidden)
    pub marker: f32,
    /// render the hull panels (structure graph) instead of / besides the marker
    pub hull: bool,
    /// station axis: "radial", "normal" (orbit normal) or "tangent"
    pub axis: String,
    /// strength / design load of the bonds, spin gravity (m/s²)
    pub safety: f64,
    pub spin_g: f64,
}
impl Default for StationCfg {
    fn default() -> Self {
        StationCfg {
            init: "elliptic".into(),
            r0: 30.0,
            rp: 8.0,
            ra: 30.0,
            pos: [30.0, 0.0, 0.0],
            vel: [0.0, 0.15, 0.0],
            marker: 0.25,
            hull: false,
            axis: "radial".into(),
            safety: 1.4,
            spin_g: 9.81,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct CameraCfg {
    /// "static" (hovering observer), "rain" (free fall from rest at infinity),
    /// "orbit" (circular geodesic), "station" (rides with the station, offset from the keys)
    pub kinematics: String,
    pub keys: Vec<CamKey>,
}
impl Default for CameraCfg {
    fn default() -> Self {
        CameraCfg { kinematics: "static".into(), keys: vec![CamKey::default()] }
    }
}

#[derive(Deserialize, Clone, Copy, Debug)]
#[serde(default)]
pub struct CamKey {
    /// real time (s) of the key
    pub time: f64,
    pub r: f64,
    /// elevation above the disk plane (degrees)
    pub incl: f64,
    pub azim: f64,
    pub fov: f64,
    /// camera offset from the station (km) for the "station" kinematics (0 = use r in M)
    pub offset_km: f64,
    /// look at the station instead of the hole
    pub look_station: bool,
    /// look offsets relative to the direction to the hole (degrees)
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
}
impl Default for CamKey {
    fn default() -> Self {
        CamKey { time: 0.0, r: 30.0, incl: 15.0, azim: 0.0, fov: 60.0, offset_km: 0.0, look_station: false, yaw: 0.0, pitch: 0.0, roll: 0.0 }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct PostCfg {
    pub bloom: f32,
    pub chromatic: f32,
    pub grain: f32,
    pub exposure_ev: f32,
    pub auto_exposure: bool,
    pub tonemap: String,
}
impl Default for PostCfg {
    fn default() -> Self {
        PostCfg { bloom: 0.04, chromatic: 0.004, grain: 0.25, exposure_ev: 0.0, auto_exposure: true, tonemap: "agx".into() }
    }
}

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct DebugCfg {
    pub zones: bool,
    pub steps: bool,
    pub redshift: bool,
    /// heat map of the panel clock rates dτ/dt
    pub clocks: bool,
}

/// Loads a scenario from a path, or by bare name from `scenarios/<name>.toml`.
pub fn load(path: &str) -> anyhow::Result<Scenario> {
    let named = format!("scenarios/{path}.toml");
    let path = if std::path::Path::new(path).is_file() || !std::path::Path::new(&named).is_file() { path } else { named.as_str() };
    let s = std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    toml::from_str(&s).map_err(|e| anyhow::anyhow!("{path}: {e}"))
}

/// Catmull-Rom interpolation of the camera keys (r in log space).
pub fn sample_keys(keys: &[CamKey], t: f64) -> CamKey {
    if keys.is_empty() {
        return CamKey::default();
    }
    if keys.len() == 1 || t <= keys[0].time {
        return keys[0];
    }
    let last = keys[keys.len() - 1];
    if t >= last.time {
        return last;
    }
    let i = keys.windows(2).position(|w| t >= w[0].time && t < w[1].time).unwrap_or(0);
    let k1 = keys[i];
    let k2 = keys[i + 1];
    let k0 = if i > 0 { keys[i - 1] } else { k1 };
    let k3 = if i + 2 < keys.len() { keys[i + 2] } else { k2 };
    let u = (t - k1.time) / (k2.time - k1.time).max(1e-9);
    let cr = |a: f64, b: f64, c: f64, d: f64| {
        let u2 = u * u;
        let u3 = u2 * u;
        0.5 * ((2.0 * b) + (-a + c) * u + (2.0 * a - 5.0 * b + 4.0 * c - d) * u2 + (-a + 3.0 * b - 3.0 * c + d) * u3)
    };
    CamKey {
        time: t,
        r: cr(k0.r.ln(), k1.r.ln(), k2.r.ln(), k3.r.ln()).exp(),
        incl: cr(k0.incl, k1.incl, k2.incl, k3.incl),
        azim: cr(k0.azim, k1.azim, k2.azim, k3.azim),
        fov: cr(k0.fov, k1.fov, k2.fov, k3.fov),
        offset_km: if k1.offset_km > 0.0 && k2.offset_km > 0.0 {
            cr(k0.offset_km.max(1e-3).ln(), k1.offset_km.ln(), k2.offset_km.ln(), k3.offset_km.max(1e-3).ln()).exp()
        } else {
            k1.offset_km
        },
        look_station: k1.look_station,
        yaw: cr(k0.yaw, k1.yaw, k2.yaw, k3.yaw),
        pitch: cr(k0.pitch, k1.pitch, k2.pitch, k3.pitch),
        roll: cr(k0.roll, k1.roll, k2.roll, k3.roll),
    }
}
