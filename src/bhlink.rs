//! Black-hole mode of the colony: the station of the `bh` simulation *is* this
//! colony. Each frame the exact simulation advances (PG geodesics, structure
//! graph, clocks), one face of an environment cube is ray traced from the
//! player's fragment (lensed sky, Doppler-boosted disk, aberration of the
//! falling frame), and the colony gets:
//! * the env cube behind the windows / through the gaps of the broken hull,
//! * light from the disk entering through the windows (it sweeps the valleys
//!   with the spin: a "day" lasts the ~10 s the source stays in a window),
//! * the tidal acceleration and, after a break-up, the rotating frame of the
//!   player's own fragment for the player and the balls,
//! * per-panel rigid transforms of every other fragment (vertex shaders),
//! * air pressure (venting) → haze, debris / glass / air particles.
use std::sync::Arc;

use anyhow::Result;
use glam::{DMat3, DQuat, DVec3, Mat3, Mat4, Vec3};
use oneill_colony::bh::frame::settings_from;
use oneill_colony::bh::hull_gpu::HullFrame;
use oneill_colony::bh::physics::{self, C};
use oneill_colony::bh::render::{BhRenderer, FrameInput, GpuRef, Settings};
use oneill_colony::bh::scenario::{self, Scenario};
use oneill_colony::bh::sim::{CamKin, Sim, TimeMode};
use oneill_colony::bh::structure::{N_CAP, N_NODES, N_THETA, N_Z};

use crate::colony::{self, L, R};
use crate::physics::FrameExt;
use crate::render::{LineVertex, NodeXf};

pub const DEFAULT_SCENARIO: &str = "colony_sgr";

/// Cube faces (forward, up, right) matching the WebGPU/Vulkan cube layout.
const FACES: [([f64; 3], [f64; 3], [f64; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
    ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]),
];

pub struct BhLink {
    pub sc: Scenario,
    pub sim: Sim,
    pub settings: Settings,
    r: BhRenderer,
    pub cube: Arc<wgpu::Texture>,
    pub env_size: u32,
    face: u32,
    pub faces_per_frame: u32,
    pub faces_done: u64,
    /// fragment the player stands on
    pub player_frag: usize,
    pub split: bool,
    pub env_gain: f32,
    /// smoothed light level for the eye adaptation
    pub adapt: f32,
    pub show_panel: bool,
    pub heatmap: bool,
    /// distant-observer t at the start (HUD shows the elapsed value)
    pub t_remote0: f64,
    readback: Readback,
    /// per cube face: Σ L·dΩ (rgb) and Σ lum·n̂·dΩ (BH frame)
    face_stats: [([f64; 3], DVec3); 6],
    pub stats_valid: u32,
}

/// Asynchronous readback of one cube face (environment light estimate).
struct Readback {
    buf: wgpu::Buffer,
    face: Option<u32>,
    ready: Arc<std::sync::atomic::AtomicBool>,
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Colony-frame offset of the structure body frame (its z runs −L/2…L/2).
fn c0() -> DVec3 {
    DVec3::new(0.0, 0.0, 0.5 * L as f64)
}

pub fn node_of(p: Vec3) -> usize {
    let th = colony::angle_of(p);
    if (p.z < 2.0 || p.z > L - 2.0) && p.truncate().length() < R - 2.0 {
        let k = ((th / (std::f32::consts::TAU / N_CAP as f32)) as usize).min(N_CAP - 1);
        return N_THETA * N_Z + if p.z > 0.5 * L { N_CAP } else { 0 } + k;
    }
    let iz = ((p.z / (L / N_Z as f32)).floor().clamp(0.0, (N_Z - 1) as f32)) as usize;
    let j = ((th / (std::f32::consts::TAU / N_THETA as f32)) as usize).min(N_THETA - 1);
    iz * N_THETA + j
}

fn v3(v: DVec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

impl BhLink {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, name: &str) -> Result<BhLink> {
        let sc = scenario::load(name)?;
        let settings = settings_from(&sc);
        let mut sim = Sim::new(&sc);
        // the camera is the player, riding the station
        sim.cam_kin = CamKin::Station;
        let gpu = GpuRef { device, queue, timestamps: false, surface_format: None };
        let r = BhRenderer::new(&gpu, &settings);
        let env_size = std::env::var("ONEILL_ENV").ok().and_then(|s| s.parse().ok()).unwrap_or(192u32);
        let cube = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bh env"),
            size: wgpu::Extent3d { width: env_size, height: env_size, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        // the player rides the station: camera clocks start on the station's
        sim.cam.pos = sim.station.x;
        sim.cam.t_pg = sim.station.t_pg;
        sim.cam.tau = sim.station.tau;
        sim.cam.t0 = sim.cam_t().unwrap_or(sim.t_remote);
        sim.update_camera_state();
        let t_remote0 = sim.t_remote;
        let readback = Readback {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("env readback"),
                size: (env_size * 8).div_ceil(256) as u64 * 256 * env_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            face: None,
            ready: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        log::info!("black-hole mode: scenario '{}' · {} · env cube {}²", sc.name, sim.mass_label(), env_size);
        Ok(BhLink {
            sc,
            sim,
            settings,
            r,
            cube,
            env_size,
            face: 0,
            faces_per_frame: 1,
            faces_done: 0,
            player_frag: 0,
            split: false,
            env_gain: 1.0,
            adapt: 0.0,
            show_panel: true,
            heatmap: false,
            t_remote0,
            readback,
            face_stats: [([0.0; 3], DVec3::ZERO); 6],
            stats_valid: 0,
        })
    }

    /// Fast-forwards the selected clock by `secs` seconds (or M when the scenario counts in M).
    pub fn skip(&mut self, secs: f64) {
        let total = if self.sim.unit_seconds { secs / self.sim.m_seconds() } else { secs };
        let n = (total / 0.02).ceil().max(1.0) as usize;
        for _ in 0..n {
            self.sim.advance_clock(total / n as f64);
        }
        self.sim.update_camera_state();
        let st = &self.sim.structure;
        log::info!(
            "skip {secs:.1}: station r = {:.3} M, τ = {:.2} s, fragments {}, first break {:?}, pressure {:.0} Pa",
            self.sim.station.r(),
            self.sim.station.tau * self.sim.m_seconds(),
            st.live_fragments(),
            st.first_break.as_ref().map(|b| b.r),
            st.pressure
        );
    }

    pub fn cube_view(&self) -> wgpu::TextureView {
        self.cube.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() })
    }

    /// Advances the simulation by `dt` real seconds; `player` in colony coordinates.
    pub fn update(&mut self, dt: f64, player: Vec3) {
        self.sim.step(dt.clamp(0.0, 0.5));
        let st = &self.sim.structure;
        let f = st.nodes[node_of(player)].frag;
        self.player_frag = if st.frags[f].alive { f } else { st.frags.iter().position(|f| f.alive).unwrap_or(0) };
        self.split = !st.intact();
    }

    fn pfrag(&self) -> &oneill_colony::bh::structure::Fragment {
        &self.sim.structure.frags[self.player_frag]
    }

    /// Rotation colony frame → black-hole frame.
    pub fn rot(&self) -> DQuat {
        self.pfrag().q
    }

    /// Rotating + tidal frame of the player's fragment for the colony physics.
    pub fn frame_ext(&self) -> FrameExt {
        let f = self.pfrag();
        let qi = f.q.inverse();
        let rm = DMat3::from_quat(f.q);
        let k = rm.transpose() * f.k * rm;
        FrameExt { spin: (qi * f.omega).as_vec3(), com: (f.com + c0()).as_vec3(), tidal: k.as_mat3() }
    }

    /// Per-node transforms from the rest geometry to the current place in the
    /// player's fragment frame.
    pub fn node_xf(&self) -> Vec<NodeXf> {
        let st = &self.sim.structure;
        let m = self.sim.m_metres();
        let p = self.pfrag();
        let qpi = p.q.inverse();
        let mut cache: Vec<Option<Mat4>> = vec![None; st.frags.len()];
        st.nodes
            .iter()
            .map(|n| {
                let fi = n.frag;
                let mm = *cache[fi].get_or_insert_with(|| {
                    let f = &st.frags[fi];
                    let delta = (f.body.x - p.body.x) * m;
                    let t1 = c0() + p.com + qpi * delta;
                    let rot = qpi * f.q;
                    let a = glam::DAffine3::from_translation(t1) * glam::DAffine3::from_quat(rot) * glam::DAffine3::from_translation(-(f.com + c0()));
                    glam::DMat4::from(a).as_mat4()
                });
                let detached = fi != self.player_frag;
                NodeXf { m: mm.to_cols_array_2d(), info: [fi as f32, detached as u32 as f32, 0.0, 0.0] }
            })
            .collect()
    }

    /// Renders `faces_per_frame` faces of the environment cube (all six on the first call).
    pub fn render_env(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let n = if self.faces_done == 0 { 6 } else { self.faces_per_frame };
        let gpu = GpuRef { device, queue, timestamps: false, surface_format: None };
        let (pos, beta) = {
            let f = self.pfrag();
            (f.body.x, if f.body.dead { DVec3::ZERO } else { f.body.rain_velocity(1.0) })
        };
        for _ in 0..n {
            let (fw, up, rt) = FACES[self.face as usize];
            let fi = FrameInput {
                cam_pos: v3(pos),
                cam_fwd: fw.map(|x| x as f32),
                cam_up: up.map(|x| x as f32),
                cam_right: rt.map(|x| x as f32),
                fov_y_deg: 90.0,
                beta: v3(beta),
                disk_time: self.sim.disk_time as f32,
                beacon_pos: [0.0; 3],
                beacon_radius: 0.0,
                beacon_u: [1.0, 0.0, 0.0, 0.0],
                tidal_radius: self.sim.tidal_radius_m().min(1e6) as f32,
                dt: 0.0,
                frame: self.sim.frame as u32,
                instant_exposure: true,
                hull: HullFrame::default(),
                show_hull: false,
            };
            let enc = self.r.render_env_face(&gpu, self.env_size, &self.settings, &fi, &self.cube, self.face);
            queue.submit(Some(enc.finish()));
            if self.readback.face.is_none() {
                self.request_readback(device, queue, self.face);
            }
            self.face = (self.face + 1) % 6;
            self.faces_done += 1;
        }
    }

    fn request_readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, face: u32) {
        let n = self.env_size;
        let bpr = (n * 8).div_ceil(256) * 256;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("env rb") });
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture: &self.cube, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: face }, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer {
                buffer: &self.readback.buf,
                layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(n) },
            },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let flag = self.readback.ready.clone();
        flag.store(false, std::sync::atomic::Ordering::SeqCst);
        self.readback.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        });
        self.readback.face = Some(face);
    }

    /// Integrates a finished face readback: radiance × solid angle per texel.
    pub fn poll_readback(&mut self, device: &wgpu::Device) {
        device.poll(wgpu::Maintain::Poll);
        let Some(face) = self.readback.face else { return };
        if !self.readback.ready.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let n = self.env_size as usize;
        let bpr = (n * 8).div_ceil(256) * 256;
        let (fw, up, rt) = FACES[face as usize];
        let (fw, up, rt) = (DVec3::from(fw), DVec3::from(up), DVec3::from(rt));
        let mut sum = [0.0f64; 3];
        let mut vec = DVec3::ZERO;
        let dump = std::env::var("ONEILL_ENV_DUMP").ok();
        let mut img = Vec::new();
        {
            let data = self.readback.buf.slice(..).get_mapped_range();
            for y in 0..n {
                let v = 1.0 - (y as f64 + 0.5) / n as f64 * 2.0;
                for x in 0..n {
                    let u = (x as f64 + 0.5) / n as f64 * 2.0 - 1.0;
                    let o = y * bpr + x * 8;
                    let px = |k: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * k], data[o + 2 * k + 1]])) as f64;
                    let (r, g, b) = (px(0), px(1), px(2));
                    if dump.is_some() {
                        let e: f64 = dump.as_deref().and_then(|d| d.parse().ok()).unwrap_or(1.0);
                        for c in [r, g, b] {
                            let t = c * e / (1.0 + c * e);
                            img.push((t.max(0.0).powf(1.0 / 2.2) * 255.0) as u8);
                        }
                        img.push(255);
                    }
                    if !(r.is_finite() && g.is_finite() && b.is_finite()) {
                        continue;
                    }
                    let d = fw + rt * u + up * v;
                    let l2 = d.length_squared();
                    let dw = 4.0 / (n * n) as f64 / (l2 * l2.sqrt());
                    sum[0] += r * dw;
                    sum[1] += g * dw;
                    sum[2] += b * dw;
                    vec += d / l2.sqrt() * ((0.2126 * r + 0.7152 * g + 0.0722 * b) * dw);
                }
            }
        }
        self.readback.buf.unmap();
        self.readback.face = None;
        if dump.is_some() {
            let path = format!("/data/env_face{face}.png");
            let _ = oneill_colony::bh::render::write_png(std::path::Path::new(&path), n as u32, n as u32, &img);
        }
        self.face_stats[face as usize] = (sum, vec);
        self.stats_valid |= 1 << face;
    }

    /// Light from the traced sky: (direction to the light in the colony
    /// frame, directional irradiance / π (rgb), mean sky radiance (rgb)).
    pub fn env_light(&self) -> Option<(Vec3, Vec3, Vec3)> {
        if self.stats_valid != 0x3f {
            return None;
        }
        let mut sum = DVec3::ZERO;
        let mut vec = DVec3::ZERO;
        for (s, v) in &self.face_stats {
            sum += DVec3::from(*s);
            vec += *v;
        }
        let lum = 0.2126 * sum.x + 0.7152 * sum.y + 0.0722 * sum.z;
        let dir_bh = vec.normalize_or(-self.pfrag().body.x.normalize_or(DVec3::X));
        let dir = (self.pfrag().q.inverse() * dir_bh).as_vec3();
        // directional part: |Σ L n̂ dΩ| with the sky colour
        let e_dir = vec.length() / lum.max(1e-12);
        let sun = (sum * e_dir / std::f64::consts::PI * self.env_gain as f64).as_vec3();
        // diffuse part: what is left after removing the directional flux
        // (a point source has |Σ L n̂ dΩ| = Σ L dΩ and no diffuse part)
        let diffuse = (1.0 - vec.length() / lum.max(1e-12)).clamp(0.0, 1.0);
        let mean = (sum * diffuse / (4.0 * std::f64::consts::PI) * self.env_gain as f64).as_vec3();
        Some((dir, sun, mean))
    }

    /// Direction to the hole (colony frame) and the disk-light colour/intensity
    /// arriving at the station (blackbody of the inner disk, Doppler factor of
    /// the falling frame to the 4th power, 1/r² dilution).
    pub fn light(&self) -> (Vec3, Vec3) {
        let f = self.pfrag();
        let x = f.body.x;
        let r = x.length().max(2.0);
        let s_bh = -x / r;
        let s = (f.q.inverse() * s_bh).as_vec3();
        if f.body.dead {
            return (s, Vec3::ZERO);
        }
        // energy of a photon coming radially from the hole, measured by the station
        let u = f.body.four_velocity(1.0);
        let k = physics::radial_photon_g(1.0, x, [u[0], u[1], u[2], u[3]]).unwrap_or(1.0);
        let t = self.settings.t_peak as f64 * k;
        let c = oneill_colony::bh::blackbody::rgb(t.clamp(800.0, 40000.0));
        let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let i = 9.0 * (6.0 / r).powi(2) * k.powi(4) * self.settings.brightness as f64;
        let col = DVec3::new(c[0], c[1], c[2]) / lum.max(1e-6) * i.min(400.0);
        (s, col.as_vec3())
    }

    /// Per-valley light directions: the source must be seen through the
    /// window opposite the valley (±15° in azimuth as seen from the floor).
    pub fn light_dirs(&self) -> ([[f32; 4]; 3], Vec3) {
        let (s, col) = match self.env_light() {
            Some((d, sun, _)) => (d, sun),
            None => self.light(),
        };
        let mut out = [[0.0f32; 4]; 3];
        let sxy = s.truncate();
        let horiz = sxy.length();
        for (k, o) in out.iter_mut().enumerate() {
            let up = colony::up_of(k as f32 * colony::STRIP_ANGLE).truncate();
            let c = if horiz > 1e-4 { sxy.dot(up) / horiz } else { 0.0 };
            let a = smooth(20f32.to_radians().cos(), 9f32.to_radians().cos(), c) * smooth(0.05, 0.3, horiz);
            *o = [s.x, s.y, s.z, a];
        }
        (out, col)
    }

    pub fn pressure_frac(&self) -> f32 {
        (self.sim.structure.pressure / self.sim.structure.spec.pressure).clamp(0.0, 1.0) as f32
    }

    /// Debris, glass and venting air as small glowing quads (colony frame).
    pub fn particles(&self, cam: Vec3, out: &mut Vec<LineVertex>, max: usize, light: Vec3) {
        let st = &self.sim.structure;
        if st.particles.is_empty() {
            return;
        }
        let m = self.sim.m_metres();
        let p = self.pfrag();
        let qpi = p.q.inverse();
        let dref = (p.body.x - st.reference.x) * m;
        let base = c0() + p.com;
        let lum = light.length().min(30.0) * 0.02 + 0.02;
        let mut n = 0;
        for q in st.particles.iter().filter(|q| q.life > q.age) {
            if n >= max {
                break;
            }
            let pc = (base + qpi * (q.d - dref)).as_vec3();
            let dist = pc.distance(cam);
            if !(dist.is_finite()) || dist > 120_000.0 {
                continue;
            }
            let fade = 1.0 - q.age / q.life.max(1e-3);
            let (col, size) = match q.kind {
                0 => ([0.55 * lum, 0.5 * lum, 0.45 * lum], q.size.max(2.0)),
                1 => ([1.6 * lum, 1.7 * lum, 2.0 * lum], q.size.max(1.0)),
                _ => ([0.25 * lum, 0.28 * lum, 0.32 * lum], q.size.max(30.0)),
            };
            let s = (size + dist * 0.0015) * 0.5;
            let to = (cam - pc).normalize_or_zero();
            let a = to.any_orthonormal_vector();
            let b = to.cross(a);
            let c = [col[0] * fade, col[1] * fade, col[2] * fade, 1.0];
            let v = |p: Vec3| LineVertex { pos: p.to_array(), color: c };
            let (p0, p1, p2, p3) = (pc - a * s - b * s, pc + a * s - b * s, pc + a * s + b * s, pc - a * s + b * s);
            out.extend_from_slice(&[v(p0), v(p1), v(p2), v(p0), v(p2), v(p3)]);
            n += 1;
        }
    }

    pub fn mode_label(&self) -> &'static str {
        self.sim.mode.label()
    }

    pub fn set_mode(&mut self, m: TimeMode) {
        self.sim.mode = m;
    }

    /// Spin gravity of the station (m/s²) and its angular rate — for the HUD.
    pub fn omega(&self) -> f64 {
        self.pfrag().omega.length()
    }

    /// Seconds per M.
    pub fn m_s(&self) -> f64 {
        self.sim.m_metres() / C
    }
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[allow(dead_code)]
fn _assert_nodes() {
    let _: [(); N_NODES] = [(); crate::render::N_NODES];
}
#[allow(dead_code)]
fn _unused(_: Mat3) {}

fn fmt_s(s: f64) -> String {
    let a = s.abs();
    if !s.is_finite() {
        "∞".into()
    } else if a < 1e-6 {
        format!("{:.2} нс", s * 1e9)
    } else if a < 1e-3 {
        format!("{:.2} мкс", s * 1e6)
    } else if a < 1.0 {
        format!("{:.2} мс", s * 1e3)
    } else if a < 600.0 {
        format!("{:.2} с", s)
    } else if a < 36_000.0 {
        format!("{:.1} мин", s / 60.0)
    } else if a < 86_400.0 * 400.0 {
        format!("{:.2} ч", s / 3600.0)
    } else {
        format!("{:.2} лет", s / 3.156e7)
    }
}

impl BhLink {
    /// HUD and controls of the black-hole mode (right-top panel).
    pub fn ui(&mut self, ctx: &egui::Context, player: Vec3, g_felt: f32) {
        let frame = egui::Frame::window(&ctx.style()).fill(egui::Color32::from_rgba_unmultiplied(20, 10, 8, 215));
        let m_s = self.m_s();
        let mut mode = self.sim.mode;
        egui::Window::new("Чёрная дыра").default_pos([900.0, 8.0]).default_width(360.0).frame(frame).show(ctx, |u| {
            u.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
            let s = &self.sim;
            let f = &s.structure.frags[self.player_frag];
            u.label(egui::RichText::new(s.mass_label()).color(egui::Color32::from_rgb(255, 200, 120)));
            let r = f.body.r();
            u.label(format!("r = {:.3} M ({:.0} тыс. км)   r_s = 2 M", r, r * s.m_metres() / 1e6));
            u.label(format!("t удалённый = {} (от старта)", fmt_s((s.t_remote - self.t_remote0) * m_s)));
            u.label(format!("τ станции   = {}", fmt_s(s.station.tau * m_s)));
            let node = &s.structure.nodes[node_of(player)];
            let tau_cam = (f.body.tau + node.dtau) * m_s;
            u.label(format!("τ камеры    = {}", fmt_s(tau_cam)));
            let rate_st = s.station.dtau_dt(1.0);
            let rate_cam = physics::rates(1.0, node.x, node.w).map(|(a, b)| a / b);
            let fr = |v: Option<f64>| v.map(|x| format!("{x:.6}")).unwrap_or_else(|| "— (под горизонтом)".into());
            u.label(format!("dτ/dt станции {}   камеры {}", fr(rate_st), fr(rate_cam)));
            u.label(format!("снаружи прошло {}", fmt_s(s.elapsed_outside() * m_s)));
            let (near_far, spread) = s.structure.clock_spread(s.m_metres());
            u.label(format!("часы: ближний − дальний конец {}  разброс {}", fmt_s(near_far), fmt_s(spread)));
            u.label(format!("прилив {:.3} g на длину · ощущаемое g {:.2}", s.station_tidal_g(), g_felt));
            let st = &s.structure;
            let state = if st.intact() {
                "корпус цел".to_string()
            } else {
                format!("РАЗРУШЕНИЕ: {} фрагм., вы на №{}", st.live_fragments(), self.player_frag)
            };
            u.label(egui::RichText::new(state).color(if st.intact() { egui::Color32::LIGHT_GREEN } else { egui::Color32::from_rgb(255, 90, 70) }));
            u.label(format!("давление {:.0} гПа · частиц {}", st.pressure / 100.0, st.particles.len()));
            if let Some(b) = &st.first_break {
                u.label(egui::RichText::new(format!("первый разрыв при r = {:.2} M", b.r)).weak());
            }
            u.separator();
            u.horizontal(|u| {
                for (m, l) in [(TimeMode::Station, "τ станции"), (TimeMode::Camera, "τ камеры"), (TimeMode::Remote, "t удалённый")] {
                    u.radio_value(&mut mode, m, l);
                }
            });
            u.horizontal(|u| {
                u.label("×");
                u.add(egui::Slider::new(&mut self.sim.speed, 0.01..=1000.0).logarithmic(true));
                u.checkbox(&mut self.sim.unit_seconds, "в секундах");
            });
            u.horizontal(|u| {
                u.checkbox(&mut self.sim.paused, "пауза");
                if u.button("сброс").clicked() {
                    self.sim.reset_station();
                }
            });
            u.horizontal(|u| {
                u.label("небо");
                u.add(egui::Slider::new(&mut self.env_gain, 0.05..=8.0).logarithmic(true));
                u.label("граней/кадр");
                u.add(egui::Slider::new(&mut self.faces_per_frame, 1..=6));
            });
        });
        self.sim.mode = mode;
    }
}
