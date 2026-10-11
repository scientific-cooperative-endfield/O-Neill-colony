//! wgpu renderer of the black-hole mode: compute passes trace → resolve →
//! bilateral upscale → metering/exposure → bloom → final, then the egui HUD and
//! either a blit to the window surface or a PNG readback (headless).
//! Every pass is bracketed by timestamp queries when the adapter supports them.

use super::{blackbody, disk_tex, noise::f16_bits, sky};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const COMMON: &str = include_str!("../../shaders/blackhole/common.wgsl");
pub const TRACE_WGSL: &str = include_str!("../../shaders/blackhole/trace.wgsl");
pub const RESOLVE_WGSL: &str = include_str!("../../shaders/blackhole/resolve.wgsl");
pub const UPSCALE_WGSL: &str = include_str!("../../shaders/blackhole/upscale.wgsl");
pub const POST_WGSL: &str = include_str!("../../shaders/blackhole/post.wgsl");
pub const BLIT_WGSL: &str = include_str!("../../shaders/blackhole/blit.wgsl");

pub fn with_common(body: &str) -> String {
    format!("{COMMON}\n{body}")
}

pub const PASS_NAMES: [&str; 6] = ["трассировка", "небо", "апскейл", "экспозиция", "bloom", "финал"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    pub scale: f32,
    pub max_steps: u32,
    pub spp: u32,
    pub halo: bool,
    pub sky_size: u32,
    pub bloom_levels: usize,
}

pub const QUALITY_NAMES: [&str; 4] = ["Низкие", "Средние", "Высокие", "Ультра"];
pub const QUALITY: [Quality; 4] = [
    Quality { scale: 0.5, max_steps: 256, spp: 1, halo: false, sky_size: 512, bloom_levels: 4 },
    Quality { scale: 0.75, max_steps: 384, spp: 1, halo: true, sky_size: 1024, bloom_levels: 5 },
    Quality { scale: 1.0, max_steps: 512, spp: 1, halo: true, sky_size: 1024, bloom_levels: 6 },
    Quality { scale: 1.0, max_steps: 512, spp: 4, halo: true, sky_size: 1024, bloom_levels: 6 },
];

pub fn quality_index(name: &str) -> usize {
    match name.to_ascii_lowercase().as_str() {
        "low" | "низкие" => 0,
        "medium" | "средние" => 1,
        "ultra" | "ультра" => 3,
        _ => 2,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub quality_idx: usize,
    pub quality: Quality,
    pub step_scale: f32,
    pub r_in: f32,
    pub r_out: f32,
    pub t_peak: f32,
    /// Kerr spin a/M (0 = Schwarzschild tracer with the full PG time model)
    pub spin: f32,
    pub brightness: f32,
    pub opacity: f32,
    pub halo: f32,
    pub halo_h: f32,
    pub sky_intensity: f32,
    pub auto_exposure: bool,
    pub exposure_ev: f32,
    pub adapt_speed: f32,
    pub bloom: f32,
    pub chromatic: f32,
    pub grain: f32,
    pub vignette: f32,
    pub tonemap: u32,
    pub debug_zones: bool,
    pub debug_steps: bool,
    pub debug_g: bool,
    pub debug_clocks: bool,
    pub debug_stress: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            quality_idx: 2,
            quality: QUALITY[2],
            step_scale: 1.0,
            r_in: 6.0,
            r_out: 20.0,
            t_peak: 6000.0,
            spin: 0.0,
            brightness: 1.0,
            opacity: 0.85,
            halo: 0.12,
            halo_h: 0.08,
            sky_intensity: 1.0,
            auto_exposure: true,
            exposure_ev: 0.0,
            adapt_speed: 1.5,
            bloom: 0.04,
            chromatic: 0.004,
            grain: 0.25,
            vignette: 0.15,
            tonemap: 0,
            debug_zones: false,
            debug_steps: false,
            debug_g: false,
            debug_clocks: false,
            debug_stress: false,
        }
    }
}

/// Per-frame camera and scene state (BH frame, M = 1).
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInput {
    pub cam_pos: [f32; 3],
    pub cam_fwd: [f32; 3],
    pub cam_up: [f32; 3],
    pub cam_right: [f32; 3],
    pub fov_y_deg: f32,
    /// camera velocity relative to the local rain frame
    pub beta: [f32; 3],
    pub disk_time: f32,
    pub beacon_pos: [f32; 3],
    pub beacon_radius: f32,
    /// contravariant PG 4-velocity (u^T, u^x, u^y, u^z)
    pub beacon_u: [f32; 4],
    pub tidal_radius: f32,
    pub dt: f32,
    pub frame: u32,
    pub instant_exposure: bool,
    /// station hull / snapshot parameters (from `HullGpu::upload`)
    pub hull: super::hull_gpu::HullFrame,
    pub show_hull: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct TraceParams {
    cam_pos: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_fwd: [f32; 4],
    cam_beta: [f32; 4],
    disk: [f32; 4],
    disk2: [f32; 4],
    beacon: [f32; 4],
    beacon_u: [f32; 4],
    beacon_col: [f32; 4],
    size: [u32; 4],
    flags: [u32; 4],
    misc: [f32; 4],
    hull: [f32; 4],
    hull_light: [f32; 4],
    heat: [f32; 4],
    kerr: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct PostParams {
    size: [u32; 4],
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub timestamps: bool,
    pub surface: Option<(wgpu::Surface<'static>, wgpu::SurfaceConfiguration)>,
}

/// Borrowed device/queue: lets the tracer run on a device owned by another app
/// (the colony renders its environment probe with it).
pub struct GpuRef<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub timestamps: bool,
    pub surface_format: Option<wgpu::TextureFormat>,
}

impl Gpu {
    pub fn r(&self) -> GpuRef<'_> {
        GpuRef { device: &self.device, queue: &self.queue, timestamps: self.timestamps, surface_format: self.surface.as_ref().map(|(_, c)| c.format) }
    }

    pub async fn new(window: Option<Arc<winit::window::Window>>, vsync: bool) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..Default::default()
        });
        let surface = match &window {
            Some(w) => Some(instance.create_surface(w.clone())?),
            None => None,
        };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable GPU adapter (Vulkan driver / Mesa lavapipe installed?)")?;
        let info = adapter.get_info();
        log::info!("adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let features = if timestamps { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bh-device"),
                    required_features: features,
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;
        let surface = match (surface, window) {
            (Some(s), Some(w)) => {
                let caps = s.get_capabilities(&adapter);
                let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
                let size = w.inner_size();
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format,
                    width: size.width.max(1),
                    height: size.height.max(1),
                    present_mode: if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync },
                    desired_maximum_frame_latency: 2,
                    alpha_mode: caps.alpha_modes[0],
                    view_formats: vec![format.remove_srgb_suffix()],
                };
                s.configure(&device, &config);
                Some((s, config))
            }
            _ => None,
        };
        Ok(Gpu { device, queue, adapter_name: format!("{} ({:?})", info.name, info.backend), timestamps, surface })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if let Some((s, c)) = &mut self.surface {
            c.width = w.max(1);
            c.height = h.max(1);
            s.configure(&self.device, c);
        }
    }
}

/// Small non-blocking GPU → CPU readback.
struct Readback {
    buf: wgpu::Buffer,
    state: Arc<Mutex<u8>>, // 0 idle, 1 mapping, 2 mapped
}

impl Readback {
    fn new(device: &wgpu::Device, size: u64, label: &str) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Readback { buf, state: Arc::new(Mutex::new(0)) }
    }
    fn idle(&self) -> bool {
        *self.state.lock().expect("readback lock") == 0
    }
    fn request(&self) {
        *self.state.lock().expect("readback lock") = 1;
        let st = self.state.clone();
        self.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *st.lock().expect("readback lock") = if r.is_ok() { 2 } else { 0 };
        });
    }
    fn take(&self) -> Option<Vec<u8>> {
        let mut s = self.state.lock().expect("readback lock");
        if *s != 2 {
            return None;
        }
        let v = self.buf.slice(..).get_mapped_range().to_vec();
        self.buf.unmap();
        *s = 0;
        Some(v)
    }
}

struct Targets {
    hdr_low_tex: wgpu::Texture,
    full: (u32, u32),
    low: (u32, u32),
    tiles: (u32, u32),
    final_tex: wgpu::Texture,
    final_view: wgpu::TextureView,
    bloom_sizes: Vec<(u32, u32)>,
    trace_bg: wgpu::BindGroup,
    resolve_bg: wgpu::BindGroup,
    upscale_bg: wgpu::BindGroup,
    meter_bg: wgpu::BindGroup,
    expose_bg: wgpu::BindGroup,
    down_bgs: Vec<wgpu::BindGroup>,
    up_bgs: Vec<wgpu::BindGroup>,
    final_bg: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
    key: (u32, u32, u32, usize),
}

pub struct BhRenderer {
    trace: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    upscale: wgpu::ComputePipeline,
    meter: wgpu::ComputePipeline,
    expose: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    up: wgpu::ComputePipeline,
    final_p: wgpu::ComputePipeline,
    blit: Option<wgpu::RenderPipeline>,
    sky_view: wgpu::TextureView,
    sky_size: u32,
    sky_mips: u32,
    lut_view: wgpu::TextureView,
    disk_view: wgpu::TextureView,
    samp_rep: wgpu::Sampler,
    samp_clamp: wgpu::Sampler,
    params_buf: wgpu::Buffer,
    post_buf: wgpu::Buffer,
    stats_buf: wgpu::Buffer,
    expo_buf: wgpu::Buffer,
    tiles_buf: wgpu::Buffer,
    pub hull: super::hull_gpu::HullGpu,
    stats_rb: Readback,
    ts: Option<(wgpu::QuerySet, wgpu::Buffer, Readback)>,
    ts_period: f32,
    targets: Option<Targets>,
    /// when set, `render` stops after the resolve pass and copies the linear HDR
    /// result into this cube face (environment probe of the colony)
    env_target: Option<(Arc<wgpu::Texture>, u32)>,
    pub egui: egui_wgpu::Renderer,
    /// GPU time per pass (ms), from timestamp queries
    pub timings_ms: [f32; 6],
    pub mean_steps: f32,
    pub exposure: f32,
}

fn tex2d(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn compute(device: &wgpu::Device, src: &str, entry: &str, label: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: entry,
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bg(device: &wgpu::Device, p: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries })
}

fn tv(b: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::TextureView(v) }
}
fn sm(b: u32, s: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::Sampler(s) }
}
fn bu(b: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() }
}

fn groups(n: u32, g: u32) -> u32 {
    n.div_ceil(g)
}

const STORE: wgpu::TextureUsages = wgpu::TextureUsages::STORAGE_BINDING.union(wgpu::TextureUsages::TEXTURE_BINDING);

impl BhRenderer {
    pub fn new(gpu: &GpuRef, settings: &Settings) -> BhRenderer {
        let device = &gpu.device;
        let trace = compute(device, &with_common(TRACE_WGSL), "main", "bh-trace");
        let resolve = compute(device, &with_common(RESOLVE_WGSL), "main", "bh-resolve");
        let upscale = compute(device, &with_common(UPSCALE_WGSL), "main", "bh-upscale");
        let meter = compute(device, POST_WGSL, "meter", "bh-meter");
        let expose = compute(device, POST_WGSL, "expose", "bh-expose");
        let down = compute(device, POST_WGSL, "down", "bh-down");
        let up = compute(device, POST_WGSL, "up", "bh-up");
        let final_p = compute(device, POST_WGSL, "final_main", "bh-final");
        let blit = gpu.surface_format.map(|fmt| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blit"), source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()) });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit"),
                layout: None,
                vertex: wgpu::VertexState { module: &module, entry_point: "vs", buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: "fs_unorm",
                    targets: &[Some(wgpu::ColorTargetState { format: fmt.remove_srgb_suffix(), blend: None, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        });
        let samp_rep = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rep"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let samp_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // black-body LUT (RGBA16F, filterable)
        let lut: Vec<u16> = blackbody::lut().iter().flat_map(|c| [f16_bits(c[0]), f16_bits(c[1]), f16_bits(c[2]), f16_bits(1.0)]).collect();
        let lut_tex = device.create_texture_with_data(
            gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("bb-lut"),
                size: wgpu::Extent3d { width: blackbody::LUT_SIZE as u32, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&lut),
        );
        let lut_view = lut_tex.create_view(&Default::default());
        // disk turbulence texture with mips
        let dt = disk_tex::generate(0xd15c);
        let disk = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("disk"),
            size: wgpu::Extent3d { width: disk_tex::W as u32, height: disk_tex::H as u32, depth_or_array_layers: 1 },
            mip_level_count: dt.mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (lvl, (w, h, data)) in dt.mips.iter().enumerate() {
            gpu.queue.write_texture(
                wgpu::ImageCopyTexture { texture: &disk, mip_level: lvl as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                data,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(*w as u32 * 4), rows_per_image: Some(*h as u32) },
                wgpu::Extent3d { width: *w as u32, height: *h as u32, depth_or_array_layers: 1 },
            );
        }
        let disk_view = disk.create_view(&Default::default());
        let (sky_view, sky_size, sky_mips) = Self::upload_sky(gpu, settings.quality.sky_size);
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<TraceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let post_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post"),
            size: std::mem::size_of::<PostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let expo_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("expo"),
            contents: bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let tiles_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiles"),
            size: 8 * 16384,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let stats_rb = Readback::new(device, 32, "stats-rb");
        let ts = gpu.timestamps.then(|| {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("ts"), ty: wgpu::QueryType::Timestamp, count: 12 });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts-resolve"),
                size: 8 * 12,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            (qs, resolve, Readback::new(device, 8 * 12, "ts-rb"))
        });
        let egui = egui_wgpu::Renderer::new(device, wgpu::TextureFormat::Rgba8Unorm, None, 1, false);
        BhRenderer {
            trace,
            resolve,
            upscale,
            meter,
            expose,
            down,
            up,
            final_p,
            blit,
            sky_view,
            sky_size,
            sky_mips,
            lut_view,
            disk_view,
            samp_rep,
            samp_clamp,
            params_buf,
            post_buf,
            stats_buf,
            expo_buf,
            tiles_buf,
            hull: super::hull_gpu::HullGpu::new(device),
            env_target: None,
            stats_rb,
            ts,
            ts_period: gpu.queue.get_timestamp_period(),
            targets: None,
            egui,
            timings_ms: [0.0; 6],
            mean_steps: 0.0,
            exposure: 1.0,
        }
    }

    fn upload_sky(gpu: &GpuRef, size: u32) -> (wgpu::TextureView, u32, u32) {
        let s = sky::load_or_generate(size);
        let mips = s.mips.len() as u32;
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut sz = size;
        for (lvl, faces) in s.mips.iter().enumerate() {
            for (f, data) in faces.iter().enumerate() {
                gpu.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &tex,
                        mip_level: lvl as u32,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: f as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(data),
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(sz * 8), rows_per_image: Some(sz) },
                    wgpu::Extent3d { width: sz, height: sz, depth_or_array_layers: 1 },
                );
            }
            sz = (sz / 2).max(1);
        }
        let view = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
        (view, size, mips)
    }

    fn ensure_targets(&mut self, gpu: &GpuRef, w: u32, h: u32, s: &Settings) {
        if s.quality.sky_size != self.sky_size {
            let (v, sz, m) = Self::upload_sky(gpu, s.quality.sky_size);
            self.sky_view = v;
            self.sky_size = sz;
            self.sky_mips = m;
            self.targets = None;
        }
        let lw = ((w as f32 * s.quality.scale).round() as u32).max(1);
        let lh = ((h as f32 * s.quality.scale).round() as u32).max(1);
        let key = (w, h, lw * 65536 + lh, s.quality.bloom_levels);
        if self.targets.as_ref().is_some_and(|t| t.key == key) {
            return;
        }
        let d = &gpu.device;
        let f16 = wgpu::TextureFormat::Rgba16Float;
        let emit = tex2d(d, "emit", lw, lh, f16, STORE).create_view(&Default::default());
        let dir = tex2d(d, "dir", lw, lh, f16, STORE).create_view(&Default::default());
        let hdr_low_tex = tex2d(d, "hdr-low", lw, lh, f16, STORE | wgpu::TextureUsages::COPY_SRC);
        let hdr_low = hdr_low_tex.create_view(&Default::default());
        let hdr_full = tex2d(d, "hdr-full", w, h, f16, STORE).create_view(&Default::default());
        let final_tex =
            tex2d(d, "final", w, h, wgpu::TextureFormat::Rgba8Unorm, STORE | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let final_view = final_tex.create_view(&Default::default());
        let n = s.quality.bloom_levels.max(2);
        let mut bloom_sizes = Vec::new();
        let (mut bw, mut bh) = (w, h);
        for _ in 0..n {
            bw = (bw / 2).max(1);
            bh = (bh / 2).max(1);
            bloom_sizes.push((bw, bh));
        }
        let downs: Vec<wgpu::TextureView> =
            bloom_sizes.iter().map(|&(a, b)| tex2d(d, "bloom-down", a, b, f16, STORE).create_view(&Default::default())).collect();
        let ups: Vec<wgpu::TextureView> =
            bloom_sizes[..n - 1].iter().map(|&(a, b)| tex2d(d, "bloom-up", a, b, f16, STORE).create_view(&Default::default())).collect();
        let trace_bg = bg(
            d,
            &self.trace,
            &[
                bu(0, &self.params_buf),
                sm(1, &self.samp_rep),
                tv(2, &self.disk_view),
                tv(3, &self.lut_view),
                tv(4, &self.sky_view),
                sm(5, &self.samp_clamp),
                tv(6, &emit),
                tv(7, &dir),
                bu(8, &self.stats_buf),
                bu(9, &self.hull.snaps_buf),
                bu(10, &self.hull.nodes_buf),
            ],
        );
        let resolve_bg = bg(
            d,
            &self.resolve,
            &[bu(0, &self.params_buf), tv(1, &emit), tv(2, &dir), tv(3, &self.sky_view), sm(4, &self.samp_clamp), tv(5, &self.lut_view), tv(6, &hdr_low)],
        );
        let upscale_bg = bg(d, &self.upscale, &[bu(0, &self.params_buf), tv(1, &hdr_low), tv(2, &hdr_full)]);
        let meter_bg = bg(d, &self.meter, &[bu(0, &self.post_buf), tv(1, &hdr_full), bu(2, &self.tiles_buf)]);
        let expose_bg = bg(d, &self.expose, &[bu(0, &self.post_buf), bu(2, &self.tiles_buf), bu(3, &self.expo_buf)]);
        let mut down_bgs = Vec::new();
        for i in 0..n {
            let src = if i == 0 { &hdr_full } else { &downs[i - 1] };
            down_bgs.push(bg(d, &self.down, &[bu(0, &self.post_buf), tv(4, src), sm(5, &self.samp_clamp), tv(6, &downs[i])]));
        }
        // up_bgs[i] writes ups[i] = downs[i] + tent(src), src = downs[n-1] or ups[i+1]
        let mut up_bgs = Vec::new();
        for i in 0..n - 1 {
            let src = if i == n - 2 { &downs[n - 1] } else { &ups[i + 1] };
            up_bgs.push(bg(d, &self.up, &[tv(4, src), sm(5, &self.samp_clamp), tv(6, &ups[i]), tv(7, &downs[i])]));
        }
        let final_bg = bg(
            d,
            &self.final_p,
            &[bu(0, &self.post_buf), tv(8, &hdr_full), tv(9, &ups[0]), sm(10, &self.samp_clamp), bu(11, &self.expo_buf), tv(12, &final_view)],
        );
        let blit_bg = match &self.blit {
            Some(p) => d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit"),
                layout: &p.get_bind_group_layout(0),
                entries: &[tv(0, &final_view), sm(1, &self.samp_clamp)],
            }),
            None => d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit-dummy"),
                layout: &self.upscale.get_bind_group_layout(0),
                entries: &[bu(0, &self.params_buf), tv(1, &hdr_low), tv(2, &hdr_full)],
            }),
        };
        self.targets = Some(Targets {
            hdr_low_tex,
            full: (w, h),
            low: (lw, lh),
            tiles: (groups(w, 64), groups(h, 64)),
            final_tex,
            final_view,
            bloom_sizes,
            trace_bg,
            resolve_bg,
            upscale_bg,
            meter_bg,
            expose_bg,
            down_bgs,
            up_bgs,
            final_bg,
            blit_bg,
            key,
        });
    }

    pub fn trace_size(&self) -> (u32, u32) {
        self.targets.as_ref().map(|t| t.low).unwrap_or((0, 0))
    }

    /// Collects finished asynchronous readbacks (timings, step statistics).
    pub fn poll_readbacks(&mut self) {
        if let Some(v) = self.stats_rb.take() {
            let steps = u32::from_le_bytes(v[0..4].try_into().expect("4 bytes"));
            let (lw, lh) = self.trace_size();
            self.mean_steps = steps as f32 / (lw * lh).max(1) as f32;
            self.exposure = f32::from_le_bytes(v[16..20].try_into().expect("4 bytes"));
        }
        if let Some((_, _, rb)) = &self.ts {
            if let Some(v) = rb.take() {
                let t: Vec<u64> = v.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
                for i in 0..6 {
                    let dt = t[2 * i + 1].saturating_sub(t[2 * i]) as f32 * self.ts_period * 1e-6;
                    self.timings_ms[i] = dt;
                }
            }
        }
    }

    fn pass<'a>(&'a self, enc: &'a mut wgpu::CommandEncoder, idx: u32, label: &str) -> wgpu::ComputePass<'a> {
        let tw = self.ts.as_ref().map(|(qs, _, _)| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(2 * idx),
            end_of_pass_write_index: Some(2 * idx + 1),
        });
        enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: tw })
    }

    /// Traces one 90° face of an environment cube map (linear HDR, `size`²) into
    /// layer `face` of `cube` (Rgba16Float, COPY_DST). Only trace + resolve run.
    pub fn render_env_face(&mut self, gpu: &GpuRef, size: u32, s: &Settings, f: &FrameInput, cube: &Arc<wgpu::Texture>, face: u32) -> wgpu::CommandEncoder {
        let mut s2 = s.clone();
        s2.quality.scale = 1.0;
        s2.quality.spp = 1;
        self.env_target = Some((Arc::clone(cube), face));
        let enc = self.render(gpu, size, size, &s2, f);
        self.env_target = None;
        enc
    }

    /// Encodes the black-hole frame into the final RGBA8 target (w × h).
    pub fn render(&mut self, gpu: &GpuRef, w: u32, h: u32, s: &Settings, f: &FrameInput) -> wgpu::CommandEncoder {
        self.ensure_targets(gpu, w, h, s);
        let t = self.targets.as_ref().expect("targets");
        let (lw, lh) = t.low;
        let th = (f.fov_y_deg.to_radians() * 0.5).tan();
        let b2 = f.beta[0] * f.beta[0] + f.beta[1] * f.beta[1] + f.beta[2] * f.beta[2];
        let gamma = 1.0 / (1.0 - b2.min(0.999_999)).sqrt();
        let mut debug = 0u32;
        if s.debug_zones {
            debug |= 1;
        }
        if s.debug_steps {
            debug |= 2;
        }
        if s.debug_g {
            debug |= 8;
        }
        if s.debug_clocks {
            debug |= 16;
        }
        if s.debug_stress {
            debug |= 32;
        }
        if f.show_hull {
            debug |= 64;
        }
        if s.quality.halo && s.halo > 0.0 {
            debug |= 4;
        }
        let tp = TraceParams {
            cam_pos: [f.cam_pos[0], f.cam_pos[1], f.cam_pos[2], th],
            cam_right: [f.cam_right[0], f.cam_right[1], f.cam_right[2], w as f32 / h as f32],
            cam_up: [f.cam_up[0], f.cam_up[1], f.cam_up[2], f.disk_time],
            cam_fwd: [f.cam_fwd[0], f.cam_fwd[1], f.cam_fwd[2], gamma],
            cam_beta: [f.beta[0], f.beta[1], f.beta[2], 1000.0],
            // Kerr: the disk starts at the ISCO of the spinning hole
            disk: {
                let r_in = if s.spin.abs() > 1e-4 { super::kerr::isco(s.spin.abs().min(0.998) as f64, s.spin >= 0.0) as f32 } else { s.r_in };
                [r_in, s.r_out.max(r_in + 0.1), s.t_peak, s.brightness]
            },
            disk2: [s.opacity, s.halo, s.halo_h, s.sky_intensity],
            beacon: [f.beacon_pos[0], f.beacon_pos[1], f.beacon_pos[2], f.beacon_radius],
            beacon_u: [f.beacon_u[1], f.beacon_u[2], f.beacon_u[3], f.beacon_u[0]],
            beacon_col: [4.0, 4.0, 4.0, 5200.0],
            size: [lw, lh, s.quality.max_steps, s.quality.spp],
            flags: [debug, f.frame, w, h],
            misc: [s.step_scale, std::f32::consts::FRAC_PI_2 / self.sky_size as f32, self.sky_mips as f32, f.tidal_radius],
            hull: [f.hull.n as f32, f.hull.near, f.hull.union_r, if f.hull.delay { 1.0 } else { 0.0 }],
            hull_light: f.hull.light,
            heat: f.hull.heat,
            kerr: if s.spin.abs() > 1e-4 {
                let a = s.spin.clamp(-0.998, 0.998) as f64;
                [a as f32, super::kerr::isco(a.abs(), a >= 0.0) as f32, super::kerr::horizon(a) as f32, 1.0]
            } else {
                [0.0; 4]
            },
        };
        gpu.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&tp));
        let n = t.bloom_sizes.len();
        let (emin, emax) = if s.auto_exposure { (1e-3, 1e4) } else { (2f32.powf(s.exposure_ev), 2f32.powf(s.exposure_ev)) };
        let pp = PostParams {
            size: [w, h, f.frame, s.tonemap],
            a: [f.dt, if s.auto_exposure { s.exposure_ev } else { 0.0 }, s.adapt_speed, if f.instant_exposure { 1.0 } else { 0.0 }],
            b: [s.bloom, s.chromatic, s.grain, s.vignette],
            c: [t.tiles.0 as f32, t.tiles.1 as f32, emin, emax],
            d: [1.0 / n as f32, 0.0, 0.0, 0.0],
        };
        gpu.queue.write_buffer(&self.post_buf, 0, bytemuck::bytes_of(&pp));
        gpu.queue.write_buffer(&self.stats_buf, 0, &[0u8; 16]);

        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bh") });
        {
            let mut p = self.pass(&mut enc, 0, "trace");
            p.set_pipeline(&self.trace);
            p.set_bind_group(0, &t.trace_bg, &[]);
            p.dispatch_workgroups(groups(lw, 8), groups(lh, 8), 1);
        }
        {
            let mut p = self.pass(&mut enc, 1, "resolve");
            p.set_pipeline(&self.resolve);
            p.set_bind_group(0, &t.resolve_bg, &[]);
            p.dispatch_workgroups(groups(lw, 8), groups(lh, 8), 1);
        }
        if let Some((cube, face)) = &self.env_target {
            enc.copy_texture_to_texture(
                wgpu::ImageCopyTexture { texture: &t.hdr_low_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::ImageCopyTexture {
                    texture: cube.as_ref(),
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: *face },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
            );
            return enc;
        }
        {
            let mut p = self.pass(&mut enc, 2, "upscale");
            p.set_pipeline(&self.upscale);
            p.set_bind_group(0, &t.upscale_bg, &[]);
            p.dispatch_workgroups(groups(w, 8), groups(h, 8), 1);
        }
        {
            let mut p = self.pass(&mut enc, 3, "exposure");
            p.set_pipeline(&self.meter);
            p.set_bind_group(0, &t.meter_bg, &[]);
            p.dispatch_workgroups(t.tiles.0, t.tiles.1, 1);
            p.set_pipeline(&self.expose);
            p.set_bind_group(0, &t.expose_bg, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut p = self.pass(&mut enc, 4, "bloom");
            p.set_pipeline(&self.down);
            for (i, b) in t.down_bgs.iter().enumerate() {
                p.set_bind_group(0, b, &[]);
                let (bw, bh) = t.bloom_sizes[i];
                p.dispatch_workgroups(groups(bw, 8), groups(bh, 8), 1);
            }
            p.set_pipeline(&self.up);
            for i in (0..n - 1).rev() {
                p.set_bind_group(0, &t.up_bgs[i], &[]);
                let (bw, bh) = t.bloom_sizes[i];
                p.dispatch_workgroups(groups(bw, 8), groups(bh, 8), 1);
            }
        }
        {
            let mut p = self.pass(&mut enc, 5, "final");
            p.set_pipeline(&self.final_p);
            p.set_bind_group(0, &t.final_bg, &[]);
            p.dispatch_workgroups(groups(w, 8), groups(h, 8), 1);
        }
        if self.stats_rb.idle() {
            enc.copy_buffer_to_buffer(&self.stats_buf, 0, &self.stats_rb.buf, 0, 16);
            enc.copy_buffer_to_buffer(&self.expo_buf, 0, &self.stats_rb.buf, 16, 16);
        }
        if let Some((qs, resolve, rb)) = &self.ts {
            if rb.idle() {
                enc.resolve_query_set(qs, 0..12, resolve, 0);
                enc.copy_buffer_to_buffer(resolve, 0, &rb.buf, 0, 8 * 12);
            }
        }
        enc
    }

    /// Paints the egui overlay onto the final target.
    pub fn paint_ui(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, jobs: &[egui::ClippedPrimitive], textures: &egui::TexturesDelta, ppp: f32) {
        let Some(t) = self.targets.as_ref() else { return };
        for (id, delta) in &textures.set {
            self.egui.update_texture(&gpu.device, &gpu.queue, *id, delta);
        }
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [t.full.0, t.full.1], pixels_per_point: ppp };
        let extra = self.egui.update_buffers(&gpu.device, &gpu.queue, enc, jobs, &sd);
        if !extra.is_empty() {
            gpu.queue.submit(extra);
        }
        {
            let rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.final_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut rp = rp.forget_lifetime();
            self.egui.render(&mut rp, jobs, &sd);
        }
        for id in &textures.free {
            self.egui.free_texture(id);
        }
    }

    /// Submits the frame; afterwards requests the asynchronous readbacks.
    pub fn submit(&mut self, gpu: &Gpu, enc: wgpu::CommandEncoder) {
        gpu.queue.submit(Some(enc.finish()));
        if self.stats_rb.idle() {
            self.stats_rb.request();
        }
        if let Some((_, _, rb)) = &self.ts {
            if rb.idle() {
                rb.request();
            }
        }
    }

    pub fn blit(&self, enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let (Some(p), Some(t)) = (&self.blit, &self.targets) else { return };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(p);
        rp.set_bind_group(0, &t.blit_bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Blocking readback of the final RGBA8 frame (headless mode).
    pub fn read_frame(&self, gpu: &Gpu) -> Result<(u32, u32, Vec<u8>)> {
        let t = self.targets.as_ref().context("nothing rendered")?;
        let (w, h) = t.full;
        let row = (w * 4).div_ceil(256) * 256;
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("png-rb"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture: &t.final_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        gpu.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let data = buf.slice(..).get_mapped_range();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let s = (y * row) as usize;
            out.extend_from_slice(&data[s..s + (w * 4) as usize]);
        }
        drop(data);
        buf.unmap();
        Ok((w, h, out))
    }
}

pub fn write_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) -> Result<()> {
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header()?;
    wr.write_image_data(rgba)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
        if let Err(e) = v.validate(&module) {
            panic!("{name}: validation error:\n{}", e.emit_to_string(src));
        }
    }

    #[test]
    fn black_hole_shaders_validate() {
        validate("trace", &with_common(TRACE_WGSL));
        validate("resolve", &with_common(RESOLVE_WGSL));
        validate("upscale", &with_common(UPSCALE_WGSL));
        validate("post", POST_WGSL);
        validate("blit", BLIT_WGSL);
    }

    #[test]
    fn uniform_layouts_are_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<TraceParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<TraceParams>(), 17 * 16);
        assert_eq!(std::mem::size_of::<PostParams>(), 5 * 16);
    }
}
