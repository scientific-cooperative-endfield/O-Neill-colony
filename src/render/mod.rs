//! The renderer: GPU-driven grass and object culling (compute), cascaded LOD
//! terrain, the colony hull with space outside the windows, aircar traffic,
//! balls, a sun shadow map, HDR + bloom + tonemapping and the egui overlay.

pub mod meshes;
pub mod post;
pub mod props;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::colony::{self, CIRC, L, R, STRIP_ANGLE};
use crate::settings::Graphics;
use crate::shaders;
use crate::world::{self, ObjInstance, World, HH, HW, NUM_KINDS, ZH, ZW};
use meshes::{Mesh, Vertex};
use post::{Post, PostSettings, HDR_FORMAT};

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const FIELD_CELL: f32 = 0.25;
const MAX_LINE_VERTS: usize = 49152;
const MAX_TRAIN: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub shadow_mat: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub cam_surf: [f32; 4],
    pub light_dir: [[f32; 4]; 3],
    pub sun_color: [f32; 4],
    pub sky_color: [f32; 4],
    pub bounce_color: [f32; 4],
    pub colony: [f32; 4],
    pub wind: [f32; 4],
    pub frustum: [[f32; 4]; 6],
    pub grass: [f32; 4],
    pub grid: [i32; 4],
    pub field_base: [i32; 4],
    pub field_params: [f32; 4],
    pub balls: [[f32; 4]; 16],
    pub ball_vel: [[f32; 4]; 16],
    pub lights: [[f32; 4]; 4],
    pub params: [f32; 4],
    pub params2: [f32; 4],
    pub space: [f32; 4],
    /// black-hole mode: on, env gain, air pressure (0..1), hull split
    pub bh: [f32; 4],
    /// colony direction -> black-hole frame
    pub bh_rot: [[f32; 4]; 4],
    /// cloud cover, -, -, -
    pub bh_misc: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<Globals>() == 1232);

/// Rigid transform of one structure node (see `common.wgsl` `NodeXf`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NodeXf {
    pub m: [[f32; 4]; 4],
    pub info: [f32; 4],
}
pub const N_NODES: usize = 396;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct BallGpu {
    pub pos: [f32; 3],
    pub radius: f32,
    pub rot: [f32; 4],
    pub style: u32,
    pub emissive: f32,
    /// [stretch along the local y axis (0 or 1 = sphere), colour hue for styles ≥ 8]
    pub ext: [f32; 2],
}

/// Instanced spheres / ellipsoids per frame (balls, combatants, debris).
pub const MAX_RENDER_BALLS: usize = 2048;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct DrawIndexed {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

const MAX_RANGES: usize = 16384;

struct ObjTile {
    start: u32,
    count: u32,
    centre: Vec3,
    radius: f32,
    tree: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    counts: [u32; 4],
    kinds: [[f32; 4]; 16],
    buckets: [[u32; 4]; 48],
}

pub struct Gpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter: String,
}

impl Gpu {
    pub async fn new(window: Arc<winit::window::Window>, vsync: bool) -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable GPU adapter (Vulkan driver installed?)")?;
        let info = adapter.get_info();
        log::info!("adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            // COPY_SRC (where supported) lets F12 save screenshots
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | (caps.usages & wgpu::TextureUsages::COPY_SRC),
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);
        Ok(Gpu { surface, device, queue, config, adapter: format!("{} ({:?})", info.name, info.backend) })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.config.width = w.max(1);
        self.config.height = h.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.config.present_mode = if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        self.surface.configure(&self.device, &self.config);
    }
}

#[derive(Clone, Copy)]
struct MeshRange {
    first_index: u32,
    count: u32,
    base_vertex: i32,
}

struct Chunk {
    inst: [f32; 4],
    centre: Vec3,
    radius: f32,
    strip: usize,
}

struct ScenePipes {
    grass_near: wgpu::RenderPipeline,
    grass_far: wgpu::RenderPipeline,
    terrain: wgpu::RenderPipeline,
    shell: wgpu::RenderPipeline,
    objects: wgpu::RenderPipeline,
    balls: wgpu::RenderPipeline,
    car_body: wgpu::RenderPipeline,
    car_glow: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
}

struct Grass {
    near: wgpu::Buffer,
    far: wgpu::Buffer,
    cap: (u32, u32),
    grid_n: u32,
    near_bg: wgpu::BindGroup,
    far_bg: wgpu::BindGroup,
    cull_bg: [wgpu::BindGroup; 2],
}

struct Targets {
    size: [u32; 2],
    samples: u32,
    msaa: Option<wgpu::TextureView>,
    hdr: wgpu::TextureView,
    depth: wgpu::TextureView,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub blades_near: u32,
    pub blades_far: u32,
    pub objects: u32,
    pub chunks: u32,
}

struct Readback {
    buf: wgpu::Buffer,
    pending: Arc<AtomicBool>,
    in_flight: bool,
}

pub struct Renderer {
    /// save the next presented frame as PNG (F12 / `--screenshot`)
    pub screenshot: Option<std::path::PathBuf>,
    env_view: wgpu::TextureView,
    nodes_buf: wgpu::Buffer,
    modules: Modules,
    layouts: Layouts,
    globals: wgpu::Buffer,
    height_view: wgpu::TextureView,
    zone_view: wgpu::TextureView,
    lin_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    field_sampler: wgpu::Sampler,
    shadow: wgpu::TextureView,
    shadow_res: u32,
    dummy_shadow: wgpu::TextureView,
    scene_bg: wgpu::BindGroup,
    scene_bg_noshadow: wgpu::BindGroup,
    field_pipe: wgpu::ComputePipeline,
    cull_pipe: wgpu::ComputePipeline,
    cull_fin: wgpu::ComputePipeline,
    obj_cull: wgpu::ComputePipeline,
    obj_fin: wgpu::ComputePipeline,
    obj_shadow: wgpu::RenderPipeline,
    ball_shadow: wgpu::RenderPipeline,
    props: props::Props,
    terrain_shadow: wgpu::RenderPipeline,
    pipes: ScenePipes,
    grass: Grass,
    grass_args: wgpu::Buffer,
    grass_args_tpl: wgpu::Buffer,
    field_views: Vec<wgpu::TextureView>,
    field_bg: [wgpu::BindGroup; 2],
    field_parity: usize,
    field_res: u32,
    field_prev: [i32; 2],
    // objects
    obj_count: u32,
    obj_tiles: Vec<ObjTile>,
    obj_ranges: wgpu::Buffer,
    obj_sel: u32,
    obj_draws: wgpu::Buffer,
    obj_draws_tpl: wgpu::Buffer,
    obj_cull_bg: wgpu::BindGroup,
    obj_bg: wgpu::BindGroup,
    obj_params: wgpu::Buffer,
    buckets: Vec<(u32, u32)>,
    kind_counts: [u32; NUM_KINDS],
    train_buf: wgpu::Buffer,
    train_bg: wgpu::BindGroup,
    train_count: u32,
    balls_buf: wgpu::Buffer,
    balls_bg: wgpu::BindGroup,
    ball_count: u32,
    mesh_v: wgpu::Buffer,
    mesh_i: wgpu::Buffer,
    ranges: Vec<MeshRange>,
    // terrain
    grids: Vec<(wgpu::Buffer, wgpu::Buffer, u32)>,
    chunk_inst: Vec<wgpu::Buffer>,
    chunk_counts: Vec<u32>,
    shadow_chunks: wgpu::Buffer,
    shadow_chunk_count: u32,
    chunks: Vec<Chunk>,
    shell_v: wgpu::Buffer,
    shell_i: wgpu::Buffer,
    shell_count: u32,
    lines_buf: wgpu::Buffer,
    line_count: u32,
    targets: Targets,
    post: Post,
    pub egui: egui_wgpu::Renderer,
    frame: u32,
    readback: Readback,
    pub stats: Stats,
    shadow_dist: f32,
    skip: String,
}

struct Modules {
    field: wgpu::ShaderModule,
    grass_cull: wgpu::ShaderModule,
    grass: wgpu::ShaderModule,
    terrain: wgpu::ShaderModule,
    shell: wgpu::ShaderModule,
    objects_cull: wgpu::ShaderModule,
    objects: wgpu::ShaderModule,
    balls: wgpu::ShaderModule,
    aircars: wgpu::ShaderModule,
    lines: wgpu::ShaderModule,
}

struct Layouts {
    scene: wgpu::BindGroupLayout,
    field: wgpu::BindGroupLayout,
    cull: wgpu::BindGroupLayout,
    blades: wgpu::BindGroupLayout,
    objcull: wgpu::BindGroupLayout,
    objs: wgpu::BindGroupLayout,
    p_scene: wgpu::PipelineLayout,
    p_field: wgpu::PipelineLayout,
    p_cull: wgpu::PipelineLayout,
    p_blades: wgpu::PipelineLayout,
    p_objcull: wgpu::PipelineLayout,
    p_objs: wgpu::PipelineLayout,
}

// ------------------------------------------------------------ helpers -----
fn storage(binding: u32, read_only: bool, vis: wgpu::ShaderStages, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: dynamic, min_binding_size: None },
        count: None,
    }
}
fn texture(binding: u32, sample_type: wgpu::TextureSampleType, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}
fn sampler(binding: u32, ty: wgpu::SamplerBindingType, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty: wgpu::BindingType::Sampler(ty), count: None }
}
fn bge(binding: u32, resource: wgpu::BindingResource) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry { binding, resource }
}
fn buf(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
}

const ALL: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::FRAGMENT).union(wgpu::ShaderStages::COMPUTE);
const VS: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX;
const CS: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;

const VERTEX_ATTRS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32];
const POS_ONLY: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
const GRID_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
const CHUNK_ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Float32x4];
const SHELL_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32];
const LINE_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

fn mesh_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout { array_stride: 28, step_mode: wgpu::VertexStepMode::Vertex, attributes: &VERTEX_ATTRS }
}
fn terrain_layouts() -> [wgpu::VertexBufferLayout<'static>; 2] {
    [
        wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &GRID_ATTRS },
        wgpu::VertexBufferLayout { array_stride: 16, step_mode: wgpu::VertexStepMode::Instance, attributes: &CHUNK_ATTRS },
    ]
}

#[derive(Clone, Copy, PartialEq)]
enum Blend {
    Opaque,
    Additive,
}

#[allow(clippy::too_many_arguments)]
fn scene_pipe(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vs: &str,
    fs: &str,
    buffers: &[wgpu::VertexBufferLayout],
    topology: wgpu::PrimitiveTopology,
    cull: Option<wgpu::Face>,
    samples: u32,
    blend: Blend,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState { module, entry_point: vs, buffers, compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: fs,
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: match blend {
                    Blend::Opaque => None,
                    Blend::Additive => Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                },
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology, cull_mode: cull, front_face: wgpu::FrontFace::Ccw, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: blend == Blend::Opaque,
            depth_compare: wgpu::CompareFunction::Greater,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
        multiview: None,
        cache: None,
    })
}

fn shadow_pipe(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vs: &str,
    buffers: &[wgpu::VertexBufferLayout],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState { module, entry_point: vs, buffers, compilation_options: Default::default() },
        fragment: None,
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.5, clamp: 0.0 },
        }),
        multisample: Default::default(),
        multiview: None,
        cache: None,
    })
}

fn compute(device: &wgpu::Device, label: &str, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: entry,
        compilation_options: Default::default(),
        cache: None,
    })
}

fn dummy_cube(device: &wgpu::Device) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy env"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() })
}

fn depth_tex(device: &wgpu::Device, label: &str, size: [u32; 2], samples: u32, sampled: bool) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | if sampled { wgpu::TextureUsages::TEXTURE_BINDING } else { wgpu::TextureUsages::empty() },
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn color_tex(device: &wgpu::Device, label: &str, size: [u32; 2], samples: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn make_targets(device: &wgpu::Device, size: [u32; 2], samples: u32) -> Targets {
    let hdr = color_tex(device, "hdr", size, 1, HDR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
    let msaa = (samples > 1).then(|| color_tex(device, "hdr msaa", size, samples, HDR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT));
    Targets { size, samples, msaa, hdr, depth: depth_tex(device, "depth", size, samples, false) }
}

fn render_size(gpu: &Gpu, g: &Graphics) -> [u32; 2] {
    [((gpu.config.width as f32 * g.render_scale) as u32).max(1), ((gpu.config.height as f32 * g.render_scale) as u32).max(1)]
}

pub fn frustum_planes(vp: Mat4) -> [[f32; 4]; 6] {
    let r = [vp.row(0), vp.row(1), vp.row(2), vp.row(3)];
    let planes = [r[3] + r[0], r[3] - r[0], r[3] + r[1], r[3] - r[1], r[3] - r[2], r[2]];
    planes.map(|p: Vec4| {
        let l = p.truncate().length();
        if l < 1e-6 {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            (p / l).to_array()
        }
    })
}

fn sphere_visible(planes: &[[f32; 4]; 6], c: Vec3, r: f32) -> bool {
    planes.iter().all(|p| p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] >= -r)
}

// ------------------------------------------------------------ renderer ----
impl Renderer {
    pub fn new(gpu: &Gpu, world: &World, g: &Graphics) -> Renderer {
        let device = &gpu.device;
        let queue = &gpu.queue;
        let module = |name: &str, body: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(shaders::compose(body).into()) })
        };
        let modules = Modules {
            field: module("field", shaders::FIELD),
            grass_cull: module("grass_cull", shaders::GRASS_CULL),
            grass: module("grass", shaders::GRASS),
            terrain: module("terrain", shaders::TERRAIN),
            shell: module("shell", shaders::SHELL),
            objects_cull: module("objects_cull", shaders::OBJECTS_CULL),
            objects: module("objects", shaders::OBJECTS),
            balls: module("balls", shaders::BALLS),
            aircars: module("aircars", shaders::AIRCARS),
            lines: module("lines", shaders::LINES),
        };

        // ---- layouts
        use wgpu::TextureSampleType as T;
        let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ALL,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                texture(1, T::Float { filterable: false }, ALL),
                texture(2, T::Float { filterable: true }, ALL),
                sampler(3, wgpu::SamplerBindingType::Filtering, ALL),
                texture(4, T::Depth, ALL),
                sampler(5, wgpu::SamplerBindingType::Comparison, ALL),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ALL,
                    ty: wgpu::BindingType::Texture {
                        sample_type: T::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(7, true, ALL, false),
            ],
        });
        let field = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field"),
            entries: &[
                texture(0, T::Float { filterable: true }, CS),
                texture(1, T::Float { filterable: true }, CS),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: CS,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: CS,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let cull = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grass cull"),
            entries: &[
                texture(0, T::Float { filterable: true }, CS),
                texture(1, T::Float { filterable: true }, CS),
                sampler(2, wgpu::SamplerBindingType::Filtering, CS),
                storage(3, false, CS, false),
                storage(4, false, CS, false),
                storage(5, false, CS, false),
            ],
        });
        let blades = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("blades"), entries: &[storage(0, true, VS, false)] });
        let objcull = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("objcull"),
            entries: &[
                storage(0, true, CS, false),
                storage(1, false, CS, false),
                storage(2, false, CS, false),
                storage(4, true, CS, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: CS,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let objs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("objs"), entries: &[storage(0, true, VS, true)] });
        let pl = |l: &[&wgpu::BindGroupLayout]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: l, push_constant_ranges: &[] })
        };
        let layouts = Layouts {
            p_scene: pl(&[&scene]),
            p_field: pl(&[&scene, &field]),
            p_cull: pl(&[&scene, &cull]),
            p_blades: pl(&[&scene, &blades]),
            p_objcull: pl(&[&scene, &objcull]),
            p_objs: pl(&[&scene, &objs]),
            scene,
            field,
            cull,
            blades,
            objcull,
            objs,
        };

        // ---- world textures
        let height_view = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("height"),
                    size: wgpu::Extent3d { width: HW as u32, height: HH as u32, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&world.height),
            )
            .create_view(&Default::default());
        let zone_view = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("zone"),
                    size: wgpu::Extent3d { width: ZW as u32, height: ZH as u32, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&world.zone),
            )
            .create_view(&Default::default());
        let lin_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lin"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let field_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("field"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let globals = buf(device, "globals", std::mem::size_of::<Globals>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let dummy_shadow = depth_tex(device, "dummy shadow", [1, 1], 1, true);
        let env_view = dummy_cube(device);
        let nodes_buf = buf(device, "node xf", (N_NODES * std::mem::size_of::<NodeXf>()) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let shadow_res = g.shadow_res.max(1);
        let shadow = depth_tex(device, "shadow", [shadow_res, shadow_res], 1, true);

        let field_res = field_res_for(g);
        let (field_views, field_bg) = make_field(device, &layouts, field_res);

        let grass_args = buf(
            device,
            "grass args",
            32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let grass_args_tpl = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass args tpl"),
            contents: bytemuck::cast_slice(&[13u32, 0, 0, 0, 5, 0, 0, 0]),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // ---- meshes
        let mut all: Vec<Mesh> = vec![
            meshes::unit_box(),
            meshes::unit_box(),
            meshes::tower(),
            meshes::unit_box(),
            meshes::dome(18),
            meshes::dome(8),
            meshes::conifer(0),
            meshes::conifer(1),
            meshes::broadleaf(0),
            meshes::broadleaf(1),
            meshes::cylinder(16),
            meshes::cylinder(6),
            meshes::train_car(),
            meshes::train_car(),
            meshes::unit_box(),
            meshes::unit_box(),
            meshes::house(),
            meshes::unit_box(),
            meshes::aircar(),
            meshes::ball_sphere(),
        ];
        let mut verts: Vec<Vertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        let mut ranges = Vec::new();
        for m in all.iter_mut() {
            m.fix_winding();
            ranges.push(MeshRange { first_index: idx.len() as u32, count: m.indices.len() as u32, base_vertex: verts.len() as i32 });
            verts.extend_from_slice(&m.verts);
            idx.extend_from_slice(&m.indices);
        }
        let mesh_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh v"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mesh_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh i"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });

        // ---- objects: buckets
        let kind_counts = world.kind_counts();
        let align = (device.limits().min_storage_buffer_offset_alignment / 32).max(1);
        let nk = NUM_KINDS as u32;
        let mut buckets = vec![(0u32, 0u32); (nk * 3) as usize];
        let mut cursor = 0u32;
        for b in 0..nk * 3 {
            let kind = if b < nk * 2 { b / 2 } else { b - nk * 2 };
            // Trees: only the near field is ever geometry (the far forest is
            // painted as canopy by the terrain shader), so cap their buckets.
            let lim = if kind == world::K_CONIFER || kind == world::K_BROADLEAF { 262_144 } else { u32::MAX };
            let cap = if kind == world::K_TRAIN { 0 } else { kind_counts[kind as usize].min(lim) };
            buckets[b as usize] = (cursor, cap);
            cursor += cap.div_ceil(align) * align;
        }
        let max_cap = buckets.iter().map(|b| b.1).max().unwrap_or(1).max(1);
        let total = cursor + max_cap;
        let mut sorted: Vec<ObjInstance> = world.objects.iter().filter(|o| o.kind() != world::K_TRAIN).copied().collect();
        // Spatial tiles: the CPU rejects whole tiles, the GPU culls instances
        // of the surviving tiles only (~1.4 M objects at canonical scale).
        let tile_key = |o: &ObjInstance| -> u64 {
            let sf = colony::world_to_surf(Vec3::from(o.pos));
            let tree = matches!(o.kind(), world::K_CONIFER | world::K_BROADLEAF);
            let ts = if tree { 256.0 } else { 512.0 };
            ((tree as u64) << 40) | ((((sf.x / ts).floor() as i64 + 100_000) as u64) << 20) | (((sf.y / ts).floor() as i64 + 1000) as u64)
        };
        sorted.sort_by_cached_key(|o| tile_key(o));
        let obj_count = sorted.len() as u32;
        let mut obj_tiles: Vec<ObjTile> = Vec::new();
        {
            let mut i = 0usize;
            while i < sorted.len() {
                let key = tile_key(&sorted[i]);
                let mut j = i;
                let mut c = Vec3::ZERO;
                while j < sorted.len() && tile_key(&sorted[j]) == key {
                    c += Vec3::from(sorted[j].pos);
                    j += 1;
                }
                let c = c / (j - i) as f32;
                let r = sorted[i..j].iter().map(|o| c.distance(Vec3::from(o.pos)) + 0.5 * Vec3::from(o.size).length() + o.size[1]).fold(0.0, f32::max);
                obj_tiles.push(ObjTile { start: i as u32, count: (j - i) as u32, centre: c, radius: r, tree: (key >> 40) != 0 });
                i = j;
            }
        }
        log::info!("object tiles: {}", obj_tiles.len());
        let obj_ranges = buf(device, "obj ranges", (MAX_RANGES * 8) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("objects"),
            contents: bytemuck::cast_slice(&sorted),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = buf(device, "objects visible", total as u64 * 32, wgpu::BufferUsages::STORAGE);
        let mut tpl = vec![DrawIndexed::default(); (nk * 3) as usize];
        for b in 0..nk * 3 {
            let (kind, lod) = if b < nk * 2 { (b / 2, b % 2) } else { (b - nk * 2, 1) };
            let r = ranges[(kind * 2 + lod) as usize];
            tpl[b as usize] =
                DrawIndexed { index_count: r.count, instance_count: 0, first_index: r.first_index, base_vertex: r.base_vertex, first_instance: 0 };
        }
        let obj_draws_tpl = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("draws tpl"),
            contents: bytemuck::cast_slice(&tpl),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let obj_draws = buf(
            device,
            "draws",
            (tpl.len() * 20) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let obj_params = buf(device, "cull params", std::mem::size_of::<CullParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let obj_cull_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("objcull"),
            layout: &layouts.objcull,
            entries: &[
                bge(0, inst_buf.as_entire_binding()),
                bge(1, out_buf.as_entire_binding()),
                bge(2, obj_draws.as_entire_binding()),
                bge(3, obj_params.as_entire_binding()),
                bge(4, obj_ranges.as_entire_binding()),
            ],
        });
        let obj_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("objs"),
            layout: &layouts.objs,
            entries: &[bge(
                0,
                wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &out_buf, offset: 0, size: wgpu::BufferSize::new(max_cap as u64 * 32) }),
            )],
        });
        let train_buf = buf(device, "trains", (MAX_TRAIN * 32) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let train_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trains"),
            layout: &layouts.objs,
            entries: &[bge(
                0,
                wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &train_buf, offset: 0, size: wgpu::BufferSize::new((MAX_TRAIN * 32) as u64) }),
            )],
        });
        let balls_buf = buf(device, "balls", (MAX_RENDER_BALLS * 48) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let balls_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("balls"),
            layout: &layouts.blades,
            entries: &[bge(0, balls_buf.as_entire_binding())],
        });

        // ---- terrain
        let grids: Vec<(wgpu::Buffer, wgpu::Buffer, u32)> = [32u32, 16, 8, 4]
            .iter()
            .map(|&n| {
                let (v, i) = meshes::terrain_grid(n);
                (
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("grid v"),
                        contents: bytemuck::cast_slice(&v),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("grid i"),
                        contents: bytemuck::cast_slice(&i),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    i.len() as u32,
                )
            })
            .collect();
        let chunks = build_chunks(world);
        let chunk_inst = (0..4).map(|_| buf(device, "chunks", (chunks.len() * 16) as u64, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST)).collect();
        let shadow_chunks = buf(device, "shadow chunks", (chunks.len() * 16) as u64, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);

        // ---- shell
        let (sv, si) = shell_mesh();
        let shell_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shell v"),
            contents: bytemuck::cast_slice(&sv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let shell_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shell i"),
            contents: bytemuck::cast_slice(&si),
            usage: wgpu::BufferUsages::INDEX,
        });
        let lines_buf = buf(device, "lines", (MAX_LINE_VERTS * 28) as u64, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);

        // ---- pipelines
        let l = &layouts;
        let m = &modules;
        let field_pipe = compute(device, "field", &l.p_field, &m.field, "cs_main");
        let cull_pipe = compute(device, "grass cull", &l.p_cull, &m.grass_cull, "cs_main");
        let cull_fin = compute(device, "grass fin", &l.p_cull, &m.grass_cull, "cs_finalize");
        let obj_cull = compute(device, "obj cull", &l.p_objcull, &m.objects_cull, "cs_main");
        let obj_fin = compute(device, "obj fin", &l.p_objcull, &m.objects_cull, "cs_finalize");
        let obj_shadow = shadow_pipe(device, "obj shadow", &l.p_objs, &m.objects, "vs_shadow", &[mesh_layout()]);
        let ball_shadow = shadow_pipe(
            device,
            "ball shadow",
            &l.p_blades,
            &m.balls,
            "vs_shadow",
            &[wgpu::VertexBufferLayout { array_stride: 28, step_mode: wgpu::VertexStepMode::Vertex, attributes: &POS_ONLY }],
        );
        let terrain_shadow = shadow_pipe(device, "terrain shadow", &l.p_scene, &m.terrain, "vs_shadow", &terrain_layouts());

        let samples = g.aa.samples();
        let pipes = build_scene_pipes(device, &layouts, &modules, samples);
        let props = props::Props::new(device, &layouts, samples);
        let size = render_size(gpu, g);
        let targets = make_targets(device, size, samples);
        let post = Post::new(device, gpu.config.format, &targets.hdr, size);
        let egui = egui_wgpu::Renderer::new(device, gpu.config.format.remove_srgb_suffix(), None, 1, false);

        let scene_bg_for = |shadow: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene"),
                layout: &layouts.scene,
                entries: &[
                    bge(0, globals.as_entire_binding()),
                    bge(1, wgpu::BindingResource::TextureView(&height_view)),
                    bge(2, wgpu::BindingResource::TextureView(&zone_view)),
                    bge(3, wgpu::BindingResource::Sampler(&lin_sampler)),
                    bge(4, wgpu::BindingResource::TextureView(shadow)),
                    bge(5, wgpu::BindingResource::Sampler(&shadow_sampler)),
                    bge(6, wgpu::BindingResource::TextureView(&env_view)),
                    bge(7, nodes_buf.as_entire_binding()),
                ],
            })
        };
        let scene_bg = scene_bg_for(&shadow);
        let scene_bg_noshadow = scene_bg_for(&dummy_shadow);
        let grass = make_grass(device, &layouts, g, &field_views, &field_sampler, &grass_args);
        let readback = Readback {
            buf: buf(device, "readback", 256 + (NUM_KINDS * 3 * 20) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            pending: Arc::new(AtomicBool::new(false)),
            in_flight: false,
        };
        let nchunks = chunks.len();
        Renderer {
            screenshot: None,
            modules,
            layouts,
            globals,
            height_view,
            zone_view,
            lin_sampler,
            shadow_sampler,
            field_sampler,
            shadow,
            shadow_res: g.shadow_res,
            dummy_shadow,
            scene_bg,
            scene_bg_noshadow,
            field_pipe,
            cull_pipe,
            cull_fin,
            obj_cull,
            obj_fin,
            obj_shadow,
            ball_shadow,
            props,
            terrain_shadow,
            pipes,
            grass,
            grass_args,
            grass_args_tpl,
            field_views,
            field_bg,
            field_parity: 0,
            field_res,
            field_prev: [i32::MIN / 2, i32::MIN / 2],
            obj_count,
            obj_tiles,
            obj_ranges,
            obj_sel: 0,
            obj_draws,
            obj_draws_tpl,
            obj_cull_bg,
            obj_bg,
            obj_params,
            buckets,
            kind_counts,
            train_buf,
            train_bg,
            train_count: 0,
            balls_buf,
            balls_bg,
            ball_count: 0,
            mesh_v,
            mesh_i,
            ranges,
            grids,
            chunk_inst,
            chunk_counts: vec![0; 4],
            shadow_chunks,
            shadow_chunk_count: 0,
            chunks,
            shell_v,
            shell_i,
            shell_count: si.len() as u32,
            lines_buf,
            line_count: 0,
            targets,
            post,
            egui,
            frame: 0,
            readback,
            stats: Stats { chunks: nchunks as u32, ..Default::default() },
            shadow_dist: g.shadow_dist,
            skip: std::env::var("ONEILL_SKIP").unwrap_or_default(),
            env_view,
            nodes_buf,
        }
    }

    /// Black-hole mode: binds the environment cube (view dimension Cube).
    pub fn set_env(&mut self, device: &wgpu::Device, view: wgpu::TextureView) {
        self.env_view = view;
        self.rebuild_scene_bgs(device);
    }

    /// Black-hole mode: per-node rigid transforms of the broken hull.
    pub fn upload_nodes(&self, queue: &wgpu::Queue, xf: &[NodeXf]) {
        queue.write_buffer(&self.nodes_buf, 0, bytemuck::cast_slice(&xf[..xf.len().min(N_NODES)]));
    }

    fn rebuild_scene_bgs(&mut self, device: &wgpu::Device) {
        let make = |shadow: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene"),
                layout: &self.layouts.scene,
                entries: &[
                    bge(0, self.globals.as_entire_binding()),
                    bge(1, wgpu::BindingResource::TextureView(&self.height_view)),
                    bge(2, wgpu::BindingResource::TextureView(&self.zone_view)),
                    bge(3, wgpu::BindingResource::Sampler(&self.lin_sampler)),
                    bge(4, wgpu::BindingResource::TextureView(shadow)),
                    bge(5, wgpu::BindingResource::Sampler(&self.shadow_sampler)),
                    bge(6, wgpu::BindingResource::TextureView(&self.env_view)),
                    bge(7, self.nodes_buf.as_entire_binding()),
                ],
            })
        };
        self.scene_bg = make(&self.shadow);
        self.scene_bg_noshadow = make(&self.dummy_shadow);
    }

    /// Recreate whatever depends on changed settings.
    pub fn apply_settings(&mut self, gpu: &Gpu, g: &Graphics) {
        let device = &gpu.device;
        let samples = g.aa.samples();
        let size = render_size(gpu, g);
        if samples != self.targets.samples {
            self.pipes = build_scene_pipes(device, &self.layouts, &self.modules, samples);
            self.props.rebuild(device, samples);
        }
        if samples != self.targets.samples || size != self.targets.size {
            self.targets = make_targets(device, size, samples);
            self.post.resize(device, &self.targets.hdr, size);
        }
        if g.shadow_res != self.shadow_res {
            self.shadow_res = g.shadow_res;
            let r = g.shadow_res.max(1);
            self.shadow = depth_tex(device, "shadow", [r, r], 1, true);
            self.rebuild_scene_bgs(device);
        }
        self.shadow_dist = g.shadow_dist;
        let fr = field_res_for(g);
        let field_changed = fr != self.field_res;
        if field_changed {
            self.field_res = fr;
            let (v, b) = make_field(device, &self.layouts, fr);
            self.field_views = v;
            self.field_bg = b;
            self.field_prev = [i32::MIN / 2, i32::MIN / 2];
        }
        if field_changed || g.grass_capacity() != self.grass.cap || grid_n(g) != self.grass.grid_n {
            self.grass = make_grass(device, &self.layouts, g, &self.field_views, &self.field_sampler, &self.grass_args);
        }
    }

    pub fn resize(&mut self, gpu: &Gpu, g: &Graphics) {
        let size = render_size(gpu, g);
        if size != self.targets.size {
            self.targets = make_targets(&gpu.device, size, self.targets.samples);
            self.post.resize(&gpu.device, &self.targets.hdr, size);
        }
    }

    fn poll_readback(&mut self, device: &wgpu::Device) {
        if self.readback.in_flight && self.readback.pending.load(Ordering::Acquire) {
            {
                let data = self.readback.buf.slice(..).get_mapped_range();
                let a: &[u32] = bytemuck::cast_slice(&data[..32]);
                self.stats.blades_near = a[1];
                self.stats.blades_far = a[5];
                let d: &[u32] = bytemuck::cast_slice(&data[256..256 + NUM_KINDS * 2 * 20]);
                self.stats.objects = (0..NUM_KINDS * 2).map(|b| d[b * 5 + 1]).sum();
            }
            self.readback.buf.unmap();
            self.readback.in_flight = false;
            self.readback.pending.store(false, Ordering::Release);
        }
        device.poll(wgpu::Maintain::Poll);
    }

    /// Fill the renderer-owned parts of the globals (grids, culling, shadows).
    #[allow(clippy::too_many_arguments)]
    fn prepare(&mut self, gpu: &Gpu, gl: &mut Globals, s: &Graphics, cam: Vec3, view_fwd: Vec3, vp: Mat4, fd: &FrameData) {
        let queue = &gpu.queue;
        // after a break-up, panels of other fragments are moved in the vertex
        // shaders: frustum culling of the rest geometry would be wrong
        let planes = if fd.no_cull { [[0.0, 0.0, 0.0, 1.0e9]; 6] } else { frustum_planes(vp) };
        gl.frustum = planes;
        let sf = colony::world_to_surf(cam);
        let (strip, _) = colony::strip_of(sf.x / R);
        gl.cam_surf = [sf.x, sf.y, sf.z, strip as f32];
        // Grass grid anchored to the world.
        let sp = s.grass_spacing;
        let n = self.grass.grid_n as i32;
        gl.grass = [sp, s.grass_radius, s.grass_lod, s.grass_lod * 0.6];
        gl.grid = [(sf.x / sp).floor() as i32 - n / 2, (sf.y / sp).floor() as i32 - n / 2, n, 1337];
        let base = [(sf.x / FIELD_CELL).floor() as i32 - self.field_res as i32 / 2, (sf.y / FIELD_CELL).floor() as i32 - self.field_res as i32 / 2];
        gl.field_base = [base[0], base[1], self.field_prev[0], self.field_prev[1]];
        self.field_prev = base;
        gl.field_params[0] = FIELD_CELL;
        gl.field_params[1] = self.field_res as f32;
        // Shadow map: ortho box along the light of the camera's valley.
        let shadows = s.shadow_res > 0 && gl.light_dir[strip][3] > 0.01;
        if shadows {
            let l = Vec3::from_slice(&gl.light_dir[strip][..3]).normalize();
            let up_loc = colony::up_of(sf.x / R);
            let flat = (view_fwd - up_loc * view_fwd.dot(up_loc)).normalize_or_zero();
            let ground = colony::surf_to_world(sf.x, sf.y, sf.z.min(30.0)) - up_loc * 0.0;
            let centre = ground + flat * s.shadow_dist * 0.45 - up_loc * (sf.z.min(30.0) - 2.0);
            let r = s.shadow_dist;
            let hint = if l.z.abs() < 0.95 { Vec3::Z } else { Vec3::X };
            let dist = 1200.0;
            let view = Mat4::look_at_rh(Vec3::ZERO, -l, hint);
            // Snap the centre to shadow texels to avoid shimmering.
            let texel = 2.0 * r / s.shadow_res as f32;
            let c_ls = view.transform_point3(centre);
            let snapped = Vec3::new((c_ls.x / texel).round() * texel, (c_ls.y / texel).round() * texel, c_ls.z);
            let centre = view.inverse().transform_point3(snapped);
            let view = Mat4::look_at_rh(centre + l * dist, centre, hint);
            let proj = Mat4::orthographic_rh(-r, r, -r, r, 1.0, dist * 2.0);
            gl.shadow_mat = (proj * view).to_cols_array_2d();
            gl.params[0] = 1.0;
            gl.params[1] = 1.0 / s.shadow_res as f32;
        } else {
            gl.params[0] = 0.0;
        }
        gl.params[2] = s.aircars as f32;
        gl.params[3] = self.frame as f32;
        gl.params2[0] = self.targets.size[0] as f32;
        gl.params2[1] = self.targets.size[1] as f32;
        gl.params2[3] = s.tree_dist;
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(gl));

        // Object culling parameters.
        let nk = NUM_KINDS as u32;
        let mut cp = CullParams { counts: [self.obj_count, nk, shadows as u32, 0], kinds: [[0.0; 4]; 16], buckets: [[0; 4]; 48] };
        let sd = s.shadow_dist * 1.4;
        let td = s.tree_dist;
        let info: [[f32; 3]; NUM_KINDS] = [
            [900.0, 48000.0, sd + 60.0],
            [1200.0, 48000.0, sd + 150.0],
            [450.0, 9000.0, sd + 30.0],
            [(td * 0.18).max(40.0), td, sd],
            [(td * 0.18).max(40.0), td, sd],
            [300.0, 5000.0, sd],
            [0.0, 0.0, 0.0],
            [700.0, 14000.0, sd],
            [300.0, 7000.0, sd],
        ];
        for (dst, inf) in cp.kinds.iter_mut().zip(info.iter()).take(NUM_KINDS) {
            *dst = [inf[0], inf[1], inf[2], 0.0];
        }
        for (i, b) in self.buckets.iter().enumerate() {
            cp.buckets[i] = [b.0, b.1, 0, 0];
        }
        {
            let mut ranges: Vec<[u32; 2]> = Vec::with_capacity(4096);
            let mut total = 0u32;
            let tree_reach = td.max(if shadows { sd } else { 0.0 }) + 60.0;
            for t in &self.obj_tiles {
                let d = t.centre.distance(cam) - t.radius;
                if d > if t.tree { tree_reach } else { 48000.0 } {
                    continue;
                }
                let shadow_ok = shadows && d < sd + 200.0;
                if !shadow_ok && !sphere_visible(&planes, t.centre, t.radius) {
                    continue;
                }
                if ranges.len() >= MAX_RANGES {
                    break;
                }
                ranges.push([t.start, total]);
                total += t.count;
            }
            cp.counts[0] = total;
            cp.counts[3] = ranges.len() as u32;
            self.obj_sel = total;
            if !ranges.is_empty() {
                queue.write_buffer(&self.obj_ranges, 0, bytemuck::cast_slice(&ranges));
            }
        }
        queue.write_buffer(&self.obj_params, 0, bytemuck::bytes_of(&cp));

        // Terrain chunk selection (CPU, ~1000 chunks).
        let mut lists: [Vec<[f32; 4]>; 4] = Default::default();
        let mut shadow_list = Vec::new();
        let det = s.terrain_detail.max(0.25);
        let lod_d = [220.0 * det, 700.0 * det, 2200.0 * det];
        let sc = Vec3::from_slice(&gl.cam_pos[..3]);
        for c in &self.chunks {
            let d = (c.centre.distance(sc) - c.radius).max(0.0);
            if shadows && c.strip == strip && d < s.shadow_dist * 1.6 {
                shadow_list.push(c.inst);
            }
            if !sphere_visible(&planes, c.centre, c.radius) {
                continue;
            }
            let lod = lod_d.iter().position(|&x| d < x).unwrap_or(3);
            lists[lod].push(c.inst);
        }
        for (i, l) in lists.iter().enumerate() {
            self.chunk_counts[i] = l.len() as u32;
            if !l.is_empty() {
                queue.write_buffer(&self.chunk_inst[i], 0, bytemuck::cast_slice(l));
            }
        }
        self.shadow_chunk_count = shadow_list.len() as u32;
        if !shadow_list.is_empty() {
            queue.write_buffer(&self.shadow_chunks, 0, bytemuck::cast_slice(&shadow_list));
        }
        // Dynamic geometry.
        self.ball_count = fd.balls.len().min(MAX_RENDER_BALLS) as u32;
        if self.ball_count > 0 {
            queue.write_buffer(&self.balls_buf, 0, bytemuck::cast_slice(&fd.balls[..self.ball_count as usize]));
        }
        self.train_count = fd.trains.len().min(MAX_TRAIN) as u32;
        if self.train_count > 0 {
            queue.write_buffer(&self.train_buf, 0, bytemuck::cast_slice(&fd.trains[..self.train_count as usize]));
        }
        self.props.prepare(queue, &planes, cam, fd);
        self.line_count = fd.lines.len().min(MAX_LINE_VERTS) as u32 / 3 * 3;
        if self.line_count > 0 {
            queue.write_buffer(&self.lines_buf, 0, bytemuck::cast_slice(&fd.lines[..self.line_count as usize]));
        }
    }

    pub fn render(&mut self, gpu: &Gpu, gl: &mut Globals, s: &Graphics, fd: &FrameData, ui: UiFrame) -> Result<()> {
        let device = &gpu.device;
        self.poll_readback(device);
        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                gpu.surface.configure(device, &gpu.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let out_view = frame.texture.create_view(&Default::default());
        let ui_view = frame.texture.create_view(&wgpu::TextureViewDescriptor { format: Some(gpu.config.format.remove_srgb_suffix()), ..Default::default() });
        let vp = Mat4::from_cols_array_2d(&gl.view_proj);
        self.prepare(gpu, gl, s, fd.cam, fd.fwd, vp, fd);
        let shadows = gl.params[0] > 0.5;

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        enc.copy_buffer_to_buffer(&self.grass_args_tpl, 0, &self.grass_args, 0, 32);
        enc.copy_buffer_to_buffer(&self.obj_draws_tpl, 0, &self.obj_draws, 0, (NUM_KINDS * 3 * 20) as u64);
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compute"), timestamp_writes: None });
            cp.set_bind_group(0, &self.scene_bg_noshadow, &[]);
            // grass dynamics field (ping-pong)
            let p = self.field_parity;
            cp.set_pipeline(&self.field_pipe);
            cp.set_bind_group(1, &self.field_bg[p], &[]);
            if s.grass_radius > 1.0 && !self.skip.contains('f') {
                cp.dispatch_workgroups(self.field_res / 8, self.field_res / 8, 1);
            }
            // grass generation + culling reads the freshly written set
            cp.set_pipeline(&self.cull_pipe);
            cp.set_bind_group(1, &self.grass.cull_bg[1 - p], &[]);
            let groups = self.grass.grid_n / 16;
            if s.grass_radius > 1.0 && !self.skip.contains('c') {
                cp.dispatch_workgroups(groups, groups, 1);
            }
            cp.set_pipeline(&self.cull_fin);
            cp.dispatch_workgroups(1, 1, 1);
            // objects
            cp.set_pipeline(&self.obj_cull);
            cp.set_bind_group(1, &self.obj_cull_bg, &[]);
            let groups = self.obj_sel.div_ceil(64).max(1);
            cp.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
            cp.set_pipeline(&self.obj_fin);
            cp.dispatch_workgroups(1, 1, 1);
        }
        self.field_parity = 1 - self.field_parity;

        let nk = NUM_KINDS as u32;
        if shadows {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_bind_group(0, &self.scene_bg_noshadow, &[]);
            rp.set_pipeline(&self.obj_shadow);
            rp.set_vertex_buffer(0, self.mesh_v.slice(..));
            rp.set_index_buffer(self.mesh_i.slice(..), wgpu::IndexFormat::Uint32);
            for k in 0..nk {
                let b = (nk * 2 + k) as usize;
                if self.buckets[b].1 == 0 {
                    continue;
                }
                rp.set_bind_group(1, &self.obj_bg, &[self.buckets[b].0 * 32]);
                rp.draw_indexed_indirect(&self.obj_draws, b as u64 * 20);
            }
            if self.train_count > 0 {
                let r = self.ranges[(world::K_TRAIN * 2) as usize];
                rp.set_bind_group(1, &self.train_bg, &[0]);
                rp.draw_indexed(r.first_index..r.first_index + r.count, r.base_vertex, 0..self.train_count);
            }
            if self.ball_count > 0 {
                let r = self.ranges[19];
                rp.set_pipeline(&self.ball_shadow);
                rp.set_bind_group(1, &self.balls_bg, &[]);
                rp.draw_indexed(r.first_index..r.first_index + r.count, r.base_vertex, 0..self.ball_count);
            }
            self.props.draw_shadow(&mut rp);
            if self.shadow_chunk_count > 0 {
                let g = &self.grids[1];
                rp.set_pipeline(&self.terrain_shadow);
                rp.set_vertex_buffer(0, g.0.slice(..));
                rp.set_vertex_buffer(1, self.shadow_chunks.slice(..));
                rp.set_index_buffer(g.1.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..g.2, 0, 0..self.shadow_chunk_count);
            }
        }

        {
            let (view, resolve) = match &self.targets.msaa {
                Some(m) => (m, Some(&self.targets.hdr)),
                None => (&self.targets.hdr, None),
            };
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: resolve,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_bind_group(0, &self.scene_bg, &[]);
            rp.set_vertex_buffer(0, self.mesh_v.slice(..));
            rp.set_index_buffer(self.mesh_i.slice(..), wgpu::IndexFormat::Uint32);
            if self.ball_count > 0 {
                let r = self.ranges[19];
                rp.set_pipeline(&self.pipes.balls);
                rp.set_bind_group(1, &self.balls_bg, &[]);
                rp.draw_indexed(r.first_index..r.first_index + r.count, r.base_vertex, 0..self.ball_count);
            }
            self.props.draw_opaque(&mut rp);
            rp.set_vertex_buffer(0, self.mesh_v.slice(..));
            rp.set_index_buffer(self.mesh_i.slice(..), wgpu::IndexFormat::Uint32);
            // grass
            if !self.skip.contains('g') {
                rp.set_pipeline(&self.pipes.grass_near);
                rp.set_bind_group(1, &self.grass.near_bg, &[]);
                rp.draw_indirect(&self.grass_args, 0);
                rp.set_pipeline(&self.pipes.grass_far);
                rp.set_bind_group(1, &self.grass.far_bg, &[]);
                rp.draw_indirect(&self.grass_args, 16);
            }
            // objects
            rp.set_pipeline(&self.pipes.objects);
            for b in 0..(nk * 2) as usize {
                if self.buckets[b].1 == 0 || self.skip.contains('o') {
                    continue;
                }
                rp.set_bind_group(1, &self.obj_bg, &[self.buckets[b].0 * 32]);
                rp.draw_indexed_indirect(&self.obj_draws, b as u64 * 20);
            }
            if self.train_count > 0 {
                let r = self.ranges[(world::K_TRAIN * 2) as usize];
                rp.set_bind_group(1, &self.train_bg, &[0]);
                rp.draw_indexed(r.first_index..r.first_index + r.count, r.base_vertex, 0..self.train_count);
            }
            // aircar bodies
            if s.aircars > 0 && !self.skip.contains('a') {
                let r = self.ranges[18];
                rp.set_pipeline(&self.pipes.car_body);
                rp.draw_indexed(r.first_index..r.first_index + r.count, r.base_vertex, 0..s.aircars);
            }
            // terrain
            rp.set_pipeline(&self.pipes.terrain);
            for (i, g) in self.grids.iter().enumerate() {
                if self.chunk_counts[i] == 0 || self.skip.contains('t') {
                    continue;
                }
                rp.set_vertex_buffer(0, g.0.slice(..));
                rp.set_vertex_buffer(1, self.chunk_inst[i].slice(..));
                rp.set_index_buffer(g.1.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..g.2, 0, 0..self.chunk_counts[i]);
            }
            // hull
            rp.set_pipeline(&self.pipes.shell);
            rp.set_vertex_buffer(0, self.shell_v.slice(..));
            rp.set_index_buffer(self.shell_i.slice(..), wgpu::IndexFormat::Uint32);
            if !self.skip.contains('s') {
                rp.draw_indexed(0..self.shell_count, 0, 0..1);
            }
            // additive
            if s.aircars > 0 {
                rp.set_pipeline(&self.pipes.car_glow);
                rp.draw(0..4, 0..s.aircars * 2);
            }
            if self.line_count > 0 {
                rp.set_pipeline(&self.pipes.lines);
                rp.set_vertex_buffer(0, self.lines_buf.slice(..));
                rp.draw(0..self.line_count, 0..1);
            }
            self.props.draw_particles(&mut rp);
        }

        let ps = PostSettings {
            bloom: s.bloom,
            bloom_strength: s.bloom_strength,
            exposure: fd.exposure,
            tonemap: match s.tonemap {
                crate::settings::Tonemap::AgX => 0,
                crate::settings::Tonemap::Aces => 1,
            },
            sharpen: s.sharpen,
            fxaa: s.aa.fxaa(),
            vignette: s.vignette,
            saturation: s.saturation,
            time: gl.cam_pos[3],
        };
        self.post.run(&gpu.queue, &mut enc, &out_view, &ps);

        // egui overlay
        for (id, delta) in &ui.textures.set {
            self.egui.update_texture(device, &gpu.queue, *id, delta);
        }
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [gpu.config.width, gpu.config.height], pixels_per_point: ui.ppp };
        let extra = self.egui.update_buffers(device, &gpu.queue, &mut enc, ui.jobs, &sd);
        {
            let rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &ui_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut rp = rp.forget_lifetime();
            self.egui.render(&mut rp, ui.jobs, &sd);
        }
        let want_readback = !self.readback.in_flight && self.frame.is_multiple_of(15);
        if want_readback {
            enc.copy_buffer_to_buffer(&self.grass_args, 0, &self.readback.buf, 0, 32);
            enc.copy_buffer_to_buffer(&self.obj_draws, 0, &self.readback.buf, 256, (NUM_KINDS * 3 * 20) as u64);
        }
        let shot = match self.screenshot.take() {
            Some(path) if gpu.config.usage.contains(wgpu::TextureUsages::COPY_SRC) => {
                let (w, h) = (gpu.config.width, gpu.config.height);
                let row = (w * 4).div_ceil(256) * 256;
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("screenshot"),
                    size: (row * h) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                enc.copy_texture_to_buffer(
                    frame.texture.as_image_copy(),
                    wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                Some((path, buf, w, h, row))
            }
            Some(path) => {
                log::warn!("screenshot {}: surface does not support COPY_SRC", path.display());
                None
            }
            None => None,
        };
        let mut cmds = extra;
        cmds.push(enc.finish());
        gpu.queue.submit(cmds);
        if let Some((path, buf, w, h, row)) = shot {
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = buf.slice(..).get_mapped_range();
            let bgra = matches!(gpu.config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h {
                for px in data[(y * row) as usize..(y * row + w * 4) as usize].as_chunks::<4>().0 {
                    if bgra {
                        rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
                    } else {
                        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
                    }
                }
            }
            drop(data);
            match oneill_colony::bh::render::write_png(&path, w, h, &rgba) {
                Ok(()) => log::info!("screenshot saved: {}", path.display()),
                Err(e) => log::error!("screenshot: {e:#}"),
            }
        }
        if want_readback {
            let flag = self.readback.pending.clone();
            self.readback.in_flight = true;
            self.readback.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        }
        for id in &ui.textures.free {
            self.egui.free_texture(id);
        }
        frame.present();
        self.frame = self.frame.wrapping_add(1);
        Ok(())
    }

    pub fn kind_counts(&self) -> [u32; NUM_KINDS] {
        self.kind_counts
    }
}

pub struct FrameData<'a> {
    pub props: &'a [(u16, props::PropInst)],
    pub particles: &'a [props::ParticleGpu],
    pub cam: Vec3,
    pub fwd: Vec3,
    pub exposure: f32,
    pub balls: &'a [BallGpu],
    pub trains: &'a [ObjInstance],
    pub lines: &'a [LineVertex],
    pub no_cull: bool,
}

pub struct UiFrame<'a> {
    pub jobs: &'a [egui::ClippedPrimitive],
    pub textures: &'a egui::TexturesDelta,
    pub ppp: f32,
}

fn field_res_for(g: &Graphics) -> u32 {
    ((2.0 * g.grass_radius / FIELD_CELL / 64.0).ceil() as u32 * 64).clamp(128, 512)
}

fn make_field(device: &wgpu::Device, layouts: &Layouts, res: u32) -> (Vec<wgpu::TextureView>, [wgpu::BindGroup; 2]) {
    let views: Vec<wgpu::TextureView> = (0..4)
        .map(|i| {
            color_tex(
                device,
                &format!("field{i}"),
                [res, res],
                1,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            )
        })
        .collect();
    // set p: state = views[p], trample = views[2 + p]
    let bgs = [0usize, 1].map(|p| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field"),
            layout: &layouts.field,
            entries: &[
                bge(0, wgpu::BindingResource::TextureView(&views[p])),
                bge(1, wgpu::BindingResource::TextureView(&views[2 + p])),
                bge(2, wgpu::BindingResource::TextureView(&views[1 - p])),
                bge(3, wgpu::BindingResource::TextureView(&views[3 - p])),
            ],
        })
    });
    (views, bgs)
}

fn grid_n(g: &Graphics) -> u32 {
    let n = (2.0 * g.grass_radius / g.grass_spacing).ceil() as u32;
    n.div_ceil(16).max(1) * 16
}

fn make_grass(device: &wgpu::Device, l: &Layouts, g: &Graphics, field: &[wgpu::TextureView], samp: &wgpu::Sampler, args: &wgpu::Buffer) -> Grass {
    let cap = g.grass_capacity();
    let near = buf(device, "near blades", cap.0 as u64 * 48, wgpu::BufferUsages::STORAGE);
    let far = buf(device, "far blades", cap.1 as u64 * 48, wgpu::BufferUsages::STORAGE);
    let near_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("near"), layout: &l.blades, entries: &[bge(0, near.as_entire_binding())] });
    let far_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("far"), layout: &l.blades, entries: &[bge(0, far.as_entire_binding())] });
    let cull_bg = [0usize, 1].map(|p| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grass cull"),
            layout: &l.cull,
            entries: &[
                bge(0, wgpu::BindingResource::TextureView(&field[p])),
                bge(1, wgpu::BindingResource::TextureView(&field[2 + p])),
                bge(2, wgpu::BindingResource::Sampler(samp)),
                bge(3, near.as_entire_binding()),
                bge(4, far.as_entire_binding()),
                bge(5, args.as_entire_binding()),
            ],
        })
    });
    Grass { near, far, cap, grid_n: grid_n(g), near_bg, far_bg, cull_bg }
}

fn build_scene_pipes(device: &wgpu::Device, l: &Layouts, m: &Modules, samples: u32) -> ScenePipes {
    use wgpu::PrimitiveTopology as P;
    let shell_layout = [wgpu::VertexBufferLayout { array_stride: 16, step_mode: wgpu::VertexStepMode::Vertex, attributes: &SHELL_ATTRS }];
    let line_layout = [wgpu::VertexBufferLayout { array_stride: 28, step_mode: wgpu::VertexStepMode::Vertex, attributes: &LINE_ATTRS }];
    let pos_only = [wgpu::VertexBufferLayout { array_stride: 28, step_mode: wgpu::VertexStepMode::Vertex, attributes: &POS_ONLY }];
    ScenePipes {
        grass_near: scene_pipe(device, "grass near", &l.p_blades, &m.grass, "vs_near", "fs_main", &[], P::TriangleStrip, None, samples, Blend::Opaque),
        grass_far: scene_pipe(device, "grass far", &l.p_blades, &m.grass, "vs_far", "fs_main", &[], P::TriangleStrip, None, samples, Blend::Opaque),
        terrain: scene_pipe(device, "terrain", &l.p_scene, &m.terrain, "vs_main", "fs_main", &terrain_layouts(), P::TriangleList, None, samples, Blend::Opaque),
        shell: scene_pipe(device, "shell", &l.p_scene, &m.shell, "vs_main", "fs_main", &shell_layout, P::TriangleList, None, samples, Blend::Opaque),
        objects: scene_pipe(
            device,
            "objects",
            &l.p_objs,
            &m.objects,
            "vs_main",
            "fs_main",
            &[mesh_layout()],
            P::TriangleList,
            Some(wgpu::Face::Back),
            samples,
            Blend::Opaque,
        ),
        balls: scene_pipe(
            device,
            "balls",
            &l.p_blades,
            &m.balls,
            "vs_main",
            "fs_main",
            &pos_only,
            P::TriangleList,
            Some(wgpu::Face::Back),
            samples,
            Blend::Opaque,
        ),
        car_body: scene_pipe(device, "car body", &l.p_scene, &m.aircars, "vs_body", "fs_body", &[mesh_layout()], P::TriangleList, None, samples, Blend::Opaque),
        car_glow: scene_pipe(device, "car glow", &l.p_scene, &m.aircars, "vs_glow", "fs_glow", &[], P::TriangleStrip, None, samples, Blend::Additive),
        lines: scene_pipe(device, "lines", &l.p_scene, &m.lines, "vs_main", "fs_main", &line_layout, P::TriangleList, None, samples, Blend::Additive),
    }
}

/// 7 x 50 chunks per valley.
fn build_chunks(world: &World) -> Vec<Chunk> {
    let hw = colony::half_land_width();
    let (nu, nv) = (24usize, 200usize);
    let cw = 2.0 * hw / nu as f32;
    let cl = L / nv as f32;
    let mut out = Vec::new();
    for k in 0..3 {
        let uc = k as f32 * STRIP_ANGLE * R;
        for i in 0..nu {
            for j in 0..nv {
                let u0 = uc - hw + i as f32 * cw;
                let v0 = j as f32 * cl;
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for a in 0..=8 {
                    for b in 0..=8 {
                        let h = world.height(u0 + cw * a as f32 / 8.0, v0 + cl * b as f32 / 8.0).max(0.0);
                        lo = lo.min(h);
                        hi = hi.max(h);
                    }
                }
                let um = u0 + cw * 0.5;
                let vm = v0 + cl * 0.5;
                let centre = colony::surf_to_world(um, vm, (lo + hi) * 0.5);
                let radius = ((cw * 0.5).powi(2) + (cl * 0.5).powi(2) + ((hi - lo) * 0.5 + 6.0).powi(2)).sqrt() + 8.0;
                out.push(Chunk { inst: [u0, v0, cw, cl], centre, radius, strip: k });
            }
        }
    }
    out
}

/// Windows (glass strips between the valleys), end caps and the spindle.
fn shell_mesh() -> (Vec<[f32; 4]>, Vec<u32>) {
    use std::f32::consts::{PI, TAU};
    let mut v: Vec<[f32; 4]> = Vec::new();
    let mut idx = Vec::new();
    let mut grid = |v: &mut Vec<[f32; 4]>, nu: usize, nv: usize, f: &dyn Fn(f32, f32) -> Vec3, kind: f32| {
        let base = v.len() as u32;
        for j in 0..=nv {
            for i in 0..=nu {
                let p = f(i as f32 / nu as f32, j as f32 / nv as f32);
                v.push([p.x, p.y, p.z, kind]);
            }
        }
        let row = nu as u32 + 1;
        for j in 0..nv as u32 {
            for i in 0..nu as u32 {
                let a = base + j * row + i;
                idx.extend_from_slice(&[a, a + 1, a + row, a + 1, a + row + 1, a + row]);
            }
        }
    };
    let hl = colony::HALF_LAND;
    let sr = crate::physics::SPINDLE_R;
    for k in 0..3 {
        let t0 = k as f32 * STRIP_ANGLE + hl - 0.002;
        let t1 = (k as f32 + 1.0) * STRIP_ANGLE - hl + 0.002;
        grid(
            &mut v,
            24,
            256,
            &|a, b| {
                let th = t0 + (t1 - t0) * a;
                Vec3::new(R * th.cos(), R * th.sin(), b * L)
            },
            0.0,
        );
    }
    for z in [0.0, L] {
        grid(
            &mut v,
            192,
            48,
            &|a, b| {
                let th = a * TAU;
                let r = sr + (R + 2.0 - sr) * b;
                Vec3::new(r * th.cos(), r * th.sin(), z)
            },
            1.0,
        );
    }
    grid(
        &mut v,
        24,
        256,
        &|a, b| {
            let th = a * TAU;
            Vec3::new(sr * th.cos(), sr * th.sin(), b * L)
        },
        2.0,
    );
    // Station modules along the spindle.
    for st in 1..16 {
        let zc = L * st as f32 / 16.0;
        let (rm, hl2) = (if st % 4 == 0 { 120.0 } else { 70.0 }, if st % 4 == 0 { 400.0 } else { 220.0 });
        grid(
            &mut v,
            32,
            6,
            &|a, b| {
                let th = a * TAU;
                // capsule-ish profile: tapered ends
                let t = b * 2.0 - 1.0;
                let r = sr + (rm - sr) * (1.0 - t.powi(8));
                Vec3::new(r * th.cos(), r * th.sin(), zc + t * hl2)
            },
            7.0,
        );
    }
    // ---- exterior ----
    // mirrors (positions are computed in the vertex shader)
    for k in 0..3 {
        grid(&mut v, 16, 64, &|a, b| Vec3::new(a, b, k as f32), 3.0);
    }
    // hull behind the land strips
    for k in 0..3 {
        let t0 = k as f32 * STRIP_ANGLE - hl - 0.002;
        let t1 = k as f32 * STRIP_ANGLE + hl + 0.002;
        grid(
            &mut v,
            48,
            128,
            &|a, b| {
                let th = t0 + (t1 - t0) * a;
                Vec3::new((R + 12.0) * th.cos(), (R + 12.0) * th.sin(), b * L)
            },
            4.0,
        );
    }
    // agricultural ring at the sun end
    grid(
        &mut v,
        256,
        4,
        &|a, b| {
            let th = a * TAU;
            let r = R * 1.05 + b * R * 0.35;
            Vec3::new(r * th.cos(), r * th.sin(), L + 300.0)
        },
        5.0,
    );
    // sky sphere (unit; follows the camera in the vertex shader)
    grid(
        &mut v,
        48,
        24,
        &|a, b| {
            let th = a * TAU;
            let ph = (b - 0.5) * PI;
            Vec3::new(ph.cos() * th.cos(), ph.cos() * th.sin(), ph.sin())
        },
        6.0,
    );
    let _ = CIRC;
    (v, idx)
}
