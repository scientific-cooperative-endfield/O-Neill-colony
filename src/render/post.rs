//! HDR post-processing: physically based bloom (13-tap downsample with Karis
//! average, tent upsample), AgX / ACES tonemapping, saturation & vignette,
//! then FXAA + contrast-adaptive sharpening while upscaling to the window.

use bytemuck::{Pod, Zeroable};

use crate::shaders;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const LEVELS: usize = 6;
const SLOT: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct PostParams {
    pub texel: [f32; 4],
    pub tone: [f32; 4],
    pub misc: [f32; 4],
}

pub struct PostSettings {
    pub bloom: bool,
    pub bloom_strength: f32,
    pub exposure: f32,
    pub tonemap: u32,
    pub sharpen: f32,
    pub fxaa: bool,
    pub vignette: f32,
    pub saturation: f32,
    pub time: f32,
}

pub struct Post {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    final_: wgpu::RenderPipeline,
    mips: Vec<(wgpu::TextureView, [u32; 2])>,
    ldr: wgpu::TextureView,
    dummy: wgpu::TextureView,
    groups: Vec<wgpu::BindGroup>, // prefilter, down x5, up x5, composite, final
    size: [u32; 2],
}

fn tex(device: &wgpu::Device, label: &str, size: [u32; 2], format: wgpu::TextureFormat) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

impl Post {
    pub fn new(device: &wgpu::Device, out_format: wgpu::TextureFormat, hdr: &wgpu::TextureView, size: [u32; 2]) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("post"), source: wgpu::ShaderSource::Wgsl(shaders::POST.into()) });
        let tex_entry = |b| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                tex_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex_entry(3),
            ],
        });
        let pl =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("post"), bind_group_layouts: &[&layout], push_constant_ranges: &[] });
        let make = |fs: &str, format: wgpu::TextureFormat, additive: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fs),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: "vs_fullscreen", buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: fs,
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: if additive {
                            Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent::OVER,
                            })
                        } else {
                            None
                        },
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post params"),
            size: SLOT * 13,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut p = Post {
            prefilter: make("fs_bloom_prefilter", HDR_FORMAT, false),
            down: make("fs_bloom_down", HDR_FORMAT, false),
            up: make("fs_bloom_up", HDR_FORMAT, true),
            composite: make("fs_composite", LDR_FORMAT, false),
            final_: make("fs_final", out_format, false),
            layout,
            sampler,
            params,
            mips: Vec::new(),
            ldr: tex(device, "ldr", [1, 1], LDR_FORMAT),
            dummy: tex(device, "dummy", [1, 1], HDR_FORMAT),
            groups: Vec::new(),
            size,
        };
        p.resize(device, hdr, size);
        p
    }

    pub fn resize(&mut self, device: &wgpu::Device, hdr: &wgpu::TextureView, size: [u32; 2]) {
        self.size = size;
        self.mips.clear();
        let mut s = size;
        for i in 0..LEVELS {
            s = [(s[0] / 2).max(1), (s[1] / 2).max(1)];
            self.mips.push((tex(device, &format!("bloom{i}"), s, HDR_FORMAT), s));
        }
        self.ldr = tex(device, "ldr", size, LDR_FORMAT);
        let group = |slot: u64, src: &wgpu::TextureView, extra: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params,
                            offset: slot * SLOT,
                            size: wgpu::BufferSize::new(48),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(extra) },
                ],
            })
        };
        let m0 = &self.mips[0].0;
        let d = &self.dummy;
        let mut groups = vec![group(0, hdr, d)];
        for i in 1..LEVELS {
            groups.push(group(i as u64, &self.mips[i - 1].0, d));
        }
        for i in (0..LEVELS - 1).rev() {
            groups.push(group((LEVELS + (LEVELS - 2 - i)) as u64, &self.mips[i + 1].0, d));
        }
        groups.push(group(11, hdr, m0));
        groups.push(group(12, &self.ldr, d));
        self.groups = groups;
    }

    pub fn run(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, out: &wgpu::TextureView, s: &PostSettings) {
        let texel = |sz: [u32; 2]| [1.0 / sz[0] as f32, 1.0 / sz[1] as f32];
        let mut data = vec![0u8; (SLOT * 13) as usize];
        let mut put = |slot: usize, p: PostParams| {
            data[slot * SLOT as usize..slot * SLOT as usize + 48].copy_from_slice(bytemuck::bytes_of(&p));
        };
        let t = texel(self.size);
        put(0, PostParams { texel: [t[0], t[1], 1.6, 0.8], ..Default::default() });
        for i in 1..LEVELS {
            let t = texel(self.mips[i - 1].1);
            put(i, PostParams { texel: [t[0], t[1], 0.0, 0.0], ..Default::default() });
        }
        for (n, i) in (0..LEVELS - 1).rev().enumerate() {
            let t = texel(self.mips[i + 1].1);
            put(LEVELS + n, PostParams { texel: [t[0], t[1], 0.0, 0.0], ..Default::default() });
        }
        let tone = [s.exposure, if s.bloom { s.bloom_strength } else { 0.0 }, s.tonemap as f32, s.sharpen];
        let misc = [if s.fxaa { 1.0 } else { 0.0 }, s.vignette, s.time, s.saturation];
        put(11, PostParams { texel: [t[0], t[1], 0.0, 0.0], tone, misc });
        put(12, PostParams { texel: [t[0], t[1], 0.0, 0.0], tone, misc });
        queue.write_buffer(&self.params, 0, &data);

        let pass = |encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, load: bool, pipe: &wgpu::RenderPipeline, g: &wgpu::BindGroup| {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if load { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color::BLACK) },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, g, &[]);
            rp.draw(0..3, 0..1);
        };
        if s.bloom {
            pass(encoder, &self.mips[0].0, false, &self.prefilter, &self.groups[0]);
            for i in 1..LEVELS {
                pass(encoder, &self.mips[i].0, false, &self.down, &self.groups[i]);
            }
            for (n, i) in (0..LEVELS - 1).rev().enumerate() {
                pass(encoder, &self.mips[i].0, true, &self.up, &self.groups[LEVELS + n]);
            }
        }
        pass(encoder, &self.ldr, false, &self.composite, &self.groups[11]);
        pass(encoder, out, false, &self.final_, &self.groups[12]);
    }
}
