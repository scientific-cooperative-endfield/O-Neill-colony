//! GPU side of the Blender-built props (instanced rigid parts) and of the
//! camera-facing particles. Both draw inside the main scene pass: props with
//! the opaque geometry (and into the sun shadow map), particles last.

use super::*;
use crate::models::{self, PropVertex};

pub const MAX_PROPS: usize = 4096;
pub const MAX_PARTICLES: usize = 8192;

/// One placed part: rows of a 3×4 world transform, team tint, effect flags (see props.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PropInst {
    pub rows: [[f32; 4]; 3],
    pub tint: [f32; 4],
    pub ext: [f32; 4],
}

impl PropInst {
    pub fn new(m: glam::Affine3A, tint: [f32; 4], ext: [f32; 4]) -> PropInst {
        let c = m.matrix3;
        let t = m.translation;
        PropInst {
            rows: [[c.x_axis.x, c.y_axis.x, c.z_axis.x, t.x], [c.x_axis.y, c.y_axis.y, c.z_axis.y, t.y], [c.x_axis.z, c.y_axis.z, c.z_axis.z, t.z]],
            tint,
            ext,
        }
    }
    pub fn pos(&self) -> Vec3 {
        Vec3::new(self.rows[0][3], self.rows[1][3], self.rows[2][3])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleGpu {
    pub pos: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub axis: [f32; 3],
    pub kind: u32,
}
const _: () = assert!(std::mem::size_of::<ParticleGpu>() == 48);
const _: () = assert!(std::mem::size_of::<PropInst>() == 80);

const PROP_ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4, 3 => Unorm8x4];

fn prop_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<PropVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &PROP_ATTRS }
}

pub struct Props {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    inst_buf: wgpu::Buffer,
    inst_bg: wgpu::BindGroup,
    part_buf: wgpu::Buffer,
    part_bg: wgpu::BindGroup,
    module: wgpu::ShaderModule,
    pmodule: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    pipe: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    ppipe: wgpu::RenderPipeline,
    /// (part, first instance, count) for this frame
    draws: Vec<(u16, u32, u32)>,
    particles: u32,
    scratch: Vec<PropInst>,
    order: Vec<(u16, u32)>,
}

impl Props {
    pub(super) fn new(device: &wgpu::Device, l: &Layouts, samples: u32) -> Props {
        let lib = models::lib();
        let dummy = [PropVertex { pos: [0.0; 3], normal: [0.0; 3], color: [0; 4], mat: [0; 4] }];
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prop verts"),
            contents: bytemuck::cast_slice(if lib.verts.is_empty() { &dummy[..] } else { &lib.verts }),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prop indices"),
            contents: bytemuck::cast_slice(if lib.indices.is_empty() { &[0u32; 3][..] } else { &lib.indices }),
            usage: wgpu::BufferUsages::INDEX,
        });
        let inst_buf = buf(device, "prop instances", (MAX_PROPS * 80) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let part_buf = buf(device, "particles", (MAX_PARTICLES * 48) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("props"),
            entries: &[storage(0, true, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, false)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("props"),
            bind_group_layouts: &[&l.scene, &bgl],
            push_constant_ranges: &[],
        });
        let bg = |b: &wgpu::Buffer, label| {
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: &bgl, entries: &[bge(0, b.as_entire_binding())] })
        };
        let inst_bg = bg(&inst_buf, "props");
        let part_bg = bg(&part_buf, "particles");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("props"),
            source: wgpu::ShaderSource::Wgsl(shaders::compose(shaders::PROPS).into()),
        });
        let pmodule = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles"),
            source: wgpu::ShaderSource::Wgsl(shaders::compose(shaders::PARTICLES).into()),
        });
        let shadow = shadow_pipe(device, "prop shadow", &layout, &module, "vs_shadow", &[prop_layout()]);
        let (pipe, ppipe) = Self::pipes(device, &layout, &module, &pmodule, samples);
        Props {
            bgl,
            layout,
            vbuf,
            ibuf,
            inst_buf,
            inst_bg,
            part_buf,
            part_bg,
            module,
            pmodule,
            pipe,
            shadow,
            ppipe,
            draws: Vec::new(),
            particles: 0,
            scratch: Vec::new(),
            order: Vec::new(),
        }
    }

    fn pipes(
        device: &wgpu::Device,
        l: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        pmodule: &wgpu::ShaderModule,
        samples: u32,
    ) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
        let pipe =
            scene_pipe(device, "props", l, module, "vs_main", "fs_main", &[prop_layout()], wgpu::PrimitiveTopology::TriangleList, None, samples, Blend::Opaque);
        let ppipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particles"),
            layout: Some(l),
            vertex: wgpu::VertexState { module: pmodule, entry_point: "vs_main", buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: pmodule,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            multiview: None,
            cache: None,
        });
        (pipe, ppipe)
    }

    pub(super) fn rebuild(&mut self, device: &wgpu::Device, samples: u32) {
        let (a, b) = Self::pipes(device, &self.layout, &self.module, &self.pmodule, samples);
        self.pipe = a;
        self.ppipe = b;
    }

    /// Culls, groups instances by part and uploads; particles arrive sorted back to front.
    pub fn prepare(&mut self, queue: &wgpu::Queue, planes: &[[f32; 4]; 6], cam: Vec3, fd: &FrameData) {
        let lib = models::lib();
        self.order.clear();
        for (i, (part, inst)) in fd.props.iter().enumerate() {
            let Some(p) = lib.parts.get(*part as usize) else { continue };
            let r = p.radius * 1.05 + 0.05;
            let c = inst.pos();
            // near ones stay for the shadow map even when off screen
            if sphere_visible(planes, c, r) || c.distance(cam) < 40.0 {
                self.order.push((*part, i as u32));
            }
        }
        self.order.sort_unstable_by_key(|x| x.0);
        self.order.truncate(MAX_PROPS);
        self.scratch.clear();
        self.draws.clear();
        for &(part, i) in &self.order {
            match self.draws.last_mut() {
                Some(d) if d.0 == part => d.2 += 1,
                _ => self.draws.push((part, self.scratch.len() as u32, 1)),
            }
            self.scratch.push(fd.props[i as usize].1);
        }
        if !self.scratch.is_empty() {
            queue.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&self.scratch));
        }
        let n = fd.particles.len().min(MAX_PARTICLES);
        self.particles = n as u32;
        if n > 0 {
            queue.write_buffer(&self.part_buf, 0, bytemuck::cast_slice(&fd.particles[..n]));
        }
    }

    fn draw_parts<'a>(&'a self, rp: &mut wgpu::RenderPass<'a>) {
        let lib = models::lib();
        rp.set_bind_group(1, &self.inst_bg, &[]);
        rp.set_vertex_buffer(0, self.vbuf.slice(..));
        rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        for &(part, first, count) in &self.draws {
            let p = &lib.parts[part as usize];
            rp.draw_indexed(p.first_index..p.first_index + p.count, p.base_vertex, first..first + count);
        }
    }

    pub fn draw_shadow<'a>(&'a self, rp: &mut wgpu::RenderPass<'a>) {
        if !self.draws.is_empty() {
            rp.set_pipeline(&self.shadow);
            self.draw_parts(rp);
        }
    }

    pub fn draw_opaque<'a>(&'a self, rp: &mut wgpu::RenderPass<'a>) {
        if !self.draws.is_empty() {
            rp.set_pipeline(&self.pipe);
            self.draw_parts(rp);
        }
    }

    pub fn draw_particles<'a>(&'a self, rp: &mut wgpu::RenderPass<'a>) {
        if self.particles > 0 {
            rp.set_pipeline(&self.ppipe);
            rp.set_bind_group(1, &self.part_bg, &[]);
            rp.draw(0..6, 0..self.particles);
        }
    }
}
