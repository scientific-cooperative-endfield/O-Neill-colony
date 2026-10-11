//! GPU side of the station snapshots: a ring of node slots plus a small header
//! array (newest first) that the tracer searches by emission time.
use super::sim::{NodeGpu, Sim};
use super::structure::N_NODES;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;

pub const SLOTS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SnapGpu {
    pub c: [f32; 4],
    pub u: [f32; 4],
    pub info: [f32; 4],
}

/// Per-frame hull parameters for the trace uniform.
#[derive(Clone, Copy, Debug, Default)]
pub struct HullFrame {
    pub n: u32,
    pub near: f32,
    pub union_r: f32,
    pub delay: bool,
    pub light: [f32; 4],
    pub heat: [f32; 4],
}

pub struct HullGpu {
    pub snaps_buf: wgpu::Buffer,
    pub nodes_buf: wgpu::Buffer,
    keys: [u64; SLOTS],
}

impl HullGpu {
    pub fn new(device: &wgpu::Device) -> HullGpu {
        let mk = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        HullGpu {
            snaps_buf: mk("hull-snaps", (SLOTS * std::mem::size_of::<SnapGpu>()) as u64),
            nodes_buf: mk("hull-nodes", (SLOTS * N_NODES * std::mem::size_of::<NodeGpu>()) as u64),
            keys: [u64::MAX; SLOTS],
        }
    }

    /// Uploads the live state and any new history snapshots; `cam` is the camera
    /// position (M) the headers are made relative to.
    pub fn upload(&mut self, queue: &wgpu::Queue, sim: &Sim, cam: DVec3, cam_t_pg: f64) -> HullFrame {
        let snaps = &sim.snapshots;
        if snaps.is_empty() {
            return HullFrame::default();
        }
        let node_size = (N_NODES * std::mem::size_of::<NodeGpu>()) as u64;
        let wanted: Vec<u64> = snaps.iter().skip(1).map(|s| s.t_pg.to_bits()).collect();
        let mut headers = Vec::with_capacity(snaps.len());
        for (i, s) in snaps.iter().enumerate() {
            let slot = if i == 0 {
                queue.write_buffer(&self.nodes_buf, 0, bytemuck::cast_slice(&s.nodes));
                0
            } else {
                let key = s.t_pg.to_bits();
                match (1..SLOTS).find(|&j| self.keys[j] == key) {
                    Some(j) => j,
                    None => {
                        let j = (1..SLOTS).find(|&j| !wanted.contains(&self.keys[j])).unwrap_or(SLOTS - 1);
                        self.keys[j] = key;
                        queue.write_buffer(&self.nodes_buf, j as u64 * node_size, bytemuck::cast_slice(&s.nodes));
                        j
                    }
                }
            };
            let c = s.centre - cam;
            headers.push(SnapGpu {
                c: [c.x as f32, c.y as f32, c.z as f32, (s.t_pg - cam_t_pg) as f32],
                u: [s.u[1] as f32, s.u[2] as f32, s.u[3] as f32, s.u[0] as f32],
                info: [s.radius as f32, slot as f32, s.nodes.len() as f32, 0.0],
            });
        }
        queue.write_buffer(&self.snaps_buf, 0, bytemuck::cast_slice(&headers));
        let s0 = &snaps[0];
        let union_r = snaps.iter().map(|s| (s.centre - s0.centre).length() + s.radius).fold(s0.radius, f64::max);
        let r_cam = cam.length().max(1e-6);
        let near = if s0.radius < 0.02 * r_cam { (0.05 * r_cam).max((s0.centre - cam).length() + 2.0 * s0.radius) } else { 0.0 };
        // disk illumination of the hull (rest frame), warm white
        let r_st = s0.centre.length().max(2.0);
        let inten = (2.5 * (10.0 / r_st).powi(2)).clamp(0.05, 20.0) as f32;
        let (mut lo, mut hi, mut desync) = (f32::MAX, f32::MIN, 0.0f32);
        for n in &s0.nodes {
            if n.h[3] >= 0.0 {
                lo = lo.min(n.h[3]);
                hi = hi.max(n.h[3]);
            }
            desync = desync.max(n.e[0].abs());
        }
        if lo > hi {
            (lo, hi) = (0.0, 1.0);
        }
        let p = (sim.structure.pressure / sim.structure.spec.pressure) as f32;
        HullFrame {
            n: snaps.len() as u32,
            near: near as f32,
            union_r: union_r as f32,
            delay: sim.light_delay,
            light: [inten, inten * 0.86, inten * 0.72, p],
            heat: [lo, hi, desync, 0.0],
        }
    }
}
