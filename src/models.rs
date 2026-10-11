//! Rigid-part models built in Blender (`tools/blender/*.py` → `assets/models/*.omdl`).
//!
//! Every part is a small indexed mesh in its own bone frame (game axes: X = left,
//! Y = up, Z = forward); sockets are named points on a model (grip, muzzle, sight…).
//! The files are embedded so the binary has no runtime asset paths.

use std::collections::HashMap;
use std::sync::OnceLock;

use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PropVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// sqrt-encoded albedo, baked ambient occlusion
    pub color: [u8; 4],
    /// metal, roughness, flags (1 team, 2 emissive, 4 visor), emissive×16
    pub mat: [u8; 4],
}

#[derive(Clone, Debug)]
pub struct Part {
    pub name: String,
    pub radius: f32,
    pub first_index: u32,
    pub count: u32,
    pub base_vertex: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct Socket {
    pub pos: Vec3,
    pub fwd: Vec3,
    pub up: Vec3,
}

#[derive(Default)]
pub struct ModelLib {
    pub parts: Vec<Part>,
    by_name: HashMap<String, u16>,
    sockets: HashMap<(String, String), Socket>,
    pub verts: Vec<PropVertex>,
    pub indices: Vec<u32>,
}

const FILES: [&[u8]; 2] = [include_bytes!("../assets/models/operative.omdl"), include_bytes!("../assets/models/weapons.omdl")];

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.at..self.at + n).ok_or("omdl: truncated")?;
        self.at += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn vec3(&mut self) -> Result<Vec3, String> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

impl ModelLib {
    pub fn parse(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut r = Reader { b: bytes, at: 0 };
        if r.take(4)? != b"OMDL" {
            return Err("omdl: bad magic".into());
        }
        let version = r.u32()?;
        if version != 1 {
            return Err(format!("omdl: version {version}"));
        }
        let n = r.u32()?;
        for _ in 0..n {
            let name = r.str()?;
            let radius = r.f32()?;
            let nv = r.u32()? as usize;
            let ni = r.u32()? as usize;
            let raw = r.take(nv * 32)?;
            let base_vertex = self.verts.len() as i32;
            self.verts.extend(raw.as_chunks::<32>().0.iter().map(|c| bytemuck::pod_read_unaligned::<PropVertex>(c)));
            let first_index = self.indices.len() as u32;
            let raw = r.take(ni * 4)?;
            self.indices.extend(raw.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)));
            self.by_name.insert(name.clone(), self.parts.len() as u16);
            self.parts.push(Part { name, radius, first_index, count: ni as u32, base_vertex });
        }
        let ns = r.u32()?;
        for _ in 0..ns {
            let model = r.str()?;
            let name = r.str()?;
            let (pos, fwd, up) = (r.vec3()?, r.vec3()?, r.vec3()?);
            self.sockets.insert((model, name), Socket { pos, fwd, up });
        }
        Ok(())
    }

    pub fn part(&self, name: &str) -> u16 {
        *self.by_name.get(name).unwrap_or_else(|| panic!("model part '{name}' missing"))
    }

    pub fn socket(&self, model: &str, name: &str) -> Socket {
        self.sockets.get(&(model.to_string(), name.to_string())).copied().unwrap_or(Socket { pos: Vec3::ZERO, fwd: Vec3::Z, up: Vec3::Y })
    }

    pub fn tris(&self) -> usize {
        self.indices.len() / 3
    }
}

/// The embedded model library (parsed once).
pub fn lib() -> &'static ModelLib {
    static LIB: OnceLock<ModelLib> = OnceLock::new();
    LIB.get_or_init(|| {
        let mut l = ModelLib::default();
        for f in FILES {
            if let Err(e) = l.parse(f) {
                log::error!("{e}");
            }
        }
        log::info!("models: {} parts, {} tris", l.parts.len(), l.tris());
        l
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn embedded_models_parse() {
        let l = super::lib();
        for p in ["torso", "head", "thigh_L", "hand_R", "rifle", "pistol", "sniper", "rifle_mag", "casing"] {
            let i = l.part(p) as usize;
            assert!(l.parts[i].count > 0, "{p}");
        }
        let g = l.socket("rifle", "muzzle");
        assert!(g.pos.z > 0.5, "muzzle in front of the grip: {:?}", g.pos);
        assert!(l.verts.iter().all(|v| v.pos.iter().all(|x| x.is_finite() && x.abs() < 2.0)));
    }
}
