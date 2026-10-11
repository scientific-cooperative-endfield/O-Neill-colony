//! Graphics settings with presets from "Чайник" (anything with a GPU) to Ultra.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Teapot,
    Low,
    Medium,
    High,
    Ultra,
    Custom,
}

impl Preset {
    pub const ALL: [Preset; 5] = [Preset::Teapot, Preset::Low, Preset::Medium, Preset::High, Preset::Ultra];
    pub fn label(self) -> &'static str {
        match self {
            Preset::Teapot => "Чайник",
            Preset::Low => "Низкие",
            Preset::Medium => "Средние",
            Preset::High => "Высокие",
            Preset::Ultra => "Ультра",
            Preset::Custom => "Свои",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AaMode {
    Off,
    Fxaa,
    Msaa2,
    Msaa4,
}

impl AaMode {
    pub const ALL: [AaMode; 4] = [AaMode::Off, AaMode::Fxaa, AaMode::Msaa2, AaMode::Msaa4];
    pub fn label(self) -> &'static str {
        match self {
            AaMode::Off => "Выкл",
            AaMode::Fxaa => "FXAA",
            AaMode::Msaa2 => "MSAA 2x + FXAA",
            AaMode::Msaa4 => "MSAA 4x",
        }
    }
    pub fn samples(self) -> u32 {
        match self {
            AaMode::Msaa2 => 2,
            AaMode::Msaa4 => 4,
            _ => 1,
        }
    }
    pub fn fxaa(self) -> bool {
        matches!(self, AaMode::Fxaa | AaMode::Msaa2)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tonemap {
    AgX,
    Aces,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graphics {
    pub preset: Preset,
    pub render_scale: f32,
    pub aa: AaMode,
    /// 0 = shadows off, otherwise shadow-map resolution.
    pub shadow_res: u32,
    pub shadow_dist: f32,
    pub grass_spacing: f32,
    pub grass_radius: f32,
    pub grass_lod: f32,
    pub tree_dist: f32,
    pub terrain_detail: f32,
    pub aircars: u32,
    pub bloom: bool,
    pub bloom_strength: f32,
    pub sharpen: f32,
    pub tonemap: Tonemap,
    pub exposure_bias: f32,
    pub saturation: f32,
    pub vignette: f32,
    pub fog: f32,
    pub vsync: bool,
    pub fps_cap: u32,
}

impl Graphics {
    pub fn preset(p: Preset) -> Self {
        let base = Graphics {
            preset: p,
            render_scale: 1.0,
            aa: AaMode::Fxaa,
            shadow_res: 1024,
            shadow_dist: 120.0,
            grass_spacing: 0.09,
            grass_radius: 55.0,
            grass_lod: 15.0,
            tree_dist: 450.0,
            terrain_detail: 1.0,
            aircars: 600,
            bloom: true,
            bloom_strength: 0.05,
            sharpen: 0.3,
            tonemap: Tonemap::AgX,
            exposure_bias: 0.0,
            saturation: 1.05,
            vignette: 0.25,
            fog: 1.0,
            vsync: true,
            fps_cap: 0,
        };
        match p {
            Preset::Teapot => Graphics {
                render_scale: 0.5,
                shadow_res: 0,
                grass_spacing: 0.16,
                grass_radius: 28.0,
                grass_lod: 8.0,
                tree_dist: 170.0,
                terrain_detail: 0.5,
                aircars: 150,
                sharpen: 0.7,
                vignette: 0.15,
                ..base
            },
            Preset::Low => Graphics {
                render_scale: 0.7,
                shadow_res: 0,
                grass_spacing: 0.12,
                grass_radius: 40.0,
                grass_lod: 11.0,
                tree_dist: 280.0,
                terrain_detail: 0.75,
                aircars: 300,
                sharpen: 0.5,
                ..base
            },
            Preset::Medium | Preset::Custom => base,
            Preset::High => Graphics {
                aa: AaMode::Msaa4,
                shadow_res: 2048,
                shadow_dist: 180.0,
                grass_spacing: 0.07,
                grass_radius: 75.0,
                grass_lod: 20.0,
                tree_dist: 700.0,
                terrain_detail: 1.4,
                aircars: 1000,
                sharpen: 0.2,
                ..base
            },
            Preset::Ultra => Graphics {
                aa: AaMode::Msaa4,
                shadow_res: 4096,
                shadow_dist: 260.0,
                grass_spacing: 0.055,
                grass_radius: 95.0,
                grass_lod: 26.0,
                tree_dist: 1100.0,
                terrain_detail: 2.0,
                aircars: 1600,
                sharpen: 0.15,
                ..base
            },
        }
    }

    pub fn keep(&self, d: f32) -> f32 {
        let thin = self.grass_lod * 0.6;
        (thin / d.max(thin)).powf(1.55).clamp(0.05, 1.0)
    }

    /// Instance-buffer capacities for grass (near, far).
    pub fn grass_capacity(&self) -> (u32, u32) {
        use std::f32::consts::PI;
        let per_m2 = 1.0 / (self.grass_spacing * self.grass_spacing);
        let near = PI * self.grass_lod * self.grass_lod * per_m2 * 0.6;
        let steps = 256;
        let dr = (self.grass_radius - self.grass_lod).max(1.0) / steps as f32;
        let mut far = 0.0;
        for k in 0..steps {
            let d = self.grass_lod + (k as f32 + 0.5) * dr;
            far += self.keep(d) * 2.0 * PI * d * dr;
        }
        ((near as u32).max(8192), ((far * per_m2 * 0.5) as u32).max(8192))
    }
}

/// Command line: `--preset teapot|low|medium|high|ultra`, `--no-vsync`.
pub fn from_args() -> Graphics {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut g = Graphics::preset(Preset::Medium);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--preset" | "-p" | "--quality" | "-q" => {
                let p = match args.get(i + 1).map(String::as_str) {
                    Some("teapot") | Some("chainik") | Some("potato") => Preset::Teapot,
                    Some("low") => Preset::Low,
                    Some("high") => Preset::High,
                    Some("ultra") => Preset::Ultra,
                    _ => Preset::Medium,
                };
                let vs = g.vsync;
                g = Graphics::preset(p);
                g.vsync = vs;
                i += 1;
            }
            "--no-vsync" => g.vsync = false,
            "-h" | "--help" => {
                println!("oneill_colony [--preset teapot|low|medium|high|ultra] [--no-vsync]");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    g
}
