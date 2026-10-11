//! Black-hole mode: Schwarzschild ray tracing of the shadow, photon ring,
//! lensed sky and Doppler-boosted accretion disk (wgpu compute), with a
//! station marker following a timelike geodesic and three time modes.
//!
//! * `cargo run --release --bin blackhole`                       — window (egui)
//! * `cargo run --release --bin blackhole -- --headless --scenario scenarios/edge_on.toml --frames 1 --out out/`
//! * `cargo run --release --bin blackhole -- --info`             — preset numbers
//! * `cargo test --release --bin blackhole`                      — physics tests

mod app;
mod ui;

use anyhow::Result;
use glam::DVec3;
use oneill_colony::bh::physics::*;
use oneill_colony::bh::render::{BhRenderer, Gpu};
use oneill_colony::bh::scenario::Scenario;
use oneill_colony::bh::sim::Sim;
use oneill_colony::bh::{fracture, render, scenario};

fn info() {
    for p in PRESETS.iter() {
        let m = m_geo(p.solar);
        let rt = tidal_radius(m, 32_000.0, G_EARTH);
        println!("{}: M = {:.4e} m, r_s = {:.4e} m, r_tidal(1g, 32 km) = {:.4e} m = {:.1} M", p.name, m, 2.0 * m, rt, rt / m);
        let r0 = 20.0 * m;
        println!("   tau(fall from rest at 20M to horizon) = {:.4e} s", fall_time_analytic(m, r0, 2.0 * m) / C);
    }
    let mut g = Geo::at_rest(1.0, DVec3::new(10.0, 0.0, 0.0));
    while g.r() > 1.0 && !g.dead {
        g.step_t(1.0, 0.05);
    }
    println!("geodesic crossed the horizon regularly: r = {:.3} M, tau = {:.4} M", g.r(), g.tau);
    let a = fracture::load_or_generate("assets/fracture.bin");
    println!("fracture asset: {} pieces, {} vertices, {} indices", a.pieces.len(), a.vertices.len(), a.indices.len());
}

pub struct Options {
    pub headless: bool,
    pub scenario: Option<String>,
    pub frames: Option<u32>,
    pub out: String,
    pub size: Option<(u32, u32)>,
    pub quality: Option<String>,
    pub hide_ui: bool,
    pub vsync: bool,
}

fn parse_options() -> (Options, bool) {
    let mut o = Options { headless: false, scenario: None, frames: None, out: "out".into(), size: None, quality: None, hide_ui: false, vsync: true };
    let mut info = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--info" => info = true,
            "--headless" => o.headless = true,
            "--scenario" => {
                o.scenario = Some(next);
                i += 1;
            }
            "--frames" => {
                o.frames = next.parse().ok();
                i += 1;
            }
            "--out" => {
                o.out = next;
                i += 1;
            }
            "--size" => {
                o.size = next.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)));
                i += 1;
            }
            "--quality" | "--preset" => {
                o.quality = Some(next);
                i += 1;
            }
            "--hide-ui" => o.hide_ui = true,
            "--no-vsync" => o.vsync = false,
            other => eprintln!("unknown argument: {other}"),
        }
        i += 1;
    }
    (o, info)
}

pub use oneill_colony::bh::frame::{frame_input, settings_from};

fn run_headless(o: &Options) -> Result<()> {
    let mut sc = match &o.scenario {
        Some(p) => scenario::load(p)?,
        None => Scenario::default(),
    };
    if let Some(n) = o.frames {
        sc.frames = n;
    }
    if let Some((w, h)) = o.size {
        sc.width = w;
        sc.height = h;
    }
    if let Some(q) = &o.quality {
        sc.quality = q.clone();
    }
    if o.hide_ui {
        sc.hud = false;
    }
    let out = std::path::PathBuf::from(&o.out);
    std::fs::create_dir_all(&out)?;
    let gpu = pollster::block_on(Gpu::new(None, false))?;
    let settings = settings_from(&sc);
    let mut sim = Sim::new(&sc);
    let mut r = BhRenderer::new(&gpu.r(), &settings);
    let ctx = egui::Context::default();
    ctx.set_visuals(egui::Visuals::dark());
    let mut csv = String::from("frame,real_time_s,mean_steps,trace_w,trace_h,gpu_trace_ms,gpu_sky_ms,gpu_upscale_ms,gpu_expo_ms,gpu_bloom_ms,gpu_final_ms,station_r_M,station_tau_M,t_remote_M,state_hash,fragments,first_break_r_M\n");
    log::info!("scenario '{}' · {} frames · {}×{} · quality {}", sc.name, sc.frames, sc.width, sc.height, sc.quality);
    for f in 0..sc.frames {
        let key = scenario::sample_keys(&sc.camera.keys, sim.real_time);
        let basis = sim.place_camera(&key);
        let hf = r.hull.upload(&gpu.queue, &sim, sim.cam.pos, sim.cam.t_pg);
        let fi = frame_input(&sim, key.fov, basis, sc.station.marker, sc.dt, f == 0, hf);
        let enc = r.render(&gpu.r(), sc.width, sc.height, &settings, &fi);
        r.submit(&gpu, enc);
        gpu.device.poll(wgpu::Maintain::Wait);
        r.poll_readbacks();
        if sc.hud {
            // the HUD of a headless frame shows this frame's own GPU timings
            let info = ui::HudInfo {
                sim: &sim,
                timings: r.timings_ms,
                has_timestamps: gpu.timestamps,
                mean_steps: r.mean_steps,
                trace_size: r.trace_size(),
                out_size: (sc.width, sc.height),
                exposure: r.exposure,
                fps: 0.0,
                adapter: &gpu.adapter_name,
            };
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(sc.width as f32, sc.height as f32))),
                ..Default::default()
            };
            // egui lays out new areas invisibly on their first pass: run twice and
            // keep the texture deltas of both passes
            let first = ctx.run(raw.clone(), |c| ui::hud(c, &info));
            let mut outp = ctx.run(raw, |c| ui::hud(c, &info));
            let mut tex = first.textures_delta;
            tex.append(std::mem::take(&mut outp.textures_delta));
            let jobs = ctx.tessellate(outp.shapes, outp.pixels_per_point);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            r.paint_ui(&gpu, &mut enc, &jobs, &tex, outp.pixels_per_point);
            gpu.queue.submit(Some(enc.finish()));
        }
        let capture = sc.capture.is_empty() || sc.capture.contains(&f);
        if capture {
            let (w, h, px) = r.read_frame(&gpu)?;
            let name = sc.capture.iter().position(|&c| c == f).and_then(|i| sc.names.get(i).cloned()).unwrap_or_else(|| format!("{}_{:05}", sc.name, f));
            let path = out.join(format!("{name}.png"));
            render::write_png(&path, w, h, &px)?;
            log::info!("frame {f}: {} · {:.1} steps/ray · trace {:.1} ms (lavapipe)", path.display(), r.mean_steps, r.timings_ms[0]);
        }
        let t = r.timings_ms;
        csv += &format!(
            "{f},{:.4},{:.2},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:016x},{},{:.4}\n",
            sim.real_time,
            r.mean_steps,
            r.trace_size().0,
            r.trace_size().1,
            t[0],
            t[1],
            t[2],
            t[3],
            t[4],
            t[5],
            sim.station.r(),
            sim.station.tau,
            sim.t_remote,
            sim.state_hash(),
            sim.structure.live_fragments(),
            sim.structure.first_break.as_ref().map(|b| b.r).unwrap_or(0.0)
        );
        sim.step(sc.dt);
    }
    std::fs::write(out.join(format!("{}_stats.csv", sc.name)), csv)?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=warn,naga=warn")).init();
    let (o, info_only) = parse_options();
    if info_only {
        info();
        return Ok(());
    }
    if o.headless {
        return run_headless(&o);
    }
    app::run(o)
}
