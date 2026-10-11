//! Interactive window (winit + egui): orbit / free / station / cinematic
//! cameras, live settings, HUD with t, τ, dτ/dt and per-pass GPU timings.

use crate::ui::{self, CamMode, UiState};
use crate::Options;
use anyhow::Result;
use glam::DVec3;
use oneill_colony::bh::render::{BhRenderer, Gpu, Settings};
use oneill_colony::bh::scenario::{self, CamKey, Scenario};
use oneill_colony::bh::sim::{CamKin, Sim};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    r: BhRenderer,
    sim: Sim,
    sc: Scenario,
    settings: Settings,
    ui: UiState,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    keys: HashSet<KeyCode>,
    dragging: bool,
    looking: bool,
    free_pos: DVec3,
    free_yaw: f64,
    free_pitch: f64,
    last: Instant,
    fps: (f32, f32, u32),
    hide_ui: bool,
    first: bool,
}

struct App {
    opts: Option<Options>,
    state: Option<State>,
}

pub fn run(opts: Options) -> Result<()> {
    let el = EventLoop::new()?;
    let mut app = App { opts: Some(opts), state: None };
    el.run_app(&mut app)?;
    Ok(())
}

impl State {
    fn camera(&mut self, dt: f64) -> (f64, (DVec3, DVec3, DVec3)) {
        let fov = self.ui.orbit.3;
        match self.ui.cam_mode {
            CamMode::Cinematic => {
                let t = self.sim.real_time % self.sc.camera.keys.last().map(|k| k.time.max(1.0)).unwrap_or(1.0);
                let k = scenario::sample_keys(&self.sc.camera.keys, t);
                (k.fov, self.sim.place_camera(&k))
            }
            CamMode::Orbit | CamMode::Station => {
                let (r, incl, azim, _) = self.ui.orbit;
                let prev = self.sim.cam_kin;
                if self.ui.cam_mode == CamMode::Station {
                    self.sim.cam_kin = CamKin::Station;
                } else if prev == CamKin::Station {
                    self.sim.cam_kin = CamKin::Static;
                }
                let k = CamKey { r, incl, azim, fov, ..CamKey::default() };
                (fov, self.sim.place_camera(&k))
            }
            CamMode::Free => {
                if self.sim.cam_kin == CamKin::Station {
                    self.sim.cam_kin = CamKin::Static;
                }
                let (cy, sy) = (self.free_yaw.cos(), self.free_yaw.sin());
                let (cp, sp) = (self.free_pitch.cos(), self.free_pitch.sin());
                let fwd = DVec3::new(cy * cp, sy * cp, sp);
                let right = fwd.cross(DVec3::Z).normalize_or(DVec3::X);
                let up = right.cross(fwd);
                let mut mv = DVec3::ZERO;
                let k = |c: KeyCode| if self.keys.contains(&c) { 1.0 } else { 0.0 };
                mv += fwd * (k(KeyCode::KeyW) - k(KeyCode::KeyS));
                mv += right * (k(KeyCode::KeyD) - k(KeyCode::KeyA));
                mv += up * (k(KeyCode::KeyE) - k(KeyCode::KeyQ));
                let boost = if self.keys.contains(&KeyCode::ShiftLeft) { 10.0 } else { 1.0 };
                // speed scales smoothly with the distance from the hole
                let r = self.free_pos.length().max(2.2);
                self.free_pos += mv * self.ui.fly_speed * boost * dt * (r / 30.0).clamp(0.02, 50.0);
                self.sim.set_camera_pos(self.free_pos);
                (fov, (fwd, up, right))
            }
        }
    }

    fn redraw(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f64().min(0.1);
        self.last = now;
        self.fps.1 += dt as f32;
        self.fps.2 += 1;
        if self.fps.1 > 0.5 {
            self.fps.0 = self.fps.2 as f32 / self.fps.1;
            self.fps = (self.fps.0, 0.0, 0);
        }
        if self.ui.restart {
            self.sim.reset_station();
            self.ui.restart = false;
        }
        self.r.poll_readbacks();
        let (fov, basis) = self.camera(dt);
        let size = self.window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));
        let marker = self.sim.station_cfg.marker;
        let hf = self.r.hull.upload(&self.gpu.queue, &self.sim, self.sim.cam.pos, self.sim.cam.t_pg);
        let fi = crate::frame_input(&self.sim, fov, basis, marker, dt, self.first, hf);
        self.first = false;
        let mut enc = self.r.render(&self.gpu.r(), w, h, &self.settings, &fi);
        let raw = self.egui_state.take_egui_input(&self.window);
        let info = ui::HudInfo {
            sim: &self.sim,
            timings: self.r.timings_ms,
            has_timestamps: self.gpu.timestamps,
            mean_steps: self.r.mean_steps,
            trace_size: self.r.trace_size(),
            out_size: (w, h),
            exposure: self.r.exposure,
            fps: self.fps.0,
            adapter: &self.gpu.adapter_name,
        };
        let mut sim = self.sim.clone();
        let mut settings = self.settings.clone();
        let hide = self.hide_ui;
        let ui_state = &mut self.ui;
        let out = self.egui_ctx.run(raw, |ctx| {
            if !hide {
                ui::hud(ctx, &info);
                ui::panels(ctx, ui_state, &mut sim, &mut settings);
            }
        });
        self.sim = sim;
        self.settings = settings;
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        let jobs = self.egui_ctx.tessellate(out.shapes, out.pixels_per_point);
        self.r.paint_ui(&self.gpu, &mut enc, &jobs, &out.textures_delta, out.pixels_per_point);
        let Some((surface, _)) = &self.gpu.surface else { return };
        match surface.get_current_texture() {
            Ok(frame) => {
                let view =
                    frame.texture.create_view(&wgpu::TextureViewDescriptor { format: Some(frame.texture.format().remove_srgb_suffix()), ..Default::default() });
                self.r.blit(&mut enc, &view);
                self.r.submit(&self.gpu, enc);
                frame.present();
            }
            Err(e) => {
                log::warn!("surface: {e}");
                let (w, h) = (size.width, size.height);
                self.gpu.resize(w, h);
            }
        }
        self.sim.step(dt);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let Some(opts) = self.opts.take() else { return };
        let sc = match &opts.scenario {
            Some(p) => scenario::load(p).unwrap_or_else(|e| {
                eprintln!("{e}");
                Scenario::default()
            }),
            None => Scenario::default(),
        };
        let (w, h) = opts.size.unwrap_or((1600, 900));
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("Чёрная дыра — Шварцшильд").with_inner_size(winit::dpi::PhysicalSize::new(w, h)))
                .expect("window"),
        );
        let gpu = pollster::block_on(Gpu::new(Some(window.clone()), opts.vsync)).expect("GPU init failed");
        println!("GPU: {}", gpu.adapter_name);
        let mut settings = crate::settings_from(&sc);
        if let Some(q) = &opts.quality {
            settings.quality_idx = crate::render::quality_index(q);
            settings.quality = crate::render::QUALITY[settings.quality_idx];
        }
        let r = BhRenderer::new(&gpu.r(), &settings);
        let sim = Sim::new(&sc);
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        let k0 = sc.camera.keys.first().copied().unwrap_or_default();
        let mut ui = UiState { orbit: (k0.r, k0.incl, k0.azim, k0.fov), ..UiState::default() };
        if opts.scenario.is_some() && sc.camera.keys.len() > 1 {
            ui.cam_mode = CamMode::Cinematic;
        }
        let free_pos = DVec3::new(k0.r, 0.0, 0.0);
        self.state = Some(State {
            window: window.clone(),
            gpu,
            r,
            sim,
            sc,
            settings,
            ui,
            egui_ctx,
            egui_state,
            keys: HashSet::new(),
            dragging: false,
            looking: false,
            free_pos,
            free_yaw: std::f64::consts::PI,
            free_pitch: 0.0,
            last: Instant::now(),
            fps: (0.0, 0.0, 0),
            hide_ui: opts.hide_ui,
            first: true,
        });
        window.request_redraw();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else { return };
        let consumed = st.egui_state.on_window_event(&st.window, &event).consumed;
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => st.gpu.resize(s.width, s.height),
            WindowEvent::RedrawRequested => st.redraw(),
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                if event.state == ElementState::Pressed {
                    if consumed && st.egui_ctx.wants_keyboard_input() {
                        return;
                    }
                    st.keys.insert(code);
                    match code {
                        KeyCode::F1 => st.hide_ui = !st.hide_ui,
                        KeyCode::Space => st.sim.paused = !st.sim.paused,
                        KeyCode::Tab => st.ui.show_panels = !st.ui.show_panels,
                        KeyCode::Digit1 => st.ui.cam_mode = CamMode::Orbit,
                        KeyCode::Digit2 => st.ui.cam_mode = CamMode::Free,
                        KeyCode::Digit3 => st.ui.cam_mode = CamMode::Station,
                        KeyCode::Digit4 => st.ui.cam_mode = CamMode::Cinematic,
                        KeyCode::KeyR => st.ui.restart = true,
                        KeyCode::Escape => el.exit(),
                        _ => {}
                    }
                } else {
                    st.keys.remove(&code);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed && !st.egui_ctx.wants_pointer_input();
                match button {
                    MouseButton::Left => st.dragging = down,
                    MouseButton::Right => st.looking = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if st.egui_ctx.wants_pointer_input() {
                    return;
                }
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                match st.ui.cam_mode {
                    CamMode::Free => st.ui.fly_speed = (st.ui.fly_speed * 1.2f64.powf(d)).clamp(0.01, 1000.0),
                    _ => st.ui.orbit.0 = (st.ui.orbit.0 * 0.9f64.powf(d)).clamp(1.2, 5000.0),
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        let Some(st) = self.state.as_mut() else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if st.dragging && st.ui.cam_mode != CamMode::Free {
                st.ui.orbit.2 -= delta.0 * 0.25;
                st.ui.orbit.1 = (st.ui.orbit.1 + delta.1 * 0.25).clamp(-89.9, 89.9);
            }
            if (st.looking || st.dragging) && st.ui.cam_mode == CamMode::Free {
                st.free_yaw -= delta.0 * 0.003;
                st.free_pitch = (st.free_pitch - delta.1 * 0.003).clamp(-1.55, 1.55);
            }
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let Some(st) = self.state.as_mut() else { return };
        el.set_control_flow(ControlFlow::Poll);
        st.window.request_redraw();
    }
}
