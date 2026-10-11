#![allow(dead_code)]
//! O'Neill colony (Island Three) simulator.
//! wgpu + WGSL compute shaders, procedural world, egui interface.

mod avatar;
mod bhlink;
mod combat;
mod fx;
mod models;
use oneill_world::{colony, physics, world};
mod render;
mod settings;
mod shaders;
mod transport;
mod ui;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use colony::{world_to_surf, CIRC, R};
use physics::{Ball, Frame, MoveInput, Player, BALL_TYPES, MAX_BALLS};
use render::{BallGpu, FrameData, Globals, LineVertex, Renderer, UiFrame};
use settings::Graphics;
use ui::{Action, Hud, Sim, UiState};

struct Options {
    gfx: Graphics,
    size: (u32, u32),
    teleport: Option<usize>,
    tod: Option<f32>,
    look: Option<(f32, f32)>,
    hide_ui: bool,
    rain: bool,
    aim: bool,
    exit_after: Option<u32>,
    blackhole: Option<String>,
    bh_skip: f64,
    screenshot: Option<String>,
    skirmish: Option<combat::Setup>,
    skirmish_skip: f64,
    neuro: bool,
}

fn parse_options() -> Options {
    let gfx = settings::from_args();
    let args: Vec<String> = std::env::args().collect();
    let mut o = Options {
        gfx,
        size: (1280, 720),
        teleport: None,
        tod: None,
        look: None,
        hide_ui: false,
        rain: false,
        aim: false,
        exit_after: None,
        blackhole: None,
        bh_skip: 0.0,
        screenshot: None,
        skirmish: None,
        skirmish_skip: 0.0,
        neuro: false,
    };
    let mut i = 1;
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--size" => {
                if let Some((w, h)) = next.split_once('x') {
                    o.size = (w.parse().unwrap_or(1280), h.parse().unwrap_or(720));
                }
            }
            "--teleport" => o.teleport = next.parse().ok(),
            "--time" => o.tod = next.parse().ok(),
            "--look" => {
                if let Some((a, b)) = next.split_once(',') {
                    o.look = Some((a.parse().unwrap_or(0.0), b.parse().unwrap_or(0.0)));
                }
            }
            "--hide-ui" => o.hide_ui = true,
            "--rain" => o.rain = true,
            "--aim" => o.aim = true,
            "--frames" => o.exit_after = next.parse().ok(),
            "--screenshot" => o.screenshot = Some(next.clone()),
            "--skirmish" => {
                let mut s = o.skirmish.take().unwrap_or_default();
                if let Ok(n) = next.parse() {
                    s.bots = n;
                }
                o.skirmish = Some(s);
            }
            "--spectate" => o.skirmish.get_or_insert_with(Default::default).spectate = true,
            "--bots-level" => o.skirmish.get_or_insert_with(Default::default).level = next.clone(),
            "--seed" => o.skirmish.get_or_insert_with(Default::default).seed = next.parse().unwrap_or(1),
            "--neuro" => o.neuro = true,
            "--skirmish-skip" => o.skirmish_skip = next.parse().unwrap_or(0.0),
            "--bh-skip" => o.bh_skip = next.parse().unwrap_or(0.0),
            "--blackhole" => o.blackhole = Some(if next.is_empty() || next.starts_with("--") { bhlink::DEFAULT_SCENARIO.to_string() } else { next.clone() }),
            _ => {}
        }
        i += 1;
    }
    o
}

struct State {
    window: Arc<Window>,
    gpu: render::Gpu,
    renderer: Renderer,
    world: world::World,
    transport: transport::Transport,
    player: Player,
    balls: Vec<Ball>,
    gfx: Graphics,
    applied: Graphics,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    ui: UiState,
    sim: Sim,
    keys: HashSet<KeyCode>,
    captured: bool,
    aiming: bool,
    jump: bool,
    last: Instant,
    fps_acc: (f32, u32),
    fps: f32,
    ms: f32,
    scratch: Vec<u32>,
    rng: u64,
    trains: Vec<world::ObjInstance>,
    lines: Vec<LineVertex>,
    deflection: Option<f32>,
    frames: u32,
    exit_after: Option<u32>,
    next_frame: Instant,
    bh: Option<bhlink::BhLink>,
    combat: Option<combat::Combat>,
    combat_setup: combat::Setup,
    combat_menu: bool,
    fire: bool,
    cam: Option<(Vec3, Vec3, Vec3)>,
    last_vp: Mat4,
    screenshot: Option<String>,
    shot_counter: u32,
}

struct App {
    opts: Option<Options>,
    state: Option<State>,
}

impl State {
    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / 16_777_216.0
    }

    fn frame(&self) -> Frame {
        Frame { omega: colony::omega_for(self.sim.g_floor), coriolis: self.sim.coriolis, ext: self.bh.as_ref().map(|b| b.frame_ext()) }
    }

    fn set_capture(&mut self, on: bool) {
        self.captured = on;
        if on {
            if self.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                let _ = self.window.set_cursor_grab(CursorGrabMode::Confined);
            }
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        }
        self.window.set_cursor_visible(!on);
    }

    fn spawn_ball(&mut self, pos: Vec3, vel: Vec3, ty: usize) {
        if self.balls.len() >= MAX_BALLS {
            // Drop the oldest one.
            let (i, _) = self.balls.iter().enumerate().max_by(|a, b| a.1.age.partial_cmp(&b.1.age).unwrap()).unwrap();
            self.balls.swap_remove(i);
        }
        let rot = Quat::from_rotation_y(self.rand() * std::f32::consts::TAU) * Quat::from_rotation_x(self.rand() * std::f32::consts::TAU);
        self.balls.push(Ball { pos, vel, rot, ang: Vec3::ZERO, ty, age: 0.0, contact: 1.0 });
    }

    /// Starts a skirmish in the rendered world; `origin` = arena corner (u, v).
    fn start_combat(&mut self, origin: Option<(f64, f64)>) {
        let origin = origin.unwrap_or_else(|| {
            let s = world_to_surf(self.player.pos);
            (s.x as f64 - 0.5 * combat::ARENA_SIZE, s.y as f64 - 0.5 * combat::ARENA_SIZE)
        });
        let c = combat::Combat::new(&self.world, origin, self.combat_setup.clone());
        self.player.fly = false;
        self.player.ride = None;
        self.combat = Some(c);
        self.ui.show_help = false;
    }

    fn stop_combat(&mut self) {
        if let Some((eye, fwd, _)) = self.cam.take() {
            self.player.pos = eye;
            let (e, up, n) = self.player.basis();
            let _ = (e, n);
            self.player.pitch = fwd.dot(up).clamp(-1.0, 1.0).asin();
        }
        self.combat = None;
    }

    fn combat_controls(&self) -> combat::Controls {
        if !self.captured {
            // dev/screenshot runs: ONEILL_INPUT="fire,ads,fwd" holds those controls
            if let Ok(v) = std::env::var("ONEILL_INPUT") {
                return combat::Controls {
                    fire: v.contains("fire"),
                    ads: v.contains("ads"),
                    fwd: if v.contains("fwd") { 1.0 } else { 0.0 },
                    run: v.contains("run"),
                    ..Default::default()
                };
            }
            return combat::Controls::default();
        }
        let k = |c: KeyCode| self.keys.contains(&c) as i32 as f32;
        let jet = self.combat.as_ref().and_then(|c| c.sim.ecs.bodies.get(c.viewed())).is_some_and(|b| b.mode == gameplay::MoveMode::Jetpack);
        combat::Controls {
            fwd: k(KeyCode::KeyW) - k(KeyCode::KeyS),
            right: k(KeyCode::KeyD) - k(KeyCode::KeyA),
            up: if jet { k(KeyCode::Space) - k(KeyCode::KeyC).max(k(KeyCode::ControlLeft)) } else { 0.0 },
            roll: k(KeyCode::KeyX) - k(KeyCode::KeyZ),
            run: self.keys.contains(&KeyCode::ShiftLeft),
            crouch: !jet && (self.keys.contains(&KeyCode::ControlLeft) || self.keys.contains(&KeyCode::KeyC)),
            jump: !jet && self.keys.contains(&KeyCode::Space),
            fire: self.fire,
            ads: self.aiming,
            boots: self.keys.contains(&KeyCode::KeyM),
        }
    }

    fn enable_bh(&mut self, name: &str) {
        match bhlink::BhLink::new(&self.gpu.device, &self.gpu.queue, name) {
            Ok(b) => {
                self.renderer.set_env(&self.gpu.device, b.cube_view());
                self.bh = Some(b);
            }
            Err(e) => log::error!("black-hole scenario '{name}': {e:#}"),
        }
    }

    fn throw_origin(&self) -> (Vec3, Vec3) {
        let f = self.player.forward();
        let r = BALL_TYPES[self.ui.ball_type].radius;
        (self.player.pos + f * (0.6 + r) - self.player.up() * 0.25, f * self.ui.throw_speed + self.player.vel)
    }

    fn throw(&mut self) {
        let (p, v) = self.throw_origin();
        self.spawn_ball(p, v, self.ui.ball_type);
    }

    fn rain(&mut self) {
        let sf = world_to_surf(self.player.pos);
        let th0 = sf.x / R;
        for _ in 0..24 {
            let th = th0 + (self.rand() - 0.5) * 0.25;
            let r = 20.0 + self.rand() * 60.0;
            let z = (sf.y + (self.rand() - 0.5) * 200.0).clamp(50.0, colony::L - 50.0);
            let ty = [0, 1, 2, 4, 4, 6, 3, 0][(self.rand() * 7.99) as usize];
            let p = Vec3::new(r * th.cos(), r * th.sin(), z);
            self.spawn_ball(p, Vec3::ZERO, ty);
        }
    }

    fn kick(&mut self) {
        let pp = self.player.pos;
        let f = self.player.forward();
        let up = self.player.up();
        if let Some(b) = self.balls.iter_mut().filter(|b| b.pos.distance(pp) < 4.0).min_by(|a, b| a.pos.distance(pp).partial_cmp(&b.pos.distance(pp)).unwrap())
        {
            let flat = (f - up * f.dot(up)).normalize_or_zero();
            let m = BALL_TYPES[b.ty].mass;
            b.vel += (flat + up * 0.45) * 14.0 * (6.0 / m.max(6.0)).sqrt().min(1.0) * if m > 100.0 { 0.2 } else { 1.0 };
        }
    }

    fn toggle_ride(&mut self) {
        if self.player.ride.take().is_some() {
            let v = self.transport.velocity(&self.world, 0);
            let _ = v;
            return;
        }
        let t = self.transport.nearest(&self.world, self.player.pos);
        self.player.ride = Some(t);
        self.player.fly = false;
    }

    fn handle_action(&mut self, a: Action) {
        match a {
            Action::Teleport(i) => {
                let t = &self.world.teleports[i];
                let (p, fly) = (t.pos, t.fly);
                self.player.teleport(p, fly);
                if i == self.world.teleports.len() - 1 {
                    self.player.yaw = 0.0;
                    self.player.pitch = 0.25;
                }
            }
            Action::Rain => self.rain(),
            Action::Clear => self.balls.clear(),
            Action::Spawn => self.throw(),
            Action::Ride => self.toggle_ride(),
        }
    }

    fn update(&mut self, dt: f32) {
        let raw_dt = dt;
        let dt = dt.min(1.0 / 20.0);
        let sdt = dt * self.sim.time_scale;
        if !self.sim.paused {
            self.sim.tod = (self.sim.tod + sdt / (self.sim.day_minutes * 60.0)).rem_euclid(1.0);
        }
        self.sim.time += sdt;
        if let Some(b) = self.bh.as_mut() {
            b.update(raw_dt as f64, self.player.pos);
        }
        let frame = self.frame();
        self.sim.spin = (self.sim.spin + frame.omega * sdt).rem_euclid(std::f32::consts::TAU);
        self.transport.update(&self.world, sdt);
        physics::step_balls(&self.world, &mut self.balls, sdt, frame, &mut self.scratch);
        self.balls.retain(|b| b.pos.is_finite());

        if self.combat.is_some() {
            let controls = self.combat_controls();
            let Some(c) = self.combat.as_mut() else { unreachable!() };
            c.update(raw_dt.min(0.25) as f64, &controls);
            let (eye, fwd, up) = c.camera();
            let b = c.sim.ecs.bodies.get(c.viewed());
            self.player.pos = eye;
            self.player.vel = b.map_or(Vec3::ZERO, |b| b.vel.as_vec3());
            self.player.grounded = b.is_some_and(|b| b.grounded);
            self.cam = Some((eye, fwd, up));
            self.lines.clear();
            self.deflection = None;
            self.transport.instances(&self.world, &mut self.trains);
            return;
        }
        self.cam = None;
        let k = |c: KeyCode| self.keys.contains(&c) as i32 as f32;
        let input = if self.captured {
            MoveInput {
                fwd: k(KeyCode::KeyW) - k(KeyCode::KeyS),
                right: k(KeyCode::KeyD) - k(KeyCode::KeyA),
                up: k(KeyCode::Space) - k(KeyCode::KeyC).max(k(KeyCode::ControlLeft)),
                run: self.keys.contains(&KeyCode::ShiftLeft),
                jump: self.jump,
            }
        } else {
            MoveInput::default()
        };
        self.jump = false;
        if let Some(t) = self.player.ride {
            if input.jump {
                self.player.ride = None;
                self.player.vel = self.transport.velocity(&self.world, t) + self.player.up() * 4.0;
            } else {
                self.player.pos = self.transport.roof(&self.world, t);
                self.player.vel = self.transport.velocity(&self.world, t);
                self.player.grounded = true;
            }
        }
        if self.player.ride.is_none() {
            self.player.step(&self.world, input, sdt, frame, &mut self.scratch);
        }

        // Trajectory preview.
        self.lines.clear();
        self.deflection = None;
        if self.aiming || self.ui.always_trajectory {
            let (p, v) = self.throw_origin();
            let (real, naive) = physics::predict(&self.world, p, v, self.ui.ball_type, frame, &mut self.scratch);
            let cam = self.player.pos;
            ribbon(&mut self.lines, &real, cam, [0.25, 1.0, 0.45], 1.6);
            ribbon(&mut self.lines, &naive, cam, [1.0, 0.25, 0.15], 1.0);
            if let (Some(a), Some(b)) = (real.last(), naive.last()) {
                self.deflection = Some(a.distance(*b));
                marker(&mut self.lines, *a, cam, [0.25, 1.0, 0.45]);
                marker(&mut self.lines, *b, cam, [1.0, 0.25, 0.15]);
            }
        }
        self.transport.instances(&self.world, &mut self.trains);
    }

    fn globals(&self) -> (Globals, Mat4, f32) {
        let s = &self.sim;
        let tod = s.tod;
        let day = smooth(0.22, 0.30, tod) * (1.0 - smooth(0.70, 0.78, tod));
        let beta = ((tod - 0.25) / 0.5).clamp(0.0, 1.0) * 110f32.to_radians() - 55f32.to_radians();
        let warm = 1.0 - (((tod - 0.25) / 0.5).clamp(0.0, 1.0) * std::f32::consts::PI).sin();
        let warm = warm * warm;
        let sun = Vec3::new(3.2, 3.0, 2.7).lerp(Vec3::new(3.1, 1.7, 0.8), warm) * 1.0;
        let sky = Vec3::new(0.30, 0.45, 0.75).lerp(Vec3::new(0.45, 0.38, 0.4), warm * 0.6) * day * 0.9 + Vec3::new(0.004, 0.006, 0.012);
        let bounce = Vec3::new(0.10, 0.12, 0.07) * day + Vec3::new(0.010, 0.010, 0.016);
        let mut exposure = (3.2 + (1.0 - 3.2) * day) * 2f32.powf(self.gfx.exposure_bias);
        let lights = colony::light_dirs(beta, day);
        let omega = colony::omega_for(s.g_floor);

        // ONEILL_CAM=x,y,z,tx,ty,tz overrides the view (cinematic / screenshots).
        static CAM: std::sync::OnceLock<Option<(Vec3, Vec3)>> = std::sync::OnceLock::new();
        let ov = CAM.get_or_init(|| {
            std::env::var("ONEILL_CAM").ok().and_then(|s| {
                let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (v.len() == 6).then(|| (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5])))
            })
        });
        let (eye, fwd, up) = match ov {
            Some((e, t)) => (*e, (*t - *e).normalize(), if (*t - *e).normalize().y.abs() > 0.95 { Vec3::X } else { Vec3::Y }),
            None => self.cam.unwrap_or((self.player.pos, self.player.forward(), self.player.up())),
        };
        // In open space the eye adapts to the much darker surroundings, so the
        // sunlit interior seen through the windows glows like in photographs.
        if eye.truncate().length() > colony::R + 1.0 || eye.z < 0.0 || eye.z > colony::L {
            exposure = exposure.max(1.0) * 2.2;
        }
        let size = self.window.inner_size();
        let aspect = size.width.max(1) as f32 / size.height.max(1) as f32;
        let view = Mat4::look_to_rh(eye, fwd, up);
        let fov = self.combat.as_ref().map_or(70.0, |c| c.fov_deg());
        let proj = Mat4::perspective_infinite_reverse_rh(fov.to_radians(), aspect, 0.05);
        let vp = proj * view;
        let mut g: Globals = bytemuck::Zeroable::zeroed();
        g.view_proj = vp.to_cols_array_2d();
        g.inv_view_proj = vp.inverse().to_cols_array_2d();
        g.cam_pos = [eye.x, eye.y, eye.z, s.time];
        for (i, l) in lights.iter().enumerate() {
            g.light_dir[i] = l.to_array();
        }
        g.sun_color = [sun.x, sun.y, sun.z, exposure];
        g.sky_color = [sky.x, sky.y, sky.z, 0.000042 * self.gfx.fog];
        g.bounce_color = [bounce.x, bounce.y, bounce.z, 1.0 - day];
        g.colony = [R, colony::L, omega, s.spin];
        g.wind = [s.wind_dir.cos(), s.wind_dir.sin(), s.wind, 6.0];
        g.params2[2] = beta;
        let e = Vec3::new(0.9, 0.0, -0.44).normalize();
        g.space = [e.x, e.y, e.z, if s.paused && false { 0.0 } else { 1.0 / 60.0 }];
        // Imprints into the grass field: player's feet + nearest balls.
        let cam_u = world_to_surf(eye).x;
        let mut n = 0;
        if self.player.grounded && !self.player.fly && self.player.ride.is_none() {
            let sf = world_to_surf(eye);
            let (ee, _, nn) = self.player.basis();
            g.balls[0] = [sf.x, sf.y, 0.0, 0.32];
            g.ball_vel[0] = [self.player.vel.dot(ee), self.player.vel.dot(nn), 0.9, 0.0];
            n = 1;
        }
        let mut idx: Vec<(f32, usize)> = self.balls.iter().enumerate().map(|(i, b)| (b.pos.distance(eye), i)).filter(|(d, _)| *d < 80.0).collect();
        idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(_, i) in idx.iter() {
            if n >= 16 {
                break;
            }
            let b = &self.balls[i];
            let t = BALL_TYPES[b.ty];
            if b.contact > 0.5 {
                continue;
            }
            let sf = world_to_surf(b.pos);
            let (ee, _, nn) = colony::basis_at(b.pos).unwrap_or((Vec3::Y, Vec3::X, Vec3::Z));
            let strength = (0.35 + 0.65 * (t.mass / 6.0).min(1.0)).min(1.0);
            g.balls[n] = [colony::unwrap_u(sf.x, cam_u), sf.y, b.contact, t.radius.max(0.15)];
            g.ball_vel[n] = [b.vel.dot(ee), b.vel.dot(nn), strength, 0.0];
            n += 1;
        }
        g.field_params = [0.0, 0.0, n as f32, if s.debug_field { 1.0 } else { 0.0 }];
        // Glowing orbs light their surroundings.
        let mut l = 0;
        for &(_, i) in idx.iter() {
            let b = &self.balls[i];
            if BALL_TYPES[b.ty].emissive > 0.0 && l < 4 {
                g.lights[l] = [b.pos.x, b.pos.y, b.pos.z, 6.0];
                l += 1;
            }
        }
        let _ = CIRC;
        (g, vp, exposure)
    }

    fn redraw(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        self.fps_acc.0 += dt;
        self.fps_acc.1 += 1;
        if self.fps_acc.0 > 0.5 {
            self.fps = self.fps_acc.1 as f32 / self.fps_acc.0;
            self.ms = 1000.0 * self.fps_acc.0 / self.fps_acc.1 as f32;
            self.fps_acc = (0.0, 0);
        }
        self.update(dt);

        // UI
        let raw = self.egui_state.take_egui_input(&self.window);
        let sf = world_to_surf(self.player.pos);
        let (strip, _) = colony::strip_of(sf.x / R);
        let land = colony::is_land(sf.x / R);
        let hud = Hud {
            fps: self.fps,
            ms: self.ms,
            g: self.player.g_force(&self.frame()),
            altitude: sf.z - physics::EYE,
            axis_dist: self.player.pos.truncate().length(),
            speed: self.player.vel.length(),
            valley: if land { colony::STRIP_NAMES[strip] } else { "Окно" },
            blades: self.renderer.stats.blades_near + self.renderer.stats.blades_far,
            objects: self.renderer.stats.objects,
            balls: self.balls.len(),
            fly: self.player.fly,
            riding: self.player.ride.is_some(),
            captured: self.captured,
            adapter: self.gpu.adapter.clone(),
            deflection: self.deflection,
            n_buildings: self.world.n_buildings,
            n_trees: self.world.n_trees,
        };
        let names: Vec<&str> = self.world.teleports.iter().map(|t| t.name).collect();
        let mut actions = Vec::new();
        let mut bh = self.bh.take();
        let (hidden, pos, cam_g) = (self.ui.hidden, self.player.pos, hud.g);
        let combat = self.combat.take();
        let (vp, captured) = (self.last_vp, self.captured);
        let mut menu_action = None;
        let (setup, show_menu) = (&mut self.combat_setup, &mut self.combat_menu);
        let out = self.egui_ctx.run(raw, |ctx| {
            if !(captured && combat.is_some()) {
                actions = ui::draw(ctx, &mut self.ui, &mut self.sim, &mut self.gfx, &hud, &names);
            }
            if let Some(c) = &combat {
                if !hidden {
                    c.hud(ctx, vp);
                }
            }
            if !captured && !hidden {
                menu_action = combat_menu(ctx, setup, show_menu, combat.as_ref());
            }
            if let Some(b) = bh.as_mut() {
                if !hidden {
                    b.ui(ctx, pos, cam_g);
                }
            }
        });
        self.bh = bh;
        self.combat = combat;
        match menu_action {
            Some(true) => self.start_combat(None),
            Some(false) => self.stop_combat(),
            None => {}
        }
        self.egui_state.handle_platform_output(&self.window, out.platform_output);
        for a in actions {
            self.handle_action(a);
        }
        if self.gfx != self.applied {
            if self.gfx.vsync != self.applied.vsync {
                self.gpu.set_vsync(self.gfx.vsync);
            }
            self.renderer.apply_settings(&self.gpu, &self.gfx);
            self.applied = self.gfx.clone();
        }
        let jobs = self.egui_ctx.tessellate(out.shapes, out.pixels_per_point);

        let (mut g, vp, mut exposure) = self.globals();
        if let Some(c) = &self.combat {
            let free: Vec<usize> = (0..4).filter(|&k| g.lights[k][3] == 0.0).collect();
            let mut free = free.into_iter();
            for l in c.lights(4) {
                if let Some(k) = free.next() {
                    g.lights[k] = l;
                }
            }
        }
        self.last_vp = vp;
        g.space[3] = dt * self.sim.time_scale;
        g.bh_misc = [self.sim.clouds, 1.0, 0.0, 0.0];
        let mut no_cull = false;
        if let Some(b) = self.bh.as_mut() {
            b.poll_readback(&self.gpu.device);
            b.render_env(&self.gpu.device, &self.gpu.queue);
            exposure = bh_globals(b, &mut g, dt, self.gfx.exposure_bias, self.gfx.fog);
            if b.split {
                self.renderer.upload_nodes(&self.gpu.queue, &b.node_xf());
                no_cull = true;
            }
            let sun = Vec3::new(g.sun_color[0], g.sun_color[1], g.sun_color[2]);
            b.particles(self.player.pos, &mut self.lines, 6000, sun);
        }
        let balls: Vec<BallGpu> = self
            .balls
            .iter()
            .map(|b| {
                let t = BALL_TYPES[b.ty];
                BallGpu { pos: b.pos.to_array(), radius: t.radius, rot: b.rot.to_array(), style: b.ty as u32, emissive: t.emissive, ext: [0.0; 2] }
            })
            .collect();
        let mut balls = balls;
        let (cam_pos, cam_fwd) = self.cam.map_or((self.player.pos, self.player.forward()), |c| (c.0, c.1));
        if let Some(c) = &self.combat {
            c.draw(cam_pos, &mut balls, &mut self.lines);
        }
        if let (Some(n), Some(path)) = (self.exit_after, &self.screenshot) {
            if self.frames + 1 == n {
                self.renderer.screenshot = Some(path.into());
            }
        }
        let (props, particles) = self.combat.as_ref().map_or((&[][..], &[][..]), |c| c.frame());
        let fd = FrameData { props, particles, cam: cam_pos, fwd: cam_fwd, exposure, balls: &balls, trains: &self.trains, lines: &self.lines, no_cull };
        let uf = UiFrame { jobs: &jobs, textures: &out.textures_delta, ppp: out.pixels_per_point };
        if let Err(e) = self.renderer.render(&self.gpu, &mut g, &self.gfx, &fd, uf) {
            log::error!("render: {e:#}");
        }
        self.frames += 1;
    }
}

/// Black-hole mode lighting: the disk shines through the windows, the sky
/// is the traced environment, the air thins out while the hull vents.
fn bh_globals(b: &mut bhlink::BhLink, g: &mut Globals, dt: f32, bias: f32, fog: f32) -> f32 {
    let (dirs, sun) = b.light_dirs();
    let mean = b.env_light().map(|e| e.2).unwrap_or(sun * 0.03);
    g.light_dir = dirs;
    let pressure = b.pressure_frac();
    let lit = dirs.iter().map(|d| d[3]).fold(0.0f32, f32::max).min(1.0);
    let lum = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    g.sun_color[0] = sun.x;
    g.sun_color[1] = sun.y;
    g.sun_color[2] = sun.z;
    // the air scatters what comes in through the windows (≈ half the view)
    let sky = (mean * 0.08 + sun * (lit * 0.012)) * pressure + Vec3::new(0.0004, 0.0006, 0.0012);
    g.sky_color = [sky.x, sky.y, sky.z, 0.000042 * fog];
    let bounce = sun * (lit * 0.03) + mean * 0.15 + Vec3::new(0.010, 0.010, 0.016);
    g.bounce_color = [bounce.x, bounce.y, bounce.z, 1.0 - lit];
    // eye adaptation (~2 s): the classic colony day (sun 3) maps to exposure 1
    let target = ((lum(sun) * lit + lum(mean) * 1.5) / 3.0).max(0.004);
    let k = 1.0 - (-dt / 2.0).exp();
    b.adapt = if b.adapt <= 0.0 { target } else { b.adapt * (target / b.adapt).powf(k) };
    let exposure = (1.0 / b.adapt).clamp(0.002, 250.0) * 2f32.powf(bias);
    g.sun_color[3] = exposure;
    let rot = Mat4::from_quat(b.rot().as_quat());
    g.bh_rot = rot.to_cols_array_2d();
    g.bh = [1.0, b.env_gain, pressure, if b.split { 1.0 } else { 0.0 }];
    exposure
}

/// Skirmish window; returns Some(true) = start here, Some(false) = stop.
fn combat_menu(ctx: &egui::Context, s: &mut combat::Setup, open: &mut bool, active: Option<&combat::Combat>) -> Option<bool> {
    let mut act = None;
    egui::Area::new(egui::Id::new("combat_btn")).anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -12.0)).show(ctx, |ui| {
        let label = if active.is_some() { "⚔ Схватка идёт" } else { "⚔ Схватка" };
        if ui.button(label).clicked() {
            *open = !*open;
        }
    });
    if !*open {
        return None;
    }
    egui::Window::new("Схватка").anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -44.0)).resizable(false).collapsible(false).show(ctx, |ui| {
        match active {
            Some(c) => {
                ui.label(format!("{} · {}", c.sim.arena.st.name, c.status()));
                ui.label("ЛКМ огонь · ПКМ прицел · 1-3 оружие · R перезарядка\nQ устройство · E режим · N нейроинтерфейс · J/F ранец (Z/X крен)\nM магнитные ботинки · Tab счёт · [ ] наблюдение · K выход");
                if ui.button("Завершить").clicked() {
                    act = Some(false);
                }
            }
            None => {
                ui.add(egui::Slider::new(&mut s.bots, 1..=15).text("ботов"));
                egui::ComboBox::from_label("уровень").selected_text(s.level.clone()).show_ui(ui, |ui| {
                    for l in ["mixed", "novice", "regular", "veteran"] {
                        ui.selectable_value(&mut s.level, l.to_string(), l);
                    }
                });
                ui.checkbox(&mut s.spectate, "только наблюдать (боты против ботов)");
                ui.add(egui::DragValue::new(&mut s.seed).prefix("seed "));
                ui.label("Арена 2×2 км вокруг вас, из настоящих зданий мира.");
                if ui.button("Начать здесь (K)").clicked() {
                    act = Some(true);
                }
            }
        }
    });
    act
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Camera-facing glowing ribbon through `pts`.
fn ribbon(out: &mut Vec<LineVertex>, pts: &[Vec3], cam: Vec3, col: [f32; 3], k: f32) {
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let d = (b - a).normalize_or_zero();
        let w0 = 0.025 + 0.0025 * a.distance(cam);
        let w1 = 0.025 + 0.0025 * b.distance(cam);
        let sa = d.cross(cam - a).normalize_or_zero() * w0;
        let sb = d.cross(cam - b).normalize_or_zero() * w1;
        let c = [col[0] * k, col[1] * k, col[2] * k, 1.0];
        let v = |p: Vec3| LineVertex { pos: p.to_array(), color: c };
        out.extend_from_slice(&[v(a - sa), v(a + sa), v(b + sb), v(a - sa), v(b + sb), v(b - sb)]);
    }
}

fn marker(out: &mut Vec<LineVertex>, p: Vec3, cam: Vec3, col: [f32; 3]) {
    let up = colony::basis_at(p).map(|b| b.1).unwrap_or(Vec3::Z);
    let pts: Vec<Vec3> = (0..=24)
        .map(|i| {
            let a = i as f32 / 24.0 * std::f32::consts::TAU;
            let e = up.any_orthonormal_vector();
            let n = up.cross(e);
            p + (e * a.cos() + n * a.sin()) * 0.8 + up * 0.05
        })
        .collect();
    ribbon(out, &pts, cam, col, 2.5);
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let opts = self.opts.take().unwrap();
        let window = Arc::new(
            el.create_window(
                Window::default_attributes().with_title("Колония О'Нила — Остров III").with_inner_size(winit::dpi::PhysicalSize::new(opts.size.0, opts.size.1)),
            )
            .expect("window"),
        );
        println!("Генерация мира…");
        let world = world::World::generate();
        let gpu = pollster::block_on(render::Gpu::new(window.clone(), opts.gfx.vsync)).expect("GPU init failed");
        println!("GPU: {}", gpu.adapter);
        let renderer = Renderer::new(&gpu, &world, &opts.gfx);
        let transport = transport::Transport::new(&world);
        let start = opts.teleport.unwrap_or(0).min(world.teleports.len() - 1);
        let mut player = Player::new(world.teleports[start].pos);
        player.fly = world.teleports[start].fly;
        if let Some((y, p)) = opts.look {
            player.yaw = y.to_radians();
            player.pitch = p.to_radians();
        }
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        let mut sim = Sim::default();
        if let Some(t) = opts.tod {
            sim.tod = t;
        }
        let ui = UiState { hidden: opts.hide_ui, always_trajectory: opts.aim, ..Default::default() };
        let mut st = State {
            window: window.clone(),
            gpu,
            renderer,
            world,
            transport,
            player,
            balls: Vec::new(),
            applied: opts.gfx.clone(),
            gfx: opts.gfx,
            egui_ctx,
            egui_state,
            ui,
            sim,
            keys: HashSet::new(),
            captured: false,
            aiming: false,
            jump: false,
            last: Instant::now(),
            fps_acc: (0.0, 0),
            fps: 0.0,
            ms: 0.0,
            scratch: Vec::new(),
            rng: 0x1234_5678_9abc_def1,
            trains: Vec::new(),
            lines: Vec::new(),
            deflection: None,
            frames: 0,
            exit_after: opts.exit_after,
            next_frame: Instant::now(),
            bh: None,
            combat: None,
            combat_setup: opts.skirmish.clone().unwrap_or_default(),
            combat_menu: false,
            fire: false,
            cam: None,
            last_vp: Mat4::IDENTITY,
            screenshot: opts.screenshot.clone(),
            shot_counter: 0,
        };
        if let Some(name) = &opts.blackhole {
            st.enable_bh(name);
            if let Some(b) = st.bh.as_mut() {
                b.skip(opts.bh_skip);
                // ONEILL_BH_LIT=k: continue until the hole shines into valley k
                if let Some(k) = std::env::var("ONEILL_BH_LIT").ok().and_then(|v| v.parse::<usize>().ok()) {
                    for _ in 0..2000 {
                        if b.light_dirs().0[k.min(2)][3] > 0.95 {
                            break;
                        }
                        b.skip(0.1);
                    }
                }
            }
        }
        if opts.rain {
            st.rain();
        }
        if opts.skirmish.is_some() {
            st.start_combat(Some(combat::default_origin()));
            if let Some(c) = st.combat.as_mut() {
                if opts.neuro {
                    c.press(gameplay::buttons::NEURO);
                }
                let skip = (opts.skirmish_skip * gameplay::TICK_HZ as f64) as u32;
                for _ in 0..skip {
                    c.update(gameplay::DT, &combat::Controls::default());
                }
            }
        }
        self.state = Some(st);
        window.request_redraw();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else { return };
        let consumed = if st.captured { false } else { st.egui_state.on_window_event(&st.window, &event).consumed };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                st.gpu.resize(s.width, s.height);
                st.renderer.resize(&st.gpu, &st.gfx);
            }
            WindowEvent::Focused(false) => {
                st.keys.clear();
                if st.captured {
                    st.set_capture(false);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                let pressed = event.state == ElementState::Pressed;
                if pressed {
                    st.keys.insert(code);
                } else {
                    st.keys.remove(&code);
                }
                if code == KeyCode::Tab {
                    if let Some(c) = st.combat.as_mut() {
                        c.scoreboard = pressed;
                    }
                }
                if !pressed || event.repeat {
                    return;
                }
                if st.combat.is_some() && code == KeyCode::Tab {
                    return;
                }
                if code == KeyCode::F12 {
                    st.shot_counter += 1;
                    st.renderer.screenshot = Some(format!("screenshot_{:03}.png", st.shot_counter).into());
                    return;
                }
                if code == KeyCode::Tab {
                    let c = !st.captured;
                    st.set_capture(c);
                    return;
                }
                if code == KeyCode::Escape && st.captured {
                    st.set_capture(false);
                    return;
                }
                if consumed && !st.captured {
                    return;
                }
                if code == KeyCode::KeyK {
                    if st.combat.is_some() {
                        st.stop_combat();
                    } else {
                        st.start_combat(None);
                    }
                    return;
                }
                if let Some(c) = st.combat.as_mut() {
                    use gameplay::buttons as bt;
                    match code {
                        KeyCode::Digit1 => c.select_weapon(1),
                        KeyCode::Digit2 => c.select_weapon(2),
                        KeyCode::Digit3 => c.select_weapon(3),
                        KeyCode::KeyR => c.press(bt::RELOAD),
                        KeyCode::KeyQ => c.press(bt::DEVICE),
                        KeyCode::KeyE => c.press(bt::DEVICE_NEXT),
                        KeyCode::KeyN => c.press(bt::NEURO),
                        KeyCode::KeyJ | KeyCode::KeyF => c.press(bt::JETPACK),
                        KeyCode::BracketLeft => c.cycle_follow(-1),
                        KeyCode::BracketRight => c.cycle_follow(1),
                        KeyCode::KeyP => st.sim.paused = !st.sim.paused,
                        KeyCode::F1 => st.ui.hidden = !st.ui.hidden,
                        _ => {}
                    }
                    return;
                }
                match code {
                    KeyCode::Space => st.jump = true,
                    KeyCode::KeyF => {
                        st.player.fly = !st.player.fly;
                        st.player.ride = None;
                    }
                    KeyCode::KeyG => st.kick(),
                    KeyCode::KeyR => st.rain(),
                    KeyCode::KeyT => st.toggle_ride(),
                    KeyCode::KeyP => st.sim.paused = !st.sim.paused,
                    KeyCode::KeyV => st.sim.debug_field = !st.sim.debug_field,
                    KeyCode::KeyB => {
                        if st.bh.is_some() {
                            st.bh = None;
                        } else {
                            st.enable_bh(bhlink::DEFAULT_SCENARIO);
                        }
                    }
                    KeyCode::F1 => st.ui.hidden = !st.ui.hidden,
                    KeyCode::Digit1
                    | KeyCode::Digit2
                    | KeyCode::Digit3
                    | KeyCode::Digit4
                    | KeyCode::Digit5
                    | KeyCode::Digit6
                    | KeyCode::Digit7
                    | KeyCode::Digit8 => {
                        let i = match code {
                            KeyCode::Digit1 => 0,
                            KeyCode::Digit2 => 1,
                            KeyCode::Digit3 => 2,
                            KeyCode::Digit4 => 3,
                            KeyCode::Digit5 => 4,
                            KeyCode::Digit6 => 5,
                            KeyCode::Digit7 => 6,
                            _ => 7,
                        };
                        st.ui.ball_type = i;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if consumed {
                    return;
                }
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left if st.combat.is_some() => {
                        if st.captured || !pressed {
                            st.fire = pressed && st.captured;
                        } else if !st.egui_ctx.wants_pointer_input() {
                            st.set_capture(true);
                        }
                    }
                    MouseButton::Left if pressed => {
                        if st.captured {
                            st.throw();
                        } else if !st.egui_ctx.wants_pointer_input() {
                            st.set_capture(true);
                        }
                    }
                    MouseButton::Right => st.aiming = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } if st.captured || !consumed => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                if st.player.fly {
                    st.player.fly_speed = (st.player.fly_speed * 1.2f32.powf(d)).clamp(2.0, 6000.0);
                } else {
                    st.ui.throw_speed = (st.ui.throw_speed + d * 2.0).clamp(2.0, 60.0);
                }
            }
            WindowEvent::RedrawRequested => {
                st.redraw();
                if let Some(n) = st.exit_after {
                    if st.frames >= n {
                        el.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
        let Some(st) = self.state.as_mut() else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if st.captured && st.combat.is_some() {
                if let Some(c) = st.combat.as_mut() {
                    let sens = if st.aiming { 0.0011 } else { 0.0022 };
                    c.mouse(delta.0, delta.1, sens);
                }
            } else if st.captured {
                st.player.yaw -= delta.0 as f32 * 0.0022;
                st.player.pitch = (st.player.pitch - delta.1 as f32 * 0.0022).clamp(-1.55, 1.55);
            }
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let Some(st) = self.state.as_mut() else { return };
        if st.gfx.fps_cap > 0 {
            let period = Duration::from_secs_f64(1.0 / st.gfx.fps_cap as f64);
            let now = Instant::now();
            if now < st.next_frame {
                el.set_control_flow(ControlFlow::WaitUntil(st.next_frame));
                return;
            }
            st.next_frame = (st.next_frame + period).max(now);
        }
        el.set_control_flow(ControlFlow::Poll);
        st.window.request_redraw();
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,oneill_colony=info")).init();
    let opts = parse_options();
    let el = EventLoop::new().expect("event loop");
    let mut app = App { opts: Some(opts), state: None };
    if let Err(e) = el.run_app(&mut app) {
        eprintln!("{e}");
    }
}
