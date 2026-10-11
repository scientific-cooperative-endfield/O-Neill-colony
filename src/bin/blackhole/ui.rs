//! egui HUD and control panels of the black-hole mode.

use oneill_colony::bh::physics::{self, PRESETS};
use oneill_colony::bh::render::{Settings, PASS_NAMES, QUALITY, QUALITY_NAMES};
use oneill_colony::bh::sim::{CamKin, Sim, TimeMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CamMode {
    Orbit,
    Free,
    Cinematic,
    Station,
}

pub struct UiState {
    pub show_panels: bool,
    pub cam_mode: CamMode,
    pub restart: bool,
    pub fly_speed: f64,
    pub orbit: (f64, f64, f64, f64), // r, incl, azim, fov
}

impl Default for UiState {
    fn default() -> Self {
        UiState { show_panels: true, cam_mode: CamMode::Orbit, restart: false, fly_speed: 2.0, orbit: (30.0, 15.0, 0.0, 60.0) }
    }
}

pub struct HudInfo<'a> {
    pub sim: &'a Sim,
    pub timings: [f32; 6],
    pub has_timestamps: bool,
    pub mean_steps: f32,
    pub trace_size: (u32, u32),
    pub out_size: (u32, u32),
    pub exposure: f32,
    pub fps: f32,
    pub adapter: &'a str,
}

/// Time in M plus physical units.
pub fn fmt_time(t_m: f64, m_s: f64) -> String {
    let s = t_m * m_s;
    let a = s.abs();
    let phys = if a >= 86400.0 * 365.25 {
        format!("{:.3} лет", s / (86400.0 * 365.25))
    } else if a >= 86400.0 {
        format!("{:.3} сут", s / 86400.0)
    } else if a >= 3600.0 {
        format!("{:.3} ч", s / 3600.0)
    } else if a >= 1.0 {
        format!("{s:.3} с")
    } else if a >= 1e-3 {
        format!("{:.3} мс", s * 1e3)
    } else {
        format!("{:.3} мкс", s * 1e6)
    };
    format!("{t_m:.2} M ({phys})")
}

pub fn fmt_secs(s: f64) -> String {
    let a = s.abs();
    if a >= 1.0 {
        format!("{s:+.3} с")
    } else if a >= 1e-3 {
        format!("{:+.3} мс", s * 1e3)
    } else if a >= 1e-6 {
        format!("{:+.3} мкс", s * 1e6)
    } else if a >= 1e-9 {
        format!("{:+.3} нс", s * 1e9)
    } else {
        format!("{:+.3} пс", s * 1e12)
    }
}

pub fn fmt_len(r_m: f64, m_metres: f64) -> String {
    let km = r_m * m_metres / 1000.0;
    if km.abs() >= 1e6 {
        format!("{r_m:.3} M ({:.3e} км)", km)
    } else {
        format!("{r_m:.3} M ({km:.1} км)")
    }
}

pub fn hud_text(h: &HudInfo) -> String {
    let s = h.sim;
    let ms = s.m_seconds();
    let mm = s.m_metres();
    let mut o = String::new();
    o += &format!("{} · M = {:.3e} м · r_s = {}\n", s.mass_label(), mm, fmt_len(2.0, mm));
    o += &format!("Время: {} · ×{:.3}{}\n", s.mode.label(), s.speed, if s.paused { " · ПАУЗА" } else { "" });
    let cr = s.cam.pos.length();
    let na = |v: Option<f64>| v.map(|v| format!("{v:.6}")).unwrap_or_else(|| "— (за горизонтом)".into());
    o += &format!("t удалённого наблюдателя = {}\n", fmt_time(s.t_remote, ms));
    o += &format!(
        "Камера ({}): r = {} · τ = {} · dτ/dt = {} · во внешней Вселенной прошло {}\n",
        s.cam_kin.label(),
        fmt_len(cr, mm),
        fmt_time(s.cam.tau, ms),
        na(s.cam.dtau_dt),
        fmt_time(s.elapsed_outside(), ms)
    );
    let st = &s.station;
    let t_st = st.t_schw(1.0).map(|t| fmt_time(t, ms)).unwrap_or_else(|| "∞ (за горизонтом)".into());
    o += &format!(
        "Станция: r = {} · t = {} · τ = {} · dτ/dt = {} · прилив по длине {:.3e} g\n",
        fmt_len(st.r(), mm),
        t_st,
        fmt_time(st.tau - s.tau0_station, ms),
        na(st.dtau_dt(1.0)),
        s.station_tidal_g()
    );
    let stc = &s.structure;
    let (near_far, spread) = stc.clock_spread(mm);
    let state = match &stc.first_break {
        None => "цела".to_string(),
        Some(e) => format!("разрушается с r = {:.2} M · фрагментов {} · разрывов {}", e.r, stc.live_fragments(), stc.events.len()),
    };
    o += &format!(
        "Корпус: {} · давление {:.0} гПа · часы: ближний − дальний конец = {}, разброс {}\n",
        state,
        stc.pressure / 100.0,
        fmt_secs(near_far),
        fmt_secs(spread)
    );
    if h.has_timestamps {
        let total: f32 = h.timings.iter().sum();
        let parts: Vec<String> = PASS_NAMES.iter().zip(h.timings.iter()).map(|(n, t)| format!("{n} {t:.2}")).collect();
        o += &format!("GPU, мс: {} · всего {:.2}\n", parts.join(" | "), total);
    } else {
        o += "GPU: timestamp queries недоступны\n";
    }
    o += &format!(
        "Шагов/луч: {:.1} · трассировка {}×{} → {}×{} · экспозиция ×{:.3} · {:.1} FPS · {}",
        h.mean_steps, h.trace_size.0, h.trace_size.1, h.out_size.0, h.out_size.1, h.exposure, h.fps, h.adapter
    );
    o
}

pub fn hud(ctx: &egui::Context, h: &HudInfo) {
    egui::Area::new(egui::Id::new("hud")).anchor(egui::Align2::LEFT_TOP, [8.0, 8.0]).show(ctx, |ui| {
        egui::Frame::none().fill(egui::Color32::from_black_alpha(150)).rounding(4.0).inner_margin(6.0).show(ui, |ui| {
            ui.add(egui::Label::new(egui::RichText::new(hud_text(h)).monospace().size(12.0).color(egui::Color32::from_gray(225))).extend());
        });
    });
}

fn log_slider(ui: &mut egui::Ui, v: &mut f64, lo: f64, hi: f64, label: &str) -> bool {
    ui.add(egui::Slider::new(v, lo..=hi).logarithmic(true).text(label)).changed()
}

pub fn panels(ctx: &egui::Context, st: &mut UiState, sim: &mut Sim, s: &mut Settings) {
    if !st.show_panels {
        return;
    }
    egui::Window::new("Чёрная дыра").default_pos([10.0, 140.0]).default_width(300.0).show(ctx, |ui| {
        ui.label("Масса:");
        for (i, p) in PRESETS.iter().enumerate() {
            ui.radio_value(&mut sim.mass_idx, i, p.name);
        }
        let m = physics::m_geo(sim.preset().solar);
        ui.label(format!("r_s = {:.4e} м · радиус приливного разрушения станции = {:.1} M", 2.0 * m, sim.tidal_radius_m()));
        ui.add_enabled(false, egui::Slider::new(&mut 0.0f32, 0.0..=0.998).text("спин a (этап 10, Керр)"));
        ui.separator();
        ui.collapsing("Аккреционный диск", |ui| {
            ui.add(egui::Slider::new(&mut s.r_in, 2.0..=12.0).text("r_in, M (ISCO = 6)"));
            ui.add(egui::Slider::new(&mut s.r_out, 8.0..=60.0).text("r_out, M"));
            ui.add(egui::Slider::new(&mut s.t_peak, 2000.0..=30000.0).logarithmic(true).text("T_max, K"));
            ui.add(egui::Slider::new(&mut s.brightness, 0.05..=10.0).logarithmic(true).text("яркость"));
            ui.add(egui::Slider::new(&mut s.opacity, 0.05..=1.0).text("непрозрачность"));
            ui.add(egui::Slider::new(&mut s.halo, 0.0..=2.0).text("гало (корона)"));
            ui.add(egui::Slider::new(&mut s.sky_intensity, 0.0..=8.0).text("яркость неба"));
        });
        ui.collapsing("Начальные условия станции", |ui| {
            let c = &mut sim.station_cfg;
            ui.horizontal(|ui| {
                ui.radio_value(&mut c.init, "radial".to_string(), "падение с покоя");
                ui.radio_value(&mut c.init, "elliptic".to_string(), "эллипс");
                ui.radio_value(&mut c.init, "custom".to_string(), "x, v");
            });
            match c.init.as_str() {
                "radial" => {
                    ui.add(egui::Slider::new(&mut c.r0, 2.5..=500.0).logarithmic(true).text("r₀, M"));
                }
                "elliptic" => {
                    ui.add(egui::Slider::new(&mut c.rp, 4.1..=100.0).logarithmic(true).text("перицентр, M"));
                    ui.add(egui::Slider::new(&mut c.ra, 4.1..=500.0).logarithmic(true).text("апоцентр, M"));
                }
                _ => {
                    for (i, n) in ["x", "y", "z"].iter().enumerate() {
                        ui.add(egui::DragValue::new(&mut c.pos[i]).speed(0.1).prefix(format!("{n} = ")));
                    }
                    for (i, n) in ["vx", "vy", "vz"].iter().enumerate() {
                        ui.add(egui::DragValue::new(&mut c.vel[i]).speed(0.005).range(-0.99..=0.99).prefix(format!("{n} = ")));
                    }
                }
            }
            ui.add(egui::Slider::new(&mut c.marker, 0.0..=2.0).text("маркер станции, M"));
            ui.checkbox(&mut c.hull, "корпус (граф конструкции)");
            ui.add(egui::Slider::new(&mut c.safety, 0.8..=5.0).text("запас прочности"));
            if ui.button("Перезапустить станцию").clicked() {
                st.restart = true;
            }
        });
    });
    egui::Window::new("Время и камера").default_pos([10.0, 520.0]).default_width(300.0).show(ctx, |ui| {
        for m in [TimeMode::Remote, TimeMode::Station, TimeMode::Camera] {
            ui.radio_value(&mut sim.mode, m, m.label());
        }
        log_slider(ui, &mut sim.speed, 0.01, 1000.0, "множитель скорости");
        ui.horizontal(|ui| {
            ui.checkbox(&mut sim.unit_seconds, "единица — секунда (иначе GM/c³)");
            ui.checkbox(&mut sim.paused, "пауза");
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.radio_value(&mut st.cam_mode, CamMode::Orbit, "орбитальная");
            ui.radio_value(&mut st.cam_mode, CamMode::Free, "свободная");
        });
        ui.horizontal(|ui| {
            ui.radio_value(&mut st.cam_mode, CamMode::Station, "у станции");
            ui.radio_value(&mut st.cam_mode, CamMode::Cinematic, "кино (сценарий)");
        });
        ui.horizontal(|ui| {
            ui.radio_value(&mut sim.cam_kin, CamKin::Static, "висит");
            ui.radio_value(&mut sim.cam_kin, CamKin::Rain, "падает");
            ui.radio_value(&mut sim.cam_kin, CamKin::Orbit, "орбита");
        });
        ui.checkbox(&mut sim.light_delay, "запаздывание света (снимки состояния)");
        if st.cam_mode == CamMode::Free {
            log_slider(ui, &mut st.fly_speed, 0.01, 1000.0, "скорость, M/с");
            let v = st.fly_speed * sim.m_metres();
            ui.label(format!("≈ {:.3e} м за секунду экрана (WASD, Q/E, ПКМ — взгляд)", v));
        }
        ui.add(egui::Slider::new(&mut st.orbit.3, 20.0..=120.0).text("поле зрения, °"));
    });
    egui::Window::new("Качество и отладка").default_pos([10.0, 800.0]).default_width(300.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            for (i, n) in QUALITY_NAMES.iter().enumerate() {
                if ui.selectable_label(s.quality_idx == i, *n).clicked() {
                    s.quality_idx = i;
                    s.quality = QUALITY[i];
                }
            }
        });
        ui.add(egui::Slider::new(&mut s.quality.scale, 0.5..=1.0).text("разрешение прохода"));
        ui.add(egui::Slider::new(&mut s.quality.max_steps, 64..=512).text("макс. шагов"));
        ui.add(egui::Slider::new(&mut s.step_scale, 0.3..=2.0).text("масштаб шага"));
        ui.horizontal(|ui| {
            ui.radio_value(&mut s.quality.spp, 1, "1 луч/пиксель");
            ui.radio_value(&mut s.quality.spp, 4, "4 луча/пиксель");
        });
        ui.checkbox(&mut s.quality.halo, "гало над диском");
        ui.separator();
        ui.checkbox(&mut s.auto_exposure, "автоэкспозиция");
        ui.add(egui::Slider::new(&mut s.exposure_ev, -6.0..=6.0).text("экспокоррекция, EV"));
        ui.add(egui::Slider::new(&mut s.bloom, 0.0..=0.15).text("bloom"));
        ui.add(egui::Slider::new(&mut s.chromatic, 0.0..=0.02).text("хром. аберрация"));
        ui.add(egui::Slider::new(&mut s.grain, 0.0..=1.0).text("зерно"));
        ui.horizontal(|ui| {
            ui.radio_value(&mut s.tonemap, 0, "AgX");
            ui.radio_value(&mut s.tonemap, 1, "ACES");
        });
        ui.separator();
        ui.checkbox(&mut s.debug_zones, "границы зон (2M, 3M, 6M, r_прил)");
        ui.checkbox(&mut s.debug_steps, "тепловая карта шагов интегратора");
        ui.checkbox(&mut s.debug_g, "частотный сдвиг g на диске");
        ui.checkbox(&mut s.debug_clocks, "тепловая карта dτ/dt панелей станции");
        ui.checkbox(&mut s.debug_stress, "нагрузка связей корпуса");
    });
}
